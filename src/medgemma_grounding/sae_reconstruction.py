"""Small model-independent helpers for residual-stream SAE checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

SAE_REPO = "google/gemma-scope-2-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_RELEASE_ALL = "gemma-scope-2-4b-it-res-all"
SAE_ID = "layer_17_width_16k_l0_medium"
SAE_SOURCE_MODEL = "google/gemma-3-4b-it"
HOOK_NAME = "model.language_model.layers.17"


def checkpoint_spec(layer: int = 17, *, all_layers: bool = False):
    """Identify the released residual SAE and matching MedGemma layer hook."""
    if not 0 <= layer < 34:
        raise ValueError("Layer must be between 0 and 33.")
    if not all_layers and layer != 17:
        raise ValueError("The original residual SAE release is only used at layer 17.")
    release = SAE_RELEASE_ALL if all_layers else SAE_RELEASE
    sae_id = f"layer_{layer}_width_16k_l0_big" if all_layers else SAE_ID
    folder = "resid_post_all" if all_layers else "resid_post"
    hook_name = f"model.language_model.layers.{layer}"
    return release, sae_id, folder, hook_name


def load_scope_sae(layer: int = 17, *, all_layers: bool = False):
    """Load and validate one Gemma 3 residual SAE for the matching layer."""
    from huggingface_hub import try_to_load_from_cache
    from sae_lens import SAE

    release, sae_id, folder, _ = checkpoint_spec(layer, all_layers=all_layers)
    sae, raw_cfg, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release, sae_id=sae_id, device="cpu", dtype="float32"
    )
    sae.eval()
    config_path = try_to_load_from_cache(SAE_REPO, f"{folder}/{sae_id}/config.json")
    weights_path = try_to_load_from_cache(SAE_REPO, f"{folder}/{sae_id}/params.safetensors")
    if not isinstance(config_path, str) or not isinstance(weights_path, str):
        raise RuntimeError("Could not identify the downloaded SAE config and checkpoint.")
    sae_revision = Path(weights_path).parents[2].name
    if Path(config_path).parents[2].name != sae_revision:
        raise ValueError("SAE config and weights came from different repository revisions.")
    checkpoint_cfg = json.loads(Path(config_path).read_text())
    hf_hook = checkpoint_cfg.get("hf_hook_point_in", "")
    match = re.fullmatch(r"model\.layers\.(\d+)\.output", hf_hook)
    expected_hook = f"blocks.{layer}.hook_resid_post"
    if (
        checkpoint_cfg.get("model_name") != SAE_SOURCE_MODEL
        or raw_cfg.get("model_name", SAE_SOURCE_MODEL) != SAE_SOURCE_MODEL
        or raw_cfg.get("hook_name", expected_hook) != expected_hook
        or match is None or int(match[1]) != layer
        or sae.cfg.normalize_activations != "none"
        or sae.cfg.d_in != 2560
        or sae.cfg.d_sae != checkpoint_cfg.get("width")
    ):
        raise ValueError("SAE checkpoint source, hook, width, or normalization differs from the expected control.")
    return sae, checkpoint_cfg, raw_cfg, sae_revision


def token_groups(input_ids, attention_mask, image_token_id, special_ids):
    """Return disjoint CPU masks; image placeholders take precedence over special IDs."""
    ids = input_ids.detach().cpu()[0]
    valid = attention_mask.detach().cpu()[0].bool()
    image = valid & (ids == image_token_id)
    special = torch.isin(ids, torch.tensor(special_ids, dtype=ids.dtype))
    return {"image": image, "text": valid & ~image & ~special}


def capture_residual(model, inputs, hook_name):
    """Record one prefill pass without changing outputs; always remove the hook."""
    captured = []

    def record(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach().to(device="cpu", dtype=torch.float32).clone())

    handle = model.get_submodule(hook_name).register_forward_hook(record)
    try:
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits
            logits = logits.detach().float().cpu()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one prefill capture, received {len(captured)}.")
    return captured[0], logits


def replace_residual(model, inputs, hook_name, replacement):
    """Run one prefill pass with a fixed layer output, then remove the hook."""
    calls = 0

    def replace(_module, _args, output):
        nonlocal calls
        calls += 1
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape != replacement.shape:
            raise ValueError(f"Replacement shape {replacement.shape} != {hidden.shape}")
        new_hidden = replacement.to(device=hidden.device, dtype=hidden.dtype)
        return (new_hidden, *output[1:]) if isinstance(output, tuple) else new_hidden

    handle = model.get_submodule(hook_name).register_forward_hook(replace)
    try:
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits
            logits = logits.detach().float().cpu()
    finally:
        handle.remove()
    if calls != 1:
        raise RuntimeError(f"Expected one replacement, received {calls}.")
    return logits


def encode_decode(sae, hidden, chunk_size=64):
    """Keep SAE computation on CPU and bound temporary feature memory."""
    if hidden.ndim != 2:
        raise ValueError("Expected [tokens, dimensions] activations.")
    decoded, encoded = [], []
    with torch.inference_mode():
        for chunk in hidden.split(chunk_size):
            features = sae.encode(chunk)
            encoded.append(features)
            decoded.append(sae.decode(features))
    return torch.cat(decoded), torch.cat(encoded)


def reconstruction_metrics(x, reconstructed, features):
    """Return reconstruction error, centered explained variance, and feature L0."""
    if x.ndim != 2 or x.shape != reconstructed.shape or features.shape[0] != x.shape[0]:
        raise ValueError("Expected aligned [tokens, dimensions/features] tensors.")
    if not all(torch.isfinite(t).all() for t in (x, reconstructed, features)):
        raise ValueError("Non-finite activations, features, or reconstructions.")
    if len(x) == 0:
        return {"tokens": 0}
    error = (x - reconstructed).square().sum().item()
    energy = x.square().sum().item()
    centered_energy = (x - x.mean(dim=0)).square().sum().item()
    return {
        "tokens": len(x),
        "finite": True,
        "mse": error / x.numel(),
        "relative_squared_error": error / energy if energy else None,
        "explained_variance": 1 - error / centered_energy if centered_energy else None,
        "mean_active_features": (features != 0).sum(dim=-1).float().mean().item(),
    }


def logit_metrics(reference, comparison):
    """Compare next-token distributions after a layer intervention."""
    if reference.shape != comparison.shape or reference.ndim != 3 or reference.shape[:2] != (1, 1):
        raise ValueError("Expected matching [1, 1, vocabulary] next-token logits.")
    if not torch.isfinite(reference).all() or not torch.isfinite(comparison).all():
        raise ValueError("Non-finite next-token logits.")
    baseline_logp = reference[0, 0].log_softmax(dim=-1)
    changed_logp = comparison[0, 0].log_softmax(dim=-1)
    baseline_p = baseline_logp.exp()
    return {
        "kl_from_baseline": (baseline_p * (baseline_logp - changed_logp)).sum().item(),
        "top_token_preserved": reference.argmax(dim=-1).item() == comparison.argmax(dim=-1).item(),
        "top_token_id": comparison.argmax(dim=-1).item(),
    }
