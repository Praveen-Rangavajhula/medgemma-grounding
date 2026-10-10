"""One Gemma 3 image/prompt -> residual activations -> pretrained SAE reconstruction.

References:
https://huggingface.co/google/gemma-scope-2-4b-it
https://github.com/decoderesearch/SAELens (Gemma 3 pretrained checkpoint converter)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from importlib.metadata import version
from pathlib import Path

import torch

from medgemma_grounding.inference import load_image, load_model, prepare_inputs

MODEL_ID = "google/gemma-3-4b-it"
SAE_REPO = "google/gemma-scope-2-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_17_width_16k_l0_medium"
PROMPT = "Is pleural effusion present in this chest X-ray? Answer exactly Yes or No."
IMAGE = Path("data/CheXpert/val/patient64545/study1/view1_frontal.jpg")


def token_groups(input_ids, attention_mask, image_token_id, special_ids):
    """Return disjoint CPU masks; image placeholders take precedence over special IDs."""
    ids = input_ids.detach().cpu()[0]
    valid = attention_mask.detach().cpu()[0].bool()
    image = valid & (ids == image_token_id)
    special = torch.isin(ids, torch.tensor(special_ids, dtype=ids.dtype))
    return {"image": image, "text": valid & ~image & ~special}


def capture_residual(model, inputs, hook_name):
    """Record one prefill pass without changing outputs; remove hook even on failure."""
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


def reconstruction_metrics(x, reconstructed, features):
    """Metrics over tokens x dimensions, with a per-dimension token mean baseline.

    Explained variance here is 1 - SSE / SST (centered reconstruction R²).
    Return null when a denominator is zero instead of producing invalid JSON.
    """
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


def run(image_path: Path, prompt: str, revision: str | None = None):
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
    from sae_lens import SAE

    # Check access before downloading large model weights. Pin model and processor
    # to the same commit even when the user requested the moving default branch.
    model_revision = HfApi().model_info(MODEL_ID, revision=revision).sha
    hf_hub_download(MODEL_ID, "config.json", revision=model_revision)
    sae, raw_cfg, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=SAE_RELEASE, sae_id=SAE_ID, device="cpu", dtype="float32"
    )
    sae.eval()

    # Validate the checkpoint's own metadata. SAELens currently returns a
    # normalized config dict too, but don't make the run depend on optional
    # fields being present in that returned dict across SAELens versions.
    sae_config_path = try_to_load_from_cache(
        SAE_REPO, f"resid_post/{SAE_ID}/config.json"
    )
    if not isinstance(sae_config_path, str):
        raise RuntimeError("Could not identify the downloaded SAE config revision.")
    sae_checkpoint_cfg = json.loads(Path(sae_config_path).read_text())
    expected_hook = "blocks.17.hook_resid_post"
    if (
        sae_checkpoint_cfg.get("model_name") != MODEL_ID
        or raw_cfg.get("model_name", MODEL_ID) != MODEL_ID
        or raw_cfg.get("hook_name", expected_hook) != expected_hook
    ):
        raise ValueError(
            "Unexpected SAE training site. Checkpoint config: "
            f"{sae_checkpoint_cfg}; SAELens config keys: {sorted(raw_cfg)}"
        )
    if sae.cfg.normalize_activations != "none":
        raise ValueError("Review the checkpoint normalization before using this experiment.")
    # Google's converter specifies a text-model-relative output hook. Our model
    # is the multimodal wrapper, whose text backbone is model.language_model.
    match = re.fullmatch(
        r"model\.layers\.(\d+)\.output",
        sae_checkpoint_cfg.get("hf_hook_point_in", ""),
    )
    if match is None or int(match[1]) != 17:
        raise ValueError(
            "Unexpected Hugging Face hook in SAE checkpoint config: "
            f"{sae_checkpoint_cfg.get('hf_hook_point_in')}"
        )
    hook_name = f"model.language_model.layers.{match[1]}"
    sae_path = try_to_load_from_cache(SAE_REPO, f"resid_post/{SAE_ID}/params.safetensors")
    if not isinstance(sae_path, str):
        raise RuntimeError("Could not identify the downloaded SAE checkpoint revision.")
    sae_revision = Path(sae_path).parents[2].name

    image = load_image(image_path)
    bundle = load_model(MODEL_ID, revision=model_revision)
    inputs = prepare_inputs(bundle, image, prompt)
    with torch.inference_mode():
        baseline = bundle.model(**inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
    x, hooked_logits = capture_residual(bundle.model, inputs, hook_name)
    torch.testing.assert_close(hooked_logits, baseline, rtol=1e-5, atol=1e-5)
    if x.shape != (*inputs["input_ids"].shape, sae.cfg.d_in):
        raise ValueError(f"Activation/token/SAE dimensions disagree: {x.shape}, {sae.cfg.d_in}")
    groups = token_groups(
        inputs["input_ids"], inputs["attention_mask"],
        bundle.model.config.image_token_index, bundle.processor.tokenizer.all_special_ids,
    )
    if not all(mask.any() for mask in groups.values()):
        raise ValueError("Expected both image and ordinary text tokens in this input.")
    # Only the selected layer is cached. Small CPU batches bound temporary SAE memory.
    decoded, encoded = [], []
    with torch.inference_mode():
        for chunk in x[0].split(64):
            z = sae.encode(chunk)
            encoded.append(z)
            decoded.append(sae.decode(z))
    z, x_hat = torch.cat(encoded), torch.cat(decoded)
    if not all(torch.isfinite(t).all() for t in (x, z, x_hat, baseline, hooked_logits)):
        raise ValueError("Non-finite tensor in reconstruction check.")
    return {
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "processor_revision": model_revision,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "sae_revision": sae_revision,
        "sae_config": sae_checkpoint_cfg,
        "sae_lens_config": raw_cfg,
        "hook": hook_name,
        "layer_indexing": "zero-based, block output before final model normalization",
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "prompt": prompt,
        "device": bundle.device,
        "model_dtype": str(bundle.model.dtype),
        "sae_device_dtype": "cpu/float32",
        "versions": {name: version(name) for name in ("torch", "transformers", "sae-lens")},
        "activation_shape": list(x.shape),
        "feature_shape": list(z.shape),
        "input_tokens": inputs["input_ids"].shape[-1],
        "excluded_tokens": int((~(groups["image"] | groups["text"])).sum()),
        "hook_max_logit_difference": (baseline - hooked_logits).abs().max().item(),
        "metrics": {name: reconstruction_metrics(x[0, mask], x_hat[mask], z[mask])
                    for name, mask in groups.items()},
    }


def main():
    from huggingface_hub.errors import GatedRepoError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=IMAGE)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--revision", help="Optional Gemma 3 checkpoint commit")
    parser.add_argument("--output", type=Path, default=Path("results/gemma_scope/gemma3_reconstruction.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        result = run(args.image, args.prompt, args.revision)
    except GatedRepoError:
        parser.exit(1, "Gemma 3 access is required. Accept the terms at "
                    "https://huggingface.co/google/gemma-3-4b-it using the account "
                    "associated with HF_TOKEN, then rerun. MedGemma access is separate.\n")
    serialized = json.dumps(result, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")
    print(serialized)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
