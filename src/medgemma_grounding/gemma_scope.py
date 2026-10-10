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
from importlib.metadata import version
from pathlib import Path

import torch

from medgemma_grounding.inference import load_image, load_model, prepare_inputs
from medgemma_grounding.sae_reconstruction import (
    HOOK_NAME,
    SAE_ID,
    SAE_RELEASE,
    SAE_REPO,
    capture_residual,
    encode_decode,
    load_scope_sae,
    reconstruction_metrics,
    token_groups,
)

MODEL_ID = "google/gemma-3-4b-it"
PROMPT = "Is pleural effusion present in this chest X-ray? Answer exactly Yes or No."
IMAGE = Path("data/CheXpert/val/patient64545/study1/view1_frontal.jpg")


def run(image_path: Path, prompt: str, revision: str | None = None):
    from huggingface_hub import HfApi, hf_hub_download

    # Check access before downloading large model weights. Pin model and processor
    # to the same commit even when the user requested the moving default branch.
    model_revision = HfApi().model_info(MODEL_ID, revision=revision).sha
    hf_hub_download(MODEL_ID, "config.json", revision=model_revision)
    sae, sae_checkpoint_cfg, raw_cfg, sae_revision = load_scope_sae()
    hook_name = HOOK_NAME

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
    x_hat, z = encode_decode(sae, x[0])
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
