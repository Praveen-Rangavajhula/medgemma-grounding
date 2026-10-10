"""Check Gemma Scope 2 layer-17 SAE transfer on MedGemma 1.5.

Run one image first, or use --pilot for a fixed 20-image CheXpert validation set.
The checkpoint was trained on Gemma 3; these measurements test transfer, not
feature meaning or clinical validity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import statistics
from importlib.metadata import version
from pathlib import Path

import torch

from medgemma_grounding.inference import DEFAULT_MODEL_ID, load_image, load_model, prepare_inputs
from medgemma_grounding.sae_reconstruction import (
    HOOK_NAME,
    SAE_ID,
    SAE_RELEASE,
    capture_residual,
    encode_decode,
    load_scope_sae,
    logit_metrics,
    reconstruction_metrics,
    replace_residual,
    token_groups,
)

PROMPT = "Is pleural effusion present in this chest X-ray? Answer exactly Yes or No."
IMAGE = Path("data/CheXpert/val/patient64545/study1/view1_frontal.jpg")
LABELS = Path("data/CheXpert/val_labels.csv")
PILOT_SEED = 0


def pilot_cases(labels_path: Path = LABELS, image_root: Path = Path("data/CheXpert/val")):
    """Select 10 definite positives and 10 negatives from distinct patients."""
    candidates = {0: [], 1: []}
    with labels_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            label = row["Pleural Effusion"]
            if row["Frontal/Lateral"] != "Frontal" or label not in {"0.0", "1.0"}:
                continue
            relative = Path(row["Path"])
            if len(relative.parts) < 4 or relative.parts[1] != "valid":
                raise ValueError(f"Unexpected CheXpert validation path: {relative}")
            patient = relative.parts[2]
            path = image_root.joinpath(*relative.parts[2:])
            if path.is_file():
                candidates[int(float(label))].append({
                    "image_path": str(path), "patient": patient, "label": int(float(label))
                })
    rng = random.Random(PILOT_SEED)
    chosen, patients = [], set()
    for label in (1, 0):
        pool = sorted(candidates[label], key=lambda case: case["image_path"])
        rng.shuffle(pool)
        for case in pool:
            if case["patient"] not in patients:
                chosen.append(case)
                patients.add(case["patient"])
            if sum(item["label"] == label for item in chosen) == 10:
                break
        if sum(item["label"] == label for item in chosen) != 10:
            raise ValueError(f"Need 10 available frontal CheXpert cases with label {label}.")
    return chosen


def inspect_case(bundle, sae, case, prompt):
    """Measure reconstruction and next-token effects for one MedGemma input."""
    path = Path(case["image_path"])
    inputs = prepare_inputs(bundle, load_image(path), prompt)
    with torch.inference_mode():
        baseline = bundle.model(**inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
    x, passive_logits = capture_residual(bundle.model, inputs, HOOK_NAME)
    torch.testing.assert_close(passive_logits, baseline, rtol=1e-5, atol=1e-5)
    if x.shape != (*inputs["input_ids"].shape, sae.cfg.d_in) or sae.cfg.d_in != 2560:
        raise ValueError(f"MedGemma activation/token/SAE width mismatch: {x.shape}, {sae.cfg.d_in}")
    groups = token_groups(
        inputs["input_ids"], inputs["attention_mask"],
        bundle.model.config.image_token_index, bundle.processor.tokenizer.all_special_ids,
    )
    if not all(mask.any() for mask in groups.values()):
        raise ValueError("Expected both image placeholders and ordinary text tokens.")
    x_hat, features = encode_decode(sae, x[0])
    if not all(torch.isfinite(t).all() for t in (x, x_hat, features)):
        raise ValueError("Non-finite SAE activations or reconstruction.")
    reconstructed_logits = replace_residual(bundle.model, inputs, HOOK_NAME, x_hat.unsqueeze(0))
    zero_logits = replace_residual(bundle.model, inputs, HOOK_NAME, torch.zeros_like(x))
    return {
        **case,
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "activation_shape": list(x.shape),
        "feature_shape": list(features.shape),
        "input_tokens": inputs["input_ids"].shape[-1],
        "excluded_tokens": int((~(groups["image"] | groups["text"])).sum()),
        "hook_max_logit_difference": (baseline - passive_logits).abs().max().item(),
        "metrics": {
            name: reconstruction_metrics(x[0, mask], x_hat[mask], features[mask])
            for name, mask in groups.items()
        },
        "next_token": {
            "baseline_top_token_id": baseline.argmax(dim=-1).item(),
            "reconstruction": logit_metrics(baseline, reconstructed_logits),
            "zero_ablation": logit_metrics(baseline, zero_logits),
        },
    }


def summarize(cases):
    """Aggregate only the quantities used to judge direct SAE transfer."""
    return {
        "cases": len(cases),
        "median_relative_squared_error": {
            group: statistics.median(case["metrics"][group]["relative_squared_error"] for case in cases)
            for group in ("image", "text")
        },
        "median_kl_from_baseline": {
            condition: statistics.median(case["next_token"][condition]["kl_from_baseline"] for case in cases)
            for condition in ("reconstruction", "zero_ablation")
        },
        "reconstruction_top_token_preserved": sum(
            case["next_token"]["reconstruction"]["top_token_preserved"] for case in cases
        ),
    }


def run(cases, prompt: str = PROMPT, revision: str | None = None):
    from huggingface_hub import HfApi, hf_hub_download

    # Resolve and record one model/processor commit before loading either.
    model_revision = HfApi().model_info(DEFAULT_MODEL_ID, revision=revision).sha
    hf_hub_download(DEFAULT_MODEL_ID, "config.json", revision=model_revision)
    sae, sae_config, sae_lens_config, sae_revision = load_scope_sae()
    bundle = load_model(DEFAULT_MODEL_ID, revision=model_revision)
    measured = []
    for index, case in enumerate(cases, 1):
        logging.info("MedGemma SAE case %d/%d: %s", index, len(cases), case["image_path"])
        measured.append(inspect_case(bundle, sae, case, prompt))
    return {
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": model_revision,
        "processor_revision": model_revision,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "sae_revision": sae_revision,
        "sae_config": sae_config,
        "sae_lens_config": sae_lens_config,
        "hook": HOOK_NAME,
        "layer_indexing": "zero-based, block output before final model normalization",
        "prompt": prompt,
        "device": bundle.device,
        "model_dtype": str(bundle.model.dtype),
        "sae_device_dtype": "cpu/float32",
        "versions": {name: version(name) for name in ("torch", "transformers", "sae-lens")},
        "pilot_seed": PILOT_SEED if len(cases) == 20 else None,
        "summary": summarize(measured),
        "cases": measured,
    }


def main():
    from huggingface_hub.errors import GatedRepoError

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--image", type=Path, default=IMAGE)
    mode.add_argument("--pilot", action="store_true", help="Run fixed 20-image CheXpert validation pilot")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--revision", help="Optional MedGemma checkpoint commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cases = pilot_cases() if args.pilot else [{"image_path": str(args.image)}]
    output = args.output or Path(
        "results/gemma_scope/medgemma_pilot.json" if args.pilot
        else "results/gemma_scope/medgemma_reconstruction.json"
    )
    try:
        result = run(cases, args.prompt, args.revision)
    except GatedRepoError:
        parser.exit(1, "MedGemma access is required. Accept its Hugging Face terms using the account associated with HF_TOKEN.\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
