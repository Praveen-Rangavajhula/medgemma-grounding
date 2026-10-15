"""Save layer-14 SAE features and answers for the fixed 20-case MedGemma pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch

OUTPUT = Path("results/gemma_scope/layer14_features")
PILOT = Path("results/gemma_scope/medgemma_pilot.json")
LAYER = 14


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spatial_layout(bundle, inputs, original_size):
    """Describe the verified single-image Gemma 3 resize/pooling layout.

    Image-position order has a spatial anchor, not a localized receptive field:
    vision attention and decoder attention already mix information.
    """
    config = bundle.model.config
    processor = bundle.processor.image_processor
    side = math.isqrt(config.mm_tokens_per_image)
    pixels = inputs["pixel_values"]
    n_image = int((inputs["input_ids"] == config.image_token_index).sum())
    valid = (
        type(processor).__name__.startswith("Gemma3ImageProcessor")
        and processor.do_resize
        and not processor.do_pan_and_scan
        and not getattr(processor, "do_center_crop", False)
        and not getattr(processor, "do_pad", False)
        and pixels.shape == (1, 3, config.vision_config.image_size, config.vision_config.image_size)
        and n_image == side * side == config.mm_tokens_per_image
        and config.vision_config.image_size % (config.vision_config.patch_size * side) == 0
    )
    if not valid:
        return {"verified": False, "reason": "Unsupported image preprocessing or token layout; overlays disabled."}
    return {
        "verified": True,
        "mapping": "direct_resize_row_major",
        "grid": [side, side],
        "original_size": list(original_size),
        "processed_size": list(pixels.shape[-2:]),
        "pool_kernel": config.vision_config.image_size // config.vision_config.patch_size // side,
        "processor_class": type(processor).__name__,
        "resample": int(processor.resample),
    }


def collect_case(bundle, sae, case, prompt):
    """Capture the intact model's prefill features; generate after removing hooks."""
    from medgemma_grounding.inference import generate, load_image, prepare_inputs
    from medgemma_grounding.sae_reconstruction import capture_residual, encode_decode, token_groups

    image = load_image(case["image_path"])
    inputs = prepare_inputs(bundle, image, prompt)
    hook = f"model.language_model.layers.{LAYER}"
    with torch.inference_mode():
        baseline = bundle.model(**inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
    hidden, passive_logits = capture_residual(bundle.model, inputs, hook)
    torch.testing.assert_close(passive_logits, baseline, rtol=1e-5, atol=1e-5)
    groups = token_groups(
        inputs["input_ids"], inputs["attention_mask"], bundle.model.config.image_token_index,
        bundle.processor.tokenizer.all_special_ids,
    )
    if hidden.shape != (*inputs["input_ids"].shape, sae.cfg.d_in) or not all(m.any() for m in groups.values()):
        raise ValueError("Activation, token, or SAE dimensions differ from the pilot.")
    _, features = encode_decode(sae, hidden[0])
    if not torch.isfinite(features).all() or (features < 0).any():
        raise ValueError("Expected finite, nonnegative JumpReLU features.")
    # Check the encoding independently at both ends of the sequence, including text.
    with torch.inference_mode():
        for start in (0, (len(features) - 1) // 64 * 64):
            torch.testing.assert_close(features[start:start + 64], sae.encode(hidden[0, start:start + 64]))
    tokenizer = bundle.processor.tokenizer
    ids = inputs["input_ids"][0].cpu().tolist()
    pieces = tokenizer.convert_ids_to_tokens(ids)
    tokens = [
        {
            "index": i, "id": token_id, "piece": piece,
            "text": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "group": "image" if groups["image"][i] else "text" if groups["text"][i] else "special",
        }
        for i, (token_id, piece) in enumerate(zip(ids, pieces, strict=True))
    ]
    record = {
        **case,
        "case_id": "_".join(Path(case["image_path"]).with_suffix("").parts[-3:]),
        "answer": generate(bundle, image, prompt, max_new_tokens=32),
        "tokens": tokens,
        "feature_shape": list(features.shape),
        "spatial_layout": spatial_layout(bundle, inputs, image.size),
        "passive_hook_max_logit_difference": (baseline - passive_logits).abs().max().item(),
        "observed_l0": {
            name: features[mask].count_nonzero().item() / int(mask.sum())
            for name, mask in groups.items()
        },
    }
    return record, features.to_sparse().coalesce()


def save_cache(directory, manifest, activations):
    """Publish the manifest last; its checksum detects interrupted replacement."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "activations.pt.tmp"
    torch.save(activations, temporary)
    manifest = {**manifest, "activations_sha256": file_sha256(temporary)}
    temporary.replace(directory / "activations.pt")
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "manifest.json")


def load_cache(directory):
    """Read CPU features and validate alignment without loading a model or SAE."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported feature-cache version.")
    path = directory / "activations.pt"
    if file_sha256(path) != manifest["activations_sha256"]:
        raise ValueError("Feature cache checksum mismatch; extraction may have been interrupted.")
    activations = torch.load(path, map_location="cpu", weights_only=True)
    cases = manifest["cases"]
    if len({c["case_id"] for c in cases}) != len(cases) or set(activations) != {c["case_id"] for c in cases}:
        raise ValueError("Feature cache has missing, duplicate, or unexpected cases.")
    for case in cases:
        x = activations[case["case_id"]]
        expected = (len(case["tokens"]), manifest["sae_width"])
        if x.layout != torch.sparse_coo or tuple(x.shape) != expected:
            raise ValueError(f"Token/feature alignment mismatch for {case['case_id']}.")
        x = x.coalesce()
        if not torch.isfinite(x.values()).all() or (x.values() < 0).any():
            raise ValueError("Cache contains invalid feature values.")
        if [t["index"] for t in case["tokens"]] != list(range(len(case["tokens"]))):
            raise ValueError("Token indices are out of order.")
        image_path = Path(manifest["project_root"]) / case["image_path"]
        if file_sha256(image_path) != case["image_sha256"]:
            raise ValueError(f"Image changed since extraction: {image_path}")
        activations[case["case_id"]] = x
    return manifest, activations


def run(pilot_path=PILOT, output=OUTPUT, *, overwrite=False):
    from medgemma_grounding.inference import load_model
    from medgemma_grounding.medgemma_scope_layers import read_pilot
    from medgemma_grounding.sae_reconstruction import checkpoint_spec, load_scope_sae

    output = Path(output)
    if (output / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"Cache already exists at {output}; use --overwrite to replace it.")
    pilot = read_pilot(Path(pilot_path))
    sae, config, _, sae_revision = load_scope_sae(LAYER, all_layers=True)
    bundle = load_model(pilot["model_id"], revision=pilot["model_revision"])
    cases, activations = [], {}
    for index, case in enumerate(pilot["cases"], 1):
        logging.info("Extracting layer-%d features %d/%d: %s", LAYER, index, len(pilot["cases"]), case["image_path"])
        record, features = collect_case(bundle, sae, case, pilot["prompt"])
        cases.append(record)
        activations[record["case_id"]] = features
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    release, sae_id, _, hook = checkpoint_spec(LAYER, all_layers=True)
    manifest = {
        "schema_version": 1, "project_root": str(Path.cwd().resolve()),
        "model_id": pilot["model_id"], "model_revision": pilot["model_revision"],
        "processor_revision": pilot["model_revision"], "prompt": pilot["prompt"],
        "sae_release": release, "sae_id": sae_id, "sae_revision": sae_revision,
        "sae_width": sae.cfg.d_sae, "target_l0": config["l0"], "hook": hook,
        "layer": LAYER, "device": bundle.device, "model_dtype": str(bundle.model.dtype),
        "sae_device_dtype": "cpu/float32", "generation": {"do_sample": False, "max_new_tokens": 32},
        "versions": {name: version(name) for name in ("torch", "transformers", "sae-lens")},
        "code_revision": revision.stdout.strip() or None, "code_dirty": bool(dirty.stdout.strip()),
        "source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in (
            "medgemma_features.py", "inference.py", "sae_reconstruction.py", "medgemma_scope_layers.py",
        )},
        "cases": cases,
    }
    save_cache(output, manifest, activations)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-json", type=Path, default=PILOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing extraction")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    manifest = run(args.pilot_json, args.output_dir, overwrite=args.overwrite)
    print(f"Saved layer-{LAYER} features and answers for {len(manifest['cases'])} cases: {args.output_dir}")


if __name__ == "__main__":
    main()
