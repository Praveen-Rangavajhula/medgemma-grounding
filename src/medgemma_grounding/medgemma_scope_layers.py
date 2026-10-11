"""Screen matching Gemma Scope 2 residual SAEs across MedGemma's 34 layers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import statistics
from importlib.metadata import version
from pathlib import Path

import torch

from medgemma_grounding.inference import DEFAULT_MODEL_ID, load_image, load_model, prepare_inputs
from medgemma_grounding.sae_reconstruction import (
    SAE_RELEASE_ALL,
    SAE_REVISION,
    capture_residual,
    checkpoint_spec,
    encode_decode,
    load_scope_sae,
    logit_metrics,
    replace_residual,
    token_groups,
)

PILOT = Path("results/gemma_scope/medgemma_pilot.json")
OUTPUT = Path("results/gemma_scope/medgemma_layer_profile.json")


class ReconstructionTotals:
    """Pooled error and variance for one token group at one layer."""

    def __init__(self, width: int):
        self.tokens = 0
        self.active_features = 0
        self.sse = 0.0
        self.sum_x = torch.zeros(width, dtype=torch.float64)
        self.sum_x2 = torch.zeros(width, dtype=torch.float64)

    def add(self, x, reconstructed, features):
        if x.ndim != 2 or x.shape != reconstructed.shape or features.shape[0] != len(x):
            raise ValueError("Expected aligned [tokens, dimensions/features] tensors.")
        if not all(torch.isfinite(t).all() for t in (x, reconstructed, features)):
            raise ValueError("Non-finite SAE activations or reconstruction.")
        values = x.double()
        self.tokens += len(x)
        self.active_features += torch.count_nonzero(features).item()
        self.sse += (values - reconstructed.double()).square().sum().item()
        self.sum_x += values.sum(dim=0)
        self.sum_x2 += values.square().sum(dim=0)

    def result(self):
        if not self.tokens:
            raise ValueError("No eligible tokens for pooled reconstruction metrics.")
        sst = (self.sum_x2 - self.sum_x.square() / self.tokens).sum().item()
        return {
            "tokens": self.tokens,
            "reconstruction_sse": self.sse,
            "mean_baseline_sse": sst,
            "fvu": self.sse / sst if sst > 0 else None,
            "observed_l0": self.active_features / self.tokens,
        }


def parse_layers(value: str):
    """Use every decoder layer by default; allow short local smoke runs."""
    if value == "all":
        return list(range(34))
    try:
        layers = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise ValueError("--layers must be 'all' or comma-separated integers.") from error
    if not layers or len(layers) != len(set(layers)) or any(not 0 <= layer < 34 for layer in layers):
        raise ValueError("--layers must contain distinct layer indices from 0 to 33.")
    return sorted(layers)


def read_pilot(path: Path):
    """Use the exact model, prompt, and patient-disjoint images from the pilot."""
    pilot = json.loads(path.read_text())
    if pilot["model_id"] != DEFAULT_MODEL_ID or len(pilot["cases"]) != 20:
        raise ValueError("Expected the 20-case MedGemma pilot result.")
    cases = [
        {key: case[key] for key in ("image_path", "image_sha256", "patient", "label")}
        for case in pilot["cases"]
    ]
    if (
        len({case["patient"] for case in cases}) != 20
        or sum(case["label"] == 1 for case in cases) != 10
        or sum(case["label"] == 0 for case in cases) != 10
    ):
        raise ValueError("Pilot cases must be 10 positives and 10 negatives from distinct patients.")
    for case in cases:
        path_to_image = Path(case["image_path"])
        if hashlib.sha256(path_to_image.read_bytes()).hexdigest() != case["image_sha256"]:
            raise ValueError(f"Image has changed since the pilot: {path_to_image}")
    manifest = {
        "model_id": pilot["model_id"],
        "model_revision": pilot["model_revision"],
        "prompt": pilot["prompt"],
        "cases": cases,
    }
    return manifest


def load_profile(output: Path, manifest: dict, layers: list[int]):
    """Create or resume an output after checking its experiment identity."""
    identity = {**manifest, "planned_layers": layers, "sae_release": SAE_RELEASE_ALL}
    if not output.exists():
        return {**identity, "layers": []}
    profile = json.loads(output.read_text())
    if any(profile.get(key) != value for key, value in identity.items()):
        raise ValueError("Existing layer profile has different cases, model, prompt, or SAE settings.")
    completed = [entry["layer"] for entry in profile["layers"]]
    if len(completed) != len(set(completed)) or any(layer not in layers for layer in completed):
        raise ValueError("Existing layer profile has duplicate or unexpected layer results.")
    if any(entry.get("sae_revision") != SAE_REVISION for entry in profile["layers"]):
        raise ValueError("Existing layer profile uses a different SAE revision.")
    return profile


def save_profile(output: Path, profile: dict):
    """Keep the last complete layer result if a run is interrupted."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def screen_layer(bundle, prepared, layer: int):
    """Evaluate one SAE on all cases, then return only scalar measurements."""
    release, sae_id, _, hook_name = checkpoint_spec(layer, all_layers=True)
    sae, checkpoint_cfg, _, sae_revision = load_scope_sae(layer, all_layers=True)
    totals = {name: ReconstructionTotals(sae.cfg.d_in) for name in ("image", "text")}
    measured = []
    for case, inputs, baseline, groups in prepared:
        hidden, passive_logits = capture_residual(bundle.model, inputs, hook_name)
        torch.testing.assert_close(passive_logits, baseline, rtol=1e-5, atol=1e-5)
        if hidden.shape != (*inputs["input_ids"].shape, sae.cfg.d_in):
            raise ValueError(f"Activation/token/SAE dimensions disagree at layer {layer}.")
        reconstructed, features = encode_decode(sae, hidden[0])
        group_counts = {}
        for name, mask in groups.items():
            selected = hidden[0, mask]
            selected_hat = reconstructed[mask]
            selected_features = features[mask]
            totals[name].add(selected, selected_hat, selected_features)
            group_counts[name] = {
                "tokens": len(selected),
                "reconstruction_sse": (selected.double() - selected_hat.double()).square().sum().item(),
                "active_features": torch.count_nonzero(selected_features).item(),
            }
        # KL measures replacement at every position, including the special
        # chat tokens excluded from the image/text FVU and L0 summaries.
        replaced_logits = replace_residual(bundle.model, inputs, hook_name, reconstructed.unsqueeze(0))
        measured.append({
            "image_path": case["image_path"],
            "image_sha256": case["image_sha256"],
            "label": case["label"],
            "groups": group_counts,
            "next_token_kl": logit_metrics(baseline, replaced_logits)["kl_from_baseline"],
            "passive_hook_max_logit_difference": (baseline - passive_logits).abs().max().item(),
        })
    result = {
        "layer": layer,
        "hook": hook_name,
        "sae_release": release,
        "sae_id": sae_id,
        "sae_revision": sae_revision,
        "sae_width": checkpoint_cfg["width"],
        "target_l0": checkpoint_cfg["l0"],
        "summary": {
            "image": totals["image"].result(),
            "text": totals["text"].result(),
            "median_next_token_kl": statistics.median(case["next_token_kl"] for case in measured),
        },
        "cases": measured,
    }
    del sae
    gc.collect()
    return result


def run(pilot_path: Path = PILOT, output: Path = OUTPUT, layers: list[int] | None = None):
    """Load MedGemma once and save a resumable layer profile."""
    layers = list(range(34)) if layers is None else layers
    manifest = read_pilot(pilot_path)
    profile = load_profile(output, manifest, layers)
    done = {entry["layer"] for entry in profile["layers"]}
    if done == set(layers):
        return profile

    versions = {name: version(name) for name in ("torch", "transformers", "sae-lens")}
    if profile["layers"] and profile.get("versions") != versions:
        raise ValueError("Existing layer profile uses different package versions.")
    bundle = load_model(DEFAULT_MODEL_ID, revision=manifest["model_revision"])
    profile.update({
        "device": bundle.device,
        "model_dtype": str(bundle.model.dtype),
        "sae_device_dtype": "cpu/float32",
        "versions": versions,
    })
    prepared = []
    for case in manifest["cases"]:
        inputs = prepare_inputs(bundle, load_image(case["image_path"]), manifest["prompt"])
        groups = token_groups(
            inputs["input_ids"], inputs["attention_mask"],
            bundle.model.config.image_token_index, bundle.processor.tokenizer.all_special_ids,
        )
        if not all(mask.any() for mask in groups.values()):
            raise ValueError(f"Expected image and text tokens in {case['image_path']}.")
        with torch.inference_mode():
            baseline = bundle.model(**inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
        prepared.append((case, inputs, baseline, groups))

    for layer in layers:
        if layer in done:
            continue
        logging.info("Screening MedGemma layer %d (%d/%d)", layer, len(done) + 1, len(layers))
        profile["layers"].append(screen_layer(bundle, prepared, layer))
        profile["layers"].sort(key=lambda entry: entry["layer"])
        save_profile(output, profile)
        done.add(layer)
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-json", type=Path, default=PILOT)
    parser.add_argument("--layers", default="all", help="all or comma-separated layer indices")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        layers = parse_layers(args.layers)
    except ValueError as error:
        parser.error(str(error))
    profile = run(args.pilot_json, args.output, layers)
    print(f"Completed {len(profile['layers'])}/{len(layers)} layers. Saved: {args.output}")


if __name__ == "__main__":
    main()
