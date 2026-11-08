"""Section 2.4: passive residual captures for the verified occlusion pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

from medgemma_grounding.baseline_pilot import check_candidate_tokens
from medgemma_grounding.grounding_baseline import json_text
from medgemma_grounding.grounding_protocol import sha256
from medgemma_grounding.occlusion_pilot import (
    check_sanity, configure_cache, load_completed, tasks_for, tensor_record, verify_prepared,
)


def verified_occlusion(prepared: Path, run: Path) -> tuple[dict, list[dict]]:
    manifest = verify_prepared(prepared)
    summary = json.loads((run / "summary.json").read_text())
    settings = json.loads((run / "run.json").read_text())["settings"]
    digest = sha256(prepared / "manifest.json")
    if (summary["prepared_manifest_sha256"] != digest
            or settings["prepared_manifest_sha256"] != digest
            or summary["scores_sha256"] != sha256(run / "scores.jsonl")
            or summary["status"] != "single_case_pilot_complete"):
        raise ValueError("Section 2.3 results do not match the prepared experiment")
    tasks = tasks_for(manifest)
    rows = load_completed(run / "scores.jsonl", tasks)
    if len(rows) != len(tasks):
        raise ValueError("Section 2.3 must be complete before activation collection")
    check_sanity(rows, manifest["baseline_yes_score"])
    # All region conditions are retained; never select them using score changes.
    return manifest, [r for r in rows if r["kind"] != "patch"]


def condition_id(row: dict) -> str:
    return f"{row['fill']}_{row['condition']}"


def input_metadata(bundle, inputs) -> dict:
    from medgemma_grounding.sae_reconstruction import token_groups

    ids = inputs["input_ids"].detach().cpu()
    valid = inputs["attention_mask"].detach().cpu().bool()
    if ids.ndim != 2 or ids.shape[0] != 1 or not valid.all():
        raise ValueError("Pilot requires one unpadded input sequence")
    groups = token_groups(ids, valid, bundle.model.config.image_token_index,
                          bundle.processor.tokenizer.all_special_ids)
    positions = {name: mask.nonzero().flatten().tolist() for name, mask in groups.items()}
    positions["decision"] = [ids.shape[1] - 1]
    if not all(positions.values()):
        raise ValueError("Expected nonempty image, text and decision token groups")
    return {"input_ids": ids[0].tolist(), "positions": positions,
            "last_prefill_token_id": int(ids[0, -1]),
            "last_prefill_token_text": bundle.processor.tokenizer.decode(ids[0, -1:].tolist()),
            "position_policy": "Zero-based positions in the unpadded prefill. Decision is the last input token predicting the first answer, not a generated Yes/No token. Text excludes image and special tokens; decision may overlap text.",
            "tensors": {k: tensor_record(v, bundle.torch) for k, v in inputs.items()}}


def capture_score(bundle, image, prompt: str, hooks: list[str]) -> tuple[float, np.ndarray | None, np.ndarray]:
    """Call the unchanged scorer; hooks return None and copy outputs to CPU."""
    from medgemma_grounding.classification import compute_yes_score

    torch = bundle.torch
    hidden, logits, handles = {}, [], []

    def record(name):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            if name in hidden or value.ndim != 3 or value.shape[0] != 1:
                raise ValueError("Expected exactly one [1, tokens, width] prefill per layer")
            hidden[name] = value[0].detach().to(device="cpu", dtype=torch.float32).clone().numpy()
        return hook

    def record_logits(_module, _args, output):
        logits.append(output.logits[0, -1].detach().to(device="cpu", dtype=torch.float32).clone().numpy())

    try:
        for name in hooks:
            handles.append(bundle.model.get_submodule(name).register_forward_hook(record(name)))
        handles.append(bundle.model.register_forward_hook(record_logits))
        score = compute_yes_score(bundle, image, prompt)
    finally:
        for handle in handles:
            handle.remove()
    if len(logits) != 1 or set(hidden) != set(hooks):
        raise ValueError("Missing or repeated prefill captures")
    residuals = np.stack([hidden[name] for name in hooks]) if hooks else None
    if (not math.isfinite(score) or not 0 <= score <= 1 or not np.isfinite(logits[0]).all()
            or (residuals is not None and not np.isfinite(residuals).all())):
        raise ValueError("Nonfinite activation/logit or invalid Yes score")
    return score, residuals, logits[0]


def agree(actual, expected, label, *, atol=1e-5, rtol=1e-4) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f"{label} check failed; do not mix these results")


def source_hashes() -> dict:
    return {name: sha256(Path(__file__).with_name(name)) for name in
            ("activation_pilot.py", "sae_reconstruction.py", "baseline_pilot.py")}


def read_condition(directory: Path, shape: list[int]) -> tuple[dict, np.ndarray, np.ndarray]:
    meta = json.loads((directory / "condition.json").read_text())
    for name, expected in meta["artifacts_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Saved activation artifact changed: {directory.name}/{name}")
    with np.load(directory / "residuals.npz", allow_pickle=False) as archive:
        values = archive["resid_post"]
    logits = np.load(directory / "next_token_logits.npy", allow_pickle=False)
    if (list(values.shape) != shape or values.dtype != np.float32 or not np.isfinite(values).all()
            or logits.ndim != 1 or logits.dtype != np.float32 or not np.isfinite(logits).all()):
        raise ValueError("Saved activation/logit shape, dtype or finiteness check failed")
    return meta, values, logits


def verify_collection(output: Path) -> tuple[dict, list[dict]]:
    """Validate provenance and every completed checkpoint without loading a model."""
    run = json.loads((output / "run.json").read_text())
    identity = run["identity"]
    manifest, prior = verified_occlusion(Path(identity["prepared_path"]), Path(identity["occlusion_path"]))
    if (identity["prepared_sha256"] != sha256(Path(identity["prepared_path"]) / "manifest.json")
            or identity["occlusion_scores_sha256"] != sha256(Path(identity["occlusion_path"]) / "scores.jsonl")
            or identity["sources"] != source_hashes()
            or identity["condition_ids"] != [condition_id(r) for r in prior]
            or identity["part1_contract"] != manifest["part1_contract"]):
        raise ValueError("Activation run no longer matches its sources or inputs")
    rows, gap = [], False
    for old in prior:
        directory = output / condition_id(old)
        if not directory.exists():
            gap = True
            continue
        if gap:
            raise ValueError("Activation checkpoints are not a consecutive prefix")
        meta, values, logits = read_condition(directory, run["activation_shape"])
        if (meta["id"] != condition_id(old) or meta["prior_condition"] != old
                or meta["run_sha256"] != sha256(output / "run.json")):
            raise ValueError("Activation checkpoint belongs to another run/condition")
        agree(meta["yes_score"], old["yes_score"], "Saved score")
        if meta["input_tensors"]["pixel_values"] != old["pixel_tensor"]:
            raise ValueError("Saved pixels differ from the section 2.3 condition")
        for key, stamp in run["tokens"]["tensors"].items():
            if key != "pixel_values" and meta["input_tensors"].get(key) != stamp:
                raise ValueError("Saved non-image inputs differ across conditions")
        rows.append(meta)
        del values, logits
    return run, rows


def run_collection(args) -> dict:
    manifest, prior = verified_occlusion(args.prepared, args.occlusion)
    runtime = manifest["runtime"]
    identity = {"prepared_path": str(args.prepared.resolve()), "occlusion_path": str(args.occlusion.resolve()),
                "prepared_sha256": sha256(args.prepared / "manifest.json"),
                "occlusion_scores_sha256": sha256(args.occlusion / "scores.jsonl"),
                "sources": source_hashes(), "part1_contract": manifest["part1_contract"],
                "condition_ids": [condition_id(r) for r in prior],
                "selection": "All decoder resid_post layers and all prefill positions, original, empty/no-op, all eight region conditions, end repeat. No patch selection."}
    rows = []
    if args.output.exists():
        if not args.resume:
            raise ValueError("Output exists; use --resume or a new directory")
        run, rows = verify_collection(args.output)
        if run["identity"] != identity:
            raise ValueError("Resume identity differs")
        if len(rows) == len(prior):
            return finalize(args.output)
    configure_cache(args.hf_home)
    from medgemma_grounding import inference
    from medgemma_grounding.sae_reconstruction import checkpoint_spec
    import torch

    for name, version in runtime["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Runtime changed since section 2.2: {name}")
    if runtime["device"] == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; run with access to the Mac GPU")
    torch.manual_seed(0)
    print(f"Loading pinned MedGemma; {len(rows)}/{len(prior)} activation conditions saved", flush=True)
    bundle = inference.load_model(runtime["model_id"], device=runtime["device"], revision=runtime["model_revision"])
    if (str(bundle.model.dtype) != runtime["dtype"]
            or bundle.model.config._commit_hash != runtime["model_revision"]
            or bundle.model.config._attn_implementation != runtime["attention_implementation"]):
        raise ValueError("Loaded model differs from section 2.2 runtime")
    config = bundle.model.config.text_config
    if config.num_hidden_layers != 34 or config.hidden_size != 2560:
        raise ValueError("Unexpected MedGemma decoder dimensions")
    hooks = [checkpoint_spec(layer, all_layers=True)[3] for layer in range(config.num_hidden_layers)]
    original = inference.load_image(args.prepared / "original.png")
    inputs = inference.prepare_inputs(bundle, original, identity["part1_contract"]["prompt"])
    tokens = input_metadata(bundle, inputs)
    boundary = check_candidate_tokens(bundle, original, identity["part1_contract"]["prompt"], inputs)
    if tokens["tensors"] != manifest["baseline_tensors"]:
        raise ValueError("Original model inputs differ from the verified baseline")
    del inputs
    shape = [len(hooks), len(tokens["input_ids"]), config.hidden_size]
    if not args.output.exists():
        args.output.mkdir(parents=True)
        run = {"section": "2.4", "identity": identity, "case": manifest["case"], "tokens": tokens,
               "tokenization": boundary, "hooks": hooks, "activation_shape": shape,
               "axes": ["decoder_layer_zero_based", "prefill_position_zero_based", "hidden_dimension"],
               "capture_dtype": f"float32 CPU, cast from {bundle.model.dtype}; no normalization",
               "hook_semantics": "Output of each decoder layer after attention and MLP residual additions, before the final model RMSNorm. Same hook paths as the inherited Gemma Scope residual code.",
               "runtime": {**{k: v for k, v in runtime.items() if not k.startswith("mps_")},
                           "python": platform.python_version(), "platform": platform.platform()},
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
        (args.output / "run.json").write_text(json_text(run))
    elif run["hooks"] != hooks or run["tokens"] != tokens or run["activation_shape"] != shape:
        raise ValueError("Resume hook/token/shape contract changed")
    if rows:
        score, _, _ = capture_score(bundle, original, identity["part1_contract"]["prompt"], [])
        agree(score, prior[0]["yes_score"], "Resume baseline")
    for index in range(len(rows), len(prior)):
        old = prior[index]
        cid = condition_id(old)
        started = time.monotonic()
        filename = f"{cid}.png" if old["kind"] == "region" else "original.png"
        image = inference.load_image(args.prepared / filename)
        inputs = inference.prepare_inputs(bundle, image, identity["part1_contract"]["prompt"])
        stamps = {k: tensor_record(v, torch) for k, v in inputs.items()}
        if stamps["pixel_values"] != old["pixel_tensor"]:
            raise ValueError("Image tensor differs from section 2.3")
        if any(stamps.get(k) != v for k, v in tokens["tensors"].items() if k != "pixel_values"):
            raise ValueError("Non-image token/input alignment changed")
        del inputs
        # Both passes call Part 1's exact scorer. The first has only a passive
        # top-level logit recorder; the second additionally records decoder outputs.
        plain_score, _, plain_logits = capture_score(bundle, image, identity["part1_contract"]["prompt"], [])
        score, values, logits = capture_score(bundle, image, identity["part1_contract"]["prompt"], hooks)
        agree(score, old["yes_score"], "Section 2.3 score")
        agree(score, plain_score, "Passive hook score", atol=1e-7, rtol=1e-6)
        agree(logits, plain_logits, "Passive hook vocabulary logits", atol=1e-5, rtol=1e-5)
        if list(values.shape) != shape:
            raise ValueError("Captured residual shape differs from the declared shape")
        repeated = None
        if old["kind"] == "sanity":
            _, baseline, baseline_logits = read_condition(args.output / condition_id(prior[0]), shape)
            agree(values, baseline, "No-op/repeat activations", atol=1e-6, rtol=1e-5)
            agree(logits, baseline_logits, "No-op/repeat logits", atol=1e-5, rtol=1e-5)
            repeated = float(np.max(np.abs(values - baseline)))
            del baseline, baseline_logits
        meta = {"id": cid, "condition": old["condition"], "fill": old["fill"], "kind": old["kind"],
                "run_sha256": sha256(args.output / "run.json"), "prior_condition": old,
                "yes_score": score, "unhooked_yes_score": plain_score,
                "response": old["response"], "prediction": old["prediction"],
                "response_source": "Reused section 2.3 generation for these identical hashed inputs; this prefill run does not generate another answer.",
                "input_tensors": stamps, "passive_max_logit_difference": float(np.max(np.abs(logits - plain_logits))),
                "repeat_max_activation_difference": repeated, "seconds": time.monotonic() - started}
        # Publish only complete conditions; interrupted staging directories vanish.
        with tempfile.TemporaryDirectory(prefix=".capture-", dir=args.output) as temporary:
            stage = Path(temporary) / "condition"
            stage.mkdir()
            np.savez_compressed(stage / "residuals.npz", resid_post=values)
            np.save(stage / "next_token_logits.npy", logits, allow_pickle=False)
            meta["artifacts_sha256"] = {p.name: sha256(p) for p in stage.iterdir()}
            (stage / "condition.json").write_text(json_text(meta))
            stage.rename(args.output / cid)
        del values, logits, plain_logits
        rows.append(meta)
        print(f"{index+1}/{len(prior)} {cid}: Yes={score:.8f}; passive logit difference={meta['passive_max_logit_difference']:.3g}", flush=True)
    return finalize(args.output)


def finalize(output: Path) -> dict:
    run, rows = verify_collection(output)
    if len(rows) != len(run["identity"]["condition_ids"]):
        raise ValueError("Activation collection is incomplete; resume it")
    summary = {"section": "2.4", "status": "single_case_activation_pilot_complete",
               "case_id": run["case"]["case_id"], "conditions_completed": len(rows),
               "activation_shape_per_condition": run["activation_shape"],
               "run_sha256": sha256(output / "run.json"),
               "condition_metadata_sha256": {r["id"]: sha256(output / r["id"] / "condition.json") for r in rows},
               "max_passive_logit_difference": max(r["passive_max_logit_difference"] for r in rows),
               "sanity_max_activation_differences": {r["id"]: r["repeat_max_activation_difference"] for r in rows if r["kind"] == "sanity"},
               "total_inference_seconds": sum(r["seconds"] for r in rows),
               "scope": "One discovery case; no SAE encoding or activation intervention. All saved raw positions remain available for later compatible SAE analysis."}
    (output / "summary.json").write_text(json_text(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("outputs/part2/occlusion-prepared-v1"))
    parser.add_argument("--occlusion", type=Path, default=Path("outputs/part2/occlusion-run-v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/part2/activations-v1"))
    parser.add_argument("--hf-home", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json_text(run_collection(args)))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Activation pilot failed: {error}\n")


if __name__ == "__main__":
    main()
