"""Run one discovery image through unchanged Part 1 functions, with provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import time

from medgemma_grounding.grounding_baseline import json_text, load_protocol, save_artifacts
from medgemma_grounding.grounding_protocol import sha256


def select_case(protocol_dir: Path, audit_dir: Path, case_id: str) -> tuple[dict, dict]:
    protocol, cases = load_protocol(protocol_dir)
    summary = json.loads((audit_dir / "summary.json").read_text())
    if (summary["protocol_sha256"] != sha256(protocol_dir / "protocol.json")
            or summary["manifest_sha256"] != sha256(protocol_dir / "cohort_manifest.json")
            or summary["part1_contract"] != protocol["part1"]
            or summary["baseline_cases_sha256"] != sha256(audit_dir / "baseline_cases.json")):
        raise ValueError("Audited baseline does not match the frozen protocol/files")
    selected = next((r for r in cases if r["case_id"] == case_id), None)
    if selected is None or selected["partition"] != "discovery":
        raise ValueError("The pilot must use a case from the discovery partition")
    saved = next(r for r in json.loads((audit_dir / "baseline_cases.json").read_text()) if r["case_id"] == case_id)
    if any(saved.get(key) != value for key, value in selected.items()):
        raise ValueError("Audited case metadata differ from the frozen manifest")
    return protocol, saved


def check_candidate_tokens(bundle, image, prompt: str, inputs) -> dict:
    """Check that Part 1's bare tokens are also single-token answer continuations."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": prompt}]}]
    rendered = bundle.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokenizer = bundle.processor.tokenizer
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    actual = inputs["input_ids"][0].tolist()
    # Visual placeholders can expand in actual model inputs. The answer boundary
    # must still end with the same text tokens as the rendered chat template.
    if len(prefix) < 8 or actual[-8:] != prefix[-8:]:
        raise ValueError("Rendered chat template and actual model answer boundary disagree")
    candidates = {}
    for answer in ("No", "Yes"):
        bare = tokenizer.encode(answer, add_special_tokens=False)
        combined = tokenizer.encode(rendered + answer, add_special_tokens=False)
        if len(bare) != 1 or combined != prefix + bare:
            raise ValueError(f"Part 1 candidate {answer} is not a matching single-token continuation")
        candidates[answer] = bare[0]
    if candidates["No"] == candidates["Yes"]:
        raise ValueError("Yes and No have the same token ID")
    return {"candidate_ids": candidates, "answer_boundary_ids": actual[-8:],
            "contextual_single_token_check": "passed",
            "rendered_chat_sha256": hashlib.sha256(rendered.encode()).hexdigest()}


def compare_scores(scores: list[float], historical: float, *, atol: float, rtol: float) -> dict:
    if len(scores) != 2 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in [*scores, historical]):
        raise ValueError("Expected two finite repeat scores and one historical score in [0, 1]")
    return {"atol": atol, "rtol": rtol, "repeated_scores": scores,
            "repeat_absolute_difference": abs(scores[0] - scores[1]),
            "repeat_agrees": math.isclose(scores[0], scores[1], abs_tol=atol, rel_tol=rtol),
            "historical_score": historical, "historical_absolute_difference": abs(scores[0] - historical),
            "historical_score_agrees": math.isclose(scores[0], historical, abs_tol=atol, rel_tol=rtol)}


def evaluate_case(bundle, image, contract: dict, historical: dict, *, atol: float, rtol: float) -> dict:
    from medgemma_grounding import classification, inference

    inputs = inference.prepare_inputs(bundle, image, contract["prompt"])
    boundary = check_candidate_tokens(bundle, image, contract["prompt"], inputs)
    tensors = {}
    for key, tensor in inputs.items():
        if hasattr(tensor, "shape"):
            raw = tensor.detach().cpu().contiguous().view(bundle.torch.uint8).numpy().tobytes()
            tensors[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "sha256": hashlib.sha256(raw).hexdigest()}
    del inputs
    started = time.monotonic()
    response = inference.generate(bundle, image, contract["prompt"], max_new_tokens=contract["max_new_tokens"])
    prediction = classification.parse_prediction(response)
    scores = [classification.compute_yes_score(bundle, image, contract["prompt"]) for _ in range(2)]
    comparison = compare_scores(scores, historical["yes_score"], atol=atol, rtol=rtol)
    comparison.update({"historical_prediction": historical["prediction"],
                       "parsed_prediction_agrees": prediction == historical["prediction"],
                       "historical_response": historical["response"],
                       "response_exact_match": response == historical["response"]})
    return {"response": response, "prediction": prediction, "yes_score": scores[0],
            "comparison": comparison, "tokenization": boundary, "prepared_tensors": tensors,
            "inference_seconds": time.monotonic() - started}


def run_pilot(args) -> dict:
    if args.output.exists():
        raise ValueError("Pilot output already exists; use a new directory for another run")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Pass an immutable 40-character model commit as --revision")
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        raise ValueError("Agreement tolerances must be finite and nonnegative")
    # Configure before importing Transformers/Hugging Face. Run from the project
    # root so the default cache is ignored by Git alongside the dataset.
    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"  # The setup step downloads the pinned checkpoint.
    os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
    from medgemma_grounding import inference
    import torch

    if not os.environ.get("HF_TOKEN"):
        raise ValueError("Set HF_TOKEN in the project's ignored .env file")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this process; run with access to the Mac GPU")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this process")
    protocol, saved = select_case(args.protocol, args.audit, args.case_id)
    torch.manual_seed(0)
    print(f"Loading pinned checkpoint on {args.device}...", flush=True)
    bundle = inference.load_model(protocol["part1"]["model_id"], device=args.device, revision=args.revision)
    loaded_revision = getattr(bundle.model.config, "_commit_hash", None)
    if loaded_revision is not None and loaded_revision != args.revision:
        raise ValueError("Loaded model revision differs from the requested checkpoint")
    image = inference.load_image(Path(protocol["data_root"]) / saved["image_path"])
    print("Checking tokenization, generating the answer, and scoring twice...", flush=True)
    result = evaluate_case(bundle, image, protocol["part1"], saved, atol=args.atol, rtol=args.rtol)
    agreement = result["comparison"]
    packages = {}
    for name in ("torch", "transformers", "huggingface-hub", "tokenizers", "numpy", "Pillow", "pandas", "scikit-learn"):
        packages[name] = importlib.metadata.version(name)
    result.update({
        "section": "2.2", "case_id": args.case_id, "patient_id": saved["patient_id"],
        "label": saved["label"], "partition": "discovery", "grounding_status": "unassessed",
        "image_sha256": saved["image_sha256"], "original_image_size": list(image.size),
        "protocol_sha256": sha256(args.protocol / "protocol.json"),
        "audit_summary_sha256": sha256(args.audit / "summary.json"),
        "pilot_source_sha256": sha256(Path(__file__)), "part1_contract": protocol["part1"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"model_id": bundle.model_id, "model_revision": args.revision,
                    "processor_revision": args.revision, "loaded_config_revision": loaded_revision,
                    "device": bundle.device, "dtype": str(bundle.model.dtype), "seed": 0,
                    "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
                    "attention_implementation": getattr(bundle.model.config, "_attn_implementation", None)},
        "status": ("single_case_baseline_agrees" if agreement["repeat_agrees"] and agreement["historical_score_agrees"]
                   and agreement["parsed_prediction_agrees"] else "baseline_discrepancy_review_required"),
        "scope": "One discovery case only. Historical model revision/device/dtype are unknown; numerical agreement does not establish runtime identity or full-cohort reproduction.",
    })
    if bundle.device == "mps":
        result["runtime"]["mps_driver_allocated_bytes_at_end"] = torch.mps.driver_allocated_memory()
        result["runtime"]["mps_recommended_max_bytes"] = torch.mps.recommended_max_memory()
    processor_config = bundle.processor.to_dict()
    image_processor = getattr(bundle.processor, "image_processor", None)
    if image_processor is not None:
        processor_config["image_processor"] = image_processor.to_dict()
    save_artifacts(args.output, {"result.json": json_text(result),
                                 "processor_config.json": json_text(processor_config),
                                 "model_config.json": bundle.model.config.to_json_string(),
                                 "generation_config.json": bundle.model.generation_config.to_json_string()})
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("outputs/part2/protocol-v1"))
    parser.add_argument("--audit", type=Path, default=Path("outputs/part2/baseline-v1"))
    parser.add_argument("--case-id", default="patient64545_study1_view1_frontal")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--hf-home", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("outputs/part2/pilot-v1"))
    args = parser.parse_args(argv)
    # Keep errors concise: upstream authentication/network tracebacks can include
    # request details. Full runtime configurations are saved only after success.
    try:
        result = run_pilot(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Baseline pilot failed: {error}\n")
    print(json_text({key: result[key] for key in ("case_id", "response", "prediction", "yes_score", "comparison", "status")}))
    print(f"Runtime and processor metadata saved to {args.output}")
    if result["status"] != "single_case_baseline_agrees":
        parser.exit(1, "Review baseline discrepancies before comparing perturbations with Part 1.\n")


if __name__ == "__main__":
    main()
