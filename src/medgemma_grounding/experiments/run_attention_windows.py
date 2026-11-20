"""E-009: fixed discovery-only image-to-answer attention-knockout depth profile.

prepare -> benchmark (includes controls) -> run -> report. Same CLI on MPS/CUDA;
different hardware requires a separate protocol/output directory. No SAE loading."""


from pathlib import Path
import argparse
import logging
import time
import traceback

from medgemma_grounding.experiments.replay import replay_if_historical
from medgemma_grounding.interventions.attention_runtime import (
    OUTPUT,
    complete_case,
    engineering,
    execute_case,
    prepare,
    read_record,
    runtime,
    validate,
    write_record,
)
from medgemma_grounding.model import weight_identity
from medgemma_grounding.utils.io import atomic_json


def run(stage, output, device, data_root, manifest_path=None):
    historical = replay_if_historical('medgemma_attention', stage=stage, output=output, device=device, data_root=data_root)
    if historical is not None:
        return historical
    p = prepare(output, device, manifest_path)
    manifest = validate(output, p)
    if stage == "prepare":
        return p
    if stage == "report":
        from medgemma_grounding.analysis.attention_windows import report
        return report(output, p, manifest)
    if list(output.glob("failure_*.json")):
        raise ValueError("Technical failure recorded; diagnose without relaxing checks")
    if stage == "run" and all(complete_case(output, p, c) for c in p["case_ids"]):
        return run("report", output, device, data_root)
    bundle = None
    try:
        bundle, info = runtime(p)
        weights = weight_identity(bundle.model)
        benchmark = read_record(output / "benchmark.json", p) if (output / "benchmark.json").exists() else engineering(bundle, p, manifest, output, data_root, info)
        if stage == "run":
            if not benchmark["backend_approved"]:
                raise RuntimeError("Compute gate: local forecast impractical; prepare single-GPU launch and request missing remote access")
            start = time.perf_counter()
            for i, case in enumerate(manifest["cases"], 1):
                if complete_case(output, p, case["case_id"]):
                    continue
                logging.info("Case %d/142 %s", i, case["case_id"])
                execute_case(bundle, p, case, output, data_root)
                atomic_json(output / "progress.json", {"completed_cases": i, "elapsed_seconds": time.perf_counter() - start,
                                                       "last_case": case["case_id"]})
        if weights != weight_identity(bundle.model) or any(v.requires_grad or v.grad is not None for v in bundle.model.parameters()):
            raise ValueError("Parameter state changed")
        write_record(output / "runtime_checks.json", {"protocol_hash": p["protocol_hash"], "frozen_unchanged_no_gradients": True, **info})
        return benchmark if stage == "benchmark" else run("report", output, device, data_root)
    except Exception as error:
        write_record(output / f"failure_{stage}.json", {"protocol_hash": p["protocol_hash"], "error": str(error),
            "diagnostics": getattr(error, "diagnostics", None), "traceback": traceback.format_exc(), "settings_changed": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["prepare", "benchmark", "run", "report"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["mps", "cuda"], default="mps")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.stage, args.output_dir, args.device, args.data_root, args.manifest)
    logging.info("Stage %s finished: %s", args.stage, {k: result[k] for k in ("complete", "backend_decision") if k in result})


if __name__ == "__main__":
    main()
