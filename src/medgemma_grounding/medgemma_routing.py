"""E-010: discovery-only single-layer routing and correct/error comparisons.

Stages: prepare -> existing (offline E-009 analysis) -> run -> report.
run includes the engineering benchmark and continues only when checks pass.
"""

import argparse
import csv
import logging
import platform
import subprocess
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from medgemma_grounding import medgemma_attention as attention
from medgemma_grounding import medgemma_interventions as base
from medgemma_grounding import routing_analysis as analysis
from medgemma_grounding.attention_knockout import conditions as old_conditions
from medgemma_grounding.medgemma_features import atomic_json, file_sha256, identity_hash
from medgemma_grounding.medgemma_preflight import tree_hashes, weight_identity

OUTPUT = Path("results/attention_errors_v3")
SOURCE = Path("results/attention_depth_v1")


def conditions():
    return [{"id": f"B_{i:02d}_{i:02d}", "route": "B", "start": i, "end": i} for i in range(34)]


def source_rows(source):
    p = base.read_json(source / "protocol.json")
    attention.validate(source, p)
    report = base.read_json(source / "report.json")
    if not report["complete"] or report["knockout_records"] != 2556:
        raise ValueError("Completed E-009 required")
    return [attention.read_record(source / "cases" / cid / f"{c['id']}.json", p)
            for cid in p["case_ids"] for c in p["conditions"]]


def metadata(manifest, data_root):
    """Filter known discovery paths immediately; never export other CSV rows."""
    ids = {c["case_id"]: c for c in manifest["cases"]}
    path = data_root / "CheXpert/val_labels.csv"
    result = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = "_".join(Path(row["Path"]).parts[-3:]).removesuffix(".jpg")
            if key not in ids:
                continue
            case = ids[key]
            if float(row["Pleural Effusion"]) != case["label"]:
                raise ValueError("CSV discovery label mismatch")
            result.append({"case_id": key, "patient": case["patient"],
                           "Sex": row["Sex"], "Age": float(row["Age"]),
                           "projection": row["AP/PA"], "view": row["Frontal/Lateral"],
                           "height": case["image_size"][0], "width": case["image_size"][1],
                           "annotation_status": str(case["annotation_status"]),
                           **{k: float(v) if v else None for k, v in row.items()
                              if k not in ("Path", "Sex", "Age", "AP/PA", "Frontal/Lateral", "Pleural Effusion")}})
    if len(result) != len(ids):
        raise ValueError("Incomplete discovery metadata")
    return result, file_sha256(path)


def prepare(output, device, data_root):
    path = output / "protocol.json"
    if path.exists():
        p = base.read_json(path)
        validate(output, p)
        if p["device"] != device:
            raise ValueError("New backend needs a separate protocol")
        return p
    output.mkdir(parents=True, exist_ok=True)
    old = base.read_json(SOURCE / "protocol.json")
    manifest = attention.validate(SOURCE, old)
    source_rows(SOURCE)
    if device != old["device"]:
        raise ValueError("This protocol implements matched local MPS; register fresh joint windows for another backend")
    if not (output / "methods.md").exists():
        raise ValueError("Reviewed methods required before protocol freeze")
    atomic_json(output / "discovery_manifest.json", manifest)
    meta, csv_sha = metadata(manifest, data_root)
    atomic_json(output / "discovery_metadata.json", meta)
    rows = source_rows(SOURCE)
    patients = analysis.patient_rows([r for r in rows if r["id"] == old["conditions"][0]["id"]])
    folds = analysis.fixed_folds(patients)
    atomic_json(output / "folds.json", folds)
    keep = ("model_id", "model_revision", "processor_revision", "prompt", "scoring", "dtype", "attention",
            "tolerance", "seed", "batch_size", "layer_types", "sliding_window", "implementation_sha256", "benchmark_ids")
    p = {"experiment": "E-010", **{k: old[k] for k in keep}, "device": device,
         "conditions": conditions(), "case_ids": old["case_ids"], "source_directory": str(SOURCE),
         "expected_knockouts": 4828, "expected_baselines": 142, "expected_cohort_forwards": 5112,
         "bootstrap": {"unit": "patient", "resamples": analysis.RESAMPLES, "seed": 42},
         "primary_family": [c["id"] for c in conditions()], "primary_contrast": "D_neg = FP minus TN delta_S",
         "folds": {"count": 5, "stratification": "expert label", "patient_disjoint": True, "seed": 42},
         "affine_slope_floor": 1e-8, "threshold_reference": 2.375,
         "local_runtime_limit_seconds": 10800, "rss_limit_bytes": 26 * 1024**3,
         "source_artifacts": {str(SOURCE): tree_hashes(SOURCE)}, "metadata_csv_sha256": csv_sha,
         "versions": {n: version(n) for n in ("torch", "transformers", "numpy", "pillow", "scipy")},
         "source_sha256": {f.name: file_sha256(f) for f in Path(__file__).parent.glob("*.py")},
         "files_sha256": {str(f.relative_to(output)): file_sha256(f) for f in output.rglob("*") if f.is_file()},
         "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
         "non_claims": ["Not an effusion circuit or clinical correction policy", "No held-out generalization",
                        "Knockouts can recruit fallback paths; group contrasts are confounded"]}
    p["protocol_hash"] = identity_hash(p)
    atomic_json(path, p)
    return p


def validate(output, p):
    if p["protocol_hash"] != identity_hash({k: v for k, v in p.items() if k != "protocol_hash"}):
        raise ValueError("Protocol hash mismatch")
    if p["conditions"] != conditions() or p["attention"] != attention.ATTENTION or p["tolerance"] != attention.TOLERANCE:
        raise ValueError("Fixed settings changed")
    for name, sha in p["source_sha256"].items():
        if file_sha256(Path(__file__).with_name(name)) != sha:
            raise ValueError(f"Runtime source changed: {name}; new protocol required")
    for name, sha in p["files_sha256"].items():
        if file_sha256(output / name) != sha:
            raise ValueError(f"Frozen input changed: {name}")
    manifest = base.read_json(output / "discovery_manifest.json")
    if ([r["case_id"] for r in manifest["cases"]] != p["case_ids"] or len(p["case_ids"]) != 142
            or len({r["patient"] for r in manifest["cases"]}) != 140 or manifest["test_access"]):
        raise ValueError("Discovery cohort changed")
    for directory, expected in p["source_artifacts"].items():
        if tree_hashes(Path(directory)) != expected:
            raise ValueError("Previous artifacts changed")
    return manifest


def existing(output, p):
    rows = source_rows(Path(p["source_directory"]))
    result = analysis.analyze(rows, output / "existing_windows", base.read_json(output / "folds.json"))
    expected = {"TP": 41, "FP": 14, "FN": 4, "TN": 83}
    if any(result["reference_threshold_2.375"][k] != v for k, v in expected.items()):
        raise ValueError("Retrospective threshold result did not reproduce")
    analysis.metadata_report(rows, base.read_json(output / "discovery_metadata.json"), output)
    return result


def engineering(bundle, p, manifest, output, data_root, info):
    # Reuse E-009's unchanged/pass/placebo, mask, full-depth isolation diagnostics.
    # The reused helper publishes a preliminary E-009 budget. Keep it internal:
    # interruption here must never publish an approved E-010 benchmark.
    benchmark = attention.engineering(bundle, {**p, "conditions": old_conditions()}, manifest,
                                      output / "engineering" / "core", data_root, info)
    times = list(benchmark["timed_forwards"])
    checks = []
    for cid in p["benchmark_ids"]:
        case = next(c for c in manifest["cases"] if c["case_id"] == cid)
        inputs, groups = attention.image_inputs(bundle, p, case, data_root)
        baseline, _ = attention.timed_forward(bundle, inputs)
        historical = torch.load(Path(p["source_directory"]) / "cases" / cid / "baseline.pt", weights_only=True, map_location="cpu")
        compatibility = base.identity_control(historical, baseline, p["scoring"], "E009_baseline_compatibility")
        for i in (0, 8, 13, 17, 33):
            c = p["conditions"][i]
            changed, detail = attention.timed_forward(bundle, inputs, groups, c, inspect=True, audit=True)
            times.append(detail["seconds"])
            attention.write_record(output / "engineering" / cid / f"{c['id']}.json",
                                   attention.measurement(case, p, baseline, changed, c, detail, attention.tensor_hash(inputs)))
        checks.append({"case_id": cid, "historical_compatibility": compatibility})
    forecast = float(np.quantile(times, .9)) * p["expected_cohort_forwards"] * 1.2
    memory = benchmark["memory"]
    comfortable = (memory["peak_process_rss_bytes"] < p["rss_limit_bytes"] and
                   memory.get("mps_driver_allocated", 0) < memory.get("mps_recommended_max", float("inf")))
    approved = forecast <= p["local_runtime_limit_seconds"] and comfortable
    result = {**benchmark, "timed_forwards": times, "median_seconds": float(np.median(times)),
              "p90_seconds": float(np.quantile(times, .9)), "forecast_forwards_upper": p["expected_cohort_forwards"],
              "forecast_seconds_with_20pct_allowance": forecast, "backend_approved": approved,
              "comfortable_memory": comfortable, "backend_decision": "local MPS" if approved else "single BF16 GPU access required",
              "historical_checks": checks, "single_layer_probability_checks": [0, 8, 13, 17, 33]}
    attention.write_record(output / "benchmark.json", {k: v for k, v in result.items() if k != "record_hash"})
    return result


def complete_case(output, p, cid):
    directory = output / "cases" / cid
    paths = [directory / "baseline.json", *[directory / f"{c['id']}.json" for c in p["conditions"]]]
    complete = True
    for path in paths:
        if path.exists():
            row = attention.read_record(path, p)
            if path.name != "baseline.json":
                condition = next(c for c in p["conditions"] if c["id"] + ".json" == path.name)
                if any(row[k] != v for k, v in condition.items()) or row["case_id"] != cid:
                    raise ValueError("Treatment condition/case drift")
        else:
            complete = False
    if (directory / "baseline.json").exists():
        header = attention.read_record(directory / "baseline.json", p)
        if header["case_id"] != cid or file_sha256(directory / "baseline.pt") != header["tensor_sha256"]:
            raise ValueError("Baseline checksum drift")
    return complete


def execute_case(bundle, p, case, output, data_root):
    cid = case["case_id"]
    directory = output / "cases" / cid
    inputs, groups = attention.image_inputs(bundle, p, case, data_root)
    digest = attention.tensor_hash(inputs)
    baseline, timing = attention.timed_forward(bundle, inputs)
    old_dir = Path(p["source_directory"]) / "cases" / cid
    old_header = base.read_json(old_dir / "baseline.json")
    if digest != old_header["input_sha256"] or file_sha256(old_dir / "baseline.pt") != old_header["tensor_sha256"]:
        raise ValueError("Historical input/cache provenance mismatch")
    old = torch.load(old_dir / "baseline.pt", weights_only=True, map_location="cpu")
    historical = base.identity_control(old, baseline, p["scoring"], "E009_baseline_compatibility")
    header_path = directory / "baseline.json"
    if header_path.exists():
        header = attention.read_record(header_path, p)
        if header["input_sha256"] != digest or file_sha256(directory / "baseline.pt") != header["tensor_sha256"]:
            raise ValueError("Resume baseline/input drift")
        base.identity_control(torch.load(directory / "baseline.pt", weights_only=True, map_location="cpu"), baseline, p["scoring"])
    else:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(baseline, directory / "baseline.pt.tmp")
        (directory / "baseline.pt.tmp").replace(directory / "baseline.pt")
        attention.write_record(header_path, {"protocol_hash": p["protocol_hash"], "case_id": cid,
            "input_sha256": digest, "tensor_sha256": file_sha256(directory / "baseline.pt"),
            "token_groups": groups, "historical_compatibility": historical, **timing})
    passed, _ = attention.timed_forward(bundle, inputs, groups, p["conditions"][-1], mode="pass")
    identity = base.identity_control(baseline, passed, p["scoring"])
    for condition in p["conditions"]:
        path = directory / f"{condition['id']}.json"
        if path.exists():
            row = attention.read_record(path, p)
            if row["input_sha256"] != digest or any(row[k] != v for k, v in condition.items()):
                raise ValueError("Resumed treatment input/condition drift")
            continue
        changed, detail = attention.timed_forward(bundle, inputs, groups, condition)
        row = attention.measurement(case, p, baseline, changed, condition, detail, digest)
        attention.write_record(path, {**row, "token_groups": groups, "identity_control": identity})


def run(stage, output=OUTPUT, device="mps", data_root=Path("data")):
    p = prepare(output, device, data_root)
    manifest = validate(output, p)
    if stage == "prepare":
        return p
    if stage == "existing":
        return existing(output, p)
    if stage == "report":
        return report(output, p, manifest)
    if stage == "verify":
        return verify(output, p, manifest)
    if list(output.glob("failure_*.json")):
        raise ValueError("Technical failure recorded; diagnose before dependent execution")
    if stage == "run" and all(complete_case(output, p, cid) for cid in p["case_ids"]):
        return report(output, p, manifest)  # completed resume never loads weights
    existing(output, p)
    try:
        bundle, info = attention.runtime(p)
        weights = weight_identity(bundle.model)
        previous = base.read_json(output / "runtime_checks.json") if (output / "runtime_checks.json").exists() else None
        if previous and any(previous[k] != info[k] for k in ("hardware", "platform", "device", "dtype", "attention")):
            raise ValueError("Execution environment changed; new protocol required")
        benchmark = attention.read_record(output / "benchmark.json", p) if (output / "benchmark.json").exists() else engineering(bundle, p, manifest, output, data_root, info)
        validate_benchmark(benchmark, p)
        if any(benchmark["runtime"][k] != info[k] for k in ("hardware", "platform", "device", "dtype", "attention")):
            raise ValueError("Benchmark execution environment changed")
        if not benchmark["backend_approved"]:
            raise RuntimeError("Compute gate failed: require actual single-GPU access/paths; no impractical local fallback")
        if stage == "run":
            start, newly = time.perf_counter(), 0
            completed_ids = {cid for cid in p["case_ids"] if complete_case(output, p, cid)}
            for case in manifest["cases"]:
                if case["case_id"] in completed_ids:
                    continue
                logging.info("Running %s", case["case_id"])
                execute_case(bundle, p, case, output, data_root)
                newly += 1
                elapsed = time.perf_counter() - start
                completed_ids.add(case["case_id"])
                completed = len(completed_ids)
                atomic_json(output / "progress.json", {"completed_cases": completed, "newly_completed": newly,
                    "elapsed_seconds": elapsed, "eta_seconds": (142 - completed) * elapsed / newly,
                    "last_case": case["case_id"]})
                logging.info("Saved %d/142; elapsed %.0fs, ETA %.0fs", completed, elapsed, (142-completed)*elapsed/newly)
        if weights != weight_identity(bundle.model) or any(v.requires_grad or v.grad is not None for v in bundle.model.parameters()):
            raise ValueError("Frozen parameter state changed")
        if any(m._forward_hooks or m._forward_pre_hooks for m in bundle.model.modules()):
            raise ValueError("Hooks leaked")
        attention.write_record(output / "runtime_checks.json", {"protocol_hash": p["protocol_hash"],
            "frozen_unchanged_no_gradients": True, "no_hooks_remain": True, **info})
        return benchmark if stage == "benchmark" else report(output, p, manifest)
    except Exception as error:
        attention.write_record(output / f"failure_{stage}.json", {"protocol_hash": p["protocol_hash"],
            "error": str(error), "traceback": traceback.format_exc(), "settings_changed": False})
        raise


def validate_benchmark(benchmark, p):
    if (benchmark.get("forecast_forwards_upper") != p["expected_cohort_forwards"]
            or benchmark.get("single_layer_probability_checks") != [0, 8, 13, 17, 33]
            or len(benchmark.get("historical_checks", [])) != 4
            or not benchmark.get("controls_passed") or not benchmark.get("full_depth_B_image_independence")):
        raise ValueError("Incomplete E-010 engineering prerequisites")


def report(output, p, manifest):
    rows = []
    for cid in p["case_ids"]:
        complete_case(output, p, cid)
        for c in p["conditions"]:
            path = output / "cases" / cid / f"{c['id']}.json"
            if path.exists():
                rows.append(attention.read_record(path, p))
    complete = len(rows) == p["expected_knockouts"] and all(complete_case(output, p, cid) for cid in p["case_ids"])
    if complete and not (output / "runtime_checks.json").exists():
        raise ValueError("Final runtime verification missing")
    if complete:
        result = analysis.analyze(rows, output / "single_layers", base.read_json(output / "folds.json"), primary_family=True)
        joint = analysis.joint_comparison(rows, source_rows(Path(p["source_directory"])))
        base.write_csv(output / "individual_vs_joint.csv", joint)
        analysis.joint_summary(joint, output)
        analysis.study_figures(output, result)
    else:
        result = {"status": "partial; no incomplete-family BH or score fit", "records": len(rows)}
    flat_keys = ("case_id", "patient", "label", "outcome", "id", "start", "end", "S", "changed_S", "M", "changed_M",
                 "delta_S", "delta_M", "corrected", "introduced", "answer_format_valid", "kl_from_baseline", "blocked_edges", "seconds")
    base.write_csv(output / "effects.csv", [{k: r[k] for k in flat_keys} for r in rows])
    result = {"experiment": "E-010", "protocol_hash": p["protocol_hash"], "complete": complete,
              "knockout_records": len(rows), "expected_knockouts": 4828,
              "completed_cases": sum(complete_case(output, p, cid) for cid in p["case_ids"]),
              "failures": [base.read_json(f) for f in output.glob("failure_*.json")],
              "analysis": result, "artifacts_sha256": {str(f.relative_to(output)): file_sha256(f)
                  for directory in (output / "single_layers", output / "existing_windows")
                  for f in directory.glob("*") if f.is_file()}}
    for name in ("effects.csv", "layer_profiles.png", "individual_vs_joint.csv", "individual_vs_joint_patients.csv", "individual_vs_joint_summary.json", "metadata_imbalance.json"):
        if (output / name).exists():
            result["artifacts_sha256"][name] = file_sha256(output / name)
    atomic_json(output / "report.json", result)
    return result


def verify(output, p, manifest):
    """Finite saved-artifact audit, including independently counted mask edges."""
    validate(output, p)
    rows = []
    for case in manifest["cases"]:
        cid = case["case_id"]
        if not complete_case(output, p, cid):
            raise ValueError("Incomplete cohort")
        directory = output / "cases" / cid
        header = attention.read_record(directory / "baseline.json", p)
        logits = torch.load(directory / "baseline.pt", map_location="cpu", weights_only=True)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite baseline")
        original = attention.answer_score(logits, p["scoring"])
        for c in p["conditions"]:
            r = attention.read_record(directory / f"{c['id']}.json", p)
            groups = r["token_groups"]
            expected_edges = len(groups["image"]) * len(groups["post_image"])
            if max(groups["valid"]) >= p["sliding_window"]:
                raise ValueError("Independent edge formula needs window-aware extension")
            if (r["blocked_edges"] != expected_edges or r["input_sha256"] != header["input_sha256"]
                    or r["image_sha256"] != case["image_sha256"] or r["patient"] != case["patient"]
                    or r["label"] != case["label"] or r["baseline"] != original):
                raise ValueError("Treatment accounting/provenance mismatch")
            if any(not np.isfinite(r[k]) for k in ("S", "changed_S", "delta_S", "M", "changed_M", "delta_M", "kl_from_baseline")):
                raise ValueError("Nonfinite treatment")
            if (r["changed_S"] - r["S"] != r["delta_S"]
                    or r["delta_M"] != (2*case["label"]-1)*r["delta_S"]):
                raise ValueError("Score algebra failed")
            for layer in r["layers"].values():
                if not layer["non_target_unchanged"] or not layer["no_fully_blocked_rows"]:
                    raise ValueError("Mask checks failed")
            rows.append(r)
    if len(rows) != 4828 or len({r["patient"] for r in rows}) != 140:
        raise ValueError("Population/coverage mismatch")
    checks = attention.read_record(output / "runtime_checks.json", p)
    if not checks["frozen_unchanged_no_gradients"] or not checks["no_hooks_remain"]:
        raise ValueError("Runtime checks failed")
    for f in diagnostic_paths(output):
        for layer in attention.read_record(f, p)["layers"].values():
            if layer["target_max_probability"] != 0:
                raise ValueError("Nonzero targeted attention")
    report_data = base.read_json(output / "report.json")
    before = {name: file_sha256(output / name) for name in ["report.json", *report_data["artifacts_sha256"]]}
    saved_runtime = attention.runtime
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume attempted model loading")
    try:
        attention.runtime = forbidden
        run("run", output, p["device"])
    finally:
        attention.runtime = saved_runtime
    if any(file_sha256(output / name) != sha for name, sha in before.items()):
        raise ValueError("Completed resume changed report artifacts")
    result = {"protocol_hash": p["protocol_hash"], "complete": True, "knockouts": len(rows),
              "images": 142, "patients": 140, "all_34_layers": True,
              "finite_score_mask_provenance_checks": True, "model_free_byte_identical_resume": True,
              "previous_artifacts_unchanged": True, "reserved_cohorts_opened": False,
              "artifacts_sha256": before}
    atomic_json(output / "verification.json", result)
    return result


def diagnostic_paths(output):
    """Only treatment diagnostics; internal benchmark summaries are not masks."""
    return sorted(f for f in (output / "engineering").rglob("*.json")
                  if f.name.startswith(("A_", "B_")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["prepare", "existing", "benchmark", "run", "report", "verify"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["mps", "cuda"], default="mps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.stage, args.output_dir, args.device, args.data_root)
    logging.info("Stage %s finished; complete=%s", args.stage, result.get("complete"))


if __name__ == "__main__":
    main()
