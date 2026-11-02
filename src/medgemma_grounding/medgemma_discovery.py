"""E-007: fresh eager-backend discovery with full-cohort calibration and fixed gates.

--stage run executes smoke -> full calibration -> qualified selection/ablation
-> offline report. Historical encodings are compatibility references only.
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from medgemma_grounding import medgemma_interventions as base
from medgemma_grounding.feature_analysis import feature_rankings
from medgemma_grounding.feature_interventions import GROUPS
from medgemma_grounding.medgemma_features import (
    atomic_json, file_sha256, identity_hash, load_discovery_case, answer_score,
)
from medgemma_grounding.medgemma_preflight import tree_hashes, weight_identity
from medgemma_grounding.sae_reconstruction import capture_residual, encode_decode

OUTPUT = Path("results/gemma_scope/interventions_v3_discovery")
PREFLIGHT = Path("results/gemma_scope/interventions_v2_preflight")


def protocol_for(source, output, preflight):
    source, output, preflight = (Path(p).resolve() for p in (source, output, preflight))
    previous = base.read_json(preflight / "protocol.json")
    report = base.read_json(preflight / "preflight_report.json")
    failed = Path(previous["failed_directory"])
    for protected in (source, preflight, failed):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Output must be separate from every preserved experiment")
    if (not report["complete"] or not report["repair_mechanics_passed"]
            or previous["protocol_hash"] != report["protocol_hash"]
            or previous["protocol_hash"] != identity_hash({k: v for k, v in previous.items() if k != "protocol_hash"})
            or previous["conditional_fix"] != {"text_config": "eager", "vision_config": "sdpa"}
            or previous["source_protocol_hash"] != base.read_json(source / "protocol.json")["protocol_hash"]):
        raise ValueError("Verified repaired-backend preflight required")
    repaired = {"smoke_ids": previous["case_ids"], "preflight_sha256": file_sha256(preflight / "preflight_report.json"),
                "previous_artifacts": {str(p): tree_hashes(p) for p in (preflight, failed)}}
    protocol, records = base.experiment_protocol(source, output, repaired=repaired)
    if len(protocol["smoke_ids"]) != 2 or not set(protocol["smoke_ids"]) <= set(protocol["discovery_case_ids"]):
        raise ValueError("Expected the two fixed discovery smoke cases")
    return protocol, records


def ensure_case(bundle, sae, protocol, output, case_id):
    path = output / "cases" / f"{case_id}.json"
    if not path.exists():
        original, reference = load_discovery_case(Path(protocol["source_directory"]), case_id)
        record, payload = base.collect_attribution(bundle, sae, protocol, original, reference, pilot=True)
        base.save_case(output, protocol, record, payload)
    return base.load_case(output, protocol, case_id)


def treatment_context(bundle, sae, protocol, record, payload):
    """Recheck original state, and ALWAYS use the new backend's saved encoding."""
    original, _ = load_discovery_case(Path(protocol["source_directory"]), record["case_id"])
    inputs, _ = base.prepared_inputs(bundle, protocol, original)
    baseline = base.forward(bundle, inputs)
    base.identity_control(payload["baseline_logits"], baseline, protocol["scoring"], "resumed_baseline")
    hidden, logits = capture_residual(bundle.model, inputs, protocol["hook"])
    base.identity_control(baseline, logits, protocol["scoring"], "resumed_capture")
    torch.testing.assert_close(hidden, payload["hidden"], rtol=1e-5, atol=1e-5)
    _, encoded = encode_decode(sae, hidden[0])
    features = payload["features"].to_dense()
    torch.testing.assert_close(encoded, features, rtol=1e-5, atol=1e-5)
    return inputs, features, {**record, "answer_score": answer_score(baseline, protocol["scoring"])}, {
        **payload, "baseline_logits": baseline}


def measure_specs(bundle, sae, protocol, output, record, payload, specs):
    missing = []
    for feature, group, alpha in specs:
        path = base.treatment_path(output, record["case_id"], feature, group, alpha)
        if path.exists():
            base.load_treatment(path, protocol, record["case_id"], feature, group, alpha)
        else:
            missing.append((feature, group, alpha))
    if not missing:
        return
    inputs, features, record, payload = treatment_context(bundle, sae, protocol, record, payload)
    for feature, group, alpha in missing:
        started = time.perf_counter()
        treatment = base.measure_treatment(bundle, protocol, record, payload, features, sae.W_dec,
                                           inputs, feature, group, alpha)
        treatment["runtime_seconds"] = time.perf_counter() - started
        base.save_treatment(base.treatment_path(output, record["case_id"], feature, group, alpha), treatment)
        logging.info("Saved %s j=%d alpha=%g actual=%+.4f predicted=%+.4f (%.1fs)",
                     group, feature, alpha, treatment["actual_delta"], treatment["predicted_delta"], treatment["runtime_seconds"])


def calibration_specs(record, payload, *, smoke=False):
    specs = []
    for group in GROUPS:
        features = base.calibration_features(record["case_id"], group, payload["groups"][group])
        for feature in features[:1] if smoke else features:
            specs.extend((feature, group, alpha) for alpha in base.FRACTIONS)
    return specs


def case_pending(output, protocol, case_id, *, calibration=False, ablations=None):
    if not (output / "cases" / f"{case_id}.json").exists():
        return True
    record, payload = base.load_case(output, protocol, case_id)
    specs = calibration_specs(record, payload) if calibration else ablations or []
    missing = False
    for feature, group, alpha in specs:
        path = base.treatment_path(output, case_id, feature, group, alpha)
        if path.exists():
            base.load_treatment(path, protocol, case_id, feature, group, alpha)
        else:
            missing = True
    return missing


def smoke_report(output, protocol):
    records, treatments = [], {}
    for case_id in protocol["smoke_ids"]:
        record, payload = base.load_case(output, protocol, case_id)
        if not all(c["allclose"] and c["decision_preserved"] for c in record["controls"].values()):
            raise ValueError("Failed numerical smoke controls")
        if len(record["repeated_scores"]) != 4:
            raise ValueError("Numerical smoke lacks repeat controls")
        for feature, group, alpha in calibration_specs(record, payload, smoke=True):
            path = base.treatment_path(output, case_id, feature, group, alpha)
            base.load_treatment(path, protocol, case_id, feature, group, alpha)
            treatments[str(path.relative_to(output))] = file_sha256(path)
        records.append({"case_id": case_id, "tensor_sha256": record["tensor_sha256"], "controls": record["controls"]})
    result = {"protocol_hash": protocol["protocol_hash"], "complete": len(records) == 2,
              "cases": records, "treatment_sha256": treatments, "purpose": "numerical mechanics, not calibration qualification"}
    result["smoke_hash"] = identity_hash(result)
    atomic_json(output / "smoke_report.json", result)
    return result


def supporting_rankings(records, payloads, calibration):
    """Attribution primary; measured support explicit; associations descriptive."""
    rows = base.patient_rankings(records, payloads)
    patients = sorted({r["patient"] for r in records})
    metadata, activities = [], {g: [] for g in GROUPS}
    for patient in patients:
        indices = [i for i, r in enumerate(records) if r["patient"] == patient]
        metadata.append({"label": max(records[i]["label"] for i in indices),
                         "answer_score": np.mean([records[i]["answer_score"]["score"] for i in indices])})
        for group in GROUPS:
            activities[group].append(np.mean([payloads[i]["groups"][group]["activation_mean"].numpy() for i in indices], axis=0))
    descriptions = feature_rankings(metadata, {g: np.stack(v) for g, v in activities.items()},
                                    min_active=0, representations=GROUPS)
    lookup = {(r["representation"], r["feature"]): r for r in descriptions if r["target"] == "label"}
    measured = defaultdict(list)
    for measurement in calibration["measurements"]:
        measured[(measurement["group"], measurement["feature"], measurement["alpha"])].append(measurement)
    for row in rows:
        group, feature = row["group"], row["feature"]
        desc = lookup[group, feature]
        row.update(supporting_label_auc=desc["label_auc"], supporting_answer_spearman=desc["answer_spearman"],
                   group_qualified=group in calibration["qualifying_groups"])
        for alpha, name in ((.5, "half"), (1., "full")):
            measurements = measured[group, feature, alpha]
            row[f"calibration_{name}_cases"] = len(measurements)
            if measurements:
                grouped = defaultdict(list)
                for measurement in measurements:
                    grouped[measurement["patient"]].append(measurement["actual_delta"])
                values = [np.mean(v) for v in grouped.values()]
                row[f"measured_{name}_mean_signed"] = float(np.mean(values))
                row[f"measured_{name}_mean_absolute"] = float(np.mean(np.abs(values)))
            else:
                row[f"measured_{name}_mean_signed"] = None
                row[f"measured_{name}_mean_absolute"] = None
    return rows


def rank_all(output, protocol, calibration):
    records, payloads = [], []
    for case_id in protocol["discovery_case_ids"]:
        record, payload = base.load_case(output, protocol, case_id)
        records.append(record)
        payloads.append({"groups": payload["groups"]})
    rows = supporting_rankings(records, payloads, calibration)
    base.write_csv(output / "attribution_rankings.csv", rows)
    return rows


def report(output, protocol):
    """Offline final or partial evidence, including calibration when no group passes."""
    for directory, expected in protocol["previous_artifacts"].items():
        if tree_hashes(Path(directory)) != expected:
            raise ValueError("A preserved historical experiment changed")
    calibration = base.pilot_report(output, protocol)
    rows = calibration["measurements"]
    base.write_csv(output / "calibration_effects.csv", rows)
    result = base.offline_report(output, protocol)
    records = [base.load_case(output, protocol, case_id)[0] for case_id in protocol["discovery_case_ids"]
               if (output / "cases" / f"{case_id}.json").exists()]
    if len(records) == len(protocol["discovery_case_ids"]):
        rank_all(output, protocol, calibration)
    failures = sorted(output.glob("failure_*.json"))
    reason = base.read_json(failures[-1])["error"] if failures else (
        "No position group meets the fixed calibration criteria" if calibration["complete"] and not calibration["qualifying_groups"] else None)
    selected = (output / "candidates.json").exists()
    stopped = calibration["complete"] and not calibration["qualifying_groups"]
    result.pop("pilot_complete", None)
    result.update(experiment="E-007", fresh_encodings=True, calibration_complete=calibration["complete"],
        calibration_cases=calibration["completed_cases"], calibration_measurements=len(rows), stop_reason=reason,
        repaired_outcomes=dict(Counter(r["outcome"] for r in records)),
        patients=len({r["patient"] for r in records}),
        changed_decisions_vs_reference=sum(r["answer_score"]["decision"] != r["reference_sdpa"]["answer_score"]["decision"] for r in records),
        score_ties=sum(r["answer_score"]["tie"] for r in records),
        minimum_candidate_probability_mass=min((r["answer_score"]["candidate_probability_mass"] for r in records), default=None),
        stages={"smoke": "complete" if (output / "smoke_report.json").exists() else "not_complete",
                "fresh_attribution": "complete" if len(records) == len(protocol["discovery_case_ids"]) else "partial",
                "calibration": "complete" if calibration["complete"] else "partial",
                "selection": "skipped_calibration" if stopped else "complete" if selected else "not_started",
                "ablations": "skipped_calibration" if stopped else "complete" if selected and result["measured_treatments"] == result["expected_treatments"] else "not_complete",
                "report": "complete"},
        calibration_by_group={g: {"status": calibration["groups"][g]["status"],
            "agreement_half": calibration["groups"][g]["half"], "agreement_full": calibration["groups"][g]["full"],
            "all": base.bootstrap_summary([r for r in rows if r["group"] == g]),
            "correct_answers": base.bootstrap_summary([r for r in rows if r["group"] == g and r["original_correct"]]),
            "outcomes": {k: base.bootstrap_summary([r for r in rows if r["group"] == g and r["outcome"] == k]) for k in base.QUOTAS}}
            for g in GROUPS},
        interpretation="Calibration summaries mix checked feature IDs; fixed-feature cohort effects exist only after gated selection/ablations",
        preserved_previous_artifacts=True)
    checks_path = output / "runtime_checks.json"
    if checks_path.exists():
        result["runtime_checks"] = base.read_json(checks_path)
    if failures:
        result.update(complete=False, status="technical_failure")
    atomic_json(output / "intervention_report.json", result)
    return result


def run(stage, source=base.SOURCE, output=OUTPUT, preflight=PREFLIGHT, limit=None):
    if stage not in {"run", "smoke", "calibrate", "select", "ablate", "report"}:
        raise ValueError("Explicit workflow stage required")
    if limit is not None and (limit < 1 or stage not in {"calibrate", "ablate"}):
        raise ValueError("--limit supports calibrate/ablate only and counts newly processed cases")
    output = Path(output)
    protocol, _ = protocol_for(source, output, preflight)
    if stage == "report":
        return report(output, protocol)
    if any(output.glob("failure_*.json")):
        raise ValueError("A technical failure is recorded; diagnose it before authorizing any resumed model work")
    bundle = sae = weights = None
    active_stage = stage

    def runtime():
        nonlocal bundle, sae, weights
        if bundle is None:
            bundle, sae = base.frozen_runtime(protocol, output)
            weights = weight_identity(bundle.model)
        return bundle, sae

    def finish():
        if bundle is not None:
            checks = {"protocol_hash": protocol["protocol_hash"],
                      "parameter_identity_versions_unchanged": weights == weight_identity(bundle.model),
                      "parameter_gradients_unset": all(p.grad is None for p in bundle.model.parameters()),
                      "parameters_frozen": all(not p.requires_grad for p in bundle.model.parameters())}
            atomic_json(output / "runtime_checks.json", checks)
            if not all(checks[k] for k in checks if k != "protocol_hash"):
                raise AssertionError("Model parameters/gradient state changed")
        return report(output, protocol)

    def process(case_ids, specs_for, phase, *, calibration=False):
        pending = [c for c in case_ids if case_pending(output, protocol, c, calibration=calibration,
                                                     ablations=specs_for if isinstance(specs_for, list) else None)]
        if limit is not None:
            pending = pending[:limit]
        start = time.perf_counter()
        for i, case_id in enumerate(pending, 1):
            logging.info("%s %d/%d %s", phase, i, len(pending), case_id)
            runtime()
            record, payload = ensure_case(bundle, sae, protocol, output, case_id)
            specs = specs_for(record, payload) if callable(specs_for) else specs_for
            measure_specs(bundle, sae, protocol, output, record, payload, specs)
            elapsed = time.perf_counter() - start
            atomic_json(output / "progress.json", {"stage": phase, "new_cases": i, "pending_initially": len(pending),
                "last_case": case_id, "elapsed_seconds": elapsed, "eta_seconds": elapsed / i * (len(pending) - i)})
            logging.info("%s saved; elapsed %.1fm; empirical ETA %.1fm", phase, elapsed / 60, elapsed / i * (len(pending) - i) / 60)

    try:
        if stage in {"run", "smoke"}:
            active_stage = "smoke"
            # Smoke uses the same selected top calibration feature; its work is retained.
            for case_id in protocol["smoke_ids"]:
                if case_pending(output, protocol, case_id):
                    runtime()
                    ensure_case(bundle, sae, protocol, output, case_id)
                record, payload = base.load_case(output, protocol, case_id)
                specs = calibration_specs(record, payload, smoke=True)
                missing = any(not base.treatment_path(output, case_id, *spec).exists() for spec in specs)
                if missing:
                    runtime()
                    measure_specs(bundle, sae, protocol, output, record, payload, specs)
            smoke_report(output, protocol)
            logging.info("Two-case numerical smoke passed")
        else:
            if not (output / "smoke_report.json").exists():
                raise ValueError("Numerical smoke prerequisite is incomplete")
            smoke_report(output, protocol)
        if stage in {"run", "calibrate"}:
            active_stage = "calibrate"
            ids = protocol["smoke_ids"] + [c for c in protocol["discovery_case_ids"] if c not in protocol["smoke_ids"]]
            process(ids, calibration_specs, "calibrate", calibration=True)
        if stage in {"run", "select", "ablate"}:
            calibration = base.pilot_report(output, protocol)
            if not calibration["complete"]:
                raise ValueError("Full-discovery calibration is incomplete")
            if not calibration["qualifying_groups"]:
                logging.info("Stopping: no groups meet fixed full-cohort calibration criteria")
                return finish()
            if stage in {"run", "select"}:
                active_stage = "select"
                base.cohort_selection(output, protocol, rank_all(output, protocol, calibration))
            if stage in {"run", "ablate"}:
                active_stage = "ablate"
                candidates = base.read_candidates(output, protocol)
                specs = [(j, g, a) for j, g, a, _ in base.treatment_specs(candidates)]
                process(protocol["discovery_case_ids"], specs, "ablate")
        return finish()
    except Exception as error:
        atomic_json(output / f"failure_{active_stage}.json", {"protocol_hash": protocol["protocol_hash"],
            "stage": active_stage, "error": str(error), "diagnostics": getattr(error, "diagnostics", None),
            "traceback": traceback.format_exc()})
        try:
            report(output, protocol)
        except Exception:
            logging.exception("Could not regenerate partial report; original diagnostic retained")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["run", "smoke", "calibrate", "select", "ablate", "report"])
    parser.add_argument("--source-dir", type=Path, default=base.SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--preflight-dir", type=Path, default=PREFLIGHT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.stage, args.source_dir, args.output_dir, args.preflight_dir, args.limit)
    logging.info("Workflow status=%s stages=%s reason=%s", result["status"], result["stages"], result["stop_reason"])


if __name__ == "__main__":
    main()
