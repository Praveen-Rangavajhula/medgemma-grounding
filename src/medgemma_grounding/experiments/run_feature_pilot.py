"""Run the gated E-005 pilot stages; saved reporting is model-free."""


from __future__ import annotations

from pathlib import Path
import argparse
import logging
import time
import traceback

import medgemma_grounding.interventions.feature_runtime as feature_runtime
import torch

from medgemma_grounding.analysis.feature_effects import (
    calibration_features,
    cohort_selection,
    offline_report,
    pilot_report,
    read_candidates,
)
from medgemma_grounding.experiments.replay import replay_if_historical
from medgemma_grounding.interventions.features import GROUPS
from medgemma_grounding.interventions.protocol import DESTINATION, FRACTIONS, SOURCE, experiment_protocol
from medgemma_grounding.model import answer_score
from medgemma_grounding.sae.reconstruction import capture_residual
from medgemma_grounding.utils.io import atomic_json


def run(stage, source=SOURCE, output=DESTINATION, limit=None):
    """Explicit stage entry point; prerequisites precede model loading."""
    if stage not in {"pilot", "attribute", "select", "ablate", "report"}:
        raise ValueError("An explicit valid stage is required")
    if limit is not None and (limit < 1 or stage not in {"attribute", "ablate"}):
        raise ValueError("--limit is positive and supported only for attribute/ablate")
    historical = replay_if_historical('medgemma_interventions', stage=stage, source=source, output=output, limit=limit)
    if historical is not None:
        return historical
    directory = Path(output)
    protocol, source_records = experiment_protocol(source, directory)
    try:
        if stage == "report":
            return offline_report(directory, protocol)
        if stage == "select":
            return cohort_selection(directory, protocol)
        if stage != "pilot":
            pilot = pilot_report(directory, protocol)
            if not pilot["complete"] or not pilot["qualifying_groups"]:
                raise ValueError("Pilot must complete and qualify a group before model expansion")
        candidates = read_candidates(directory, protocol) if stage == "ablate" else None
        by_id = {r["case_id"]: r for r in source_records}
        ids = protocol["pilot"]["case_ids"] if stage == "pilot" else protocol["discovery_case_ids"]
        pending = []
        for case_id in ids:
            path = directory / "cases" / f"{case_id}.json"
            loaded = feature_runtime.load_case(directory, protocol, case_id) if path.exists() else None
            if stage == "ablate" and loaded is None:
                raise ValueError("Ablation requires complete saved attribution")
            if stage == "attribute" and loaded:
                continue
            if stage == "pilot" and loaded:
                _, payload = loaded
                specs = [(j, g, a, "calibration") for g in GROUPS
                         for j in calibration_features(case_id, g, payload["groups"][g]) for a in FRACTIONS]
            elif stage == "ablate":
                specs = feature_runtime.treatment_specs(candidates)
            else:
                specs = []
            missing = loaded is None
            for feature, group, alpha, _ in specs:
                treatment = feature_runtime.treatment_path(directory, case_id, feature, group, alpha)
                if treatment.exists():
                    feature_runtime.load_treatment(treatment, protocol, case_id, feature, group, alpha)
                else:
                    missing = True
            if missing:
                pending.append(case_id)
        if not pending:
            return pilot_report(directory, protocol) if stage == "pilot" else {"new_cases": 0}
        if limit is not None:
            pending = pending[:limit]
        bundle, sae = feature_runtime.frozen_runtime(protocol, directory)
        started = time.perf_counter()
        for index, case_id in enumerate(pending, 1):
            case_start = time.perf_counter()
            logging.info("%s %d/%d %s", stage, index, len(pending), case_id)
            source_record, source_payload = feature_runtime.load_discovery_case(Path(source), case_id)
            path = directory / "cases" / f"{case_id}.json"
            if not path.exists():
                record, payload = feature_runtime.collect_attribution(bundle, sae, protocol, source_record, source_payload, pilot=stage == "pilot")
                feature_runtime.save_case(directory, protocol, record, payload)
            record, payload = feature_runtime.load_case(directory, protocol, case_id)
            if stage in {"pilot", "ablate"}:
                inputs, _ = feature_runtime.prepared_inputs(bundle, protocol, source_record)
                baseline = feature_runtime.forward(bundle, inputs)
                feature_runtime.identity_control(payload["baseline_logits"], baseline, protocol["scoring"])
                # All direct effects use this pass's baseline after saved-state alignment.
                payload = {**payload, "baseline_logits": baseline}
                record = {**record, "answer_score": answer_score(baseline, protocol["scoring"])}
                hidden, captured = capture_residual(bundle.model, inputs, protocol["hook"])
                feature_runtime.identity_control(baseline, captured, protocol["scoring"])
                torch.testing.assert_close(hidden, payload["hidden"], rtol=1e-5, atol=1e-5)
                features = source_payload["features"].to_dense()
                specs = feature_runtime.treatment_specs(candidates) if stage == "ablate" else [
                    (j, g, a, "calibration") for g in GROUPS
                    for j in calibration_features(case_id, g, payload["groups"][g]) for a in FRACTIONS]
                for feature, group, alpha, _ in specs:
                    path = feature_runtime.treatment_path(directory, case_id, feature, group, alpha)
                    if path.exists():
                        feature_runtime.load_treatment(path, protocol, case_id, feature, group, alpha)
                        continue
                    treatment_start = time.perf_counter()
                    measured = feature_runtime.measure_treatment(bundle, protocol, record, payload, features, sae.W_dec,
                                                 inputs, feature, group, alpha)
                    measured["runtime_seconds"] = time.perf_counter() - treatment_start
                    feature_runtime.save_treatment(path, measured)
                    logging.info("Saved %s j=%d alpha=%g delta=%+.4f (%.1fs)", group, feature, alpha,
                                 measured["actual_delta"], measured["runtime_seconds"])
            logging.info("Saved case %.1fs; elapsed %.1fm; empirical remaining ETA %.1fm", time.perf_counter() - case_start,
                         (time.perf_counter() - started) / 60,
                         (time.perf_counter() - started) / index * (len(pending) - index) / 60)
        return pilot_report(directory, protocol) if stage == "pilot" else {"new_cases": len(pending)}
    except Exception as error:
        atomic_json(directory / f"failure_{stage}.json", {"protocol_hash": protocol["protocol_hash"],
            "stage": stage, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "diagnostics": getattr(error, "diagnostics", None),
            "case_id": locals().get("case_id"), "settings_changed": False})
        if stage == "pilot":
            pilot_report(directory, protocol)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["pilot", "attribute", "select", "ablate", "report"])
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.stage, args.source_dir, args.output_dir, args.limit)
    logging.info("Stage %s finished: %s", args.stage, {k: v for k, v in result.items() if k in
                 {"complete", "status", "qualifying_groups", "cases", "new_cases"}})


if __name__ == "__main__":
    main()
