"""Patient-level calibration, selection and saved feature-intervention reports."""


from __future__ import annotations

from pathlib import Path

from scipy.stats import spearmanr
import numpy as np
import torch

from medgemma_grounding.interventions.feature_runtime import (
    load_case,
    load_treatment,
    treatment_path,
    treatment_specs,
)
from medgemma_grounding.interventions.features import GROUPS
from medgemma_grounding.interventions.protocol import FRACTIONS, QUOTAS, THRESHOLDS
from medgemma_grounding.utils.io import (
    atomic_json,
    file_sha256,
    hashed,
    identity_hash,
    read_json,
    write_csv,
)


def calibration_features(case_id, group, summary):
    active = summary["active"].nonzero().flatten().tolist()
    ranked = sorted(active, key=lambda j: (-abs(summary["effect"][j].item()), j))
    if len(ranked) <= 2:
        return ranked
    chosen = [ranked[0], ranked[len(ranked) // 2]]
    remaining = [j for j in active if j not in chosen]
    chosen.append(min(remaining, key=lambda j: (hashed(f"42:{case_id}:{group}:{j}"), j)))
    return chosen


def agreement(rows, tau):
    measurable = [r for r in rows if abs(r["actual_delta"]) >= tau]
    if not measurable:
        return {"pairs": 0, "spearman": None, "sign_agreement": None, "median_relative_error": None}
    predicted, actual = (np.array([r[k] for r in measurable]) for k in ("predicted_delta", "actual_delta"))
    rho = float(spearmanr(predicted, actual).statistic) if np.ptp(predicted) and np.ptp(actual) else None
    return {"pairs": len(measurable), "spearman": rho,
            "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
            "median_relative_error": float(np.median(abs(predicted - actual) / np.maximum(abs(actual), tau)))}


def pilot_report(directory, protocol):
    """Legacy pilot or full-discovery calibration, as fixed in the protocol."""
    records, rows = [], []
    complete = True
    completed_cases = 0
    full = "calibration" in protocol
    for case_id in (protocol["calibration"]["case_ids"] if full else protocol["pilot"]["case_ids"]):
        if not (directory / "cases" / f"{case_id}.json").exists():
            complete = False
            continue
        record, payload = load_case(directory, protocol, case_id)
        records.append(record)
        case_complete = True
        if len(record["repeated_scores"]) != 4 or not (record.get("fresh_encodings") if full else record["pilot"]):
            raise ValueError("Calibration lacks fresh encoding or repeated baselines")
        for group in GROUPS:
            for feature in calibration_features(case_id, group, payload["groups"][group]):
                for alpha in FRACTIONS:
                    path = treatment_path(directory, case_id, feature, group, alpha)
                    if path.exists():
                        rows.append(load_treatment(path, protocol, case_id, feature, group, alpha))
                    else:
                        complete = False
                        case_complete = False
        completed_cases += int(case_complete)
    noise = max((r["baseline_variation"] for r in records), default=0)
    tau = max(THRESHOLDS["floor_logits"], THRESHOLDS["noise_multiplier"] * noise)
    groups = {}
    for group in GROUPS:
        half = agreement([r for r in rows if r["group"] == group and r["alpha"] == 0.5], tau)
        enough = half["pairs"] >= THRESHOLDS["min_pairs"]
        qualified = complete and enough and half["spearman"] is not None and (
            half["spearman"] >= THRESHOLDS["spearman"] and half["sign_agreement"] >= THRESHOLDS["sign_agreement"]
            and half["median_relative_error"] <= THRESHOLDS["relative_error"])
        groups[group] = {"status": "qualifies" if qualified else "insufficient_signal" if not enough else "poor_approximation",
                         "half": half, "full": agreement([r for r in rows if r["group"] == group and r["alpha"] == 1], tau)}
    result = {"protocol_hash": protocol["protocol_hash"], "complete": complete, "cases": len(records),
              "tau": tau, "largest_baseline_variation": noise, "groups": groups,
              "qualifying_groups": [k for k, v in groups.items() if v["status"] == "qualifies"],
              "measurements": rows, "failure_diagnostics": sorted(p.name for p in directory.glob(
                  "failure_calibrate.json" if full else "failure_pilot.json"))}
    result["report_hash"] = identity_hash(result)
    if full:
        result.update(completed_cases=completed_cases, calibration_kind="full_discovery")
        result["report_hash"] = identity_hash({k: v for k, v in result.items() if k != "report_hash"})
    atomic_json(directory / ("calibration_report.json" if full else "pilot_report.json"), result)
    return result


def patient_rankings(records, payloads):
    """Average images within patients before any cohort mean or ranking."""
    patients = sorted({r["patient"] for r in records})
    rows = []
    for group in GROUPS:
        effects = np.stack([p["groups"][group]["effect"].numpy() for p in payloads])
        active = np.stack([p["groups"][group]["active"].numpy() for p in payloads])
        rms = np.stack([p["groups"][group]["removal_rms"].numpy() for p in payloads])
        l2 = np.stack([p["groups"][group]["removal_l2"].numpy() for p in payloads])
        patient_effects, patient_active, patient_rms, patient_l2 = [], [], [], []
        for patient in patients:
            ix = [i for i, r in enumerate(records) if r["patient"] == patient]
            patient_effects.append(effects[ix].mean(0))
            patient_active.append(active[ix].any(0))
            patient_rms.append((rms[ix] ** 2).mean(0))
            patient_l2.append((l2[ix] ** 2).mean(0))
        means = np.stack(patient_effects)
        activity = np.stack(patient_active).sum(0)
        rms_mean, l2_mean = np.sqrt(np.mean(patient_rms, axis=0)), np.sqrt(np.mean(patient_l2, axis=0))
        by_outcome = {}
        for category in QUOTAS:
            grouped = [effects[[i for i, r in enumerate(records) if r["patient"] == patient and r["outcome"] == category]].mean(0)
                       for patient in patients if any(r["patient"] == patient and r["outcome"] == category for r in records)]
            by_outcome[category] = np.mean(grouped, axis=0) if grouped else None
        for feature in range(means.shape[1]):
            rows.append({"feature": feature, "group": group,
                         "mean_absolute_effect": float(abs(means[:, feature]).mean()),
                         "mean_signed_effect": float(means[:, feature].mean()),
                         "active_patients": int(activity[feature]), "activity_fraction": float(activity[feature] / len(patients)),
                         "removal_rms": float(rms_mean[feature]), "removal_l2": float(l2_mean[feature]),
                         **{f"{k}_mean_signed": float(v[feature]) if v is not None else None for k, v in by_outcome.items()}})
    for group in GROUPS:
        ordered = sorted((r for r in rows if r["group"] == group), key=lambda r: (-r["mean_absolute_effect"], r["feature"]))
        for rank, row in enumerate(ordered, 1):
            row["rank"] = rank
    return rows


def matched_candidates(rows, qualifying):
    pairs = []
    for group in qualifying:
        pool = [r for r in rows if r["group"] == group and r["active_patients"] > 0]
        supported = [r for r in pool if r.get("calibration_half_cases", 1) and r.get("calibration_full_cases", 1)]
        targets = sorted(supported, key=lambda r: (-r["mean_absolute_effect"], r["feature"]))[:5]
        excluded = {r["feature"] for r in targets}
        for target in targets:
            available = [r for r in pool if r["feature"] not in excluded]
            if not available:
                raise ValueError("Insufficient active comparison features")
            def distance(row):
                return ((row["activity_fraction"] - target["activity_fraction"]) ** 2
                        + (np.log(row["removal_rms"] + 1e-12) - np.log(target["removal_rms"] + 1e-12)) ** 2)
            comparison = min(available, key=lambda r: (distance(r), r["feature"]))
            excluded.add(comparison["feature"])
            pairs.append({"group": group, "target": target, "comparison": comparison,
                          "matching_distance": float(distance(comparison))})
    return pairs


def cohort_selection(directory, protocol, ranking_rows=None):
    pilot = pilot_report(directory, protocol)
    if not pilot["complete"] or not pilot["qualifying_groups"]:
        raise ValueError("Completed qualifying pilot required for selection")
    records, payloads = [], []
    for case_id in protocol["discovery_case_ids"]:
        if not (directory / "cases" / f"{case_id}.json").exists():
            raise ValueError("Selection requires complete cohort attribution")
        record, payload = load_case(directory, protocol, case_id)
        records.append(record)
        # Keep only compact CPU summaries, not full shards/hidden states.
        payloads.append({"groups": payload["groups"]})
    rows = patient_rankings(records, payloads) if ranking_rows is None else ranking_rows
    pairs = matched_candidates(rows, pilot["qualifying_groups"])
    old = read_json(Path(protocol["source_directory"]) / "shortlist.json")
    calibration_key = "calibration_report_hash" if protocol.get("fresh_encodings") else "pilot_report_hash"
    result = {"protocol_hash": protocol["protocol_hash"], calibration_key: pilot["report_hash"],
              "input_tensor_sha256": {r["case_id"]: r["tensor_sha256"] for r in records},
              "pairs": pairs, "old_shortlist_ranks": [r for r in rows if r["feature"] in {c["feature"] for c in old["candidates"]}],
              "qualifying_groups": pilot["qualifying_groups"], "interpretation": "Discovery-selected interventions; comparisons are not nulls"}
    result["candidates_hash"] = identity_hash(result)
    path = directory / "candidates.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("Frozen intervention candidates changed")
    write_csv(directory / "attribution_rankings.csv", rows)
    atomic_json(path, result)
    return result


def read_candidates(directory, protocol):
    candidates = read_json(directory / "candidates.json")
    if (candidates["protocol_hash"] != protocol["protocol_hash"] or
            candidates["candidates_hash"] != identity_hash({k: v for k, v in candidates.items() if k != "candidates_hash"})):
        raise ValueError("Candidate freeze checksum mismatch")
    pilot = pilot_report(directory, protocol)
    calibration_key = "calibration_report_hash" if protocol.get("fresh_encodings") else "pilot_report_hash"
    if candidates[calibration_key] != pilot["report_hash"]:
        raise ValueError("Pilot calibration changed after selection")
    for case_id, digest in candidates["input_tensor_sha256"].items():
        record, _ = load_case(directory, protocol, case_id)
        if record["tensor_sha256"] != digest:
            raise ValueError("Attribution shard changed after selection")
    return candidates


def bootstrap_summary(rows):
    """Descriptive patient bootstrap, conditional on discovery selection."""
    patients = sorted({r["patient"] for r in rows})
    keys = ("actual_delta", "absolute_effect", "predicted_delta", "kl_from_baseline",
            "realized_l2", "corrected", "introduced")
    data = np.array([[np.mean([abs(r["actual_delta"]) if key == "absolute_effect" else r[key]
                              for r in rows if r["patient"] == patient]) for key in keys] for patient in patients])
    if not len(data):
        return {"patients": 0}
    samples = np.random.default_rng(42).integers(0, len(data), size=(1000, len(data)))
    boot = data[samples].mean(1)
    return {"patients": len(patients), "images": len(rows),
            **{key: {"mean": float(data[:, i].mean()), "ci95": np.quantile(boot[:, i], [.025, .975]).tolist()}
               for i, key in enumerate(keys)}}


def offline_report(directory, protocol):
    pilot = pilot_report(directory, protocol)
    rows, summaries, contrasts = [], [], []
    expected = 0
    candidates = read_candidates(directory, protocol) if (directory / "candidates.json").exists() else None
    if candidates:
        specs = treatment_specs(candidates)
        expected = len(specs) * len(protocol["discovery_case_ids"])
        for case_id in protocol["discovery_case_ids"]:
            for feature, group, alpha, role in specs:
                path = treatment_path(directory, case_id, feature, group, alpha)
                if path.exists():
                    rows.append({**load_treatment(path, protocol, case_id, feature, group, alpha), "role": role})
        for feature, group, alpha, role in specs:
            selected = [r for r in rows if (r["feature"], r["group"], r["alpha"]) == (feature, group, alpha)]
            summaries.append({"feature": feature, "group": group, "alpha": alpha, "role": role,
                              "overall": bootstrap_summary(selected),
                              "correct_answers": bootstrap_summary([r for r in selected if r["original_correct"]]),
                              "outcomes": {k: bootstrap_summary([r for r in selected if r["outcome"] == k]) for k in QUOTAS},
                              "corrected_images": sum(r["corrected"] for r in selected),
                              "introduced_images": sum(r["introduced"] for r in selected),
                              "agreement": agreement(selected, pilot["tau"])})
        for pair in candidates["pairs"]:
            for alpha in FRACTIONS:
                targets = {r["case_id"]: r for r in rows if r["group"] == pair["group"] and
                           r["feature"] == pair["target"]["feature"] and r["alpha"] == alpha}
                comparisons = {r["case_id"]: r for r in rows if r["group"] == pair["group"] and
                               r["feature"] == pair["comparison"]["feature"] and r["alpha"] == alpha}
                differences = []
                for case_id in targets.keys() & comparisons.keys():
                    target, control = targets[case_id], comparisons[case_id]
                    differences.append({**target, "actual_delta": target["actual_delta"] - control["actual_delta"],
                                        "predicted_delta": target["predicted_delta"] - control["predicted_delta"]})
                # Paired signed differences only; do not reinterpret control effects as nulls.
                contrasts.append({"group": pair["group"], "target": pair["target"]["feature"],
                                  "comparison": pair["comparison"]["feature"], "alpha": alpha,
                                  "signed_difference": bootstrap_summary(differences).get("actual_delta")})
    write_csv(directory / "intervention_effects.csv", rows)
    completed_cases = [c for c in protocol["discovery_case_ids"] if (directory / "cases" / f"{c}.json").exists()]
    limits = torch.zeros(protocol["sae_width"])
    activity_limits = torch.zeros_like(limits)
    compact = {g: [] for g in GROUPS}
    for case_id in completed_cases:
        _, payload = load_case(directory, protocol, case_id)
        sparse = payload["attribution"]
        limits.scatter_reduce_(0, sparse.indices()[1], sparse.values().abs(), reduce="amax")
        if protocol.get("fresh_encodings"):
            features = payload["features"]
            activity_limits.scatter_reduce_(0, features.indices()[1], features.values(), reduce="amax")
        for group in GROUPS:
            compact[group].append(payload["groups"][group]["effect"])
    tensor_path = directory / "intervention_summaries.pt"
    temporary = tensor_path.with_suffix(".pt.tmp")
    torch.save({"case_ids": completed_cases, "signed_color_limits": limits, "activation_color_limits": activity_limits,
                "group_effects": {g: torch.stack(v) if v else torch.empty(0, protocol["sae_width"])
                                  for g, v in compact.items()}}, temporary)
    temporary.replace(tensor_path)
    report = {"protocol_hash": protocol["protocol_hash"], "pilot_complete": pilot["complete"],
              "qualifying_groups": pilot["qualifying_groups"], "attributed_cases": len(completed_cases),
              "expected_cases": len(protocol["discovery_case_ids"]), "measured_treatments": len(rows),
              "expected_treatments": expected, "complete": pilot["complete"] and (
                  not pilot["qualifying_groups"] or candidates is not None and len(rows) == expected),
              "status": ("calibration_stopped" if protocol.get("fresh_encodings") else "pilot_stopped") if pilot["complete"] and not pilot["qualifying_groups"] else
                  "cohort_complete" if candidates and len(rows) == expected else "partial",
              "summaries": summaries, "target_minus_comparison": contrasts,
              "limitations": ["Discovery-selected estimates; bootstrap intervals are descriptive, not independent confirmation",
                  "Matched comparisons are not null features", "Position attribution is contextual; no pixel-level causality or localization",
                  "One transferred SAE at layer 14; no circuit or unseen-patient mitigation claim"],
              "failure_diagnostics": sorted(p.name for p in directory.glob("failure_*.json"))}
    report["summary_tensor_sha256"] = file_sha256(tensor_path)
    report["effects_csv_sha256"] = file_sha256(directory / "intervention_effects.csv")
    atomic_json(directory / "intervention_report.json", report)
    return report
