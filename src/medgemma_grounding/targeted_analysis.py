"""E-008 offline diagnostics, transparent nomination and patient-level summaries.

No model loading. Historical E-007 files are read-only. Means/contrasts nominate
features; they are not semantic labels, tests of significance, or causal scores.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from medgemma_grounding import medgemma_interventions as base
from medgemma_grounding.feature_analysis import feature_rankings
from medgemma_grounding.medgemma_features import atomic_json, identity_hash, file_sha256

GROUPS = ("image", "final")
NOMINATION = {"minimum_active_patients": 5, "minimum_active_error_patients": 2,
              "per_track_per_group": 1, "reference_final_features": [414, 359],
              "maximum_candidates": 8}


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(spearmanr(x, y).statistic) if len(x) >= 3 and np.ptp(x) and np.ptp(y) else None


def agreement(rows, tau):
    """ALL supplied pairs, including zero/small effects. Denominators explicit."""
    p = np.array([r["predicted_delta"] for r in rows], dtype=float)
    a = np.array([r["actual_delta"] for r in rows], dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("Nonfinite attribution comparison")
    large = abs(p) >= tau
    both = (p != 0) & (a != 0)
    return {"pairs": len(rows), "patients": len({r["patient"] for r in rows}),
            "spearman": correlation(p, a),
            "mae": float(abs(p - a).mean()) if len(a) else None,
            "rmse": float(np.sqrt(np.mean((p - a) ** 2))) if len(a) else None,
            "median_normalized_error": float(np.median(abs(p - a) / np.maximum(abs(a), tau))) if len(a) else None,
            "sign_agreement_including_zero": float(np.mean(np.sign(p) == np.sign(a))) if len(a) else None,
            "both_nonzero": int(both.sum()),
            "sign_agreement_both_nonzero": float(np.mean(np.sign(p[both]) == np.sign(a[both]))) if both.any() else None,
            "opposite_sign": int((p * a < 0).sum()),
            "opposite_sign_both_substantial": int(((p * a < 0) & large & (abs(a) >= tau)).sum()),
            "actual_zero": int((a == 0).sum()), "predicted_zero": int((p == 0).sum()),
            "predicted_substantial": int(large.sum()),
            "predicted_large_actual_small": int((large & (abs(a) < tau)).sum()),
            "false_alarm_fraction_of_large_predictions": float(np.mean(abs(a[large]) < tau)) if large.any() else None}


def audit_pairs(rows, tau, collection):
    summaries = []
    for group in base.GROUPS:
        for alpha in base.FRACTIONS:
            selected = [r for r in rows if r["group"] == group and r["alpha"] == alpha]
            for name, subset in (("all", selected), ("substantial_actual", [r for r in selected if abs(r["actual_delta"]) >= tau])):
                summaries.append({"collection": collection, "group": group, "alpha": alpha, "subset": name,
                                  "feature": None, **agreement(subset, tau)})
            for feature in sorted({r["feature"] for r in selected}):
                subset = [r for r in selected if r["feature"] == feature]
                for name, part in (("all", subset), ("substantial_actual", [r for r in subset if abs(r["actual_delta"]) >= tau])):
                    summaries.append({"collection": collection, "group": group, "alpha": alpha, "subset": name,
                                      "feature": feature, **agreement(part, tau)})
    return summaries


def audit_e007(source, parent, records, output):
    """Independent extension; never call E-007 report writers."""
    calibration = base.read_json(source / "calibration_report.json")
    historical = base.read_json(source / "intervention_report.json")
    union = {}
    for path in sorted((source / "treatments").glob("*/*.json")):
        r = base.read_json(path)
        base.load_treatment(path, parent, r["case_id"], r["feature"], r["group"], r["alpha"])
        union[(r["case_id"], r["feature"], r["group"], r["alpha"])] = r
    selected = []
    candidates = base.read_json(source / "candidates.json")
    for case in parent["discovery_case_ids"]:
        for j, g, a, _ in base.treatment_specs(candidates):
            selected.append(union[case, j, g, a])
    rows = list(union.values())
    summaries = []
    for name, measurements in (("calibration", calibration["measurements"]), ("frozen_cohort", selected), ("unique_union", rows)):
        summaries.extend(audit_pairs(measurements, calibration["tau"], name))
    base.write_csv(output / "e007_agreement.csv", summaries)
    alarms = sorted([r for r in rows if abs(r["predicted_delta"]) >= calibration["tau"] and abs(r["actual_delta"]) < calibration["tau"]],
                    key=lambda r: (-abs(r["predicted_delta"]), r["case_id"], r["group"], r["feature"], r["alpha"]))
    opposite = sorted([r for r in rows if r["predicted_delta"] * r["actual_delta"] < 0],
                      key=lambda r: (-abs(r["predicted_delta"] - r["actual_delta"]), r["case_id"], r["feature"]))
    introduced = [r for r in selected if r["introduced"]]
    references = []
    for j in NOMINATION["reference_final_features"]:
        measurements = [r for r in selected if r["feature"] == j and r["group"] == "final" and r["alpha"] == 1]
        for outcome in base.QUOTAS:
            part = [r for r in measurements if r["outcome"] == outcome and r["original_score"] != 0]
            references.append({"feature": j, "outcome": outcome, "events": len(part),
                               "strengthens_existing_answer": int(sum(r["actual_delta"] * np.sign(r["original_score"]) > 0 for r in part)),
                               "mean_delta": float(np.mean([r["actual_delta"] for r in part])) if part else None})
    ratios = [r["realized_l2"] / r["requested_l2"] for r in rows if r["requested_l2"] > 0]
    result = {"source_protocol_hash": parent["protocol_hash"], "cases": len(records),
              "patients": len({r["patient"] for r in records}), "unique_treatment_events": len(rows),
              "historical_qualifying_groups": historical["qualifying_groups"],
              "historical_calibration_unchanged": calibration["groups"],
              "agreement": [r for r in summaries if r["feature"] is None],
              "false_alarm_examples": alarms[:20], "opposite_sign_examples": opposite[:20],
              "reference_strengthening": references,
              "realized_over_requested_l2": {"min": min(ratios), "median": float(np.median(ratios)), "max": max(ratios)},
              "frozen_corrected_events": sum(r["corrected"] for r in selected),
              "frozen_introduced_events": len(introduced),
              "frozen_introduced_patients": sorted({r["patient"] for r in introduced}),
              "natural_comparison_matching": [{"group": p["group"], "target": p["target"]["feature"],
                   "comparison": p["comparison"]["feature"], "distance": p["matching_distance"],
                   "target_removal_rms": p["target"]["removal_rms"], "comparison_removal_rms": p["comparison"]["removal_rms"]} for p in candidates["pairs"]],
              "limits": ["Even all tested pairs are selected, not an unbiased dictionary-wide ranking evaluation",
                         "Substantial-actual conditioning cannot establish screening accuracy on small/zero effects",
                         "Natural comparison features are not size-matched random directions"]}
    atomic_json(output / "e007_audit.json", result)
    return result


def aggregate_patients(records, arrays):
    patients = sorted({r["patient"] for r in records})
    result = {g: {k: [] for k in ("activity", "attribution")} for g in GROUPS}
    metadata = []
    for patient in patients:
        ix = [i for i, r in enumerate(records) if r["patient"] == patient]
        outcomes = {records[i]["outcome"] for i in ix}
        if len(outcomes) != 1 or len({records[i]["label"] for i in ix}) != 1:
            raise ValueError("Mixed-label/outcome patient requires a new explicit aggregation policy")
        metadata.append({"patient": patient, "label": records[ix[0]]["label"], "outcome": next(iter(outcomes)),
                         "answer_score": float(np.mean([records[i]["answer_score"]["score"] for i in ix])),
                         "images": len(ix)})
        for g in GROUPS:
            for k in result[g]:
                result[g][k].append(arrays[g][k][ix].mean(0))
    return metadata, {g: {k: np.stack(v) for k, v in a.items()} for g, a in result.items()}


def nominate(records, arrays, settings=None):
    """Three SEPARATE class-mean contrasts per group; references do not consume them.

    Disease: absolute positive-minus-negative difference. Errors: positive
    FP-minus-TP / FN-minus-TN activity enrichment within the same original answer.
    Signed attribution is supporting pattern evidence, never a combined score.
    """
    settings = NOMINATION if settings is None else settings
    patients, pooled = aggregate_patients(records, arrays)
    labels = np.array([p["label"] for p in patients])
    scores = np.array([p["answer_score"] for p in patients])
    outcomes = np.array([p["outcome"] for p in patients])
    if set(outcomes) != set(base.QUOTAS):
        raise ValueError("All TP/TN/FP/FN outcomes required for the fixed nomination tracks")
    associations = feature_rankings(patients, {g: pooled[g]["activity"] for g in GROUPS},
                                    min_active=0, representations=GROUPS)
    assoc = {(r["representation"], r["feature"]): r for r in associations if r["target"] == "label"}
    rows, chosen = [], {}
    for g in GROUPS:
        x, attribution = pooled[g]["activity"], pooled[g]["attribution"]
        means = {c: x[outcomes == c].mean(0) for c in base.QUOTAS}
        effects = {c: attribution[outcomes == c].mean(0) for c in base.QUOTAS}
        active = (x > 0).sum(0)
        contrasts = {"effusion": x[labels == 1].mean(0) - x[labels == 0].mean(0),
                     "FP_vs_TP": means["FP"] - means["TP"], "FN_vs_TN": means["FN"] - means["TN"]}
        group_rows = []
        for j in range(x.shape[1]):
            constant = np.ptp(x[:, j]) == 0
            row = {"group": g, "feature": j, "active_patients": int(active[j]), "patients": len(patients),
                   "constant": bool(constant), "label_auc": assoc[g, j]["label_auc"],
                   "score_spearman": assoc[g, j]["answer_spearman"],
                   "absolute_score_spearman": correlation(x[:, j], abs(scores)),
                   "positive_mean_activity": float(x[labels == 1, j].mean()),
                   "negative_mean_activity": float(x[labels == 0, j].mean())}
            for c in base.QUOTAS:
                row.update({f"{c}_patients": int((outcomes == c).sum()),
                            f"{c}_active_patients": int((x[outcomes == c, j] > 0).sum()),
                            f"{c}_mean_activity": float(means[c][j]), f"{c}_mean_attribution": float(effects[c][j])})
            for track, values in contrasts.items():
                reasons = []
                if constant:
                    reasons.append("constant")
                if active[j] < settings["minimum_active_patients"]:
                    reasons.append("fewer_than_five_active_patients")
                if track != "effusion":
                    error = "FP" if track == "FP_vs_TP" else "FN"
                    if row[f"{error}_active_patients"] < settings["minimum_active_error_patients"]:
                        reasons.append("fewer_than_two_active_error_patients")
                    if values[j] <= 0:
                        reasons.append("not_error_enriched")
                row[f"{track}_contrast"] = float(values[j])
                row[f"{track}_exclusion"] = ";".join(reasons)
            group_rows.append(row)
        for track in contrasts:
            ordered = sorted(group_rows, key=lambda r: (-(abs(r[f"{track}_contrast"]) if track == "effusion" else r[f"{track}_contrast"]), r["feature"]))
            eligible_rank = 0
            for raw_rank, row in enumerate(ordered, 1):
                row[f"{track}_raw_rank"] = raw_rank
                eligible = not row[f"{track}_exclusion"]
                eligible_rank += int(eligible)
                row[f"{track}_eligible_rank"] = eligible_rank if eligible else None
                if eligible and eligible_rank <= settings["per_track_per_group"]:
                    chosen.setdefault((g, row["feature"]), {"group": g, "feature": row["feature"], "tracks": []})["tracks"].append(track)
        rows.extend(group_rows)
    lookup = {(r["group"], r["feature"]): r for r in rows}
    for j in settings["reference_final_features"]:
        chosen.setdefault(("final", j), {"group": "final", "feature": j, "tracks": []})["tracks"].append("E007_reference")
    candidates = [{**entry, "statistics": lookup[key], "semantic_hypothesis": None} for key, entry in sorted(chosen.items())]
    if len(candidates) > settings["maximum_candidates"]:
        raise ValueError("Prespecified candidate budget exceeded")
    return candidates, rows, patients


def uncertainty(rows, key, resamples=1000, seed=42):
    """Equal patient weighting; bootstrap patients, never treatment events."""
    grouped = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            grouped[r["patient"]].append(r[key])
    if not grouped:
        return {"patients": 0, "images": 0, "mean": None, "ci95": None}
    values = np.array([np.mean(grouped[p]) for p in sorted(grouped)])
    ix = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    return {"patients": len(values), "images": sum(map(len, grouped.values())), "mean": float(values.mean()),
            "ci95": np.quantile(values[ix].mean(1), [.025, .975]).tolist()}


def audit_figure(audit, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(10, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    colors = {"all": "#23658a", "substantial_actual": "#b76e18"}
    for col, alpha in enumerate(base.FRACTIONS, 1):
        axis = figure.add_subplot(1, 2, col)
        for name, shift in (("all", -.16), ("substantial_actual", .16)):
            values = [next(r for r in audit["agreement"] if r["collection"] == "calibration" and
                          r["group"] == g and r["alpha"] == alpha and r["subset"] == name) for g in base.GROUPS]
            axis.bar(np.arange(3) + shift, [v["spearman"] if v["spearman"] is not None else np.nan for v in values], .3,
                     label=name.replace("_", " "), color=colors[name])
            for i, v in enumerate(values):
                if v["spearman"] is not None:
                    axis.text(i + shift, v["spearman"] + .025, f"n={v['pairs']}", ha="center", fontsize=8)
        axis.set(xticks=range(3), xticklabels=["Image", "Final input", "Other non-image"], ylim=(-.1, 1.15),
                 ylabel="Spearman (predicted vs measured)", title=f"E-007 calibration · removal {alpha:g}")
        axis.axhline(0, color="#444444", linewidth=.7)
        axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", fontsize=8)
    figure.savefig(path, dpi=150)


def candidate_summary(rows):
    metrics = ("actual_delta", "absolute_effect", "predicted_delta", "kl_from_baseline",
               "realized_l2", "toward_expert_delta", "strengthening_delta", "corrected", "introduced")
    result = {k: uncertainty(rows, k) for k in metrics}
    result.update(images=len(rows), patients=len({r["patient"] for r in rows}),
                  corrected_images=sum(bool(r["corrected"]) for r in rows),
                  introduced_images=sum(bool(r["introduced"]) for r in rows),
                  corrected_patients=sorted({r["patient"] for r in rows if r["corrected"]}),
                  introduced_patients=sorted({r["patient"] for r in rows if r["introduced"]}))
    return result


def paired_cases(targets, controls, bank_size):
    rows = []
    for t in targets:
        key = (t["case_id"], t["group"], t["feature"], t["alpha"])
        bank = controls.get(key, [])
        matched = (t["target_state"] == "positive_size" and len(bank) == bank_size
                   and {c["bank"] for c in bank} == set(range(bank_size)) and all(c["matched"] for c in bank))
        x = {k: t[k] for k in ("case_id", "patient", "label", "outcome", "group", "feature", "alpha",
                              "original_score", "original_decision", "actual_delta", "predicted_delta", "kl_from_baseline",
                              "requested_l2", "realized_l2", "target_state", "corrected", "introduced", "execution")}
        x.update(absolute_effect=abs(t["actual_delta"]),
                 toward_expert_delta=(1 if t["label"] else -1) * t["actual_delta"],
                 strengthening_delta=(1 if t["original_decision"] == "Yes" else -1) * t["actual_delta"],
                 primary_matched=matched, matching_exclusion=None if matched else t["target_state"] if t["target_state"] != "positive_size" else "incomplete_or_unmatched_bank")
        mean_signed = float(np.mean([c["actual_delta"] for c in bank])) if matched else None
        mean_abs = float(np.mean([abs(c["actual_delta"]) for c in bank])) if matched else None
        x.update(control_mean_delta=mean_signed, control_mean_absolute=mean_abs,
                 matched_signed_excess=t["actual_delta"] - mean_signed if matched else None,
                 matched_abs_excess=abs(t["actual_delta"]) - mean_abs if matched else None,
                 fraction_bank_abs_smaller_or_equal=float(np.mean([abs(t["actual_delta"]) >= abs(c["actual_delta"]) for c in bank])) if matched else None)
        rows.append(x)
    return rows


def report(output, protocol, parent):
    """Saved-only results; no model/SAE loading and no historical artifact writes."""
    from medgemma_grounding.medgemma_targeted import measurement_path, load_measurement, specs, case_complete
    candidates = base.read_json(output / "candidates.json")["candidates"]
    targets, random, all_rows = [], defaultdict(list), []
    for case in protocol["discovery_case_ids"]:
        for g, j, a, k in specs(candidates, protocol["control_bank_size"]):
            path = measurement_path(output, case, g, j, a, k)
            if not path.exists():
                continue
            r = load_measurement(path, protocol, case, g, j, a, k)
            all_rows.append(r)
            if k is None:
                targets.append(r)
            else:
                random[case, g, j, a].append(r)
    rows = paired_cases(targets, random, protocol["control_bank_size"])
    base.write_csv(output / "case_effects.csv", rows)
    fields = ("case_id", "patient", "label", "outcome", "group", "feature", "alpha", "bank", "role", "execution",
              "target_state", "status", "requested_l2", "realized_l2", "max_position_relative_error", "total_relative_error",
              "original_score", "actual_delta", "predicted_delta", "kl_from_baseline", "corrected", "introduced", "matched")
    base.write_csv(output / "measurements.csv", [{k: r.get(k) for k in fields} for r in all_rows])
    summaries = []
    for c in candidates:
        for a in base.FRACTIONS:
            subset = [r for r in rows if r["group"] == c["group"] and r["feature"] == c["feature"] and r["alpha"] == a]
            by_patient = defaultdict(list)
            for r in subset:
                by_patient[r["patient"]].append(r)
            s = {"group": c["group"], "feature": c["feature"], "alpha": a, "tracks": c["tracks"],
                 "all_cases": candidate_summary(subset),
                 "correct_answers": candidate_summary([r for r in subset if r["outcome"].startswith("T")]),
                 "outcomes": {o: candidate_summary([r for r in subset if r["outcome"] == o]) for o in base.QUOTAS},
                 "matched": {key: uncertainty(subset, key) for key in ("control_mean_delta", "control_mean_absolute", "matched_signed_excess", "matched_abs_excess", "fraction_bank_abs_smaller_or_equal")},
                 "matching_exclusions": {reason: sum(r["matching_exclusion"] == reason for r in subset)
                     for reason in ("inactive", "rounded_away", "incomplete_or_unmatched_bank")},
                 "score_dependence": correlation([np.mean([r["original_score"] for r in v]) for v in by_patient.values()],
                                                 [np.mean([r["actual_delta"] for r in v]) for v in by_patient.values()]),
                 "absolute_score_dependence": correlation([np.mean([abs(r["original_score"]) for r in v]) for v in by_patient.values()],
                                                          [np.mean([abs(r["actual_delta"]) for r in v]) for v in by_patient.values()]),
                 "agreement_all": agreement(subset, .25),
                 "agreement_substantial_actual": agreement([r for r in subset if abs(r["actual_delta"]) >= .25], .25)}
            s["evidence"] = (f"Nominated by {', '.join(c['tracks'])}. Measured {len(subset)} discovery images; "
                f"corrected {s['all_cases']['corrected_images']} and introduced {s['all_cases']['introduced_images']} image-level errors. "
                "Association is not disease meaning; direct influence is not localization or independent confirmation.")
            summaries.append(s)
    controls = [r for r in all_rows if r["bank"] is not None]
    positive = [r for r in controls if r["target_state"] == "positive_size"]
    matched = [r for r in positive if r["matched"]]
    complete_cases = sum(case_complete(output, protocol, case, candidates) for case in protocol["discovery_case_ids"])
    failures = [base.read_json(p) for p in sorted(output.glob("failure_*.json"))]
    complete = (complete_cases == len(protocol["discovery_case_ids"]) and not failures
                and (output / "smoke_report.json").exists() and (output / "runtime_checks.json").exists())
    result = {"protocol_hash": protocol["protocol_hash"], "experiment": "E-008", "complete": complete,
        "status": "technical_failure" if failures else "complete" if complete else "partial",
        "stages": {"audit": "complete", "nomination": "complete", "protocol": "frozen",
                   "smoke": "complete" if (output / "smoke_report.json").exists() else "not_complete",
                   "cohort": "complete" if complete else "partial", "report": "complete"},
        "expected_cases": len(protocol["discovery_case_ids"]), "completed_cases": complete_cases,
        "measurement_records": len(all_rows), "expected_measurement_records": len(specs(candidates, protocol["control_bank_size"])) * len(protocol["discovery_case_ids"]),
        "new_treatment_forwards": sum(r["execution"] == "new_forward" for r in all_rows),
        "reused_target_measurements": sum(r["execution"] == "reused_exact_e007" for r in targets),
        "summaries": summaries,
        "control_matching": {"positive_size_controls": len(positive), "matched_positive_controls": len(matched),
            "unmatched_controls": len(positive) - len(matched), "zero_target_controls": len(controls) - len(positive),
            "maximum_matched_position_relative_error": max((r["max_position_relative_error"] for r in matched), default=None),
            "maximum_matched_total_relative_error": max((r["total_relative_error"] for r in matched), default=None),
            "primary_matched_case_feature_strengths": sum(r["primary_matched"] for r in rows)},
        "distinct_corrected_patients": sorted({r["patient"] for r in rows if r["corrected"]}),
        "distinct_introduced_patients": sorted({r["patient"] for r in rows if r["introduced"]}),
        "natural_comparison_features": "Historical E-007 natural-feature comparisons remain separate; see e007_audit.json",
        "failures": failures,
        "limitations": ["Discovery-selected and exploratory; no held-out inference or generalization evidence",
            "Only four FN patients; same-answer contrasts do not remove score/acquisition/other confounding",
            "Three fixed directions control realized magnitude only; not semantic, on-manifold or population-null controls",
            "Primary control contrasts exclude zero/rounded-away/unmatched banks; unconditional target effects retain every measured case",
            "Decoder position overlays are contextual, not anatomy labels or causal image-region localization",
            "No IG, extra SAE layer, training, image-region manipulation or correction policy"]}
    checks = output / "runtime_checks.json"
    if checks.exists():
        result["runtime_checks"] = base.read_json(checks)
    result["artifacts_sha256"] = {name: file_sha256(output / name) for name in ("case_effects.csv", "measurements.csv")}
    atomic_json(output / "report.json", result)
    result_figures(result, rows, output)
    result["artifacts_sha256"].update({name: file_sha256(output / name) for name in
                                      ("e007_agreement.png", "controlled_effects.png", "score_effects.png")})
    atomic_json(output / "report.json", result)
    lines = ["# E-008: representation, influence and errors remain separate questions", "",
        f"Status: **{result['status']}**. Complete cases: {complete_cases}/{result['expected_cases']}. "
        f"Saved measurements: {len(all_rows)}/{result['expected_measurement_records']}.", "",
        "This is the fixed discovery cohort, not confirmation or test evaluation.", "",
        "## Direct full-removal results", "",
        "| Position / feature | Tracks | Mean absolute ΔS (all cases) | Paired absolute excess over size controls | Corrected / introduced images |",
        "|---|---|---:|---:|---:|"]
    for s in summaries:
        if s["alpha"] != 1:
            continue
        absolute = s["all_cases"]["absolute_effect"]["mean"]
        excess = s["matched"]["matched_abs_excess"]["mean"]
        value = lambda x: "unavailable" if x is None else f"{x:.4f}"
        lines.append(f"| {s['group']} / {s['feature']} | {', '.join(s['tracks'])} | {value(absolute)} | {value(excess)} | {s['all_cases']['corrected_images']} / {s['all_cases']['introduced_images']} |")
    lines += ["", "Paired excess uses only complete positive-size matched banks; populations can differ from the all-case target column. Intervals and subgroup counts are in report.json and the viewer.", "",
        "![All-pair versus substantial-actual calibration](e007_agreement.png)", "",
        "![Paired effects with patient-bootstrap intervals](controlled_effects.png)", "",
        "![Original score versus actual removal effect](score_effects.png)", "", "## Limits", ""]
    lines += [f"- {v}" for v in result["limitations"]]
    lines += ["", "[Frozen method adaptations](methods.md) · [Protocol](protocol.json) · [Detailed results](report.json)"]
    if (output / "interpretation.md").exists():
        lines += ["", "[Research interpretation and one proposed next experiment](interpretation.md)"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def result_figures(report, rows, output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(10, 6), layout="constrained")
    FigureCanvasAgg(figure)
    palette = {"image": "#23658a", "final": "#b76e18"}
    for col, alpha in enumerate(base.FRACTIONS, 1):
        axis = figure.add_subplot(1, 2, col)
        selected = [s for s in report["summaries"] if s["alpha"] == alpha]
        for i, s in enumerate(selected):
            v = s["matched"]["matched_abs_excess"]
            if v["mean"] is not None:
                lo, hi = v["ci95"]
                axis.errorbar(v["mean"], i, xerr=[[max(0, v["mean"] - lo)], [max(0, hi - v["mean"])]],
                              fmt="o", color=palette[s["group"]], capsize=3)
        axis.set(yticks=range(len(selected)), yticklabels=[f"{s['group']} {s['feature']}" for s in selected],
                 xlabel="Mean |target ΔS| − mean |bank ΔS| (logits)", title=f"Removal {alpha:g} · paired matched cases")
        axis.axvline(0, color="#444444", linewidth=.8)
        axis.spines[["top", "right"]].set_visible(False)
    # Comparable strengths use the same domain, preserving zero and interval extent.
    axes = figure.axes
    lo, hi = min(a.get_xlim()[0] for a in axes), max(a.get_xlim()[1] for a in axes)
    for axis in axes:
        axis.set_xlim(lo, hi)
    figure.savefig(output / "controlled_effects.png", dpi=150)
    candidates = [s for s in report["summaries"] if s["alpha"] == 1]
    figure = Figure(figsize=(10, max(3, len(candidates) * 2.1)), layout="constrained")
    FigureCanvasAgg(figure)
    colors = {"TP": "#23658a", "TN": "#888888", "FP": "#b76e18", "FN": "#b14e85"}
    markers = {"TP": "o", "TN": "o", "FP": "x", "FN": "x"}
    limits = {g: max([abs(r["actual_delta"]) for r in rows if r["group"] == g and r["alpha"] == 1] or [.25]) * 1.1 for g in GROUPS}
    for i, s in enumerate(candidates, 1):
        axis = figure.add_subplot((len(candidates) + 1) // 2, 2, i)
        part = [r for r in rows if (r["group"], r["feature"], r["alpha"]) == (s["group"], s["feature"], 1)]
        for o in base.QUOTAS:
            selected = [r for r in part if r["outcome"] == o]
            axis.scatter([r["original_score"] for r in selected], [r["actual_delta"] for r in selected],
                         s=14, alpha=.65, color=colors[o], marker=markers[o], label=o)
        axis.axhline(0, color="#444444", linewidth=.7)
        axis.set(title=f"{s['group']} {s['feature']} · full removal", xlabel="Original Yes-minus-No score",
                 ylabel="Measured ΔS", ylim=(-limits[s["group"]], limits[s["group"]]))
        axis.spines[["top", "right"]].set_visible(False)
        if i == 1:
            axis.legend(fontsize=8)
    figure.suptitle("Discovery images · y scales shared within position group, not across groups", fontsize=11)
    figure.savefig(output / "score_effects.png", dpi=140)


def save_visual_hypothesis(directory, protocol, group, feature, text, counterexamples):
    candidates = base.read_json(directory / "candidates.json")["candidates"]
    if (not any(c["group"] == group and c["feature"] == feature for c in candidates)
            or not set(counterexamples) <= set(protocol["discovery_case_ids"])):
        raise ValueError("Hypotheses/counterexamples must belong to frozen discovery candidates/cases")
    path = directory / "visual_hypotheses.json"
    saved = base.read_json(path) if path.exists() else {"protocol_hash": protocol["protocol_hash"], "features": {}}
    if saved["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Hypothesis notes belong to another protocol")
    saved["features"][f"{group}:{feature}"] = {"hypothesis": text.strip(), "counterexample_cases": counterexamples,
        "source": "human-proposed", "status": "unconfirmed; not an anatomical label or ground truth"}
    atomic_json(path, saved)
