"""Patient-level error contrasts and score-only controls; saved results only.

No model, SAE or held-out image loading. Original outcome strata never change."""


from collections import defaultdict

from scipy.stats import rankdata
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
import medgemma_grounding.utils.io as io
import numpy as np

from medgemma_grounding.utils.io import atomic_json


OUTCOMES = ("TP", "TN", "FP", "FN")


CONTRASTS = {"D_neg": ("FP", "TN"), "D_yes": ("FP", "TP"), "D_pos": ("FN", "TP")}


RESAMPLES = 2000


def patient_rows(rows):
    """One row per patient AND condition, retaining image-level counts elsewhere."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["id"], row["patient"])].append(row)
    result = []
    for (condition, patient), part in sorted(grouped.items()):
        for key in ("label", "outcome"):
            if len({r[key] for r in part}) != 1:
                raise ValueError(f"Ambiguous within-patient {key}: {patient}")
        result.append({"id": condition, "patient": patient, "label": part[0]["label"],
                       "outcome": part[0]["outcome"], "images": len(part),
                       **{k: float(np.mean([r[k] for r in part])) for k in
                          ("S", "changed_S", "delta_S", "delta_M", "kl_from_baseline")}})
    return result


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "ci95": None}
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite analysis input")
    draws = np.random.default_rng(42).integers(len(a), size=(RESAMPLES, len(a)))
    return {"n": len(a), "mean": float(a.mean()),
            "ci95": np.quantile(a[draws].mean(1), [.025, .975]).tolist(),
            "quantiles": np.quantile(a, [0, .25, .5, .75, 1]).tolist()}


def contrast(first, second):
    """Independent resampling within the two fixed patient groups."""
    x, y = np.asarray(first, float), np.asarray(second, float)
    if min(len(x), len(y)) < 2:
        raise ValueError("At least two patients per contrast group required")
    rng = np.random.default_rng(42)
    boot = x[rng.integers(len(x), size=(RESAMPLES, len(x)))].mean(1)
    boot -= y[rng.integers(len(y), size=(RESAMPLES, len(y)))].mean(1)
    effect = float(x.mean() - y.mean())
    if np.var(x, ddof=1) + np.var(y, ddof=1) == 0:
        statistic, p = None, 1. if effect == 0 else 0.
        status = "both_zero_variance"
    else:
        test = ttest_ind(x, y, equal_var=False)
        statistic, p, status = float(test.statistic), float(test.pvalue), "welch"
    return {"first": distribution(x), "second": distribution(y), "effect": effect,
            "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "welch_t": statistic, "p": p, "test_status": status}


def bh_adjust(pvalues):
    p = np.asarray(pvalues, float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(q, 1)
    return result.tolist()


def fixed_folds(patients):
    labels = {}
    for r in patients:
        if r["patient"] in labels and labels[r["patient"]] != r["label"]:
            raise ValueError("Patient labels disagree")
        labels[r["patient"]] = r["label"]
    rng, folds = np.random.default_rng(42), {}
    for label in (0, 1):
        ids = sorted(p for p, y in labels.items() if y == label)
        for i, p in enumerate(rng.permutation(ids)):
            folds[str(p)] = i % 5
    return folds


def fit_score_control(x, y, kind):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid control training data")
    if kind == "shift":
        a, b, status = float(np.mean(y - x)), 1., "shift"
    elif kind == "affine":
        variance = float(np.sum((x - x.mean())**2))
        if variance == 0:
            b, status = 1., "constant_x_shift_fallback"
        else:
            slope = float(np.sum((x - x.mean()) * (y - y.mean())) / variance)
            b, status = max(1e-8, slope), "positive_slope_clamped" if slope <= 1e-8 else "affine"
        a = float(y.mean() - b * x.mean())
    else:
        raise ValueError("Unknown score control")
    return {"intercept": a, "slope": b, "status": status, "training_patients": len(x)}


def score_controls(part, folds):
    """Train on patient means; predict held-out patients without outcome parameters."""
    predictions, fits, summaries = [], [], []
    for kind in ("shift", "affine"):
        pred = []
        for fold in range(5):
            train = [r for r in part if folds[r["patient"]] != fold]
            test = [r for r in part if folds[r["patient"]] == fold]
            fitted = fit_score_control([r["S"] for r in train], [r["changed_S"] for r in train], kind)
            fits.append({"id": part[0]["id"], "control": kind, "fold": fold, **fitted})
            for row in test:
                estimate = fitted["intercept"] + fitted["slope"] * row["S"]
                pred.append({**row, "control": kind, "fold": fold, "predicted_S": estimate,
                             "residual": row["changed_S"] - estimate})
        errors = np.array([r["residual"] for r in pred])
        actual = np.array([r["changed_S"] for r in pred])
        denom = float(np.sum((actual - actual.mean())**2))
        summaries.append({"id": part[0]["id"], "control": kind,
                          "mae": float(np.mean(abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2))),
                          "oof_r2": 1 - float(np.sum(errors**2)) / denom if denom else None,
                          "answer_agreement": float(np.mean([(r["predicted_S"] > 0) == (r["changed_S"] > 0) for r in pred])),
                          "residual_by_outcome": {o: distribution([r["residual"] for r in pred if r["outcome"] == o]) for o in OUTCOMES},
                          "residual_D_neg": contrast([r["residual"] for r in pred if r["outcome"] == "FP"],
                                                     [r["residual"] for r in pred if r["outcome"] == "TN"])})
        predictions.extend(pred)
    return predictions, fits, summaries


def classification(rows, score="changed_S", threshold=0.):
    counts = {o: 0 for o in OUTCOMES}
    for r in rows:
        answer = r[score] > threshold
        correct = answer == bool(r["label"])
        counts[("T" if correct else "F") + ("P" if answer else "N")] += 1
    sensitivity = counts["TP"] / (counts["TP"] + counts["FN"])
    specificity = counts["TN"] / (counts["TN"] + counts["FP"])
    return {**counts, "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "accuracy": (counts["TP"] + counts["TN"]) / len(rows)}


def ranking(rows, score):
    x, y = np.array([r[score] for r in rows]), np.array([r["label"] for r in rows])
    npos, nneg = int(y.sum()), int((y == 0).sum())
    auc = (rankdata(x)[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return float(auc)


def correlation(x, y):
    return float(spearmanr(x, y).statistic) if len(x) > 2 and np.ptp(x) and np.ptp(y) else None


def roc_thresholds(rows):
    scores = sorted({float(r["S"]) for r in rows})
    return [{"threshold": t, **classification(rows, "S", t)} for t in [scores[0] - 1, *scores]]


def patient_metric_intervals(rows):
    """Patient-stratified bootstrap; image-level clinical counts are separate."""
    rng = np.random.default_rng(42)
    arrays = {y: [r for r in rows if r["label"] == y] for y in (0, 1)}
    values = defaultdict(list)
    for _ in range(RESAMPLES):
        sample = [part[i] for part in arrays.values() for i in rng.integers(len(part), size=len(part))]
        metrics = classification(sample)
        for key in ("sensitivity", "specificity", "balanced_accuracy"):
            values[key].append(metrics[key])
        values["auc"].append(ranking(sample, "changed_S"))
    return {k: np.quantile(v, [.025, .975]).tolist() for k, v in values.items()}


def analyze(rows, output, folds, *, primary_family=False):
    """Write separate analysis products; never mutate raw records or E-009."""
    output.mkdir(parents=True, exist_ok=True)
    patients = patient_rows(rows)
    io.write_csv(output / "patients.csv", patients)
    baseline = list({r["case_id"]: r for r in rows}.values())
    thresholds = roc_thresholds(baseline)
    io.write_csv(output / "thresholds.csv", thresholds)
    reference = classification(baseline, "S", 2.375)
    contrasts, groups, loo, predictions, fits, controls, behavior = [], [], [], [], [], [], []
    for cid in sorted({r["id"] for r in patients}):
        part = [r for r in patients if r["id"] == cid]
        image_part = [r for r in rows if r["id"] == cid]
        for outcome in OUTCOMES:
            subset = [r for r in part if r["outcome"] == outcome]
            groups.append({"id": cid, "outcome": outcome,
                           "S": distribution([r["S"] for r in subset]),
                           "delta_S": distribution([r["delta_S"] for r in subset]),
                           "baseline_effect_spearman": correlation([r["S"] for r in subset], [r["delta_S"] for r in subset])})
        for name, (first, second) in CONTRASTS.items():
            a = [r for r in part if r["outcome"] == first]
            b = [r for r in part if r["outcome"] == second]
            contrasts.append({"id": cid, "contrast": name,
                              **contrast([r["delta_S"] for r in a], [r["delta_S"] for r in b])})
            if name == "D_neg":
                for omitted in a + b:
                    other_a = [r["delta_S"] for r in a if r["patient"] != omitted["patient"]]
                    other_b = [r["delta_S"] for r in b if r["patient"] != omitted["patient"]]
                    loo.append({"id": cid, "omitted_patient": omitted["patient"],
                                "outcome": omitted["outcome"], "D_neg": float(np.mean(other_a) - np.mean(other_b))})
        pred, fitted, summary = score_controls(part, folds)
        predictions.extend(pred)
        fits.extend(fitted)
        controls.extend(summary)
        changed = classification(image_part)
        best_tp = max(t["TP"] for t in thresholds if t["FP"] <= changed["FP"])
        best_fp = min(t["FP"] for t in thresholds if t["TP"] >= changed["TP"])
        behavior.append({"id": cid, "image_counts": changed,
                         "patient_mean_score_counts": classification(part),
                         "patient_metric_ci95": patient_metric_intervals(part),
                         "patient_auc": ranking(part, "changed_S"), "image_auc": ranking(image_part, "changed_S"),
                         "baseline_patient_auc": ranking(part, "S"), "baseline_image_auc": ranking(image_part, "S"),
                         "rank_spearman": correlation([r["S"] for r in part], [r["changed_S"] for r in part]),
                         "invalid_answer_events": sum(not r["answer_format_valid"] for r in image_part),
                         "corrected_events": sum(r["corrected"] for r in image_part),
                         "introduced_events": sum(r["introduced"] for r in image_part),
                         "corrected_patients": sorted({r["patient"] for r in image_part if r["corrected"]}),
                         "introduced_patients": sorted({r["patient"] for r in image_part if r["introduced"]}),
                         "mean_KL": float(np.mean([r["kl_from_baseline"] for r in part])),
                         "threshold_envelope_best_TP_at_most_this_FP": best_tp,
                         "threshold_envelope_min_FP_at_least_this_TP": best_fp,
                         "exceeds_original_threshold_envelope": changed["TP"] > best_tp})
    primary = [r for r in contrasts if r["contrast"] == "D_neg"]
    if primary_family:
        if len(primary) != 34:
            raise ValueError("BH primary family requires all 34 layers")
        for r, q in zip(primary, bh_adjust([r["p"] for r in primary])):
            r["BH_q_34"] = q
    io.write_csv(output / "contrasts.csv", [{k: v for k, v in r.items() if k not in ("first", "second")} |
                                             {"n_first": r["first"]["n"], "n_second": r["second"]["n"]} for r in contrasts])
    io.write_csv(output / "leave_one_patient_out.csv", loo)
    io.write_csv(output / "score_predictions.csv", predictions)
    io.write_csv(output / "score_fits.csv", fits)
    special = [r for r in rows if r["outcome"] == "FN" or r["S"] == 0]
    io.write_csv(output / "FN_and_tied_cases.csv", [{k: r[k] for k in
        ("case_id", "patient", "outcome", "id", "S", "changed_S", "delta_S", "corrected", "introduced")} for r in special])
    result = {"complete_conditions": len(primary), "images": len(baseline), "patients": len({r["patient"] for r in patients}),
              "reference_threshold_2.375": reference, "baseline_image_counts": classification(baseline, "S"),
              "contrasts": contrasts, "groups": groups, "controls": controls, "behavior": behavior,
              "bootstrap_resamples": RESAMPLES, "seed": 42,
              "limitations": ["Naturally occurring groups; metadata and score confounding remain",
                              "Exploratory discovery intervals/tests; no independent confirmation",
                              "Knockouts renormalize remaining attention and may activate fallback pathways"]}
    atomic_json(output / "analysis.json", result)
    return result


def joint_comparison(single, joint):
    lookup = {(r["case_id"], r["start"]): r for r in single}
    result = []
    for row in joint:
        if row["route"] != "B" or row["end"] - row["start"] != 5:
            continue
        part = [lookup[(row["case_id"], i)] for i in range(row["start"], row["end"] + 1)]
        if any(not np.isclose(r["S"], row["S"], rtol=1e-5, atol=1e-5) for r in part):
            raise ValueError("Unmatched single/joint baselines; fresh matched windows required")
        summed = sum(r["delta_S"] for r in part)
        result.append({"case_id": row["case_id"], "patient": row["patient"], "outcome": row["outcome"],
                       "id": row["id"], "joint_delta_S": row["delta_S"],
                       "sum_single_delta_S": summed, "joint_minus_sum": row["delta_S"] - summed})
    return result


def metadata_report(rows, metadata, output):
    """Descriptive imbalance, with patient weighting and explicit unknowns."""
    by_case = {r["case_id"]: r for r in rows}
    groups = defaultdict(list)
    for r in metadata:
        if r["case_id"] not in by_case:
            raise ValueError("Metadata case outside discovery records")
        groups[r["patient"]].append(r)
    records = []
    for outcome in OUTCOMES:
        patients = [part for part in groups.values() if by_case[part[0]["case_id"]]["outcome"] == outcome]
        for key in metadata[0]:
            if key in ("case_id", "patient"):
                continue
            if key in ("Sex", "projection", "view", "annotation_status"):
                categories = sorted({str(r[key]) for part in patients for r in part})
                for category in categories:
                    values = [np.mean([str(r[key]) == category for r in part]) for part in patients]
                    records.append({"outcome": outcome, "field": key, "category": category,
                                    "patients": len(values), "patient_weighted_fraction": float(np.mean(values))})
            else:
                values = []
                for part in patients:
                    valid = [r[key] for r in part if r[key] is not None and r[key] >= 0]
                    if valid:
                        values.append(float(np.mean(valid)))
                records.append({"outcome": outcome, "field": key, "category": "numeric",
                                "patients": len(values), "unknown_patients": len(patients) - len(values),
                                "summary": distribution(values)})
    atomic_json(output / "metadata_imbalance.json", records)


def joint_summary(rows, output):
    values = defaultdict(list)
    for r in rows:
        values[(r["id"], r["outcome"], r["patient"])].append(r)
    patients = []
    for (condition, outcome, patient), part in sorted(values.items()):
        patients.append({"id": condition, "outcome": outcome, "patient": patient,
                         **{k: float(np.mean([r[k] for r in part])) for k in
                            ("joint_delta_S", "sum_single_delta_S", "joint_minus_sum")}})
    io.write_csv(output / "individual_vs_joint_patients.csv", patients)
    summaries = []
    for cid in sorted({r["id"] for r in patients}):
        for outcome in ("all", *OUTCOMES):
            part = [r for r in patients if r["id"] == cid and (outcome == "all" or r["outcome"] == outcome)]
            summaries.append({"id": cid, "outcome": outcome, **{k: distribution([r[k] for r in part]) for k in
                              ("joint_delta_S", "sum_single_delta_S", "joint_minus_sum")}})
    atomic_json(output / "individual_vs_joint_summary.json", summaries)


def study_figures(output, result):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(13, 9), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(2, 1, 1)
    for outcome in OUTCOMES:
        part = [r for r in result["groups"] if r["outcome"] == outcome]
        x = np.arange(len(part))
        y = np.array([r["delta_S"]["mean"] for r in part])
        bounds = np.array([r["delta_S"]["ci95"] for r in part])
        line, = ax.plot(x, y, label=f"{outcome} (n={part[0]['delta_S']['n']})")
        ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=line.get_color(), alpha=.15)
    ax.axhline(0, color="gray", lw=.7)
    ax.set(title="Individual route-B knockouts · patient means and exploratory 95% intervals", ylabel="ΔS (Yes − No logits)")
    ax.legend()
    ax = fig.add_subplot(2, 1, 2)
    for name in CONTRASTS:
        part = [r for r in result["contrasts"] if r["contrast"] == name]
        x = np.arange(len(part))
        line, = ax.plot(x, [r["effect"] for r in part], label=name)
        bounds = np.array([r["ci95"] for r in part])
        ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=line.get_color(), alpha=.12)
    ax.axhline(0, color="gray", lw=.7)
    ax.set(xlabel="Zero-based decoder layer", ylabel="Difference in mean ΔS", xticks=range(0, 34, 2),
           title="D_neg FP−TN (primary); D_yes FP−TP; D_pos FN−TP (four FN patients)")
    ax.legend()
    fig.savefig(output / "layer_profiles.png", dpi=150)
