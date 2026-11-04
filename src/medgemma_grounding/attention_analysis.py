"""Saved-only E-009 patient-cluster summaries and Streamlit inspection."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from medgemma_grounding import medgemma_interventions as base
from medgemma_grounding.medgemma_features import atomic_json, file_sha256


def patient_values(rows, metric):
    groups = defaultdict(list)
    for r in rows:
        groups[r["patient"]].append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def summary(rows, metric):
    values = np.array(list(patient_values(rows, metric).values()))
    if not len(values):
        return {"patients": 0, "images": 0, "mean": None, "ci95": None}
    ix = np.random.default_rng(42).integers(len(values), size=(1000, len(values)))
    return {"patients": len(values), "images": len(rows), "mean": float(values.mean()),
            "ci95": np.quantile(values[ix].mean(1), [.025, .975]).tolist()}


def report(output, p, manifest):
    from medgemma_grounding.medgemma_attention import read_record, complete_case
    rows, blanks = [], []
    for cid in p["case_ids"]:
        for c in p["conditions"]:
            path = output / "cases" / cid / f"{c['id']}.json"
            if path.exists():
                r = read_record(path, p)
                if r["id"] != c["id"]:
                    raise ValueError("Condition identity mismatch")
                rows.append(r)
        path = output / "cases" / cid / "blank.json"
        if path.exists():
            blanks.append(read_record(path, p))
    flat = [{k: r[k] for k in ("case_id", "patient", "label", "outcome", "id", "route", "start", "end",
             "S", "M", "changed_S", "changed_M", "delta_S", "delta_M", "corrected", "introduced",
             "original_correct", "changed_correct", "answer_format_valid", "kl_from_baseline", "blocked_edges", "seconds",
             "historical_outcome", "historical_score_difference")} for r in rows]
    base.write_csv(output / "effects.csv", flat)
    strata = {"all": lambda r: True, "correct": lambda r: r["original_correct"],
              "incorrect": lambda r: not r["original_correct"], "positive": lambda r: r["label"] == 1,
              "negative": lambda r: r["label"] == 0,
              **{o: lambda r, o=o: r["outcome"] == o for o in ("TP", "TN", "FP", "FN")}}
    profiles = []
    for c in p["conditions"]:
        part = [r for r in rows if r["id"] == c["id"]]
        for name, predicate in strata.items():
            selected = [r for r in part if predicate(r)]
            x, y = patient_values(selected, "M"), patient_values(selected, "delta_M")
            rho = float(spearmanr(list(x.values()), list(y.values())).statistic) if len(x) >= 3 and np.ptp(list(x.values())) and np.ptp(list(y.values())) else None
            profiles.append({**c, "stratum": name, "delta_S": summary(selected, "delta_S"),
                "delta_M": summary(selected, "delta_M"), "KL": summary(selected, "kl_from_baseline"),
                "corrected_events": sum(r["corrected"] for r in selected), "introduced_events": sum(r["introduced"] for r in selected),
                "corrected_patients": sorted({r["patient"] for r in selected if r["corrected"]}),
                "introduced_patients": sorted({r["patient"] for r in selected if r["introduced"]}),
                "baseline_margin_effect_spearman": rho})
    ncomplete = sum(complete_case(output, p, c) for c in p["case_ids"])
    failures = [base.read_json(f) for f in output.glob("failure_*.json")]
    complete = ncomplete == 142 and len(rows) == 2556 and len(blanks) == 142 and not failures
    if complete and not (output / "runtime_checks.json").exists():
        raise ValueError("Final runtime verification missing")
    result = {"experiment": "E-009", "protocol_hash": p["protocol_hash"], "complete": complete,
              "complete_cases": ncomplete, "knockout_records": len(rows), "expected_knockouts": 2556,
              "blank_links": len(blanks), "unique_blank_forwards": len(list((output / "blanks").glob("*.pt"))),
              "profiles": profiles, "failures": failures,
              "baseline_compatibility": [{"case_id": r["case_id"], "historical_outcome": r["historical_outcome"],
                   "current_outcome": r["outcome"], "score_difference": r["historical_score_difference"], "S": r["S"], "M": r["M"]}
                    for r in rows if r["id"] == p["conditions"][0]["id"]],
              "blank_diagnostic": {"delta_S": summary(blanks, "delta_S"), "agreement": summary(blanks, "answer_agreement"),
                                   "interpretation": "Altered image, not text-only, no clinical blank-image accuracy"},
              "limitations": ["Joint overlapping-window sensitivity, not additive importance or a complete circuit",
                  "A/B differences do not isolate indirect mediation; remaining attention is renormalized",
                  "Discovery-only descriptive intervals; no effusion/anatomy or natural-error-cause claim",
                  "Baseline strata fixed on this backend; treatment events are not independent patients"]}
    figures(output, rows, profiles, p)
    result["artifacts_sha256"] = {f: file_sha256(output / f) for f in ("effects.csv", "profiles.png", "patient_heatmaps.png")}
    atomic_json(output / "report.json", result)
    lines = ["# E-009 decoder-depth routing sensitivity", "", f"Complete: {complete}. Cases: {ncomplete}/142. Knockouts: {len(rows)}/2556.",
             "", "This localizes sensitivity to a mask intervention, not an effusion feature or complete circuit.", "",
             "![Route profiles](profiles.png)", "", "![Patient effects](patient_heatmaps.png)", "",
             "[Detailed summaries](report.json) · [Protocol](protocol.json) · [Methods](methods.md)", "",
             "## Full-depth diagnostics", "", "| Route | Stratum | Patients | Mean ΔM | Corrected events | Introduced events |", "|---|---|---:|---:|---:|---:|"]
    for r in profiles:
        if r["start"] == 0 and r["end"] == 33 and r["stratum"] in ("all", "TP", "TN", "FP", "FN"):
            lines.append(f"| {r['route']} | {r['stratum']} | {r['delta_M']['patients']} | {r['delta_M']['mean']} | {r['corrected_events']} | {r['introduced_events']} |")
    lines += ["", "## Limitations", ""] + [f"- {s}" for s in result["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def figures(output, rows, profiles, p):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(fig)
    for i, route in enumerate(("A", "B"), 1):
        ax = fig.add_subplot(1, 2, i)
        for stratum in ("TP", "TN", "FP", "FN"):
            part = [r for r in profiles if r["route"] == route and r["stratum"] == stratum and r["end"] - r["start"] != 33]
            ax.plot(range(8), [r["delta_M"]["mean"] if r["delta_M"]["mean"] is not None else np.nan for r in part], marker="o", label=stratum)
        ax.axhline(0, color="gray", lw=.8)
        ax.set(xticks=range(8), xticklabels=[f"{c['start']}–{c['end']}" for c in p["conditions"][:8]],
               ylabel="Patient-weighted Δ correctness margin (logits)", title=f"Route {route} · joint window effect")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    lo, hi = min(ax.get_ylim()[0] for ax in fig.axes), max(ax.get_ylim()[1] for ax in fig.axes)
    for ax in fig.axes:
        ax.set_ylim(lo, hi)
    fig.savefig(output / "profiles.png", dpi=150)
    patients = sorted({r["patient"] for r in rows})
    fig = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(fig)
    arrays = []
    for route in ("A", "B"):
        data = np.full((max(1, len(patients)), 9), np.nan)
        for j, c in enumerate([c for c in p["conditions"] if c["route"] == route]):
            values = patient_values([r for r in rows if r["id"] == c["id"]], "delta_M")
            for i, patient in enumerate(patients):
                data[i, j] = values.get(patient, np.nan)
        arrays.append(data)
    limit = max([abs(v) for data in arrays for v in data.ravel() if np.isfinite(v)] or [1.]) or 1.
    for i, (route, data) in enumerate(zip(("A", "B"), arrays), 1):
        ax = fig.add_subplot(1, 2, i)
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(title=f"Route {route} · ΔM (missing ≠ zero)", ylabel="Discovery patients, lexicographic order",
               xticks=range(9), xticklabels=[f"{a['start']}–{a['end']}" for a in p["conditions"][:9]])
        ax.tick_params(axis="x", rotation=45)
    fig.colorbar(im, ax=fig.axes, label="Δ correctness margin (logits)")
    fig.savefig(output / "patient_heatmaps.png", dpi=150)


def viewer(directory):
    import csv
    import streamlit as st
    from medgemma_grounding.medgemma_attention import validate, read_record
    p = base.read_json(directory / "protocol.json")
    manifest = validate(directory, p)
    st.subheader("E-009 image-to-answer routing sensitivity")
    st.caption("Discovery only. Joint windows are not additive single-layer importance; A/B difference is not isolated mediation. No anatomical-localization claim.")
    report_path = directory / "report.json"
    if not report_path.exists():
        st.warning("No cohort report yet. Engineering results are not the complete study.")
    else:
        r = base.read_json(report_path)
        if r["protocol_hash"] != p["protocol_hash"]:
            raise ValueError("Report identity changed")
        for name, sha in r["artifacts_sha256"].items():
            if file_sha256(directory / name) != sha:
                raise ValueError("Saved report artifact changed")
        st.write(f"Completed cases {r['complete_cases']}/142; knockouts {r['knockout_records']}/2556")
        if not r["complete"]:
            st.warning("Partial results; absent effects are not zeros.")
        profiles, patients, cases_tab, blank, checks = st.tabs(["Route A/B profiles", "Patient heatmaps", "Individual discovery cases", "Gray-image diagnostic", "Numerics and compute"])
        with profiles:
            st.image(str(directory / "profiles.png"))
            st.dataframe([{k: v for k, v in t.items() if k not in ("delta_S", "delta_M", "KL")} |
                {"mean_delta_M": t["delta_M"]["mean"], "CI95": t["delta_M"]["ci95"], "patients": t["delta_M"]["patients"]} for t in r["profiles"]], hide_index=True)
        with patients:
            st.image(str(directory / "patient_heatmaps.png"))
        with cases_tab:
            route = st.selectbox("Route", ["A", "B"])
            condition = st.selectbox("Window", [c["id"] for c in p["conditions"] if c["route"] == route])
            with (directory / "effects.csv").open() as f:
                rows = [v for v in csv.DictReader(f) if v["id"] == condition]
            for row in rows:
                for key in ("M", "delta_M", "S", "delta_S"):
                    row[key] = float(row[key])
            st.vega_lite_chart(rows, {"mark": "point", "encoding": {"x": {"field": "M", "type": "quantitative"},
                "y": {"field": "delta_M", "type": "quantitative"}, "color": {"field": "outcome", "type": "nominal"},
                "tooltip": [{"field": k} for k in ("case_id", "outcome", "M", "delta_M")]}}, width="stretch")
            st.dataframe(rows, hide_index=True)
            cid = st.selectbox("Discovery case", p["case_ids"])
            path = directory / "cases" / cid / f"{condition}.json"
            st.json(read_record(path, p) if path.exists() else {"status": "not measured"})
            st.caption("Original and changed answers, correctness and tied cases are saved in this record. No depth-profile overlay is an anatomical map.")
        with blank:
            st.json(r["blank_diagnostic"])
            st.caption("Same image-containing format with uniform gray; not text-only and not clinical accuracy on a negative image.")
        with checks:
            for name in ("benchmark.json", "runtime_checks.json", "verification.json"):
                if (directory / name).exists():
                    st.json(base.read_json(directory / name))
