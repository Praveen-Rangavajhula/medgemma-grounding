"""Saved E-010 layer contrasts, score controls and threshold tradeoffs."""


import medgemma_grounding.experiments.run_attention_layers as run_attention_layers
import medgemma_grounding.utils.io as io

from medgemma_grounding.utils.io import file_sha256


def viewer(directory):
    import csv
    import streamlit as st
    import medgemma_grounding.experiments.run_attention_layers as study
    p = io.read_json(directory / "protocol.json")
    run_attention_layers.validate(directory, p)
    st.subheader("E-010 · correct/error routing comparisons")
    st.caption("Discovery only. Natural groups and score dependence are confounded; routing sensitivity is not an effusion-specific circuit or correction policy.")
    path = directory / "report.json"
    if not path.exists():
        st.warning("No cohort report yet. Existing-window analysis may be inspected below.")
        if (directory / "existing_windows/analysis.json").exists():
            st.json(io.read_json(directory / "existing_windows/analysis.json"))
        return
    report = io.read_json(path)
    if report["protocol_hash"] != p["protocol_hash"]:
        raise ValueError("Report identity changed")
    for name, sha in report["artifacts_sha256"].items():
        if file_sha256(directory / name) != sha:
            raise ValueError("Saved analysis artifact changed")
    st.write(f"{report['completed_cases']}/142 images; {report['knockout_records']}/4,828 single-layer knockouts")
    if not report["complete"]:
        st.warning("Partial study; no completed-family statistical result.")
        return
    data = report["analysis"]
    profiles, distributions, controls, tradeoffs, joint, checks = st.tabs([
        "Layer contrasts", "Patients and scores", "Score-only controls", "Errors and thresholds", "Single versus joint", "Numerics and provenance"])
    condition = st.selectbox("Individual layer", [c["id"] for c in p["conditions"]])
    def csv_rows(name):
        with (directory / name).open() as stream:
            return list(csv.DictReader(stream))
    patients = [r for r in csv_rows("single_layers/patients.csv") if r["id"] == condition]
    for row in patients:
        for key in ("S", "changed_S", "delta_S", "delta_M"):
            row[key] = float(row[key])
    with profiles:
        st.image(str(directory / "layer_profiles.png"))
        st.dataframe(csv_rows("single_layers/contrasts.csv"), hide_index=True)
        st.json([r for r in data["groups"] if r["id"] == condition])
        st.caption("BH family is the 34 primary D_neg tests only. Intervals/tests are exploratory; overlapping historical windows are not independent.")
    with distributions:
        chart = {"mark": "point", "encoding": {"x": {"field": "S", "type": "quantitative"},
                 "y": {"field": "changed_S", "type": "quantitative"}, "color": {"field": "outcome", "type": "nominal"},
                 "tooltip": [{"field": k} for k in ("patient", "outcome", "S", "changed_S", "delta_S")]}}
        st.vega_lite_chart(patients, chart, width="stretch")
        st.vega_lite_chart(patients, {"mark": {"type": "point", "opacity": .5}, "encoding": {
            "x": {"field": "outcome", "type": "nominal"}, "y": {"field": "delta_S", "type": "quantitative"},
            "tooltip": [{"field": k} for k in ("patient", "S", "delta_S")]}}, width="stretch")
        st.dataframe(patients, hide_index=True)
        st.dataframe(csv_rows("single_layers/FN_and_tied_cases.csv"), hide_index=True)
        st.json(io.read_json(directory / "metadata_imbalance.json"))
    with controls:
        st.json([r for r in data["controls"] if r["id"] == condition])
        selected = [r for r in csv_rows("single_layers/score_predictions.csv") if r["id"] == condition]
        for row in selected:
            for key in ("S", "residual", "predicted_S", "changed_S"):
                row[key] = float(row[key])
        st.vega_lite_chart(selected, {"mark": "point", "encoding": {"x": {"field": "S", "type": "quantitative"},
            "y": {"field": "residual", "type": "quantitative"}, "color": {"field": "outcome", "type": "nominal"},
            "column": {"field": "control", "type": "nominal"}}}, width="stretch")
        st.dataframe(selected, hide_index=True)
        st.caption("OOF fits use only original/intervened scores from training patients; residuals do not identify anatomy or disease semantics.")
    with tradeoffs:
        st.json(next(r for r in data["behavior"] if r["id"] == condition))
        thresholds = csv_rows("single_layers/thresholds.csv")
        for row in thresholds:
            for key in ("sensitivity", "specificity", "TP", "FP"):
                row[key] = float(row[key])
            row["FPR"] = 1 - row["specificity"]
        st.vega_lite_chart(thresholds, {"mark": "line", "encoding": {"x": {"field": "FPR", "type": "quantitative"},
            "y": {"field": "sensitivity", "type": "quantitative"}}}, width="stretch")
        st.dataframe(thresholds, hide_index=True)
        st.json(data["reference_threshold_2.375"])
        st.caption("Retrospective discovery threshold comparison; not a deployable policy or held-out clinical result. Treatment events differ from distinct patients.")
    with joint:
        st.json(io.read_json(directory / "individual_vs_joint_summary.json"))
        st.dataframe(csv_rows("individual_vs_joint_patients.csv"), hide_index=True)
        st.caption("Joint minus sum is descriptive non-additivity, not a unique redundancy/cooperation mechanism. Knockouts may induce fallback routes.")
    with checks:
        for name in ("benchmark.json", "runtime_checks.json", "verification.json"):
            if (directory / name).exists():
                st.json(io.read_json(directory / name))
