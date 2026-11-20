"""Saved E-009 window-routing profiles and per-patient evidence."""


import medgemma_grounding.utils.io as io

from medgemma_grounding.utils.io import file_sha256


def viewer(directory):
    import csv
    import streamlit as st
    from medgemma_grounding.interventions.attention_runtime import validate
    from medgemma_grounding.interventions.attention_runtime import read_record
    p = io.read_json(directory / "protocol.json")
    manifest = validate(directory, p)
    st.subheader("E-009 image-to-answer routing sensitivity")
    st.caption("Discovery only. Joint windows are not additive single-layer importance; A/B difference is not isolated mediation. No anatomical-localization claim.")
    report_path = directory / "report.json"
    if not report_path.exists():
        st.warning("No cohort report yet. Engineering results are not the complete study.")
    else:
        r = io.read_json(report_path)
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
                    st.json(io.read_json(directory / name))
