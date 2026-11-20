"""Browse saved MedGemma SAE features with a local Streamlit gallery.

Gallery/ranking/overlay adapted from Imageomics saev-demo, demo/feature_vis.py:
https://github.com/Imageomics/saev-demo/tree/5fa496c9f8b63f503b1c8d9b0b9ef25dcd81b09e
The DINOv2/saev inference path is replaced by a saved MedGemma feature adapter.

Upstream MIT License — Copyright (c) 2023 Joseph Bloom

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from medgemma_grounding.utils.io import file_sha256
from medgemma_grounding.visualization.overlays import (
    feature_values,
    image_map,
    make_overlay,
    signed_overlay,
)
from medgemma_grounding.visualization.targeted_features import run_targeted_viewer


def run_intervention_viewer(directory, source):
    """Optional model-free E-005 section. Reads partial and failed runs too."""
    import streamlit as st
    from medgemma_grounding.utils.io import read_json
    from medgemma_grounding.interventions.feature_runtime import load_case
    from medgemma_grounding.utils.io import identity_hash
    from medgemma_grounding.sae.cache import load_discovery_case

    st.subheader("Layer-14 feature interventions")
    try:
        protocol = read_json(directory / "protocol.json")
        if protocol.get("experiment") == "E-008":
            run_targeted_viewer(directory)
            return
        if (identity_hash({k: v for k, v in protocol.items() if k != "protocol_hash"}) != protocol["protocol_hash"]
                or Path(protocol["source_directory"]).resolve() != source.resolve()):
            raise ValueError("Intervention protocol/source directory mismatch")
        for name, digest in protocol["source_artifact_sha256"].items():
            if file_sha256(source / name) != digest:
                raise ValueError("Source discovery artifact changed")
        report = read_json(directory / "intervention_report.json") if (directory / "intervention_report.json").exists() else None
        fresh = protocol.get("fresh_encodings", False)
        calibration_path = directory / ("calibration_report.json" if fresh else "pilot_report.json")
        pilot = read_json(calibration_path) if calibration_path.exists() else None
        for saved in (report, pilot):
            if saved and saved["protocol_hash"] != protocol["protocol_hash"]:
                raise ValueError("Saved report belongs to another protocol")
        if report and (file_sha256(directory / "intervention_summaries.pt") != report["summary_tensor_sha256"]
                       or file_sha256(directory / "intervention_effects.csv") != report["effects_csv_sha256"]):
            raise ValueError("Intervention report artifact checksum mismatch")
        if not report or not report["complete"]:
            st.warning("Intervention run is incomplete. Available measurements are partial.")
        if pilot:
            st.write("Full-discovery calibration by position group" if fresh else "Pilot calibration by position group")
            if not pilot["complete"]:
                st.warning("Pilot calibration is incomplete. Group labels and the effect floor are provisional; inspect the failure diagnostics before interpreting them.")
            st.json({"tau": pilot["tau"], "groups": pilot["groups"]})
        for path in sorted(directory.glob("failure_*.json")):
            st.error(read_json(path)["error"])
        candidates = read_json(directory / "candidates.json") if (directory / "candidates.json").exists() else None
        if candidates:
            if candidates["candidates_hash"] != identity_hash({k: v for k, v in candidates.items() if k != "candidates_hash"}):
                raise ValueError("Candidate checksum mismatch")
            st.dataframe([{"group": p["group"], "target": p["target"]["feature"],
                           "comparison": p["comparison"]["feature"], "matching_distance": p["matching_distance"]}
                          for p in candidates["pairs"]], hide_index=True)
            st.caption("Matched comparisons are not null features. Matching minimizes activity/norm distance without a maximum-distance cutoff; the nearest available comparison can still be a poor match. Inspect distances and perturbation norms.")
        import csv
        rankings_path = directory / "attribution_rankings.csv"
        if fresh and rankings_path.exists():
            st.write("Attribution rankings with direct calibration evidence")
            st.caption("Estimated removal effects are primary; half/full measurements show where estimates were checked. AUC/Spearman describe associations, not selection criteria. Nonqualifying groups are not trusted for selection.")
            rankings = list(csv.DictReader(rankings_path.open()))
            ranking_group = st.selectbox("Ranking position group", ["image", "final", "other_non_image"])
            st.dataframe(sorted([r for r in rankings if r["group"] == ranking_group],
                                key=lambda r: int(r["rank"]))[:50], hide_index=True)
        effects_path = directory / "intervention_effects.csv"
        effects = list(csv.DictReader(effects_path.open())) if effects_path.exists() else []
        # Pilot measurements are available even when calibration stops expansion.
        measurements = effects or (pilot["measurements"] if pilot else [])
        if measurements:
            chart = [{**r, "predicted_delta": float(r["predicted_delta"]), "actual_delta": float(r["actual_delta"]),
                      "alpha": float(r["alpha"])} for r in measurements]
            bound = max(max(abs(r["predicted_delta"]), abs(r["actual_delta"])) for r in chart) or 1
            st.vega_lite_chart({"layer": [
                {"data": {"values": chart}, "mark": "point", "encoding": {
                    "x": {"field": "predicted_delta", "type": "quantitative"},
                    "y": {"field": "actual_delta", "type": "quantitative"},
                    "color": {"field": "group", "type": "nominal"},
                    "tooltip": [{"field": k} for k in ("case_id", "feature", "group", "alpha", "outcome")]}},
                {"data": {"values": [{"x": -bound, "y": -bound}, {"x": bound, "y": bound}]},
                 "mark": {"type": "line", "strokeDash": [4, 4], "color": "gray"},
                 "encoding": {"x": {"field": "x", "type": "quantitative"}, "y": {"field": "y", "type": "quantitative"}}}
            ]}, width="stretch")
            st.dataframe(chart, hide_index=True)
            st.vega_lite_chart(chart, {"mark": "point", "encoding": {
                "x": {"field": "alpha", "type": "quantitative", "title": "Removal fraction"},
                "y": {"field": "actual_delta", "type": "quantitative"},
                "color": {"field": "outcome", "type": "nominal"},
                "tooltip": [{"field": k} for k in ("case_id", "feature", "group", "actual_delta")]}}, width="stretch")
        if report:
            st.write("Patient-weighted effects and descriptive bootstrap intervals")
            if report["summaries"]:
                st.write("Direct cohort measurements for frozen features (not activation associations)")
                measured_rows = []
                for summary in report["summaries"]:
                    overall = summary["overall"]
                    if not overall["patients"]:
                        continue
                    def mean_delta(stratum):
                        return stratum.get("actual_delta", {}).get("mean")
                    measured_rows.append({
                        "group": summary["group"], "feature": summary["feature"], "role": summary["role"],
                        "removal": summary["alpha"], "patients": overall["patients"],
                        "mean_abs_delta": overall["absolute_effect"]["mean"],
                        "abs_delta_ci95": overall["absolute_effect"]["ci95"],
                        "mean_signed_delta": mean_delta(overall),
                        "correct_mean_delta": mean_delta(summary["correct_answers"]),
                        "FP_mean_delta": mean_delta(summary["outcomes"]["FP"]),
                        "FN_mean_delta": mean_delta(summary["outcomes"]["FN"]),
                        "mean_realized_l2": overall["realized_l2"]["mean"],
                        "corrected_images": summary["corrected_images"],
                        "introduced_images": summary["introduced_images"],
                    })
                st.dataframe(measured_rows, hide_index=True)
            st.json({k: report[k] for k in ("status", "summaries", "target_minus_comparison", "limitations")})
            if fresh:
                st.json({k: report[k] for k in ("stages", "stop_reason", "repaired_outcomes", "calibration_by_group")})
                st.caption("Calibration summaries pool different checked features; only the selected-feature ablations estimate fixed-feature cohort effects. Correct answers, false positives and false negatives use the repaired baseline.")
        available = [case_id for case_id in protocol["discovery_case_ids"] if (directory / "cases" / f"{case_id}.json").exists()]
        if not available:
            st.info("No attribution shards saved yet. Check the pilot diagnostics above.")
            return
        case_id = st.selectbox("Intervention case", available)
        feature = int(st.number_input("Intervention feature ID", min_value=0, max_value=protocol["sae_width"] - 1,
                                      value=int(measurements[0]["feature"]) if measurements else 0))
        record, payload = load_case(directory, protocol, case_id)
        original, source_payload = load_discovery_case(source, case_id)
        activity = feature_values({case_id: payload["features"] if fresh else source_payload["features"]}, feature)[case_id]
        attribution = feature_values({case_id: payload["attribution"]}, feature)[case_id]
        st.dataframe([{**t, "activation": float(activity[i]), "predicted_full_removal_delta": float(attribution[i]),
                       "position_group": next((k for k, v in record["positions"].items() if i in v), "padding")}
                      for i, t in enumerate(record["tokens"])], hide_index=True)
        # Shared scales from compact offline summaries, never per-image maxima.
        compact = torch.load(directory / "intervention_summaries.pt", map_location="cpu", weights_only=True) if report else None
        scale = compact["signed_color_limits"][feature].item() if compact else 0
        image = Image.open(original["image_path"]).convert("RGB")
        heat = image_map(original, attribution)
        columns = st.columns(2)
        answer = record["answer_score"]["decision"] if fresh else original["answer"]
        columns[0].image(image, caption=f"{case_id} · expert {original['label']} · baseline answer {answer}")
        if heat is not None:
            columns[1].image(signed_overlay(image, heat, scale),
                            caption=f"Predicted full-removal score change · blue negative / red positive · ±{scale:.4g}")
        if not compact:
            st.info("Run the offline report stage to compute shared signed color scales.")
        source_report_path = source / "discovery_report.json"
        activity_heat = image_map(original, activity)
        if activity_heat is not None and ((fresh and compact) or (not fresh and source_report_path.exists())):
            activity_scale = (compact["activation_color_limits"][feature].item() if fresh
                              else read_json(source_report_path)["discovery_color_limits"].get(str(feature), 0))
            st.image(make_overlay(image, activity_heat, activity_scale),
                     caption="Fresh eager-backend feature activation" if fresh else "Original feature activation (separate from signed attribution)")
        st.caption("Contextual decoder-position attribution; it does not establish pixel-level causality or anatomical localization. Expert-mask display restrictions remain in the discovery section.")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        st.error(f"Cannot open intervention results: {error}")
