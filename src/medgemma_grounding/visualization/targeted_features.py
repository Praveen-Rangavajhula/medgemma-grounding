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
import medgemma_grounding.analysis.targeted_features as targeted_analysis
import medgemma_grounding.experiments.run_targeted_features as run_targeted_features
import medgemma_grounding.interventions.targeted_features as targeted_edits
import numpy as np
import torch

from medgemma_grounding.utils.io import file_sha256
from medgemma_grounding.visualization.overlays import (
    feature_values,
    image_map,
    make_overlay,
    mask_display,
)


def run_targeted_viewer(directory):
    """E-008: saved discovery evidence, never model loading or held-out browsing."""
    import csv
    import json
    import streamlit as st
    import medgemma_grounding.experiments.run_targeted_features as study
    import medgemma_grounding.analysis.targeted_features as analysis
    from medgemma_grounding.utils.io import read_json
    from medgemma_grounding.sae.cache import load_discovery_case
    from medgemma_grounding.data.chexlocalize import expert_mask

    st.subheader("Targeted discovery: representation, influence and errors")
    try:
        protocol = read_json(directory / "protocol.json")
        parent = run_targeted_features.validate_protocol(directory, protocol, full=False)
        candidates = read_json(directory / "candidates.json")["candidates"]
        report_path = directory / "report.json"
        report = read_json(report_path) if report_path.exists() else None
        if report:
            if report["protocol_hash"] != protocol["protocol_hash"]:
                raise ValueError("Targeted report protocol differs")
            for name, digest in report["artifacts_sha256"].items():
                if file_sha256(directory / name) != digest:
                    raise ValueError(f"Changed targeted report artifact: {name}")
        if not report or not report["complete"]:
            st.warning("Partial experiment. Missing measurements are not zeros or negative evidence.")
        if report and report["failures"]:
            st.error(report["failures"])
        audit_tab, candidate_tab = st.tabs(["E-007 calibration audit", "Candidate evidence and visual hypotheses"])
        with audit_tab:
            audit = read_json(directory / "e007_audit.json")
            st.caption("Historical substantial-actual metrics are retained. All-pair metrics also include small and zero effects; tested pairs are still a selected sample, not all 16k features.")
            st.image(str(directory / "e007_agreement.png"))
            st.dataframe(audit["agreement"], hide_index=True)
            with (directory / "e007_agreement.csv").open() as f:
                per_feature = list(csv.DictReader(f))
            st.write("Measured-feature diagnostics (small counts and undefined correlations remain explicit)")
            st.dataframe(per_feature, hide_index=True)
            st.write("Predicted-large / measured-small examples")
            st.dataframe(audit["false_alarm_examples"], hide_index=True)
            st.write("Opposite-sign examples")
            st.dataframe(audit["opposite_sign_examples"], hide_index=True)
            st.write("Historical natural comparisons, not size-matched controls")
            st.dataframe(audit["natural_comparison_matching"], hide_index=True)
        with candidate_tab:
            labels = [f"{c['group']} / {c['feature']} · {', '.join(c['tracks'])}" for c in candidates]
            chosen = st.selectbox("Frozen candidate", labels)
            candidate = candidates[labels.index(chosen)]
            group, feature = candidate["group"], candidate["feature"]
            st.caption("Discovery patients only. Association is not disease meaning; influence is not error selectivity or visual grounding.")
            st.json(candidate)

            @st.cache_data(show_spinner=False)
            def compact(path, digest):
                if file_sha256(Path(path)) != digest:
                    raise ValueError("Nomination data changed")
                return torch.load(path, map_location="cpu", weights_only=True)

            data = compact(str(directory / "nomination_data.pt"), protocol["artifacts_sha256"]["nomination_data.pt"])
            source = Path(protocol["source_directory"])
            records = []
            for case in protocol["discovery_case_ids"]:
                path = source / "cases" / f"{case}.json"
                if file_sha256(path) != protocol["case_record_sha256"][case]:
                    raise ValueError("Source case metadata changed")
                records.append(read_json(path))
            if data["case_ids"] != [r["case_id"] for r in records]:
                raise ValueError("Compact case alignment differs")
            activities = data["groups"][group]["activity"][:, feature].numpy()
            point_rows = [{"case_id": r["case_id"], "patient": r["patient"], "label": r["label"], "outcome": r["outcome"],
                           "original_score": r["answer_score"]["score"], "activation": float(activities[i])} for i, r in enumerate(records)]
            by_patient = {}
            for r in point_rows:
                by_patient.setdefault(r["patient"], []).append(r)
            patient_rows = [{**v[0], "activation": float(np.mean([r["activation"] for r in v])),
                             "original_score": float(np.mean([r["original_score"] for r in v]))} for v in by_patient.values()]
            st.write("Patient-level activation distributions and answer-score dependence")
            st.caption("Activity means the average over image-associated positions, or the single final-input position. These distributions average images within each patient; example rankings below are image-level.")
            st.vega_lite_chart(patient_rows, {"mark": "boxplot", "encoding": {
                "x": {"field": "outcome", "type": "nominal"}, "y": {"field": "activation", "type": "quantitative"}}}, width="stretch")
            st.vega_lite_chart(patient_rows, {"mark": "point", "encoding": {
                "x": {"field": "original_score", "type": "quantitative"}, "y": {"field": "activation", "type": "quantitative"},
                "color": {"field": "outcome", "type": "nominal"}, "tooltip": [{"field": k} for k in ("patient", "label", "outcome", "activation", "original_score")]}}, width="stretch")
            st.caption("FP–TP and FN–TN hold the original answer fixed, not its margin or other confounds. Only four FN patients exist.")
            original_directory = Path(parent["source_directory"])

            def raw_case(case_id):
                if case_id not in protocol["discovery_case_ids"]:
                    raise ValueError("Confirmation/test browsing is disabled")
                path = original_directory / "cases" / f"{case_id}.json"
                if file_sha256(path) != parent["source_artifact_sha256"][f"cases/{case_id}.json"]:
                    raise ValueError("Image metadata changed")
                raw = read_json(path)
                if raw["partition"] != "discovery" or file_sha256(Path(raw["image_path"])) != raw["image_sha256"]:
                    raise ValueError("Image changed or is not discovery")
                return raw

            for column, title, predicate in zip(st.columns(4),
                    ("Strong activation", "Activating expert negatives", "False positives", "False negatives"),
                    (lambda r: True, lambda r: r["label"] == 0, lambda r: r["outcome"] == "FP", lambda r: r["outcome"] == "FN")):
                ranked = sorted([r for r in point_rows if predicate(r)], key=lambda r: (-r["activation"], r["case_id"]))
                seen, examples = set(), []
                for r in ranked:
                    if r["patient"] not in seen:
                        examples.append(r)
                        seen.add(r["patient"])
                with column:
                    st.write(title)
                    if examples:
                        raw = raw_case(examples[0]["case_id"])
                        st.image(raw["image_path"], caption=f"{examples[0]['case_id']} · activation {examples[0]['activation']:.3g}")
                    st.dataframe(examples[:5], hide_index=True)
            alpha = st.select_slider("Removal strength", options=[.5, 1.], value=1.)
            if report:
                summary = next(s for s in report["summaries"] if (s["group"], s["feature"], s["alpha"]) == (group, feature, alpha))
                st.write(summary["evidence"])
                strata = {"all": summary["all_cases"], "correct": summary["correct_answers"], **summary["outcomes"]}
                st.dataframe([{"stratum": name, "patients": s["patients"], "images": s["images"],
                    "mean_delta": s["actual_delta"]["mean"], "delta_ci95": s["actual_delta"]["ci95"],
                    "mean_abs_delta": s["absolute_effect"]["mean"], "corrected_images": s["corrected_images"],
                    "introduced_images": s["introduced_images"], "corrected_patients": len(s["corrected_patients"]),
                    "introduced_patients": len(s["introduced_patients"])} for name, s in strata.items()], hide_index=True)
                st.json({"matched_size_results": summary["matched"], "exclusions": summary["matching_exclusions"],
                         "matching_diagnostics": report["control_matching"]})
                with (directory / "case_effects.csv").open() as f:
                    effects = [r for r in csv.DictReader(f) if r["group"] == group and int(r["feature"]) == feature and float(r["alpha"]) == alpha]
                for r in effects:
                    for k in ("original_score", "actual_delta", "requested_l2", "realized_l2", "control_mean_delta", "control_mean_absolute", "matched_signed_excess", "matched_abs_excess"):
                        r[k] = float(r[k]) if r[k] else None
                st.write("Matched-size comparison by original outcome")
                comparison_rows = []
                for outcome in ("TP", "TN", "FP", "FN"):
                    outcome_rows = [r for r in effects if r["outcome"] == outcome]
                    signed = targeted_analysis.uncertainty(outcome_rows, "matched_signed_excess")
                    absolute = targeted_analysis.uncertainty(outcome_rows, "matched_abs_excess")
                    comparison_rows.append({"outcome": outcome, "matched_patients": absolute["patients"],
                        "matched_images": absolute["images"], "signed_excess": signed["mean"],
                        "signed_ci95": signed["ci95"], "absolute_excess": absolute["mean"], "absolute_ci95": absolute["ci95"]})
                st.dataframe(comparison_rows, hide_index=True)
                st.caption("These comparisons require all three positive-size matches; excluded cases are not assigned zero effects. Patient intervals condition on discovery selection and this fixed direction bank.")
                with (directory / "measurements.csv").open() as f:
                    control_rows = [r for r in csv.DictReader(f) if r["group"] == group
                                    and int(r["feature"]) == feature and float(r["alpha"]) == alpha
                                    and r["bank"] != "" and r["actual_delta"] != ""]
                for r in control_rows:
                    for key in ("actual_delta", "requested_l2", "realized_l2", "kl_from_baseline"):
                        r[key] = float(r[key])
                    r["absolute_effect"] = abs(r["actual_delta"])
                control_outcomes = []
                for bank in range(protocol["control_bank_size"]):
                    for outcome in ("TP", "TN", "FP", "FN"):
                        part = [r for r in control_rows if int(r["bank"]) == bank and r["outcome"] == outcome]
                        signed = targeted_analysis.uncertainty(part, "actual_delta")
                        control_outcomes.append({"bank": bank, "outcome": outcome, "patients": signed["patients"],
                            "images": signed["images"], "mean_delta": signed["mean"], "delta_ci95": signed["ci95"],
                            "mean_absolute": targeted_analysis.uncertainty(part, "absolute_effect")["mean"],
                            "corrected_events": sum(r["corrected"] == "True" for r in part),
                            "introduced_events": sum(r["introduced"] == "True" for r in part),
                            "distinct_corrected_patients": len({r["patient"] for r in part if r["corrected"] == "True"}),
                            "distinct_introduced_patients": len({r["patient"] for r in part if r["introduced"] == "True"})})
                st.write("Individual random directions: outcomes and error events")
                st.dataframe(control_outcomes, hide_index=True)
                st.caption("All measured controls, including zero-size edits. Unmatched unexecuted controls are omitted, not assigned zero. Direction/strength events can involve the same patient; accidental corrections are not a mitigation policy.")
                with st.expander("Per-case random-direction measurements, executed norms and KL"):
                    st.dataframe(control_rows, hide_index=True)
                st.vega_lite_chart(effects, {"mark": "point", "encoding": {
                    "x": {"field": "original_score", "type": "quantitative"}, "y": {"field": "actual_delta", "type": "quantitative"},
                    "color": {"field": "outcome", "type": "nominal"}, "tooltip": [{"field": k} for k in ("case_id", "patient", "outcome", "actual_delta", "control_mean_delta", "matched_abs_excess")]}}, width="stretch")
                st.vega_lite_chart(effects, {"mark": "point", "encoding": {
                    "x": {"field": "requested_l2", "type": "quantitative"}, "y": {"field": "realized_l2", "type": "quantitative"},
                    "tooltip": [{"field": k} for k in ("case_id", "target_state", "requested_l2", "realized_l2")]}}, width="stretch")
                st.dataframe(effects, hide_index=True)
            case_id = st.selectbox("Inspect candidate discovery case", protocol["discovery_case_ids"])
            record, payload = targeted_edits.source_case(protocol, parent, case_id)
            raw = raw_case(case_id)
            with Image.open(raw["image_path"]) as im:
                image = im.convert("RGB")
            displays = [("Original X-ray", image)]
            cohort = read_json(original_directory / "cohort.json")
            if file_sha256(original_directory / "cohort.json") != parent["source_artifact_sha256"]["cohort.json"]:
                raise ValueError("Cohort changed")
            for path, digest in cohort["metadata_sha256"].items():
                if file_sha256(Path(path)) != digest:
                    raise ValueError("Expert annotations changed")
            root = Path(cohort["data_root"]) / "CheXlocalize"
            status, mask, detail = expert_mask(raw, read_json(root / "gt_annotations_val.json"), read_json(root / "gt_segmentations_val.json"))
            st.caption(f"Expert mask: {status} · {detail}. Missing mask does not label an image negative.")
            if status == "invalid/mismatched":
                st.warning("Mismatched expert annotation: outline disabled.")
            elif mask is not None:
                displays.append(("Expert outline", mask_display(image, mask)))
            activity = feature_values({case_id: payload["features"]}, feature)[case_id]
            attributed = feature_values({case_id: payload["attribution"]}, feature)[case_id]
            if group == "image":
                heat = image_map(raw, activity)
                if heat is not None:
                    # Use a true token-maximum shared scale, not a mean-activity scale.
                    scale_digest = read_json(source / "intervention_report.json")["summary_tensor_sha256"]
                    summary_tensors = torch.load(source / "intervention_summaries.pt", map_location="cpu", weights_only=True)
                    if file_sha256(source / "intervention_summaries.pt") != scale_digest:
                        raise ValueError("Fresh activity colour scales changed")
                    displays.append(("Contextual image-position activity", make_overlay(image, heat, summary_tensors["activation_color_limits"][feature].item())))
            for column, (caption, displayed) in zip(st.columns(len(displays)), displays):
                column.image(displayed, caption=caption)
            st.dataframe([{**record["tokens"][i], "position": i, "activity": float(activity[i]),
                           "predicted_full_removal": float(attributed[i])} for i in record["positions"][group]], hide_index=True)
            st.caption("Decoder-position overlays are contextual. They establish neither anatomical localization nor causal dependence on those pixels. Final-input activity is not overlaid on the X-ray.")
            notes_path = directory / "visual_hypotheses.json"
            notes = read_json(notes_path)["features"].get(f"{group}:{feature}", {}) if notes_path.exists() else {}
            hypothesis = st.text_area("Human-proposed visual hypothesis (unconfirmed)", value=notes.get("hypothesis", ""), key=f"hypothesis_{group}_{feature}")
            counterexamples = st.multiselect("Counterexample discovery cases", protocol["discovery_case_ids"], default=notes.get("counterexample_cases", []), key=f"counterexamples_{group}_{feature}")
            if st.button("Save human hypothesis", key=f"save_{group}_{feature}"):
                targeted_analysis.save_visual_hypothesis(directory, protocol, group, feature, hypothesis, counterexamples)
                st.success("Saved separately from frozen measurements; not a ground-truth feature label.")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        st.error(f"Cannot open targeted discovery: {error}")
