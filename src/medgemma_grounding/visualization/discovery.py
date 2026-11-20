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
    mask_display,
    token_highlights,
)


def run_discovery_viewer(directory):
    """Read small summaries; load token activations only for displayed cases."""
    import csv
    import json
    import streamlit as st
    from medgemma_grounding.data.chexlocalize import expert_mask
    from medgemma_grounding.sae.cache import load_discovery_case
    from medgemma_grounding.utils.io import identity_hash

    st.caption("Discovery only · Confirmation inference and browsing are disabled.")
    try:
        protocol = json.loads((directory / "protocol.json").read_text())
        report = json.loads((directory / "discovery_report.json").read_text())
        shortlist = json.loads((directory / "shortlist.json").read_text())
        if report["protocol_hash"] != protocol["protocol_hash"] or shortlist["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("Analysis belongs to another protocol")
        if identity_hash({k: v for k, v in protocol.items() if k != "protocol_hash"}) != protocol["protocol_hash"]:
            raise ValueError("Protocol hash mismatch")
        for name, digest in report["artifacts_sha256"].items():
            if file_sha256(directory / name) != digest:
                raise ValueError(f"Changed analysis artifact: {name}")
        if identity_hash({k: v for k, v in shortlist.items() if k != "shortlist_hash"}) != shortlist["shortlist_hash"]:
            raise ValueError("Frozen shortlist hash mismatch")
        with (directory / "rankings.csv").open() as handle:
            rankings = list(csv.DictReader(handle))
        summaries = torch.load(directory / "summaries.pt", map_location="cpu", weights_only=True)
        cases = protocol["discovery_cases"]
        for case in cases:
            path = directory / "cases" / f"{case['case_id']}.json"
            if file_sha256(path) != shortlist["input_records_sha256"][case["case_id"]]:
                raise ValueError("Case record changed after shortlist freeze")
        records = [json.loads((directory / "cases" / f"{c['case_id']}.json").read_text()) for c in cases]
        if any(c["partition"] != "discovery" for c in records):
            raise ValueError("Confirmation case in discovery results")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        st.error(f"Cannot open discovery results: {error}")
        st.info('After extraction finishes, run: uv run python -m medgemma_grounding.analysis.feature_discovery --experiment-dir ' + str(directory))
        return
    st.info("Exploratory associations, not validated effusion features. No causal effects have been tested.")
    st.subheader("Frozen candidate shortlist")
    candidates = shortlist["candidates"]
    st.dataframe([{k: v for k, v in c.items() if k != "secondary_measurements"} for c in candidates], hide_index=True)
    feature = int(st.sidebar.number_input("Feature ID", min_value=0, max_value=protocol["sae_width"] - 1,
                                          value=candidates[0]["feature"] if candidates else 0))
    representation = st.sidebar.selectbox("Representation", ["image_mean", "final"])
    for rep in ("image_mean", "final"):
        for target in ("label", "answer_score"):
            with st.expander(f"Ranking: {rep} versus {target}"):
                table = [r for r in rankings if r["representation"] == rep and r["target"] == target]
                table.sort(key=lambda r: (r["eligible"] != "True", -float(r["strength"]), int(r["feature"])))
                st.dataframe(table[:100], hide_index=True)
    chosen = next((c for c in candidates if c["feature"] == feature), None)
    if chosen:
        st.write(f"Selected via {chosen['selection_track']} · Direction: {chosen['direction']}")
        st.caption(f"Hypothesis: {chosen['hypothesis'] or 'not yet interpreted'} · Source: {chosen['hypothesis_source']}")
        with st.expander("Candidate provenance and secondary measurements"):
            st.json(chosen)
    else:
        st.write("Not automatically selected; arbitrary features remain browsable.")
    scatter = [{"case_id": c["case_id"], "patient": c["patient"], "expert_effusion": "present" if c["label"] else "absent",
                "parsed_answer": c["parsed_answer"], "answer_score": c["answer_score"]["score"],
                "activation": float(summaries[representation][i, feature])} for i, c in enumerate(records)]
    st.subheader(f"Feature {feature}: {representation} versus answer score")
    st.vega_lite_chart(scatter, {
        "mark": {"type": "point", "filled": True},
        "encoding": {"x": {"field": "activation", "type": "quantitative"},
                     "y": {"field": "answer_score", "type": "quantitative"},
                     "color": {"field": "expert_effusion", "type": "nominal"},
                     "tooltip": [{"field": name} for name in
                                 ("case_id", "patient", "expert_effusion", "parsed_answer", "activation", "answer_score")]},
    }, width="stretch")
    st.dataframe(scatter, hide_index=True)
    st.caption("Plot rows are images; selection statistics average images within each patient.")
    st.subheader("Counterexamples")
    categories = (
        ("Activating positives", lambda c: c["label"] == 1, True),
        ("Activating negatives", lambda c: c["label"] == 0, True),
        ("Weak/inactive positives", lambda c: c["label"] == 1, False),
        ("Model mistakes / invalid answers", lambda c: c["parsed_answer"] != ("Yes" if c["label"] else "No"), True),
    )
    for column, (title, predicate, descending) in zip(st.columns(4), categories):
        indices = [i for i, c in enumerate(records) if predicate(c)]
        indices.sort(key=lambda i: ((-1 if descending else 1) * float(summaries[representation][i, feature]), records[i]["case_id"]))
        with column:
            st.write(title)
            if indices:
                shown = records[indices[0]]
                try:
                    path = Path(shown["image_path"])
                    if file_sha256(path) != shown["image_sha256"]:
                        raise ValueError("Image changed since extraction")
                    with Image.open(path) as source:
                        st.image(source.convert("RGB"), caption=shown["case_id"], width="stretch")
                except (OSError, ValueError) as error:
                    st.error(str(error))
            st.dataframe([scatter[i] for i in indices[:5]], hide_index=True)
    case_id = st.selectbox("Inspect discovery case", [c["case_id"] for c in records], key="discovery_case_id")
    try:
        case, payload = load_discovery_case(directory, case_id, protocol)
        cohort = json.loads((directory / "cohort.json").read_text())
        if file_sha256(directory / "cohort.json") != protocol["cohort_sha256"]:
            raise ValueError("Cohort changed since extraction")
        for path, digest in cohort["metadata_sha256"].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError("Label/annotation metadata changed since extraction")
        root = Path(cohort["data_root"])
        # Only the selected discovery case is decoded, never confirmation masks.
        annotations = json.loads((root / "CheXlocalize" / "gt_annotations_val.json").read_text())
        segmentations = json.loads((root / "CheXlocalize" / "gt_segmentations_val.json").read_text())
        status, mask, detail = expert_mask(case, annotations, segmentations)
        with Image.open(case["image_path"]) as source:
            image = source.convert("RGB")
        values = feature_values({case_id: payload["features"]}, feature)[case_id]
        heat = image_map(case, values) if representation == "image_mean" else None
        scale = report["discovery_color_limits"][str(feature)]
        st.caption(f"Fixed discovery feature colour range: 0–{scale:.6g}; no per-image rescaling.")
        if scale == 0:
            st.info("This feature is inactive at eligible image/text positions in discovery.")
        elif values[[i for i, t in enumerate(case["tokens"]) if t["group"] in {"image", "text"}]].max() > scale:
            st.warning("Activations exceed the frozen discovery colour limit and are clipped.")
        st.write(f"Patient {case['patient']} · Expert: {case['label']} · Answer: {case['answer']!r} · Parsed: {case['parsed_answer']}")
        st.write(f"Expert mask: {status} — {detail}")
        if status == "invalid/mismatched":
            st.warning("Expert outline disabled: released mask did not pass alignment/agreement checks. Image remains in nonspatial discovery.")
        displays = [("Original X-ray", image), ("Expert-mask outline", mask_display(image, mask))]
        if heat is not None:
            displays.extend([("Decoder-position activation map", make_overlay(image, heat, scale)),
                             ("Expert outline + decoder-position activation map", mask_display(image, mask, feature_heat=heat, scale=scale))])
        for column, (caption, shown) in zip(st.columns(len(displays)), displays):
            with column:
                st.image(shown, caption=caption, width="stretch")
        if representation == "final":
            st.info("Final-position activity has no spatial overlay. Switch to image_mean to inspect image-position activity of this same feature.")
        elif heat is None:
            st.warning("Unsupported token layout; decoder-position overlays disabled.")
        st.caption("Decoder-position activation maps are contextual features, not anatomical localization or causal attribution. Final-position values have no spatial mapping.")
        st.write("Final-position activation:", float(payload["summaries"]["final"][feature]))
        st.caption("Text highlights retain the legacy pilot group, including some template tokens; exact ordinary-text indices are listed separately below.")
        st.markdown(token_highlights(case["tokens"], values, scale), unsafe_allow_html=True)
        with st.expander("Token values and position groups"):
            st.dataframe([{**t, "activation": float(values[i])} for i, t in enumerate(case["tokens"])], hide_index=True)
            st.json(case["positions"])
        with st.expander("Reconstruction distortion — compare score changes with original margin"):
            st.json(case["fidelity"])
        with st.expander("Original CSV labels and co-occurring findings"):
            st.json(case.get("csv_record", {}))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        st.error(f"Cannot open discovery case: {error}")
    with st.expander("Pooled fidelity and protocol"):
        st.json(report["fidelity"])
        st.json(protocol)
