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

import html
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from medgemma_grounding.medgemma_features import OUTPUT, file_sha256, load_cache


def feature_values(activations, feature):
    """Adapt sparse [tokens, features] tensors without densifying the dictionary."""
    selected = {}
    for case_id, sparse in activations.items():
        indices, values = sparse.indices(), sparse.values()
        mask = indices[1] == feature
        tokens = torch.zeros(sparse.shape[0], dtype=values.dtype)
        tokens[indices[0, mask]] = values[mask]
        selected[case_id] = tokens.numpy()
    return selected


def feature_catalog(manifest, activations):
    maxima = {group: torch.zeros(manifest["sae_width"]) for group in ("image", "text")}
    for case in manifest["cases"]:
        sparse = activations[case["case_id"]]
        rows, columns = sparse.indices()
        for group, maximum in maxima.items():
            eligible = torch.tensor([t["group"] == group for t in case["tokens"]])
            mask = eligible[rows]
            maximum.scatter_reduce_(0, columns[mask], sparse.values()[mask], reduce="amax")
    active = ((maxima["image"] > 0) | (maxima["text"] > 0)).nonzero().flatten().tolist()
    active.sort(key=lambda i: (-maxima["image"][i].item(), i))
    return [
        {"feature": i, "image_max": maxima["image"][i].item(), "text_max": maxima["text"][i].item()}
        for i in active
    ]


def case_scores(manifest, values, sort_by="image_max"):
    """Keep all cases, including zeros; labels never determine the ranking."""
    rows = []
    for case in manifest["cases"]:
        row = {"case_id": case["case_id"], "expert_effusion": "present" if case["label"] else "absent",
               "answer": case["answer"]}
        for group in ("image", "text"):
            mask = np.array([t["group"] == group for t in case["tokens"]])
            selected = values[case["case_id"]][mask]
            row.update({
                f"{group}_max": float(selected.max()) if selected.size else 0.0,
                f"{group}_mean": float(selected.mean()) if selected.size else 0.0,
                f"{group}_active_fraction": float(np.count_nonzero(selected) / selected.size) if selected.size else 0.0,
            })
        rows.append(row)
    return sorted(rows, key=lambda row: (-row[sort_by], row["case_id"]))


def color_scale(manifest, values):
    """One feature scale across image/text positions in all cases; omit specials."""
    return max((float(values[c["case_id"]][i]) for c in manifest["cases"]
                for i, t in enumerate(c["tokens"]) if t["group"] in {"image", "text"}), default=0.0)


def image_map(case, values):
    layout = case["spatial_layout"]
    selected = values[np.array([t["group"] == "image" for t in case["tokens"]])]
    grid = layout.get("grid", [])
    if (not layout.get("verified") or layout.get("mapping") != "direct_resize_row_major"
            or len(grid) != 2 or min(grid) <= 0 or grid[0] * grid[1] != selected.size):
        return None
    return selected.reshape(grid)


def make_overlay(image, heat, scale):
    """Imageomics-style overlay with shared scaling and exact output dimensions.

    Nearest-neighbor resize avoids transposition, smoothing, and the upstream
    integer-tile truncation when image dimensions are not divisible by the grid.
    """
    image = image.convert("RGB")
    if scale <= 0:
        return image.copy()
    normalized = np.clip(heat / scale, 0, 1).astype(np.float32)
    heat_up = np.asarray(Image.fromarray(normalized).resize(image.size, Image.Resampling.NEAREST))
    rgb = np.asarray(image, dtype=np.float32) / 255
    alpha = (0.8 * heat_up)[..., None]
    blended = (1 - alpha) * rgb + alpha * colormaps["inferno"](heat_up)[..., :3]
    return Image.fromarray(np.clip(blended * 255, 0, 255).astype(np.uint8))


def token_highlights(tokens, values, scale):
    spans = []
    for token, value in zip(tokens, values, strict=True):
        if token["group"] != "text":
            continue
        rgb = colormaps["inferno"](float(value) / scale if scale > 0 else 0)[:3]
        color = ",".join(str(round(c * 255)) for c in rgb)
        foreground = "black" if sum(rgb) > 1.5 else "white"
        # Literal newlines in attributes break Markdown's HTML block parsing.
        title = html.escape(
            f"position {token['index']} | id {token['id']} | {token['piece']!r} | activation {value:.6g}",
            quote=True,
        )
        text = html.escape(token["text"]).replace("\n", "↵").replace(" ", "·") or "∅"
        spans.append(f'<span title="{title}" style="background:rgb({color});color:{foreground};padding:3px;margin:1px;display:inline-block">{text}</span>')
    return "".join(spans)


def run_app():
    import streamlit as st

    st.set_page_config(page_title="MedGemma SAE features", layout="wide")
    st.title("MedGemma layer-14 SAE features")
    discovery = Path("results/gemma_scope/discovery_v1")
    default = discovery if (discovery / "discovery_report.json").exists() else OUTPUT
    directory = Path(st.sidebar.text_input("Saved feature directory", str(default))).expanduser()
    interventions = st.sidebar.text_input("Optional intervention results directory", "")
    if interventions:
        selected = Path(interventions).expanduser()
        if (selected / "protocol.json").exists() and read_viewer_experiment(selected) == "E-009":
            from medgemma_grounding.attention_analysis import viewer
            try:
                viewer(selected)
                attention_case_images(selected)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                st.error(f"Could not open attention experiment: {error}")
            return
        run_intervention_viewer(Path(interventions).expanduser(), directory)
        if not st.sidebar.checkbox("Also show original SDPA discovery reference", value=False):
            return
    if (directory / "protocol.json").exists():
        run_discovery_viewer(directory)
        return
    st.caption("20-case workflow pilot · Feature descriptions require further validation on separate patients.")

    @st.cache_data(show_spinner=False)
    def cached_data(directory, manifest_time, tensor_time):
        manifest, activations = load_cache(directory)
        return manifest, activations, feature_catalog(manifest, activations)

    try:
        manifest, activations, catalog = cached_data(
            str(directory.resolve()), (directory / "manifest.json").stat().st_mtime_ns,
            (directory / "activations.pt").stat().st_mtime_ns,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        st.error(f"Could not open feature cache: {error}")
        st.info("Run: uv run python -m medgemma_grounding.medgemma_features")
        return
    default = catalog[0]["feature"] if catalog else 0
    feature = int(st.sidebar.number_input("Feature ID", min_value=0, max_value=manifest["sae_width"] - 1, value=default))
    sort_by = st.sidebar.selectbox("Rank cases by", ["image_max", "text_max"], key="sort_by")
    overlay = st.sidebar.checkbox("Overlay image-position activations", value=False)
    values = feature_values(activations, feature)
    rows = case_scores(manifest, values, sort_by)
    scale = color_scale(manifest, values)
    st.subheader(f"Feature {feature}")
    st.caption(f"Shared feature scale: 0–{scale:.6g} across eligible image/text tokens in all cases. Scores are raw SAE activations.")
    with st.expander(f"Features active in this cohort ({len(catalog)})"):
        st.dataframe(catalog, hide_index=True, width="stretch")
    st.dataframe(rows, hide_index=True, width="stretch", height=300)
    if scale == 0:
        st.info("This feature is inactive at all image and text positions in these cases.")
    by_id = {c["case_id"]: c for c in manifest["cases"]}

    def open_image(case):
        path = Path(manifest["project_root"]) / case["image_path"]
        if file_sha256(path) != case["image_sha256"]:
            raise ValueError(f"Image changed since extraction: {path}")
        with Image.open(path) as source:
            return source.convert("RGB")

    st.subheader("Top cases")
    columns = st.columns(5)
    for column, row in zip(columns, rows[:5]):
        case = by_id[row["case_id"]]
        with column:
            try:
                image = open_image(case)
                heat = image_map(case, values[case["case_id"]])
                st.image(make_overlay(image, heat, scale) if overlay and heat is not None else image,
                         width="stretch")
            except (OSError, ValueError) as error:
                st.error(f"Cannot open X-ray: {error}")
            st.caption(f"{case['case_id']} · {sort_by}={row[sort_by]:.4g}")
            st.caption(f"Expert effusion: {row['expert_effusion']} · MedGemma: {row['answer']}")

    case_id = st.selectbox("Inspect case (includes inactive examples)", [r["case_id"] for r in rows], key="case_id")
    case = by_id[case_id]
    selected = values[case_id]
    st.write(f"Expert effusion: {'present' if case['label'] else 'absent'} · MedGemma answer: {case['answer']}")
    st.caption(case["image_path"])
    heat = image_map(case, selected)
    left, right = st.columns(2)
    with left:
        try:
            image = open_image(case)
            st.image(make_overlay(image, heat, scale) if overlay and heat is not None else image,
                     caption="X-ray", width="stretch")
        except (OSError, ValueError) as error:
            st.error(f"Cannot open X-ray: {error}")
    with right:
        if heat is None:
            st.warning("Spatial mapping unverified: overlay disabled. Inspect numeric token activations below.")
        else:
            rgb = (colormaps["inferno"](heat / scale if scale > 0 else np.zeros_like(heat))[..., :3] * 255).astype(np.uint8)
            st.image(Image.fromarray(rgb).resize((320, 320), Image.Resampling.NEAREST), caption="Image-associated positions: row-major 16×16")
            st.caption("SAE activation at image-associated decoder positions. Attention mixes information; this is not a pathology-localization or causal-attribution map.")
    st.subheader("Input text-token activations")
    st.caption("Hover for position, token ID and raw value. These are prefill activations; generated answers are context only.")
    st.markdown(token_highlights(case["tokens"], selected, scale), unsafe_allow_html=True)
    token_rows = [{**t, "activation": float(selected[i])} for i, t in enumerate(case["tokens"])]
    with st.expander("All token values (image tokens follow row-major grid order)"):
        st.dataframe([t for t in token_rows if t["group"] != "special"], hide_index=True, width="stretch")
    with st.expander("Special-token values (excluded from color scaling and ranking)"):
        st.dataframe([t for t in token_rows if t["group"] == "special"], hide_index=True, width="stretch")
    with st.expander("Experiment settings"):
        st.json({k: v for k, v in manifest.items() if k != "cases"})
        st.json(case["spatial_layout"])


def mask_display(image, mask, *, feature_heat=None, scale=0):
    """Original-coordinate expert boundary, optionally on a decoder-position map."""
    from scipy.ndimage import binary_erosion

    shown = make_overlay(image, feature_heat, scale) if feature_heat is not None else image.convert("RGB")
    if mask is None:
        return shown
    if mask.shape != (image.height, image.width):
        raise ValueError("Expert-mask dimensions differ from original image")
    boundary = mask & ~binary_erosion(mask, iterations=max(1, round(image.width / 600)))
    rgb = np.asarray(shown).copy()
    rgb[boundary] = [0, 255, 100]
    return Image.fromarray(rgb)


def run_discovery_viewer(directory):
    """Read small summaries; load token activations only for displayed cases."""
    import csv
    import json
    import streamlit as st
    from medgemma_grounding.chexlocalize import expert_mask
    from medgemma_grounding.medgemma_features import load_discovery_case, identity_hash

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
        st.info("After extraction finishes, run: uv run python -m medgemma_grounding.feature_analysis --experiment-dir " + str(directory))
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


def signed_overlay(image, heat, scale):
    """Diverging signed attribution; zero is transparent, symmetric fixed limits."""
    image = image.convert("RGB")
    if scale <= 0:
        return image.copy()
    normalized = np.clip(np.asarray(heat, dtype=np.float32) / scale, -1, 1)
    up = np.asarray(Image.fromarray(normalized).resize(image.size, Image.Resampling.NEAREST))
    rgb = np.asarray(image, dtype=np.float32) / 255
    alpha = (0.8 * np.abs(up))[..., None]
    colors = colormaps["coolwarm"]((up + 1) / 2)[..., :3]
    return Image.fromarray(np.clip(((1 - alpha) * rgb + alpha * colors) * 255, 0, 255).astype(np.uint8))


def run_intervention_viewer(directory, source):
    """Optional model-free E-005 section. Reads partial and failed runs too."""
    import streamlit as st
    from medgemma_grounding.medgemma_interventions import read_json, load_case
    from medgemma_grounding.medgemma_features import identity_hash, load_discovery_case

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


def read_viewer_experiment(directory):
    import json
    return json.loads((directory / "protocol.json").read_text()).get("experiment")


def attention_case_images(directory):
    """Original discovery X-rays, never a spatial overlay inferred from depth."""
    import streamlit as st
    from medgemma_grounding.medgemma_interventions import read_json
    from medgemma_grounding.chexlocalize import expert_mask
    manifest = read_json(directory / "discovery_manifest.json")
    with st.expander("Inspect original discovery X-rays and valid expert outlines"):
        case_id = st.selectbox("X-ray discovery case", [c["case_id"] for c in manifest["cases"]])
        case = next(c for c in manifest["cases"] if c["case_id"] == case_id)
        root = Path(st.text_input("Local data root for original images", "data")).expanduser()
        path = root / case["relative_image"]
        if not path.exists():
            st.warning("Original discovery image unavailable at this data root. No placeholder image is substituted.")
            return
        if file_sha256(path) != case["image_sha256"]:
            raise ValueError("Discovery X-ray checksum changed")
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        st.write({"case_id": case_id, "patient": case["patient"], "expert_label": case["label"],
                  "historical_outcome": case["historical_outcome"], "historical_answer_score": case["historical_answer_score"]})
        displays = [("Original X-ray (no depth overlay)", image)]
        annotation = root / "CheXlocalize" / "gt_annotations_val.json"
        segmentation = root / "CheXlocalize" / "gt_segmentations_val.json"
        if annotation.exists() and segmentation.exists():
            status, mask, detail = expert_mask(case, read_json(annotation), read_json(segmentation))
            if status == "invalid/mismatched":
                st.warning("Mismatched expert annotation: outline disabled.")
            elif mask is not None:
                displays.append(("Expert outline, independent of knockout effects", mask_display(image, mask)))
            st.caption(f"Mask status: {status}; {detail}. Missing masks do not label a case negative.")
        else:
            st.caption("Expert annotation files unavailable; no outline substituted.")
        for column, (caption, displayed) in zip(st.columns(len(displays)), displays):
            column.image(displayed, caption=caption)
        st.caption("Depth-profile sensitivity is not anatomical localization or a causal pixel-region map.")


def run_targeted_viewer(directory):
    """E-008: saved discovery evidence, never model loading or held-out browsing."""
    import csv
    import json
    import streamlit as st
    from medgemma_grounding import medgemma_targeted as study, targeted_analysis as analysis
    from medgemma_grounding.medgemma_interventions import read_json
    from medgemma_grounding.medgemma_features import load_discovery_case
    from medgemma_grounding.chexlocalize import expert_mask

    st.subheader("Targeted discovery: representation, influence and errors")
    try:
        protocol = read_json(directory / "protocol.json")
        parent = study.validate_protocol(directory, protocol, full=False)
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
                    signed = analysis.uncertainty(outcome_rows, "matched_signed_excess")
                    absolute = analysis.uncertainty(outcome_rows, "matched_abs_excess")
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
                        signed = analysis.uncertainty(part, "actual_delta")
                        control_outcomes.append({"bank": bank, "outcome": outcome, "patients": signed["patients"],
                            "images": signed["images"], "mean_delta": signed["mean"], "delta_ci95": signed["ci95"],
                            "mean_absolute": analysis.uncertainty(part, "absolute_effect")["mean"],
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
            record, payload = study.source_case(protocol, parent, case_id)
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
                analysis.save_visual_hypothesis(directory, protocol, group, feature, hypothesis, counterexamples)
                st.success("Saved separately from frozen measurements; not a ground-truth feature label.")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        st.error(f"Cannot open targeted discovery: {error}")


if __name__ == "__main__":
    run_app()
