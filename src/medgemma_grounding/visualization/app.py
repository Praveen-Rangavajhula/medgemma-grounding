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
from matplotlib import colormaps
import numpy as np

from medgemma_grounding.sae.cache import load_cache
from medgemma_grounding.sae.config import OUTPUT
from medgemma_grounding.utils.io import file_sha256
from medgemma_grounding.visualization.discovery import run_discovery_viewer
from medgemma_grounding.visualization.feature_effects import run_intervention_viewer
from medgemma_grounding.visualization.images import attention_case_images, read_viewer_experiment
from medgemma_grounding.visualization.overlays import (
    case_scores,
    color_scale,
    feature_catalog,
    feature_values,
    image_map,
    make_overlay,
    token_highlights,
)


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
        if (selected / "protocol.json").exists() and read_viewer_experiment(selected) == "E-010":
            from medgemma_grounding.visualization.attention_layers import viewer
            try:
                viewer(selected)
                attention_case_images(selected)
            except (ValueError, KeyError, OSError) as error:
                st.error(f"Could not open routing experiment: {error}")
            return
        if (selected / "protocol.json").exists() and read_viewer_experiment(selected) == "E-009":
            from medgemma_grounding.visualization.attention_windows import viewer
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
        st.info("Run: uv run python -m medgemma_grounding.experiments.compute_sae_activations")
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


if __name__ == "__main__":
    run_app()
