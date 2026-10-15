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
    st.caption("20-case workflow pilot · Feature descriptions require further validation on separate patients.")
    directory = Path(st.sidebar.text_input("Saved feature directory", str(OUTPUT))).expanduser()

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


if __name__ == "__main__":
    run_app()
