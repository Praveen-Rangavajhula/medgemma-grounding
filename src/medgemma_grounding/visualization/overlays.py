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

from PIL import Image
from matplotlib import colormaps
import numpy as np
import torch


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
