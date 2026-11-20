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

from medgemma_grounding.utils.io import file_sha256
from medgemma_grounding.visualization.overlays import mask_display


def read_viewer_experiment(directory):
    import json
    return json.loads((directory / "protocol.json").read_text()).get("experiment")


def attention_case_images(directory):
    """Original discovery X-rays, never a spatial overlay inferred from depth."""
    import streamlit as st
    from medgemma_grounding.utils.io import read_json
    from medgemma_grounding.data.chexlocalize import expert_mask
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
