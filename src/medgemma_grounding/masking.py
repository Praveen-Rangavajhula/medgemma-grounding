"""Pixel-exact masks and summaries for the section 2.3 occlusion pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from pycocotools import mask as coco_mask


def load_effusion_mask(path: Path, case_id: str, shape: tuple[int, int]) -> np.ndarray:
    record = json.loads(path.read_text()).get(case_id, {})
    rle = record.get("Pleural Effusion")
    if not isinstance(rle, dict) or rle.get("size") != list(shape):
        raise ValueError("Missing effusion segmentation or image/mask dimension mismatch")
    if isinstance(rle.get("counts"), list):
        rle = coco_mask.frPyObjects(rle, *shape)
    result = coco_mask.decode(rle)
    if result.shape != shape or not np.isin(result, [0, 1]).all() or not result.any():
        raise ValueError("Expected a nonempty binary effusion mask at native resolution")
    return result.astype(bool)


def translate_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate the full shape, rejecting clipping and never wrapping pixels."""
    y, x = np.where(mask)
    if not len(y):
        raise ValueError("Cannot translate an empty mask")
    y, x = y + dy, x + dx
    if y.min() < 0 or x.min() < 0 or y.max() >= mask.shape[0] or x.max() >= mask.shape[1]:
        raise ValueError("Control translation would clip the mask")
    shifted = np.zeros_like(mask, dtype=bool)
    shifted[y, x] = True
    return shifted


def control_masks(target: np.ndarray, offsets: list[list[float]], chest_box: list[float]) -> list[dict]:
    """Use predeclared placements, independent of model outputs.

    The box is a manually chosen coarse chest window, not a lung segmentation.
    Shape/area matching does not make different anatomy interchangeable.
    """
    height, width = target.shape
    if (len(chest_box) != 4 or not 0 <= chest_box[0] < chest_box[2] <= 1
            or not 0 <= chest_box[1] < chest_box[3] <= 1):
        raise ValueError("Invalid normalized chest window")
    if len(offsets) != 3:
        raise ValueError("This pilot requires three predeclared control positions")
    x0, y0, x1, y1 = [round(v * s) for v, s in zip(chest_box, [width, height, width, height])]
    controls, seen = [], set()
    for offset in offsets:
        if len(offset) != 2 or not np.isfinite(offset).all():
            raise ValueError("Expected finite normalized x/y translations")
        dx, dy = round(offset[0] * width), round(offset[1] * height)
        if (dx, dy) in seen:
            raise ValueError("Duplicate control position")
        seen.add((dx, dy))
        shifted = translate_mask(target, dx, dy)
        y, x = np.where(shifted)
        if x.min() < x0 or x.max() >= x1 or y.min() < y0 or y.max() >= y1:
            raise ValueError("Control falls outside the predeclared chest window")
        if np.any(shifted & target):
            raise ValueError("Control overlaps the effusion annotation")
        controls.append({"mask": shifted, "dx": dx, "dy": dy})
    return controls


def replacement_images(image: Image.Image, model_size: tuple[int, int], blur_sigma: float) -> tuple[dict, dict]:
    if not np.isfinite(blur_sigma) or blur_sigma <= 0:
        raise ValueError("Blur sigma must be finite and positive")
    rgb = np.asarray(image.convert("RGB"))
    gray = int(np.rint(rgb.mean()))
    radius = (blur_sigma * image.width / model_size[0], blur_sigma * image.height / model_size[1])
    replacements = {"mean": np.full_like(rgb, gray),
                    "blur": np.asarray(image.convert("RGB").filter(ImageFilter.GaussianBlur(radius)))}
    return replacements, {"mean_gray_value": gray, "blur_sigma_model_pixels": blur_sigma,
                          "blur_radius_native_xy": list(radius),
                          "blur_method": "Pillow GaussianBlur approximation on original RGB; paste inside mask only"}


def apply_mask(image: np.ndarray, mask: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or replacement.shape != image.shape or replacement.dtype != np.uint8
            or mask.shape != image.shape[:2] or mask.dtype != bool):
        raise ValueError("Expected RGB uint8 image/replacement and same-size boolean mask")
    result = image.copy()
    result[mask] = replacement[mask]
    return result


def grid_patches(shape: tuple[int, int], grid: int) -> list[dict]:
    height, width = shape
    if not isinstance(grid, int) or not 1 <= grid <= min(height, width):
        raise ValueError("Grid must give nonempty patches")
    xs = np.rint(np.linspace(0, width, grid + 1)).astype(int)
    ys = np.rint(np.linspace(0, height, grid + 1)).astype(int)
    return [{"name": f"patch_{r:02d}_{c:02d}", "row": r, "column": c,
             "bbox_xyxy": [int(xs[c]), int(ys[r]), int(xs[c + 1]), int(ys[r + 1])]}
            for r in range(grid) for c in range(grid)]


def rectangle_mask(shape: tuple[int, int], bbox: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    if not 0 <= x0 < x1 <= shape[1] or not 0 <= y0 < y1 <= shape[0]:
        raise ValueError("Patch outside image")
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def summarize_regions(rows: list[dict], original: float) -> dict:
    result = {}
    for fill in ("mean", "blur"):
        target = next(r for r in rows if r["condition"] == "target" and r["fill"] == fill)
        controls = [r for r in rows if r["condition"].startswith("control_") and r["fill"] == fill]
        if len(controls) != 3:
            raise ValueError("Need all three controls for each fill")
        target_drop = original - target["yes_score"]
        drops = [original - r["yes_score"] for r in controls]
        result[fill] = {"target_drop": target_drop, "control_drops": drops,
                        "mean_control_drop": float(np.mean(drops)),
                        "target_minus_mean_control_drop": float(target_drop - np.mean(drops))}
    return result


def summarize_patches(rows: list[dict], target: np.ndarray, original: float, grid: int) -> tuple[dict, dict]:
    maps, summaries = {}, {}
    for fill in ("mean", "blur"):
        selected = [r for r in rows if r["fill"] == fill and r["kind"] == "patch"]
        if len(selected) != grid * grid:
            raise ValueError("Incomplete patch sweep")
        matrix = np.empty((grid, grid), dtype=float)
        positive_mass = inside_mass = 0.0
        for row in selected:
            x0, y0, x1, y1 = row["bbox_xyxy"]
            drop = original - row["yes_score"]
            matrix[row["row"], row["column"]] = drop
            positive_mass += max(drop, 0) * (x1 - x0) * (y1 - y0)
            inside_mass += max(drop, 0) * target[y0:y1, x0:x1].sum()
        fraction = float(inside_mass / positive_mass) if positive_mass > 0 else None
        summaries[fill] = {"positive_drop_mass_in_mask": fraction,
                           "mask_area_fraction": float(target.mean()),
                           "mass_enrichment_over_area": fraction / float(target.mean()) if fraction is not None else None,
                           "positive_patches": int((matrix > 0).sum()),
                           "negative_patches": int((matrix < 0).sum()),
                           "zero_patches": int((matrix == 0).sum()),
                           "minimum_signed_drop": float(matrix.min()), "maximum_signed_drop": float(matrix.max()),
                           "definition": "Positive score-drop density constant within each native patch; area-weighted mass inside official mask. Raw maps retain negative drops; no positive mass is null."}
        maps[fill] = matrix
    return summaries, maps
