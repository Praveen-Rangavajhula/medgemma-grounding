"""Load one labeled, localized case from the CheXlocalize release.

The official release keeps CheXpert images and labels beside CheXlocalize's
radiologist annotations.  This module intentionally exposes only a small
case-level interface; it does not iterate over a split or run a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CheXlocalizeDataError(RuntimeError):
    """Raised when an expected CheXlocalize file or record is unavailable."""


@dataclass(frozen=True)
class CheXlocalizeCase:
    """One image, its expert label, and its raw expert-drawn contours."""

    case_id: str
    finding: str
    image_path: Path
    label: float
    image_size: tuple[int, int]
    contours: tuple[tuple[tuple[float, float], ...], ...]


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    options = "\n  ".join(str(path) for path in paths)
    raise CheXlocalizeDataError(f"Could not find any of:\n  {options}")


def _split_directory_name(split: str) -> str:
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    return "valid" if split == "val" else "test"


def _metadata_paths(data_root: Path, split: str) -> tuple[Path, Path]:
    """Return label and annotation paths for the official release layout."""

    label_path = _first_existing(
        data_root / "CheXpert" / f"{split}_labels.csv",
        data_root / f"{split}_labels.csv",
    )
    annotation_path = _first_existing(
        data_root / "CheXlocalize" / f"gt_annotations_{split}.json",
        data_root / f"gt_annotations_{split}.json",
    )
    return label_path, annotation_path


def _case_id(label_path: str) -> str:
    """Match CheXpert's image path to CheXlocalize's annotation key."""

    return "_".join(Path(label_path).with_suffix("").parts[-3:])


def _image_path(data_root: Path, split: str, label_path: str) -> Path:
    """Resolve both the release layout and CheXpert's CSV-relative paths."""

    relative = Path(label_path)
    split_directory = _split_directory_name(split)
    try:
        image_tail = relative.parts[relative.parts.index(split_directory) + 1 :]
    except ValueError as error:
        raise CheXlocalizeDataError(
            f"Label path does not contain the expected '{split_directory}' directory: {label_path}"
        ) from error

    return _first_existing(
        data_root / relative,
        data_root / "CheXpert" / split / Path(*image_tail),
        data_root / "CheXpert" / split_directory / Path(*image_tail),
    )


def _label_rows(label_path: Path) -> list[dict[str, str]]:
    with label_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def find_positive_case(
    data_root: str | Path,
    finding: str = "Pleural Effusion",
    *,
    split: str = "val",
) -> CheXlocalizeCase:
    """Return the first positive case with an expert contour for ``finding``.

    CheXlocalize's raw annotation JSON contains only positive expert findings,
    so requiring both a 1.0 label and a contour prevents accidental mismatches.
    """

    root = Path(data_root)
    label_path, annotation_path = _metadata_paths(root, split)
    with annotation_path.open() as handle:
        annotations: dict[str, dict[str, Any]] = json.load(handle)

    for row in _label_rows(label_path):
        if finding not in row:
            raise CheXlocalizeDataError(
                f"Finding '{finding}' is not a column in {label_path}"
            )
        if row[finding] != "1.0":
            continue

        label_image_path = row["Path"]
        case_id = _case_id(label_image_path)
        annotation = annotations.get(case_id, {})
        if finding not in annotation:
            continue

        contours = tuple(
            tuple((float(x), float(y)) for x, y in contour)
            for contour in annotation[finding]
        )
        return CheXlocalizeCase(
            case_id=case_id,
            finding=finding,
            image_path=_image_path(root, split, label_image_path),
            label=float(row[finding]),
            image_size=tuple(annotation["img_size"]),
            contours=contours,
        )

    raise CheXlocalizeDataError(
        f"No positive {finding!r} case with a raw expert annotation was found in {split}."
    )


def inspect_case(case: CheXlocalizeCase) -> tuple[int, int]:
    """Open the image and return its (width, height), proving it is readable."""

    from PIL import Image

    with Image.open(case.image_path) as image:
        image.load()
        return image.size


def expert_mask(case, annotations, segmentations):
    """Decode official RLE and independently check Pillow polygon rasterization.

    Matches cheXlocalize annotation_to_segmentation.py: ImageDraw.polygon on
    an original-resolution 1-bit image. Missing records never become negative labels.
    Returns (status, binary mask or None, detail).
    """
    import numpy as np
    from PIL import Image, ImageDraw
    from pycocotools import mask as coco_mask

    finding = "Pleural Effusion"
    raw = annotations.get(case["case_id"])
    released = segmentations.get(case["case_id"])
    if raw is None and released is None:
        return "missing", None, "No annotation record"
    try:
        size = tuple(case["image_size"])
        if raw is not None and tuple(raw["img_size"]) != size:
            raise ValueError("Annotation dimensions differ from image")
        polygons = raw.get(finding, []) if raw is not None else []
        canvas = Image.new("1", (size[1], size[0]), 0)
        draw = ImageDraw.Draw(canvas)
        for polygon in polygons:
            draw.polygon([tuple(p) for p in polygon], outline=1, fill=1)
        expected = np.asarray(canvas, dtype=bool)
        rle = released.get(finding) if released is not None else None
        if rle is None or raw is None:
            raise ValueError("Raw record or released finding mask missing from annotated case")
        decoded = coco_mask.decode(rle).astype(bool)
        if decoded.shape != size:
            raise ValueError(f"Released mask shape {decoded.shape} differs from image {size}")
        if not np.array_equal(decoded, expected):
            raise ValueError(f"Released mask and original-coordinate polygons disagree at {np.count_nonzero(decoded != expected)} pixels")
        return "present nonempty" if decoded.any() else "present empty", decoded, "RLE/polygon agreement"
    except (ValueError, KeyError, TypeError, IndexError, RuntimeError, OverflowError) as error:
        return "invalid/mismatched", None, str(error)


def validation_cohort(data_root, pilot_patients, *, expected=(202, 200, 64), targets=(45, 95)):
    """Audit validation only and split patients with the preregistered hash order.

    Every eligible image is included independently of annotation availability.
    Confirmation images are read only for eligibility/hash auditing, never inference.
    """
    from PIL import Image

    root = Path(data_root)
    labels, annotation_path = _metadata_paths(root, "val")
    annotations = json.loads(annotation_path.read_text())
    cases, exclusions = [], []
    for row in _label_rows(labels):
        reasons = []
        if row["Frontal/Lateral"] != "Frontal":
            reasons.append("not frontal")
        if row["Pleural Effusion"] not in {"0", "1", "0.0", "1.0"}:
            reasons.append("not explicit binary label")
        if reasons:
            exclusions.append({"csv_path": row["Path"], "reasons": reasons})
            continue
        path = _image_path(root, "val", row["Path"])
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        case_id = _case_id(row["Path"])
        annotation = annotations.get(case_id)
        cases.append({
            "case_id": case_id, "image_path": str(path.resolve()),
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "patient": Path(row["Path"]).parts[-3], "label": int(float(row["Pleural Effusion"])),
            "image_size": [height, width], "csv_record": row,
            "annotation_status": "missing" if annotation is None else
                "present nonempty" if annotation.get("Pleural Effusion") else "present empty",
        })
    patients = sorted({c["patient"] for c in cases})
    if (len(cases), len(patients), sum(c["label"] for c in cases)) != expected:
        raise ValueError("Unexpected validation eligibility counts; revise protocol explicitly")
    if not set(pilot_patients) <= set(patients):
        raise ValueError("Pilot patients are not all eligible")
    positive = {p for p in patients if any(c["label"] for c in cases if c["patient"] == p)}
    discovery = set(pilot_patients)
    for stratum, count in ((positive, targets[0]), (set(patients) - positive, targets[1])):
        forced = discovery & stratum
        if len(forced) > count or len(stratum) < count:
            raise ValueError("Cannot satisfy fixed stratified split")
        remaining = sorted(stratum - forced, key=lambda p: hashlib.sha256(
            f"medgemma-effusion-v1|seed=42|{p}".encode()).hexdigest())
        discovery.update(remaining[:count - len(forced)])
    hashes = {}
    for case in cases:
        case["partition"] = "discovery" if case["patient"] in discovery else "confirmation"
        previous = hashes.setdefault(case["image_sha256"], case["partition"])
        if previous != case["partition"]:
            raise ValueError("Duplicate image hash crosses partitions")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case ID")
    segmentation_path = root / "CheXlocalize" / "gt_segmentations_val.json"
    return {"schema_version": 2, "data_root": str(root.resolve()), "seed": 42,
            "split_rule": "medgemma-effusion-v1|seed=42|<patient_id>",
            "pilot_patients": sorted(pilot_patients), "cases": cases, "exclusions": exclusions,
            "metadata_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (labels, annotation_path, segmentation_path)},
            "prior_exposure": "20 pilot patients inspected; aggregate validation previously viewed; pretraining overlap unknown"}


def validate_partition(cohort, pilot_patients, *, expected=(202, 200, 64), targets=(45, 95)):
    """Recompute the frozen membership rule; do not trust a partition label alone."""
    cases = cohort["cases"]
    patients = {c["patient"] for c in cases}
    if ((len(cases), len(patients), sum(c["label"] for c in cases)) != expected
            or len({c["case_id"] for c in cases}) != len(cases)
            or any(c["label"] not in (0, 1) or '/test/' in c['image_path'] for c in cases)
            or not set(pilot_patients) <= patients
            or cohort["pilot_patients"] != sorted(pilot_patients)):
        raise ValueError("Invalid discovery cohort identity/counts")
    positive = {c["patient"] for c in cases if c["label"] == 1}
    chosen = set(pilot_patients)
    for stratum, target in ((positive, targets[0]), (patients - positive, targets[1])):
        forced = chosen & stratum
        if len(forced) > target:
            raise ValueError("Pilot exceeds split target")
        remaining = sorted(stratum - forced, key=lambda p: hashlib.sha256(
            f"medgemma-effusion-v1|seed=42|{p}".encode()).hexdigest())
        chosen.update(remaining[:target - len(forced)])
    seen = {}
    for c in cases:
        required = "discovery" if c["patient"] in chosen else "confirmation"
        if c["partition"] != required:
            raise ValueError("Partition differs from frozen patient split")
        if seen.setdefault(c["image_sha256"], required) != required:
            raise ValueError("Duplicate image crosses partitions")
    if len(chosen) != sum(targets):
        raise ValueError("Unexpected discovery patient count")
    return [c for c in cases if c["partition"] == "discovery"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and inspect one localized CheXlocalize finding."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--finding", default="Pleural Effusion")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--cohort-output", type=Path, help="Audit and partition validation; no model inference")
    parser.add_argument("--pilot-json", type=Path, default=Path("results/gemma_scope/medgemma_pilot.json"))
    arguments = parser.parse_args()

    if arguments.cohort_output:
        if arguments.split != "val":
            parser.error("Discovery cohort supports validation only")
        pilot = json.loads(arguments.pilot_json.read_text())
        cohort = validation_cohort(arguments.data_root, {c["patient"] for c in pilot["cases"]})
        from medgemma_grounding.medgemma_features import atomic_json
        if arguments.cohort_output.exists():
            if json.loads(arguments.cohort_output.read_text()) != cohort:
                raise ValueError("Existing cohort differs; do not overwrite experiment identity")
        else:
            atomic_json(arguments.cohort_output, cohort)
        print(f"Audited {len(cohort['cases'])} validation images: {arguments.cohort_output}")
        return

    case = find_positive_case(arguments.data_root, arguments.finding, split=arguments.split)
    loaded_size = inspect_case(case)
    print(f"case_id: {case.case_id}")
    print(f"image_path: {case.image_path}")
    print(f"label ({case.finding}): {case.label}")
    print(f"annotation image_size (height, width): {case.image_size}")
    print(f"image loaded_size (width, height): {loaded_size}")
    print(f"contours: {len(case.contours)}")
    print(f"first contour points: {len(case.contours[0])}")
    print(f"first contour preview: {case.contours[0][:3]}")


if __name__ == "__main__":
    main()
