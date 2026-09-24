"""Load one labeled, localized case from the CheXlocalize release.

The official release keeps CheXpert images and labels beside CheXlocalize's
radiologist annotations.  This module intentionally exposes only a small
case-level interface; it does not iterate over a split or run a model.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and inspect one localized CheXlocalize finding."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--finding", default="Pleural Effusion")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    arguments = parser.parse_args()

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
