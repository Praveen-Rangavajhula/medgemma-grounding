"""Inspect validation cases or build the fixed patient-disjoint discovery cohort."""


from __future__ import annotations

from pathlib import Path
import argparse
import json

from medgemma_grounding.data.chexlocalize import find_positive_case, inspect_case, validation_cohort


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
        from medgemma_grounding.utils.io import atomic_json
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
