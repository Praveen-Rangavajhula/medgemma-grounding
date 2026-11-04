"""Prepare the Part 2 protocol and patient partitions without loading a model."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

from PIL import Image


FINDING = "Pleural Effusion"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_part1_contract() -> dict:
    """Read literal settings and source fingerprints without importing plotting code.

    Later inference must call Part 1's functions and verify their fingerprints;
    this preparation step does not reimplement or execute the scorer/parser.
    """
    package = Path(__file__).parent
    settings = {}
    for filename, names in (
        ("classification.py", ("PROMPT", "MAX_NEW_TOKENS")),
        ("inference.py", ("DEFAULT_MODEL_ID",)),
    ):
        source = package / filename
        tree = ast.parse(source.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        settings[target.id] = ast.literal_eval(node.value)
        if any(name not in settings for name in names):
            raise ValueError(f"Required Part 1 settings missing from {filename}")
    return {
        "prompt": settings["PROMPT"],
        "max_new_tokens": settings["MAX_NEW_TOKENS"],
        "model_id": settings["DEFAULT_MODEL_ID"],
        "score_function": "medgemma_grounding.classification.compute_yes_score",
        "parse_function": "medgemma_grounding.classification.parse_prediction",
        "preprocessing_function": "medgemma_grounding.inference.prepare_inputs",
        "classification_source_sha256": sha256(package / "classification.py"),
        "inference_source_sha256": sha256(package / "inference.py"),
        "score_definition": "softmax([logit_No, logit_Yes])[Yes] at the first answer token",
        "score_interpretation": "Forced-choice preference, not a calibrated disease probability",
    }


def load_records(data_root: Path) -> tuple[list[dict], list[dict]]:
    """Keep all frontal binary-label cases, including negatives with other findings."""
    labels_path = data_root / "CheXpert" / "val_labels.csv"
    annotations = json.loads(
        (data_root / "CheXlocalize" / "gt_annotations_val.json").read_text()
    )
    records, excluded = [], []
    seen = set()
    with labels_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Path", "Frontal/Lateral", FINDING}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Labels CSV is missing required path, view, or target columns")
        for row in reader:
            original_path = row["Path"]
            if row["Frontal/Lateral"] != "Frontal":
                excluded.append({"label_path": original_path, "reason": "non_frontal"})
                continue
            try:
                label = float(row[FINDING])
            except (TypeError, ValueError):
                label = math.nan
            if label not in (0.0, 1.0):
                excluded.append({"label_path": original_path, "reason": "missing_or_uncertain_label"})
                continue
            parts = PurePosixPath(original_path).parts
            if (len(parts) != 5 or parts[0] not in {"CheXpert-v1.0", "CheXpert"}
                    or parts[1] not in {"valid", "val"} or ".." in parts):
                raise ValueError(f"Unexpected validation image path: {original_path}")
            patient, study, filename = parts[-3:]
            if not PurePosixPath(filename).stem.endswith("_frontal"):
                raise ValueError(f"Frontal metadata/filename mismatch: {original_path}")
            case_id = "_".join((patient, study, PurePosixPath(filename).stem))
            if case_id in seen:
                raise ValueError(f"Duplicate case ID: {case_id}")
            seen.add(case_id)
            relative = Path("CheXpert", "val", patient, study, filename)
            path = data_root / relative
            with Image.open(path) as image:
                image.load()
                width, height = image.size
            annotation = annotations.get(case_id)
            contours = annotation.get(FINDING, []) if annotation else []
            if annotation and annotation.get("img_size") != [height, width]:
                raise ValueError(f"Annotation/image dimensions disagree: {case_id}")
            for contour in contours:
                if len(contour) < 3 or any(
                    len(point) != 2 or not all(math.isfinite(float(v)) for v in point)
                    for point in contour
                ):
                    raise ValueError(f"Invalid target contour: {case_id}")
            if contours and label == 0:
                raise ValueError(f"Positive contour disagrees with negative expert label: {case_id}")
            records.append({
                "case_id": case_id,
                "patient_id": patient,
                "image_path": relative.as_posix(),
                "image_sha256": sha256(path),
                "image_size_height_width": [height, width],
                "label": int(label),
                "has_effusion_contour": bool(contours),
                "localization_eligible": label == 1 and bool(contours),
                "localization_exclusion": (
                    None if label == 1 and contours else
                    "negative_target_label" if label == 0 else "missing_positive_contour"
                ),
                "other_positive_findings": sorted(
                    key for key, value in row.items()
                    if key not in {"Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", FINDING, "No Finding"}
                    and value in {"1", "1.0"}
                ),
                "grounding_status": "unassessed",
            })
    if not records:
        raise ValueError("No eligible frontal binary-label validation images")
    return sorted(records, key=lambda row: row["case_id"]), excluded


def assign_patients(
    records: list[dict], *, seed: int, discovery_fraction: float,
    discovery_patients: tuple[str, ...] = (),
) -> dict[str, str]:
    """Stratify by any-positive patient; keep every view of a patient together."""
    if not 0 < discovery_fraction < 1:
        raise ValueError("Discovery fraction must be between 0 and 1")
    patients = defaultdict(list)
    for row in records:
        patients[row["patient_id"]].append(row["label"])
    pinned = set(discovery_patients)
    if pinned - patients.keys():
        raise ValueError("Pinned discovery patient is absent from the eligible cohort")
    assignments = {}
    for positive in (False, True):
        group = [p for p, labels in patients.items() if bool(any(labels)) == positive]
        if len(group) < 2:
            raise ValueError("Need at least two patients in each target-label stratum")
        count = max(1, min(len(group) - 1, int(len(group) * discovery_fraction + 0.5)))
        if len(pinned.intersection(group)) > count:
            raise ValueError("Pinned patients exceed the discovery allocation")
        group.sort(key=lambda p: (
            p not in pinned, hashlib.sha256(f"{seed}:{int(positive)}:{p}".encode()).hexdigest()
        ))
        for index, patient in enumerate(group):
            assignments[patient] = "discovery" if index < count else "confirmation"
    return assignments


def prepare_protocol(
    data_root: Path, *, seed: int = 42, discovery_fraction: float = 0.5,
    discovery_patients: tuple[str, ...] = (),
) -> dict[str, object]:
    data_root = data_root.resolve()
    records, excluded = load_records(data_root)
    assignments = assign_patients(
        records, seed=seed, discovery_fraction=discovery_fraction,
        discovery_patients=discovery_patients,
    )
    for row in records:
        row["partition"] = assignments[row["patient_id"]]
    summary = {"images": len(records), "patients": len(assignments),
               "excluded": dict(Counter(row["reason"] for row in excluded)), "partitions": {}}
    for partition in ("discovery", "confirmation"):
        subset = [row for row in records if row["partition"] == partition]
        summary["partitions"][partition] = {
            "images": len(subset), "patients": len({r["patient_id"] for r in subset}),
            "positive": sum(r["label"] == 1 for r in subset),
            "negative": sum(r["label"] == 0 for r in subset),
            "localization_eligible": sum(r["localization_eligible"] for r in subset),
        }
    git = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "classification.py", "inference.py"],
        cwd=Path(__file__).parent, capture_output=True, text=True,
    )
    metadata_files = [data_root / "CheXpert/val_labels.csv", data_root / "CheXlocalize/gt_annotations_val.json"]
    mask_path = data_root / "CheXlocalize/gt_segmentations_val.json"
    if mask_path.exists():
        metadata_files.append(mask_path)
    protocol = {
        "schema_version": 1, "section": "2.1", "finding": FINDING, "dataset_split": "val",
        "data_root": str(data_root), "part1_code_commit": git.stdout.strip() if git.returncode == 0 else None,
        "preparation_source_sha256": sha256(Path(__file__)),
        "metadata_sha256": {str(p.relative_to(data_root)): sha256(p) for p in metadata_files},
        "part1": read_part1_contract(),
        "patient_partition": {
            "seed": seed, "discovery_fraction": discovery_fraction,
            "method": "SHA256 ranking within any-positive vs negative-only patient strata",
            "forced_discovery_patients": sorted(set(discovery_patients)),
            "confirmation_role": "Held out from perturbation and feature selection; Part 1 already evaluated all validation images",
        },
        "output_categories": {
            "classification": ["true_positive", "false_negative", "true_negative", "false_positive", "unparsed"],
            "parsing": "Reuse Part 1 parse_prediction; retain unparsed cases for continuous-score analysis",
            "grounding": "Unassessed at baseline; classification correctness alone is not grounding evidence",
        },
        "localization_population": "All annotated positives, including baseline false negatives and unparsed responses",
        "negative_cases": "Retain other abnormalities; use as target-negative controls, not empty-mask localization examples",
        "planned_conditions": ["original", "effusion_region_occluded", "matched_control_region_occluded"],
        "runtime_to_verify_in_section_2_2": ["model_revision", "processor_revision", "device", "dtype", "candidate_token_ids", "baseline_score_agreement"],
        "test_set_policy": "Reserved for final evaluation; not read by this command",
    }
    return {"protocol.json": protocol, "cohort_manifest.json": {"cases": records, "excluded": excluded},
            "cohort_summary.json": summary}


def save_protocol(output: Path, documents: dict[str, object]) -> None:
    """Permit identical reruns; never replace an existing different experiment."""
    serialized = {name: json.dumps(value, indent=2, sort_keys=True) + "\n" for name, value in documents.items()}
    if output.exists():
        if all((output / name).is_file() and (output / name).read_text() == text
               for name, text in serialized.items()):
            return
        raise ValueError("Output already contains different files/settings; choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".protocol-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        for name, text in serialized.items():
            (stage / name).write_text(text)
        stage.rename(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/part2/protocol-v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discovery-fraction", type=float, default=0.5)
    parser.add_argument("--discovery-patient", action="append", default=[], help="Keep a previously inspected patient in discovery")
    args = parser.parse_args(argv)
    try:
        documents = prepare_protocol(args.data_root, seed=args.seed, discovery_fraction=args.discovery_fraction,
                                     discovery_patients=tuple(args.discovery_patient))
        save_protocol(args.output, documents)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Protocol preparation failed: {error}\n")
    print(json.dumps(documents["cohort_summary.json"], indent=2))
    print(f"Protocol and patient assignments saved to {args.output}")


if __name__ == "__main__":
    main()
