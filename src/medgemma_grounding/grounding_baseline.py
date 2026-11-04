"""Reconcile saved Part 1 predictions with the frozen Part 2 cohort (section 2.2)."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from medgemma_grounding.grounding_protocol import prepare_protocol, read_part1_contract, sha256


def load_protocol(directory: Path) -> tuple[dict, list[dict]]:
    """Recheck source, dataset, image hashes, labels, and patient assignments."""
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["part1"] != read_part1_contract():
        raise ValueError("Part 1 source/settings differ from the frozen protocol")
    split = protocol["patient_partition"]
    current = prepare_protocol(
        Path(protocol["data_root"]), seed=split["seed"],
        discovery_fraction=split["discovery_fraction"],
        discovery_patients=tuple(split["forced_discovery_patients"]),
    )
    if current["protocol.json"]["metadata_sha256"] != protocol["metadata_sha256"]:
        raise ValueError("Dataset metadata differ from the frozen protocol")
    manifest = json.loads((directory / "cohort_manifest.json").read_text())
    if manifest != current["cohort_manifest.json"]:
        raise ValueError("Cohort, patient assignments, or images differ from the frozen manifest")
    return protocol, manifest["cases"]


def read_predictions(source: Path) -> tuple[list[dict], dict]:
    provenance = {"path": str(source.resolve()), "sha256": sha256(source)}
    if zipfile.is_zipfile(source):
        member = "classification/predictions.csv"
        with zipfile.ZipFile(source) as archive:
            if archive.namelist().count(member) != 1:
                raise ValueError("Archive must contain exactly one classification/predictions.csv")
            payload = archive.read(member)
            if "classification/metrics.txt" in archive.namelist():
                report = archive.read("classification/metrics.txt").decode("utf-8-sig")
                for name in ("Prompt", "Model"):
                    values = [line[len(name) + 2:] for line in report.splitlines()
                              if line.startswith(name + ": ")]
                    if len(values) == 1:
                        provenance["reported_" + name.lower()] = values[0]
        provenance["archive_member"] = member
    else:
        payload = source.read_bytes()
    provenance["predictions_csv_sha256"] = hashlib.sha256(payload).hexdigest()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {"image_path", "ground_truth", "response", "prediction", "yes_score"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Predictions CSV is missing required Part 1 columns")
    return list(reader), provenance


def case_id_from_path(value: str) -> str:
    parts = PurePosixPath(value.replace("\\", "/")).parts
    if (len(parts) < 3 or ".." in parts or not parts[-3].startswith("patient")
            or not parts[-2].startswith("study") or not PurePosixPath(parts[-1]).stem.endswith("_frontal")):
        raise ValueError(f"Unexpected frontal prediction path: {value}")
    return "_".join((*parts[-3:-1], PurePosixPath(parts[-1]).stem))


def binary_value(value: str, *, allow_missing: bool = False) -> int | None:
    if allow_missing and value.strip().lower() in {"", "nan", "none"}:
        return None
    number = float(value)
    if number not in (0.0, 1.0):
        raise ValueError("Expected a binary label/prediction")
    return int(number)


def reconcile(rows: list[dict], cases: list[dict]) -> list[dict]:
    from medgemma_grounding.classification import parse_prediction

    expected = {case["case_id"]: case for case in cases}
    if len(expected) != len(cases):
        raise ValueError("Duplicate case in cohort manifest")
    joined = {}
    for row in rows:
        case_id = case_id_from_path(row["image_path"])
        if case_id in joined:
            raise ValueError(f"Duplicate prediction: {case_id}")
        if case_id not in expected:
            raise ValueError(f"Prediction outside the frozen cohort: {case_id}")
        case = expected[case_id]
        label = binary_value(row["ground_truth"])
        prediction = binary_value(row["prediction"], allow_missing=True)
        if label != case["label"]:
            raise ValueError(f"Expert label mismatch: {case_id}")
        if parse_prediction(row["response"]) != prediction:
            raise ValueError(f"Stored prediction disagrees with Part 1 parser: {case_id}")
        score = float(row["yes_score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid continuous score: {case_id}")
        category = ("unparsed" if prediction is None else
                    {(1, 1): "true_positive", (1, 0): "false_negative",
                     (0, 0): "true_negative", (0, 1): "false_positive"}[label, prediction])
        joined[case_id] = dict(case, response=row["response"], prediction=prediction,
                              yes_score=score, ground_truth=label, classification_category=category)
    if expected.keys() != joined.keys():
        raise ValueError(f"Missing predictions for {len(expected.keys() - joined.keys())} cohort cases")
    return [joined[key] for key in sorted(joined)]


def metrics_for(rows: list[dict]) -> dict:
    import pandas as pd
    from medgemma_grounding.classification import compute_metrics

    if not rows:
        return {"total_images": 0, "status": "empty_partition"}
    frame = pd.DataFrame(rows)
    if frame["prediction"].notna().any():
        metrics = compute_metrics(frame)
    else:
        # The inherited function raises when nothing is parsed. Continuous scores
        # remain valid; retain that population without inventing binary decisions.
        from sklearn.metrics import average_precision_score, roc_auc_score
        both = frame["ground_truth"].nunique() == 2
        metrics = {"total_images": len(rows), "parsed_images": 0, "unparsed_images": len(rows),
                   "binary_metrics": None,
                   "roc_auc": roc_auc_score(frame.ground_truth, frame.yes_score) if both else None,
                   "average_precision": average_precision_score(frame.ground_truth, frame.yes_score) if both else None}
    metrics = {key: None if isinstance(value, float) and not math.isfinite(value) else value
               for key, value in metrics.items()}
    metrics["categories"] = dict(sorted(Counter(row["classification_category"] for row in rows).items()))
    metrics["unparsed_positive"] = sum(r["prediction"] is None and r["label"] == 1 for r in rows)
    metrics["unparsed_negative"] = sum(r["prediction"] is None and r["label"] == 0 for r in rows)
    metrics["localization_eligible"] = sum(r["localization_eligible"] for r in rows)
    return metrics


def curate_cases(rows: list[dict], seed: int, per_category: int = 2) -> dict:
    selected = []
    for category in ("true_positive", "false_negative", "true_negative", "false_positive", "unparsed"):
        candidates = [r for r in rows if r["partition"] == "discovery" and r["classification_category"] == category]
        candidates.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['case_id']}".encode()).hexdigest())
        selected.extend(candidates[:per_category])
    return {"selection": "Up to two per parsed-answer category, discovery only, SHA256(seed:case_id) order",
            "seed": seed, "purpose": "Qualitative pilot examples; category sampling is not a population estimate",
            "localization_population": "All annotated positives remain eligible, regardless of these selections",
            "cases": selected}


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def save_artifacts(output: Path, files: dict[str, str]) -> None:
    if output.exists():
        if all((output / name).is_file() and (output / name).read_text() == content
               for name, content in files.items()):
            return
        raise ValueError("Output already exists with different content; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        for name, content in files.items():
            (stage / name).write_text(content)
        stage.rename(output)


def audit(protocol_dir: Path, source: Path, output: Path) -> dict:
    protocol, cases = load_protocol(protocol_dir)
    raw, provenance = read_predictions(source)
    if provenance.get("reported_prompt", protocol["part1"]["prompt"]) != protocol["part1"]["prompt"]:
        raise ValueError("Archived report prompt differs from Part 1")
    rows = reconcile(raw, cases)
    summary = {
        "section": "2.2", "status": "saved_predictions_verified; real_model_pilot_separate",
        "source": provenance, "part1_contract": protocol["part1"],
        "protocol_sha256": sha256(protocol_dir / "protocol.json"),
        "manifest_sha256": sha256(protocol_dir / "cohort_manifest.json"),
        "audit_source_sha256": sha256(Path(__file__)),
        "baseline_cases_sha256": hashlib.sha256(json_text(rows).encode()).hexdigest(),
        "historical_runtime": {"model_revision": None, "processor_revision": None, "device": None, "dtype": None},
        "denominators": "ROC-AUC and average precision use all scores. Binary metrics use only parsed responses. AP is not trapezoidal PR-AUC.",
        "grounding_status": "unassessed", "metrics": {},
    }
    for partition in ("all", "discovery", "confirmation"):
        summary["metrics"][partition] = metrics_for(rows if partition == "all" else [r for r in rows if r["partition"] == partition])
    table = io.StringIO(newline="")
    columns = ["case_id", "patient_id", "partition", "image_path", "ground_truth", "response", "prediction",
               "yes_score", "classification_category", "localization_eligible", "grounding_status"]
    writer = csv.DictWriter(table, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    save_artifacts(output, {"summary.json": json_text(summary), "baseline_cases.json": json_text(rows),
                            "baseline_table.csv": table.getvalue(),
                            "curated_cases.json": json_text(curate_cases(rows, protocol["patient_partition"]["seed"]))})
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("outputs/part2/protocol-v1"))
    parser.add_argument("--predictions", type=Path, required=True, help="Part 1 CSV or result.zip")
    parser.add_argument("--output", type=Path, default=Path("outputs/part2/baseline-v1"))
    args = parser.parse_args(argv)
    try:
        summary = audit(args.protocol, args.predictions, args.output)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Baseline audit failed: {error}\n")
    print(json_text(summary["metrics"]["all"]))
    print(f"Baseline table and discovery examples saved to {args.output}")


if __name__ == "__main__":
    main()
