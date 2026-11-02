"""Discovery-only patient-level SAE screening; no confirmation inference/tests.

Image means follow multimodal-saes/probing_approximation/compute_sae_activations.py;
the four-track individual-feature selection is our recorded screening protocol,
not a reproduction of their sparse trained probe.
"""

from __future__ import annotations

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from medgemma_grounding.medgemma_features import (
    atomic_json, file_sha256, identity_hash, load_discovery_case,
)

TRACKS = (("image_mean", "label"), ("final", "label"),
          ("image_mean", "answer_score"), ("final", "answer_score"))


def aggregate_patients(records, summaries):
    """One patient contributes one equally weighted observation per track."""
    patients = sorted({r["patient"] for r in records})
    metadata, arrays = [], {name: [] for name in ("image_mean", "final")}
    for patient in patients:
        indices = [i for i, r in enumerate(records) if r["patient"] == patient]
        labels = {records[i]["label"] for i in indices}
        metadata.append({"patient": patient, "label": max(labels), "mixed_label": len(labels) > 1,
                         "images": len(indices), "case_ids": [records[i]["case_id"] for i in indices],
                         "answer_score": float(np.mean([records[i]["answer_score"]["score"] for i in indices]))})
        for name in arrays:
            arrays[name].append(np.asarray(summaries[name])[indices].mean(0))
    return metadata, {k: np.stack(v) for k, v in arrays.items()}


def feature_rankings(patients, summaries, min_active=5, representations=("image_mean", "final")):
    """ROC-AUC via tie-aware ranks; Spearman via centered rank correlation.

    No significance tests are performed on features selected in this same data.
    """
    labels = np.array([p["label"] for p in patients])
    scores = np.array([p["answer_score"] for p in patients])
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    if not positives or not negatives:
        raise ValueError("Both expert labels required for discovery rankings")
    target = rankdata(scores).astype(float)
    target -= target.mean()
    rows = []
    for representation in representations:
        x = np.asarray(summaries[representation], dtype=float)
        if not np.isfinite(x).all() or len(x) != len(labels):
            raise ValueError("Invalid patient-feature summaries")
        ranks = rankdata(x, axis=0)
        auc = (ranks[labels == 1].sum(0) - positives * (positives + 1) / 2) / (positives * negatives)
        centered = ranks - ranks.mean(0)
        denominator = np.linalg.norm(centered, axis=0) * np.linalg.norm(target)
        rho = np.divide(target @ centered, denominator, out=np.zeros(x.shape[1]), where=denominator > 0)
        active = (x > 0).sum(0)
        constant = np.ptp(x, axis=0) == 0
        for feature in range(x.shape[1]):
            common = {"feature": feature, "representation": representation,
                      "active_patients": int(active[feature]), "patient_count": len(x),
                      "prevalence": float(active[feature] / len(x)),
                      "constant": bool(constant[feature]),
                      "label_auc": float(auc[feature]),
                      "answer_spearman": float(rho[feature]) if denominator[feature] > 0 else None,
                      "mean": float(x[:, feature].mean()), "maximum": float(x[:, feature].max())}
            eligible = not constant[feature] and active[feature] >= min_active
            for target_name in ("label", "answer_score"):
                effect = auc[feature] - .5 if target_name == "label" else rho[feature]
                direction = "positive" if effect > 0 else "negative" if effect < 0 else "zero"
                if target_name == "answer_score" and denominator[feature] == 0:
                    direction = "undefined"
                rows.append({**common, "target": target_name,
                             "eligible": bool(eligible and (target_name == "label" or np.linalg.norm(target) > 0)),
                             "strength": float(abs(effect)), "direction": direction})
    return rows


def select_candidates(rows):
    selected, seen = [], set()
    for representation, target in TRACKS:
        ranked = sorted((r for r in rows if r["representation"] == representation
                         and r["target"] == target and r["eligible"] and r["strength"] > 0),
                        key=lambda r: (-r["strength"], r["feature"]))
        count = 0
        for row in ranked:
            if row["feature"] in seen:
                continue
            if target == "label":
                association = "expert effusion presence" if row["direction"] == "positive" else "expert effusion absence"
            else:
                association = "higher Yes-minus-No scores" if row["direction"] == "positive" else "lower Yes-minus-No scores"
            selected.append({**row, "selection_track": f"{representation}:{target}",
                             "hypothesis": f"Higher {representation} activity associates with {association}. Feature meaning and causal role are unknown.",
                             "hypothesis_source": "automatically generated association hypothesis, not semantic interpretation",
                             "semantic_interpretation": None,
                             "secondary_measurements": [r for r in rows if r["feature"] == row["feature"]]})
            seen.add(row["feature"])
            count += 1
            if count == 2:
                break
    return selected


def pooled_fidelity(records, sufficient):
    results = {}
    for group in records[0]["fidelity"]:
        counts = [r["fidelity"][group] for r in records]
        tokens = sum(r["tokens"] for r in counts)
        sx = sum(s[group]["sum_x"].double() for s in sufficient)
        sx2 = sum(s[group]["sum_x2"].double() for s in sufficient)
        sse = sum(r["sse"] for r in counts)
        energy = sum(r["energy"] for r in counts)
        sst = (sx2 - sx.square() / tokens).sum().item()
        results[group] = {"tokens": tokens, "sse": sse, "energy": energy, "centered_sst": sst,
                          "fvu": sse / sst if sst > 0 else None,
                          "relative_squared_error": sse / energy if energy else None,
                          "observed_l0": sum(r["active_features"] for r in counts) / tokens}
        if "behavior" in counts[0]:
            behaviors = [r["behavior"] for r in counts]
            results[group]["behavior"] = {
                "median_kl": float(np.median([b["kl_from_baseline"] for b in behaviors])),
                "mean_signed_score_change": float(np.mean([b["score_change"] for b in behaviors])),
                "median_absolute_score_change": float(np.median([b["absolute_score_change"] for b in behaviors])),
                "decision_flips": sum(b["decision_flip"] for b in behaviors),
                "distortion_exceeds_original_margin": sum(b["absolute_score_change"] > b["original_margin"] for b in behaviors)}
    return results


def audit_masks(directory, cohort, cases):
    """Discovery-only released/raw mask checks and non-square alignment figures."""
    from PIL import Image
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from medgemma_grounding.chexlocalize import expert_mask
    from medgemma_grounding.feature_viewer import mask_display

    directory = Path(directory)
    root = Path(cohort["data_root"]) / "CheXlocalize"
    annotations = json.loads((root / "gt_annotations_val.json").read_text())
    segmentations = json.loads((root / "gt_segmentations_val.json").read_text())
    audit, examples = [], []
    for case in cases:
        if case["partition"] != "discovery":
            raise ValueError("Mask audit is discovery-only")
        status, mask, detail = expert_mask(case, annotations, segmentations)
        audit.append({"case_id": case["case_id"], "status": status, "detail": detail,
                      "pixels": int(mask.sum()) if mask is not None else None,
                      "image_size": case["image_size"]})
        if status == "present nonempty" and len(examples) < 3:
            with Image.open(case["image_path"]) as source:
                original = source.convert("RGB")
            processed = original.resize((896, 896), Image.Resampling.BILINEAR)
            processed_mask = np.asarray(Image.fromarray(mask).resize((896, 896), Image.Resampling.NEAREST))
            examples.append((case["case_id"], mask_display(original, mask), mask_display(processed, processed_mask)))
    if examples:
        figure = Figure(figsize=(10, len(examples) * 4))
        FigureCanvasAgg(figure)
        for row, (case_id, original, processed) in enumerate(examples):
            for column, (title, image) in enumerate((("Original coordinates", original), ("Direct-resize mask check (nearest)", processed))):
                axis = figure.add_subplot(len(examples), 2, row * 2 + column + 1)
                axis.imshow(image)
                axis.set_title(f"{case_id}\n{title}", fontsize=8)
                axis.axis("off")
        figure.tight_layout()
        figure.savefig(directory / "mask_alignment.png", dpi=130)
    atomic_json(directory / "mask_audit.json", {"partition": "discovery", "cases": audit})
    return audit


def run(directory):
    """Regenerate discovery rankings and freeze an immutable shortlist, offline."""
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["analysis"]["confirmation_enabled"]:
        raise ValueError("Confirmation is not implemented in this milestone")
    cohort = json.loads((directory / "cohort.json").read_text())
    if file_sha256(directory / "cohort.json") != protocol["cohort_sha256"]:
        raise ValueError("Saved cohort hash differs from protocol")
    for path, digest in cohort["metadata_sha256"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError("Annotation/label metadata changed")
    records, sufficient = [], []
    summaries = {k: [] for k in ("image_mean", "final", "image_max", "image_active_fraction")}
    color_limits = torch.zeros(protocol["sae_width"])
    for case in protocol["discovery_cases"]:
        record, payload = load_discovery_case(directory, case["case_id"], protocol)
        records.append(record)
        sufficient.append(payload["sufficient_statistics"])
        for name in summaries:
            summaries[name].append(payload["summaries"][name].numpy())
        sparse = payload["features"]
        eligible = torch.tensor([t["group"] in {"image", "text"} for t in record["tokens"]], dtype=torch.bool)
        mask = eligible[sparse.indices()[0]]
        color_limits.scatter_reduce_(0, sparse.indices()[1, mask], sparse.values()[mask], reduce="amax")
    summaries = {k: np.stack(v) for k, v in summaries.items()}
    mask_audit = audit_masks(directory, cohort, records)
    patients, patient_summaries = aggregate_patients(records, summaries)
    rows = feature_rankings(patients, patient_summaries, protocol["analysis"]["min_active_patients"])
    candidates = select_candidates(rows)
    shortlist = {"protocol_hash": protocol["protocol_hash"], "partition": "discovery",
                 "candidates": candidates, "interpretation": "Exploratory associations, not confirmed pathology features",
                 "follow_up_protocol_frozen": False,
                 "input_records_sha256": {r["case_id"]: file_sha256(directory / "cases" / f"{r['case_id']}.json") for r in records}}
    shortlist["shortlist_hash"] = identity_hash(shortlist)
    path = directory / "shortlist.json"
    if path.exists() and json.loads(path.read_text()) != shortlist:
        raise ValueError("Frozen shortlist differs; create a versioned experiment rather than overwrite")
    temporary = directory / "rankings.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / "rankings.csv")
    # Compact arrays for model-free scatterplots; full token features stay sharded.
    temporary = directory / "summaries.pt.tmp"
    torch.save({k: torch.from_numpy(v) for k, v in summaries.items()}, temporary)
    temporary.replace(directory / "summaries.pt")
    atomic_json(path, shortlist)
    colors = {str(feature): value for feature, value in enumerate(color_limits.tolist())}
    report = {"protocol_hash": protocol["protocol_hash"], "cases": len(records), "patients": len(patients),
              "mixed_label_patients": [p["patient"] for p in patients if p["mixed_label"]],
              "score_ties": sum(r["answer_score"]["tie"] for r in records),
              "invalid_answers": sum(r["parsed_answer"] == "invalid" for r in records),
              "parsed_score_disagreements": sum(r["parsed_score_disagreement"] for r in records),
              "fidelity": pooled_fidelity(records, sufficient),
              "runtime_seconds": sum(r["runtime_seconds"] for r in records),
              "peak_rss_bytes": max(r["peak_rss_bytes"] for r in records),
              "patients_metadata": patients, "discovery_color_limits": colors,
              "mask_status_counts": {status: sum(r["status"] == status for r in mask_audit)
                                     for status in {r["status"] for r in mask_audit}},
              "analysis_source_sha256": file_sha256(Path(__file__)),
              "viewer_source_sha256": file_sha256(Path(__file__).with_name("feature_viewer.py")),
              "analysis_versions": {name: version(name) for name in ("numpy", "scipy", "pillow", "matplotlib", "pycocotools", "streamlit")},
              "uv_lock_sha256": file_sha256(Path("uv.lock")) if Path("uv.lock").exists() else None,
              "artifacts_sha256": {name: file_sha256(directory / name) for name in
                                    ("rankings.csv", "shortlist.json", "summaries.pt", "mask_audit.json")},
              "limitations": ["Discovery selection is not a significance test", "Decoder-position maps are not localization",
                              "No feature interventions performed", "Confirmation inference disabled; test unused"]}
    atomic_json(directory / "discovery_report.json", report)
    print(f"Discovery complete: {len(records)} images; {len(patients)} patients; {len(candidates)} candidates")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment_dir)


if __name__ == "__main__":
    main()
