import csv
import json

from PIL import Image
import pytest

from medgemma_grounding.grounding_protocol import assign_patients, prepare_protocol, save_protocol


def fixture_dataset(root):
    """Repeated patient views, another abnormality, uncertain target, and lateral."""
    cases = [
        ("patient1", "view1_frontal", "1.0", "Frontal"),
        ("patient2", "view1_frontal", "1", "Frontal"),
        ("patient3", "view1_frontal", "0.0", "Frontal"),
        ("patient3", "view2_frontal", "0.0", "Frontal"),
        ("patient4", "view1_frontal", "0", "Frontal"),
        ("patient5", "view1_frontal", "-1", "Frontal"),
        ("patient6", "view1_frontal", "", "Frontal"),
        ("patient7", "view1_lateral", "1.0", "Lateral"),
    ]
    rows, annotations = [], {}
    for patient, view, label, orientation in cases:
        path = root / "CheXpert" / "val" / patient / "study1" / f"{view}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (8, 6)).save(path)
        rows.append({"Path": f"CheXpert-v1.0/valid/{patient}/study1/{view}.jpg",
                     "Pleural Effusion": label, "Frontal/Lateral": orientation,
                     "Cardiomegaly": "1.0" if patient == "patient3" else "0.0"})
        if patient in {"patient1", "patient2"}:
            annotations[f"{patient}_study1_{view}"] = {
                "img_size": [6, 8], "Pleural Effusion": [[[1, 1], [4, 1], [4, 4]]],
            }
    with (root / "CheXpert/val_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "CheXlocalize").mkdir()
    (root / "CheXlocalize/gt_annotations_val.json").write_text(json.dumps(annotations))


def test_patient_split_is_disjoint_order_independent_and_keeps_inspected_patient():
    rows = [{"patient_id": f"p{i}", "label": int(i < 4)} for i in range(8)]
    rows.append({"patient_id": "p0", "label": 0})  # Mixed labels still one positive patient.
    assignments = assign_patients(rows, seed=42, discovery_fraction=0.5, discovery_patients=("p0",))
    assert assignments == assign_patients(list(reversed(rows)), seed=42, discovery_fraction=0.5,
                                           discovery_patients=("p0",))
    assert assignments["p0"] == "discovery"
    for group in (range(4), range(4, 8)):
        assert sum(assignments[f"p{i}"] == "discovery" for i in group) == 2


def test_protocol_keeps_abnormal_negatives_and_excludes_uncertain_and_lateral(tmp_path):
    fixture_dataset(tmp_path)
    docs = prepare_protocol(tmp_path, discovery_patients=("patient1",))
    cases = docs["cohort_manifest.json"]["cases"]
    assert len(cases) == 5
    assert docs["cohort_summary.json"]["patients"] == 4
    assert docs["cohort_summary.json"]["excluded"] == {"non_frontal": 1, "missing_or_uncertain_label": 2}
    patient3 = [r for r in cases if r["patient_id"] == "patient3"]
    assert len({r["partition"] for r in patient3}) == 1
    assert all(r["other_positive_findings"] == ["Cardiomegaly"] for r in patient3)
    assert all(not r["localization_eligible"] and r["grounding_status"] == "unassessed" for r in patient3)
    assert sum(r["localization_eligible"] for r in cases) == 2


@pytest.mark.parametrize("problem", ["dimensions", "negative_contour", "duplicate"])
def test_protocol_rejects_misalignment_and_duplicate_cases(tmp_path, problem):
    fixture_dataset(tmp_path)
    path = tmp_path / "CheXlocalize/gt_annotations_val.json"
    annotations = json.loads(path.read_text())
    if problem == "dimensions":
        annotations["patient1_study1_view1_frontal"]["img_size"] = [8, 6]
    elif problem == "negative_contour":
        annotations["patient3_study1_view1_frontal"] = annotations["patient1_study1_view1_frontal"]
    else:
        labels = tmp_path / "CheXpert/val_labels.csv"
        with labels.open("a") as f:
            f.write(labels.read_text().splitlines()[1] + "\n")
    path.write_text(json.dumps(annotations))
    with pytest.raises(ValueError, match="dimensions|negative expert label|Duplicate"):
        prepare_protocol(tmp_path)


def test_missing_positive_contour_is_not_silently_converted_to_negative(tmp_path):
    fixture_dataset(tmp_path)
    (tmp_path / "CheXlocalize/gt_annotations_val.json").write_text("{}")
    docs = prepare_protocol(tmp_path)
    positive = [r for r in docs["cohort_manifest.json"]["cases"] if r["label"] == 1]
    assert len(positive) == 2
    assert all(r["localization_exclusion"] == "missing_positive_contour" for r in positive)


def test_saved_protocol_is_repeatable_and_cannot_be_replaced(tmp_path):
    output = tmp_path / "protocol"
    docs = {"protocol.json": {"seed": 42}}
    save_protocol(output, docs)
    original = (output / "protocol.json").read_bytes()
    save_protocol(output, docs)
    with pytest.raises(ValueError, match="different files/settings"):
        save_protocol(output, {"protocol.json": {"seed": 7}})
    assert (output / "protocol.json").read_bytes() == original


def test_unknown_or_excess_pinned_patients_are_rejected():
    rows = [{"patient_id": f"p{i}", "label": int(i < 2)} for i in range(4)]
    with pytest.raises(ValueError, match="absent"):
        assign_patients(rows, seed=42, discovery_fraction=0.5, discovery_patients=("missing",))
    with pytest.raises(ValueError, match="exceed"):
        assign_patients(rows, seed=42, discovery_fraction=0.5, discovery_patients=("p0", "p1"))
