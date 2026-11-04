import csv
import io
import json
from types import SimpleNamespace
import zipfile

import pytest

from medgemma_grounding import grounding_baseline as baseline
from medgemma_grounding import baseline_pilot as pilot
from medgemma_grounding.grounding_protocol import prepare_protocol, save_protocol
from test_grounding_protocol import fixture_dataset


def sample_rows():
    cases, predictions = [], []
    for i, (label, response, prediction, score) in enumerate([
        (1, "Yes", "1", ".9"), (1, "No", "0", ".2"),
        (0, "No", "0", ".1"), (0, "Yes", "1", ".8"),
        (1, "Cannot determine", "", ".75"), (0, "Uncertain", "", ".25"),
    ]):
        case_id = f"patient{i}_study1_view1_frontal"
        cases.append({"case_id": case_id, "patient_id": f"patient{i}", "label": label,
                      "partition": "discovery", "localization_eligible": label == 1,
                      "grounding_status": "unassessed"})
        predictions.append({"image_path": f"data/CheXpert/val/patient{i}/study1/view1_frontal.jpg",
                            "ground_truth": str(label), "response": response,
                            "prediction": prediction, "yes_score": score})
    return cases, predictions


def test_join_is_by_id_and_retains_unparsed_and_false_negative_localization_cases():
    cases, predictions = sample_rows()
    rows = baseline.reconcile(list(reversed(predictions)), cases)
    assert [r["case_id"] for r in rows] == [r["case_id"] for r in cases]
    metrics = baseline.metrics_for(rows)
    assert (metrics["total_images"], metrics["parsed_images"], metrics["unparsed_images"]) == (6, 4, 2)
    assert metrics["accuracy"] == .5
    assert metrics["unparsed_positive"] == metrics["unparsed_negative"] == 1
    assert metrics["localization_eligible"] == 3
    assert metrics["roc_auc"] is not None
    assert all(r["grounding_status"] == "unassessed" for r in rows)
    assert {r["classification_category"] for r in rows if r["localization_eligible"]} == {
        "true_positive", "false_negative", "unparsed"}


@pytest.mark.parametrize("problem", ["duplicate", "missing", "extra", "label", "parser", "nan", "out_of_range"])
def test_invalid_or_misaligned_saved_predictions_are_rejected(problem):
    cases, predictions = sample_rows()
    if problem == "duplicate":
        predictions.append(dict(predictions[0]))
    elif problem == "missing":
        predictions.pop()
    elif problem == "extra":
        predictions[0]["image_path"] = "patient999/study1/view1_frontal.jpg"
    elif problem == "label":
        predictions[0]["ground_truth"] = "0"
    elif problem == "parser":
        predictions[0]["prediction"] = "0"
    else:
        predictions[0]["yes_score"] = "nan" if problem == "nan" else "1.01"
    with pytest.raises(ValueError):
        baseline.reconcile(predictions, cases)


def test_all_unparsed_keeps_continuous_metrics():
    cases, predictions = sample_rows()
    for row in predictions:
        row.update(response="Uncertain", prediction="")
    metrics = baseline.metrics_for(baseline.reconcile(predictions, cases))
    assert metrics["parsed_images"] == 0
    assert metrics["binary_metrics"] is None
    assert metrics["roc_auc"] is not None
    assert metrics["unparsed_images"] == 6


def test_curation_never_selects_confirmation_and_is_order_independent():
    cases, predictions = sample_rows()
    rows = baseline.reconcile(predictions, cases)
    rows[0]["partition"] = "confirmation"
    chosen = baseline.curate_cases(rows, 42)
    assert chosen == baseline.curate_cases(list(reversed(rows)), 42)
    assert all(r["partition"] == "discovery" for r in chosen["cases"])
    assert "false_negative" in {r["classification_category"] for r in chosen["cases"]}


def test_zip_read_does_not_extract_files(tmp_path):
    _, predictions = sample_rows()
    content = io.StringIO()
    writer = csv.DictWriter(content, fieldnames=predictions[0].keys())
    writer.writeheader()
    writer.writerows(predictions)
    archive = tmp_path / "result.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("classification/predictions.csv", content.getvalue())
        handle.writestr("../../unrelated.txt", "not a prediction")
    rows, provenance = baseline.read_predictions(archive)
    assert rows == predictions
    assert provenance["archive_member"] == "classification/predictions.csv"
    assert list(tmp_path.iterdir()) == [archive]


def test_frozen_cohort_and_source_changes_are_detected(tmp_path, monkeypatch):
    data = tmp_path / "data"
    fixture_dataset(data)
    directory = tmp_path / "protocol"
    save_protocol(directory, prepare_protocol(data))
    _, cases = baseline.load_protocol(directory)
    assert len(cases) == 5
    path = directory / "cohort_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["cases"][0]["partition"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        baseline.load_protocol(directory)
    monkeypatch.setattr(baseline, "read_part1_contract", lambda: {})
    with pytest.raises(ValueError, match="Part 1 source"):
        baseline.load_protocol(directory)


def test_pilot_calls_inherited_generation_and_scorer(monkeypatch):
    import torch
    from medgemma_grounding import classification, inference

    prefix = list(range(10, 20))
    encodings = {"rendered": prefix, "No": [40], "Yes": [41],
                 "renderedNo": prefix + [40], "renderedYes": prefix + [41]}
    tokenizer = SimpleNamespace(encode=lambda value, **kwargs: encodings[value])
    processor = SimpleNamespace(tokenizer=tokenizer, apply_chat_template=lambda *args, **kwargs: "rendered")
    bundle = SimpleNamespace(processor=processor, torch=torch)
    calls = []
    monkeypatch.setattr(inference, "prepare_inputs", lambda *args: {"input_ids": torch.tensor([prefix])})
    def generate(bundle_arg, image, prompt, **kwargs):
        calls.append(("generate", prompt, kwargs))
        assert bundle_arg is bundle
        return "Yes"
    def score(bundle_arg, image, prompt):
        calls.append(("score", prompt))
        assert bundle_arg is bundle
        return .8
    monkeypatch.setattr(inference, "generate", generate)
    monkeypatch.setattr(classification, "compute_yes_score", score)
    contract = {"prompt": classification.PROMPT, "max_new_tokens": classification.MAX_NEW_TOKENS}
    result = pilot.evaluate_case(bundle, object(), contract,
                                 {"response": "Yes", "prediction": 1, "yes_score": .8}, atol=1e-5, rtol=1e-4)
    assert calls == [("generate", contract["prompt"], {"max_new_tokens": 16}),
                     ("score", contract["prompt"]), ("score", contract["prompt"])]
    assert result["comparison"]["repeat_agrees"]
    assert result["prediction"] == 1
    assert result["tokenization"]["candidate_ids"] == {"No": 40, "Yes": 41}
    encodings["renderedYes"] = prefix + [99]
    with pytest.raises(ValueError, match="continuation"):
        pilot.evaluate_case(bundle, object(), contract, {}, atol=1e-5, rtol=1e-4)


def test_score_agreement_reports_discrepancy_without_replacing_historical_score():
    report = pilot.compare_scores([.7, .7], .8, atol=1e-5, rtol=1e-4)
    assert report["repeat_agrees"]
    assert not report["historical_score_agrees"]
    assert report["historical_score"] == .8
    assert report["historical_absolute_difference"] == pytest.approx(.1)


def test_audit_roundtrip_and_pilot_selection_guardrails(tmp_path):
    data = tmp_path / "data"
    fixture_dataset(data)
    protocol = tmp_path / "protocol"
    prepared = prepare_protocol(data)
    save_protocol(protocol, prepared)
    cases = prepared["cohort_manifest.json"]["cases"]
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "ground_truth", "response", "prediction", "yes_score"])
        writer.writeheader()
        for case in cases:
            writer.writerow({"image_path": case["image_path"], "ground_truth": case["label"],
                             "response": "Yes" if case["label"] else "No", "prediction": case["label"],
                             "yes_score": .8 if case["label"] else .2})
    output = tmp_path / "audit"
    first = baseline.audit(protocol, predictions, output)
    assert baseline.audit(protocol, predictions, output) == first
    assert first["metrics"]["all"]["total_images"] == 5
    discovery = next(r["case_id"] for r in cases if r["partition"] == "discovery")
    confirmation = next(r["case_id"] for r in cases if r["partition"] == "confirmation")
    _, selected = pilot.select_case(protocol, output, discovery)
    assert selected["case_id"] == discovery
    with pytest.raises(ValueError, match="discovery"):
        pilot.select_case(protocol, output, confirmation)
    payload = output / "baseline_cases.json"
    changed = json.loads(payload.read_text())
    changed[0]["yes_score"] = .123
    payload.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="frozen protocol/files"):
        pilot.select_case(protocol, output, discovery)
