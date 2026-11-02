"""Full-cohort repaired discovery gates, using a small exactly linear model."""

from types import SimpleNamespace

import pytest
import torch
from streamlit.testing.v1 import AppTest
from PIL import Image

from test_feature_interventions import Toy, CANDIDATES
from medgemma_grounding import medgemma_discovery as workflow
from medgemma_grounding import medgemma_interventions as base
from medgemma_grounding import medgemma_features
from medgemma_grounding.medgemma_features import atomic_json, file_sha256, identity_hash


@pytest.fixture
def fresh_run(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "cases").mkdir(parents=True)
    atomic_json(source / "shortlist.json", {"candidates": [{"feature": 1}]})
    image = source / "image.png"
    Image.new("RGB", (20, 20), "gray").save(image)
    masks = {"image": torch.tensor([False, True, True, True, True, False]),
             "final": torch.tensor([False, False, False, False, False, True]),
             "other_non_image": torch.tensor([True, False, False, False, False, False])}
    records, reference, inputs = {}, {}, {}
    for i, (label, positive) in enumerate(((1, True), (0, False), (0, True), (1, False), (1, True))):
        case = f"case{i}"
        x = torch.ones(1, 6, 2) * (i + 1)
        x[:, :, 1] *= 1 if positive else 4
        inputs[case] = {"x": x}
        record = {"case_id": case, "patient": f"patient{min(i, 3)}", "label": label,
                  "tokens": [{"id": t, "text": str(t), "group": "image" if 1 <= t <= 4 else "text"} for t in range(6)],
                  "image_path": str(image), "answer": "Yes", "positions": {},
                  "spatial_layout": {"overlay_supported": False},
                  "answer_score": {"decision": "Yes", "score": 999.}}
        records[case] = record
        # Old support deliberately incompatible with the newly encoded support.
        reference[case] = {"features": torch.zeros(6, 16).to_sparse().coalesce()}
        atomic_json(source / "cases" / f"{case}.json", record)
    protocol = {"schema_version": 3, "fresh_encodings": True, "experiment": "E-007",
                "source_directory": str(source), "sae_width": 16, "hook": "layer", "scoring": CANDIDATES,
                "discovery_case_ids": list(records), "calibration": {"case_ids": list(records)},
                "smoke_ids": ["case0", "case1"], "previous_artifacts": {},
                "source_artifact_sha256": {f"cases/{c}.json": file_sha256(source / "cases" / f"{c}.json") for c in records}}
    protocol["protocol_hash"] = identity_hash(protocol)
    atomic_json(output / "protocol.json", protocol)
    model = Toy()
    model.dtype = torch.float32
    bundle = SimpleNamespace(model=model)
    scales = torch.linspace(.4, 2, 16)
    sae = SimpleNamespace(W_dec=torch.stack([torch.ones(16), torch.zeros(16)], dim=1),
                          encode=lambda h: h[:, :1].abs() * scales,
                          decode=lambda z: z @ torch.stack([torch.ones(16), torch.zeros(16)], dim=1))
    monkeypatch.setattr(workflow, "protocol_for", lambda *a: (protocol, list(records.values())))
    for module in (base, workflow, medgemma_features):
        monkeypatch.setattr(module, "load_discovery_case", lambda directory, case: (records[case], reference[case]))
    monkeypatch.setattr(base, "prepared_inputs", lambda b, p, r: (inputs[r["case_id"]], masks))
    calls = []
    monkeypatch.setattr(base, "frozen_runtime", lambda *a: (calls.append(True) or bundle, sae))
    return source, output, protocol, calls


def test_auto_full_cohort_fresh_support_and_model_free_resume(fresh_run, monkeypatch):
    source, output, protocol, calls = fresh_run
    result = workflow.run("run", source, output)
    assert result["complete"] and result["status"] == "cohort_complete"
    assert result["calibration_cases"] == 5 and result["patients"] == 4
    assert result["calibration_measurements"] == 90
    assert all(v == "complete" for v in result["stages"].values())
    assert result["repaired_outcomes"] == {"TP": 2, "TN": 1, "FP": 1, "FN": 1}
    assert calls == [True]
    record, payload = base.load_case(output, protocol, "case1")
    assert record["outcome"] == "TN" and payload["features"]._nnz() == 96
    assert torch.equal(payload["features"].indices(), payload["attribution"].indices())
    calibration = base.pilot_report(output, protocol)
    assert set(calibration["qualifying_groups"]) == {"image", "final", "other_non_image"}
    for pair in base.read_candidates(output, protocol)["pairs"]:
        assert pair["target"]["calibration_half_cases"] > 0
        assert pair["target"]["calibration_full_cases"] > 0
        assert "supporting_label_auc" in pair["target"]
    def forbid(*a, **k):
        raise AssertionError("Completed resume/report must be model-free")
    monkeypatch.setattr(base, "frozen_runtime", forbid)
    assert workflow.run("run", source, output)["complete"]
    assert workflow.run("report", source, output)["complete"]
    app = AppTest.from_string(
        "from pathlib import Path\nfrom medgemma_grounding.feature_viewer import run_intervention_viewer\n"
        f"run_intervention_viewer(Path({str(output)!r}), Path({str(source)!r}))\n").run(timeout=30)
    assert not app.exception and not app.error, [e.value for e in app.error]


def test_nonqualifying_groups_stop_before_selection(fresh_run, monkeypatch):
    source, output, _, _ = fresh_run
    monkeypatch.setattr(base, "agreement", lambda rows, tau: {
        "pairs": len(rows), "spearman": .1, "sign_agreement": .5, "median_relative_error": 1.})
    result = workflow.run("run", source, output)
    assert result["complete"] and result["status"] == "calibration_stopped"
    assert result["attributed_cases"] == 5 and result["calibration_cases"] == 5
    assert result["stages"]["selection"] == "skipped_calibration"
    assert not (output / "candidates.json").exists()


def test_smoke_failure_stops_expansion(fresh_run, monkeypatch):
    source, output, _, _ = fresh_run
    def fail(*a, **k):
        raise RuntimeError("unsupported backward")
    monkeypatch.setattr(base, "capture_gradient", fail)
    with pytest.raises(RuntimeError, match="unsupported backward"):
        workflow.run("run", source, output)
    result = base.read_json(output / "intervention_report.json")
    assert not result["complete"] and result["status"] == "technical_failure"
    assert result["attributed_cases"] == 0
    assert (output / "failure_smoke.json").exists()
    with pytest.raises(ValueError, match="technical failure"):
        workflow.run("run", source, output)


def test_limit_counts_new_cases_and_requires_complete_calibration(fresh_run):
    source, output, _, _ = fresh_run
    workflow.run("smoke", source, output)
    first = workflow.run("calibrate", source, output, limit=1)
    assert first["calibration_cases"] == 1
    second = workflow.run("calibrate", source, output, limit=1)
    assert second["calibration_cases"] == 2
    with pytest.raises(ValueError, match="incomplete"):
        workflow.run("select", source, output)


def test_new_protocol_has_full_cohort_no_old_quota_and_rejects_drift(tmp_path, monkeypatch):
    from medgemma_grounding.sae_reconstruction import checkpoint_spec, SAE_REVISION
    release, sae_id, _, hook = checkpoint_spec(14, all_layers=True)
    original = {"protocol_hash": "source", "model_id": "model", "model_revision": "revision",
                "processor_revision": "revision", "prompt": "question", "sae_revision": SAE_REVISION,
                "sae_release": release, "sae_id": sae_id, "sae_width": 16384, "hook": hook,
                "scoring": CANDIDATES}
    records = [{"case_id": f"case{i}", "image_sha256": "image"} for i in range(3)]
    monkeypatch.setattr(base, "validate_source", lambda *a: (original, records, {}))
    monkeypatch.setattr(base, "choose_pilot", lambda *a: pytest.fail("No old quota selection"))
    repaired = {"smoke_ids": ["case0", "case1"], "previous_artifacts": {}, "preflight_sha256": "hash"}
    protocol, _ = base.experiment_protocol(tmp_path / "source", tmp_path / "output", repaired=repaired)
    assert "pilot" not in protocol and protocol["calibration"]["case_ids"] == ["case0", "case1", "case2"]
    assert protocol["thresholds"] == base.THRESHOLDS
    assert protocol["runtime"]["attention_implementation"] == {"text_config": "eager", "vision_config": "sdpa"}
    assert base.experiment_protocol(tmp_path / "source", tmp_path / "output", repaired=repaired)[0] == protocol
    repaired["preflight_sha256"] = "changed"
    with pytest.raises(ValueError, match="resume"):
        base.experiment_protocol(tmp_path / "source", tmp_path / "output", repaired=repaired)
