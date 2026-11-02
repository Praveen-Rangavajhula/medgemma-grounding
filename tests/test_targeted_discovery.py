"""E-008 contracts: inclusive diagnostics, patient nomination, BF16 controls, gates."""

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from medgemma_grounding import medgemma_targeted as study, targeted_analysis as analysis
from medgemma_grounding import medgemma_interventions as base
from medgemma_grounding.feature_controls import direction_bank, match_direction, MATCHING
from medgemma_grounding.feature_interventions import feature_removal, project_attribution
from medgemma_grounding.medgemma_features import atomic_json, identity_hash, file_sha256
from medgemma_grounding.medgemma_preflight import tree_hashes
from test_feature_interventions import Toy, CANDIDATES


def test_all_pair_diagnostics_include_zero_small_false_alarm_and_opposite_sign():
    rows = [{"patient": f"p{i}", "predicted_delta": p, "actual_delta": a} for i, (p, a) in enumerate(
        ((0, 0), (2, 0), (-2, 2), (.1, .1), (2, 3)))]
    result = analysis.agreement(rows, .25)
    assert result["pairs"] == 5 and result["actual_zero"] == 2
    assert result["predicted_substantial"] == 3 and result["predicted_large_actual_small"] == 1
    assert result["false_alarm_fraction_of_large_predictions"] == pytest.approx(1 / 3)
    assert result["opposite_sign"] == 1 and result["mae"] == pytest.approx(1.4)
    assert analysis.agreement([], .25)["spearman"] is None
    assert analysis.agreement([rows[0]], .25)["sign_agreement_including_zero"] == 1
    assert analysis.agreement([rows[0]], .25)["sign_agreement_both_nonzero"] is None


def test_patient_weighting_and_nomination_tracks_are_separate():
    records = [{"case_id": f"c{i}", "patient": f"p{p}", "label": label, "outcome": outcome,
                "answer_score": {"score": score}} for i, (p, label, outcome, score) in enumerate(
        ((0, 1, "TP", 2), (1, 0, "TN", -2), (2, 0, "FP", 1), (3, 1, "FN", -1), (0, 1, "TP", 4)))]
    x = np.array([[2., 1, 2, 1], [0, 1, 1, 1], [4, 3, 2, 1], [3, 2, 5, 1], [4, 1, 2, 1]])
    arrays = {g: {"activity": x, "attribution": -x} for g in analysis.GROUPS}
    metadata, pooled = analysis.aggregate_patients(records, arrays)
    assert pooled["image"]["activity"][0, 0] == 3 and len(metadata) == 4
    settings = {**analysis.NOMINATION, "minimum_active_patients": 1,
                "minimum_active_error_patients": 1, "reference_final_features": [0, 1]}
    candidates, rows, _ = analysis.nominate(records, arrays, settings)
    assert len(candidates) <= 8
    assert any("FP_vs_TP" in c["tracks"] for c in candidates)
    assert any("FN_vs_TN" in c["tracks"] for c in candidates)
    assert all("constant" in r["effusion_exclusion"] for r in rows if r["feature"] == 3)
    assert all("positive_mean_activity" in c["statistics"] for c in candidates)
    assert all("TP_mean_attribution" in c["statistics"] for c in candidates)
    assert analysis.uncertainty([{"patient": "a", "v": 2}, {"patient": "a", "v": 4}, {"patient": "b", "v": 1}], "v")["mean"] == 2
    records[-1]["outcome"] = "FN"
    with pytest.raises(ValueError, match="Mixed"):
        analysis.aggregate_patients(records, arrays)


def test_controls_match_executed_norms_and_preserve_positions_and_residual():
    generator = torch.Generator().manual_seed(3)
    hidden = (torch.randn(1, 5, 64, generator=generator) * 10).bfloat16().float()
    z = torch.full((5, 2), 2.)
    decoder = direction_bank(64, 2, seed=5)
    mask = torch.tensor([False, True, True, True, False])
    target, norms = feature_removal(hidden, z, decoder, 0, mask, .5, torch.bfloat16)
    for direction in direction_bank(64):
        control, diagnostics = match_direction(hidden, target, mask, direction)
        assert diagnostics["matched"] and diagnostics["status"] == "matched"
        assert diagnostics["max_position_relative_error"] <= .05 + 1e-6
        assert diagnostics["realized_l2"] == pytest.approx(norms["realized_l2"], rel=.05)
        assert torch.equal(control[:, ~mask], hidden[:, ~mask])
        assert torch.equal(control, control.bfloat16().float())
    assert torch.equal(direction_bank(64), direction_bank(64))
    # Residual and bias are preserved by the original edit, up to its BF16 cast.
    bias = torch.ones(64)
    residual = hidden[0] - z @ decoder - bias
    changed_z = z.clone()
    changed_z[mask, 0] *= .5
    ideal = changed_z @ decoder + bias + residual
    torch.testing.assert_close(target[0, mask], ideal[mask].bfloat16().float(), rtol=1e-5, atol=1e-5)


def test_zero_rounded_and_unmatchable_controls_are_explicit():
    h = torch.tensor([[[1., 1024.], [2., 3.]]])
    mask = torch.tensor([True, False])
    zero, diag = match_direction(h, h, mask, torch.tensor([0., 1.]))
    assert torch.equal(zero, h) and diag["status"] == "zero_target"
    target = h.clone()
    target[0, 0, 0] += .0078125
    control, diag = match_direction(h, target, mask, torch.tensor([0., 1.]))
    assert not diag["matched"] and diag["status"] == "unmatched"
    assert torch.equal(control[:, ~mask], h[:, ~mask])
    target, norm = feature_removal(h, torch.ones(2, 1), torch.tensor([[1e-6, 0.]]), 0, mask, 1, torch.bfloat16)
    assert norm["requested_l2"] > 0 and norm["realized_l2"] == 0
    assert match_direction(h, target, mask, torch.tensor([1., 0.]))[1]["status"] == "zero_target"


def test_paired_control_distribution_requires_complete_positive_size_bank():
    t = {"case_id": "c", "patient": "p", "label": 0, "outcome": "FP", "group": "image", "feature": 1,
         "alpha": 1., "original_score": 1., "original_decision": "Yes", "actual_delta": -2.,
         "predicted_delta": -1., "kl_from_baseline": .1, "requested_l2": 3., "realized_l2": 3.,
         "target_state": "positive_size", "corrected": True, "introduced": False, "execution": "new_forward"}
    key = ("c", "image", 1, 1.)
    bank = {key: [{"bank": i, "matched": True, "actual_delta": v} for i, v in enumerate((1, -1, 0))]}
    row = analysis.paired_cases([t], bank, 3)[0]
    assert row["primary_matched"] and row["matched_abs_excess"] == pytest.approx(4 / 3)
    assert row["toward_expert_delta"] == 2 and row["strengthening_delta"] == -2
    bank[key][0]["matched"] = False
    row = analysis.paired_cases([t], bank, 3)[0]
    assert not row["primary_matched"] and row["matched_abs_excess"] is None


def test_protocol_identity_and_cohort_resume_guards(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    parent = {"source_directory": str(tmp_path / "old"), "previous_artifacts": {}}
    parent["protocol_hash"] = identity_hash(parent)
    atomic_json(source / "protocol.json", parent)
    atomic_json(output / "source_manifest.json", {"source_tree_sha256": tree_hashes(source)})
    atomic_json(output / "candidates.json", {"candidates": []})
    p = {"source_directory": str(source), "source_protocol_hash": parent["protocol_hash"],
         "confirmation_enabled": False, "test_enabled": False, "matching": MATCHING,
         "nomination": analysis.NOMINATION, "control_bank_size": 3,
         "artifacts_sha256": {"candidates.json": file_sha256(output / "candidates.json")}, "source_sha256": {}, "versions": {},
         "discovery_case_ids": ["c"]}
    p["protocol_hash"] = identity_hash(p)
    assert study.validate_protocol(output, p) == parent
    with pytest.raises(ValueError, match="discovery"):
        study.source_case(p, parent, "confirmation")
    with pytest.raises(ValueError, match="separate"):
        study.check_output(source, source / "nested")
    atomic_json(output / "candidates.json", {"candidates": ["drift"]})
    with pytest.raises(ValueError, match="Frozen artifact"):
        study.validate_protocol(output, p)


@pytest.fixture
def toy_run(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    parent = {"source_directory": str(tmp_path / "old"), "previous_artifacts": {},
              "hook": "layer", "scoring": CANDIDATES, "protocol_hash": "source"}
    atomic_json(source / "protocol.json", parent)
    records, payloads, inputs = {}, {}, {}
    model = Toy()
    model.dtype = torch.bfloat16
    class SAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W_dec = torch.nn.Parameter(torch.tensor([[1., 0.], [1., .25]]), requires_grad=False)
    sae = SAE().eval()
    masks = {"image": torch.tensor([False, True, True, True, True, False]),
             "final": torch.tensor([False, False, False, False, False, True]),
             "other_non_image": torch.tensor([True, False, False, False, False, False])}
    for i, (label, positive) in enumerate(((1, True), (0, False), (0, True), (1, False), (1, True))):
        case = f"c{i}"
        h = torch.ones(1, 6, 2) * (i + 1) * 2
        h[:, :, 1] *= 1 if positive else 4
        features = torch.ones(6, 2) * (i + 1)
        if i == 4:
            features[masks["image"], 0] = 0
        gradient = model.weight.detach().expand_as(h).clone()
        attribution, groups = project_attribution(gradient, features, sae.W_dec, masks)
        baseline = model(x=h).logits.detach()
        score = study.answer_score(baseline, CANDIDATES)
        record = {"case_id": case, "patient": f"p{i if i < 4 else 0}", "label": label,
                  "answer_score": score, "tensor_sha256": f"tensor{i}",
                  "positions": {g: m.nonzero().flatten().tolist() for g, m in masks.items()}}
        record["outcome"] = base.outcome(record)
        records[case] = record
        payloads[case] = {"hidden": h, "gradient": gradient, "baseline_logits": baseline,
                          "features": features.to_sparse().coalesce(), "attribution": attribution, "groups": groups}
        inputs[case] = {"x": h}
    candidates = [{"group": "image", "feature": 0, "tracks": ["effusion"]},
                  {"group": "final", "feature": 1, "tracks": ["FN_vs_TN"]}]
    p = {"protocol_hash": "targeted", "source_directory": str(source), "scoring": CANDIDATES,
         "hook": "layer", "sae_width": 2, "hidden_width": 2, "discovery_case_ids": list(records),
         "smoke_ids": ["c0", "c1"], "case_tensor_sha256": {c: r["tensor_sha256"] for c, r in records.items()},
         "matching": MATCHING, "control_bank_size": 3}
    atomic_json(output / "protocol.json", p)
    atomic_json(output / "candidates.json", {"candidates": candidates})
    Image.new("RGB", (10, 10), "gray").save(output / "e007_agreement.png")
    torch.save(direction_bank(2), output / "control_bank.pt")
    monkeypatch.setattr(study, "prepare", lambda *a: p)
    monkeypatch.setattr(study, "validate_protocol", lambda *a, **k: parent)
    def load_source(protocol, parent, case):
        if case not in p["discovery_case_ids"]:
            raise ValueError("Discovery only")
        return records[case], payloads[case]
    monkeypatch.setattr(study, "source_case", load_source)
    monkeypatch.setattr(study.discovery, "treatment_context", lambda bundle, sae, parent, record, payload: (
        inputs[record["case_id"]], payload["features"].to_dense(), record, payload))
    calls = []
    bundle = SimpleNamespace(model=model)
    monkeypatch.setattr(base, "frozen_runtime", lambda *a: (calls.append(True) or bundle, sae))
    old = base.measure_treatment(bundle, parent, records['c0'], payloads['c0'], payloads['c0']['features'].to_dense(),
                                  sae.W_dec, inputs['c0'], 0, 'image', 1.)
    old_path = base.treatment_path(source, 'c0', 0, 'image', 1.)
    base.save_treatment(old_path, old)
    return source, output, p, calls, old_path


def test_complete_workflow_cache_reuse_and_model_free_resume(toy_run, monkeypatch):
    source, output, protocol, calls, old_path = toy_run
    old_hash = file_sha256(old_path)
    result = study.run("run", source, output)
    assert result["complete"] and result["completed_cases"] == 5
    assert result["measurement_records"] == 80 and result["reused_target_measurements"] == 1
    assert result["control_matching"]["unmatched_controls"] == 0
    assert file_sha256(old_path) == old_hash and calls == [True]
    before = (output / "report.json").read_bytes()
    def forbid(*a, **k):
        raise AssertionError("No model/SAE needed for completed resume")
    monkeypatch.setattr(base, "frozen_runtime", forbid)
    assert study.run("run", source, output)["complete"]
    assert (output / "report.json").read_bytes() == before
    path = study.measurement_path(output, "c0", "image", 0, 1.)
    row = base.read_json(path)
    row["actual_delta"] += 1
    atomic_json(path, row)
    with pytest.raises(ValueError, match="checksum"):
        study.run("run", source, output)


def test_failure_stops_before_cohort_and_retains_diagnostic(toy_run, monkeypatch):
    source, output, _, calls, _ = toy_run
    def fail(*a, **k):
        raise RuntimeError("gradient-forward technical failure")
    monkeypatch.setattr(study, "capture_gradient", fail)
    with pytest.raises(RuntimeError, match="technical failure"):
        study.run("run", source, output)
    result = base.read_json(output / "report.json")
    assert not result["complete"] and result["measurement_records"] == 0
    assert result["failures"][0]["settings_changed"] is False
    with pytest.raises(ValueError, match="Technical failure recorded"):
        study.run("run", source, output)


def test_partial_cohort_resume_and_unmatched_controls_do_not_run(toy_run, monkeypatch):
    source, output, protocol, calls, _ = toy_run
    assert not study.run("smoke", source, output)["complete"]
    first = study.run("cohort", source, output, limit=1)
    assert first["completed_cases"] == 3  # two smoke cases already contain every toy candidate
    assert not first["complete"]
    real_match = study.match_direction
    def unmatched(*args, **kwargs):
        edited, diagnostic = real_match(*args, **kwargs)
        diagnostic.update(matched=False, status="unmatched")
        return edited, diagnostic
    monkeypatch.setattr(study, "match_direction", unmatched)
    study.run("cohort", source, output, limit=1)
    rows = [base.read_json(p) for p in (output / "measurements" / "c3").glob("*.json")]
    failed = [r for r in rows if r["bank"] is not None and not r["matched"]]
    assert failed and all(r["actual_delta"] is None and r["execution"] == "unmatched" for r in failed)
    # Every remaining case is still processed; incomplete banks are explicit exclusions.
    result = study.run("cohort", source, output)
    assert result["complete"] and result["control_matching"]["unmatched_controls"] > 0
    assert any(s["matching_exclusions"]["incomplete_or_unmatched_bank"] for s in result["summaries"])


def test_visual_hypotheses_cannot_reference_held_out_cases(toy_run):
    _, output, protocol, _, _ = toy_run
    with pytest.raises(ValueError):
        analysis.save_visual_hypothesis(output, protocol, "image", 0, "unconfirmed", ["confirmation"])
    analysis.save_visual_hypothesis(output, protocol, "image", 0, "human idea", ["c0"])
    note = base.read_json(output / "visual_hypotheses.json")["features"]["image:0"]
    assert note["source"] == "human-proposed" and note["counterexample_cases"] == ["c0"]
