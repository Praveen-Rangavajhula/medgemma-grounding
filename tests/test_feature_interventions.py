"""Numerical contracts and stage safety without model weights or held-out data."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from streamlit.testing.v1 import AppTest

from test_feature_discovery import discovery_cache
from medgemma_grounding import medgemma_interventions as study
from medgemma_grounding.feature_interventions import (
    GROUPS, capture_gradient, feature_removal, intervention_groups, project_attribution,
)
from medgemma_grounding.feature_viewer import signed_overlay
from medgemma_grounding.medgemma_features import atomic_json, file_sha256, identity_hash, load_discovery_case
from medgemma_grounding.sae_reconstruction import replace_residual


class TupleOutput(torch.nn.Module):
    def forward(self, x):
        return (x, "preserved")


class Toy(torch.nn.Module):
    def __init__(self, tuple_output=False, nonlinear=False, double=False, fail=False):
        super().__init__()
        self.layer = TupleOutput() if tuple_output else torch.nn.Identity()
        self.weight = torch.nn.Parameter(torch.tensor([2., -1.]), requires_grad=False)
        self.nonlinear, self.double, self.fail = nonlinear, double, fail
        self.eval()

    def forward(self, x, use_cache=False, logits_to_keep=1):
        assert not use_cache and logits_to_keep == 1
        value = self.layer(x)
        hidden = value[0] if isinstance(value, tuple) else value
        if self.double:
            self.layer(x)
        if self.fail:
            raise RuntimeError("downstream failure")
        score = (hidden * self.weight).sum()
        if self.nonlinear:
            score = score ** 3
        return SimpleNamespace(logits=torch.stack([score, score * 0]).reshape(1, 1, 2))


CANDIDATES = {"ids": {"Yes": 0, "No": 1}}


@pytest.mark.parametrize("tuple_output", [False, True])
def test_linear_attribution_exact_signed_strength_and_residual(tuple_output):
    model = Toy(tuple_output)
    hidden = torch.tensor([[[2., 1.], [1., 3.], [4., 2.]]])
    features = torch.tensor([[1., 2.], [2., 1.], [1., 0.]])
    decoder = torch.tensor([[1., 0.], [0., 1.]])
    groups = {"image": torch.tensor([True, True, False]), "final": torch.tensor([False, False, True])}
    original, baseline, gradient = capture_gradient(model, {"x": hidden}, "layer", CANDIDATES)
    attribution, summaries = project_attribution(gradient, features, decoder, groups, chunk_size=1)
    assert torch.equal(attribution.indices(), features.to_sparse().coalesce().indices())
    for group, mask in groups.items():
        for feature in range(2):
            for alpha in (0, 0.5, 1):
                edited, norms = feature_removal(original, features, decoder, feature, mask, alpha, torch.float32)
                logits = replace_residual(model, {"x": hidden}, "layer", edited)
                actual = (logits - baseline)[0, 0, 0].item()
                assert actual == pytest.approx(alpha * summaries[group]["effect"][feature].item())
                assert torch.equal(edited[:, ~mask], hidden[:, ~mask])
                assert norms["realized_l2"] == pytest.approx((edited - hidden).norm().item())
                # Imperfect affine SAE: the residual and decoder bias survive exactly.
                bias = torch.tensor([7., -3.])
                residual = hidden[0] - (features @ decoder + bias)
                changed_features = features.clone()
                changed_features[mask, feature] *= 1 - alpha
                torch.testing.assert_close(edited[0], changed_features @ decoder + bias + residual)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert not model.layer._forward_hooks


@pytest.mark.parametrize("options", [{"double": True}, {"fail": True}])
def test_gradient_hook_cleanup_after_exception(options):
    model = Toy(**options)
    with pytest.raises(RuntimeError):
        capture_gradient(model, {"x": torch.ones(1, 3, 2)}, "layer", CANDIDATES)
    assert not model.layer._forward_hooks and model.weight.grad is None
    model.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        capture_gradient(model, {"x": torch.ones(1, 3, 2)}, "layer", CANDIDATES)


def test_nonlinearity_is_visible_in_calibration():
    model = Toy(nonlinear=True)
    hidden = torch.tensor([[[1., 0.]]])
    features = torch.ones(1, 1)
    decoder = torch.tensor([[1., 0.]])
    mask = torch.ones(1, dtype=torch.bool)
    _, baseline, gradient = capture_gradient(model, {"x": hidden}, "layer", CANDIDATES)
    _, summary = project_attribution(gradient, features, decoder, {"image": mask})
    edited, _ = feature_removal(hidden, features, decoder, 0, mask, .5, torch.float32)
    actual = replace_residual(model, {"x": hidden}, "layer", edited)[0, 0, 0].item() - baseline[0, 0, 0].item()
    predicted = summary["image"]["effect"][0].item() * .5
    assert actual == -7 and predicted == -12
    score = study.agreement([{"actual_delta": actual, "predicted_delta": predicted}], .25)
    assert score["median_relative_error"] > .5


def test_unsupported_backward_removes_hook():
    class Unsupported(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()
        @staticmethod
        def backward(ctx, gradient):
            raise NotImplementedError("unsupported backward")
    class Broken(Toy):
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            result.logits = Unsupported.apply(result.logits)
            return result
    model = Broken()
    with pytest.raises(NotImplementedError, match="unsupported backward"):
        capture_gradient(model, {"x": torch.ones(1, 3, 2)}, "layer", CANDIDATES)
    assert not model.layer._forward_hooks and model.weight.grad is None


def test_masks_cast_rounding_and_inactive_features():
    masks = intervention_groups({"all": torch.tensor([False, True, True, True]),
                                "image": torch.tensor([False, True, False, False]),
                                "final": torch.tensor([False, False, False, True])})
    assert masks["other_non_image"].tolist() == [False, False, True, False]
    with pytest.raises(ValueError, match="partition"):
        intervention_groups({"all": torch.ones(2, dtype=torch.bool), "image": torch.ones(2, dtype=torch.bool),
                             "final": torch.tensor([False, True])})
    hidden = torch.ones(1, 4, 2)
    features = torch.zeros(4, 2)
    features[1, 0] = .001
    decoder = torch.ones(2, 2)
    edited, norms = feature_removal(hidden, features, decoder, 0, masks["image"], 1, torch.bfloat16)
    assert torch.equal(edited, hidden) and norms["requested_l2"] > 0 and norms["realized_l2"] == 0
    inactive, norms = feature_removal(hidden, features, decoder, 1, masks["image"], 1, torch.bfloat16)
    assert torch.equal(inactive, hidden) and norms["realized_l2"] == 0


def test_pilot_patient_selection_quota_and_feature_calibration_ties():
    records = []
    for category, n in {"TP": 10, "TN": 10, "FP": 8, "FN": 4}.items():
        for i in range(n):
            record = {"patient": f"{category}{i}", "case_id": f"{category}{i}_a", "label": int(category in {"TP", "FN"}),
                      "answer_score": {"decision": "Yes" if category in {"TP", "FP"} else "No"}}
            records.extend([record, {**record, "case_id": f"{category}{i}_b"}])
    chosen = study.choose_pilot(records)
    assert chosen == study.choose_pilot(list(reversed(records)))
    assert len(chosen["case_ids"]) == 20 and all(c.endswith("_a") for c in chosen["case_ids"])
    assert len(set(chosen["case_ids"])) == 20
    assert chosen["preflight"] == [chosen["allocation"]["TP"][0], chosen["allocation"]["TN"][0]]
    with pytest.raises(ValueError, match="quotas|prespecified"):
        study.choose_pilot([r for r in records if r["patient"] != "FN0"])
    summary = {"active": torch.ones(10, dtype=torch.bool), "effect": torch.zeros(10)}
    ids = study.calibration_features("case", "image", summary)
    assert ids[:2] == [0, 5] and len(set(ids)) == 3


def test_patient_weighting_rank_matching_and_bootstrap():
    records = [{"patient": "a", "outcome": "TP"}, {"patient": "a", "outcome": "FP"}, {"patient": "b", "outcome": "TN"}]
    payloads = []
    for value in (2., 4., -1.):
        payloads.append({"groups": {g: {"effect": torch.tensor([value, 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]),
                           "active": torch.ones(12, dtype=torch.bool), "removal_rms": torch.ones(12),
                           "removal_l2": torch.ones(12)} for g in GROUPS}})
    rows = study.patient_rankings(records, payloads)
    feature = next(r for r in rows if r["group"] == "image" and r["feature"] == 0)
    assert feature["mean_signed_effect"] == 1 and feature["mean_absolute_effect"] == 2
    pairs = study.matched_candidates(rows, ["image"])
    assert len(pairs) == 5 and len({p["comparison"]["feature"] for p in pairs}) == 5
    assert not {p["target"]["feature"] for p in pairs} & {p["comparison"]["feature"] for p in pairs}
    effects = [{"patient": r["patient"], "actual_delta": v, "predicted_delta": v,
                "kl_from_baseline": 0., "realized_l2": 1., "corrected": False, "introduced": False}
               for r, v in zip(records, (2., 4., -1.))]
    assert study.bootstrap_summary(effects)["actual_delta"]["mean"] == 1
    assert study.bootstrap_summary(effects) == study.bootstrap_summary(effects)


@pytest.fixture
def intervention_cache(discovery_cache, tmp_path):
    directory = tmp_path / "interventions"
    original = study.read_json(discovery_cache / "protocol.json")
    case_ids = [c["case_id"] for c in original["discovery_cases"]]
    protocol = {"source_directory": str(discovery_cache), "source_artifact_sha256": {
                    f"cases/{c}.json": file_sha256(discovery_cache / "cases" / f"{c}.json") for c in case_ids},
                "discovery_case_ids": case_ids, "pilot": {"case_ids": case_ids[:2]}, "sae_width": 4}
    protocol["protocol_hash"] = identity_hash(protocol)
    atomic_json(directory / "protocol.json", protocol)
    for case_id in case_ids:
        source, source_payload = load_discovery_case(discovery_cache, case_id)
        masks = {"image": torch.tensor([False, True, True, True, True, False]),
                 "final": torch.tensor([False, False, False, False, False, True]),
                 "other_non_image": torch.tensor([True, False, False, False, False, False])}
        hidden = torch.ones(1, 6, 2)
        attribution, groups = project_attribution(hidden, source_payload["features"], torch.ones(4, 2), masks)
        record = {"case_id": case_id, "patient": source["patient"], "tokens": source["tokens"],
                  "source_record_sha256": protocol["source_artifact_sha256"][f"cases/{case_id}.json"],
                  "pilot": True, "repeated_scores": [1.] * 4, "baseline_variation": 0., "outcome": "TP",
                  "positions": {k: m.nonzero().flatten().tolist() for k, m in masks.items()}}
        payload = {"hidden": hidden, "gradient": hidden, "baseline_logits": torch.ones(1, 1, 2),
                   "attribution": attribution, "groups": groups}
        study.save_case(directory, protocol, record, payload)
        if case_id not in protocol["pilot"]["case_ids"]:
            continue
        for group in GROUPS:
            for feature in study.calibration_features(case_id, group, groups[group]):
                for alpha in study.FRACTIONS:
                    study.save_treatment(study.treatment_path(directory, case_id, feature, group, alpha),
                        {"protocol_hash": protocol["protocol_hash"], "case_id": case_id, "feature": feature,
                         "group": group, "alpha": alpha, "predicted_delta": -1., "actual_delta": 0.})
    return directory, protocol, discovery_cache


def test_partial_gates_checksum_and_offline_regeneration(intervention_cache, monkeypatch):
    directory, protocol, source = intervention_cache
    def forbid(*args, **kwargs):
        raise AssertionError("No inference or SAE loading allowed")
    monkeypatch.setattr(study, "frozen_runtime", forbid)
    report = study.offline_report(directory, protocol)
    assert report["complete"] and report["status"] == "pilot_stopped"
    assert all(x["status"] == "insufficient_signal" for x in study.pilot_report(directory, protocol)["groups"].values())
    original = (directory / "intervention_report.json").read_bytes()
    study.offline_report(directory, protocol)
    assert original == (directory / "intervention_report.json").read_bytes()
    with pytest.raises(ValueError, match="qualifying"):
        study.cohort_selection(directory, protocol)
    with pytest.raises(ValueError, match="discovery"):
        study.load_case(directory, protocol, "confirmation_case")
    monkeypatch.setattr(study, "experiment_protocol", lambda *args: (protocol, []))
    # Completed pilot resume must not load the model.
    assert study.run("pilot", source, directory)["complete"]
    with pytest.raises(ValueError, match="qualify"):
        study.run("attribute", source, directory)
    tensor = directory / "cases" / "case0.pt"
    tensor.write_bytes(tensor.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        study.load_case(directory, protocol, "case0")


def test_signed_viewer_synthetic_artifacts(intervention_cache, monkeypatch):
    from medgemma_grounding import feature_viewer, inference, sae_reconstruction
    directory, protocol, source = intervention_cache
    def forbid(*args, **kwargs):
        raise AssertionError("Viewer must not load models")
    monkeypatch.setattr(inference, "load_model", forbid)
    monkeypatch.setattr(sae_reconstruction, "load_scope_sae", forbid)
    study.offline_report(directory, protocol)
    app = AppTest.from_string(
        "from pathlib import Path\nfrom medgemma_grounding.feature_viewer import run_intervention_viewer\n"
        f"run_intervention_viewer(Path({str(directory)!r}), Path({str(source)!r}))\n").run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.get("image")) == 2
    image = Image.new("RGB", (6, 2), "gray")
    overlay = np.asarray(signed_overlay(image, np.array([[-1., 0., 1.]]), 1))
    assert overlay[0, 0, 2] > overlay[0, 0, 0] and overlay[0, -1, 0] > overlay[0, -1, 2]
    assert np.array_equal(overlay[:, 2:4], np.asarray(image)[:, 2:4])


def test_identity_failure_stops_pilot_and_records_incomplete_report(intervention_cache, tmp_path, monkeypatch):
    _, protocol, source = intervention_cache
    directory = tmp_path / "failed_preflight"
    directory.mkdir()
    monkeypatch.setattr(study, "experiment_protocol", lambda *args: (protocol, []))
    runtime_calls, collected = [], []
    def runtime(*args):
        runtime_calls.append(True)
        return object(), object()
    def failed_collection(bundle, sae, protocol, record, payload, **kwargs):
        collected.append(record["case_id"])
        raise AssertionError("gradient forward mismatch")
    monkeypatch.setattr(study, "frozen_runtime", runtime)
    monkeypatch.setattr(study, "collect_attribution", failed_collection)
    with pytest.raises(AssertionError, match="gradient forward mismatch"):
        study.run("pilot", source, directory)
    assert runtime_calls == [True] and collected == protocol["pilot"]["case_ids"][:1]
    failure = study.read_json(directory / "failure_pilot.json")
    assert failure["settings_changed"] is False
    report = study.offline_report(directory, protocol)
    assert not report["complete"] and report["measured_treatments"] == 0
    assert report["failure_diagnostics"] == ["failure_pilot.json"]


def test_protocol_rejects_setting_drift_and_no_explicit_stage(tmp_path, monkeypatch):
    from medgemma_grounding.sae_reconstruction import checkpoint_spec, SAE_REVISION
    source = tmp_path / "source"
    source.mkdir()
    records = [{"patient": f"{group}{i}", "case_id": f"{group}{i}", "image_sha256": "hash", "label": int(group in {"TP", "FN"}),
                "answer_score": {"decision": "Yes" if group in {"TP", "FP"} else "No"}}
               for group, count in study.QUOTAS.items() for i in range(count)]
    original = {"protocol_hash": "source", "model_id": "model", "model_revision": "revision", "processor_revision": "revision",
                "prompt": "question", "sae_revision": SAE_REVISION, "sae_release": checkpoint_spec(14, all_layers=True)[0],
                "sae_id": checkpoint_spec(14, all_layers=True)[1], "sae_width": 16384,
                "hook": checkpoint_spec(14, all_layers=True)[3], "scoring": CANDIDATES}
    monkeypatch.setattr(study, "validate_source", lambda directory: (original, records, {}))
    output = tmp_path / "output"
    protocol, _ = study.experiment_protocol(source, output)
    assert study.experiment_protocol(source, output)[0] == protocol
    original["model_revision"] = "changed"
    with pytest.raises(ValueError, match="resume"):
        study.experiment_protocol(source, output)
    with pytest.raises(ValueError, match="separate"):
        study.experiment_protocol(source, source / "nested")
    with pytest.raises(ValueError, match="explicit"):
        study.run(None)


def test_all_stages_on_linear_model_with_qualifying_groups_and_resume(tmp_path, monkeypatch):
    """Exercise the conditional cohort path, independently of a real MPS result."""
    from medgemma_grounding.medgemma_features import answer_score

    source, directory = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    model = Toy()
    model.dtype = torch.float32
    decoder = torch.ones(16, 2)
    decoder[:, 1] = 0
    class Encoder:
        W_dec = decoder
        def encode(self, hidden):
            return hidden[:, :1].abs() * torch.linspace(.4, 2., 16)
        def decode(self, features):
            return features @ self.W_dec
    sae = Encoder()
    masks = {"image": torch.tensor([False, True, True, True, True, False]),
             "final": torch.tensor([False, False, False, False, False, True]),
             "other_non_image": torch.tensor([True, False, False, False, False, False])}
    tokens = [{"index": i, "id": i, "group": "image" if 1 <= i <= 4 else "text"} for i in range(6)]
    records, payloads, hidden_by_id = [], {}, {}
    for category, n in {"TP": 7, "TN": 7, "FP": 4, "FN": 4}.items():
        for i in range(n):
            case_id = f"{category}{i}"
            hidden = torch.full((1, 6, 2), (1 + i / 10) * (1 if category in {"TP", "FP"} else -1))
            hidden_by_id[case_id] = hidden
            record = {"case_id": case_id, "patient": case_id, "label": int(category in {"TP", "FN"}),
                      "tokens": tokens, "positions": {}, "spatial_layout": {},
                      "answer_score": answer_score(model(x=hidden).logits, CANDIDATES)}
            records.append(record)
            payloads[case_id] = {"features": sae.encode(hidden[0]).to_sparse().coalesce()}
            atomic_json(source / "cases" / f"{case_id}.json", record)
    protocol = {"source_directory": str(source), "scoring": CANDIDATES, "hook": "layer", "prompt": "fixed",
                "discovery_case_ids": [r["case_id"] for r in records], "pilot": study.choose_pilot(records),
                "sae_width": 16, "source_artifact_sha256": {f"cases/{r['case_id']}.json":
                    file_sha256(source / "cases" / f"{r['case_id']}.json") for r in records}}
    protocol["protocol_hash"] = identity_hash(protocol)
    atomic_json(directory / "protocol.json", protocol)
    atomic_json(source / "shortlist.json", {"candidates": [{"feature": 1}]})
    by_id = {r["case_id"]: r for r in records}
    def source_case(_directory, case_id, *args):
        assert case_id in by_id, "No confirmation/test access"
        return by_id[case_id], payloads[case_id]
    monkeypatch.setattr(study, "experiment_protocol", lambda *args: (protocol, records))
    monkeypatch.setattr(study, "load_discovery_case", source_case)
    monkeypatch.setattr(study, "prepared_inputs", lambda bundle, protocol, record:
                        ({"x": hidden_by_id[record["case_id"]]}, masks))
    calls = []
    def runtime(*args):
        calls.append(True)
        return SimpleNamespace(model=model), sae
    monkeypatch.setattr(study, "frozen_runtime", runtime)
    pilot = study.run("pilot", source, directory)
    assert pilot["complete"] and pilot["qualifying_groups"] == list(GROUPS)
    # FP32 subtraction can break exact ties in this decimal-valued cohort.
    assert all(g["half"]["spearman"] > .99 for g in pilot["groups"].values())
    assert all(r["actual_delta"] == pytest.approx(r["predicted_delta"], abs=1e-5)
               for r in pilot["measurements"])
    n_loads = len(calls)
    study.run("pilot", source, directory)
    assert len(calls) == n_loads
    with pytest.raises(ValueError, match="complete cohort"):
        study.run("select", source, directory)
    assert study.run("attribute", source, directory, limit=1)["new_cases"] == 1
    assert study.run("attribute", source, directory)["new_cases"] == 1
    candidates = study.run("select", source, directory)
    assert len(candidates["pairs"]) == 15
    frozen = (directory / "candidates.json").read_bytes()
    assert study.run("ablate", source, directory, limit=1)["new_cases"] == 1
    partial = study.run("report", source, directory)
    assert not partial["complete"] and partial["status"] == "partial"
    study.run("ablate", source, directory)
    n_loads = len(calls)
    study.run("ablate", source, directory)
    assert len(calls) == n_loads
    report = study.run("report", source, directory)
    assert report["complete"] and report["measured_treatments"] == 22 * 60
    assert report["status"] == "cohort_complete"
    assert len(report["summaries"]) == 60 and len(report["target_minus_comparison"]) == 30
    assert all(abs(s["overall"]["actual_delta"]["mean"] - s["overall"]["predicted_delta"]["mean"]) < 1e-4
               for s in report["summaries"])
    study.run("select", source, directory)
    assert (directory / "candidates.json").read_bytes() == frozen
