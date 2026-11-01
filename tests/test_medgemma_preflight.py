"""Bounded diagnosis, structured failures and explicit attention configuration."""

from types import SimpleNamespace

import pytest
import torch

from medgemma_grounding import inference, medgemma_preflight as preflight
from medgemma_grounding.feature_interventions import (
    attention_settings, configure_attention, tensor_diagnostics,
)
from medgemma_grounding.medgemma_features import atomic_json, file_sha256, identity_hash
from medgemma_grounding.medgemma_interventions import IdentityCheckError, identity_control


class ModeSensitiveLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, value):
        offset = .125 if self.config._attn_implementation == "sdpa" and value.requires_grad and torch.is_grad_enabled() else 0.
        return value + offset


class ModeModel(torch.nn.Module):
    """Emulate a dispatch difference to test detection and conditional repair."""
    def __init__(self, tuple_output=False):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="sdpa",
            text_config=SimpleNamespace(_attn_implementation="sdpa"),
            vision_config=SimpleNamespace(_attn_implementation="sdpa"))
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(15)] + [ModeSensitiveLayer(self.config.text_config)])
        self.weight = torch.nn.Parameter(torch.tensor([2., 1.]), requires_grad=False)
        self.dtype = torch.float32
        self.eval()

    def set_attn_implementation(self, value):
        mapping = {k: value for k in attention_settings(self)} if isinstance(value, str) else value
        for key, backend in mapping.items():
            config = self.config if key == "" else getattr(self.config, key)
            config._attn_implementation = backend

    def forward(self, x, use_cache=False, logits_to_keep=1):
        assert not use_cache and logits_to_keep == 1
        for layer in self.model.language_model.layers:
            x = layer(x)
        score = (x * self.weight).sum()
        return SimpleNamespace(logits=torch.stack([score, score * 0]).reshape(1, 1, 2))


HOOK = "model.language_model.layers.14"
CANDIDATES = {"ids": {"Yes": 0, "No": 1}}


def test_diagnostics_keep_tolerance_scores_and_nonfinite_details():
    baseline = torch.tensor([[[2., -1.]]])
    changed = baseline + torch.tensor([[[.125, 0.]]])
    with pytest.raises(IdentityCheckError) as error:
        identity_control(baseline, changed, CANDIDATES, "gradient_forward")
    d = error.value.diagnostics
    assert d["control"] == "gradient_forward" and d["max_logit_difference"] == .125
    assert d["mismatched_elements"] == 1 and d["score_difference"] == .125
    assert d["decision_preserved"] and d["rtol"] == d["atol"] == 1e-5
    assert not tensor_diagnostics(baseline, baseline * float("nan"))["finite"]
    assert identity_control(baseline, baseline.clone(), CANDIDATES)["allclose"]


def test_mode_localization_and_supported_attention_fix():
    model = ModeModel()
    parameters = preflight.weight_identity(model)
    inputs = {"x": torch.ones(1, 3, 2)}
    outputs = {mode: preflight.capture_mode(model, inputs, HOOK, CANDIDATES, mode)[0]
               for mode in preflight.MODES}
    comparisons = {mode: preflight.mode_comparison(outputs["inference"], outputs[mode], CANDIDATES, mode)
                   for mode in preflight.MODES[1:]}
    traces = {"inference": {"aten::_scaled_dot_product_attention_math_for_mps": 60},
              "gradient": {"aten::_scaled_dot_product_attention_math_for_mps": 41,
                           "aten::_scaled_dot_product_attention_math": 19}}
    evidence = preflight.backend_evidence(comparisons, traces)
    assert evidence["confirmed"] and evidence["boundary_same"] and evidence["downstream_changed"]
    assert not preflight.backend_evidence(comparisons, {"inference": {}, "gradient": {}})["confirmed"]
    configure_attention(model, preflight.FIX)
    for mode in preflight.MODES:
        output, _ = preflight.capture_mode(model, inputs, HOOK, CANDIDATES, mode)
        assert preflight.mode_comparison(outputs["inference"], output, CANDIDATES, mode)["logits"]["allclose"]
    assert preflight.weight_identity(model) == parameters
    assert all(p.grad is None for p in model.parameters())
    assert not model.get_submodule(HOOK)._forward_hooks
    assert not model.get_submodule("model.language_model.layers.15")._forward_hooks
    with pytest.raises(ValueError, match="Unsupported"):
        configure_attention(model, {"unknown": "eager"})
    model.set_attn_implementation = lambda setting: None
    with pytest.raises(ValueError, match="consistently"):
        configure_attention(model, "sdpa")


def test_cache_compatibility_is_independent_of_internal_consistency():
    old = torch.tensor([[1., 0., 2.], [0., 3., 0.]])
    changed = torch.tensor([[1., 1., 2.], [0., 0., 0.]])
    masks = {"image": torch.tensor([True, False]), "final": torch.tensor([False, True])}
    d = preflight.feature_compatibility(old, changed, masks)
    assert not d["all"]["allclose"] and d["activated_entries"] == d["deactivated_entries"] == 1
    assert d["all"]["max_absolute_difference"] == 3


def test_repaired_controls_can_pass_with_an_incompatible_old_cache():
    model = ModeModel()
    configure_attention(model, preflight.FIX)
    inputs = {"x": torch.ones(1, 3, 2)}
    baseline, _ = preflight.capture_mode(model, inputs, HOOK, CANDIDATES, "inference")
    class Encoder:
        W_dec = torch.eye(2)
        def encode(self, hidden):
            return hidden.relu()
        def decode(self, features):
            return features @ self.W_dec
    masks = {"image": torch.tensor([True, False, False]), "final": torch.tensor([False, False, True]),
             "other_non_image": torch.tensor([False, True, False])}
    result, _ = preflight.verify_consistent(SimpleNamespace(model=model), Encoder(),
        {"hook": HOOK, "scoring": CANDIDATES}, {}, masks, inputs, {"inference": baseline}, torch.zeros(3, 2))
    assert result["passed"] and not result["discovery_feature_cache_compatibility"]["all"]["allclose"]
    assert len(result["verification_edits"]) == 9
    assert all(e["realized_l2"] > 0 for e in result["verification_edits"] if e["alpha"])


def test_mode_capture_cleans_both_hooks_on_failure():
    model = ModeModel()
    after = model.get_submodule("model.language_model.layers.15")
    def fail(*args):
        raise RuntimeError("diagnostic failure")
    handle = after.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="diagnostic failure"):
            preflight.capture_mode(model, {"x": torch.ones(1, 3, 2)}, HOOK, CANDIDATES, "gradient")
        assert not model.get_submodule(HOOK)._forward_hooks and not after._forward_hooks
    finally:
        handle.remove()


@pytest.mark.parametrize("attention", [None, {"text_config": "eager", "vision_config": "sdpa"}])
def test_loader_forwards_attention_only_when_explicit(monkeypatch, attention):
    calls = []
    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(kwargs)
            return cls()
        def to(self, device):
            return self
        def eval(self):
            return self
    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert "attn_implementation" not in kwargs
            return cls()
    monkeypatch.setattr(inference, "_load_dependencies", lambda: (torch, Model, Processor))
    inference.load_model("model", device="cpu", token="unit-test", revision="pinned", attn_implementation=attention)
    assert calls[0]["dtype"] == torch.float32 and calls[0]["revision"] == "pinned"
    assert ("attn_implementation" in calls[0]) == (attention is not None)
    if attention is not None:
        assert calls[0]["attn_implementation"] == attention


def source_fixture(tmp_path, monkeypatch):
    source, failed = tmp_path / "discovery", tmp_path / "interventions_v1"
    source.mkdir()
    source_protocol = {"schema": "source"}
    source_protocol["protocol_hash"] = identity_hash(source_protocol)
    atomic_json(source / "protocol.json", source_protocol)
    records, hashes = {}, {}
    for case_id in ("tp", "tn"):
        record = {"case_id": case_id, "partition": "discovery", "image_path": f"/data/CheXpert/val/{case_id}/x.jpg",
                  "image_sha256": f"{case_id}-image"}
        records[case_id] = record
        atomic_json(source / "cases" / f"{case_id}.json", record)
        torch.save({"unit": torch.ones(1)}, source / "cases" / f"{case_id}.pt")
        for extension in ("json", "pt"):
            name = f"cases/{case_id}.{extension}"
            hashes[name] = file_sha256(source / name)
    previous = {"source_protocol_hash": source_protocol["protocol_hash"], "source_directory": str(source),
                "pilot": {"preflight": ["tp", "tn"], "allocation": {"TP": ["tp"], "TN": ["tn"]}},
                "source_artifact_sha256": hashes,
                **{k: "pinned" for k in ("model_id", "model_revision", "processor_revision", "sae_revision", "sae_release", "sae_id", "hook", "prompt")},
                "scoring": CANDIDATES}
    previous["protocol_hash"] = identity_hash(previous)
    atomic_json(failed / "protocol.json", previous)
    accesses = []
    def load(directory, case_id, *args):
        assert case_id in {"tp", "tn"}
        accesses.append(case_id)
        return records[case_id], {}
    monkeypatch.setattr(preflight, "load_discovery_case", load)
    return source, failed, accesses


def test_two_case_boundary_immutable_sources_and_offline_report(tmp_path, monkeypatch):
    source, failed, accesses = source_fixture(tmp_path, monkeypatch)
    output = tmp_path / "new_preflight"
    protocol, _ = preflight.protocol_for(source, failed, output)
    assert accesses == ["tp", "tn"]
    assert preflight.protocol_for(source, failed, output)[0] == protocol
    with pytest.raises(ValueError, match="separate"):
        preflight.protocol_for(source, failed, failed)
    report = preflight.run(source, failed, output, report_only=True)
    assert not report["complete"] and not report["pilot_or_cohort_authorized"]
    with pytest.raises(ValueError, match="two preflight"):
        preflight.load_case(output, protocol, "confirmation")
    # Changes to original sources are rejected, never written into v1.
    atomic_json(failed / "unexpected.json", {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        preflight.report(output, protocol)
    with pytest.raises(ValueError, match="resume"):
        preflight.protocol_for(source, failed, output)


def test_completed_two_case_run_resumes_without_a_model(tmp_path, monkeypatch):
    source, failed, _ = source_fixture(tmp_path, monkeypatch)
    output = tmp_path / "finished"
    protocol, _ = preflight.protocol_for(source, failed, output)
    for case_id in protocol["case_ids"]:
        preflight.save_case(output, protocol, {"case_id": case_id, "consistent": {"passed": True},
                            "backend_evidence": {"confirmed": True}}, {"dummy": torch.ones(1)})
    atomic_json(output / "runtime_checks.json", {"protocol_hash": protocol["protocol_hash"],
        "parameter_identity_versions_unchanged": True, "parameter_gradients_unset": True})
    before = preflight.report(output, protocol)
    def forbid(*args, **kwargs):
        raise AssertionError("Completed preflight must not load a model")
    monkeypatch.setattr(inference, "load_model", forbid)
    assert preflight.run(source, failed, output) == before
    path = output / "tp.pt"
    path.write_bytes(path.read_bytes() + b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        preflight.load_case(output, protocol, "tp")
