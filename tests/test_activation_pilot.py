"""Passive capture correctness and research artifact recovery/provenance."""

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from medgemma_grounding import activation_pilot as pilot
from medgemma_grounding import layer_comparison as analysis


def test_capture_uses_inherited_score_preserves_tuple_outputs_and_removes_hooks(monkeypatch):
    from medgemma_grounding import classification

    class Block(torch.nn.Module):
        def forward(self, x):
            return x + 1, "auxiliary"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

        def forward(self, x):
            value, auxiliary = self.block(x)
            assert auxiliary == "auxiliary"
            return SimpleNamespace(logits=value)

    model = Model()
    bundle = SimpleNamespace(torch=torch, model=model)
    x = torch.arange(12.).reshape(1, 3, 4)
    def scorer(bundle, image, prompt):
        assert prompt == "fixed"
        return bundle.model(image).logits[0, -1].softmax(-1)[0].item()
    monkeypatch.setattr(classification, "compute_yes_score", scorer)
    plain, _, plain_logits = pilot.capture_score(bundle, x, "fixed", [])
    score, values, logits = pilot.capture_score(bundle, x, "fixed", ["block"])
    assert score == plain
    np.testing.assert_array_equal(logits, plain_logits)
    np.testing.assert_array_equal(values[0], (x + 1)[0].numpy())
    assert not model._forward_hooks and not model.block._forward_hooks
    def fail(*args):
        raise RuntimeError("interrupted")
    monkeypatch.setattr(classification, "compute_yes_score", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        pilot.capture_score(bundle, x, "fixed", ["block"])
    assert not model._forward_hooks and not model.block._forward_hooks
    with pytest.raises(AttributeError):
        pilot.capture_score(bundle, x, "fixed", ["block", "missing"])
    assert not model.block._forward_hooks


def test_token_positions_preserve_images_and_last_input_not_generated_answer():
    tokenizer = SimpleNamespace(all_special_ids=[2, 9, 3], decode=lambda ids: "newline")
    bundle = SimpleNamespace(torch=torch, processor=SimpleNamespace(tokenizer=tokenizer),
                             model=SimpleNamespace(config=SimpleNamespace(image_token_index=9)))
    inputs = {"input_ids": torch.tensor([[2, 9, 9, 8, 3]]), "attention_mask": torch.ones(1, 5)}
    meta = pilot.input_metadata(bundle, inputs)
    assert meta["positions"] == {"image": [1, 2], "text": [3], "decision": [4]}
    assert meta["last_prefill_token_id"] == 3
    inputs["attention_mask"][0, -1] = 0
    with pytest.raises(ValueError, match="unpadded"):
        pilot.input_metadata(bundle, inputs)


def test_metrics_distinguish_scale_rotation_and_undefined_zero_norms():
    a = np.array([[1., 0.], [0., 1.]])
    scaled = analysis.distances(a, 2*a)
    assert scaled["relative_l2"] == pytest.approx(1)
    assert scaled["mean_token_cosine"] == pytest.approx(1)
    rotated = analysis.distances(a, a[:, ::-1])
    assert rotated["relative_l2"] == pytest.approx(np.sqrt(2))
    assert rotated["mean_token_cosine_distance"] == pytest.approx(1)
    zero = analysis.distances(np.zeros_like(a), a)
    assert zero["relative_l2"] is None and zero["mean_token_cosine"] is None
    assert zero["cosine_valid_tokens"] == 0
    with pytest.raises(ValueError, match="finite"):
        analysis.distances(a, np.full_like(a, np.nan))


def test_contrasts_keep_individual_controls_and_require_both_fills_for_candidates():
    rows = [{"layer": layer, "group": "decision", "fill": fill, "condition": name,
             "relative_l2": distance, "mean_token_cosine_distance": distance / 10}
            for layer in (0, 1) for fill in ("mean", "blur")
            for name, distance in zip(("target", "control_1", "control_2", "control_3"),
                                     ((.5 if fill == "mean" or layer == 0 else .1), .1, .2, .3))]
    contrast = analysis.contrasts(rows)
    first = next(r for r in contrast if r["layer"] == 0 and r["fill"] == "mean")
    assert first["controls_relative_l2"] == [.1, .2, .3]
    assert first["target_minus_mean_control_relative_l2"] == pytest.approx(.3)
    assert [r["layer"] for r in analysis.candidate_layers(contrast)["decision"]] == [0]
    with pytest.raises(ValueError, match="three controls"):
        analysis.contrasts(rows[:-1])


def test_collection_resume_analysis_and_corruption_detection(tmp_path, monkeypatch):
    from medgemma_grounding import classification, inference

    class Block(torch.nn.Module):
        def forward(self, value):
            return value + .01

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block() for _ in range(34)])
            self.dtype = torch.float32
            self.config = SimpleNamespace(_commit_hash="a"*40, _attn_implementation="test", image_token_index=9,
                                          text_config=SimpleNamespace(num_hidden_layers=34, hidden_size=2560))
            self.calls, self.interrupt = 0, None

        def get_submodule(self, name):
            return self.blocks[int(name.rsplit(".", 1)[1])]

        def forward(self, input_ids, attention_mask, pixel_values):
            self.calls += 1
            if self.calls == self.interrupt:
                raise RuntimeError("test interruption")
            hidden = (pixel_values.mean() / 255).expand(1, 4, 2560).clone()
            for block in self.blocks:
                hidden = block(hidden)
            return SimpleNamespace(logits=torch.stack((hidden[..., 0], hidden[..., 0]*2), -1))

    model = Model()
    tokenizer = SimpleNamespace(all_special_ids=[2, 9, 3], decode=lambda ids: "text",
                                encode=lambda answer, **kwargs: [int(answer == "Yes")])
    bundle = SimpleNamespace(model=model, torch=torch, processor=SimpleNamespace(tokenizer=tokenizer))
    def inputs(bundle, image, prompt):
        return {"input_ids": torch.tensor([[2, 9, 3, 4]]), "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "pixel_values": torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()}
    monkeypatch.setattr(inference, "prepare_inputs", inputs)
    monkeypatch.setattr(classification, "prepare_inputs", inputs)
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(pilot, "check_candidate_tokens", lambda *args: {"candidate_ids": {"No": 0, "Yes": 1}})
    monkeypatch.setattr(pilot, "configure_cache", lambda *args: None)
    prepared, occlusion = tmp_path / "prepared", tmp_path / "occlusion"
    prepared.mkdir()
    occlusion.mkdir()
    (prepared / "manifest.json").write_text("{}")
    (occlusion / "scores.jsonl").write_text("fixture")
    image = Image.fromarray(np.full((4, 4, 3), 100, dtype=np.uint8))
    image.save(prepared / "original.png")
    tasks = [{"condition": "original", "fill": "none", "kind": "baseline"},
             {"condition": "empty_mask", "fill": "none", "kind": "sanity"}]
    tasks += [{"condition": name, "fill": fill, "kind": "region"}
              for fill in ("mean", "blur") for name in ("target", "control_1", "control_2", "control_3")]
    tasks += [{"condition": "original_repeat", "fill": "none", "kind": "sanity"}]
    prior = []
    for index, task in enumerate(tasks):
        variant = image if task["kind"] != "region" else Image.fromarray(np.full((4, 4, 3), 100-index, dtype=np.uint8))
        if task["kind"] == "region":
            variant.save(prepared / f"{pilot.condition_id(task)}.png")
        prior.append(dict(task, yes_score=classification.compute_yes_score(bundle, variant, "fixed"),
                          response="Yes", prediction=1,
                          pixel_tensor=pilot.tensor_record(inputs(bundle, variant, "fixed")["pixel_values"], torch)))
    manifest = {"case": {"case_id": "fixture"}, "part1_contract": {"prompt": "fixed"},
                "baseline_tensors": {k: pilot.tensor_record(v, torch) for k, v in inputs(bundle, image, "fixed").items()},
                "runtime": {"model_id": "test", "model_revision": "a"*40, "device": "cpu", "dtype": "torch.float32",
                            "attention_implementation": "test", "packages": {}}}
    monkeypatch.setattr(pilot, "verified_occlusion", lambda *args: (manifest, prior))
    args = SimpleNamespace(prepared=prepared, occlusion=occlusion, output=tmp_path / "captures",
                           hf_home=tmp_path / "cache", resume=False)
    model.calls, model.interrupt = 0, 5
    with pytest.raises(RuntimeError, match="interruption"):
        pilot.run_collection(args)
    assert len(pilot.verify_collection(args.output)[1]) == 2
    assert not model._forward_hooks and all(not block._forward_hooks for block in model.blocks)
    model.interrupt = None
    args.resume = True
    summary = pilot.run_collection(args)
    assert summary["conditions_completed"] == 11
    assert summary["max_passive_logit_difference"] == 0
    assert summary["sanity_max_activation_differences"] == {"none_empty_mask": 0, "none_original_repeat": 0}
    calls = model.calls
    assert pilot.run_collection(args) == summary
    assert model.calls == calls  # Completed resume does not load/run another model.
    report = analysis.analyze(args.output, tmp_path / "analysis")
    assert report["groups"] == {"decision": 1, "image": 1, "text": 1}
    assert report["sanity_max_l2"] == 0
    assert analysis.analyze(args.output, tmp_path / "analysis") == report
    assert len((tmp_path / "analysis" / "layer_metrics.csv").read_text().splitlines()) == 817
    artifact = args.output / "none_original" / "next_token_logits.npy"
    artifact.write_bytes(artifact.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="artifact changed"):
        pilot.verify_collection(args.output)
