import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pycocotools import mask as coco
import pytest

from medgemma_grounding import masking
from medgemma_grounding import occlusion_pilot as pilot


def test_official_rle_keeps_orientation_and_rejects_wrong_dimensions(tmp_path):
    expected = np.zeros((5, 9), dtype=np.uint8)
    expected[1:4, 5:8] = 1
    encoded = coco.encode(np.asfortranarray(expected))
    encoded["counts"] = encoded["counts"].decode()
    path = tmp_path / "masks.json"
    path.write_text(json.dumps({"case": {"Pleural Effusion": encoded}}))
    actual = masking.load_effusion_mask(path, "case", (5, 9))
    np.testing.assert_array_equal(actual, expected.astype(bool))
    with pytest.raises(ValueError, match="dimension"):
        masking.load_effusion_mask(path, "case", (9, 5))
    with pytest.raises(ValueError, match="Missing"):
        masking.load_effusion_mask(path, "unknown", (5, 9))


def test_translated_masks_preserve_exact_shape_area_without_wrap():
    target = np.zeros((20, 30), dtype=bool)
    target[12:15, 20:25] = True
    target[10:15, 24] = True
    controls = masking.control_masks(target, [[-.5, 0], [-.5, -.4], [0, -.4]], [0, 0, 1, 1])
    for control in controls:
        assert control["mask"].sum() == target.sum()
        assert not (control["mask"] & target).any()
        np.testing.assert_array_equal(masking.translate_mask(control["mask"], -control["dx"], -control["dy"]), target)
    with pytest.raises(ValueError, match="clip"):
        masking.translate_mask(target, 100, 0)
    with pytest.raises(ValueError, match="overlaps"):
        masking.control_masks(target, [[0, 0], [-.5, 0], [0, -.4]], [0, 0, 1, 1])
    with pytest.raises(ValueError, match="chest window"):
        masking.control_masks(target, [[-.5, 0], [-.5, -.4], [0, -.4]], [.6, 0, 1, 1])


def test_fill_is_fixed_and_outside_pixels_are_unchanged():
    original = np.random.default_rng(4).integers(0, 256, (25, 31, 3), dtype=np.uint8)
    backup = original.copy()
    replacements, settings = masking.replacement_images(Image.fromarray(original), (16, 16), 2)
    assert np.unique(replacements["mean"]).tolist() == [round(original.mean())]
    assert settings["blur_radius_native_xy"] == [2 * 31 / 16, 2 * 25 / 16]
    mask = np.zeros(original.shape[:2], dtype=bool)
    mask[6:14, 10:20] = True
    for replacement in replacements.values():
        result = masking.apply_mask(original, mask, replacement)
        np.testing.assert_array_equal(result[~mask], original[~mask])
        np.testing.assert_array_equal(result[mask], replacement[mask])
        np.testing.assert_array_equal(masking.apply_mask(original, np.zeros_like(mask), replacement), original)
    np.testing.assert_array_equal(original, backup)


def test_grid_covers_every_pixel_once_even_for_odd_dimensions():
    shape = (23, 37)
    coverage = np.zeros(shape, dtype=int)
    patches = masking.grid_patches(shape, 8)
    assert len(patches) == 64
    for patch in patches:
        coverage += masking.rectangle_mask(shape, patch["bbox_xyxy"])
    np.testing.assert_array_equal(coverage, np.ones(shape))


def test_signed_maps_and_area_weighted_mass_have_defined_zero_behavior():
    target = np.zeros((4, 4), dtype=bool)
    target[:2, :2] = True
    rows = []
    for fill, scores in (("mean", [.4, .7, .8, .8]), ("blur", [.9, .9, .9, .9])):
        for patch, score in zip(masking.grid_patches(target.shape, 2), scores):
            rows.append(dict(patch, kind="patch", fill=fill, yes_score=score))
    summaries, maps = masking.summarize_patches(rows, target, .8, 2)
    np.testing.assert_allclose(maps["mean"], [[.4, .1], [0, 0]])
    assert summaries["mean"]["positive_drop_mass_in_mask"] == pytest.approx(.8)
    assert summaries["mean"]["mass_enrichment_over_area"] == pytest.approx(3.2)
    assert summaries["blur"]["positive_drop_mass_in_mask"] is None
    assert summaries["blur"]["negative_patches"] == 4
    assert (maps["blur"] < 0).all()


def test_region_comparison_uses_all_three_controls_and_retains_negative_drops():
    rows = [dict(condition=name, fill=fill, yes_score=score) for fill in ("mean", "blur")
            for name, score in zip(("target", "control_1", "control_2", "control_3"), (.5, .7, .9, .8))]
    summary = masking.summarize_regions(rows, .8)
    assert summary["mean"]["target_drop"] == pytest.approx(.3)
    assert summary["mean"]["control_drops"] == pytest.approx([.1, -.1, 0])
    assert summary["mean"]["target_minus_mean_control_drop"] == pytest.approx(.3)


@pytest.mark.parametrize("problem", ["changed_pixels", "baseline_drift", "repeat_drift"])
def test_sanity_failures_cannot_be_finalized(problem):
    rows = [{"kind": "baseline", "condition": "original", "yes_score": .8, "changed_model_pixels": 0},
            {"kind": "sanity", "condition": "empty_mask", "yes_score": .8, "changed_model_pixels": 0}]
    if problem == "changed_pixels":
        rows[1]["changed_model_pixels"] = 1
    elif problem == "baseline_drift":
        rows[0]["yes_score"] = .7
    else:
        rows[1]["yes_score"] = .7
    with pytest.raises(ValueError, match="check failed"):
        pilot.check_sanity(rows, .8)


def test_checkpoint_rejects_wrong_order_or_nonfinite_scores(tmp_path):
    path = tmp_path / "scores.jsonl"
    tasks = [{"condition": "original", "kind": "baseline", "fill": "none"}]
    path.write_text(json.dumps(dict(tasks[0], yes_score=.8)) + "\n")
    assert len(pilot.load_completed(path, tasks)) == 1
    path.write_text(json.dumps(dict(tasks[0], condition="wrong", yes_score=.8)) + "\n")
    with pytest.raises(ValueError, match="order"):
        pilot.load_completed(path, tasks)
    path.write_text(json.dumps(dict(tasks[0], yes_score=float("nan"))) + "\n")
    with pytest.raises(ValueError, match="Invalid saved"):
        pilot.load_completed(path, tasks)


def test_run_resume_uses_inherited_functions_and_does_not_repeat_completed_conditions(tmp_path, monkeypatch):
    import torch
    from medgemma_grounding import classification, inference

    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    image = Image.fromarray(pixels)
    image.save(prepared / "original.png")
    image.save(prepared / "model_original.png")
    target = np.zeros((8, 8), dtype=bool)
    target[5:7, 5:7] = True
    masks = {"target": target, "control_1": masking.translate_mask(target, -4, 0),
             "control_2": masking.translate_mask(target, -4, -4), "control_3": masking.translate_mask(target, 0, -4)}
    replacements, fill_settings = masking.replacement_images(image, (8, 8), 2)
    for name, mask in masks.items():
        pilot.save_mask(prepared / f"mask_{name}.png", mask)
        for fill, replacement in replacements.items():
            Image.fromarray(masking.apply_mask(pixels, mask, replacement)).save(prepared / f"{fill}_{name}.png")
    calls = {"score": 0, "generate": 0, "interrupt": True}
    def inputs(bundle, img, prompt):
        assert prompt == classification.PROMPT
        return {"input_ids": torch.tensor([[10, 11]]),
                "pixel_values": torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()}
    def score(bundle, img, prompt):
        calls["score"] += 1
        if calls["interrupt"] and calls["score"] == 6:
            raise RuntimeError("test interruption")
        assert prompt == classification.PROMPT
        return .9 - np.count_nonzero(np.asarray(img) != pixels) / pixels.size * .2
    def generate(bundle, img, prompt, max_new_tokens):
        calls["generate"] += 1
        assert max_new_tokens == classification.MAX_NEW_TOKENS
        if calls["generate"] == 2:
            raise RuntimeError("MedGemma generated only special or padding tokens. Check dtype.")
        return "Yes"
    revision = "a" * 40
    bundle = SimpleNamespace(model=SimpleNamespace(dtype=torch.float32,
                             config=SimpleNamespace(_commit_hash=revision, _attn_implementation="test")))
    manifest = {"case": {"case_id": "case"}, "config": {"fills": ["mean", "blur"], "grid": 2, "blur_sigma_model_pixels": 2},
                "regions": [{"name": name} for name in masks], "patches": masking.grid_patches(target.shape, 2),
                "fill_settings": fill_settings, "model_size_width_height": [8, 8], "baseline_yes_score": .9,
                "part1_contract": {"prompt": classification.PROMPT, "max_new_tokens": classification.MAX_NEW_TOKENS},
                "runtime": {"model_id": "test", "model_revision": revision, "dtype": "torch.float32", "device": "cpu",
                            "attention_implementation": "test", "packages": {}},
                "baseline_tensors": {k: pilot.tensor_record(v, torch) for k, v in inputs(bundle, image, classification.PROMPT).items()}}
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(pilot, "verify_prepared", lambda path: manifest)
    monkeypatch.setattr(pilot, "check_candidate_tokens", lambda *args: {})
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(inference, "prepare_inputs", inputs)
    monkeypatch.setattr(inference, "generate", generate)
    monkeypatch.setattr(classification, "compute_yes_score", score)
    args = SimpleNamespace(prepared=prepared, output=tmp_path / "run", hf_home=tmp_path / "cache",
                           review_note="Synthetic fixture reviewed", resume=False)
    with pytest.raises(RuntimeError, match="interruption"):
        pilot.run(args)
    first = (args.output / "scores.jsonl").read_text()
    assert len(first.splitlines()) == 5
    calls["interrupt"] = False
    args.resume = True
    summary = pilot.run(args)
    assert (args.output / "scores.jsonl").read_text().startswith(first)
    assert summary["conditions_completed"] == 19
    assert summary["sanity"] == {"no_op_absolute_difference": 0, "end_repeat_absolute_difference": 0}
    assert calls["score"] == 21  # 19 completed, one interrupted, one resume-baseline check.
    assert calls["generate"] == 9
    saved_rows = [json.loads(line) for line in (args.output / "scores.jsonl").read_text().splitlines()]
    assert saved_rows[2]["response"] == ""
    assert saved_rows[2]["prediction"] is None
    assert saved_rows[2]["generation_status"] == "unparsed_empty_generation"
    assert saved_rows[2]["yes_score"] > 0  # Its continuous score is still retained.
    assert (args.output / "results.png").is_file()
    before = calls.copy()
    pilot.run(args)
    assert calls == before  # A completed resume only regenerates the report.
