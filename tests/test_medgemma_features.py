"""Feature-cache, spatial mapping and viewer checks without model downloads."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from streamlit.testing.v1 import AppTest

from medgemma_grounding import feature_viewer
from medgemma_grounding.feature_viewer import (
    case_scores, color_scale, feature_catalog, feature_values, image_map,
    make_overlay, token_highlights,
)
from medgemma_grounding.medgemma_features import file_sha256, load_cache, save_cache, spatial_layout


@pytest.fixture
def cache(tmp_path):
    cases, activations = [], {}
    for case_id, label in (("positive", 1), ("negative", 0)):
        path = tmp_path / f"{case_id}.png"
        Image.new("RGB", (17, 11)).save(path)
        tokens = [
            {"index": i, "id": i, "piece": "\n\n" if i == 5 else str(i), "text": "<script>" if i == 5 else str(i),
             "group": "special" if i == 0 else "text" if i == 5 else "image"}
            for i in range(6)
        ]
        dense = torch.zeros(6, 4)
        dense[0, 0], dense[0, 3] = 100, 50
        dense[1:5, 0] = torch.tensor([4., 0., 2., 1.] if label else [1., 0., 0., 0.])
        dense[5, 0], dense[5, 1] = (3, 7) if label else (8, 0)
        cases.append({
            "case_id": case_id, "label": label, "answer": "Yes" if label else "No",
            "image_path": str(path), "image_sha256": file_sha256(path), "tokens": tokens,
            "feature_shape": [6, 4],
            "spatial_layout": {"verified": True, "mapping": "direct_resize_row_major", "grid": [2, 2]},
        })
        activations[case_id] = dense.to_sparse().coalesce()
    directory = tmp_path / "features"
    save_cache(directory, {"schema_version": 1, "project_root": str(tmp_path), "sae_width": 4,
                           "cases": cases}, activations)
    return directory


def test_sparse_roundtrip_ranking_and_zeros(cache):
    manifest, activations = load_cache(cache)
    values = feature_values(activations, 0)
    np.testing.assert_array_equal(values["positive"], [100, 4, 0, 2, 1, 3])
    assert [row["feature"] for row in feature_catalog(manifest, activations)] == [0, 1]
    rows = case_scores(manifest, values)
    assert [r["case_id"] for r in rows] == ["positive", "negative"]
    assert rows[0]["image_mean"] == 1.75
    assert rows[0]["image_active_fraction"] == .75
    assert rows[1]["expert_effusion"] == "absent"
    assert case_scores(manifest, values, "text_max")[0]["case_id"] == "negative"
    assert color_scale(manifest, values) == 8  # special-token activation 100 excluded
    zeros = feature_values(activations, 2)
    assert len(case_scores(manifest, zeros)) == 2
    assert color_scale(manifest, zeros) == 0
    torch.testing.assert_close(activations["positive"].to_dense()[:, 0], torch.from_numpy(values["positive"]))


def test_cache_rejects_misalignment_and_changed_images(cache):
    manifest, activations = load_cache(cache)
    activations["positive"] = torch.zeros(5, 4).to_sparse()
    save_cache(cache, manifest, activations)
    with pytest.raises(ValueError, match="alignment"):
        load_cache(cache)
    activations["positive"] = torch.zeros(6, 4).to_sparse()
    save_cache(cache, manifest, activations)
    Image.new("RGB", (17, 11), "white").save(manifest["cases"][0]["image_path"])
    with pytest.raises(ValueError, match="Image changed"):
        load_cache(cache)


def test_cache_detects_interrupted_replacement(cache):
    path = cache / "activations.pt"
    path.write_bytes(path.read_bytes() + b"interrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_cache(cache)


def test_spatial_mapping_and_overlay_orientation(cache):
    manifest, activations = load_cache(cache)
    case = manifest["cases"][0]
    heat = image_map(case, feature_values(activations, 0)["positive"])
    np.testing.assert_array_equal(heat, [[4, 0], [2, 1]])
    image = Image.new("RGB", (17, 11))
    result = np.asarray(make_overlay(image, np.array([[0., 4.], [0., 0.]]), 4))
    assert result.shape == (11, 17, 3)
    assert result[0, -1].max() > 0
    assert not result[0, 0].any() and not result[-1, -1].any()
    assert not result[-1, 0].any()
    assert not np.array_equal(np.asarray(make_overlay(image, np.ones((2, 2)), 4)),
                              np.asarray(make_overlay(image, np.ones((2, 2)) * 2, 4)))
    np.testing.assert_array_equal(make_overlay(image, heat, 0), image)
    case["spatial_layout"]["verified"] = False
    assert image_map(case, feature_values(activations, 0)["positive"]) is None


def test_gemma_projector_pooling_preserves_row_major_order():
    from transformers import Gemma3Config
    from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

    cfg = Gemma3Config(vision_config={"image_size": 896, "patch_size": 14, "hidden_size": 16},
                      text_config={"hidden_size": 16}, mm_tokens_per_image=256)
    projector = Gemma3MultiModalProjector(cfg)
    markers = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64)
    result = projector.avg_pool(markers).flatten(2).transpose(1, 2).squeeze()
    expected = torch.tensor([markers[0, 0, r*4:r*4+4, c*4:c*4+4].mean()
                             for r in range(16) for c in range(16)])
    torch.testing.assert_close(result, expected)


def test_spatial_layout_rejects_crops():
    processor = type("Gemma3ImageProcessorPil", (), {"do_resize": True, "do_pan_and_scan": False, "resample": 2})()
    config = SimpleNamespace(mm_tokens_per_image=4, image_token_index=99,
                             vision_config=SimpleNamespace(image_size=16, patch_size=2))
    bundle = SimpleNamespace(model=SimpleNamespace(config=config),
                             processor=SimpleNamespace(image_processor=processor))
    inputs = {"pixel_values": torch.zeros(1, 3, 16, 16), "input_ids": torch.tensor([[99] * 4])}
    layout = spatial_layout(bundle, inputs, (17, 11))
    assert layout["verified"] and layout["grid"] == [2, 2]
    assert layout["pool_kernel"] == 4
    inputs["pixel_values"] = torch.zeros(2, 3, 16, 16)
    assert not spatial_layout(bundle, inputs, (17, 11))["verified"]


def test_viewer_switches_cases_features_and_escapes_tokens(cache, monkeypatch):
    from medgemma_grounding import inference, sae_reconstruction

    def forbid_inference(*args, **kwargs):
        raise AssertionError("Browsing must not load or run a model")

    monkeypatch.setattr(inference, "load_model", forbid_inference)
    monkeypatch.setattr(inference, "generate", forbid_inference)
    monkeypatch.setattr(sae_reconstruction, "load_scope_sae", forbid_inference)
    app = AppTest.from_file(str(Path(feature_viewer.__file__))).run()
    app.sidebar.text_input[0].set_value(str(cache)).run()
    assert not app.exception
    assert len(app.dataframe[1].value) == 2
    assert app.selectbox(key="case_id").value == "positive"
    app.selectbox(key="case_id").select("negative").run()
    assert not app.exception
    assert any("MedGemma answer: No" in m.value for m in app.markdown)
    app.sidebar.number_input[0].set_value(2).run()
    assert not app.exception
    assert any("inactive" in message.value for message in app.info)
    manifest, activations = load_cache(cache)
    highlighted = token_highlights(manifest["cases"][0]["tokens"], feature_values(activations, 0)["positive"], 8)
    assert "&lt;script&gt;" in highlighted and "<script>" not in highlighted
    assert "\n" not in highlighted  # attribute newlines must not break Markdown rendering
    # The image must still match after Streamlit has cached the tensors.
    Image.new("RGB", (17, 11), "white").save(manifest["cases"][0]["image_path"])
    app.sidebar.number_input[0].set_value(0).run()
    assert not app.exception
    assert any("Image changed" in message.value for message in app.error)


def test_viewer_reports_missing_cache(tmp_path):
    app = AppTest.from_file(str(Path(feature_viewer.__file__))).run()
    app.sidebar.text_input[0].set_value(str(tmp_path / "missing")).run()
    assert not app.exception and app.error
