"""Pooled layer metrics, checkpoint selection, and safe resume checks."""

import pytest
import torch

from medgemma_grounding.medgemma_scope_layers import (
    ReconstructionTotals,
    load_profile,
    parse_layers,
    save_profile,
)
from medgemma_grounding.sae_reconstruction import SAE_RELEASE_ALL, checkpoint_spec


def test_pooled_fvu_and_l0_weight_tokens_across_cases():
    totals = ReconstructionTotals(width=2)
    totals.add(
        torch.tensor([[0., 0.], [2., 0.]]),
        torch.tensor([[0., 0.], [1., 0.]]),
        torch.tensor([[1., 0., 0.], [0., 1., 1.]]),
    )
    totals.add(
        torch.tensor([[4., 0.]]),
        torch.tensor([[3., 0.]]),
        torch.tensor([[1., 0., 0.]]),
    )

    result = totals.result()
    assert result["tokens"] == 3
    assert result["reconstruction_sse"] == 2
    assert result["mean_baseline_sse"] == 8
    assert result["fvu"] == 0.25
    assert result["observed_l0"] == pytest.approx(4 / 3)


def test_layer_ids_use_matching_all_layer_residual_checkpoint():
    for layer in (0, 17, 33):
        release, sae_id, folder, hook = checkpoint_spec(layer, all_layers=True)
        assert release == SAE_RELEASE_ALL
        assert sae_id == f"layer_{layer}_width_16k_l0_big"
        assert folder == "resid_post_all"
        assert hook == f"model.language_model.layers.{layer}"
    assert checkpoint_spec(17)[1] == "layer_17_width_16k_l0_medium"
    with pytest.raises(ValueError, match="only used at layer 17"):
        checkpoint_spec(0)
    with pytest.raises(ValueError, match="between 0 and 33"):
        checkpoint_spec(34, all_layers=True)


def test_profile_resume_requires_same_experiment(tmp_path):
    output = tmp_path / "profile.json"
    manifest = {"model_id": "medgemma", "model_revision": "commit", "prompt": "fixed", "cases": [1]}
    profile = load_profile(output, manifest, [0, 17])
    profile["layers"].append({"layer": 0})
    save_profile(output, profile)

    assert [entry["layer"] for entry in load_profile(output, manifest, [0, 17])["layers"]] == [0]
    with pytest.raises(ValueError, match="different cases, model, prompt"):
        load_profile(output, {**manifest, "prompt": "changed"}, [0, 17])
    with pytest.raises(ValueError, match="different cases, model, prompt"):
        load_profile(output, manifest, [0, 33])


def test_layer_selector():
    assert parse_layers("all") == list(range(34))
    assert parse_layers("33,0,17") == [0, 17, 33]
    with pytest.raises(ValueError, match="distinct"):
        parse_layers("0,0")
