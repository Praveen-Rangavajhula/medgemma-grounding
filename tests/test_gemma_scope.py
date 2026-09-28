"""Small mathematical and hook checks; no network or model downloads."""

from types import SimpleNamespace

import pytest
import torch

from medgemma_grounding.gemma_scope import (
    capture_residual,
    reconstruction_metrics,
    token_groups,
)


def test_metrics_known_reconstructions():
    x = torch.tensor([[1., 0.], [0., 1.]])
    perfect = reconstruction_metrics(x, x, x)
    assert perfect["relative_squared_error"] == 0
    assert perfect["explained_variance"] == 1
    assert perfect["mean_active_features"] == 1
    zero = reconstruction_metrics(x, torch.zeros_like(x), x)
    assert zero["relative_squared_error"] == 1
    assert zero["explained_variance"] == -1
    constant = reconstruction_metrics(torch.ones_like(x), torch.ones_like(x), x)
    assert constant["explained_variance"] is None
    with pytest.raises(ValueError, match="Non-finite"):
        reconstruction_metrics(x, x * float("nan"), x)


def test_token_groups_exclude_special_and_padding_but_keep_images():
    ids = torch.tensor([[1, 99, 99, 20, 21, 2, 0]])
    groups = token_groups(ids, torch.tensor([[1, 1, 1, 1, 1, 1, 0]]), 99, [0, 1, 2, 99])
    assert groups["image"].nonzero().flatten().tolist() == [1, 2]
    assert groups["text"].nonzero().flatten().tolist() == [3, 4]
    assert not (groups["image"] & groups["text"]).any()


class TinyModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.block = torch.nn.Identity()
        self.fail = fail

    def forward(self, x, **kwargs):
        output = self.block(x)
        if self.fail:
            raise RuntimeError("intentional failure")
        return SimpleNamespace(logits=output[:, -1:])


def test_hook_preserves_output_and_is_removed():
    model = TinyModel()
    x = torch.randn(1, 4, 3)
    hidden, logits = capture_residual(model, {"x": x}, "block")
    torch.testing.assert_close(hidden, x)
    torch.testing.assert_close(logits, model(x).logits)
    assert not model.block._forward_hooks
    model.fail = True
    with pytest.raises(RuntimeError, match="intentional"):
        capture_residual(model, {"x": x}, "block")
    assert not model.block._forward_hooks
