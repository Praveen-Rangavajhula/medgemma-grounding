"""Local checks for cohort selection and residual interventions."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from medgemma_grounding.medgemma_scope import pilot_cases, summarize
from medgemma_grounding.sae_reconstruction import logit_metrics, replace_residual


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Identity()

    def forward(self, x, **kwargs):
        hidden = self.block(x)
        return SimpleNamespace(logits=hidden[:, -1:])


def test_replacement_changes_logits_and_removes_hook():
    model = TinyModel()
    x = torch.ones(1, 3, 4)
    logits = replace_residual(model, {"x": x}, "block", torch.zeros_like(x))
    assert torch.equal(logits, torch.zeros(1, 1, 4))
    assert not model.block._forward_hooks
    with pytest.raises(ValueError, match="Replacement shape"):
        replace_residual(model, {"x": x}, "block", torch.zeros(1, 2, 4))
    assert not model.block._forward_hooks


def test_next_token_metrics():
    reference = torch.tensor([[[3., 1., 0.]]])
    same = logit_metrics(reference, reference)
    assert abs(same["kl_from_baseline"]) < 1e-6
    assert same["top_token_preserved"]
    changed = logit_metrics(reference, torch.tensor([[[0., 1., 3.]]]))
    assert changed["kl_from_baseline"] > 0
    assert not changed["top_token_preserved"]


def test_summary_includes_median_reconstruction_r_squared():
    cases = [
        {
            "metrics": {
                "image": {"relative_squared_error": 0.1, "explained_variance": 0.8},
                "text": {"relative_squared_error": 0.2, "explained_variance": 0.6},
            },
            "next_token": {
                "reconstruction": {"kl_from_baseline": 0.1, "top_token_preserved": True},
                "zero_ablation": {"kl_from_baseline": 1.0},
            },
        },
        {
            "metrics": {
                "image": {"relative_squared_error": 0.2, "explained_variance": 0.9},
                "text": {"relative_squared_error": 0.3, "explained_variance": 0.8},
            },
            "next_token": {
                "reconstruction": {"kl_from_baseline": 0.2, "top_token_preserved": False},
                "zero_ablation": {"kl_from_baseline": 2.0},
            },
        },
    ]

    summary = summarize(cases)

    assert summary["median_reconstruction_r_squared"] == pytest.approx(
        {"image": 0.85, "text": 0.7}
    )
    assert "explained_variance" in cases[0]["metrics"]["image"]
    json.dumps(summary, allow_nan=False)


def test_pilot_is_deterministic_balanced_and_patient_disjoint(tmp_path: Path):
    root = tmp_path / "val"
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Path", "Frontal/Lateral", "Pleural Effusion"])
        writer.writeheader()
        for label in (0, 1):
            for index in range(12):
                patient = f"patient{label}{index:03d}"
                relative = Path(patient) / "study1" / "view1_frontal.jpg"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                writer.writerow({"Path": f"CheXpert-v1.0/valid/{relative}",
                                 "Frontal/Lateral": "Frontal", "Pleural Effusion": f"{label}.0"})
    first = pilot_cases(labels, root)
    assert first == pilot_cases(labels, root)
    assert len(first) == 20
    assert sum(case["label"] == 1 for case in first) == 10
    assert len({case["patient"] for case in first}) == 20
