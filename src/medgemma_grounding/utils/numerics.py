"""Numerical identity diagnostics with the original fixed tolerances."""


from __future__ import annotations

import torch


def tensor_diagnostics(reference, changed, *, rtol=1e-5, atol=1e-5):
    """Measure equivalence without losing the evidence when a check fails."""
    if reference.shape != changed.shape:
        raise ValueError("Diagnostic tensor shapes differ")
    a, b = reference.detach().cpu().double(), changed.detach().cpu().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = {"shape": list(a.shape), "finite": finite, "rtol": rtol, "atol": atol,
              "elements": a.numel(), "allclose": False}
    if finite:
        error = b - a
        result.update(allclose=bool(torch.allclose(b, a, rtol=rtol, atol=atol)),
                      mismatched_elements=int((error.abs() > atol + rtol * a.abs()).sum()),
                      max_absolute_difference=error.abs().max().item() if a.numel() else 0.,
                      rms_difference=error.square().mean().sqrt().item() if a.numel() else 0.,
                      relative_squared_error=error.square().sum().item() / a.square().sum().item()
                      if a.square().sum() else None)
    return result
