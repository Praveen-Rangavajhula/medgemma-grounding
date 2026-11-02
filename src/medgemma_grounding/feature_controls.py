"""Size controls for fixed feature removals, matched AFTER execution-precision casting.

Adapts the per-position norm principle in NNsight's evaluating_saes tutorial.
Random orientations are not assumed semantic, null, or on-manifold controls.
"""

from __future__ import annotations

import torch

MATCHING = {"relative_tolerance": 0.05, "absolute_tolerance": 1e-6,
            "bracket_steps": 12, "bisection_steps": 32}


def direction_bank(width, count=3, seed=42):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bank = torch.randn(count, width, generator=generator, dtype=torch.float32)
    return bank / bank.norm(dim=1, keepdim=True)


def perturbation_norms(original, edited, mask):
    delta = (edited - original)[0, mask].float()
    return {"position_l2": delta.norm(dim=-1).tolist(), "realized_l2": delta.norm().item(),
            "realized_rms": delta.norm().item() / max(1, delta.numel()) ** .5}


def match_direction(hidden, target, mask, direction, dtype=torch.bfloat16, settings=None):
    """Match target's realized L2 at EVERY selected position, without reading logits.

    A fixed bank direction is shared across positions, like a decoder direction;
    its scalar amplitude varies per position to match the executed target edit.
    BF16 rounding makes the norm a monotone staircase. Search for its closest
    value; unmatched staircases are reported, never accepted by changing a bound.
    Zero target positions stay exactly unchanged.
    """
    settings = MATCHING if settings is None else settings
    if (hidden.shape != target.shape or hidden.ndim != 3 or hidden.shape[0] != 1
            or mask.shape != hidden.shape[1:2] or direction.shape != hidden.shape[-1:]
            or any(t.device.type != "cpu" for t in (hidden, target, mask, direction))
            or not torch.equal(hidden, hidden.to(dtype).float())
            or not torch.equal(target, target.to(dtype).float())
            or not torch.equal(target[:, ~mask], hidden[:, ~mask])):
        raise ValueError("Expected aligned CPU activations already representable in execution dtype")
    if not all(torch.isfinite(t).all() for t in (hidden, target, direction)) or not direction.norm():
        raise ValueError("Nonfinite/zero control direction or activation")
    original = hidden[0, mask].float()
    desired = (target[0, mask] - original).norm(dim=-1)
    result = hidden.clone()
    if not desired.count_nonzero():
        return result, {"status": "zero_target", "matched": True, "max_position_relative_error": 0.,
                        "total_relative_error": 0., "requested_l2": 0., **perturbation_norms(hidden, result, mask)}
    active = desired > 0
    x, goal = original[active], desired[active]
    unit = direction.float() / direction.norm()

    def evaluate(scale):
        replacement = (x + scale[:, None] * unit).to(dtype).float()
        return (replacement - x).norm(dim=-1)

    lo, hi = torch.zeros_like(goal), goal.clone()
    best_scale, best_error = lo.clone(), goal.clone()

    def remember(scale, norm):
        nonlocal best_scale, best_error
        error = (norm - goal).abs()
        better = error < best_error
        best_scale = torch.where(better, scale, best_scale)
        best_error = torch.minimum(error, best_error)

    for _ in range(settings["bracket_steps"]):
        norm = evaluate(hi)
        remember(hi, norm)
        short = norm < goal
        if not short.any():
            break
        hi = torch.where(short, hi * 2, hi)
    tolerance = settings["absolute_tolerance"] + settings["relative_tolerance"] * goal
    for _ in range(settings["bisection_steps"]):
        if (best_error <= tolerance).all():
            break
        mid = (lo + hi) / 2
        norm = evaluate(mid)
        remember(mid, norm)
        lo = torch.where(norm < goal, mid, lo)
        hi = torch.where(norm >= goal, mid, hi)
    replacement = original.clone()
    replacement[active] = (x + best_scale[:, None] * unit).to(dtype).float()
    result[0, mask] = replacement
    realized = perturbation_norms(hidden, result, mask)
    actual = torch.tensor(realized["position_l2"])
    error = (actual - desired).abs()
    per_position = error <= settings["absolute_tolerance"] + settings["relative_tolerance"] * desired
    target_total = desired.norm().item()
    total_error = abs(realized["realized_l2"] - target_total)
    matched = bool(per_position.all() and total_error <= settings["absolute_tolerance"] + settings["relative_tolerance"] * target_total)
    return result, {"status": "matched" if matched else "unmatched", "matched": matched,
                    "max_position_relative_error": (error[active] / desired[active]).max().item(),
                    "total_relative_error": total_error / target_total,
                    "requested_l2": best_scale.norm().item(), **realized}
