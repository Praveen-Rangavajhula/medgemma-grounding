"""Fixed decoded-feature edits and first-order attribution on the original model.

Method references (adapted, no upstream code copied):
feature-circuits at 7fbd82b: attribution.py (_pe_attrib), ablation.py
(handle_errors='keep'). We project original-state gradients onto decoder rows
directly, avoiding a gradient through JumpReLU or a reconstruction baseline.
"""

from __future__ import annotations

import torch

GROUPS = ("image", "final", "other_non_image")


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


def attention_settings(model):
    """Read the actual multimodal attention configuration after public API calls."""
    return {name: getattr(config, "_attn_implementation", None) for name, config in (
        ("", model.config), ("text_config", model.config.text_config),
        ("vision_config", model.config.vision_config))}


def configure_attention(model, implementation):
    """Use Transformers' supported API and reject unsupported/partial changes."""
    requested = ({key: implementation for key in attention_settings(model)}
                 if isinstance(implementation, str) else dict(implementation))
    if (not requested or not requested.keys() <= attention_settings(model).keys()
            or any(value not in {"sdpa", "eager"} for value in requested.values())):
        raise ValueError("Unsupported attention setting")
    model.set_attn_implementation(implementation)
    actual = attention_settings(model)
    if any(actual[key] != value for key, value in requested.items()):
        raise ValueError(f"Attention setting was not applied consistently: {actual}")
    return actual


def intervention_groups(groups):
    """Partition valid processed positions; final must be a non-image position."""
    valid, image, final = (groups[k].cpu().bool() for k in ("all", "image", "final"))
    result = {"image": image, "final": final, "other_non_image": valid & ~image & ~final}
    if (final.sum() != 1 or (image & final).any()
            or not torch.equal(sum(mask.int() for mask in result.values()), valid.int())):
        raise ValueError("Position groups must partition valid positions with one non-image final")
    return result


def feature_removal(hidden, features, decoder, feature, mask, alpha, dtype):
    """Return CPU FP32 replacement and realized perturbation after model casting.

    Hidden is [1,T,D], features [T,F], decoder [F,D]. The unchanged reconstruction
    residual and decoder bias cancel: H' = H - alpha*z_j*w_j, only at mask rows.
    Never re-encode H'. Round only selected rows, preserving unselected values.
    """
    if (hidden.ndim != 3 or hidden.shape[0] != 1 or features.ndim != 2
            or features.shape[0] != hidden.shape[1]
            or decoder.shape != (features.shape[1], hidden.shape[2])
            or mask.shape != (hidden.shape[1],) or not 0 <= feature < features.shape[1]
            or not 0 <= alpha <= 1):
        raise ValueError("Invalid feature edit dimensions, ID, mask or strength")
    if any(x.device.type != "cpu" for x in (hidden, features, decoder, mask)):
        raise ValueError("Feature edit arithmetic must run on CPU")
    delta = -alpha * features[mask, feature, None].float() * decoder[feature].float()
    edited = hidden.float().clone()
    if alpha and torch.count_nonzero(delta):
        edited[0, mask] = (hidden[0, mask].float() + delta).to(dtype).float()
    realized = edited - hidden.float()
    if not torch.isfinite(edited).all():
        raise ValueError("Nonfinite feature edit")
    return edited, {"requested_l2": delta.norm().item(), "realized_l2": realized.norm().item(),
                    "realized_rms": realized.norm().item() / max(1, delta.numel()) ** 0.5}


def capture_gradient(model, inputs, hook_name, candidates):
    """One leaf replacement, original forward, autograd only through later blocks.

    All weights must already be frozen. autograd.grad targets only the cloned
    layer output, so parameter .grad remains None. Hooks are removed on errors.
    """
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Gradient capture requires eval mode, frozen weights and unset parameter gradients")
    leaves = []

    def hook(_module, _args, output):
        if leaves:
            raise RuntimeError("Expected exactly one gradient hook invocation")
        value = output[0] if isinstance(output, tuple) else output
        leaf = value.detach().clone().requires_grad_(True)
        leaves.append(leaf)
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    handle = model.get_submodule(hook_name).register_forward_hook(hook)
    try:
        with torch.inference_mode(False), torch.enable_grad():
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits
            if len(leaves) != 1:
                raise RuntimeError(f"Expected one gradient hook invocation, got {len(leaves)}")
            ids = candidates["ids"]
            score = logits[0, -1, ids["Yes"]].float() - logits[0, -1, ids["No"]].float()
            gradient, = torch.autograd.grad(score, leaves[0])
            result = tuple(x.detach().to(device="cpu", dtype=torch.float32).clone()
                           for x in (leaves[0], logits, gradient))
    finally:
        handle.remove()
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Parameter gradient unexpectedly populated")
    if not all(torch.isfinite(x).all() for x in result):
        raise ValueError("Nonfinite gradient capture")
    return result


def project_attribution(gradient, features, decoder, masks, chunk_size=256):
    """Signed full-removal estimates on the exact original sparse support.

    A[t,j] = -z[t,j] * (G[t] dot W_dec[j]). Group effects SUM positions.
    Removal RMS is sqrt(sum_t ||z[t,j]*w_j||² / (group_tokens * hidden_width)).
    """
    gradient = gradient.detach().cpu().float()
    if gradient.ndim == 3:
        gradient = gradient[0]
    sparse = features.to_sparse().coalesce().cpu()
    decoder = decoder.detach().cpu().float()
    if (gradient.shape != (features.shape[0], decoder.shape[1])
            or features.shape[1] != decoder.shape[0] or chunk_size < 1):
        raise ValueError("Attribution dimensions or chunk size differ")
    indices, values = sparse.indices(), sparse.values().float()
    attributed = torch.empty_like(values)
    for start in range(0, decoder.shape[0], chunk_size):
        end = min(start + chunk_size, decoder.shape[0])
        select = (indices[1] >= start) & (indices[1] < end)
        projection = gradient @ decoder[start:end].T
        attributed[select] = -values[select] * projection[indices[0, select], indices[1, select] - start]
    summaries = {}
    row_norm2 = decoder.square().sum(1)
    for name, mask in masks.items():
        select = mask[indices[0]]
        feature_ids = indices[1, select]
        effect = torch.zeros(features.shape[1]).scatter_add_(0, feature_ids, attributed[select])
        norm2 = torch.zeros_like(effect).scatter_add_(
            0, feature_ids, values[select].square() * row_norm2[feature_ids])
        active = torch.zeros_like(effect, dtype=torch.bool)
        active[feature_ids] = True
        summaries[name] = {"effect": effect, "active": active, "removal_l2": norm2.sqrt(),
                           "removal_rms": (norm2 / max(1, int(mask.sum()) * decoder.shape[1])).sqrt()}
    attribution = torch.sparse_coo_tensor(indices, attributed, sparse.shape).coalesce()
    if not torch.isfinite(attributed).all():
        raise ValueError("Nonfinite attribution projection")
    return attribution, summaries
