"""Additional Gemma-3 attention restrictions: query is destination, key is source.

Hooks consume the model's already composed additive mask. They never construct
a replacement causal mask or change Transformers' installed implementation.
"""

from contextlib import contextmanager

import torch

WINDOWS = [(0, 5), (4, 9), (8, 13), (12, 17), (16, 21), (20, 25), (24, 29), (28, 33), (0, 33)]


def conditions():
    return [{"id": f"{route}_{lo:02d}_{hi:02d}", "route": route, "start": lo, "end": hi}
            for route in ("A", "B") for lo, hi in WINDOWS]


def token_groups(inputs, image_id):
    ids = inputs["input_ids"].detach().cpu()
    valid = inputs["attention_mask"].detach().cpu().bool()
    if ids.ndim != 2 or ids.shape[0] != 1 or valid.shape != ids.shape:
        raise ValueError("Only one unpadded image-containing sequence is supported")
    image = ((ids == image_id) & valid)[0].nonzero().flatten().tolist()
    all_valid = valid[0].nonzero().flatten().tolist()
    if not image or image != list(range(min(image), max(image) + 1)):
        raise ValueError("Expected a single contiguous image block")
    if "token_type_ids" not in inputs or not torch.equal(
            inputs["token_type_ids"].detach().cpu() == 1, ids == image_id):
        raise ValueError("Processor image block types disagree with image-token IDs")
    post = [i for i in all_valid if i > max(image)]
    if not post or max(all_valid) in image or len(all_valid) != ids.shape[1]:
        raise ValueError("Expected unpadded input with a final post-image position")
    return {"image": image, "final": [max(all_valid)], "post_image": post,
            "pre_image": [i for i in all_valid if i < min(image)], "valid": all_valid}


def restrict_mask(mask, queries, keys, mode="block"):
    """Edit only requested originally allowed entries; fail on unsupported masks."""
    if (mask is None or mask.ndim != 4 or mask.shape[:2] != (1, 1)
            or not mask.is_floating_point() or mask.shape[-1] != mask.shape[-2]):
        raise ValueError("Expected explicit [1,1,T,T] additive prefill mask")
    allowed = mask == 0
    forbidden = torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)
    if not (allowed | forbidden).all() or not allowed.any(-1).all():
        raise ValueError("Unsupported mask convention or entirely blocked row")
    requested = torch.zeros_like(allowed)
    if queries and keys:
        q = torch.tensor(queries, device=mask.device)
        k = torch.tensor(keys, device=mask.device)
        requested[0, 0, q[:, None], k[None, :]] = True
    if mode == "placebo":
        requested &= ~allowed
    elif mode == "pass":
        requested.zero_()
    elif mode != "block":
        raise ValueError("Unknown mask mode")
    changed = mask.clone()
    effective = requested & allowed
    changed[effective] = float("-inf")
    if not (changed == 0).any(-1).all():
        raise ValueError("Knockout would block an entire query row")
    if not torch.equal(changed[~effective], mask[~effective]):
        raise AssertionError("Non-target entries changed")
    return changed, effective, {"requested_edges": int(requested.sum()),
                               "blocked_edges": int(effective.sum()),
                               "non_target_unchanged": True, "no_fully_blocked_rows": True}


@contextmanager
def knockout(model, groups, condition, *, mode="block", inspect=False, audit_routes=False):
    """One prefill invocation per selected layer, with exception-safe cleanup.

    Diagnostic output hooks see eager attention probabilities one layer at a
    time, reduce them immediately, and retain no full attention matrices.
    """
    layers = model.model.language_model.layers
    indices = list(range(condition["start"], condition["end"] + 1))
    queries = groups["final"] if condition["route"] == "A" else groups["post_image"]
    handles, records, edges, calls = [], {}, {}, {i: 0 for i in indices}

    def before(index):
        def hook(module, args, kwargs):
            calls[index] += 1
            original = kwargs.get("attention_mask")
            changed, effective, summary = restrict_mask(original, queries, groups["image"], mode)
            if audit_routes:
                # Pre-image text must not read image positions under the ORIGINAL mask.
                pre = original[0, 0][groups["pre_image"]][:, groups["image"]]
                summary["pre_image_cannot_read_image"] = not bool((pre == 0).any())
                if not summary["pre_image_cannot_read_image"]:
                    raise ValueError("An unblocked pre-image route exists")
                if condition["route"] == "B" and mode == "block":
                    post = changed[0, 0][groups["post_image"]][:, groups["image"]]
                    if (post == 0).any():
                        raise ValueError("Incomplete full-depth image-to-text isolation")
            records[str(index)] = {**summary, "mask_dtype": str(original.dtype),
                                   "mask_shape": list(original.shape), "layer_type": module.layer_type}
            if inspect:
                edges[index] = effective
            return args, {**kwargs, "attention_mask": changed}
        return hook

    def after(index):
        def hook(module, args, output):
            probabilities = output[1]
            if probabilities is None or not torch.isfinite(probabilities).all():
                raise ValueError("Eager probabilities unavailable or nonfinite")
            selected = probabilities.masked_select(edges.pop(index).expand_as(probabilities))
            maximum = selected.abs().max().item() if selected.numel() else 0.
            records[str(index)]["target_max_probability"] = maximum
            if maximum != 0:
                raise ValueError("Targeted edges have nonzero probability")
        return hook

    try:
        for i in indices:
            handles.append(layers[i].self_attn.register_forward_pre_hook(before(i), with_kwargs=True))
            if inspect:
                handles.append(layers[i].self_attn.register_forward_hook(after(i)))
        yield records
        if any(n != 1 for n in calls.values()):
            raise ValueError(f"Expected one prefill call per layer: {calls}")
    finally:
        for handle in handles:
            handle.remove()
        edges.clear()
