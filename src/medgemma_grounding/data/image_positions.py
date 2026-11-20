"""Verified processor-to-image position mapping; decoder positions are contextual."""


from __future__ import annotations

import math


def spatial_layout(bundle, inputs, original_size):
    """Describe the verified single-image Gemma 3 resize/pooling layout.

    Image-position order has a spatial anchor, not a localized receptive field:
    vision attention and decoder attention already mix information.
    """
    config = bundle.model.config
    processor = bundle.processor.image_processor
    side = math.isqrt(config.mm_tokens_per_image)
    pixels = inputs["pixel_values"]
    n_image = int((inputs["input_ids"] == config.image_token_index).sum())
    valid = (
        type(processor).__name__.startswith("Gemma3ImageProcessor")
        and processor.do_resize
        and not processor.do_pan_and_scan
        and not getattr(processor, "do_center_crop", False)
        and not getattr(processor, "do_pad", False)
        and pixels.shape == (1, 3, config.vision_config.image_size, config.vision_config.image_size)
        and n_image == side * side == config.mm_tokens_per_image
        and config.vision_config.image_size % (config.vision_config.patch_size * side) == 0
    )
    if not valid:
        return {"verified": False, "reason": "Unsupported image preprocessing or token layout; overlays disabled."}
    return {
        "verified": True,
        "mapping": "direct_resize_row_major",
        "grid": [side, side],
        "original_size": list(original_size),
        "processed_size": list(pixels.shape[-2:]),
        "pool_kernel": config.vision_config.image_size // config.vision_config.patch_size // side,
        "processor_class": type(processor).__name__,
        "resample": int(processor.resample),
    }
