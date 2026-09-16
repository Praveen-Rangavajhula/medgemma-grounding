"""Small, explicit building blocks for running MedGemma image-text inference.

This module deliberately does *not* load a model when it is imported.  Loading
the weights is slow and requires optional machine-learning dependencies, so it
happens only when :func:`load_model` is called.

Before running this module, accept the MedGemma terms on Hugging Face and set
the ``HF_TOKEN`` environment variable to a read token.  For example::

    export HF_TOKEN=hf_...
    python -m medgemma_grounding.inference --image example.png \\
        --prompt "Describe this image."
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# Keep the model name in one place: later experiments can accept a different
# model id without duplicating the loading logic.
DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"


class OptionalDependencyError(RuntimeError):
    """Raised when inference dependencies have not been installed yet."""


@dataclass(frozen=True)
class ModelBundle:
    """The objects needed for repeated inference calls.

    ``model`` and ``processor`` are typed as ``Any`` because importing their
    concrete Transformers classes at module import time would make this file
    unusable before the optional dependencies are installed.
    """

    model: Any
    processor: Any
    device: str
    model_id: str


def _load_dependencies() -> tuple[Any, Any, Any]:
    """Import optional packages only when somebody actually runs inference."""

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise OptionalDependencyError(
            "MedGemma inference needs PyTorch and Transformers. "
            "Install the project's inference dependencies before calling "
            "load_model()."
        ) from error

    return torch, AutoModelForImageTextToText, AutoProcessor


def choose_device(torch: Any) -> str:
    """Choose the best available PyTorch device, without assuming a GPU."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device: str | None = None,
    token: str | None = None,
) -> ModelBundle:
    """Load a MedGemma model and its matching processor.

    Args:
        model_id: A Hugging Face MedGemma model identifier.
        device: ``"cuda"``, ``"mps"``, or ``"cpu"``.  If omitted, detect one.
        token: A Hugging Face read token.  If omitted, use ``HF_TOKEN``.

    Returns:
        A :class:`ModelBundle` that can be supplied to :func:`generate`.

    The official MedGemma repository is gated.  If loading fails with an
    authorization error, accept its terms on Hugging Face and use a token with
    access to that repository.
    """

    torch, model_class, processor_class = _load_dependencies()
    selected_device = device or choose_device(torch)
    if selected_device not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: 'cuda', 'mps', or 'cpu'")

    # Lower precision reduces memory use on accelerators.  CPU inference stays
    # float32 because it is the most broadly compatible starting point.
    dtype = torch.float32 if selected_device == "cpu" else torch.float16
    access_token = token or os.environ.get("HF_TOKEN")

    processor = processor_class.from_pretrained(model_id, token=access_token)
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=access_token,
    )
    model.to(selected_device)
    model.eval()  # We are generating, not training.

    return ModelBundle(
        model=model,
        processor=processor,
        device=selected_device,
        model_id=model_id,
    )


def load_image(image_path: str | Path) -> Any:
    """Open an image as RGB, the predictable format for model input."""

    try:
        from PIL import Image
    except ImportError as error:
        raise OptionalDependencyError(
            "Image loading needs Pillow. Install the project's inference "
            "dependencies before calling load_image()."
        ) from error

    with Image.open(image_path) as source:
        return source.convert("RGB")


def generate(
    bundle: ModelBundle,
    image: Any,
    prompt: str,
    *,
    max_new_tokens: int = 128,
) -> str:
    """Ask MedGemma one question about one image and return its response."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = bundle.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(bundle.device)

    generated_ids = bundle.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Deterministic outputs make first experiments easier.
    )
    # ``generate`` includes the input tokens.  Remove them before decoding so
    # callers receive only MedGemma's answer.
    answer_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return bundle.processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main() -> None:
    """Provide a minimal command-line smoke test for this module."""

    parser = argparse.ArgumentParser(description="Run MedGemma on one image.")
    parser.add_argument("--image", type=Path, required=True, help="Image to inspect")
    parser.add_argument(
        "--prompt",
        default="Describe this image.",
        help="Question to ask MedGemma",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    arguments = parser.parse_args()

    bundle = load_model(arguments.model_id)
    image = load_image(arguments.image)
    answer = generate(
        bundle,
        image,
        arguments.prompt,
        max_new_tokens=arguments.max_new_tokens,
    )
    print(answer)


if __name__ == "__main__":
    main()
