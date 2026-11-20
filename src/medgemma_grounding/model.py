"""Small, explicit building blocks for running MedGemma image-text inference.

This module deliberately does *not* load a model when it is imported.  Loading
the weights is slow and requires optional machine-learning dependencies, so it
happens only when :func:`load_model` is called.

Before running this module, accept the MedGemma terms on Hugging Face and set
the ``HF_TOKEN`` environment variable to a read token.  For example::

    export HF_TOKEN=hf_...
    python -m medgemma_grounding.experiments.run_inference --image example.png \
        --prompt "Describe this image."
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import os

from dotenv import load_dotenv
import torch


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


def weight_identity(model):
    return [(name, id(p), p._version, str(p.dtype), str(p.device)) for name, p in model.named_parameters()]


def answer_candidates(tokenizer, input_ids):
    """Verify single-token Yes/No continuations at the actual chat boundary."""
    suffix = tokenizer.decode(input_ids[0, -8:].tolist(), skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
    prefix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if prefix_ids != input_ids[0, -8:].tolist():
        raise ValueError("Chat boundary does not round-trip; revise scoring verification")
    candidates = {}
    for text in ("Yes", "No"):
        combined = tokenizer.encode(suffix + text, add_special_tokens=False)
        if combined[:-1] != prefix_ids or len(combined) != len(prefix_ids) + 1:
            raise ValueError(f"Scoring protocol invalid: {text!r} is not one boundary continuation token")
        token_id = combined[-1]
        if tokenizer.decode([token_id], clean_up_tokenization_spaces=False) != text:
            raise ValueError("Answer token does not decode to candidate string")
        candidates[text] = token_id
    return {"strings": ["Yes", "No"], "ids": candidates,
            "boundary_text": suffix, "boundary_ids": prefix_ids}


def answer_score(logits, candidates):
    vector = logits.reshape(-1).double()
    if not torch.isfinite(vector).all():
        raise ValueError("Non-finite answer logits")
    yes, no = (candidates["ids"][name] for name in ("Yes", "No"))
    probabilities = vector.softmax(-1)
    score = (vector[yes] - vector[no]).item()
    return {"yes_logit": vector[yes].item(), "no_logit": vector[no].item(),
            "yes_probability": probabilities[yes].item(), "no_probability": probabilities[no].item(),
            "candidate_probability_mass": (probabilities[yes] + probabilities[no]).item(),
            "score": score, "decision": "Yes" if score > 0 else "No", "tie": score == 0,
            "argmax_token_id": vector.argmax().item()}


load_dotenv()


DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"


LOGGER = logging.getLogger("medgemma_grounding.model")


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
    torch: Any
    device: str
    model_id: str


def _load_dependencies() -> tuple[Any, Any, Any]:
    """Import optional packages only when somebody actually runs inference."""

    try:
        import torch
        from transformers import AutoModelForImageTextToText
        from transformers import AutoProcessor
    except ImportError as error:
        raise OptionalDependencyError(
            "MedGemma inference needs PyTorch and Transformers. "
            "Install the project's inference dependencies before calling "
            "load_model()."
        ) from error

    LOGGER.info("Using PyTorch %s", torch.__version__)
    LOGGER.debug("Model class: %s", AutoModelForImageTextToText.__name__)
    LOGGER.debug("Processor class: %s", AutoProcessor.__name__)

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
    revision: str | None = None,
    attn_implementation: str | dict[str, str] | None = None,
) -> ModelBundle:
    """Load a MedGemma model and its matching processor.

    Args:
        model_id: A Hugging Face MedGemma model identifier.
        device: ``"cuda"``, ``"mps"``, or ``"cpu"``.  If omitted, detect one.
        token: A Hugging Face read token.  If omitted, use ``HF_TOKEN``.
        revision: Optional checkpoint commit to load reproducibly.
        attn_implementation: Explicit supported Transformers attention setting;
            omitted by default to preserve ordinary inference behavior.

    Returns:
        A :class:`ModelBundle` that can be supplied to :func:`generate`.

    The official MedGemma repository is gated.  If loading fails with an
    authorization error, accept its terms on Hugging Face and use a token with
    access to that repository.
    """

    torch, model_class, processor_class = _load_dependencies()
    selected_device = device or choose_device(torch)
    LOGGER.info("Using device: %s", selected_device)
    if selected_device not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: 'cuda', 'mps', or 'cpu'")

    if selected_device == "cpu":
        dtype = torch.float32
    elif selected_device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        # MedGemma emits only padding tokens with FP16 on Apple MPS, while
        # BF16 is stable and keeps the same memory footprint.
        dtype = torch.bfloat16
    access_token = token or os.environ.get("HF_TOKEN")
    if not access_token:
        raise RuntimeError("Set HF_TOKEN in .env before loading MedGemma.")

    LOGGER.info("Loading %s on %s using %s", model_id, selected_device, dtype)
    processor = processor_class.from_pretrained(model_id, token=access_token, revision=revision)
    model = model_class.from_pretrained(
        model_id,
        dtype=dtype,
        token=access_token,
        revision=revision,
        **({"attn_implementation": attn_implementation} if attn_implementation is not None else {}),
    )
    model.to(selected_device)
    model.eval()  # We are generating, not training.

    return ModelBundle(
        model=model,
        processor=processor,
        torch=torch,
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
    LOGGER.debug("Loading image: %s", image_path)

    with Image.open(image_path) as source:
        return source.convert("RGB")


def prepare_inputs(bundle: ModelBundle, image: Any, prompt: str) -> Any:
    """Use the same image/chat formatting for generation and activation capture."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return bundle.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(bundle.device, dtype=bundle.model.dtype)


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
    inputs = prepare_inputs(bundle, image, prompt)
    input_length = inputs["input_ids"].shape[-1]

    LOGGER.info("Generating response")
    with bundle.torch.inference_mode():
        generation = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # ``generate`` includes the prompt tokens. Decode only newly generated text.
    answer_ids = generation[0][input_length:]
    LOGGER.info("Generated %d tokens", answer_ids.shape[0])
    answer = bundle.processor.decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if not answer:
        raise RuntimeError(
            "MedGemma generated only special or padding tokens. "
            "Check the selected device and tensor dtype."
        )
    return answer
