"""Run one image-text inference using the shared model loader."""


from __future__ import annotations

from pathlib import Path
import argparse
import logging

from medgemma_grounding.model import DEFAULT_MODEL_ID, LOGGER, generate, load_image, load_model


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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug logging, including model classes and the image path",
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    LOGGER.setLevel(logging.DEBUG if arguments.verbose else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

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
