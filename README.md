# medgemma-grounding

Minimal Python research scaffold for mechanistic interpretability experiments on visual grounding in MedGemma 1.5.

## Stack

- Python 3.11
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- `src/` package layout

## Quick start

```bash
uv python install 3.11
uv sync --dev
uv run pytest
```

## Alternative environments

For `pip`, a standard virtual environment, or Conda, install the same runtime
dependencies listed in `pyproject.toml` with the single `requirements.txt`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For Conda:

```bash
conda create -n medgemma-grounding python=3.11
conda activate medgemma-grounding
python -m pip install -r requirements.txt
```

When changing runtime dependencies, update `pyproject.toml` and
`requirements.txt` together, then run `uv lock`.

## Project layout

```text
.
├── pyproject.toml
├── src/medgemma_grounding/
└── tests/
```

This repository intentionally includes only the reproducible scaffold and no research methodology implementation yet.

## CheXlocalize: inspect one localized case

The next experiment uses the official CheXlocalize release, which contains
CheXpert images and labels plus radiologist-drawn pathology contours. The data
is deliberately stored under `data/`, which is ignored by Git.

Follow the official project's step-by-step
[CheXlocalize download instructions](https://github.com/rajpurkarlab/cheXlocalize/blob/master/download_instructions.md).
Preserve its native layout:

```text
data/
├── CheXpert/
│   ├── val/
│   ├── val_labels.csv
│   ├── test/
│   └── test_labels.csv
└── CheXlocalize/
    ├── gt_annotations_val.json
    └── gt_annotations_test.json
```

With the validation files in place, this command loads one positive pleural
effusion case, opens its image, and prints its expert label and raw contour:

```bash
uv run python -m medgemma_grounding.chexlocalize
```

`find_positive_case()` is the case-level loader intended for the next step;
it returns the image path, known label, image dimensions, and expert contours
without performing inference or looping over the dataset.

## Gemma Scope 2: reconstruction control

Run the SAE on its intended model, **Gemma 3 4B-IT**, before studying transfer
to MedGemma. Accept the [Gemma terms](https://huggingface.co/google/gemma-3-4b-it)
and set `HF_TOKEN` in `.env`, then run:

```bash
uv run python -m medgemma_grounding.gemma_scope
```

The script saves a single-input reconstruction summary to
`results/gemma_scope/gemma3_reconstruction.json`. This control does not show
that SAE features transfer to MedGemma or have medical meaning.
