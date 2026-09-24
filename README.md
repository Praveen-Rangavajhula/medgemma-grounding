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
is deliberately stored under `data/chexlocalize/`, which is ignored by Git.

Download the official release after registering and accepting its terms at the
[Stanford AIMI CheXlocalize page](https://aimi.stanford.edu/datasets/chexlocalize).
Preserve its native layout:

```text
data/chexlocalize/
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
