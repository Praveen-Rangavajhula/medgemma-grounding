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

## Gemma Scope 2: one reconstruction check

This experiment starts with **Gemma 3 4B-IT**, the SAE's intended model.
Accept its [Hugging Face terms](https://huggingface.co/google/gemma-3-4b-it)
and set `HF_TOKEN` in `.env`. Access to MedGemma does not grant Gemma 3 access.

```bash
uv sync --dev
uv run python -m medgemma_grounding.gemma_scope
```

The defaults use `patient64545` and the fixed Yes/No pleural-effusion prompt.
Override them with `--image` and `--prompt`. Model and SAE downloads are cached
by Hugging Face. Pip users should install the project with
`python -m pip install --no-deps -e .` after installing `requirements.txt`,
then use `python -m`.

We load `layer_17_width_16k_l0_medium` from Google's
[Gemma Scope 2 4B-IT release](https://huggingface.co/google/gemma-scope-2-4b-it)
using the [SAELens loader](https://github.com/decoderesearch/SAELens).
The installed SAELens registry calls this release `gemma-scope-2-4b-it-res`;
its v6 API differs from older snippets in Google's README.
Layer 17 is zero-based: we capture the block's residual output, not an MLP
output or the model's final normalized state. The converter specifies no
additional activation normalization for this checkpoint.

Gemma runs with the existing device policy; the SAE runs on captured CPU/FP32
activations. The command checks that recording leaves next-token logits
unchanged, then prints and saves a summary to
`results/gemma_scope/gemma3_reconstruction.json` (overwritten on rerun; use
`--output` to keep multiple runs). It includes checkpoint revisions, the image
hash, shapes, and separate image/text token statistics. Special/padding tokens
are excluded from metrics.

- `relative_squared_error`: squared reconstruction error / activation energy;
  lower is better, zero is exact reconstruction.
- `explained_variance`: `1 - SSE/SST`, centered across tokens per dimension;
  one is perfect, zero matches the token-mean baseline, negative is worse.
  It is null when the baseline variance is zero.
- `mean_active_features`: average number of nonzero SAE features per token.

These are single-input diagnostics, not evidence of successful medical feature
transfer. Review this reference result before applying the SAE to MedGemma.
Use `--revision` with the recorded Gemma commit to repeat the model/processor
version; the output also records the SAE snapshot and resolved configuration.
