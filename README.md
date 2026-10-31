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

The repository contains small, reproducible experiments rather than a full analysis pipeline.

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

## Gemma Scope 2 on MedGemma

To check whether the Gemma Scope 2 SAE transfers to MedGemma 1.5, run one
image and then the fixed 20-image CheXpert validation pilot:

```bash
uv run python -m medgemma_grounding.medgemma_scope
uv run python -m medgemma_grounding.medgemma_scope --pilot
```

The JSON results are saved under `results/gemma_scope/`. The earlier
`gemma_scope` command and `gemma3_reconstruction.json` remain the Gemma 3
control; SAE reconstruction alone does not establish feature meaning.

For the 34-layer MedGemma transfer screen, run
`uv run python -m medgemma_grounding.medgemma_scope_layers`. It saves a resumable
FVU, L0, and next-token KL profile to `results/gemma_scope/medgemma_layer_profile.json`.
FVU and L0 exclude special tokens; KL measures the next-token distribution after
reconstructing every input position, including special tokens.

## Layer-14 feature viewer

Capture the exact 20 pilot cases with the layer-14 16k `l0_big` SAE, then
browse the saved features locally:

```bash
uv run python -m medgemma_grounding.medgemma_features
uv run streamlit run src/medgemma_grounding/feature_viewer.py
```

### Discovery cohort (validation only)

Build the fixed patient split, then extract **discovery only** into a new
experiment directory. The first command audits 202 eligible images / 200
patients, includes all 20 pilot patients in discovery, and reserves 60 patients
for confirmation. Original images stay in place; test data is not used.

```bash
uv run python -m medgemma_grounding.chexlocalize \
  --cohort-output results/gemma_scope/discovery_v1/cohort.json
uv run python -m medgemma_grounding.medgemma_features \
  --cohort-manifest results/gemma_scope/discovery_v1/cohort.json \
  --output-dir results/gemma_scope/discovery_v1 --limit 2
```

Inspect the smoke records and runtime estimate before continuing. Resume with
the same extraction command **without `--limit 2`**; completed cases are validated
and skipped. Changed inputs, extraction sources, revisions, or settings are
rejected. The schema-1 pilot cache is never overwritten.

After extraction, compute patient-level discovery statistics without inference:

```bash
uv run python -m medgemma_grounding.feature_analysis \
  --experiment-dir results/gemma_scope/discovery_v1
uv run streamlit run src/medgemma_grounding/feature_viewer.py
```

The viewer defaults to `results/gemma_scope/discovery_v1` when its report exists;
otherwise set **Saved feature directory** in the sidebar to that path. It displays four ranked association tables,
the frozen shortlist, score/activation plots, counterexamples, expert-mask
outlines, token values, and reconstruction distortion. Full token features are
loaded only for the selected case; browsing does not load MedGemma or an SAE.

Artifacts include `protocol.json`, `cohort.json`, per-case JSON/sparse tensors
under `cases/`, `rankings.csv`, `shortlist.json`, `discovery_report.json`,
`mask_audit.json`, and `mask_alignment.png`. Yes/No tokenization is verified at
the actual generation boundary. The score is a next-token logit difference,
not a calibrated clinical probability. Mean image-position and final-input
features are screened separately; each patient contributes one observation.
Selection is exploratory, not a significance claim over 16,384 features.

Confirmation inference/browsing is deliberately unavailable. The shortlist
freezes discovery associations, **not** a future intervention/spatial protocol.
No feature suppression, image masking, spatial performance evaluation, or
mitigation is performed by these commands. Decoder-position overlays are not
anatomical localization or causal attribution.

Methods adapted from the [Line of Sight aggregation scripts](https://github.com/multimodal-interpretability/multimodal-saes/tree/8a466949994700be3eb67e3364375083d7697562/probing_approximation)
and [official CheXlocalize mask conversion](https://github.com/rajpurkarlab/cheXlocalize/blob/2a79492e9ecdf1c3cad74d968134cadc16be4a70/annotation_to_segmentation.py).

### Layer-14 feature attribution and interventions

The E-005 workflow tests first-order feature-removal estimates against direct
residual-preserving edits on the existing discovery images. Run stages in order:

```bash
uv run python -m medgemma_grounding.medgemma_interventions --stage pilot
uv run python -m medgemma_grounding.medgemma_interventions --stage attribute
uv run python -m medgemma_grounding.medgemma_interventions --stage select
uv run python -m medgemma_grounding.medgemma_interventions --stage ablate
uv run python -m medgemma_grounding.medgemma_interventions --stage report
```

`pilot` runs the fixed 20-patient calibration, starting with its TP/TN preflight.
Continue only if a position group qualifies in `pilot_report.json`; otherwise
run `report` and stop. Later stages enforce this prerequisite. `attribute` and
`ablate` accept `--limit N` newly processed cases and resume saved work.
Selection requires complete discovery attribution. `select` and `report`
run offline. `--stage` is required; importing the module starts no experiment.

Defaults: source `results/gemma_scope/discovery_v1`, output
`results/gemma_scope/interventions_v1`; override with `--source-dir` and
`--output-dir`. Outputs include the fixed protocol, per-case CPU shards,
`pilot_report.json`, `attribution_rankings.csv`, `candidates.json`,
`intervention_effects.csv`, `intervention_report.json` and compact summaries.
Changed protocols/settings require a new output directory. Completed treatments
are saved individually; full ablations may take hours. The runtime stays
MPS/BF16 with a CPU/FP32 SAE; errors stop execution without device substitution.

In the feature viewer, set **Optional intervention results directory** to the
output path. It displays calibration, signed attribution, direct removal
effects and partial-run diagnostics. Estimates and matched comparisons remain
discovery-selected; their uncertainty intervals are descriptive. This study
does not establish a circuit, anatomical localization or unseen-patient benefit.
Confirmation and test data remain reserved.

### Existing 20-case cache

Run both commands from the repository root. Extraction saves
`results/gemma_scope/layer14_features/manifest.json` and `activations.pt`:
model/processor/SAE revisions, original image paths and hashes, expert labels,
raw greedy MedGemma answers (up to 32 tokens), token identities, and lossless
sparse features for every input token. Use `--overwrite` to replace this cache.
The viewer uses the original images at the recorded project location. Browsing
requires neither model weights nor a Hugging Face token and performs no inference.

Choose a feature ID to compare all 20 cases, inspect strongly or weakly
activating examples, and view image-position grids, text highlights and raw
token values. Cases rank by maximum image-token activation by default; text
ranking is also available. The selected feature shares one color scale across
all cases, excluding special tokens. Labels do not determine ranking.

The optional overlay follows the verified single-image Gemma 3 path: direct
resize to 896×896, 64×64 vision patches, 4×4 pooling, then a row-major 16×16
image-token grid. Unsupported layouts disable overlays. These are contextual
decoder-position activations, **not pathology-localization or causal-attribution
maps**. Only input/prefill activations are captured; the displayed generated
answer is context. The 20 cases develop the workflow, not confirm medical
feature meanings; broader discovery and separate confirmation patients come later.

The gallery, ranking and overlay code adapts Imageomics'
[`saev-demo` feature viewer](https://github.com/Imageomics/saev-demo/blob/5fa496c9f8b63f503b1c8d9b0b9ef25dcd81b09e/demo/feature_vis.py),
replacing its DINOv2/SAE inference path with saved MedGemma activations.
Source attribution and the upstream MIT notice are retained in the viewer.
