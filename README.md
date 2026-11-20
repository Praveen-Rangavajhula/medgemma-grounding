# medgemma-grounding

Research on visual grounding and mechanistic interpretability in **MedGemma 1.5**,
using CheXlocalize X-rays, transferred Gemma Scope 2 SAEs and controlled attention
interventions. Gemma 3 is a reference control.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required. Run commands from
the repository root.

```bash
uv sync --dev
uv run pytest
```

Accept the relevant Hugging Face model terms and configure a read token using
`example.env` before inference. Saved-result browsing does not need weights or a
token. For other Python environments, install `requirements.txt`.

Follow the [official CheXlocalize download instructions](https://github.com/rajpurkarlab/cheXlocalize/blob/master/download_instructions.md)
and retain the original layout:

```text
data/
├── CheXpert/{val/, val_labels.csv, test/, test_labels.csv}
└── CheXlocalize/{gt_annotations_val.json, gt_annotations_test.json}
```

## Code layout

```text
src/medgemma_grounding/
├── model.py          # model loading, input preparation and answer scoring
├── data/             # cases, labels, patient splits and image-position mapping
├── sae/              # checkpoints, reconstruction and CPU feature caches
├── interventions/    # feature edits, attention restrictions and execution helpers
├── experiments/      # runnable study stages and historical offline replay
├── analysis/         # patient-level statistics and saved-result reports
├── visualization/    # Streamlit application, overlays and study panels
└── utils/            # I/O, numerical diagnostics and source provenance
tests/                # corresponding data, SAE, intervention and study checks
```

The [migration guide](MIGRATION.md) maps previous modules and commands, explains
historical source snapshots, and records the upstream organizational references.

## Inspect saved results

```bash
uv run streamlit run src/medgemma_grounding/visualization/app.py
```

The viewer opens the discovery feature cache when available. Use **Optional
intervention results directory** to inspect a completed study:

| Directory | Contents |
| --- | --- |
| `results/gemma_scope/discovery_v1` | feature associations, shortlist, examples and expert-mask audit |
| `results/gemma_scope/interventions_v3_discovery` | calibrated feature attribution and direct removals |
| `results/gemma_scope/targeted_discovery_v1` | candidate effects and executed-size controls |
| `results/attention_depth_v1` | attention-route window knockouts |
| `results/attention_errors_v3` | individual layers, correct/error contrasts and score controls |

Regenerate or verify the existing attention reports without loading weights:

```bash
uv run python -m medgemma_grounding.experiments.run_attention_windows --stage report
uv run python -m medgemma_grounding.experiments.run_attention_layers --stage report
uv run python -m medgemma_grounding.experiments.run_attention_layers --stage verify
```

Historical attention commands automatically use the verified pre-refactor
runtime. Completed `--stage run` checks remain model-free; incomplete historical
execution is refused. Keep `results/maintenance/layout_20261120/` with the saved
results when transferring this workspace. These ignored artifacts are not
included in a fresh clone.

## Experiment entry points

Use `uv run python -m medgemma_grounding.experiments.<module> --help` for options.
Model execution is explicit; importing a module never starts an experiment.

| Module | Purpose / output |
| --- | --- |
| `inspect_data` | inspect a case or freeze the validation patient split |
| `run_inference` | one-image MedGemma inference |
| `run_gemma_control` | Gemma 3 SAE reference check |
| `run_sae_transfer` | single-image or fixed 20-case transfer pilot |
| `run_sae_layers` | layerwise FVU, L0 and next-token KL profile |
| `compute_sae_activations` | pilot or discovery CPU feature cache |
| `diagnose_attention` | bounded two-case numerical diagnosis |
| `run_feature_pilot` | historical E-005 calibration workflow |
| `run_feature_discovery` | fresh eager-text / SDPA-vision discovery attribution |
| `run_targeted_features` | candidate removals and matched-size directions |
| `run_attention_windows` | image-to-text routing windows |
| `run_attention_layers` | individual routing layers and score controls |

Reports and measurements live under ignored `results/`; original images and
Hugging Face cache locations stay unchanged. Staged studies save protocols,
per-case records, summaries and figures. Resume requires identical cohort,
runtime, settings and source identity. **Use a new output directory for execution
with the reorganized implementation**; do not reuse a historical protocol.
Attention studies additionally require reviewed methods and their source artifacts
before preparation. No new model run is part of this code migration.

## Interpretation limits

The 20 cases are a workflow pilot. Discovery uses 142 validation images / 140
patients; 60 confirmation patients and all test data remain reserved. Missing
contours do not imply a negative label, and no effusion does not imply a normal
X-ray. Decoder-position overlays are contextual feature activity, not anatomical
localization or causal pixel attribution. The Yes/No logit difference is not a
calibrated clinical probability. Reconstruction, association, causal influence
and improved classification are separate claims.

The viewer retains the attribution and MIT notice for Imageomics'
[saev-demo](https://github.com/Imageomics/saev-demo/blob/5fa496c9f8b63f503b1c8d9b0b9ef25dcd81b09e/demo/feature_vis.py).
Research protocols and findings are recorded in the local, ignored `docs/` and
saved result directories.
