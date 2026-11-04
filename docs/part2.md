# Kristian's work: sections 2.1 through 2.6

Work branch: `codex/part2-grounding`, created from
`feature/classification-benchmark` at `7fef071a2a0a4de75de3ff16568bc2477b968847`.
Part 1's classification and inference files are inherited unchanged. Each
increment adds its own modules, tests, and documentation. The earlier grounding
prototype remains in the stash until its components are needed and reviewed.

## 2.1: protocol and patient partitions

The target is pleural effusion on frontal CheXlocalize validation images. The
verified local release has 202 frontal images from 200 patients: 64 positives
and 138 negatives. Keep target-negative images with other abnormalities.
Exclude lateral images and missing/uncertain target labels. A missing annotation
does not imply a negative diagnosis.

Prepare the protocol without downloading weights or running inference:

```bash
uv run python -m medgemma_grounding.grounding_protocol \
  --data-root data \
  --output outputs/part2/protocol-v1 \
  --seed 42 \
  --discovery-fraction 0.5 \
  --discovery-patient patient64545
```

With the already-installed project environment, the equivalent command is:

```bash
PYTHONPATH=src .venv/bin/python -m medgemma_grounding.grounding_protocol \
  --data-root data \
  --output outputs/part2/protocol-v1 \
  --discovery-patient patient64545
```

Patient `patient64545` was inspected in the initial annotation preview, so it
stays in discovery. The proposed split is 50/50 by patient, stratified on whether
a patient has any positive effusion image. All images of the same patient stay
together. Selection uses fixed SHA256 rankings and is independent of CSV order.
Discovery is for debugging and selecting perturbation/feature settings;
confirmation is reserved for evaluating those fixed settings. Part 1 already
evaluated all validation images, so this is a holdout from Part 2 tuning, not an
entirely unseen classification dataset. The official test split remains reserved.

The command creates these ignored local artifacts:

- `protocol.json`: exact Part 1 prompt, scorer/parser references, source hashes,
  dataset hashes, cohort rules, output categories, and patient-split settings.
- `cohort_manifest.json`: image IDs, patient assignments, labels, image hashes,
  dimensions, annotation eligibility, other positive findings, and exclusions.
- `cohort_summary.json`: patient/image/label counts for each partition.

Identical reruns are allowed. Changed settings or source files require a new
output directory; an existing protocol is never silently replaced. This step
checks source-image/contour dimensions. Processor-space transformations and
tokenization must still be verified during the real model pilot.

### Reusing Part 1

The prompt is read from `classification.PROMPT`:

> Is pleural effusion present in this chest X-ray? Answer Yes or No.

The preparation command reads literal settings using Python's AST without
importing plotting libraries. Model execution in the next increment will call
the existing `compute_yes_score`, `parse_prediction`, and `prepare_inputs`
functions, checking source fingerprints against this protocol.

Part 1's continuous score is the first-answer-token softmax over No and Yes.
Keep this score separate from the parsed generated-text answer. Classify parsed
answers as TP/FN/TN/FP against the expert label and retain unparsed responses as
an explicit category; they still have usable continuous scores. Baseline
correctness alone does not establish visual grounding. Grounding remains
unassessed until the controlled image experiments.

Section 2.2 records the model revision, processor revision, device, dtype, and
contextual candidate token IDs for new runs. Check baseline agreement before
comparing perturbation results with Part 1. If tokenization is incompatible,
stop and document the difference instead of substituting a different scorer.

## 2.2: audit the saved baseline and run one discovery case

The audit reads Part 1's CSV directly, or reads `classification/predictions.csv`
inside its ZIP without extracting it. It rechecks the frozen cohort against the
local data, joins predictions by case ID, and rejects duplicates, missing/extra
cases, label differences, nonfinite/out-of-range scores, and answers that no
longer match the inherited parser. The inherited source files stay unchanged.

Install the additional benchmark dependencies into an existing project environment:

```bash
uv pip install --python .venv/bin/python -r requirements-part2.txt
```

Use the existing environment directly for these commands. Plain `uv run` may
resynchronize it to Part 1's dependency list, which omits these extra packages.

```bash
PYTHONPATH=src MPLCONFIGDIR=outputs/.matplotlib \
  .venv/bin/python -m medgemma_grounding.grounding_baseline \
  --predictions /path/to/result.zip \
  --protocol outputs/part2/protocol-v1 \
  --output outputs/part2/baseline-v1
```

Outputs are `baseline_table.csv`, `baseline_cases.json`, `summary.json`, and
`curated_cases.json`. Identical audit reruns are allowed; changed inputs require
a new output directory. The curated examples select up to two cases per answer
category from discovery only, using a fixed hash ranking. This is a qualitative
sample, not an estimate of category prevalence. All annotated positives remain
eligible for localization, including false negatives and unparsed responses.

The supplied archive has 202 matching predictions. ROC-AUC is 0.931442 and
average precision is 0.865113, both using all 202 continuous scores. Average
precision is the Part 1 PR summary; it is not trapezoidal PR-AUC. Binary metrics
use 171 parsed responses: TP 58, TN 83, FP 27, FN 3; accuracy 0.824561 and F1
0.794521. The 31 unparsed responses include 3 positives and 28 negatives. They
are retained in the baseline table and continuous-score evaluation.

### Local model setup

The Mac pilot uses Apple Metal (`mps`) and Part 1's bfloat16 loading policy. This
machine has 16 GiB of unified memory; the checkpoint weights occupy about 8 GiB.
Start with one image and retain PyTorch's memory limits. Do not change precision,
quantize, resize images, or substitute a scorer to make a failed run look like
the original baseline.

Accept access at [the official model repository](https://huggingface.co/google/medgemma-1.5-4b-it)
using the same account as your Hugging Face read token. Put `HF_TOKEN=...` in the
project's ignored `.env` file. Never commit the token or the dataset. The local
checkpoint is stored under the ignored `models/huggingface/` directory; its
revision and cache path are recorded in `models/medgemma_setup.json`.

To download the fixed checkpoint on another authorized machine, run from the
project root after saving its `.env`:

```bash
HF_HOME="$PWD/models/huggingface" .venv/bin/python - <<'PY'
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
load_dotenv('.env')
snapshot_download(
    'google/medgemma-1.5-4b-it',
    revision='91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b',
    token=os.environ['HF_TOKEN'],
    allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.tiktoken'],
    max_workers=2,
)
PY
```

The pilot reads this cache offline, and calls Part 1's input preparation,
generation, parser, and scorer directly:

```bash
PYTHONPATH=src .venv/bin/python -m medgemma_grounding.baseline_pilot \
  --revision 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b \
  --device mps \
  --case-id patient64545_study1_view1_frontal \
  --output outputs/part2/pilot-v1
```

It checks the actual answer boundary against the rendered chat template and
verifies that the inherited Yes/No IDs are matching single-token continuations.
It generates once with the original 16-token limit, scores twice, and compares
with the archived score using preselected tolerances (`atol=1e-5`, `rtol=1e-4`).
It records the response, parsed answer, score, tensor shapes/hashes, processor
and generation configurations, checkpoint commit, package versions, and runtime.
Another run needs a new output directory; discrepancy results are saved and
the command returns a nonzero exit code for review before perturbation work.

Part 1's historical revision, device, and dtype were not recorded in the supplied
archive. Even agreement on this one case does not establish identical runtimes
or reproduce the entire cohort. A disagreement must be investigated and may
require a separately identified Part 2 baseline on the fixed new runtime.

### Verified local pilot

The first run completed successfully on MPS with bfloat16, PyTorch 2.14.0, and
Transformers 5.17.0. Its results are in `outputs/part2/pilot-v1/result.json`.

| Check | Result |
| --- | --- |
| Case | `patient64545_study1_view1_frontal`, discovery, expert label 1 |
| Generated response | Exact match with the archived Part 1 response; parsed Yes |
| Archived Yes score | 0.9998766183853149 |
| New Yes score | 0.9998601675033569 |
| Absolute score difference | 0.000016450881958007812, within the preselected tolerance |
| Repeated score difference | 0.0 |
| Contextual candidate tokens | No = 3771; Yes = 10784; single-token boundary check passed |
| Model input | 284 tokens; RGB image tensor `[1, 3, 896, 896]` |
| Inference time | About 42 seconds for one generation plus two scores, excluding loading |

The original image is 2828 by 2320 pixels. The model uses its inherited processor
to resize it to 896 by 896. Section 2.3 must align masks/perturbations with these
recorded transformations. Baseline classification and numerical agreement do
not establish that the model used the radiologist-marked region.

## Incremental implementation order

| Section | Deliverable | Current state |
| --- | --- | --- |
| 2.1 | Fixed cohort rules, inherited prompt/scoring contract, patient partitions | Preparation command implemented |
| 2.2 | Reconciled Part 1 results and one reproduced real-model baseline | Audit of 202 predictions and one-case Mac pilot passed |
| 2.3 | Original, effusion-region occluded, and matched control-region occluded images; score changes and patch maps | Pending |
| 2.4 | Activations from the same layers/token positions for those conditions, with full metadata | Pending |
| 2.5 | Layerwise distances/similarities, plots, and candidate layers | Pending |
| 2.6 | Recurrent SAE feature observations where transfer fidelity is adequate, or a documented negative result | Pending |

For 2.3, explicitly match control-region area and filling method on the same
image; negative images and shuffled heatmaps are additional controls, not
substitutes. Select patch sizes, fills, and analysis settings on discovery only.
Evaluate localization on annotated positives regardless of the original model
answer, retaining false negatives and unparsed responses. Analyze target-negative
cases separately because they lack a positive effusion region for localization.

For 2.4, coordinate layer and token-position choices with the existing activation
code before the pilot, so 2.5 can compare like with like. For 2.6, use the team's
Gemma Scope transfer results to choose compatible dictionaries; SAE fidelity and
feature recurrence must be measured, not inferred from a convincing example.

Tests for this increment:

```bash
PYTHONPATH=src MPLCONFIGDIR=outputs/.matplotlib .venv/bin/python -m pytest -q \
  tests/test_grounding_protocol.py tests/test_grounding_baseline.py
```
