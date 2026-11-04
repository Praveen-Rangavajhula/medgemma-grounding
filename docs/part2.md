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

The model revision, processor revision, device, dtype, and contextual candidate
token IDs have not yet been verified for the new runs. Section 2.2 must record
them and check baseline agreement before any perturbation results are compared
with Part 1. If tokenization is incompatible, stop and document the difference
instead of silently substituting a different scoring rule.

## Incremental implementation order

| Section | Deliverable | Current state |
| --- | --- | --- |
| 2.1 | Fixed cohort rules, inherited prompt/scoring contract, patient partitions | Preparation command implemented |
| 2.2 | Reconciled Part 1 results and one reproduced real-model baseline | Next |
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
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_grounding_protocol.py
```
