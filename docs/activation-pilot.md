# Sections 2.4 and 2.5: activations and layer comparisons

This increment follows the completed section 2.3 discovery-case pilot. It uses
the exact saved original, target-mask and three control-mask images, under both
mean and blur fills. It adds two new modules without editing Part 1 or the
frozen masking modules. Confirmation and official test images remain unused.

## 2.4: collect comparable internal states

`activation_pilot` records the output of all 34 language-decoder layers, indexed
0 through 33, at every position in the original input sequence. The hook paths
are `model.language_model.layers.N`, taken from the inherited
`sae_reconstruction.checkpoint_spec`. These are residual states after each
layer's attention and MLP additions, before the model's final RMSNorm. They are
not attention weights, vision-encoder outputs, or SAE features.

The hooks run during the unchanged Part 1 `compute_yes_score` call. Each hook
returns `None`, copies its output to CPU and casts it to float32 for storage.
No normalization, projection or intervention is applied. Bfloat16 values are
exactly representable in float32. Hooks are removed even if inference fails.

The pilot collects 11 conditions:

- Original image, plus an identical-input no-op check.
- Effusion region and three controls, each under mean and blur fill.
- Original image again, to test repeatability at the end.

The 128 grid-patch conditions are not included in this collection. Region
conditions are all retained, regardless of whether their score increased or
decreased. All layers and positions were selected before inspecting activation
distances.

Each condition has two scoring passes. The first records only the model's
next-token logits; the second also records all decoder residuals. Full-vocabulary
logits must agree (`atol=1e-5`, `rtol=1e-5`), and their Yes scores must agree
(`atol=1e-7`, `rtol=1e-6`). Scores must also reproduce section 2.3 within its
existing tolerance (`atol=1e-5`, `rtol=1e-4`). No-op and end-repeat activations
must agree with the first capture (`atol=1e-6`, `rtol=1e-5`).

Input hashes verify identical pixel tensors to section 2.3 and identical
non-image tensors across conditions. The checkpoint revision, processor,
device, dtype, attention implementation, package versions, prompt and token
boundary checks are inherited from the verified baseline. Generated answers
are linked from section 2.3 and explicitly marked as reused; these new passes
record the prefill rather than generate another answer.

Run from the project root using the installed environment and cached model:

```bash
PYTHONPATH=src MPLCONFIGDIR=outputs/.matplotlib \
  .venv/bin/python -u -m medgemma_grounding.activation_pilot
```

On this Mac the command needs access to the MPS GPU. Default output is
`outputs/part2/activations-v1/`. Repeat the same command with `--resume` after
an interruption. Complete conditions are published atomically, checked by hash
and skipped on resume; incomplete staging files are never counted as completed
conditions. A changed source, input or runtime requires a new output
directory. A completed resume verifies artifacts without loading the model.

Artifacts:

- `run.json`: case, prompt/scoring contract, provenance, runtime, hooks, tensor
  axes, full input token IDs, and exact token-group positions.
- One directory per condition with `residuals.npz` (array `resid_post`, axes
  `[layer, input_position, hidden_dimension]`), `next_token_logits.npy`, and
  `condition.json` with hashes, score, linked response and numerical checks.
- `summary.json`: completion status, condition metadata hashes and sanity checks.

The saved arrays are numeric and load with `allow_pickle=False`. They retain
the original token-level information for later compatible SAE analysis. They
do not establish that any Gemma Scope dictionary transfers faithfully.

For `patient64545_study1_view1_frontal`, each array is `[34, 284, 2560]`.
There are 256 image positions and 25 text positions. The decision position is
283 (token ID 107, the newline ending the generation prefix), which is also
included among the text positions. These explicit indices prevent accidentally
comparing a prompt state with a generated-answer state.

## 2.5: compare representations across conditions

```bash
PYTHONPATH=src MPLCONFIGDIR=outputs/.matplotlib \
  .venv/bin/python -m medgemma_grounding.layer_comparison
```

This step reads the completed, verified capture without loading MedGemma.
Default output is `outputs/part2/layers-v1/`. Use a new `--output` if the inputs
or analysis code change; an identical rerun validates and reuses existing files.

Three token groups are analyzed separately at the same positions in every image:

1. **Decision:** the final input position, whose state predicts the first answer
   token. This is not the state after a generated “Yes” or “No.”
2. **Image:** positions equal to the configured image-placeholder token ID,
   matching the inherited SAE code's token grouping.
3. **Text:** valid input positions excluding image and special tokens. The
   decision position can also belong to this group; the groups are not claimed
   to be three independent samples.

For every layer and group, compare each masked condition with the original:

- Raw L2/Frobenius distance and RMSE.
- Relative L2: `||H_masked − H_original||_F / ||H_original||_F`.
- Mean cosine similarity of corresponding token vectors, plus `1 − cosine`.
  Cosine is computed per token before averaging, not from a pooled mean vector.

Zero reference norms produce null relative L2. Zero-norm token pairs are omitted
from cosine with their valid count reported; all-zero groups produce null. All
arithmetic for the metrics uses float64. Layer normalization is not applied to
the stored states. Relative L2 and cosine help interpret raw distances when
residual magnitudes vary across layers.

Within each fill, subtract the average of the three control distances from the
target distance. All individual control results remain available. The controls
are anatomically different, partially overlapping regions, not independent
replicates, so the report does not calculate confidence intervals or p-values.

The exploratory candidate rule is fixed in the analysis code: for each token
group, retain layers with positive target-minus-mean-control relative L2 under
both fills, rank by the smaller of the two contrasts and return up to three.
Ties use the smaller layer index. An empty list is a valid finding. Candidates
are for inspection on additional discovery cases, not a validated layer choice
for confirmation or a claim of causal importance.

Outputs include `layer_metrics.csv`, `sanity_metrics.csv`,
`target_control_contrasts.json`, `summary.json`, and three figures:
`layer_profiles.png`, `cosine_profiles.png`, `target_control_contrasts.png`.
Target/control curves share a y-axis across fills within each token group.

## Completed first pilot

The real MPS/bfloat16 run completed all 11 conditions at the inherited pinned
model revision. Every score agreed with section 2.3. Full-vocabulary passive-hook
logit differences, no-op activation differences and end-repeat activation
differences were all exactly zero. The numeric artifacts occupy about 502 MiB
locally and remain ignored by Git.

The analysis contains 816 region/group/layer rows and 204 sanity rows. All three
figures were inspected for legibility and shared axes. The relative-L2 candidate
rule produced the following zero-based indices:

| Token group | Candidate layers, ranked by minimum contrast across fills |
| --- | --- |
| Final pre-answer position | 3, 14 |
| Image tokens | None |
| Prompt text | 8, 9, 11 |

At the final pre-answer position, layer 3 has target-minus-mean-control relative
L2 of 0.004891 under mean fill and 0.005653 under blur. Layer 14 has 0.001059 and
0.002438, respectively. These are differences between activation-distance
ratios, not changes in the Yes score. The larger late-layer changes are strongly
fill-dependent: target/control contrasts become negative under mean fill and
positive under blur at the final pre-answer position. This does not justify
calling the candidates effusion-specific or causal. Section 2.3's full-region
masking still increased the already saturated Yes score in both fills.

All 49 relevant tests passed. The only warning was the pre-existing upstream
pycocotools/NumPy `__array__` deprecation. Raw activations and exact per-layer
metrics are retained so further discovery cases can test these observations.

## Limits and next work

This completes a one-case implementation and execution of sections 2.4–2.5,
not the cohort-level research study. A larger representation change need not
lower the effusion score or reflect effusion-specific evidence. Image-token
distances are not pixel localization maps. Single-case probes, clustering and
correlations treating the 34 dependent layers as independent observations would
not support reliable conclusions and are not included.

Before section 2.6, inspect the team's quantitative SAE-transfer results and
choose only compatible layers/dictionaries. Broaden discovery evaluation before
freezing layer or feature choices for patient-disjoint confirmation.

## Verification

```bash
PYTHONPATH=src MPLCONFIGDIR=outputs/.matplotlib \
  .venv/bin/python -m pytest -q tests/test_activation_pilot.py \
  tests/test_masking.py tests/test_grounding_protocol.py \
  tests/test_grounding_baseline.py tests/test_chexlocalize.py \
  tests/test_import.py tests/test_medgemma_scope.py tests/test_medgemma_scope_layers.py
```

Tests cover passive tuple/tensor hooks and cleanup on failure, token positions,
analytically known distances, zero-norm behavior, target/control contrasts,
cross-fill candidate rules, interrupted collection/resume, complete-run reuse,
analysis output and corrupted-artifact rejection.
