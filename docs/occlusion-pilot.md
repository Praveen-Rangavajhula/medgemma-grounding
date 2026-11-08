# Section 2.3: one-case masking pilot

This increment adds masking and a patch-occlusion experiment for the already
verified discovery case, `patient64545_study1_view1_frontal`. Part 1's inference,
prompt, parser, and Yes-score functions are called directly and remain unchanged.
This is a pilot implementation, not a completed cohort evaluation. Confirmation
and official test images are not used for inference or selecting settings.

## Frozen inputs and mask construction

`configs/part2/pilot-case.json` declares settings before occlusion scores are
observed. The preparation command checks the section 2.1 protocol, section 2.2
audit and baseline, source fingerprints, image hashes, and annotation hashes.
It decodes the official COCO RLE in `gt_segmentations_val.json`; contours are not
redrawn and a bounding box is not substituted for the expert mask. This follows
the [official segmentation decoding convention](https://github.com/rajpurkarlab/cheXlocalize/blob/master/eval.py).

For this case:

- Original image: width 2828, height 2320. Target mask: 253,877 pixels, about
  3.87% of the image.
- Three full-shape translations: `(dx, dy) = (-848, 0), (-848, -406), (-141, -522)`.
  They retain exactly the target's native pixel count, have zero target overlap,
  remain inside a predeclared coarse chest window, and are never clipped or wrapped.
- The chest window is a case-specific visual constraint, not an anatomical lung
  segmentation. The controls cover different anatomy and can overlap other
  abnormalities. Controls 1 and 2 share 13,429 pixels; they are not independent
  samples for significance testing.
- Mean fill: a single gray level, **129/255**, computed once from the original
  image and reused for every target, control, and patch.
- Blur fill: Pillow's GaussianBlur approximation, sigma 16 pixels at the model
  scale, mapped to native x/y radii of approximately 50.5 and 41.43 pixels. Blur
  the original image once and replace pixels only inside each mask. Blur retains
  coarse evidence and is a sensitivity analysis, not complete evidence removal.

Masks are applied to native RGB pixels. Every outside pixel is preserved
exactly, and files are saved as lossless PNG. The same Part 1 processor then
produces the model's 896-by-896 image. A processor-only CPU preparation verifies
that the original input tensors match the recorded section 2.2 hashes. Review
figures show those actual processed image tensors, inverted from normalization.

Nearest-neighbor masks are used only for drawing outlines and reporting model-
scale area. They are not multiplied into already-normalized model inputs. Image
resampling can spread changes across mask boundaries. Target and control areas
at native resolution match exactly; nearest-resized target area is 31,017 pixels,
versus 31,144, 31,128, and 31,123 for controls (all less than 0.5% different).
The runner also records actual changed model pixels and pixel-tensor hashes.

## Commands

Use the existing project environment and cached checkpoint from section 2.2:

```bash
uv pip install --python .venv/bin/python -r requirements-part2.txt

PYTHONPATH=src MPLCONFIGDIR=outputs/.matplotlib \
  .venv/bin/python -m medgemma_grounding.occlusion_pilot prepare
```

Inspect `outputs/part2/occlusion-prepared-v1/mask_review.png` before looking at
occlusion scores. Check image/mask orientation, target alignment, control
placement, and the mean/blur variants. The implemented pilot's overlay was
visually inspected by Codex; this is geometry QA, not radiologist validation of
the controls. The run records a review note along with the preparation hash:

```bash
PYTHONPATH=src MPLCONFIGDIR=outputs/.matplotlib \
  .venv/bin/python -u -m medgemma_grounding.occlusion_pilot run \
  --review-note 'Describe the visual mask review performed before scoring.'
```

For a new experiment, choose new `--output` directories. To continue an
interrupted run, repeat its exact command and review note with `--resume`.
Completed conditions are not recomputed. Resuming checks input/source hashes,
runtime versions, task order, and a newly computed baseline before continuing.
The command does not overwrite an unrelated or differently configured run.

The model runs offline at the section 2.2 checkpoint, device, precision, and
attention implementation. Changed dependencies or mismatched baseline tensors
cause a failure instead of silently mixing settings.

## Conditions and saved results

There are **139 scored conditions**:

1. Original image and an empty-mask/no-op check.
2. Target and three controls under mean fill and blur: eight region conditions.
3. An annotation-independent 8-by-8 grid under each fill: 128 patch conditions.
4. Original image again to detect baseline drift at the end.

Grid edges divide the original image into approximately equal cells. Rounding
ensures every native pixel belongs to exactly one cell, including edge pixels.
The grid corresponds to approximately 112-by-112 pixels in the resized view;
it is an image grid, not a claim about the model's visual-token boundaries.
Neither annotation overlap nor model scores are used to select patch locations.

Generation uses Part 1's 16-token limit for the original and eight region
conditions. Patches and sanity checks only require the continuous score, so
their response fields are empty by design. Generated but unparseable responses
remain in the score analysis. Each completed condition is flushed to
`outputs/part2/occlusion-run-v1/scores.jsonl` for recovery.

The run produces:

- `run.json`: review note, input hash, runtime, and start time.
- `scores.jsonl` and `scores.csv`: condition, fill, score, signed drop, optional
  response, token/pixel checks, native patch bounds, and timing.
- `signed_drop_mean.npy` and `signed_drop_blur.npy`: raw signed 8-by-8 maps.
- `summary.json`: baseline sanity checks, region contrasts, and descriptive
  patch-localization summaries.
- `results.png`: shared-scale signed heatmaps with the expert outline, plus
  target/control score-drop plots.

Patch images can be reconstructed from the frozen original, replacement method,
and saved bounds. Original, target, and control images are saved explicitly in
the preparation directory. All dataset/model/experiment artifacts stay ignored
by Git.

## Interpretation and relation to the paper

For every mask, `signed_drop = original_yes_score - occluded_yes_score`.
Positive values mean the score decreased; negative values mean it increased.
The region comparison is `target_drop - mean(control_drops)`, reported
separately for mean and blur, with all individual control values retained.

The heatmap preserves signed changes. A separate descriptive measure asks what
fraction of positive score-drop mass lies inside the expert mask, treating each
patch's score drop as a constant density and weighting by native pixel area.
It is compared with the mask's 3.87% area fraction. If there is no positive mass,
the fraction is null, not zero. No thresholded IoU or tuned localization claim
is reported from this first case.

[Sadanandan and Behzadan's paper](https://arxiv.org/html/2607.18577v2) uses a
16-by-16 occlusion grid and answer-margin shifts. Our pilot adapts the occlusion
idea with an 8-by-8 grid, the inherited Yes-vs-No probability score, and a second
blur fill. These differences are explicit; this is not a numerical reproduction
of that paper. Scores near zero or one can saturate and compress differences.

A region score change shows sensitivity to an artificial image perturbation.
It does not establish clinical causality or reliable grounding across patients.
Use additional discovery cases, including false negatives and unparsed answers,
to assess robustness before freezing settings for confirmation. Retain negative
cases for separate target-negative analysis; they are not empty-mask positive
localization examples. Later activation work should reuse the saved conditions.

## First completed pilot

The Mac run completed all 139 conditions. The original Yes score was
**0.9998601675**; the empty-mask and final repeated-original checks differed by
**0.0**. Recorded condition execution took about 15.3 minutes, excluding model
loading and preparation. All eight region variants still generated a parsed Yes.

| Measurement | Mean fill | Blur fill |
| --- | ---: | ---: |
| Full target-mask score drop | -0.000125051 | -0.000132918 |
| Average control-mask score drop | -0.000134945 | -0.000110110 |
| Target drop minus average control drop | +0.000009894 | -0.000022809 |
| Largest individual patch score drop | 0.059946835 | 0.025817573 |
| Positive patch-drop mass inside expert mask | 23.19% | 16.81% |
| Expert-mask area fraction | 3.87% | 3.87% |

Negative full-region drops mean that the score **increased** after masking. The
small target-versus-control contrast changes sign between fills. In both patch
sweeps, the largest drop occurs at zero-based row 4, column 6: native bounds
`[2121, 1160, 2474, 1450]`. About 13.34% of that patch overlaps the expert mask,
so its effect cannot be assigned exclusively to annotated pixels. Its occluded
Yes scores are approximately 0.939913 (mean) and 0.974043 (blur).

These observations are mixed: patch sensitivity is concentrated near parts of
the annotation, but covering the full annotation does not reduce the score.
The perturbations cover different pixels and can have nonlinear effects.
The positive-mass fractions are descriptive comparisons with area, not
significance tests. Preserve both results; do not label this case simply
"grounded" or "ungrounded" from this pilot. Additional discovery cases and
fill/grid sensitivity checks are needed before a confirmation protocol is frozen.

Figures were visually inspected. All 139 CSV/JSONL rows, source and artifact
fingerprints, baseline checks, and localization summaries were verified; the
mass fractions were independently recomputed by summing the full native pixel
maps. Original Part 1 source files remain unchanged.

## Checks

```bash
PYTHONPATH=src MPLCONFIGDIR=outputs/.matplotlib \
  .venv/bin/python -m pytest -q tests/test_masking.py \
  tests/test_grounding_baseline.py tests/test_grounding_protocol.py
```

Tests cover RLE orientation, strict translation, outside-pixel preservation,
fixed fills, full grid coverage, signed summaries and zero-mass behavior,
baseline/no-op failures, malformed checkpoints, and interruption/resumption
through the inherited scoring and generation functions.
