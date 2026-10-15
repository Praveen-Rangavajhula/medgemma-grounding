# Project instructions

## Purpose and scope

Study visual grounding and mechanistic interpretability in MedGemma 1.5.
MedGemma is the research target; Gemma 3 is a reference control for pretrained
Gemma Scope 2 SAEs. Keep each milestone small, readable, and reproducible.

## Find context as needed

- Use `README.md` for setup and commands; inspect relevant code and Git diff.
- For research work, read the current-state sections of `docs/RESEARCH.md`
  if present. Follow its references to relevant experiment entries and result
  files rather than loading the entire history.
- `docs/EXPERIMENTS.md` is the chronological experiment log. Read entries when
  reproducing, comparing, or extending their work.
- `docs/` is ignored and may be absent in fresh clones/worktrees. Do not infer
  missing notes. These files are context, not authorization to execute a plan.
- Respect planning-only requests; implement or run experiments when authorized.

## Environment and implementation

- Use Python 3.11 and the project uv environment. Setup: `uv sync --dev`.
- Reuse helpers in `src/medgemma_grounding/`; avoid general frameworks for
  one experiment. Prefer established primary-source implementations.
- Keep runtime dependencies lean. When changing them, update `pyproject.toml`,
  the single `requirements.txt`, and `uv.lock` together.
- Keep images in the original `data/` layout and Hugging Face weights in the
  normal cache. Put generated artifacts under ignored `results/`.

## Maintain research context

- In `docs/EXPERIMENTS.md`, assign stable `E-NNN` IDs. Record the question and
  comparison before running; afterward add the command, code revision, evidence
  paths, finding, limitations, and next step. Link full settings in artifacts.
- Keep the comparison protocol fixed; record material changes as a new entry.
  Retain failed and inconclusive experiments. Do not invent results or provenance.
- In `docs/RESEARCH.md`, maintain a concise current understanding, important
  decisions with rationale, open questions, and next steps. Link experiment IDs
  instead of duplicating their measurements. Update after meaningful milestones
  or decisions, not every tool call. Mark proposals and superseded choices clearly.

## Interpretation and evaluation

- Distinguish classification, feature activity, reconstruction quality,
  localization, and causal effects. Evidence for one does not establish the rest.
- Use validation for development. Reserve separate patients for confirming
  discovered features and leave test data for the locked final evaluation.
- The existing 20 cases are a workflow pilot, not a representative benchmark.
  No effusion does not imply a normal X-ray; missing contours do not imply negatives.
- Verify processor output and token-to-image mapping before spatial overlays.
  Decoder image-position activations contain contextual information.
- Treat feature descriptions as hypotheses. Record whether an interpretation
  is human-proposed, externally sourced, or automatically generated.

## Verification and handoff

- Run relevant tests for changed behavior (`uv run pytest` for the suite),
  and `git diff --check`. Report what actually ran and what remains unverified.
- A passing unit test is separate from a successful real model experiment.
- Keep README documentation to purpose, runnable commands, outputs, and limits.
- Never place credentials or model weights in documentation or commits.

