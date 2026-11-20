# Package migration (2026-11-20)

Scientific protocols and saved artifacts remain unchanged.

| Old module | Primary new module |
| --- | --- |
| `inference.py` | `model.py` |
| `chexlocalize.py` | `data/chexlocalize.py` |
| `sae_reconstruction.py` | `sae/reconstruction.py` |
| `feature_interventions.py` | `interventions/features.py` |
| `feature_controls.py` | `interventions/controls.py` |
| `attention_knockout.py` | `interventions/attention.py` |
| `gemma_scope.py` | `experiments/run_gemma_control.py` |
| `medgemma_scope.py` | `experiments/run_sae_transfer.py` |
| `medgemma_scope_layers.py` | `experiments/run_sae_layers.py` |
| `medgemma_features.py` | `experiments/compute_sae_activations.py` |
| `feature_analysis.py` | `analysis/feature_discovery.py` |
| `medgemma_interventions.py` | `interventions/feature_runtime.py` |
| `medgemma_preflight.py` | `experiments/diagnose_attention.py` |
| `medgemma_discovery.py` | `experiments/run_feature_discovery.py` |
| `medgemma_targeted.py` | `experiments/run_targeted_features.py` |
| `medgemma_attention.py` | `experiments/run_attention_windows.py` |
| `medgemma_routing.py` | `experiments/run_attention_layers.py` |
| `attention_analysis.py` | `analysis/attention_windows.py` |
| `routing_analysis.py` | `analysis/attention_layers.py` |
| `targeted_analysis.py` | `analysis/targeted_features.py` |
| `feature_viewer.py` | `visualization/app.py` |

Shared helpers are also extracted into `utils/io.py`, `utils/numerics.py`,
`sae/cache.py`, `data/image_positions.py`, `interventions/protocol.py`,
`interventions/feature_runtime.py`, and `analysis/feature_effects.py`.
Viewer overlays and study panels are separate modules under `visualization/`.

Exact historical runtime archived before edits:
`results/maintenance/layout_20261120/historical_runtime/medgemma_grounding/`.
Its separate `snapshot.json` records all 22 original source hashes and verifies
both frozen E-009 and E-010 protocols. Original result directories are retained.

## Structural references

These are organizational patterns adapted to this small repository; they are
not evidence for a new scientific method or a reason to copy an entire framework.

- [Pathways, `vlm_spatial/model.py`](https://github.com/israfelsr/vlm-pathways/blob/470944b3f6b44901a9055cf8d91aef18cee21671/vlm_spatial/model.py): shared model/data/hooks separated from experiments.
- [DualPath-VLM, `experiments/run_patching.py`](https://github.com/jiaxin26/DualPath-VLM/blob/c1cf59e6b632db25bddec34a6f3a8578788e75e0/experiments/run_patching.py): experiment orchestration imports intervention and analysis code.
- [SAEBench, `evals/sparse_probing/main.py`](https://github.com/adamkarvonen/SAEBench/blob/8042bb3828c6340da8d12062324e92b2077c571c/sae_bench/evals/sparse_probing/main.py): grouped evaluation responsibilities and shared activation utilities. Separate config/output dataclasses are unnecessary here.
- [Line of Sight, `compute_sae_activations.py`](https://github.com/multimodal-interpretability/multimodal-saes/blob/8a466949994700be3eb67e3364375083d7697562/probing_approximation/compute_sae_activations.py): recognizable extraction naming, separate SAE evaluation and steering.
- [PyPA src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/): keep the existing installable package boundary.

## Commands

For Python commands, retain arguments and replace the module after `python -m`:

| Previous module suffix | New module suffix (under `medgemma_grounding`) |
| --- | --- |
| `inference` | `experiments.run_inference` |
| `chexlocalize` | `experiments.inspect_data` |
| `gemma_scope` | `experiments.run_gemma_control` |
| `medgemma_scope` | `experiments.run_sae_transfer` |
| `medgemma_scope_layers` | `experiments.run_sae_layers` |
| `medgemma_features` | `experiments.compute_sae_activations` |
| `feature_analysis` | `analysis.feature_discovery` |
| `medgemma_preflight` | `experiments.diagnose_attention` |
| `medgemma_interventions` | `experiments.run_feature_pilot` |
| `medgemma_discovery` | `experiments.run_feature_discovery` |
| `medgemma_targeted` | `experiments.run_targeted_features` |
| `medgemma_attention` | `experiments.run_attention_windows` |
| `medgemma_routing` | `experiments.run_attention_layers` |

Streamlit: replace `src/medgemma_grounding/feature_viewer.py` with
`src/medgemma_grounding/visualization/app.py`.

## Historical compatibility

New protocols hash every package `.py` file recursively, with package-relative
paths. Added, changed and removed files invalidate resume. Historical protocol
files and measurements are never migrated to new hashes.

E-009/E-010 report, verify, existing-window analysis and completed resume run in
a separate Python interpreter with the exact archived package. A temporary
working tree exposes the original `src/` filenames, allowing the independent
artifact inventories to verify the original bytes. Model and SAE loaders are
blocked; incomplete attention runs are rejected before execution. No second
active implementation is maintained under the old flat source paths.

To run the original independent audits after migration:

```bash
uv run python -m medgemma_grounding.experiments.replay \
  --module medgemma_attention --stage run --output-dir results/attention_depth_v1 \
  --audit-script results/attention_depth_v1/verify_results.py \
  --audit-output /tmp/medgemma-e009-audit.json
uv run python -m medgemma_grounding.experiments.replay \
  --module medgemma_routing --stage verify --output-dir results/attention_errors_v3 \
  --audit-script results/attention_errors_v3/audit_results.py \
  --audit-output /tmp/medgemma-e010-audit.json
```

E-007/E-008 offline reporting also dispatches to the archived implementation.
Earlier E-004/E-005/E-006 source pins already differ from the pre-refactor HEAD;
those recorded pins are retained and mismatches fail closed. The current
feature-discovery reporter also refuses to overwrite historical E-004 provenance. Their frozen
reports and failed-run evidence remain inspectable. A command mapping does not
authorize resuming those old runs with new code. Exact older execution would
require an independently verified matching historical source snapshot.

Keep the snapshot with local result backups: it is ignored, like other generated
research artifacts. Fresh clones include the new code/tests but no local data,
weights, reports or historical runtime. Scientific results and local research
notes retain their original directories and historical commands.

## Removal policy

The old 21 flat source locations were removed after their implementations and
callers were migrated. No scientific workflow was declared unused. Original
snapshots, failed runs, protocols, measurements, images, proposals and slides
were preserved. No dependency changed. The root dataset ignore rule is now anchored as `/data/`
so the new `src/medgemma_grounding/data/` and `tests/data/` code is tracked;
the original dataset remains ignored. No orphaned research
artifact was deleted based only on lack of a Python import. Orphaned Python
bytecode for the moved source/test filenames was removed; it is regenerable.
