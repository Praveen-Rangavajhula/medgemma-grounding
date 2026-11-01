"""E-006: diagnose/fix attention consistency on exactly E-005's two preflight cases.

This command cannot launch the pilot or a cohort stage. A supported eager text
backend is tested only after the original SDPA traces support the hypothesis.
Vision stays frozen on SDPA for every call. Weights, BF16 and tolerances stay fixed.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import traceback
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import torch

from medgemma_grounding.feature_interventions import (
    GROUPS, attention_settings, capture_gradient, configure_attention,
    feature_removal, tensor_diagnostics,
)
from medgemma_grounding.medgemma_features import (
    answer_score, atomic_json, file_sha256, identity_hash, load_discovery_case,
)
from medgemma_grounding.medgemma_interventions import (
    identity_diagnostics, outcome, prepared_inputs, read_json,
)
from medgemma_grounding.sae_reconstruction import encode_decode, replace_residual

OUTPUT = Path("results/gemma_scope/interventions_v2_preflight")
FIX = {"text_config": "eager", "vision_config": "sdpa"}
MODES = ("inference", "no_grad", "grad_no_leaf", "gradient")


def tree_hashes(directory):
    return {str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def bounded_source(source, failed):
    """Read only the two frozen preflight shards; never audit/load other images."""
    previous = read_json(failed / "protocol.json")
    original = read_json(source / "protocol.json")
    for protocol in (previous, original):
        if protocol["protocol_hash"] != identity_hash({k: v for k, v in protocol.items() if k != "protocol_hash"}):
            raise ValueError("Source protocol checksum mismatch")
    case_ids = previous["pilot"]["preflight"]
    if (len(case_ids) != 2 or len(set(case_ids)) != 2
            or case_ids != [previous["pilot"]["allocation"]["TP"][0], previous["pilot"]["allocation"]["TN"][0]]
            or previous["source_protocol_hash"] != original["protocol_hash"]
            or Path(previous["source_directory"]).resolve() != source.resolve()):
        raise ValueError("Expected exactly the two existing TP/TN preflight cases")
    cases, hashes = [], {}
    for case_id in case_ids:
        record, _ = load_discovery_case(source, case_id, original)
        if record["partition"] != "discovery" or "/val/" not in record["image_path"]:
            raise ValueError("Preflight may use discovery validation inputs only")
        for extension in ("json", "pt"):
            name = f"cases/{case_id}.{extension}"
            digest = file_sha256(source / name)
            if previous["source_artifact_sha256"][name] != digest:
                raise ValueError("Preflight source changed since E-005")
            hashes[name] = digest
        cases.append(record)
    return previous, cases, hashes


def protocol_for(source, failed, output):
    source, failed, output = (Path(p).resolve() for p in (source, failed, output))
    if any(output == p or output in p.parents or p in output.parents for p in (source, failed)):
        raise ValueError("Use a separate new output directory; preserve discovery and interventions_v1")
    previous, cases, hashes = bounded_source(source, failed)
    keys = ("model_id", "model_revision", "processor_revision", "sae_revision", "sae_release", "sae_id", "hook", "prompt", "scoring")
    identity = {"experiment": "E-006", "scope": "two-case numerical verification only",
                "source_directory": str(source), "failed_directory": str(failed),
                "source_protocol_hash": previous["source_protocol_hash"], "previous_protocol_hash": previous["protocol_hash"],
                **{k: previous[k] for k in keys}, "case_ids": [r["case_id"] for r in cases],
                "source_sha256": hashes, "preserved_v1_sha256": tree_hashes(failed),
                "image_sha256": {r["case_id"]: r["image_sha256"] for r in cases},
                "original_attention": "sdpa", "conditional_fix": FIX,
                "execution_modes": list(MODES), "rtol": 1e-5, "atol": 1e-5,
                "runtime": {"device": "mps", "dtype": "torch.bfloat16", "sae": "cpu/float32", "batch_size": 1,
                            "use_cache": False, "logits_to_keep": 1, "frozen_weights": True},
                "verification_edits": "one largest-decoded-norm active feature per group at 0.5/1.0, not a shortlist or calibration",
                "stop": "after these two cases; no pilot/cohort/confirmation/test execution",
                "versions": {k: version(k) for k in ("torch", "transformers", "sae-lens")},
                "torch_git_version": torch.version.git_version,
                "references": {"torch_dispatch": "https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/transformers/attention.cpp#L821",
                               "attention_api": "https://huggingface.co/docs/transformers/attention_interface"},
                "code_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in (
                    "medgemma_preflight.py", "feature_interventions.py", "medgemma_interventions.py",
                    "inference.py", "medgemma_features.py", "sae_reconstruction.py")}}
    path = output / "protocol.json"
    if path.exists():
        saved = read_json(path)
        if (saved["protocol_hash"] != identity_hash({k: v for k, v in saved.items() if k != "protocol_hash"})
                or any(saved.get(k) != v for k, v in identity.items())):
            raise ValueError("Preflight resume settings/sources differ; use a new directory")
        return saved, cases
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty without a valid preflight protocol")
    identity.update(code_revision=subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip(),
                    code_dirty=bool(subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=True).stdout))
    identity["protocol_hash"] = identity_hash(identity)
    atomic_json(path, identity)
    return identity, cases


def capture_mode(model, inputs, hook_name, candidates, mode, *, profile=False):
    """One mode, layer-14/15 snapshots and optional CPU operator trace on MPS."""
    if mode not in MODES:
        raise ValueError("Unknown execution mode")
    captured, downstream = [], []

    def record(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.detach().cpu().float().clone())

    def after(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        downstream.append(value.detach().cpu().float().clone())

    next_hook = hook_name.rsplit(".", 1)[0] + ".15"
    handles = [model.get_submodule(next_hook).register_forward_hook(after)]
    if mode != "gradient":
        handles.append(model.get_submodule(hook_name).register_forward_hook(record))
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if profile else nullcontext()
    try:
        with profiler as trace:
            if mode == "gradient":
                hidden, logits, gradient = capture_gradient(model, inputs, hook_name, candidates)
                captured.append(hidden)
            else:
                context = {"inference": torch.inference_mode, "no_grad": torch.no_grad, "grad_no_leaf": torch.enable_grad}[mode]
                with torch.inference_mode(False), context():
                    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits.detach().cpu().float()
                gradient = None
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 1 or len(downstream) != 1:
        raise RuntimeError("Expected one layer-14 and one layer-15 invocation")
    tensors = {"hidden": captured[0], "next_hidden": downstream[0], "logits": logits}
    if gradient is not None:
        tensors["gradient"] = gradient
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Nonfinite preflight tensors")
    operators = ({event.key: event.count for event in trace.key_averages()
                  if "scaled_dot_product" in event.key and event.key.startswith("aten::")} if profile else {})
    return tensors, operators


def feature_compatibility(original, changed, masks):
    result = {"all": tensor_diagnostics(original, changed)}
    for group, mask in masks.items():
        result[group] = tensor_diagnostics(original[mask], changed[mask])
    old_active, new_active = original != 0, changed != 0
    result.update(activated_entries=int((~old_active & new_active).sum()),
                  deactivated_entries=int((old_active & ~new_active).sum()),
                  original_nonzeros=int(old_active.sum()), changed_nonzeros=int(new_active.sum()))
    return result


def mode_comparison(baseline, changed, candidates, name):
    return {"logits": identity_diagnostics(baseline["logits"], changed["logits"], candidates, name),
            "layer14": tensor_diagnostics(baseline["hidden"], changed["hidden"]),
            "layer15": tensor_diagnostics(baseline["next_hidden"], changed["next_hidden"])}


def backend_evidence(comparisons, traces):
    special = "aten::_scaled_dot_product_attention_math_for_mps"
    generic = "aten::_scaled_dot_product_attention_math"
    no_change_without_leaf = all(comparisons[m]["logits"]["allclose"] for m in ("no_grad", "grad_no_leaf"))
    gradient_changed = not comparisons["gradient"]["logits"]["allclose"]
    boundary_same = comparisons["gradient"]["layer14"]["allclose"]
    downstream_changed = not comparisons["gradient"]["layer15"]["allclose"]
    trace_switch = (traces["gradient"].get(special, 0) < traces["inference"].get(special, 0)
                    and traces["gradient"].get(generic, 0) > traces["inference"].get(generic, 0))
    return {"no_change_without_leaf": no_change_without_leaf, "gradient_changed": gradient_changed,
            "boundary_same": boundary_same, "downstream_changed": downstream_changed,
            "operator_path_changed": trace_switch,
            "confirmed": bool(no_change_without_leaf and gradient_changed and boundary_same and downstream_changed and trace_switch)}


def weight_identity(model):
    return [(name, id(p), p._version, str(p.dtype), str(p.device)) for name, p in model.named_parameters()]


def save_case(output, protocol, record, payload):
    path = output / f"{record['case_id']}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    atomic_json(path.with_suffix(".json"), {**record, "protocol_hash": protocol["protocol_hash"], "tensor_sha256": file_sha256(path)})


def load_case(output, protocol, case_id):
    if case_id not in protocol["case_ids"]:
        raise ValueError("Only the two preflight IDs are accessible")
    record = read_json(output / f"{case_id}.json")
    path = output / f"{case_id}.pt"
    if record["protocol_hash"] != protocol["protocol_hash"] or record["tensor_sha256"] != file_sha256(path):
        raise ValueError("Preflight shard identity/checksum mismatch")
    return record, torch.load(path, map_location="cpu", weights_only=True)


def report(output, protocol):
    records = [load_case(output, protocol, case_id)[0] for case_id in protocol["case_ids"]
               if (output / f"{case_id}.json").exists()]
    if tree_hashes(Path(protocol["failed_directory"])) != protocol["preserved_v1_sha256"]:
        raise ValueError("Preserved interventions_v1 changed")
    checks_path = output / "runtime_checks.json"
    checks = read_json(checks_path) if checks_path.exists() else None
    if checks and checks["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Runtime check identity mismatch")
    result = {"experiment": "E-006", "protocol_hash": protocol["protocol_hash"], "cases": records,
              "complete": len(records) == 2 and all("consistent" in r for r in records),
              "backend_hypothesis_confirmed": len(records) == 2 and all(r["backend_evidence"]["confirmed"] for r in records),
              "repair_mechanics_passed": bool(checks and checks["parameter_identity_versions_unchanged"]
                  and checks["parameter_gradients_unset"] and len(records) == 2
                  and all(r.get("consistent", {}).get("passed", False) for r in records)),
              "runtime_checks": checks,
              "v1_preserved": True, "pilot_or_cohort_authorized": False,
              "limits": "Two-case numerical verification; no feature calibration, representative compatibility or causal conclusions"}
    atomic_json(output / "preflight_report.json", result)
    return result


def verify_consistent(bundle, sae, protocol, record, masks, inputs, original_payload, old_features):
    modes, traces = {}, {}
    for mode in MODES:
        modes[mode], traces[mode] = capture_mode(bundle.model, inputs, protocol["hook"], protocol["scoring"], mode,
                                               profile=mode in {"inference", "gradient"})
    baseline = modes["inference"]
    comparisons = {mode: mode_comparison(baseline, modes[mode], protocol["scoring"], mode) for mode in MODES[1:]}
    _, features = encode_decode(sae, baseline["hidden"][0])
    compatible_features = feature_compatibility(old_features, features, masks)
    compatible_logits = identity_diagnostics(original_payload["inference"]["logits"], baseline["logits"],
                                             protocol["scoring"], "original_sdpa_vs_consistent")
    unchanged = replace_residual(bundle.model, inputs, protocol["hook"], baseline["hidden"])
    controls = {"unchanged": identity_diagnostics(baseline["logits"], unchanged, protocol["scoring"], "unchanged")}
    edits = []
    modes_passed = all(c["logits"]["allclose"] and c["logits"].get("decision_preserved", False)
                       and c["layer14"]["allclose"] for c in comparisons.values())
    if modes_passed and controls["unchanged"]["allclose"]:
        gradient = modes["gradient"]["gradient"][0]
        for group, mask in masks.items():
            norms = features[mask].square().sum(0) * sae.W_dec.detach().square().sum(1)
            if not norms.any():
                edits.append({"group": group, "inactive_group": True})
                continue
            feature = int(norms.argmax())
            for alpha in (0., .5, 1.):
                edited, norms_record = feature_removal(baseline["hidden"], features, sae.W_dec,
                                                      feature, mask, alpha, bundle.model.dtype)
                if not torch.equal(edited[:, ~mask], baseline["hidden"][:, ~mask]):
                    raise AssertionError("Unselected positions changed")
                logits = replace_residual(bundle.model, inputs, protocol["hook"], edited)
                measurement = identity_diagnostics(baseline["logits"], logits, protocol["scoring"], f"{group}:{alpha}")
                predicted = -alpha * (features[mask, feature, None] * sae.W_dec[feature] * gradient[mask]).sum().item()
                edits.append({"group": group, "feature": feature, "alpha": alpha, **norms_record,
                              "predicted_delta": predicted, "measurement": measurement})
                if alpha == 0:
                    controls[f"zero_{group}"] = measurement
    passed = modes_passed and all(c["allclose"] and c.get("decision_preserved", False) for c in controls.values())
    return {"attention": attention_settings(bundle.model), "comparisons": comparisons, "operators": traces,
            "controls": controls, "verification_edits": edits, "passed": passed,
            "discovery_output_compatibility": compatible_logits,
            "discovery_feature_cache_compatibility": compatible_features}, {
                "modes": modes, "features": features.to_sparse().coalesce()}


def run(source, failed, output=OUTPUT, *, report_only=False):
    source, failed, output = (Path(p).resolve() for p in (source, failed, output))
    protocol, cases = protocol_for(source, failed, output)
    if report_only:
        return report(output, protocol)
    if (output / "preflight_report.json").exists() and read_json(output / "preflight_report.json")["complete"]:
        return report(output, protocol)
    from medgemma_grounding.inference import load_model
    from medgemma_grounding.sae_reconstruction import load_scope_sae

    if not torch.backends.mps.is_available():
        raise RuntimeError("Fixed preflight requires MPS; no fallback")
    bundle = load_model(protocol["model_id"], device="mps", revision=protocol["model_revision"], attn_implementation="sdpa")
    sae, _, _, revision = load_scope_sae(14, all_layers=True)
    if bundle.model.dtype != torch.bfloat16 or revision != protocol["sae_revision"]:
        raise ValueError("Fixed model precision/SAE revision changed")
    for model in (bundle.model, sae):
        model.eval()
        model.requires_grad_(False)
    before_weights = weight_identity(bundle.model)
    def check_weights():
        checks = {"protocol_hash": protocol["protocol_hash"],
                  "parameter_identity_versions_unchanged": before_weights == weight_identity(bundle.model),
                  "parameter_gradients_unset": all(p.grad is None for p in bundle.model.parameters()),
                  "parameters_frozen": all(not p.requires_grad for p in bundle.model.parameters()),
                  "model_dtype": str(bundle.model.dtype), "attention": attention_settings(bundle.model)}
        atomic_json(output / "runtime_checks.json", checks)
        if not all(checks[key] for key in ("parameter_identity_versions_unchanged", "parameter_gradients_unset", "parameters_frozen")):
            raise AssertionError("Weights/precision or parameter gradient state changed")
    try:
        configure_attention(bundle.model, "sdpa")
        for case in cases:
            case_id = case["case_id"]
            if (output / f"{case_id}.json").exists():
                load_case(output, protocol, case_id)
                continue
            logging.info("Original SDPA modes: %s", case_id)
            _, source_payload = load_discovery_case(source, case_id)
            inputs, masks = prepared_inputs(bundle, protocol, case)
            modes, traces = {}, {}
            for mode in MODES:
                modes[mode], traces[mode] = capture_mode(bundle.model, inputs, protocol["hook"], protocol["scoring"], mode,
                                                       profile=mode in {"inference", "gradient"})
            comparisons = {mode: mode_comparison(modes["inference"], modes[mode], protocol["scoring"], mode)
                           for mode in MODES[1:]}
            _, features = encode_decode(sae, modes["inference"]["hidden"][0])
            cached = source_payload["features"].to_dense()
            compatibility = feature_compatibility(cached, features, masks)
            baseline_score = answer_score(modes["inference"]["logits"], protocol["scoring"])
            source_reproduced = all(baseline_score[k] == case["answer_score"][k] for k in ("yes_logit", "no_logit", "decision"))
            record = {"case_id": case_id, "outcome": outcome(case), "original_attention": attention_settings(bundle.model),
                      "comparisons": comparisons, "operators": traces, "backend_evidence": backend_evidence(comparisons, traces),
                      "source_outputs_reproduced": source_reproduced, "source_features": compatibility}
            save_case(output, protocol, record, {"original": modes})
            logging.info("SDPA gradient max difference %.6g; operators %s", comparisons["gradient"]["logits"]["max_absolute_difference"], traces)
        originals = [load_case(output, protocol, case["case_id"])[0] for case in cases]
        if not all(r["backend_evidence"]["confirmed"] and r["source_outputs_reproduced"] and r["source_features"]["all"]["allclose"] for r in originals):
            check_weights()
            return report(output, protocol)
        configure_attention(bundle.model, FIX)
        for case in cases:
            case_id = case["case_id"]
            record, payload = load_case(output, protocol, case_id)
            if "consistent" in record:
                continue
            logging.info("Consistent text eager / vision SDPA verification: %s", case_id)
            _, source_payload = load_discovery_case(source, case_id)
            inputs, masks = prepared_inputs(bundle, protocol, case)
            record["consistent"], payload["consistent"] = verify_consistent(
                bundle, sae, protocol, case, masks, inputs, payload["original"], source_payload["features"].to_dense())
            save_case(output, protocol, record, payload)
            logging.info("Consistency passed=%s; discovery logit max difference %.6g", record["consistent"]["passed"],
                         record["consistent"]["discovery_output_compatibility"]["max_absolute_difference"])
        check_weights()
        return report(output, protocol)
    except Exception as error:
        atomic_json(output / "failure.json", {"protocol_hash": protocol["protocol_hash"], "error": str(error),
                    "diagnostics": getattr(error, "diagnostics", None), "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("results/gemma_scope/discovery_v1"))
    parser.add_argument("--failed-dir", type=Path, default=Path("results/gemma_scope/interventions_v1"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.source_dir, args.failed_dir, args.output_dir, report_only=args.report_only)
    logging.info("Two-case verification: complete=%s, hypothesis=%s, repair=%s; stop here",
                 result["complete"], result["backend_hypothesis_confirmed"], result["repair_mechanics_passed"])


if __name__ == "__main__":
    main()
