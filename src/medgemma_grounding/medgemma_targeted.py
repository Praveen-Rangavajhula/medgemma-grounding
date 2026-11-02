"""E-008: bounded activation-nominated discovery with executed-size controls.

Explicit stages: analyze / prepare / smoke / run / report. --stage run continues
automatically from numerical smoke to the full frozen discovery cohort.
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from medgemma_grounding import medgemma_interventions as base, medgemma_discovery as discovery
from medgemma_grounding import targeted_analysis as analysis
from medgemma_grounding.feature_controls import MATCHING, direction_bank, match_direction, perturbation_norms
from medgemma_grounding.feature_interventions import feature_removal, capture_gradient
from medgemma_grounding.medgemma_features import atomic_json, file_sha256, identity_hash, answer_score
from medgemma_grounding.medgemma_preflight import tree_hashes, weight_identity
from medgemma_grounding.sae_reconstruction import replace_residual, logit_metrics

SOURCE = discovery.OUTPUT
OUTPUT = Path("results/gemma_scope/targeted_discovery_v1")
CONTROL_BANK_SIZE = 3
SEED = 42
SOURCE_FILES = ("medgemma_targeted.py", "targeted_analysis.py", "feature_controls.py",
                "feature_interventions.py", "medgemma_interventions.py", "medgemma_discovery.py",
                "medgemma_features.py", "sae_reconstruction.py", "inference.py", "chexlocalize.py")


def validate_source(source, *, full=True):
    """Read-only: do not call any E-007 report/selection writer."""
    source = Path(source).resolve()
    parent = base.read_json(source / "protocol.json")
    if (parent.get("experiment") != "E-007" or not parent.get("fresh_encodings")
            or parent["protocol_hash"] != identity_hash({k: v for k, v in parent.items() if k != "protocol_hash"})
            or parent["runtime"]["attention_implementation"] != {"text_config": "eager", "vision_config": "sdpa"}
            or parent["runtime"]["model_dtype"] != "torch.bfloat16"):
        raise ValueError("The pinned repaired E-007 runtime is required")
    report = base.read_json(source / "intervention_report.json")
    calibration = base.read_json(source / "calibration_report.json")
    candidates = base.read_json(source / "candidates.json")
    if (not report["complete"] or report["protocol_hash"] != parent["protocol_hash"]
            or not calibration["complete"] or set(calibration["qualifying_groups"]) != set(analysis.GROUPS)
            or calibration["report_hash"] != identity_hash({k: v for k, v in calibration.items() if k != "report_hash"})
            or candidates["candidates_hash"] != identity_hash({k: v for k, v in candidates.items() if k != "candidates_hash"})
            or candidates["calibration_report_hash"] != calibration["report_hash"]):
        raise ValueError("Complete, unchanged E-007 calibration and frozen results required")
    if full:
        for name, expected in parent["source_sha256"].items():
            if file_sha256(Path(__file__).with_name(name)) != expected:
                raise ValueError(f"E-007 computation code changed; exact reuse not justified: {name}")
        if any(version(name) != value for name, value in parent["versions"].items()):
            raise ValueError("E-007 runtime package versions changed")
        _, originals, hashes = base.validate_source(Path(parent["source_directory"]))
        if (hashes != parent["source_artifact_sha256"]
                or set(parent["discovery_case_ids"]) != {r["case_id"] for r in originals}):
            raise ValueError("E-004 provenance changed")
        cohort = base.read_json(Path(parent["source_directory"]) / "cohort.json")
        confirmation = {c["patient"] for c in cohort["cases"] if c["partition"] == "confirmation"}
        if len(confirmation) != 60 or confirmation & {r["patient"] for r in originals}:
            raise ValueError("Expected 60 inaccessible confirmation patients")
        for path, expected in parent["previous_artifacts"].items():
            if tree_hashes(Path(path)) != expected:
                raise ValueError("A historical experiment changed")
    records, arrays = [], {g: {"activity": [], "attribution": []} for g in analysis.GROUPS}
    for case in parent["discovery_case_ids"]:
        record, payload = base.load_case(source, parent, case)
        if (not record.get("fresh_encodings") or not all(c["allclose"] for c in record["controls"].values())
                or record["outcome"] != base.outcome(record)):
            raise ValueError("Invalid repaired case/control/outcome")
        positions = record["positions"]
        all_positions = set(record["source_positions"]["all"])
        image, final = set(positions["image"]), set(positions["final"])
        if (image != set(record["source_positions"]["image"]) or final != {max(all_positions)}
                or image & final or set(positions["other_non_image"]) != all_positions - image - final):
            raise ValueError("Image/final-input definitions changed; final is not an answer token")
        records.append(record)
        for g in arrays:
            arrays[g]["activity"].append(payload["groups"][g]["activation_mean"].numpy())
            arrays[g]["attribution"].append(payload["groups"][g]["effect"].numpy())
    if len(records) != 142 or len({r["patient"] for r in records}) != 140:
        raise ValueError("Expected exactly 142 discovery images / 140 patients")
    return parent, records, {g: {k: np.stack(v) for k, v in a.items()} for g, a in arrays.items()}


def check_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    parent = base.read_json(source / "protocol.json")
    protected = [source, Path(parent["source_directory"]), *map(Path, parent["previous_artifacts"])]
    for path in protected:
        if output == path or output in path.parents or path in output.parents:
            raise ValueError("New output must be separate from every historical source")
    return source, output


def analyze(source, output):
    source, output = check_output(source, output)
    if (output / "protocol.json").exists():
        raise ValueError("Analysis is frozen here; use report or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    parent, records, arrays = validate_source(source)
    manifest = {"source_directory": str(source), "source_protocol_hash": parent["protocol_hash"],
                "source_tree_sha256": tree_hashes(source)}
    path = output / "source_manifest.json"
    if path.exists() and base.read_json(path) != manifest:
        raise ValueError("Source changed after the initial audit")
    atomic_json(path, manifest)
    audit = analysis.audit_e007(source, parent, records, output)
    analysis.audit_figure(audit, output / "e007_agreement.png")
    return parent, records, arrays, audit


def expected_budget(source, parent, records, candidates):
    controls = new_targets = reused = inactive = 0
    timings = []
    for record in records:
        case = record["case_id"]
        _, payload = base.load_case(source, parent, case)
        for c in candidates:
            active = bool(payload["groups"][c["group"]]["active"][c["feature"]])
            for alpha in base.FRACTIONS:
                path = base.treatment_path(source, case, c["feature"], c["group"], alpha)
                if path.exists():
                    old = base.load_treatment(path, parent, case, c["feature"], c["group"], alpha)
                    reused += 1
                    if not old["inactive"] and "runtime_seconds" in old:
                        timings.append(old["runtime_seconds"])
                elif active:
                    new_targets += 1
                else:
                    inactive += 1
                controls += CONTROL_BANK_SIZE * int(active)
    # Every case has baseline/capture/unchanged. Two smoke cases add grad+zero,
    # and are revalidated when the cohort stage resumes their partial treatments.
    numerical = 3 * len(records) + 4 + 6
    upper = controls + new_targets + numerical
    if not timings:
        timings = [2., 3.]
    return {"candidate_budget": NOMINATION_LIMIT, "frozen_candidates": len(candidates),
            "bank_size": CONTROL_BANK_SIZE, "expected_treatment_records": len(candidates) * 2 * len(records) * (1 + CONTROL_BANK_SIZE),
            "reusable_target_records": reused, "new_active_targets_upper": new_targets,
            "inactive_targets_without_cache": inactive, "control_forwards_upper": controls,
            "numerical_forwards_upper": numerical, "new_forwards_upper": upper,
            "historical_active_forward_seconds_median": float(np.median(timings)),
            "estimated_hours_median_plus_10pct_overhead": upper * float(np.median(timings)) * 1.1 / 3600,
            "estimated_hours_q25_q90_plus_10pct": (upper * np.quantile(timings, [.25, .9]) * 1.1 / 3600).tolist(),
            "limits": "Upper bound before zero/rounded-away/unmatched skips; timing is hardware-dependent, not a promise"}


NOMINATION_LIMIT = analysis.NOMINATION["maximum_candidates"]


def prepare(source, output):
    source, output = check_output(source, output)
    path = output / "protocol.json"
    if path.exists():
        protocol = base.read_json(path)
        validate_protocol(output, protocol)
        return protocol
    parent, records, arrays, _ = analyze(source, output)
    if not (output / "methods.md").exists() or not (output / "references").is_dir():
        raise ValueError("Record the reviewed primary methods and pinned source files before freezing")
    candidates, rows, patients = analysis.nominate(records, arrays)
    base.write_csv(output / "nomination_rankings.csv", rows)
    atomic_json(output / "candidates.json", {"candidates": candidates, "settings": analysis.NOMINATION,
        "deduplication": "merge tracks for identical group/feature; do not backfill",
        "exclusions": "all features and track-specific exclusions retained in nomination_rankings.csv"})
    # Compact, CPU-only case summaries, not additional copies of hidden states.
    data = {"case_ids": [r["case_id"] for r in records], "groups": {
        g: {k: torch.from_numpy(v) for k, v in a.items()} for g, a in arrays.items()}}
    torch.save(data, output / "nomination_data.pt")
    _, first_payload = base.load_case(source, parent, parent["discovery_case_ids"][0])
    hidden_width = first_payload["hidden"].shape[-1]
    bank = direction_bank(hidden_width, CONTROL_BANK_SIZE, SEED)
    torch.save(bank, output / "control_bank.pt")
    budget = expected_budget(source, parent, records, candidates)
    atomic_json(output / "execution_budget.json", budget)
    artifacts = ["source_manifest.json", "methods.md", "candidates.json", "nomination_rankings.csv",
                 "nomination_data.pt", "control_bank.pt", "execution_budget.json", "e007_audit.json", "e007_agreement.csv", "e007_agreement.png"]
    artifacts += [str(p.relative_to(output)) for p in sorted((output / "references").rglob("*")) if p.is_file()]
    protocol = {"schema_version": 1, "experiment": "E-008", "source_directory": str(source),
        "source_protocol_hash": parent["protocol_hash"], "discovery_case_ids": parent["discovery_case_ids"],
        "case_tensor_sha256": {r["case_id"]: r["tensor_sha256"] for r in records},
        "case_record_sha256": {r["case_id"]: file_sha256(source / "cases" / f"{r['case_id']}.json") for r in records},
        "smoke_ids": parent["smoke_ids"], "patient_count": len(patients), "nomination": analysis.NOMINATION,
        "runtime": parent["runtime"], "model_id": parent["model_id"], "model_revision": parent["model_revision"],
        "processor_revision": parent["processor_revision"], "sae_revision": parent["sae_revision"],
        "sae_release": parent["sae_release"], "sae_id": parent["sae_id"], "hook": parent["hook"],
        "scoring": parent["scoring"], "prompt": parent["prompt"], "sae_width": parent["sae_width"], "hidden_width": hidden_width,
        "positions": {g: parent["position_definitions"][g] for g in analysis.GROUPS},
        "strengths": list(base.FRACTIONS), "control_bank_size": CONTROL_BANK_SIZE, "seed": SEED,
        "matching": MATCHING, "control_orientation": "three fixed Gaussian unit decoder-width directions, shared across cases/features/strengths; amplitudes matched per edited position",
        "zero_policy": "inactive exact zeros recorded; rounded-away targets recorded and excluded from positive-size comparisons",
        "unmatched_policy": "save diagnostics, no unmatched-control forward; target still measured; primary comparison requires all three matches",
        "reuse": "only identical E-007 model/processor/SAE/runtime/input/hidden/features/group/strength; verify edit norms and baseline",
        "bootstrap": {"unit": "patient", "resamples": 1000, "seed": SEED, "conditioned_on": "discovery selection and fixed direction bank"},
        "confirmation_enabled": False, "test_enabled": False, "integrated_gradients": False,
        "artifacts_sha256": {name: file_sha256(output / name) for name in artifacts},
        "source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES},
        "versions": {name: version(name) for name in ("torch", "transformers", "sae-lens", "scipy", "numpy")},
        "code_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "code_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout)}
    protocol["protocol_hash"] = identity_hash(protocol)
    atomic_json(path, protocol)
    logging.info("Frozen %d candidates; budget %s", len(candidates), budget)
    return protocol


def validate_protocol(output, protocol, *, full=True):
    if (protocol["protocol_hash"] != identity_hash({k: v for k, v in protocol.items() if k != "protocol_hash"})
            or protocol["confirmation_enabled"] or protocol["test_enabled"]
            or protocol["matching"] != MATCHING or protocol["nomination"] != analysis.NOMINATION
            or protocol["control_bank_size"] != CONTROL_BANK_SIZE):
        raise ValueError("Changed/unsupported targeted protocol; create a new output directory")
    for name, expected in protocol["artifacts_sha256"].items():
        if file_sha256(output / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    for name, expected in protocol["source_sha256"].items():
        if file_sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Source changed after freeze: {name}")
    if any(version(name) != expected for name, expected in protocol["versions"].items()):
        raise ValueError("Runtime package versions changed after freeze")
    manifest = base.read_json(output / "source_manifest.json")
    source = Path(protocol["source_directory"])
    if full and tree_hashes(source) != manifest["source_tree_sha256"]:
        raise ValueError("Historical E-007 artifacts changed")
    parent = base.read_json(source / "protocol.json")
    if (parent["protocol_hash"] != protocol["source_protocol_hash"]
            or parent["protocol_hash"] != identity_hash({k: v for k, v in parent.items() if k != "protocol_hash"})
            or file_sha256(source / "protocol.json") != manifest["source_tree_sha256"]["protocol.json"]):
        raise ValueError("Parent protocol changed")
    return parent


def measurement_path(output, case, group, feature, alpha, bank=None):
    role = "target" if bank is None else f"random{bank}"
    return output / "measurements" / case / f"{group}_{feature}_{alpha:g}_{role}.json"


def save_measurement(path, row):
    row["measurement_hash"] = identity_hash(row)
    atomic_json(path, row)


def load_measurement(path, protocol, case, group, feature, alpha, bank=None):
    row = base.read_json(path)
    expected = {"protocol_hash": protocol["protocol_hash"], "case_id": case, "group": group,
                "feature": feature, "alpha": alpha, "bank": bank}
    if (any(row[k] != v for k, v in expected.items())
            or row["measurement_hash"] != identity_hash({k: v for k, v in row.items() if k != "measurement_hash"})
            or case not in protocol["discovery_case_ids"] or group not in analysis.GROUPS):
        raise ValueError("Measurement identity/checksum or cohort mismatch")
    return row


def source_case(protocol, parent, case):
    if case not in protocol["discovery_case_ids"]:
        raise ValueError("Only frozen discovery cases are accessible")
    source = Path(protocol["source_directory"])
    if file_sha256(source / "cases" / f"{case}.json") != protocol["case_record_sha256"][case]:
        raise ValueError("Source metadata changed")
    record, payload = base.load_case(source, parent, case)
    if record["tensor_sha256"] != protocol["case_tensor_sha256"][case]:
        raise ValueError("Source tensor changed")
    return record, payload


def edited_score(bundle, protocol, record, payload, inputs, edited, *, skip=False):
    baseline = payload["baseline_logits"]
    changed = baseline if skip else replace_residual(bundle.model, inputs, protocol["hook"], edited)
    score = answer_score(changed, protocol["scoring"])
    original = record["answer_score"]
    correct = score["decision"] == ("Yes" if record["label"] else "No")
    was_correct = record["outcome"].startswith("T")
    return {"actual_delta": score["score"] - original["score"], "changed_score": score["score"],
            "changed_decision": score["decision"], "changed_correct": correct,
            "corrected": correct and not was_correct, "introduced": was_correct and not correct,
            "decision_flip": score["decision"] != original["decision"], **logit_metrics(baseline, changed)}


def measure_candidate(bundle, sae, protocol, parent, output, record, payload, inputs, features, candidate, alpha, bank):
    case, group, feature = record["case_id"], candidate["group"], candidate["feature"]
    paths = [measurement_path(output, case, group, feature, alpha, k)
             for k in [None, *range(protocol["control_bank_size"])]]
    for k, path in zip([None, *range(protocol["control_bank_size"])], paths):
        if path.exists():
            load_measurement(path, protocol, case, group, feature, alpha, k)
    if all(p.exists() for p in paths):
        return 0
    mask = torch.zeros(features.shape[0], dtype=torch.bool)
    mask[record["positions"][group]] = True
    edited, norms = feature_removal(payload["hidden"], features, sae.W_dec, feature, mask, alpha, bundle.model.dtype)
    inactive = not bool(features[mask, feature].count_nonzero())
    state = "inactive" if inactive else "rounded_away" if norms["realized_l2"] == 0 else "positive_size"
    common = {"protocol_hash": protocol["protocol_hash"], "source_protocol_hash": parent["protocol_hash"],
              "case_id": case, "patient": record["patient"], "label": record["label"], "outcome": record["outcome"],
              "group": group, "feature": feature, "alpha": alpha, "original_score": record["answer_score"]["score"],
              "original_decision": record["answer_score"]["decision"], "original_correct": record["outcome"].startswith("T"),
              "source_tensor_sha256": record["tensor_sha256"], "target_state": state,
              "edited_positions": record["positions"][group], "target_position_l2": perturbation_norms(payload["hidden"], edited, mask)["position_l2"]}
    forwards = 0
    if not paths[0].exists():
        old_path = base.treatment_path(Path(protocol["source_directory"]), case, feature, group, alpha)
        started = time.perf_counter()
        if old_path.exists():
            old = base.load_treatment(old_path, parent, case, feature, group, alpha)
            if old["original_score"] != common["original_score"] or old["outcome"] != common["outcome"]:
                raise ValueError("Reusable treatment has a different baseline/outcome")
            for key in ("requested_l2", "realized_l2"):
                if not np.isclose(old[key], norms[key], rtol=1e-5, atol=1e-5):
                    raise ValueError("Reusable treatment has a different executed edit")
            measured = {k: old[k] for k in ("actual_delta", "changed_score", "changed_decision", "changed_correct",
                        "corrected", "introduced", "decision_flip", "kl_from_baseline", "top_token_preserved", "top_token_id")}
            provenance = {"execution": "reused_exact_e007", "reused_path": str(old_path.resolve()),
                          "reused_sha256": file_sha256(old_path), "reused_treatment_hash": old["treatment_hash"]}
        else:
            measured = edited_score(bundle, protocol, record, payload, inputs, edited, skip=state != "positive_size")
            provenance = {"execution": "new_forward" if state == "positive_size" else state}
            forwards += int(state == "positive_size")
        predicted = alpha * payload["groups"][group]["effect"][feature].item()
        save_measurement(paths[0], {**common, "role": "feature", "bank": None, "status": state,
            "predicted_delta": predicted, **norms, **measured, **provenance,
            "runtime_seconds": time.perf_counter() - started})
    for k, direction in enumerate(bank):
        path = paths[k + 1]
        if path.exists():
            continue
        started = time.perf_counter()
        random_edit, matching = match_direction(payload["hidden"], edited, mask, direction,
                                                bundle.model.dtype, protocol["matching"])
        if matching["matched"]:
            measured = edited_score(bundle, protocol, record, payload, inputs, random_edit, skip=state != "positive_size")
            forwards += int(state == "positive_size")
        else:
            measured = {key: None for key in ("actual_delta", "changed_score", "changed_decision", "changed_correct",
                        "corrected", "introduced", "decision_flip", "kl_from_baseline", "top_token_preserved", "top_token_id")}
        save_measurement(path, {**common, "role": "random_direction", "bank": k,
            "predicted_delta": float((payload["gradient"] * (random_edit - payload["hidden"])).sum()),
            "prediction_basis": "original gradient dot realized random perturbation", **matching, **measured,
            "execution": "new_forward" if matching["matched"] and state == "positive_size" else matching["status"],
            "runtime_seconds": time.perf_counter() - started})
    return forwards


def specs(candidates, bank_size):
    return [(c["group"], c["feature"], a, k) for c in candidates for a in base.FRACTIONS
            for k in [None, *range(bank_size)]]


def case_complete(output, protocol, case, candidates):
    complete = (output / "cases" / f"{case}.json").exists()
    if complete:
        r = base.read_json(output / "cases" / f"{case}.json")
        if (r["protocol_hash"] != protocol["protocol_hash"] or r["source_tensor_sha256"] != protocol["case_tensor_sha256"][case]
                or r["record_hash"] != identity_hash({k: v for k, v in r.items() if k != "record_hash"})):
            raise ValueError("Case context identity changed")
    for g, j, a, k in specs(candidates, protocol["control_bank_size"]):
        path = measurement_path(output, case, g, j, a, k)
        if path.exists():
            load_measurement(path, protocol, case, g, j, a, k)
        else:
            complete = False
    return complete


def checked_context(bundle, sae, protocol, parent, output, case, *, smoke=False):
    original, saved = source_case(protocol, parent, case)
    inputs, features, record, payload = discovery.treatment_context(bundle, sae, parent, original, saved)
    baseline = payload["baseline_logits"]
    controls = {"saved_baseline": base.identity_control(saved["baseline_logits"], baseline, parent["scoring"]),
                "unchanged": base.identity_control(baseline,
                    replace_residual(bundle.model, inputs, protocol["hook"], payload["hidden"]), parent["scoring"])}
    if smoke:
        h, logits, grad = capture_gradient(bundle.model, inputs, protocol["hook"], parent["scoring"])
        controls["gradient_forward"] = base.identity_control(baseline, logits, parent["scoring"])
        torch.testing.assert_close(h, payload["hidden"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(grad, payload["gradient"], rtol=1e-5, atol=1e-5)
        mask = torch.zeros(features.shape[0], dtype=torch.bool)
        mask[record["positions"]["image"]] = True
        zero, _ = feature_removal(payload["hidden"], features, sae.W_dec, 0, mask, 0, bundle.model.dtype)
        controls["zero_strength"] = base.identity_control(baseline,
            replace_residual(bundle.model, inputs, protocol["hook"], zero), parent["scoring"])
    path = output / "cases" / f"{case}.json"
    header = {"protocol_hash": protocol["protocol_hash"], "case_id": case, "source_tensor_sha256": original["tensor_sha256"],
              "patient": record["patient"], "outcome": record["outcome"], "controls": controls}
    if path.exists():
        previous = base.read_json(path)
        if not all(c["allclose"] for c in previous["controls"].values()):
            raise ValueError("Previous numerical control failed")
        header["controls"] = {**previous["controls"], **controls}
    header["record_hash"] = identity_hash(header)
    atomic_json(path, header)
    return record, payload, inputs, features


def smoke_candidates(payload, candidates):
    result = []
    for group in analysis.GROUPS:
        available = [c for c in candidates if c["group"] == group]
        if available:
            result.append(min(available, key=lambda c: (-payload["groups"][group]["removal_l2"][c["feature"]].item(), c["feature"])))
    return result


def check_smoke(output, protocol, parent, candidates):
    matched_groups, hashes = set(), {}
    for case in protocol["smoke_ids"]:
        _, payload = source_case(protocol, parent, case)
        header_path = output / "cases" / f"{case}.json"
        header = base.read_json(header_path)
        if (header["protocol_hash"] != protocol["protocol_hash"]
                or header["record_hash"] != identity_hash({k: v for k, v in header.items() if k != "record_hash"})
                or not {"saved_baseline", "unchanged", "gradient_forward", "zero_strength"} <= header["controls"].keys()
                or not all(c["allclose"] and c["decision_preserved"] for c in header["controls"].values())):
            raise ValueError("Smoke numerical prerequisite failed")
        hashes[str(header_path.relative_to(output))] = file_sha256(header_path)
        for c in smoke_candidates(payload, candidates):
            all_matched = True
            for g, j, a, k in specs([c], protocol["control_bank_size"]):
                path = measurement_path(output, case, g, j, a, k)
                row = load_measurement(path, protocol, case, g, j, a, k)
                hashes[str(path.relative_to(output))] = file_sha256(path)
                if k is not None:
                    all_matched &= row["matched"] and row["target_state"] == "positive_size"
            if all_matched:
                matched_groups.add(c["group"])
    if matched_groups != {c["group"] for c in candidates}:
        raise ValueError("Smoke did not verify all three positive-size controls in every selected position group")
    result = {"protocol_hash": protocol["protocol_hash"], "complete": True, "matched_groups": sorted(matched_groups),
              "artifact_sha256": hashes, "purpose": "numerical and executed-size mechanics, not research cohort selection"}
    result["smoke_hash"] = identity_hash(result)
    atomic_json(output / "smoke_report.json", result)
    return result


def run(stage, source=SOURCE, output=OUTPUT, limit=None):
    if stage not in {"analyze", "prepare", "smoke", "cohort", "run", "report"}:
        raise ValueError("An explicit stage is required")
    if limit is not None and (stage != "cohort" or limit < 1):
        raise ValueError("--limit counts new cohort cases only")
    source, output = check_output(source, output)
    if stage == "analyze":
        return analyze(source, output)[3]
    protocol = prepare(source, output)
    parent = validate_protocol(output, protocol)
    if stage == "prepare":
        return protocol
    if stage == "report":
        return report(output, protocol, parent)
    if list(output.glob("failure_*.json")):
        raise ValueError("Technical failure recorded; diagnose before any resumed model work")
    candidates = base.read_json(output / "candidates.json")["candidates"]
    bank = torch.load(output / "control_bank.pt", map_location="cpu", weights_only=True)
    bundle = sae = weights = None
    active_stage = stage

    def runtime():
        nonlocal bundle, sae, weights
        if bundle is None:
            config = {**parent, "protocol_hash": protocol["protocol_hash"]}
            bundle, sae = base.frozen_runtime(config, output)
            if sae.W_dec.shape != (protocol["sae_width"], protocol["hidden_width"]):
                raise ValueError("Decoder dimensions changed")
            weights = (weight_identity(bundle.model), weight_identity(sae))

    try:
        if stage in {"smoke", "run"}:
            active_stage = "smoke"
            if (output / "smoke_report.json").exists():
                check_smoke(output, protocol, parent, candidates)
            else:
                started = time.perf_counter()
                forwards = 0
                for case in protocol["smoke_ids"]:
                    runtime()
                    record, payload, inputs, features = checked_context(bundle, sae, protocol, parent, output, case, smoke=True)
                    for c in smoke_candidates(payload, candidates):
                        for a in base.FRACTIONS:
                            forwards += measure_candidate(bundle, sae, protocol, parent, output, record, payload, inputs, features, c, a, bank)
                check_smoke(output, protocol, parent, candidates)
                atomic_json(output / "smoke_timing.json", {"protocol_hash": protocol["protocol_hash"],
                            "elapsed_seconds": time.perf_counter() - started, "treatment_forwards": forwards,
                            "note": "Includes model loading, controls and BF16 matching; cohort ETA updates empirically"})
                logging.info("Two-case numerical and norm-matching smoke passed; full cohort authorized")
        else:
            if not (output / "smoke_report.json").exists():
                raise ValueError("Numerical smoke prerequisite missing")
            check_smoke(output, protocol, parent, candidates)
        if stage in {"run", "cohort"}:
            active_stage = "cohort"
            pending = [c for c in protocol["discovery_case_ids"] if not case_complete(output, protocol, c, candidates)]
            if limit is not None:
                pending = pending[:limit]
            start, forwards = time.perf_counter(), 0
            for i, case in enumerate(pending, 1):
                logging.info("cohort %d/%d %s", i, len(pending), case)
                runtime()
                record, payload, inputs, features = checked_context(bundle, sae, protocol, parent, output, case)
                for c in candidates:
                    for a in base.FRACTIONS:
                        forwards += measure_candidate(bundle, sae, protocol, parent, output, record, payload, inputs, features, c, a, bank)
                elapsed = time.perf_counter() - start
                progress = {"stage": "cohort", "new_cases": i, "pending_initially": len(pending),
                            "last_case": case, "elapsed_seconds": elapsed, "eta_seconds": elapsed / i * (len(pending) - i),
                            "new_treatment_forwards": forwards}
                atomic_json(output / "progress.json", progress)
                logging.info("Saved case; %.1fm elapsed, empirical ETA %.1fm", elapsed / 60, progress["eta_seconds"] / 60)
        if bundle is not None:
            checks = {"protocol_hash": protocol["protocol_hash"],
                      "parameter_identities_versions_unchanged": weights == (weight_identity(bundle.model), weight_identity(sae)),
                      "parameters_frozen": all(not p.requires_grad for m in (bundle.model, sae) for p in m.parameters()),
                      "parameter_gradients_unset": all(p.grad is None for m in (bundle.model, sae) for p in m.parameters())}
            atomic_json(output / "runtime_checks.json", checks)
            if not all(v for k, v in checks.items() if k != "protocol_hash"):
                raise ValueError("Model/SAE parameter state changed")
        return report(output, protocol, parent)
    except Exception as error:
        atomic_json(output / f"failure_{active_stage}.json", {"protocol_hash": protocol["protocol_hash"],
            "error": str(error), "stage": active_stage, "diagnostics": getattr(error, "diagnostics", None),
            "traceback": traceback.format_exc(), "settings_changed": False})
        try:
            report(output, protocol, parent)
        except Exception:
            logging.exception("Partial report failed; original failure is preserved")
        raise


def report(output, protocol, parent):
    validate_protocol(output, protocol)
    return analysis.report(output, protocol, parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["analyze", "prepare", "smoke", "cohort", "run", "report"])
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.stage, args.source_dir, args.output_dir, args.limit)
    logging.info("E-008 finished: %s", {k: result[k] for k in ("status", "complete", "stages") if k in result})


if __name__ == "__main__":
    main()
