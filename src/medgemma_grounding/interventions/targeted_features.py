"""E-008 residual-preserving feature edits and executed-size comparisons."""


from __future__ import annotations

from pathlib import Path
import time

import medgemma_grounding.analysis.targeted_features as targeted_analysis
import medgemma_grounding.interventions.feature_runtime as feature_runtime
import medgemma_grounding.interventions.protocol as feature_protocol
import medgemma_grounding.utils.io as io
import numpy as np
import torch

from medgemma_grounding.interventions.controls import match_direction, perturbation_norms
from medgemma_grounding.interventions.features import capture_gradient, feature_removal
from medgemma_grounding.model import answer_score
from medgemma_grounding.sae.reconstruction import logit_metrics, replace_residual
from medgemma_grounding.utils.io import atomic_json, file_sha256, identity_hash


CONTROL_BANK_SIZE = 3


SEED = 42


def measurement_path(output, case, group, feature, alpha, bank=None):
    role = "target" if bank is None else f"random{bank}"
    return output / "measurements" / case / f"{group}_{feature}_{alpha:g}_{role}.json"


def save_measurement(path, row):
    row["measurement_hash"] = identity_hash(row)
    atomic_json(path, row)


def load_measurement(path, protocol, case, group, feature, alpha, bank=None):
    row = io.read_json(path)
    expected = {"protocol_hash": protocol["protocol_hash"], "case_id": case, "group": group,
                "feature": feature, "alpha": alpha, "bank": bank}
    if (any(row[k] != v for k, v in expected.items())
            or row["measurement_hash"] != identity_hash({k: v for k, v in row.items() if k != "measurement_hash"})
            or case not in protocol["discovery_case_ids"] or group not in targeted_analysis.GROUPS):
        raise ValueError("Measurement identity/checksum or cohort mismatch")
    return row


def source_case(protocol, parent, case):
    if case not in protocol["discovery_case_ids"]:
        raise ValueError("Only frozen discovery cases are accessible")
    source = Path(protocol["source_directory"])
    if file_sha256(source / "cases" / f"{case}.json") != protocol["case_record_sha256"][case]:
        raise ValueError("Source metadata changed")
    record, payload = feature_runtime.load_case(source, parent, case)
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
        old_path = feature_runtime.treatment_path(Path(protocol["source_directory"]), case, feature, group, alpha)
        started = time.perf_counter()
        if old_path.exists():
            old = feature_runtime.load_treatment(old_path, parent, case, feature, group, alpha)
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
    return [(c["group"], c["feature"], a, k) for c in candidates for a in feature_protocol.FRACTIONS
            for k in [None, *range(bank_size)]]


def case_complete(output, protocol, case, candidates):
    complete = (output / "cases" / f"{case}.json").exists()
    if complete:
        r = io.read_json(output / "cases" / f"{case}.json")
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
    inputs, features, record, payload = feature_runtime.treatment_context(bundle, sae, parent, original, saved)
    baseline = payload["baseline_logits"]
    controls = {"saved_baseline": feature_runtime.identity_control(saved["baseline_logits"], baseline, parent["scoring"]),
                "unchanged": feature_runtime.identity_control(baseline,
                    replace_residual(bundle.model, inputs, protocol["hook"], payload["hidden"]), parent["scoring"])}
    if smoke:
        h, logits, grad = capture_gradient(bundle.model, inputs, protocol["hook"], parent["scoring"])
        controls["gradient_forward"] = feature_runtime.identity_control(baseline, logits, parent["scoring"])
        torch.testing.assert_close(h, payload["hidden"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(grad, payload["gradient"], rtol=1e-5, atol=1e-5)
        mask = torch.zeros(features.shape[0], dtype=torch.bool)
        mask[record["positions"]["image"]] = True
        zero, _ = feature_removal(payload["hidden"], features, sae.W_dec, 0, mask, 0, bundle.model.dtype)
        controls["zero_strength"] = feature_runtime.identity_control(baseline,
            replace_residual(bundle.model, inputs, protocol["hook"], zero), parent["scoring"])
    path = output / "cases" / f"{case}.json"
    header = {"protocol_hash": protocol["protocol_hash"], "case_id": case, "source_tensor_sha256": original["tensor_sha256"],
              "patient": record["patient"], "outcome": record["outcome"], "controls": controls}
    if path.exists():
        previous = io.read_json(path)
        if not all(c["allclose"] for c in previous["controls"].values()):
            raise ValueError("Previous numerical control failed")
        header["controls"] = {**previous["controls"], **controls}
    header["record_hash"] = identity_hash(header)
    atomic_json(path, header)
    return record, payload, inputs, features


def smoke_candidates(payload, candidates):
    result = []
    for group in targeted_analysis.GROUPS:
        available = [c for c in candidates if c["group"] == group]
        if available:
            result.append(min(available, key=lambda c: (-payload["groups"][group]["removal_l2"][c["feature"]].item(), c["feature"])))
    return result


def check_smoke(output, protocol, parent, candidates):
    matched_groups, hashes = set(), {}
    for case in protocol["smoke_ids"]:
        _, payload = source_case(protocol, parent, case)
        header_path = output / "cases" / f"{case}.json"
        header = io.read_json(header_path)
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
