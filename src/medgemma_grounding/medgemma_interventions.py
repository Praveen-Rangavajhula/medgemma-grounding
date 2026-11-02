"""E-005: gated layer-14 attribution, residual-preserving edits and offline reports.

Run --stage pilot first. Selection and reporting are model-free. All source
images are the fixed E-004 discovery partition; source artifacts are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from medgemma_grounding.feature_interventions import (
    GROUPS, capture_gradient, feature_removal, intervention_groups, project_attribution,
    attention_settings, tensor_diagnostics,
)
from medgemma_grounding.medgemma_features import (
    answer_candidates, answer_score, atomic_json, file_sha256, identity_hash,
    load_discovery_case, spatial_layout,
)
from medgemma_grounding.sae_reconstruction import (
    SAE_REVISION, capture_residual, checkpoint_spec, encode_decode,
    logit_metrics, position_groups, prompt_positions, replace_residual,
)

SOURCE = Path("results/gemma_scope/discovery_v1")
DESTINATION = Path("results/gemma_scope/interventions_v1")
FRACTIONS = (0.5, 1.0)
QUOTAS = {"TP": 6, "TN": 6, "FP": 4, "FN": 4}
THRESHOLDS = {"floor_logits": 0.25, "noise_multiplier": 10, "min_pairs": 10,
              "spearman": 0.7, "sign_agreement": 0.8, "relative_error": 0.5}
UPSTREAM = {
    "multimodal-sae": "https://github.com/EvolvingLMMs-Lab/multimodal-sae/blob/60265c36e4b21d7b70e1951086cc6f04a7bb5659/sae_auto_interp/features/patching/attribution.py",
    "feature-circuits-attribution": "https://github.com/saprmarks/feature-circuits/blob/7fbd82b895ae16294f4e6fc7bfc675d1d680d659/attribution.py",
    "feature-circuits-ablation": "https://github.com/saprmarks/feature-circuits/blob/7fbd82b895ae16294f4e6fc7bfc675d1d680d659/ablation.py",
    "line-of-sight": "https://arxiv.org/html/2506.04706v1#S4.SS3",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def outcome(record):
    return ("T" if (record["answer_score"]["decision"] == "Yes") == bool(record["label"]) else "F") + (
        "P" if record["answer_score"]["decision"] == "Yes" else "N")


def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()


def choose_pilot(records):
    """Hash-ordered patients; use the first lexicographic case per patient."""
    patients = {}
    for record in sorted(records, key=lambda r: r["case_id"]):
        patients.setdefault(record["patient"], record)
    pools = {key: sorted((r for r in patients.values() if outcome(r) == key),
                        key=lambda r: (hashed(f"42:{r['patient']}"), r["patient"])) for key in QUOTAS}
    if len(pools["FN"]) != 4 or any(len(pools[k]) < n for k, n in QUOTAS.items()):
        raise ValueError("Source outcomes no longer support the prespecified 6 TP / 6 TN / 4 FP / all 4 FN pilot")
    selected = {k: [r["case_id"] for r in pools[k][:n]] for k, n in QUOTAS.items()}
    order = [selected["TP"][0], selected["TN"][0]]
    order += [c for group in QUOTAS for c in selected[group] if c not in order]
    return {"allocation": selected, "case_ids": order, "preflight": order[:2]}


def validate_source(source):
    """Check all discovery shards, source metadata and isolation before any run."""
    from medgemma_grounding.chexlocalize import validate_partition

    protocol = read_json(source / "protocol.json")
    cohort = read_json(source / "cohort.json")
    if file_sha256(source / "cohort.json") != protocol["cohort_sha256"]:
        raise ValueError("Source cohort checksum changed")
    cases = validate_partition(cohort, set(cohort["pilot_patients"]))
    if cases != protocol["discovery_cases"] or len(cases) != 142:
        raise ValueError("Expected the exact 142 E-004 discovery images")
    if (protocol["layer"] != 14 or protocol["sae_id"] != checkpoint_spec(14, all_layers=True)[1]
            or protocol["sae_revision"] != SAE_REVISION
            or protocol["hook"] != checkpoint_spec(14, all_layers=True)[3]
            or protocol["sae_release"] != checkpoint_spec(14, all_layers=True)[0]
            or protocol["device"] != "mps" or protocol["model_dtype"] != "torch.bfloat16"):
        raise ValueError("Expected the pinned layer-14 MPS/BF16 discovery protocol")
    for path, digest in cohort["metadata_sha256"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError("Source label/annotation metadata changed")
    records = []
    hashes = {name: file_sha256(source / name) for name in
              ("protocol.json", "cohort.json", "shortlist.json", "discovery_report.json")}
    for case in cases:
        record, _ = load_discovery_case(source, case["case_id"], protocol)
        if Path(record["image_path"]).parent.parent.parent.name != "val":
            raise ValueError("Only original validation paths may be loaded")
        records.append(record)
        for extension in ("json", "pt"):
            name = f"cases/{case['case_id']}.{extension}"
            hashes[name] = file_sha256(source / name)
    return protocol, records, hashes


def experiment_protocol(source, destination, *, repaired=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Intervention output must be separate from E-004")
    original, records, hashes = validate_source(source)
    settings = {"schema_version": 1, "experiment": "E-005", "source_directory": str(source),
                "source_protocol_hash": original["protocol_hash"], "source_artifact_sha256": hashes,
                "source_image_sha256": {r["case_id"]: r["image_sha256"] for r in records},
                "model_id": original["model_id"], "model_revision": original["model_revision"],
                "processor_revision": original["processor_revision"], "prompt": original["prompt"],
                "sae_revision": original["sae_revision"], "sae_release": original["sae_release"],
                "sae_id": original["sae_id"], "sae_width": original["sae_width"], "hook": original["hook"],
                "scoring": original["scoring"],
                "discovery_case_ids": [r["case_id"] for r in records],
                "position_definitions": {"image": "all valid processor image positions",
                    "final": "last valid input position", "other_non_image": "valid non-image minus final"},
                "fractions": list(FRACTIONS), "thresholds": THRESHOLDS, "feature_chunk_size": 256,
                "runtime": {"device": "mps", "model_dtype": "torch.bfloat16", "sae": "cpu/float32",
                    "batch_size": 1, "use_cache": False, "logits_to_keep": 1, "frozen_parameters": True},
                "analysis": {"seed": 42, "bootstrap_resamples": 1000, "targets_per_group": 5,
                    "ranking": "mean absolute full-removal estimate after averaging images within patient",
                    "matching": "squared activity-fraction difference + squared log RMS difference",
                    "rms": "sqrt(mean_patient(mean_image(sum_t ||z_j*w_j||^2 / (group_tokens*hidden_width))))",
                    "stages": ["pilot", "attribute", "select", "ablate", "report"],
                    "confirmation_enabled": False, "test_enabled": False},
                "upstream": UPSTREAM,
                "versions": {name: version(name) for name in ("torch", "transformers", "sae-lens", "scipy")},
                "source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in (
                    "feature_interventions.py", "medgemma_interventions.py", "medgemma_features.py",
                    "sae_reconstruction.py", "inference.py", "chexlocalize.py")}}
    if repaired is None:
        settings["pilot"] = choose_pilot(records)
    else:
        settings.update(schema_version=3, experiment="E-007", fresh_encodings=True,
                        calibration={"kind": "full_discovery", "case_ids": settings["discovery_case_ids"],
                                     "unit": "image-feature pair; unchanged E-005 criteria"},
                        smoke_ids=repaired["smoke_ids"], previous_artifacts=repaired["previous_artifacts"],
                        preflight_sha256=repaired["preflight_sha256"])
        settings["runtime"]["attention_implementation"] = {"text_config": "eager", "vision_config": "sdpa"}
        settings["analysis"].update(stages=["smoke", "calibrate", "select", "ablate", "report"],
            primary_evidence="attribution rank with direct half/full calibration support; measured cohort ablation effects",
            target_eligibility="active feature with direct calibration measurements at both removal strengths; no effect-size cutoff",
            supporting_evidence="patient-level activation label AUC and answer-score Spearman; not selection criteria")
        settings["source_sha256"].update({name: file_sha256(Path(__file__).with_name(name)) for name in
                                        ("medgemma_discovery.py", "feature_analysis.py")})
    path = destination / "protocol.json"
    if path.exists():
        saved = read_json(path)
        if (identity_hash({k: v for k, v in saved.items() if k != "protocol_hash"}) != saved["protocol_hash"]
                or any(saved.get(k) != v for k, v in settings.items())):
            raise ValueError("Incompatible resume: create a new output directory")
        return saved, records
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output directory is nonempty without a valid protocol")
    settings.update(code_revision=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                text=True, check=True).stdout.strip(),
                    code_dirty=bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                                  text=True, check=True).stdout))
    settings["protocol_hash"] = identity_hash(settings)
    atomic_json(path, settings)
    return settings, records


def save_case(directory, protocol, record, payload):
    path = directory / "cases" / f"{record['case_id']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    atomic_json(path.with_suffix(".json"), {**record, "protocol_hash": protocol["protocol_hash"],
                                          "tensor_sha256": file_sha256(path)})


def load_case(directory, protocol, case_id):
    if case_id not in protocol["discovery_case_ids"]:
        raise ValueError("Only fixed discovery cases can be loaded")
    record = read_json(directory / "cases" / f"{case_id}.json")
    path = directory / "cases" / f"{case_id}.pt"
    if record["protocol_hash"] != protocol["protocol_hash"] or file_sha256(path) != record["tensor_sha256"]:
        raise ValueError("Intervention shard identity/checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for name in ("hidden", "gradient", "baseline_logits"):
        if not torch.isfinite(payload[name]).all():
            raise ValueError("Nonfinite saved intervention tensor")
    attribution = payload["attribution"]
    source_record, source_payload = load_discovery_case(Path(protocol["source_directory"]), case_id)
    source_features = source_payload["features"]
    features = payload["features"] if protocol.get("fresh_encodings") else source_features
    if (record["source_record_sha256"] != protocol["source_artifact_sha256"][f"cases/{case_id}.json"]
            or file_sha256(Path(protocol["source_directory"]) / "cases" / f"{case_id}.json") != record["source_record_sha256"]
            or record["tokens"] != source_record["tokens"]
            or attribution.shape != source_features.shape or not attribution.is_coalesced()
            or features.shape != source_features.shape or not features.is_coalesced()
            or not torch.isfinite(features.values()).all() or (features.values() < 0).any()
            or not torch.equal(attribution.indices(), features.indices())
            or not torch.isfinite(attribution.values()).all()
            or payload["hidden"].shape != payload["gradient"].shape
            or payload["hidden"].shape[:2] != (1, len(record["tokens"]))):
        raise ValueError("Intervention/source tensor alignment mismatch")
    return record, payload


def frozen_runtime(protocol, destination):
    from medgemma_grounding.inference import load_model
    from medgemma_grounding.sae_reconstruction import load_scope_sae

    if not torch.backends.mps.is_available():
        raise RuntimeError("The fixed E-005 runtime requires MPS; no device substitution")
    attention = protocol["runtime"].get("attention_implementation")
    bundle = load_model(protocol["model_id"], device="mps", revision=protocol["model_revision"],
                        **({"attn_implementation": attention} if attention is not None else {}))
    if attention is not None:
        expected = ({k: attention for k in attention_settings(bundle.model)}
                    if isinstance(attention, str) else attention)
        if any(attention_settings(bundle.model)[k] != value for k, value in expected.items()):
            raise ValueError("Requested attention implementation was not applied")
    sae, _, _, revision = load_scope_sae(14, all_layers=True)
    for module in (bundle.model, sae):
        module.eval()
        module.requires_grad_(False)
        if any(p.grad is not None for p in module.parameters()):
            raise RuntimeError("Unexpected pre-existing parameter gradients")
    if bundle.model.dtype != torch.bfloat16 or revision != protocol["sae_revision"]:
        raise ValueError("Runtime precision/checkpoint differs")
    runtime = {"protocol_hash": protocol["protocol_hash"], "attention_implementation":
               {name: getattr(config, "_attn_implementation", None) for name, config in (
                   ("model", bundle.model.config),
                   ("text", bundle.model.config.text_config), ("vision", bundle.model.config.vision_config))}}
    path = destination / "runtime.json"
    if path.exists() and read_json(path) != runtime:
        raise ValueError("Attention implementation changed; no silent substitution")
    atomic_json(path, runtime)
    return bundle, sae


def prepared_inputs(bundle, protocol, source_record):
    from medgemma_grounding.inference import load_image, prepare_inputs

    image = load_image(source_record["image_path"])
    inputs = prepare_inputs(bundle, image, protocol["prompt"])
    if (inputs["input_ids"].shape[0] != 1 or inputs["input_ids"][0].cpu().tolist() !=
            [t["id"] for t in source_record["tokens"]]
            or answer_candidates(bundle.processor.tokenizer, inputs["input_ids"]) != protocol["scoring"]):
        raise ValueError("Source token IDs/scoring boundary changed")
    groups = position_groups(inputs["input_ids"], inputs["attention_mask"],
                            bundle.model.config.image_token_index, bundle.processor.tokenizer.all_special_ids)
    groups["ordinary_text"] = prompt_positions(inputs["input_ids"],
        bundle.processor.tokenizer.encode(protocol["prompt"], add_special_tokens=False))
    if any(groups[name].nonzero().flatten().tolist() != source_record["positions"][name]
           for name in groups):
        raise ValueError("Source position masks changed")
    layout = spatial_layout(bundle, inputs, image.size)
    if layout != source_record["spatial_layout"]:
        raise ValueError("Source spatial layout changed")
    return inputs, intervention_groups(groups)


def forward(bundle, inputs):
    with torch.inference_mode():
        logits = bundle.model(**inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite model output")
    return logits


class IdentityCheckError(AssertionError):
    """Retain scores and numerical diagnostics in failed-run artifacts."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"{diagnostics['control']} identity failed: {diagnostics}")


def identity_diagnostics(baseline, changed, candidates, name="identity"):
    diagnostics = {"control": name, **tensor_diagnostics(baseline, changed)}
    if diagnostics["finite"]:
        original, scored = answer_score(baseline, candidates), answer_score(changed, candidates)
        diagnostics.update(original_answer=original, changed_answer=scored,
                           decision_preserved=scored["decision"] == original["decision"],
                           score_difference=scored["score"] - original["score"],
                           max_logit_difference=diagnostics["max_absolute_difference"],
                           **logit_metrics(baseline, changed))
    return diagnostics


def identity_control(baseline, changed, candidates, name="identity"):
    diagnostics = identity_diagnostics(baseline, changed, candidates, name)
    if not diagnostics["allclose"] or not diagnostics.get("decision_preserved", False):
        raise IdentityCheckError(diagnostics)
    return diagnostics


def collect_attribution(bundle, sae, protocol, source_record, source_payload, *, pilot=False):
    start = time.perf_counter()
    inputs, masks = prepared_inputs(bundle, protocol, source_record)
    baseline = forward(bundle, inputs)
    hidden, passive = capture_residual(bundle.model, inputs, protocol["hook"])
    controls = {"passive": identity_control(baseline, passive, protocol["scoring"])}
    controls["unchanged"] = identity_control(baseline,
        replace_residual(bundle.model, inputs, protocol["hook"], hidden), protocol["scoring"])
    _, features = encode_decode(sae, hidden[0])
    if not torch.isfinite(features).all() or (features < 0).any():
        raise ValueError("Invalid fresh SAE features")
    if not protocol.get("fresh_encodings"):
        torch.testing.assert_close(features, source_payload["features"].to_dense(), rtol=1e-5, atol=1e-5)
    scored = answer_score(baseline, protocol["scoring"])
    if not protocol.get("fresh_encodings") and any(scored[key] != source_record["answer_score"][key] for key in ("yes_logit", "no_logit", "decision")):
        raise ValueError("Original answer score changed from E-004")
    zero, _ = feature_removal(hidden, features, sae.W_dec, 0, masks["image"], 0, bundle.model.dtype)
    controls["zero_strength"] = identity_control(baseline,
        replace_residual(bundle.model, inputs, protocol["hook"], zero), protocol["scoring"])
    gradient_hidden, gradient_logits, gradient = capture_gradient(
        bundle.model, inputs, protocol["hook"], protocol["scoring"])
    torch.testing.assert_close(gradient_hidden, hidden, rtol=1e-5, atol=1e-5)
    controls["gradient_forward"] = identity_control(baseline, gradient_logits, protocol["scoring"], "gradient_forward")
    repeated = [scored["score"]]
    if pilot:
        for _ in range(3):
            repeated_logits = forward(bundle, inputs)
            repeated_score = answer_score(repeated_logits, protocol["scoring"])
            if repeated_score["decision"] != scored["decision"]:
                raise ValueError("Repeated baseline changed the original decision")
            repeated.append(repeated_score["score"])
    attribution, summaries = project_attribution(gradient, features, sae.W_dec, masks)
    record = {"case_id": source_record["case_id"], "patient": source_record["patient"],
              "label": source_record["label"], "outcome": outcome({**source_record, "answer_score": scored}), "answer_score": scored,
              "source_record_sha256": protocol["source_artifact_sha256"][f"cases/{source_record['case_id']}.json"],
              "tokens": source_record["tokens"], "spatial_layout": source_record["spatial_layout"],
              "source_positions": source_record["positions"],
              "positions": {k: m.nonzero().flatten().tolist() for k, m in masks.items()},
              "controls": controls, "repeated_scores": repeated,
              "baseline_variation": max(repeated) - min(repeated), "pilot": pilot,
              "runtime_seconds": time.perf_counter() - start}
    payload = {"hidden": hidden, "gradient": gradient, "baseline_logits": baseline,
               "attribution": attribution, "groups": summaries}
    if protocol.get("fresh_encodings"):
        payload["features"] = features.to_sparse().coalesce()
        for group, mask in masks.items():
            summaries[group]["activation_mean"] = features[mask].mean(0)
        record["fresh_encodings"] = True
        record["pilot"] = False
        record["calibration_scope"] = "full_discovery"
        record["reference_sdpa"] = {"outcome": outcome(source_record), "answer_score": source_record["answer_score"],
            "feature_compatibility": tensor_diagnostics(source_payload["features"].to_dense(), features)}
    return record, payload


def calibration_features(case_id, group, summary):
    active = summary["active"].nonzero().flatten().tolist()
    ranked = sorted(active, key=lambda j: (-abs(summary["effect"][j].item()), j))
    if len(ranked) <= 2:
        return ranked
    chosen = [ranked[0], ranked[len(ranked) // 2]]
    remaining = [j for j in active if j not in chosen]
    chosen.append(min(remaining, key=lambda j: (hashed(f"42:{case_id}:{group}:{j}"), j)))
    return chosen


def treatment_path(directory, case_id, feature, group, alpha):
    if group not in GROUPS or alpha not in FRACTIONS or feature < 0:
        raise ValueError("Invalid treatment identity")
    return directory / "treatments" / case_id / f"{group}_{feature}_{alpha:g}.json"


def load_treatment(path, protocol, case_id, feature, group, alpha):
    record = read_json(path)
    expected = {"protocol_hash": protocol["protocol_hash"], "case_id": case_id,
                "feature": feature, "group": group, "alpha": alpha}
    if (any(record.get(k) != v for k, v in expected.items())
            or record.get("treatment_hash") != identity_hash({k: v for k, v in record.items() if k != "treatment_hash"})):
        raise ValueError("Treatment checksum/identity mismatch")
    return record


def measure_treatment(bundle, protocol, record, payload, features, decoder, inputs, feature, group, alpha):
    baseline = payload["baseline_logits"]
    mask = torch.zeros(features.shape[0], dtype=torch.bool)
    mask[record["positions"][group]] = True
    inactive = not bool(features[mask, feature].count_nonzero())
    edited, norms = feature_removal(payload["hidden"], features, decoder, feature, mask, alpha, bundle.model.dtype)
    changed = baseline if inactive else replace_residual(bundle.model, inputs, protocol["hook"], edited)
    scored = answer_score(changed, protocol["scoring"])
    correct = scored["decision"] == ("Yes" if record["label"] else "No")
    original_correct = record["outcome"].startswith("T")
    return {"protocol_hash": protocol["protocol_hash"], "case_id": record["case_id"],
            "patient": record["patient"], "outcome": record["outcome"], "feature": feature,
            "group": group, "alpha": alpha, "inactive": inactive, **norms,
            "predicted_delta": alpha * payload["groups"][group]["effect"][feature].item(),
            "actual_delta": scored["score"] - record["answer_score"]["score"],
            "original_score": record["answer_score"]["score"], "original_decision": record["answer_score"]["decision"],
            "original_correct": original_correct, "changed_score": scored["score"],
            "changed_decision": scored["decision"], "changed_correct": correct,
            "decision_flip": scored["decision"] != record["answer_score"]["decision"],
            "corrected": bool(correct and not original_correct), "introduced": bool(original_correct and not correct),
            **logit_metrics(baseline, changed)}


def save_treatment(path, record):
    record["treatment_hash"] = identity_hash(record)
    atomic_json(path, record)


def agreement(rows, tau):
    measurable = [r for r in rows if abs(r["actual_delta"]) >= tau]
    if not measurable:
        return {"pairs": 0, "spearman": None, "sign_agreement": None, "median_relative_error": None}
    predicted, actual = (np.array([r[k] for r in measurable]) for k in ("predicted_delta", "actual_delta"))
    rho = float(spearmanr(predicted, actual).statistic) if np.ptp(predicted) and np.ptp(actual) else None
    return {"pairs": len(measurable), "spearman": rho,
            "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
            "median_relative_error": float(np.median(abs(predicted - actual) / np.maximum(abs(actual), tau)))}


def pilot_report(directory, protocol):
    """Legacy pilot or full-discovery calibration, as fixed in the protocol."""
    records, rows = [], []
    complete = True
    completed_cases = 0
    full = "calibration" in protocol
    for case_id in (protocol["calibration"]["case_ids"] if full else protocol["pilot"]["case_ids"]):
        if not (directory / "cases" / f"{case_id}.json").exists():
            complete = False
            continue
        record, payload = load_case(directory, protocol, case_id)
        records.append(record)
        case_complete = True
        if len(record["repeated_scores"]) != 4 or not (record.get("fresh_encodings") if full else record["pilot"]):
            raise ValueError("Calibration lacks fresh encoding or repeated baselines")
        for group in GROUPS:
            for feature in calibration_features(case_id, group, payload["groups"][group]):
                for alpha in FRACTIONS:
                    path = treatment_path(directory, case_id, feature, group, alpha)
                    if path.exists():
                        rows.append(load_treatment(path, protocol, case_id, feature, group, alpha))
                    else:
                        complete = False
                        case_complete = False
        completed_cases += int(case_complete)
    noise = max((r["baseline_variation"] for r in records), default=0)
    tau = max(THRESHOLDS["floor_logits"], THRESHOLDS["noise_multiplier"] * noise)
    groups = {}
    for group in GROUPS:
        half = agreement([r for r in rows if r["group"] == group and r["alpha"] == 0.5], tau)
        enough = half["pairs"] >= THRESHOLDS["min_pairs"]
        qualified = complete and enough and half["spearman"] is not None and (
            half["spearman"] >= THRESHOLDS["spearman"] and half["sign_agreement"] >= THRESHOLDS["sign_agreement"]
            and half["median_relative_error"] <= THRESHOLDS["relative_error"])
        groups[group] = {"status": "qualifies" if qualified else "insufficient_signal" if not enough else "poor_approximation",
                         "half": half, "full": agreement([r for r in rows if r["group"] == group and r["alpha"] == 1], tau)}
    result = {"protocol_hash": protocol["protocol_hash"], "complete": complete, "cases": len(records),
              "tau": tau, "largest_baseline_variation": noise, "groups": groups,
              "qualifying_groups": [k for k, v in groups.items() if v["status"] == "qualifies"],
              "measurements": rows, "failure_diagnostics": sorted(p.name for p in directory.glob(
                  "failure_calibrate.json" if full else "failure_pilot.json"))}
    result["report_hash"] = identity_hash(result)
    if full:
        result.update(completed_cases=completed_cases, calibration_kind="full_discovery")
        result["report_hash"] = identity_hash({k: v for k, v in result.items() if k != "report_hash"})
    atomic_json(directory / ("calibration_report.json" if full else "pilot_report.json"), result)
    return result


def patient_rankings(records, payloads):
    """Average images within patients before any cohort mean or ranking."""
    patients = sorted({r["patient"] for r in records})
    rows = []
    for group in GROUPS:
        effects = np.stack([p["groups"][group]["effect"].numpy() for p in payloads])
        active = np.stack([p["groups"][group]["active"].numpy() for p in payloads])
        rms = np.stack([p["groups"][group]["removal_rms"].numpy() for p in payloads])
        l2 = np.stack([p["groups"][group]["removal_l2"].numpy() for p in payloads])
        patient_effects, patient_active, patient_rms, patient_l2 = [], [], [], []
        for patient in patients:
            ix = [i for i, r in enumerate(records) if r["patient"] == patient]
            patient_effects.append(effects[ix].mean(0))
            patient_active.append(active[ix].any(0))
            patient_rms.append((rms[ix] ** 2).mean(0))
            patient_l2.append((l2[ix] ** 2).mean(0))
        means = np.stack(patient_effects)
        activity = np.stack(patient_active).sum(0)
        rms_mean, l2_mean = np.sqrt(np.mean(patient_rms, axis=0)), np.sqrt(np.mean(patient_l2, axis=0))
        by_outcome = {}
        for category in QUOTAS:
            grouped = [effects[[i for i, r in enumerate(records) if r["patient"] == patient and r["outcome"] == category]].mean(0)
                       for patient in patients if any(r["patient"] == patient and r["outcome"] == category for r in records)]
            by_outcome[category] = np.mean(grouped, axis=0) if grouped else None
        for feature in range(means.shape[1]):
            rows.append({"feature": feature, "group": group,
                         "mean_absolute_effect": float(abs(means[:, feature]).mean()),
                         "mean_signed_effect": float(means[:, feature].mean()),
                         "active_patients": int(activity[feature]), "activity_fraction": float(activity[feature] / len(patients)),
                         "removal_rms": float(rms_mean[feature]), "removal_l2": float(l2_mean[feature]),
                         **{f"{k}_mean_signed": float(v[feature]) if v is not None else None for k, v in by_outcome.items()}})
    for group in GROUPS:
        ordered = sorted((r for r in rows if r["group"] == group), key=lambda r: (-r["mean_absolute_effect"], r["feature"]))
        for rank, row in enumerate(ordered, 1):
            row["rank"] = rank
    return rows


def matched_candidates(rows, qualifying):
    pairs = []
    for group in qualifying:
        pool = [r for r in rows if r["group"] == group and r["active_patients"] > 0]
        supported = [r for r in pool if r.get("calibration_half_cases", 1) and r.get("calibration_full_cases", 1)]
        targets = sorted(supported, key=lambda r: (-r["mean_absolute_effect"], r["feature"]))[:5]
        excluded = {r["feature"] for r in targets}
        for target in targets:
            available = [r for r in pool if r["feature"] not in excluded]
            if not available:
                raise ValueError("Insufficient active comparison features")
            def distance(row):
                return ((row["activity_fraction"] - target["activity_fraction"]) ** 2
                        + (np.log(row["removal_rms"] + 1e-12) - np.log(target["removal_rms"] + 1e-12)) ** 2)
            comparison = min(available, key=lambda r: (distance(r), r["feature"]))
            excluded.add(comparison["feature"])
            pairs.append({"group": group, "target": target, "comparison": comparison,
                          "matching_distance": float(distance(comparison))})
    return pairs


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def cohort_selection(directory, protocol, ranking_rows=None):
    pilot = pilot_report(directory, protocol)
    if not pilot["complete"] or not pilot["qualifying_groups"]:
        raise ValueError("Completed qualifying pilot required for selection")
    records, payloads = [], []
    for case_id in protocol["discovery_case_ids"]:
        if not (directory / "cases" / f"{case_id}.json").exists():
            raise ValueError("Selection requires complete cohort attribution")
        record, payload = load_case(directory, protocol, case_id)
        records.append(record)
        # Keep only compact CPU summaries, not full shards/hidden states.
        payloads.append({"groups": payload["groups"]})
    rows = patient_rankings(records, payloads) if ranking_rows is None else ranking_rows
    pairs = matched_candidates(rows, pilot["qualifying_groups"])
    old = read_json(Path(protocol["source_directory"]) / "shortlist.json")
    calibration_key = "calibration_report_hash" if protocol.get("fresh_encodings") else "pilot_report_hash"
    result = {"protocol_hash": protocol["protocol_hash"], calibration_key: pilot["report_hash"],
              "input_tensor_sha256": {r["case_id"]: r["tensor_sha256"] for r in records},
              "pairs": pairs, "old_shortlist_ranks": [r for r in rows if r["feature"] in {c["feature"] for c in old["candidates"]}],
              "qualifying_groups": pilot["qualifying_groups"], "interpretation": "Discovery-selected interventions; comparisons are not nulls"}
    result["candidates_hash"] = identity_hash(result)
    path = directory / "candidates.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("Frozen intervention candidates changed")
    write_csv(directory / "attribution_rankings.csv", rows)
    atomic_json(path, result)
    return result


def read_candidates(directory, protocol):
    candidates = read_json(directory / "candidates.json")
    if (candidates["protocol_hash"] != protocol["protocol_hash"] or
            candidates["candidates_hash"] != identity_hash({k: v for k, v in candidates.items() if k != "candidates_hash"})):
        raise ValueError("Candidate freeze checksum mismatch")
    pilot = pilot_report(directory, protocol)
    calibration_key = "calibration_report_hash" if protocol.get("fresh_encodings") else "pilot_report_hash"
    if candidates[calibration_key] != pilot["report_hash"]:
        raise ValueError("Pilot calibration changed after selection")
    for case_id, digest in candidates["input_tensor_sha256"].items():
        record, _ = load_case(directory, protocol, case_id)
        if record["tensor_sha256"] != digest:
            raise ValueError("Attribution shard changed after selection")
    return candidates


def treatment_specs(candidates):
    return [(pair[role]["feature"], pair["group"], alpha, role)
            for pair in candidates["pairs"] for role in ("target", "comparison") for alpha in FRACTIONS]


def bootstrap_summary(rows):
    """Descriptive patient bootstrap, conditional on discovery selection."""
    patients = sorted({r["patient"] for r in rows})
    keys = ("actual_delta", "absolute_effect", "predicted_delta", "kl_from_baseline",
            "realized_l2", "corrected", "introduced")
    data = np.array([[np.mean([abs(r["actual_delta"]) if key == "absolute_effect" else r[key]
                              for r in rows if r["patient"] == patient]) for key in keys] for patient in patients])
    if not len(data):
        return {"patients": 0}
    samples = np.random.default_rng(42).integers(0, len(data), size=(1000, len(data)))
    boot = data[samples].mean(1)
    return {"patients": len(patients), "images": len(rows),
            **{key: {"mean": float(data[:, i].mean()), "ci95": np.quantile(boot[:, i], [.025, .975]).tolist()}
               for i, key in enumerate(keys)}}


def offline_report(directory, protocol):
    pilot = pilot_report(directory, protocol)
    rows, summaries, contrasts = [], [], []
    expected = 0
    candidates = read_candidates(directory, protocol) if (directory / "candidates.json").exists() else None
    if candidates:
        specs = treatment_specs(candidates)
        expected = len(specs) * len(protocol["discovery_case_ids"])
        for case_id in protocol["discovery_case_ids"]:
            for feature, group, alpha, role in specs:
                path = treatment_path(directory, case_id, feature, group, alpha)
                if path.exists():
                    rows.append({**load_treatment(path, protocol, case_id, feature, group, alpha), "role": role})
        for feature, group, alpha, role in specs:
            selected = [r for r in rows if (r["feature"], r["group"], r["alpha"]) == (feature, group, alpha)]
            summaries.append({"feature": feature, "group": group, "alpha": alpha, "role": role,
                              "overall": bootstrap_summary(selected),
                              "correct_answers": bootstrap_summary([r for r in selected if r["original_correct"]]),
                              "outcomes": {k: bootstrap_summary([r for r in selected if r["outcome"] == k]) for k in QUOTAS},
                              "corrected_images": sum(r["corrected"] for r in selected),
                              "introduced_images": sum(r["introduced"] for r in selected),
                              "agreement": agreement(selected, pilot["tau"])})
        for pair in candidates["pairs"]:
            for alpha in FRACTIONS:
                targets = {r["case_id"]: r for r in rows if r["group"] == pair["group"] and
                           r["feature"] == pair["target"]["feature"] and r["alpha"] == alpha}
                comparisons = {r["case_id"]: r for r in rows if r["group"] == pair["group"] and
                               r["feature"] == pair["comparison"]["feature"] and r["alpha"] == alpha}
                differences = []
                for case_id in targets.keys() & comparisons.keys():
                    target, control = targets[case_id], comparisons[case_id]
                    differences.append({**target, "actual_delta": target["actual_delta"] - control["actual_delta"],
                                        "predicted_delta": target["predicted_delta"] - control["predicted_delta"]})
                # Paired signed differences only; do not reinterpret control effects as nulls.
                contrasts.append({"group": pair["group"], "target": pair["target"]["feature"],
                                  "comparison": pair["comparison"]["feature"], "alpha": alpha,
                                  "signed_difference": bootstrap_summary(differences).get("actual_delta")})
    write_csv(directory / "intervention_effects.csv", rows)
    completed_cases = [c for c in protocol["discovery_case_ids"] if (directory / "cases" / f"{c}.json").exists()]
    limits = torch.zeros(protocol["sae_width"])
    activity_limits = torch.zeros_like(limits)
    compact = {g: [] for g in GROUPS}
    for case_id in completed_cases:
        _, payload = load_case(directory, protocol, case_id)
        sparse = payload["attribution"]
        limits.scatter_reduce_(0, sparse.indices()[1], sparse.values().abs(), reduce="amax")
        if protocol.get("fresh_encodings"):
            features = payload["features"]
            activity_limits.scatter_reduce_(0, features.indices()[1], features.values(), reduce="amax")
        for group in GROUPS:
            compact[group].append(payload["groups"][group]["effect"])
    tensor_path = directory / "intervention_summaries.pt"
    temporary = tensor_path.with_suffix(".pt.tmp")
    torch.save({"case_ids": completed_cases, "signed_color_limits": limits, "activation_color_limits": activity_limits,
                "group_effects": {g: torch.stack(v) if v else torch.empty(0, protocol["sae_width"])
                                  for g, v in compact.items()}}, temporary)
    temporary.replace(tensor_path)
    report = {"protocol_hash": protocol["protocol_hash"], "pilot_complete": pilot["complete"],
              "qualifying_groups": pilot["qualifying_groups"], "attributed_cases": len(completed_cases),
              "expected_cases": len(protocol["discovery_case_ids"]), "measured_treatments": len(rows),
              "expected_treatments": expected, "complete": pilot["complete"] and (
                  not pilot["qualifying_groups"] or candidates is not None and len(rows) == expected),
              "status": ("calibration_stopped" if protocol.get("fresh_encodings") else "pilot_stopped") if pilot["complete"] and not pilot["qualifying_groups"] else
                  "cohort_complete" if candidates and len(rows) == expected else "partial",
              "summaries": summaries, "target_minus_comparison": contrasts,
              "limitations": ["Discovery-selected estimates; bootstrap intervals are descriptive, not independent confirmation",
                  "Matched comparisons are not null features", "Position attribution is contextual; no pixel-level causality or localization",
                  "One transferred SAE at layer 14; no circuit or unseen-patient mitigation claim"],
              "failure_diagnostics": sorted(p.name for p in directory.glob("failure_*.json"))}
    report["summary_tensor_sha256"] = file_sha256(tensor_path)
    report["effects_csv_sha256"] = file_sha256(directory / "intervention_effects.csv")
    atomic_json(directory / "intervention_report.json", report)
    return report


def run(stage, source=SOURCE, output=DESTINATION, limit=None):
    """Explicit stage entry point; prerequisites precede model loading."""
    if stage not in {"pilot", "attribute", "select", "ablate", "report"}:
        raise ValueError("An explicit valid stage is required")
    if limit is not None and (limit < 1 or stage not in {"attribute", "ablate"}):
        raise ValueError("--limit is positive and supported only for attribute/ablate")
    directory = Path(output)
    protocol, source_records = experiment_protocol(source, directory)
    try:
        if stage == "report":
            return offline_report(directory, protocol)
        if stage == "select":
            return cohort_selection(directory, protocol)
        if stage != "pilot":
            pilot = pilot_report(directory, protocol)
            if not pilot["complete"] or not pilot["qualifying_groups"]:
                raise ValueError("Pilot must complete and qualify a group before model expansion")
        candidates = read_candidates(directory, protocol) if stage == "ablate" else None
        by_id = {r["case_id"]: r for r in source_records}
        ids = protocol["pilot"]["case_ids"] if stage == "pilot" else protocol["discovery_case_ids"]
        pending = []
        for case_id in ids:
            path = directory / "cases" / f"{case_id}.json"
            loaded = load_case(directory, protocol, case_id) if path.exists() else None
            if stage == "ablate" and loaded is None:
                raise ValueError("Ablation requires complete saved attribution")
            if stage == "attribute" and loaded:
                continue
            if stage == "pilot" and loaded:
                _, payload = loaded
                specs = [(j, g, a, "calibration") for g in GROUPS
                         for j in calibration_features(case_id, g, payload["groups"][g]) for a in FRACTIONS]
            elif stage == "ablate":
                specs = treatment_specs(candidates)
            else:
                specs = []
            missing = loaded is None
            for feature, group, alpha, _ in specs:
                treatment = treatment_path(directory, case_id, feature, group, alpha)
                if treatment.exists():
                    load_treatment(treatment, protocol, case_id, feature, group, alpha)
                else:
                    missing = True
            if missing:
                pending.append(case_id)
        if not pending:
            return pilot_report(directory, protocol) if stage == "pilot" else {"new_cases": 0}
        if limit is not None:
            pending = pending[:limit]
        bundle, sae = frozen_runtime(protocol, directory)
        started = time.perf_counter()
        for index, case_id in enumerate(pending, 1):
            case_start = time.perf_counter()
            logging.info("%s %d/%d %s", stage, index, len(pending), case_id)
            source_record, source_payload = load_discovery_case(Path(source), case_id)
            path = directory / "cases" / f"{case_id}.json"
            if not path.exists():
                record, payload = collect_attribution(bundle, sae, protocol, source_record, source_payload, pilot=stage == "pilot")
                save_case(directory, protocol, record, payload)
            record, payload = load_case(directory, protocol, case_id)
            if stage in {"pilot", "ablate"}:
                inputs, _ = prepared_inputs(bundle, protocol, source_record)
                baseline = forward(bundle, inputs)
                identity_control(payload["baseline_logits"], baseline, protocol["scoring"])
                # All direct effects use this pass's baseline after saved-state alignment.
                payload = {**payload, "baseline_logits": baseline}
                record = {**record, "answer_score": answer_score(baseline, protocol["scoring"])}
                hidden, captured = capture_residual(bundle.model, inputs, protocol["hook"])
                identity_control(baseline, captured, protocol["scoring"])
                torch.testing.assert_close(hidden, payload["hidden"], rtol=1e-5, atol=1e-5)
                features = source_payload["features"].to_dense()
                specs = treatment_specs(candidates) if stage == "ablate" else [
                    (j, g, a, "calibration") for g in GROUPS
                    for j in calibration_features(case_id, g, payload["groups"][g]) for a in FRACTIONS]
                for feature, group, alpha, _ in specs:
                    path = treatment_path(directory, case_id, feature, group, alpha)
                    if path.exists():
                        load_treatment(path, protocol, case_id, feature, group, alpha)
                        continue
                    treatment_start = time.perf_counter()
                    measured = measure_treatment(bundle, protocol, record, payload, features, sae.W_dec,
                                                 inputs, feature, group, alpha)
                    measured["runtime_seconds"] = time.perf_counter() - treatment_start
                    save_treatment(path, measured)
                    logging.info("Saved %s j=%d alpha=%g delta=%+.4f (%.1fs)", group, feature, alpha,
                                 measured["actual_delta"], measured["runtime_seconds"])
            logging.info("Saved case %.1fs; elapsed %.1fm; empirical remaining ETA %.1fm", time.perf_counter() - case_start,
                         (time.perf_counter() - started) / 60,
                         (time.perf_counter() - started) / index * (len(pending) - index) / 60)
        return pilot_report(directory, protocol) if stage == "pilot" else {"new_cases": len(pending)}
    except Exception as error:
        atomic_json(directory / f"failure_{stage}.json", {"protocol_hash": protocol["protocol_hash"],
            "stage": stage, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "diagnostics": getattr(error, "diagnostics", None),
            "case_id": locals().get("case_id"), "settings_changed": False})
        if stage == "pilot":
            pilot_report(directory, protocol)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["pilot", "attribute", "select", "ablate", "report"])
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.stage, args.source_dir, args.output_dir, args.limit)
    logging.info("Stage %s finished: %s", args.stage, {k: v for k, v in result.items() if k in
                 {"complete", "status", "qualifying_groups", "cases", "new_cases"}})


if __name__ == "__main__":
    main()
