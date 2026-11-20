"""Shared layer-14 feature capture, numerical controls and direct intervention execution.

Edits preserve the reconstruction residual; reports and selection live in analysis/.
"""


from __future__ import annotations

from pathlib import Path
import time

import torch

from medgemma_grounding.analysis.metrics import outcome
from medgemma_grounding.data.image_positions import spatial_layout
from medgemma_grounding.interventions.features import (
    GROUPS,
    capture_gradient,
    feature_removal,
    intervention_groups,
    project_attribution,
)
from medgemma_grounding.interventions.protocol import FRACTIONS
from medgemma_grounding.model import answer_candidates, answer_score, attention_settings
from medgemma_grounding.sae.cache import load_discovery_case
from medgemma_grounding.sae.reconstruction import (
    capture_residual,
    encode_decode,
    logit_metrics,
    position_groups,
    prompt_positions,
    replace_residual,
)
from medgemma_grounding.utils.io import atomic_json, file_sha256, identity_hash, read_json
from medgemma_grounding.utils.numerics import tensor_diagnostics


def treatment_context(bundle, sae, protocol, record, payload):
    """Recheck original state, and ALWAYS use the new backend's saved encoding."""
    original, _ = load_discovery_case(Path(protocol["source_directory"]), record["case_id"])
    inputs, _ = prepared_inputs(bundle, protocol, original)
    baseline = forward(bundle, inputs)
    identity_control(payload["baseline_logits"], baseline, protocol["scoring"], "resumed_baseline")
    hidden, logits = capture_residual(bundle.model, inputs, protocol["hook"])
    identity_control(baseline, logits, protocol["scoring"], "resumed_capture")
    torch.testing.assert_close(hidden, payload["hidden"], rtol=1e-5, atol=1e-5)
    _, encoded = encode_decode(sae, hidden[0])
    features = payload["features"].to_dense()
    torch.testing.assert_close(encoded, features, rtol=1e-5, atol=1e-5)
    return inputs, features, {**record, "answer_score": answer_score(baseline, protocol["scoring"])}, {
        **payload, "baseline_logits": baseline}


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
    from medgemma_grounding.model import load_model
    from medgemma_grounding.sae.reconstruction import load_scope_sae

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
    from medgemma_grounding.model import load_image
    from medgemma_grounding.model import prepare_inputs

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


def treatment_specs(candidates):
    return [(pair[role]["feature"], pair["group"], alpha, role)
            for pair in candidates["pairs"] for role in ("target", "comparison") for alpha in FRACTIONS]
