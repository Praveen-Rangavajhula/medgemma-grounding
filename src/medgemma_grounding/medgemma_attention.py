"""E-009: fixed discovery-only image-to-answer attention-knockout depth profile.

prepare -> benchmark (includes controls) -> run -> report. Same CLI on MPS/CUDA;
different hardware requires a separate protocol/output directory. No SAE loading.
"""

import argparse
import hashlib
import logging
import platform
import resource
import subprocess
import time
import traceback
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from medgemma_grounding import medgemma_interventions as base
from medgemma_grounding.attention_knockout import conditions, knockout, token_groups
from medgemma_grounding.inference import load_model, load_image, prepare_inputs
from medgemma_grounding.medgemma_features import atomic_json, identity_hash, file_sha256, answer_score, answer_candidates, spatial_layout
from medgemma_grounding.medgemma_preflight import tree_hashes, weight_identity, attention_settings
from medgemma_grounding.sae_reconstruction import logit_metrics

OUTPUT = Path("results/attention_depth_v1")
SOURCE = Path("results/gemma_scope/targeted_discovery_v1")
ATTENTION = {"text_config": "eager", "vision_config": "sdpa"}
TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}
BENCHMARK_SECONDS_LIMIT = 3 * 3600


def write_record(path, row):
    row = {**row, "record_hash": identity_hash(row)}
    atomic_json(path, row)


def read_record(path, protocol):
    row = base.read_json(path)
    if (row["record_hash"] != identity_hash({k: v for k, v in row.items() if k != "record_hash"})
            or row["protocol_hash"] != protocol["protocol_hash"]
            or (row.get("case_id") is not None and row["case_id"] not in protocol["case_ids"])):
        raise ValueError(f"Invalid record identity/cohort/checksum: {path}")
    return row


def export_manifest(source, output):
    """Only discovery metadata; no held-out images or hidden-state caches opened."""
    e008 = base.read_json(source / "protocol.json")
    result = base.read_json(source / "report.json")
    if not result["complete"] or result["protocol_hash"] != e008["protocol_hash"]:
        raise ValueError("Completed E-008 required")
    e007_dir = Path(e008["source_directory"])
    e007 = base.read_json(e007_dir / "protocol.json")
    e004_dir = Path(e007["source_directory"])
    cohort = base.read_json(e004_dir / "cohort.json")
    reserved = {r["patient"] for r in cohort["cases"] if r["partition"] == "confirmation"}
    cases = []
    for case_id in e008["discovery_case_ids"]:
        old = base.read_json(e004_dir / "cases" / f"{case_id}.json")
        fresh = base.read_json(e007_dir / "cases" / f"{case_id}.json")
        if (old["partition"] != "discovery" or old["patient"] in reserved
                or file_sha256(e007_dir / "cases" / f"{case_id}.json") != e008["case_record_sha256"][case_id]):
            raise ValueError("Changed discovery provenance")
        parts = Path(old["image_path"]).parts
        offset = parts.index("CheXpert")
        relative = Path(*parts[offset:])
        if relative.parts[1] != "val" or file_sha256(Path(old["image_path"])) != old["image_sha256"]:
            raise ValueError("Expected unchanged validation discovery image")
        cases.append({k: old[k] for k in ("case_id", "patient", "label", "image_sha256", "image_size", "tokens", "positions", "spatial_layout", "annotation_status")}
                     | {"relative_image": str(relative), "historical_answer_score": fresh["answer_score"], "historical_outcome": fresh["outcome"]})
    if len(cases) != 142 or len({c["patient"] for c in cases}) != 140 or len(reserved) != 60:
        raise ValueError("Expected fixed 142/140 discovery with 60 reserved patients")
    if result["measurement_records"] != 7952 or result["distinct_corrected_patients"]:
        raise ValueError("E-008 context differs from expected completed evidence")
    image_candidates = [r for r in base.read_json(source / "candidates.json")["candidates"] if r["group"] == "image"]
    if len(image_candidates) != 3:
        raise ValueError("Expected three directly tested E-008 image candidates")
    manifest = {"cases": cases, "confirmation_patients_reserved": 60, "test_access": False,
                "source_protocol_hash": e008["protocol_hash"],
                "source_artifacts": {str(d): tree_hashes(d) for d in (source, e007_dir, e004_dir)},
                "model": {k: e007[k] for k in ("model_id", "model_revision", "processor_revision", "prompt", "scoring")}}
    atomic_json(output / "discovery_manifest.json", manifest)
    return manifest


def prepare(output, device, manifest_path=None):
    path = output / "protocol.json"
    if path.exists():
        p = base.read_json(path)
        validate(output, p)
        if p["device"] != device:
            raise ValueError("Different backend requires a separate output protocol")
        return p
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path:
        manifest = base.read_json(manifest_path)
        atomic_json(output / "discovery_manifest.json", manifest)
    else:
        manifest = export_manifest(SOURCE, output)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(manifest["model"]["model_id"], revision=manifest["model"]["model_revision"])
    text = config.text_config
    if text.num_hidden_layers != 34 or len(text.layer_types) != 34:
        raise ValueError("Expected 34-layer Gemma 3 decoder; resolve windows before execution")
    if not (output / "methods.md").exists():
        raise ValueError("Archive reviewed methods before protocol freeze")
    cases = manifest["cases"]
    if len(cases) != 142 or len({c["patient"] for c in cases}) != 140 or manifest["test_access"]:
        raise ValueError("Discovery manifest population changed")
    # Deterministic outcome-stratified engineering set, not a new research subset.
    smoke = [min((r for r in cases if r["historical_outcome"] == o),
                 key=lambda r: (base.hashed('42:' + r['patient']), r['case_id']))["case_id"] for o in base.QUOTAS]
    source_files = [Path(__file__), Path(__file__).with_name("attention_knockout.py"),
                    Path(__file__).with_name("inference.py"), Path(__file__).with_name("medgemma_features.py"),
                    Path(__file__).with_name("sae_reconstruction.py"), Path(__file__).with_name("medgemma_interventions.py"),
                    Path(__file__).with_name("attention_analysis.py"), Path(__file__).with_name("feature_interventions.py"),
                    Path(__file__).with_name("medgemma_preflight.py")]
    from transformers.models.gemma3 import modeling_gemma3
    from transformers import masking_utils
    p = {"experiment": "E-009", **manifest["model"], "device": device, "dtype": "torch.bfloat16",
         "attention": ATTENTION, "tolerance": TOLERANCE, "seed": 42, "batch_size": 1,
         "conditions": conditions(), "case_ids": [c["case_id"] for c in cases], "benchmark_ids": smoke,
         "expected_knockouts": 2556, "expected_baselines": 142, "expected_blank_links": 142,
         "blank": {"rgb": [128, 128, 128], "deduplication": "SHA256 of all processed tensors including dtype/shape"},
         "layer_types": list(text.layer_types), "sliding_window": text.sliding_window,
         "window_layer_types": {c["id"]: list(text.layer_types[c["start"]:c["end"] + 1]) for c in conditions()},
         "local_runtime_limit_seconds": BENCHMARK_SECONDS_LIMIT, "rss_limit_bytes": 26 * 1024**3,
         "independence": "full-depth B logits allclose across engineering images with identical non-image tensors/layout",
         "bootstrap": {"unit": "patient", "resamples": 1000, "seed": 42},
         "files_sha256": {"discovery_manifest.json": file_sha256(output / "discovery_manifest.json"),
                           "methods.md": file_sha256(output / "methods.md"),
                           **{str(f.relative_to(output)): file_sha256(f) for f in (output / "references").glob("*") if f.is_file()}},
         "source_sha256": {f.name: file_sha256(f) for f in source_files},
         "implementation_sha256": {"gemma3": file_sha256(Path(modeling_gemma3.__file__)), "masking_utils": file_sha256(Path(masking_utils.__file__))},
         "versions": {n: version(n) for n in ("torch", "transformers", "numpy", "pillow")},
         "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))}
    p["protocol_hash"] = identity_hash(p)
    atomic_json(path, p)
    return p


def validate(output, p):
    if p["protocol_hash"] != identity_hash({k: v for k, v in p.items() if k != "protocol_hash"}):
        raise ValueError("Changed protocol")
    for name, sha in p["files_sha256"].items():
        if file_sha256(output / name) != sha:
            raise ValueError(f"Changed frozen artifact {name}")
    for name, sha in p["source_sha256"].items():
        if file_sha256(Path(__file__).with_name(name)) != sha:
            raise ValueError(f"Changed runtime source {name}; new protocol required")
    if p["conditions"] != conditions() or p["tolerance"] != TOLERANCE or p["attention"] != ATTENTION:
        raise ValueError("Changed fixed settings")
    return base.read_json(output / "discovery_manifest.json")


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()


def tensor_hash(inputs):
    h = hashlib.sha256()
    for name, tensor in sorted(inputs.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(f"{name}:{t.dtype}:{list(t.shape)}".encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def image_inputs(bundle, p, case, data_root, blank=False):
    path = data_root / case["relative_image"]
    if file_sha256(path) != case["image_sha256"]:
        raise ValueError("Image changed")
    image = Image.new("RGB", tuple(reversed(case["image_size"])), (128, 128, 128)) if blank else load_image(path)
    inputs = prepare_inputs(bundle, image, p["prompt"])
    if (inputs["input_ids"][0].cpu().tolist() != [t["id"] for t in case["tokens"]]
            or answer_candidates(bundle.processor.tokenizer, inputs["input_ids"]) != p["scoring"]
            or spatial_layout(bundle, inputs, image.size) != case["spatial_layout"]):
        raise ValueError("Processor/scoring/layout differs from discovery manifest")
    groups = token_groups(inputs, bundle.model.config.image_token_index)
    if groups["image"] != case["positions"]["image"] or groups["final"] != [max(case["positions"]["all"])]:
        raise ValueError("Token groups changed")
    return inputs, groups


def timed_forward(bundle, inputs, groups=None, condition=None, mode="block", inspect=False, audit=False):
    synchronize(bundle.device)
    start = time.perf_counter()
    if condition:
        with knockout(bundle.model, groups, condition, mode=mode, inspect=inspect, audit_routes=audit) as masks:
            logits = base.forward(bundle, inputs)
    else:
        masks = {}
        logits = base.forward(bundle, inputs)
    synchronize(bundle.device)
    return logits, {"seconds": time.perf_counter() - start, "layers": masks,
                    "blocked_edges": sum(r["blocked_edges"] for r in masks.values())}


def measurement(case, p, baseline, changed, condition, timing, input_hash):
    original, answer = answer_score(baseline, p["scoring"]), answer_score(changed, p["scoring"])
    sign = 2 * case["label"] - 1
    oc = original["decision"] == ("Yes" if case["label"] else "No")
    cc = answer["decision"] == ("Yes" if case["label"] else "No")
    return {"protocol_hash": p["protocol_hash"], "case_id": case["case_id"], "patient": case["patient"],
            "label": case["label"], "outcome": ("T" if oc else "F") + ("P" if original["decision"] == "Yes" else "N"),
            **condition, "baseline": original, "changed": answer, "S": original["score"], "M": sign * original["score"],
            "changed_S": answer["score"], "changed_M": sign * answer["score"],
            "delta_S": answer["score"] - original["score"], "delta_M": sign * (answer["score"] - original["score"]),
            "original_correct": oc, "changed_correct": cc, "corrected": cc and not oc, "introduced": oc and not cc,
            "answer_format_valid": answer["argmax_token_id"] in p["scoring"]["ids"].values(),
            "historical_outcome": case["historical_outcome"], "historical_score_difference": original["score"] - case["historical_answer_score"]["score"],
            **logit_metrics(baseline, changed), **timing, "input_sha256": input_hash, "image_sha256": case["image_sha256"]}


def runtime(p):
    if p["device"] == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("A supported BF16 CUDA device is required; no dtype fallback")
    if p["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    for name, expected in p["versions"].items():
        if version(name) != expected:
            raise ValueError("Versions differ from frozen protocol")
    from transformers.models.gemma3 import modeling_gemma3
    from transformers import masking_utils
    for name, module in (("gemma3", modeling_gemma3), ("masking_utils", masking_utils)):
        if file_sha256(Path(module.__file__)) != p["implementation_sha256"][name]:
            raise ValueError("Installed mask/attention source changed")
    start = time.perf_counter()
    bundle = load_model(p["model_id"], device=p["device"], revision=p["model_revision"], attn_implementation=ATTENTION)
    bundle.model.requires_grad_(False)
    synchronize(bundle.device)
    if bundle.model.dtype != torch.bfloat16 or list(bundle.model.config.text_config.layer_types) != p["layer_types"]:
        raise ValueError("Architecture or precision changed")
    settings = attention_settings(bundle.model)
    if any(settings[k] != v for k, v in ATTENTION.items()):
        raise ValueError("Attention backend changed")
    if bundle.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hardware = torch.cuda.get_device_name() if bundle.device == "cuda" else platform.processor()
    return bundle, {"hardware": hardware, "platform": platform.platform(), "load_seconds": time.perf_counter() - start,
                    "attention": settings, "dtype": str(bundle.model.dtype), "device": bundle.device}


def engineering(bundle, p, manifest, output, data_root, load_info):
    full_b = next(c for c in p["conditions"] if c["route"] == "B" and c["end"] - c["start"] == 33)
    full_a = {**full_b, "route": "A", "id": "A_00_33"}
    records, timings, isolated = [], [], []
    for cid in p["benchmark_ids"]:
        case = next(c for c in manifest["cases"] if c["case_id"] == cid)
        inputs, groups = image_inputs(bundle, p, case, data_root)
        baseline, _ = timed_forward(bundle, inputs)  # warmup excluded from steady timing
        repeated, t = timed_forward(bundle, inputs)
        base.identity_control(baseline, repeated, p["scoring"])
        timings.append(t["seconds"])
        checks = {}
        for mode in ("pass", "placebo"):
            logits, detail = timed_forward(bundle, inputs, groups, full_b, mode=mode)
            checks[mode] = base.identity_control(baseline, logits, p["scoring"])
            if detail["blocked_edges"] != 0:
                raise ValueError("Identity/placebo blocked an allowed edge")
        for condition in (p["conditions"][0], full_a, full_b):
            logits, detail = timed_forward(bundle, inputs, groups, condition, inspect=True, audit=True)
            timings.append(detail["seconds"])
            row = measurement(case, p, baseline, logits, condition, detail, tensor_hash(inputs))
            row["token_groups"] = groups
            write_record(output / "engineering" / cid / f"{condition['id']}.json", row)
            if condition == full_b:
                isolated.append((logits, tensor_hash({k: v for k, v in inputs.items() if k != "pixel_values"}), groups))
        records.append({"case_id": cid, "controls": checks})
    reference, key, layout = isolated[0]
    for logits, other_key, other_layout in isolated[1:]:
        if other_key != key or other_layout != layout:
            raise ValueError("Engineering inputs have different non-image tensors/layout; resolve before comparing")
        base.identity_control(reference, logits, p["scoring"], "full_depth_B_image_independence")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    memory = {"peak_process_rss_bytes": rss}
    if bundle.device == "mps":
        memory.update(mps_current_allocated=torch.mps.current_allocated_memory(), mps_driver_allocated=torch.mps.driver_allocated_memory(),
                      mps_recommended_max=torch.mps.recommended_max_memory())
    else:
        memory.update(cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
    # 18 knockouts + baseline + one identity per case, <=142 distinct blanks.
    upper_forwards = 142 * 21
    forecast = float(np.quantile(timings, .9)) * upper_forwards * 1.2
    comfortable = rss < p["rss_limit_bytes"]
    selected = bundle.device != "mps" or (forecast <= p["local_runtime_limit_seconds"] and comfortable)
    result = {"protocol_hash": p["protocol_hash"], "controls_passed": True, "numerical_checks": records,
              "full_depth_B_image_independence": True, "timed_forwards": timings,
              "median_seconds": float(np.median(timings)), "p90_seconds": float(np.quantile(timings, .9)),
              "forecast_forwards_upper": upper_forwards, "forecast_seconds_with_20pct_allowance": forecast,
              "memory": memory, "runtime": load_info, "comfortable_memory": comfortable,
              "backend_approved": selected, "backend_decision": "local MPS" if selected and bundle.device == "mps" else "CUDA" if selected else "prefer one BF16 CUDA GPU; remote access/paths required",
              "storage_forecast_bytes": 142 * 1024 * 1024 + 50 * 1024 * 1024,
              "storage_basis": "at most 142 one-vocabulary FP32 baseline tensors, small JSON shards/figures; no hidden states/attention tensors"}
    write_record(output / "benchmark.json", result)
    return result


def execute_case(bundle, p, case, output, data_root):
    cid = case["case_id"]
    directory = output / "cases" / cid
    inputs, groups = image_inputs(bundle, p, case, data_root)
    digest = tensor_hash(inputs)
    baseline, timing = timed_forward(bundle, inputs)
    baseline_path = directory / "baseline.pt"
    header_path = directory / "baseline.json"
    if header_path.exists():
        header = read_record(header_path, p)
        if header["input_sha256"] != digest or file_sha256(baseline_path) != header["tensor_sha256"]:
            raise ValueError("Baseline cache drift")
        saved = torch.load(baseline_path, weights_only=True, map_location="cpu")
        base.identity_control(saved, baseline, p["scoring"])
    else:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(baseline, baseline_path.with_suffix(".pt.tmp"))
        baseline_path.with_suffix(".pt.tmp").replace(baseline_path)
        write_record(header_path, {"protocol_hash": p["protocol_hash"], "case_id": cid, "input_sha256": digest,
            "tensor_sha256": file_sha256(baseline_path), "token_groups": groups, "seconds": timing["seconds"]})
    unmodified, _ = timed_forward(bundle, inputs, groups, p["conditions"][-1], mode="pass")
    base.identity_control(baseline, unmodified, p["scoring"])
    for condition in p["conditions"]:
        path = directory / f"{condition['id']}.json"
        if path.exists():
            row = read_record(path, p)
            if row["input_sha256"] != digest or row["id"] != condition["id"]:
                raise ValueError("Resumed treatment input/condition mismatch")
            continue
        changed, detail = timed_forward(bundle, inputs, groups, condition)
        write_record(path, measurement(case, p, baseline, changed, condition, detail, digest) | {"token_groups": groups})
    blank_inputs, blank_groups = image_inputs(bundle, p, case, data_root, blank=True)
    blank_hash = tensor_hash(blank_inputs)
    blank_path = output / "blanks" / f"{blank_hash}.pt"
    if blank_path.exists():
        meta = read_record(blank_path.with_suffix(".json"), p)
        if file_sha256(blank_path) != meta["tensor_sha256"]:
            raise ValueError("Blank tensor changed")
        blank_logits = torch.load(blank_path, map_location="cpu", weights_only=True)
    else:
        blank_logits, blank_timing = timed_forward(bundle, blank_inputs)
        blank_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blank_logits, blank_path.with_suffix(".pt.tmp"))
        blank_path.with_suffix(".pt.tmp").replace(blank_path)
        write_record(blank_path.with_suffix(".json"), {"protocol_hash": p["protocol_hash"],
            "processed_input_sha256": blank_hash, "tensor_sha256": file_sha256(blank_path), **blank_timing})
    blank_score = answer_score(blank_logits, p["scoring"])
    original = answer_score(baseline, p["scoring"])
    write_record(directory / "blank.json", {"protocol_hash": p["protocol_hash"], "case_id": cid,
        "patient": case["patient"], "baseline_S": original["score"], "blank_S": blank_score["score"],
        "delta_S": blank_score["score"] - original["score"], "answer_agreement": blank_score["decision"] == original["decision"],
        "blank_answer": blank_score, "processed_input_sha256": blank_hash, "token_groups": blank_groups,
        "interpretation": "Altered gray image diagnostic, not text-only or a clinically negative X-ray"})


def complete_case(output, p, cid):
    paths = [output / "cases" / cid / f"{c['id']}.json" for c in p["conditions"]]
    paths += [output / "cases" / cid / "blank.json", output / "cases" / cid / "baseline.json"]
    complete = True
    for path in paths:
        if path.exists():
            read_record(path, p)
        else:
            complete = False
    if complete:
        header = read_record(paths[-1], p)
        if file_sha256(paths[-1].with_suffix(".pt")) != header["tensor_sha256"]:
            raise ValueError("Baseline tensor checksum changed")
    return complete


def run(stage, output, device, data_root, manifest_path=None):
    p = prepare(output, device, manifest_path)
    manifest = validate(output, p)
    if stage == "prepare":
        return p
    if stage == "report":
        from medgemma_grounding.attention_analysis import report
        return report(output, p, manifest)
    if list(output.glob("failure_*.json")):
        raise ValueError("Technical failure recorded; diagnose without relaxing checks")
    if stage == "run" and all(complete_case(output, p, c) for c in p["case_ids"]):
        return run("report", output, device, data_root)
    bundle = None
    try:
        bundle, info = runtime(p)
        weights = weight_identity(bundle.model)
        benchmark = read_record(output / "benchmark.json", p) if (output / "benchmark.json").exists() else engineering(bundle, p, manifest, output, data_root, info)
        if stage == "run":
            if not benchmark["backend_approved"]:
                raise RuntimeError("Compute gate: local forecast impractical; prepare single-GPU launch and request missing remote access")
            start = time.perf_counter()
            for i, case in enumerate(manifest["cases"], 1):
                if complete_case(output, p, case["case_id"]):
                    continue
                logging.info("Case %d/142 %s", i, case["case_id"])
                execute_case(bundle, p, case, output, data_root)
                atomic_json(output / "progress.json", {"completed_cases": i, "elapsed_seconds": time.perf_counter() - start,
                                                       "last_case": case["case_id"]})
        if weights != weight_identity(bundle.model) or any(v.requires_grad or v.grad is not None for v in bundle.model.parameters()):
            raise ValueError("Parameter state changed")
        write_record(output / "runtime_checks.json", {"protocol_hash": p["protocol_hash"], "frozen_unchanged_no_gradients": True, **info})
        return benchmark if stage == "benchmark" else run("report", output, device, data_root)
    except Exception as error:
        write_record(output / f"failure_{stage}.json", {"protocol_hash": p["protocol_hash"], "error": str(error),
            "diagnostics": getattr(error, "diagnostics", None), "traceback": traceback.format_exc(), "settings_changed": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["prepare", "benchmark", "run", "report"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["mps", "cuda"], default="mps")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.stage, args.output_dir, args.device, args.data_root, args.manifest)
    logging.info("Stage %s finished: %s", args.stage, {k: result[k] for k in ("complete", "backend_decision") if k in result})


if __name__ == "__main__":
    main()
