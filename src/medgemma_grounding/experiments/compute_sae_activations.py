"""Save layer-14 SAE features and answers for the fixed 20-case MedGemma pilot."""


from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import argparse
import json
import logging
import subprocess

import torch

from medgemma_grounding.data.image_positions import spatial_layout
from medgemma_grounding.model import answer_candidates, answer_score
from medgemma_grounding.sae.cache import load_discovery_case, save_cache
from medgemma_grounding.sae.config import LAYER, OUTPUT, PILOT
from medgemma_grounding.utils.io import atomic_json, file_sha256, identity_hash
from medgemma_grounding.utils.provenance import source_hashes


def collect_case(bundle, sae, case, prompt):
    """Capture the intact model's prefill features; generate after removing hooks."""
    from medgemma_grounding.model import generate
    from medgemma_grounding.model import load_image
    from medgemma_grounding.model import prepare_inputs
    from medgemma_grounding.sae.reconstruction import capture_residual
    from medgemma_grounding.sae.reconstruction import encode_decode
    from medgemma_grounding.sae.reconstruction import token_groups

    image = load_image(case["image_path"])
    inputs = prepare_inputs(bundle, image, prompt)
    hook = f"model.language_model.layers.{LAYER}"
    with torch.inference_mode():
        baseline = bundle.model(**inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
    hidden, passive_logits = capture_residual(bundle.model, inputs, hook)
    torch.testing.assert_close(passive_logits, baseline, rtol=1e-5, atol=1e-5)
    groups = token_groups(
        inputs["input_ids"], inputs["attention_mask"], bundle.model.config.image_token_index,
        bundle.processor.tokenizer.all_special_ids,
    )
    if hidden.shape != (*inputs["input_ids"].shape, sae.cfg.d_in) or not all(m.any() for m in groups.values()):
        raise ValueError("Activation, token, or SAE dimensions differ from the pilot.")
    _, features = encode_decode(sae, hidden[0])
    if not torch.isfinite(features).all() or (features < 0).any():
        raise ValueError("Expected finite, nonnegative JumpReLU features.")
    # Check the encoding independently at both ends of the sequence, including text.
    with torch.inference_mode():
        for start in (0, (len(features) - 1) // 64 * 64):
            torch.testing.assert_close(features[start:start + 64], sae.encode(hidden[0, start:start + 64]))
    tokenizer = bundle.processor.tokenizer
    ids = inputs["input_ids"][0].cpu().tolist()
    pieces = tokenizer.convert_ids_to_tokens(ids)
    tokens = [
        {
            "index": i, "id": token_id, "piece": piece,
            "text": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "group": "image" if groups["image"][i] else "text" if groups["text"][i] else "special",
        }
        for i, (token_id, piece) in enumerate(zip(ids, pieces, strict=True))
    ]
    record = {
        **case,
        "case_id": "_".join(Path(case["image_path"]).with_suffix("").parts[-3:]),
        "answer": generate(bundle, image, prompt, max_new_tokens=32),
        "tokens": tokens,
        "feature_shape": list(features.shape),
        "spatial_layout": spatial_layout(bundle, inputs, image.size),
        "passive_hook_max_logit_difference": (baseline - passive_logits).abs().max().item(),
        "observed_l0": {
            name: features[mask].count_nonzero().item() / int(mask.sum())
            for name, mask in groups.items()
        },
    }
    return record, features.to_sparse().coalesce()


def run(pilot_path=PILOT, output=OUTPUT, *, overwrite=False):
    from medgemma_grounding.model import load_model
    from medgemma_grounding.experiments.run_sae_layers import read_pilot
    from medgemma_grounding.sae.reconstruction import checkpoint_spec
    from medgemma_grounding.sae.reconstruction import load_scope_sae

    output = Path(output)
    if (output / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"Cache already exists at {output}; use --overwrite to replace it.")
    pilot = read_pilot(Path(pilot_path))
    sae, config, _, sae_revision = load_scope_sae(LAYER, all_layers=True)
    bundle = load_model(pilot["model_id"], revision=pilot["model_revision"])
    cases, activations = [], {}
    for index, case in enumerate(pilot["cases"], 1):
        logging.info("Extracting layer-%d features %d/%d: %s", LAYER, index, len(pilot["cases"]), case["image_path"])
        record, features = collect_case(bundle, sae, case, pilot["prompt"])
        cases.append(record)
        activations[record["case_id"]] = features
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    release, sae_id, _, hook = checkpoint_spec(LAYER, all_layers=True)
    manifest = {
        "schema_version": 1, "project_root": str(Path.cwd().resolve()),
        "model_id": pilot["model_id"], "model_revision": pilot["model_revision"],
        "processor_revision": pilot["model_revision"], "prompt": pilot["prompt"],
        "sae_release": release, "sae_id": sae_id, "sae_revision": sae_revision,
        "sae_width": sae.cfg.d_sae, "target_l0": config["l0"], "hook": hook,
        "layer": LAYER, "device": bundle.device, "model_dtype": str(bundle.model.dtype),
        "sae_device_dtype": "cpu/float32", "generation": {"do_sample": False, "max_new_tokens": 32},
        "versions": {name: version(name) for name in ("torch", "transformers", "sae-lens")},
        "code_revision": revision.stdout.strip() or None, "code_dirty": bool(dirty.stdout.strip()),
        "source_sha256": source_hashes(),
        "cases": cases,
    }
    save_cache(output, manifest, activations)
    return manifest


def collect_discovery_case(bundle, sae, case, prompt, candidates, generation_config):
    """One discovery image: intact answer and sequential group fidelity checks."""
    import re
    import time
    import resource
    import sys
    from medgemma_grounding.model import load_image
    from medgemma_grounding.model import prepare_inputs
    from medgemma_grounding.sae.reconstruction import capture_residual
    from medgemma_grounding.sae.reconstruction import encode_decode
    from medgemma_grounding.sae.reconstruction import position_groups
    from medgemma_grounding.sae.reconstruction import masked_replacement
    from medgemma_grounding.sae.reconstruction import replace_residual
    from medgemma_grounding.sae.reconstruction import logit_metrics
    from medgemma_grounding.sae.reconstruction import prompt_positions
    start = time.perf_counter()
    image = load_image(case["image_path"])
    inputs = prepare_inputs(bundle, image, prompt)
    verified = answer_candidates(bundle.processor.tokenizer, inputs["input_ids"])
    if verified != candidates:
        raise ValueError("Chat scoring boundary changed between cases")
    masks = position_groups(inputs["input_ids"], inputs["attention_mask"],
                            bundle.model.config.image_token_index, bundle.processor.tokenizer.all_special_ids)
    if not masks["image"].any() or not masks["non_image"].any():
        raise ValueError("Expected both image and non-image positions")
    masks["ordinary_text"] = prompt_positions(
        inputs["input_ids"], bundle.processor.tokenizer.encode(prompt, add_special_tokens=False))
    if (masks["ordinary_text"] & ~masks["non_image"]).any():
        raise ValueError("User-prompt span overlaps an image or padding position")
    hook = f"model.language_model.layers.{LAYER}"
    with torch.inference_mode():
        baseline = bundle.model(**inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
    hidden, passive = capture_residual(bundle.model, inputs, hook)
    torch.testing.assert_close(passive, baseline, rtol=1e-5, atol=1e-5)
    unchanged = replace_residual(bundle.model, inputs, hook, hidden)
    torch.testing.assert_close(unchanged, baseline, rtol=1e-5, atol=1e-5)
    reconstructed, features = encode_decode(sae, hidden[0])
    if (features.shape != (hidden.shape[1], sae.cfg.d_sae)
            or not all(torch.isfinite(x).all() for x in (hidden, reconstructed, features))
            or (features < 0).any()):
        raise ValueError("Invalid or misaligned SAE encoding")
    original_score = answer_score(baseline, candidates)
    fidelity, sufficient = {}, {}
    for name, mask in masks.items():
        x, x_hat = hidden[0, mask].double(), reconstructed[mask].double()
        sse = (x - x_hat).square().sum().item()
        energy = x.square().sum().item()
        count = len(x)
        fidelity[name] = {"tokens": count, "sse": sse, "energy": energy,
                          "relative_squared_error": sse / energy if energy else None,
                          "active_features": features[mask].count_nonzero().item(),
                          "observed_l0": features[mask].count_nonzero().item() / count if count else None}
        sufficient[name] = {"sum_x": x.sum(0), "sum_x2": x.square().sum(0)}
    for name in ("image", "non_image", "final", "all"):
        replacement = masked_replacement(hidden, reconstructed.unsqueeze(0), masks[name])
        changed = replace_residual(bundle.model, inputs, hook, replacement)
        scored = answer_score(changed, candidates)
        delta = scored["score"] - original_score["score"]
        fidelity[name]["behavior"] = {
            **logit_metrics(baseline, changed), "answer": scored,
            "score_change": delta, "absolute_score_change": abs(delta),
            "decision_flip": scored["decision"] != original_score["decision"],
            "original_margin": abs(original_score["score"]),
        }
    with torch.inference_mode():
        generated = bundle.model.generate(**inputs, generation_config=generation_config)
    generated_ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
    if not generated_ids or generated_ids[0] != original_score["argmax_token_id"]:
        raise ValueError("Greedy first token disagrees with raw argmax: revise scoring protocol")
    answer = bundle.processor.decode(generated_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False).strip()
    match = re.fullmatch(r"(Yes|No)[.!]?", answer, flags=re.IGNORECASE)
    parsed = match.group(1).capitalize() if match else "invalid"
    ids = inputs["input_ids"][0].cpu().tolist()
    tokenizer = bundle.processor.tokenizer
    tokens = [{"index": i, "id": token_id, "piece": tokenizer.convert_ids_to_tokens(token_id),
               "text": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
               "group": "image" if masks["image"][i] else "text" if masks["text"][i] else "special"}
              for i, token_id in enumerate(ids)]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {**case, "answer": answer, "parsed_answer": parsed, "generated_token_ids": generated_ids,
              "answer_score": original_score, "parsed_score_disagreement": parsed != original_score["decision"],
              "tokens": tokens, "feature_shape": list(features.shape),
              "positions": {name: mask.nonzero().flatten().tolist() for name, mask in masks.items()},
              "spatial_layout": spatial_layout(bundle, inputs, image.size), "fidelity": fidelity,
              "passive_hook_max_logit_difference": (baseline - passive).abs().max().item(),
              "unchanged_hook_max_logit_difference": (baseline - unchanged).abs().max().item(),
              "controls": {"passive": logit_metrics(baseline, passive),
                           "unchanged": logit_metrics(baseline, unchanged)},
              "runtime_seconds": time.perf_counter() - start,
              "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024}
    if bundle.device == "mps":
        record["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
        record["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
    for name, logits in (("passive", passive), ("unchanged", unchanged)):
        scored = answer_score(logits, candidates)
        delta = scored["score"] - original_score["score"]
        record["controls"][name].update(answer=scored, score_change=delta,
                                        absolute_score_change=abs(delta),
                                        decision_flip=scored["decision"] != original_score["decision"])
    payload = {"features": features.to_sparse().coalesce(), "sufficient_statistics": sufficient,
               "summaries": {"image_mean": features[masks["image"]].mean(0),
                             "image_max": features[masks["image"]].amax(0),
                             "image_active_fraction": (features[masks["image"]] > 0).float().mean(0),
                             "final": features[masks["final"]][0]}}
    return record, payload


def run_discovery(cohort_path, output, pilot_path=PILOT, *, limit=None):
    """Resume the fixed discovery partition only. A limit retains smoke outputs."""
    import copy
    from medgemma_grounding.model import load_model
    from medgemma_grounding.model import load_image
    from medgemma_grounding.model import prepare_inputs
    from medgemma_grounding.sae.reconstruction import SAE_REVISION
    from medgemma_grounding.sae.reconstruction import checkpoint_spec
    from medgemma_grounding.sae.reconstruction import load_scope_sae

    output = Path(output)
    if output.resolve() == OUTPUT.resolve() or (output / "manifest.json").exists():
        raise ValueError("Never write schema-2 results over a pilot cache")
    cohort = json.loads(Path(cohort_path).read_text())
    cases = [c for c in cohort["cases"] if c["partition"] == "discovery"]
    if not cases or any('/test/' in c['image_path'] for c in cohort['cases']):
        raise ValueError("Expected validation discovery cohort")
    pilot = json.loads(Path(pilot_path).read_text())
    release, sae_id, _, hook = checkpoint_spec(LAYER, all_layers=True)
    from medgemma_grounding.data.chexlocalize import validate_partition
    cases = validate_partition(cohort, {c["patient"] for c in pilot["cases"]})
    for path, digest in cohort["metadata_sha256"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError("Cohort metadata changed")
    sources = source_hashes()
    identity = {"schema_version": 2, "model_id": pilot["model_id"], "model_revision": pilot["model_revision"],
                "processor_revision": pilot["model_revision"], "prompt": pilot["prompt"],
                "sae_revision": SAE_REVISION, "sae_release": release, "sae_id": sae_id,
                "sae_width": 16384, "layer": LAYER, "hook": hook,
                "cohort_sha256": file_sha256(Path(cohort_path)), "discovery_cases": cases,
                "source_sha256": sources,
                "analysis": {"patient_aggregation": "image mean; label any-positive; flag mixed",
                             "representations": ["image_mean", "final"], "min_active_patients": 5,
                             "ranking": ["label_auc_distance", "answer_absolute_spearman"],
                             "selection": "top two new IDs per track in fixed order; ID tie break",
                             "confirmation_enabled": False},
                "position_definitions": {
                    "image": "valid processor image-token IDs",
                    "ordinary_text": "unique exact token span of fixed user prompt, excluding template",
                    "text": "legacy pilot: valid non-image minus tokenizer.all_special_ids; may include template",
                    "non_image": "all valid positions except image, including chat formatting",
                    "final": "last valid input position (not necessarily ordinary text)",
                    "all": "all valid positions; padding excluded"},
                "versions": {name: version(name) for name in ("torch", "transformers", "sae-lens", "scipy", "pycocotools")}}
    existing = json.loads((output / "protocol.json").read_text()) if (output / "protocol.json").exists() else None
    if existing:
        if any(existing.get(k) != v for k, v in identity.items()):
            raise ValueError("Resume protocol differs: do not mix revisions, sources or cohort")
        # Validate all completed cases before loading any model.
        for case in cases:
            if (output / "cases" / f"{case['case_id']}.json").exists():
                load_discovery_case(output, case["case_id"], existing)
        if file_sha256(output / "cohort.json") != file_sha256(Path(cohort_path)):
            raise ValueError("Saved cohort differs from input cohort")
    remaining = [c for c in cases if not (output / "cases" / f"{c['case_id']}.json").exists()]
    if not remaining:
        return existing
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    bundle = load_model(pilot["model_id"], revision=pilot["model_revision"])
    sae, config, _, _ = load_scope_sae(LAYER, all_layers=True)
    inputs = prepare_inputs(bundle, load_image(remaining[0]["image_path"]), pilot["prompt"])
    candidates = answer_candidates(bundle.processor.tokenizer, inputs["input_ids"])
    generation = copy.deepcopy(bundle.model.generation_config)
    for key, value in {"do_sample": False, "num_beams": 1, "max_new_tokens": 32,
                       "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "min_length": 0,
                       "min_new_tokens": 0, "forced_bos_token_id": None, "forced_eos_token_id": None,
                       "suppress_tokens": None, "begin_suppress_tokens": None, "bad_words_ids": None,
                       "sequence_bias": None}.items():
        setattr(generation, key, value)
    protocol = {**identity, "scoring": candidates, "generation": generation.to_dict(),
                "device": bundle.device, "model_dtype": str(bundle.model.dtype),
                "sae_device_dtype": "cpu/float32", "target_l0": config["l0"],
                "code_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "code_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout)}
    if existing:
        for key in ("scoring", "generation", "device", "model_dtype", "target_l0"):
            if existing[key] != protocol[key]:
                raise ValueError(f"Resume runtime settings differ: {key}")
        protocol = existing
    else:
        protocol["protocol_hash"] = identity_hash(protocol)
        atomic_json(output / "protocol.json", protocol)
        atomic_json(output / "cohort.json", cohort)
    (output / "cases").mkdir(exist_ok=True)
    for index, case in enumerate(remaining[:limit], 1):
        if file_sha256(Path(case["image_path"])) != case["image_sha256"]:
            raise ValueError("Input image changed before inference")
        logging.info("Discovery %d/%d %s", index, len(remaining), case["case_id"])
        record, payload = collect_discovery_case(bundle, sae, case, pilot["prompt"], candidates, generation)
        tensor_path = output / "cases" / f"{case['case_id']}.pt"
        temporary = tensor_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(tensor_path)
        record.update(protocol_hash=protocol["protocol_hash"], tensor_sha256=file_sha256(tensor_path))
        atomic_json(tensor_path.with_suffix(".json"), record)
        logging.info("Saved %.1fs; peak RSS %.2f GB; estimated remaining %.1f minutes",
                     record["runtime_seconds"], record["peak_rss_bytes"] / 1e9,
                     max(0, len(remaining) - index) * record["runtime_seconds"] / 60)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-json", type=Path, default=PILOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing extraction")
    parser.add_argument("--cohort-manifest", type=Path, help="Schema-2 validation cohort; discovery only")
    parser.add_argument("--limit", type=int, help="Save at most this many new discovery cases for a smoke check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.cohort_manifest:
        if args.overwrite:
            parser.error("Discovery is resume-only; use a new directory for a new protocol")
        run_discovery(args.cohort_manifest, args.output_dir, args.pilot_json, limit=args.limit)
        return
    if args.limit is not None:
        parser.error("--limit is only available with --cohort-manifest")
    manifest = run(args.pilot_json, args.output_dir, overwrite=args.overwrite)
    print(f"Saved layer-{LAYER} features and answers for {len(manifest['cases'])} cases: {args.output_dir}")


if __name__ == "__main__":
    main()
