"""Prepare, inspect, and run the section 2.3 discovery-case occlusion experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from medgemma_grounding.baseline_pilot import check_candidate_tokens, select_case
from medgemma_grounding.grounding_baseline import json_text
from medgemma_grounding.grounding_protocol import read_part1_contract, sha256
from medgemma_grounding import masking


def configure_cache(cache: Path) -> None:
    os.environ["HF_HOME"] = str(cache.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))


def tensor_record(tensor, torch) -> dict:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(raw).hexdigest()}


def model_view(inputs, processor) -> Image.Image:
    pixels = inputs["pixel_values"].detach().float().cpu().numpy()
    if pixels.shape[0:2] != (1, 3):
        raise ValueError("Pilot supports one resized RGB image, not multiple image crops")
    pixels = pixels[0].transpose(1, 2, 0)
    ip = processor.image_processor
    pixels = pixels * np.array(ip.image_std) + np.array(ip.image_mean)
    return Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8))


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
        if not np.isin(array, [0, 255]).all():
            raise ValueError("Saved mask is no longer binary")
        return array == 255


def prepare(args) -> dict:
    if args.output.exists():
        raise ValueError("Preparation output exists; use a new directory")
    config = json.loads(args.config.read_text())
    if config["fills"] != ["mean", "blur"]:
        raise ValueError("This pilot requires both mean and blur fills")
    protocol, case = select_case(args.protocol, args.audit, config["case_id"])
    prior = json.loads((args.baseline / "result.json").read_text())
    if (prior["case_id"] != case["case_id"] or prior["status"] != "single_case_baseline_agrees"
            or prior["protocol_sha256"] != sha256(args.protocol / "protocol.json")
            or prior["part1_contract"] != protocol["part1"]):
        raise ValueError("Need a successful section 2.2 baseline for this discovery case and protocol")
    if not case["localization_eligible"]:
        raise ValueError("Region pilot requires a positive case with a radiologist annotation")
    configure_cache(args.hf_home)
    from medgemma_grounding import inference
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(prior["runtime"]["model_id"],
                                              revision=prior["runtime"]["processor_revision"],
                                              token=os.environ.get("HF_TOKEN"), local_files_only=True)
    dtype = getattr(torch, prior["runtime"]["dtype"].removeprefix("torch."))
    bundle = SimpleNamespace(processor=processor, model=SimpleNamespace(dtype=dtype), device="cpu", torch=torch)
    image = inference.load_image(Path(protocol["data_root"]) / case["image_path"])
    pixels = np.asarray(image)
    inputs = inference.prepare_inputs(bundle, image, protocol["part1"]["prompt"])
    stamps = {key: tensor_record(value, torch) for key, value in inputs.items()}
    if stamps != prior["prepared_tensors"]:
        raise ValueError("Processor-only tensors differ from the verified section 2.2 inputs")
    check_candidate_tokens(bundle, image, protocol["part1"]["prompt"], inputs)
    view = model_view(inputs, processor)
    model_size = view.size
    target_path = Path(protocol["data_root"]) / "CheXlocalize/gt_segmentations_val.json"
    if protocol["metadata_sha256"].get("CheXlocalize/gt_segmentations_val.json") != sha256(target_path):
        raise ValueError("Official segmentation file is not the one frozen in the protocol")
    target = masking.load_effusion_mask(target_path, case["case_id"], pixels.shape[:2])
    controls = masking.control_masks(target, config["control_offsets_normalized_xy"], config["chest_box_normalized_xyxy"])
    masks = {"target": target, **{f"control_{i+1}": c["mask"] for i, c in enumerate(controls)}}
    replacements, fill_settings = masking.replacement_images(image, model_size, config["blur_sigma_model_pixels"])
    patches = masking.grid_patches(target.shape, config["grid"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".occlusion-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        image.save(stage / "original.png")
        view.save(stage / "model_original.png")
        regions = []
        for name, mask in masks.items():
            save_mask(stage / f"mask_{name}.png", mask)
            model_mask = np.asarray(Image.fromarray(mask).resize(model_size, Image.Resampling.NEAREST))
            region = {"name": name, "mask_pixels_native": int(mask.sum()),
                      "mask_pixels_model_nearest": int(model_mask.sum()),
                      "target_overlap_pixels": int((mask & target).sum())}
            if name != "target":
                control = controls[int(name[-1]) - 1]
                region["translation_native_xy"] = [control["dx"], control["dy"]]
            regions.append(region)
            for fill, replacement in replacements.items():
                changed = masking.apply_mask(pixels, mask, replacement)
                if not np.array_equal(changed[~mask], pixels[~mask]):
                    raise ValueError("Pixels outside mask changed")
                variant = Image.fromarray(changed)
                variant.save(stage / f"{fill}_{name}.png")
                prepared = inference.prepare_inputs(bundle, variant, protocol["part1"]["prompt"])
                if tensor_record(prepared["input_ids"], torch) != stamps["input_ids"]:
                    raise ValueError("Occlusion unexpectedly changed the prompt/image token sequence")
                model_view(prepared, processor).save(stage / f"model_{fill}_{name}.png")
        from medgemma_grounding.occlusion_figures import preparation_figure
        preparation_figure(stage, config, masks, model_size)
        manifest = {
            "section": "2.3", "case": case, "config": config, "fill_settings": fill_settings,
            "model_size_width_height": list(model_size), "original_size_width_height": list(image.size),
            "part1_contract": protocol["part1"], "runtime": prior["runtime"],
            "baseline_yes_score": prior["yes_score"], "baseline_tensors": stamps,
            "protocol_path": str(args.protocol.resolve()), "baseline_path": str(args.baseline.resolve()),
            "protocol_sha256": sha256(args.protocol / "protocol.json"),
            "baseline_result_sha256": sha256(args.baseline / "result.json"),
            "segmentation_sha256": sha256(target_path), "case_config_sha256": sha256(args.config),
            "source_sha256": {name: sha256(Path(__file__).with_name(name))
                              for name in ("masking.py", "occlusion_pilot.py", "occlusion_figures.py")},
            "regions": regions, "patches": patches,
            "control_pair_overlap_pixels": {f"{a+1}_{b+1}": int((controls[a]["mask"] & controls[b]["mask"]).sum())
                                             for a in range(3) for b in range(a+1, 3)},
            "spatial_policy": "Apply masks at original resolution, then unchanged Part 1 preprocessing. Nearest-neighbor masks are for overlays only; image interpolation can affect boundary pixels.",
            "control_limitations": "Same area/shape/fill, different anatomy; controls may overlap each other and other findings. Chest window is not a lung segmentation. Visual QA is required before scoring.",
            "scope": "Single discovery-case pilot. No confirmation or test inference; no population-level grounding conclusion.",
            "artifacts_sha256": {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()},
        }
        (stage / "manifest.json").write_text(json_text(manifest))
        stage.rename(args.output)
    print(f"Prepared region images and overlays: {args.output / 'mask_review.png'}", flush=True)
    return manifest


def verify_prepared(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["part1_contract"] != read_part1_contract():
        raise ValueError("Inherited Part 1 source/settings changed")
    for name, expected in manifest["source_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError("Masking/runner source changed; prepare a new experiment")
    for name, expected in manifest["artifacts_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    if sha256(Path(manifest["protocol_path"]) / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed after preparation")
    if sha256(Path(manifest["baseline_path"]) / "result.json") != manifest["baseline_result_sha256"]:
        raise ValueError("Section 2.2 baseline changed after preparation")
    if manifest["case"]["partition"] != "discovery":
        raise ValueError("This command only runs discovery pilots")
    return manifest


def tasks_for(manifest: dict) -> list[dict]:
    tasks = [{"condition": "original", "kind": "baseline", "fill": "none"},
             {"condition": "empty_mask", "kind": "sanity", "fill": "none"}]
    for fill in manifest["config"]["fills"]:
        tasks += [{"condition": r["name"], "kind": "region", "fill": fill} for r in manifest["regions"]]
    for fill in manifest["config"]["fills"]:
        tasks += [dict(p, condition=p["name"], kind="patch", fill=fill) for p in manifest["patches"]]
    return tasks + [{"condition": "original_repeat", "kind": "sanity", "fill": "none"}]


def load_completed(path: Path, tasks: list[dict]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if len(rows) > len(tasks):
        raise ValueError("Results contain more rows than planned")
    for task, row in zip(tasks, rows):
        if any(row.get(key) != value for key, value in task.items()):
            raise ValueError("Saved results do not match planned task order")
        score = row.get("yes_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid saved Yes score")
    return rows


def check_sanity(rows: list[dict], historical: float) -> None:
    if not rows:
        return
    for row in rows:
        if row["kind"] not in {"baseline", "sanity"}:
            continue
        expected = historical if row["condition"] == "original" else rows[0]["yes_score"]
        if row["changed_model_pixels"] != 0 or not math.isclose(row["yes_score"], expected, abs_tol=1e-5, rel_tol=1e-4):
            raise ValueError("Baseline/no-op/repeat check failed; results retained for diagnosis")


def run(args) -> dict:
    manifest = verify_prepared(args.prepared)
    if not args.review_note.strip():
        raise ValueError("Record your visual inspection of mask_review.png before scoring")
    configure_cache(args.hf_home)
    from medgemma_grounding import inference, classification
    import torch

    runtime = manifest["runtime"]
    for name, version in runtime["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Runtime package changed since section 2.2: {name}")
    if runtime["device"] == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; run with access to the Mac GPU")
    tasks = tasks_for(manifest)
    run_settings = {"prepared_manifest_sha256": sha256(args.prepared / "manifest.json"),
                    "visual_review_note": args.review_note, "baseline_atol": 1e-5, "baseline_rtol": 1e-4}
    if args.output.exists():
        if not args.resume or json.loads((args.output / "run.json").read_text())["settings"] != run_settings:
            raise ValueError("Use a new output, or --resume with identical preparation and review note")
    else:
        args.output.mkdir(parents=True)
        (args.output / "run.json").write_text(json_text({"settings": run_settings,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {**{k: v for k, v in runtime.items() if not k.startswith("mps_")},
                        "platform": platform.platform(), "python": platform.python_version()},
            "pycocotools_version": importlib.metadata.version("pycocotools"),
            "task_count": len(tasks)}))
    result_path = args.output / "scores.jsonl"
    rows = load_completed(result_path, tasks)
    check_sanity(rows, manifest["baseline_yes_score"])
    if len(rows) < len(tasks):
        torch.manual_seed(0)
        print(f"Loading pinned MedGemma on {runtime['device']}; {len(rows)}/{len(tasks)} conditions saved", flush=True)
        bundle = inference.load_model(runtime["model_id"], device=runtime["device"], revision=runtime["model_revision"])
        if (str(bundle.model.dtype) != runtime["dtype"]
                or getattr(bundle.model.config, "_commit_hash", None) != runtime["model_revision"]
                or getattr(bundle.model.config, "_attn_implementation", None) != runtime["attention_implementation"]):
            raise ValueError("Loaded model does not match the section 2.2 runtime")
        image = inference.load_image(args.prepared / "original.png")
        original = np.asarray(image)
        inputs = inference.prepare_inputs(bundle, image, manifest["part1_contract"]["prompt"])
        check_candidate_tokens(bundle, image, manifest["part1_contract"]["prompt"], inputs)
        if {k: tensor_record(v, torch) for k, v in inputs.items()} != manifest["baseline_tensors"]:
            raise ValueError("Original prepared tensors differ from section 2.2")
        original_pixels = inputs["pixel_values"].detach().cpu()
        del inputs
        replacements, settings = masking.replacement_images(image, tuple(manifest["model_size_width_height"]),
                                                             manifest["config"]["blur_sigma_model_pixels"])
        if settings != manifest["fill_settings"]:
            raise ValueError("Fill settings changed")
        # On resume, recompute a baseline before accepting the previous scores.
        if rows:
            fresh = classification.compute_yes_score(bundle, image, manifest["part1_contract"]["prompt"])
            if not math.isclose(fresh, rows[0]["yes_score"], abs_tol=1e-5, rel_tol=1e-4):
                raise ValueError("Baseline drift on resume; do not mix these scores")
        for index in range(len(rows), len(tasks)):
            task = tasks[index]
            started = time.monotonic()
            if task["kind"] == "region":
                variant = inference.load_image(args.prepared / f"{task['fill']}_{task['condition']}.png")
            elif task["kind"] == "patch":
                mask = masking.rectangle_mask(original.shape[:2], task["bbox_xyxy"])
                variant = Image.fromarray(masking.apply_mask(original, mask, replacements[task["fill"]]))
            elif task["condition"] == "empty_mask":
                empty = np.zeros(original.shape[:2], dtype=bool)
                variant = Image.fromarray(masking.apply_mask(original, empty, replacements["mean"]))
            else:
                variant = image
            prepared = inference.prepare_inputs(bundle, variant, manifest["part1_contract"]["prompt"])
            for key, expected in manifest["baseline_tensors"].items():
                if key != "pixel_values" and tensor_record(prepared[key], torch) != expected:
                    raise ValueError("Non-image model inputs changed across conditions")
            stamp = tensor_record(prepared["pixel_values"], torch)
            if stamp["shape"] != manifest["baseline_tensors"]["pixel_values"]["shape"]:
                raise ValueError("Image token/crop geometry changed")
            changed_pixels = int((prepared["pixel_values"].detach().cpu() != original_pixels).any(dim=1).sum().item())
            del prepared
            score = classification.compute_yes_score(bundle, variant, manifest["part1_contract"]["prompt"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Model returned an invalid Yes score")
            response = None
            generation_status = "not_requested_for_score_only_condition"
            if task["kind"] in {"baseline", "region"}:
                try:
                    response = inference.generate(bundle, variant, manifest["part1_contract"]["prompt"],
                                                  max_new_tokens=manifest["part1_contract"]["max_new_tokens"])
                    generation_status = "generated"
                except RuntimeError as error:
                    if "generated only special or padding tokens" not in str(error):
                        raise
                    response = ""
                    generation_status = "unparsed_empty_generation"
            row = dict(task, yes_score=score, response=response,
                       prediction=classification.parse_prediction(response) if response is not None else None,
                       generation_status=generation_status,
                       pixel_tensor=stamp, changed_model_pixels=changed_pixels,
                       seconds=time.monotonic() - started)
            with result_path.open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            rows.append(row)
            check_sanity(rows, manifest["baseline_yes_score"])
            if task["kind"] != "patch" or (index + 1) % 8 == 0:
                print(f"{index+1}/{len(tasks)} {task['fill']} {task['condition']}: Yes={score:.8f}", flush=True)
    return report(args.prepared, args.output)


def report(prepared: Path, output: Path) -> dict:
    manifest = verify_prepared(prepared)
    tasks = tasks_for(manifest)
    rows = load_completed(output / "scores.jsonl", tasks)
    if len(rows) != len(tasks):
        raise ValueError("Experiment is incomplete; resume it before final reporting")
    original = rows[0]["yes_score"]
    check_sanity(rows, manifest["baseline_yes_score"])
    target = read_mask(prepared / "mask_target.png")
    patch_summary, maps = masking.summarize_patches(rows, target, original, manifest["config"]["grid"])
    summary = {"section": "2.3", "status": "single_case_pilot_complete", "case_id": manifest["case"]["case_id"],
               "baseline_yes_score": original, "conditions_completed": len(rows),
               "regions": masking.summarize_regions(rows, original), "patch_localization": patch_summary,
               "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
               "scores_sha256": sha256(output / "scores.jsonl"),
               "sanity": {"no_op_absolute_difference": abs(rows[1]["yes_score"] - original),
                          "end_repeat_absolute_difference": abs(rows[-1]["yes_score"] - original)},
               "interpretation": "Descriptive one-case perturbation effects. Fills can introduce artifacts; blur retains coarse evidence. Control anatomy differs. Scores near 0/1 can saturate. Not a clinical-causality or cohort-grounding conclusion.",
               "total_condition_seconds": sum(r["seconds"] for r in rows)}
    for fill, matrix in maps.items():
        np.save(output / f"signed_drop_{fill}.npy", matrix, allow_pickle=False)
    from medgemma_grounding.occlusion_figures import results_figure, write_tables
    results_figure(prepared, output, maps, summary)
    write_tables(output, rows, original)
    (output / "summary.json").write_text(json_text(summary))
    print(json_text(summary["regions"]), flush=True)
    print(f"Completed pilot: {output / 'results.png'}", flush=True)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Create masks, images, and processor-space overlays without model inference")
    prep.add_argument("--config", type=Path, default=Path("configs/part2/pilot-case.json"))
    prep.add_argument("--protocol", type=Path, default=Path("outputs/part2/protocol-v1"))
    prep.add_argument("--audit", type=Path, default=Path("outputs/part2/baseline-v1"))
    prep.add_argument("--baseline", type=Path, default=Path("outputs/part2/pilot-v1"))
    prep.add_argument("--hf-home", type=Path, default=Path("models/huggingface"))
    prep.add_argument("--output", type=Path, default=Path("outputs/part2/occlusion-prepared-v1"))
    execute = sub.add_parser("run", help="Score the reviewed region and patch conditions; checkpoint each result")
    execute.add_argument("--prepared", type=Path, default=Path("outputs/part2/occlusion-prepared-v1"))
    execute.add_argument("--output", type=Path, default=Path("outputs/part2/occlusion-run-v1"))
    execute.add_argument("--hf-home", type=Path, default=Path("models/huggingface"))
    execute.add_argument("--review-note", required=True, help="Record inspection of mask_review.png before viewing scores")
    execute.add_argument("--resume", action="store_true")
    render = sub.add_parser("report", help="Regenerate descriptive reports from a completed run")
    render.add_argument("--prepared", type=Path, default=Path("outputs/part2/occlusion-prepared-v1"))
    render.add_argument("--output", type=Path, default=Path("outputs/part2/occlusion-run-v1"))
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "run":
            run(args)
        else:
            report(args.prepared, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Occlusion pilot failed: {error}\n")


if __name__ == "__main__":
    main()
