"""Frozen E-005/E-007 cohort, runtime and calibration protocols."""


from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import subprocess

from medgemma_grounding.analysis.metrics import outcome
from medgemma_grounding.sae.cache import load_discovery_case
from medgemma_grounding.sae.reconstruction import SAE_REVISION, checkpoint_spec
from medgemma_grounding.utils.io import atomic_json, file_sha256, hashed, identity_hash, read_json
from medgemma_grounding.utils.provenance import source_hashes


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
    from medgemma_grounding.data.chexlocalize import validate_partition

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
                "source_sha256": source_hashes()}
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
        settings["source_sha256"].update(source_hashes())
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
