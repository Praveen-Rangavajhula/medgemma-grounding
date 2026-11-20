"""CPU-only feature-cache readers, writers and integrity checks; no model loading."""


from __future__ import annotations

from pathlib import Path
import json

import torch

from medgemma_grounding.utils.io import file_sha256, identity_hash


def save_cache(directory, manifest, activations):
    """Publish the manifest last; its checksum detects interrupted replacement."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "activations.pt.tmp"
    torch.save(activations, temporary)
    manifest = {**manifest, "activations_sha256": file_sha256(temporary)}
    temporary.replace(directory / "activations.pt")
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "manifest.json")


def load_cache(directory):
    """Read CPU features and validate alignment without loading a model or SAE."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported feature-cache version.")
    path = directory / "activations.pt"
    if file_sha256(path) != manifest["activations_sha256"]:
        raise ValueError("Feature cache checksum mismatch; extraction may have been interrupted.")
    activations = torch.load(path, map_location="cpu", weights_only=True)
    cases = manifest["cases"]
    if len({c["case_id"] for c in cases}) != len(cases) or set(activations) != {c["case_id"] for c in cases}:
        raise ValueError("Feature cache has missing, duplicate, or unexpected cases.")
    for case in cases:
        x = activations[case["case_id"]]
        expected = (len(case["tokens"]), manifest["sae_width"])
        if x.layout != torch.sparse_coo or tuple(x.shape) != expected:
            raise ValueError(f"Token/feature alignment mismatch for {case['case_id']}.")
        x = x.coalesce()
        if not torch.isfinite(x.values()).all() or (x.values() < 0).any():
            raise ValueError("Cache contains invalid feature values.")
        if [t["index"] for t in case["tokens"]] != list(range(len(case["tokens"]))):
            raise ValueError("Token indices are out of order.")
        image_path = Path(manifest["project_root"]) / case["image_path"]
        if file_sha256(image_path) != case["image_sha256"]:
            raise ValueError(f"Image changed since extraction: {image_path}")
        activations[case["case_id"]] = x
    return manifest, activations


def load_discovery_case(directory, case_id, protocol=None):
    """Lazy shard reader; reject confirmation, altered tensors and input images."""
    directory = Path(directory)
    protocol = protocol or json.loads((directory / "protocol.json").read_text())
    if identity_hash({k: v for k, v in protocol.items() if k != "protocol_hash"}) != protocol["protocol_hash"]:
        raise ValueError("Protocol hash mismatch")
    expected = next((c for c in protocol["discovery_cases"] if c["case_id"] == case_id), None)
    if expected is None or expected["partition"] != "discovery":
        raise ValueError("Only discovery cases are accessible in this milestone")
    record = json.loads((directory / "cases" / f"{case_id}.json").read_text())
    if record["protocol_hash"] != protocol["protocol_hash"] or any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Case/protocol identity mismatch")
    tensor_path = directory / "cases" / f"{case_id}.pt"
    if file_sha256(tensor_path) != record["tensor_sha256"] or file_sha256(Path(record["image_path"])) != record["image_sha256"]:
        raise ValueError("Case tensor or input image checksum mismatch")
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    x = payload["features"]
    if (x.layout != torch.sparse_coo or tuple(x.shape) != (len(record["tokens"]), protocol["sae_width"])
            or not x.is_coalesced() or not torch.isfinite(x.values()).all() or (x.values() < 0).any()
            or record["feature_shape"] != list(x.shape)
            or [t["index"] for t in record["tokens"]] != list(range(len(record["tokens"])))):
        raise ValueError("Invalid case feature tensor")
    return record, payload
