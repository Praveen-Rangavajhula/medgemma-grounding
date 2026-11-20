"""Atomic result I/O and content hashes shared by extraction and interventions."""


from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json


def read_json(path):
    return json.loads(Path(path).read_text())


def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def tree_hashes(directory):
    return {str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
