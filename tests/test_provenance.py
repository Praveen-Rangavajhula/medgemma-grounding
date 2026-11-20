"""Source movement must not weaken resume checks or rewrite historical identity."""


from pathlib import Path
import json

import pytest

from medgemma_grounding.experiments import replay
from medgemma_grounding.utils import provenance
from medgemma_grounding.utils.io import file_sha256


def test_package_identity_covers_nested_added_and_removed_files(tmp_path, monkeypatch):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "__init__.py").write_text("")
    source = tmp_path / "experiments/run.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.setattr(provenance, "PACKAGE", tmp_path)
    frozen = provenance.source_hashes()
    assert "experiments/run.py" in frozen
    provenance.validate_sources(frozen)
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        provenance.validate_sources(frozen)
    source.write_text("VALUE = 1\n")
    extra = tmp_path / "extra.py"
    extra.write_text("")
    with pytest.raises(ValueError, match="source changed"):
        provenance.validate_sources(frozen)
    extra.unlink()
    source.unlink()
    with pytest.raises(ValueError, match="source changed"):
        provenance.validate_sources(frozen)


def test_historical_read_requires_verified_archive_and_explicit_opt_in(tmp_path, monkeypatch):
    package = tmp_path / "historical_runtime/medgemma_grounding"
    package.mkdir(parents=True)
    source = package / "old.py"
    source.write_text("VALUE = 1\n")
    frozen = {"old.py": file_sha256(source)}
    (tmp_path / "snapshot.json").write_text(json.dumps({"source_sha256": frozen}))
    monkeypatch.setattr(provenance, "SNAPSHOT", tmp_path)
    with pytest.raises(ValueError, match="Historical runtime"):
        provenance.validate_sources(frozen)
    provenance.validate_sources(frozen, historical=True)
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        provenance.validate_sources(frozen, historical=True)


def test_historical_protocol_cannot_claim_an_unmatched_source(tmp_path, monkeypatch):
    package = tmp_path / "historical_runtime/medgemma_grounding"
    package.mkdir(parents=True)
    source = package / "old.py"
    source.write_text("")
    (tmp_path / "snapshot.json").write_text(json.dumps({
        "source_sha256": {"old.py": file_sha256(source)}}))
    monkeypatch.setattr(provenance, "SNAPSHOT", tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        provenance.validate_sources({"old.py": "incorrect"}, historical=True)


def test_current_protocol_does_not_dispatch_to_historical_runtime(tmp_path, monkeypatch):
    (tmp_path / "protocol.json").write_text(json.dumps({
        "source_sha256": {"experiments/run.py": "current"}}))
    monkeypatch.setattr(replay, "run_historical", lambda *a, **k: pytest.fail("Wrong implementation"))
    assert replay.replay_if_historical("medgemma_routing", output=tmp_path, stage="run") is None


def test_historical_execution_is_blocked_before_any_subprocess(tmp_path, monkeypatch):
    (tmp_path / "protocol.json").write_text(json.dumps({"source_sha256": {"old.py": "hash"}}))
    monkeypatch.setattr(replay, "historical_sources", lambda expected: Path("archive"))
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **k: pytest.fail("Execution attempted"))
    with pytest.raises(ValueError, match="saved-only"):
        replay.run_historical("medgemma_routing", {"output": tmp_path, "stage": "benchmark"})
    with pytest.raises(ValueError, match="Only completed attention"):
        replay.run_historical("medgemma_discovery", {"output": tmp_path, "stage": "run"})


def test_replay_audit_cannot_overwrite_frozen_results(tmp_path, monkeypatch):
    (tmp_path / "protocol.json").write_text(json.dumps({"source_sha256": {"old.py": "hash"}}))
    original = (tmp_path / "protocol.json").read_bytes()
    monkeypatch.setattr(replay, "historical_sources", lambda expected: Path("archive"))
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **k: pytest.fail("Execution attempted"))
    with pytest.raises(ValueError, match="outside the frozen"):
        replay.run_historical("medgemma_routing", {"output": tmp_path, "stage": "verify"},
                              script=tmp_path / "audit.py", audit_output=tmp_path / "protocol.json")
    assert (tmp_path / "protocol.json").read_bytes() == original
