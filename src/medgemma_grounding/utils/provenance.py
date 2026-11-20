"""Source identity for new runs and explicit, read-only historical validation."""


from pathlib import Path

from medgemma_grounding.utils.io import file_sha256, read_json


PACKAGE = Path(__file__).resolve().parents[1]
# Commands run from the repository root, including when testing an installed wheel.
PROJECT = Path.cwd().resolve()
SNAPSHOT = PROJECT / "results/maintenance/layout_20261120"


def source_hashes():
    """Hash the complete package using package-relative paths, including subpackages."""
    return {str(p.relative_to(PACKAGE)): file_sha256(p)
            for p in sorted(PACKAGE.rglob("*.py"))}


def historical_sources(expected):
    """Verify the archived bytes; never substitute current files for old filenames."""
    manifest = read_json(SNAPSHOT / "snapshot.json")
    package = SNAPSHOT / "historical_runtime/medgemma_grounding"
    for name, digest in manifest["source_sha256"].items():
        if Path(name).name != name or file_sha256(package / name) != digest:
            raise ValueError(f"Historical runtime snapshot changed: {name}")
    if any(manifest["source_sha256"].get(name) != digest for name, digest in expected.items()):
        raise ValueError("The archived implementation does not match this historical protocol")
    return package


def is_historical(protocol):
    hashes = protocol.get("source_sha256", protocol.get("code_sha256", {}))
    return bool(hashes) and all("/" not in name for name in hashes)


def validate_sources(expected, *, historical=False):
    """Validate archived sources only when the caller explicitly permits historical reading.

    New execution/resume requires the entire current package identity, so adding,
    moving or changing a source file invalidates that resume as well.
    """
    if not expected:
        return  # Some small test fixtures do not carry a runtime identity.
    if all("/" not in name for name in expected):
        if not historical:
            raise ValueError("Historical runtime: use offline replay; new execution needs a new output directory")
        historical_sources(expected)
    elif expected != source_hashes():
        raise ValueError("Runtime source changed; new protocol/output directory required")
