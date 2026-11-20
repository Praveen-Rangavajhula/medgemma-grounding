"""Replay saved attention reports/audits with the exact historical runtime, without weights.

The source snapshot and scientific artifacts remain outside the active package.
No historical inference or incomplete-run continuation is supported here.
"""


from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile

from medgemma_grounding.utils.io import read_json
from medgemma_grounding.utils.provenance import PROJECT, historical_sources, is_historical


# This bootstrap runs in a separate interpreter: old imports and old source-path
# checks cannot collide with the reorganized package. The temporary working tree
# exposes the frozen src paths expected by independent artifact inventories.
BOOTSTRAP = r'''
import importlib, json, runpy, sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
def forbidden(*args, **kwargs):
    raise RuntimeError("Historical replay is model-free; incomplete runs need a new implementation/protocol")
for name, attr in (("inference", "load_model"), ("sae_reconstruction", "load_scope_sae"),
                   ("medgemma_interventions", "frozen_runtime"), ("medgemma_attention", "runtime")):
    setattr(importlib.import_module("medgemma_grounding." + name), attr, forbidden)
module = importlib.import_module("medgemma_grounding." + request["module"])
kwargs = request["kwargs"]
for key in ("output", "source", "failed", "data_root", "preflight", "directory"):
    if key in kwargs:
        kwargs[key] = Path(kwargs[key])
if request.get("script"):
    sys.argv = [request["script"], request["audit_output"]]
    runpy.run_path(request["script"], run_name="__main__")
    result = json.loads(Path(request["audit_output"]).read_text())
else:
    if kwargs.get("stage") == "run":
        output = kwargs["output"]
        p = module.base.read_json(output / "protocol.json")
        module.validate(output, p)
        if not all(module.complete_case(output, p, cid) for cid in p["case_ids"]):
            raise ValueError("Historical replay refuses incomplete-run execution")
    result = module.run(**kwargs)
Path(request["result"]).write_text(json.dumps(result, allow_nan=False))
'''


def run_historical(module, kwargs, *, script=None, audit_output=None):
    """Use archived code only for saved-only operations; return its JSON result."""
    output = Path(kwargs["output"]).resolve()
    protocol = read_json(output / "protocol.json")
    package = historical_sources(protocol.get("source_sha256", protocol.get("code_sha256", {})))
    stage = kwargs.get("stage")
    if stage not in {"report", "existing", "verify", "run"}:
        raise ValueError("Historical replay supports saved-only reports and completed attention resumes")
    if stage == "run" and module not in {"medgemma_attention", "medgemma_routing"}:
        raise ValueError("Only completed attention runs support historical resume verification")
    if script:
        if audit_output is None:
            raise ValueError("An independent audit needs a separate output path")
        destination = Path(audit_output).resolve()
        if destination == output or output in destination.parents:
            raise ValueError("Write the audit outside the frozen historical result directory")
    with tempfile.TemporaryDirectory(prefix="medgemma-replay-") as tmp:
        work = Path(tmp)
        (work / "src").mkdir()
        (work / "src/medgemma_grounding").symlink_to(package, target_is_directory=True)
        for name in ("data", "results"):
            (work / name).symlink_to(PROJECT / name, target_is_directory=True)
        request = {"module": module, "kwargs": {k: str(v.resolve()) if isinstance(v, Path) else v
                                                for k, v in kwargs.items()},
                   "script": str(Path(script).resolve()) if script else None,
                   "audit_output": str(Path(audit_output).resolve()) if audit_output else None,
                   "result": str(work / "result.json")}
        (work / "request.json").write_text(json.dumps(request))
        env = {**os.environ, "PYTHONPATH": str(work / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run([sys.executable, "-c", BOOTSTRAP, str(work / "request.json")],
                       cwd=work, env=env, check=True)
        return read_json(work / "result.json")


def replay_if_historical(module, **kwargs):
    path = Path(kwargs["output"]) / "protocol.json"
    if path.exists() and is_historical(read_json(path)):
        return run_historical(module, kwargs)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, choices=["medgemma_attention", "medgemma_routing"])
    parser.add_argument("--stage", choices=["report", "existing", "verify", "run"], default="report")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.audit_script and not args.audit_output:
        parser.error("--audit-script requires a separate --audit-output path")
    kwargs = {"stage": args.stage, "output": args.output_dir, "device": "mps", "data_root": Path("data")}
    result = run_historical(args.module, kwargs, script=args.audit_script, audit_output=args.audit_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
