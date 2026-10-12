"""Load the private backend in this process and propagate it to spawned workers."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys

from .runtime import atomic_json, verify_runtime
from .policy import POLICIES


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--trace-dir", type=Path, required=True)
    args, remaining = p.parse_known_args(argv)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    runtime = args.runtime.resolve(strict=True)
    manifest = verify_runtime(runtime, args.policy)
    if "vllm" in sys.modules:
        raise RuntimeError("Bootstrap must run before importing vLLM")
    sys.path.insert(0, str(runtime))
    os.environ.update(PYTHONPATH=str(runtime), PYTHONNOUSERSITE="1",
                      AR_RUNTIME_ROOT=str(runtime), AR_TRACE_DIR=str(args.trace_dir.resolve()),
                      AR_POLICY=args.policy)
    os.environ.pop("PYTHONHOME", None)
    spec = importlib.util.find_spec("vllm")
    if not spec or Path(spec.origin).resolve() != runtime / "vllm/__init__.py":
        raise RuntimeError("Isolated vLLM did not take precedence over the installed package")
    atomic_json(args.trace_dir / ("bootstrap_" + str(os.getpid()) + ".json"), {
        "schema": "kvbench.residual.bootstrap.v1", "pid": os.getpid(),
        "policy": args.policy, "epsilon": manifest["epsilon"], "vllm_origin": spec.origin,
        "runtime": str(runtime), "executed_source_sha256": manifest["executed_source_sha256"],
        "python": sys.executable, "worker_pythonpath": os.environ["PYTHONPATH"]})
    sys.argv = ["vllm.entrypoints.openai.api_server", *remaining]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
