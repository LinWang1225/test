"""Sparse provenance receipts, not per-token GPU execution counters.

The first accepted launch of each kernel and first call for each driver shape
are recorded. CUDA graph replay does not run these Python hooks. All policies
use the same instrumentation; no GPU atomics or per-token file writes are added.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import time

from .policy import POLICIES


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(directory, payload):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    event = {"unix_time": time.time(), "pid": os.getpid(), **payload}
    raw = (json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path / (str(os.getpid()) + ".jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, raw) != len(raw):
            raise OSError("Incomplete AR provenance receipt")
    finally:
        os.close(fd)


class LaunchReceipt:
    def __init__(self, inner, name, directory, identity):
        self.inner, self.name = inner, name
        self.directory, self.identity = directory, identity
        self.recorded_pid = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __getitem__(self, grid):
        launch = self.inner[grid]
        if self.recorded_pid == os.getpid():
            return launch

        def first_launch(*args, **kwargs):
            result = launch(*args, **kwargs)
            config = getattr(self.inner, "best_config", None)
            constants = {k: kwargs[k] for k in ["D", "GROUP", "Q_PER_KV", "Q_PER_KV_PAD",
                         "K_BITS", "V_BITS", "NUM_KV_SPLITS", "SLIDING_WINDOW", "VQ_INDIRECT", "AR_POLICY", "AR_EPSILON"] if k in kwargs}
            _write(self.directory, {**self.identity, "event": "first_python_kernel_launch",
                   "kernel": self.name, "grid": list(grid) if isinstance(grid, tuple) else str(grid),
                   "constants": constants, "autotune_config": str(config) if config is not None else None,
                   "meaning": "Launch returned without a synchronous error; may be warmup or graph capture. Not a GPU replay count."})
            self.recorded_pid = os.getpid()
            return result
        return first_launch


def install(namespace, source, policy):
    directory = os.environ.get("AR_TRACE_DIR")
    root = os.environ.get("AR_RUNTIME_ROOT")
    if not directory and not root:
        return
    if not directory or not root:
        raise RuntimeError("Both AR_TRACE_DIR and AR_RUNTIME_ROOT are required")
    from .runtime import verify_runtime
    root = Path(root).resolve(strict=True)
    manifest = verify_runtime(root, policy, full=False)
    actual_path = Path(source).resolve(strict=True)
    expected_path = root / "vllm/v1/attention/ops/triton_kvarn_decode.py"
    actual_sha = _sha(actual_path)
    if (actual_path != expected_path or policy != manifest.get("policy")
            or actual_sha != manifest.get("executed_source_sha256")):
        raise RuntimeError("AR worker loaded an unexpected kernel source or policy")
    identity = {"policy": policy, "epsilon": manifest["epsilon"], "source": str(actual_path), "source_sha256": actual_sha,
                "transformed_source_sha256": manifest["transformed_source_sha256"]}
    _write(directory, {**identity, "event": "kernel_module_import"})
    for name in ["_kvarn_fused_decode_kernel", "_kvarn_fused_decode_stage1", "_kvarn_fused_decode_stage2",
                 "_kvarn_build_packed_kv_kernel"]:
        if name in namespace:
            namespace[name] = LaunchReceipt(namespace[name], name, directory, identity)
    for name in ["kvarn_decode_attention", "kvarn_verify_attention"]:
        if name not in namespace:
            continue
        namespace[name] = _driver_receipt(namespace[name], name, directory, identity)


def _driver_receipt(inner, name, directory, identity):
    seen = set()

    @functools.wraps(inner)
    def wrapped(*args, **kwargs):
        q = args[0] if args else kwargs["query"]
        shape = tuple(q.shape)
        key = (os.getpid(), shape)
        result = inner(*args, **kwargs)
        if key not in seen:
            _write(directory, {**identity, "event": "first_python_driver_shape", "driver": name,
                   "query_shape": list(shape), "output_shape": list(result.shape),
                   "fused_decode": os.environ.get("KVARN_FUSED_DECODE", "1"),
                   "split_override": os.environ.get("KVARN_SPLIT_K"),
                   "meaning": "Successful Python driver return for this shape; graph replays are not counted."})
            seen.add(key)
        return result
    return wrapped


def summarize(directory, expected_policy=None, expected_sha=None, expected_epsilon=None):
    events = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    known_events = {"kernel_module_import", "first_python_kernel_launch", "first_python_driver_shape"}
    imported = {(e.get("pid"), e.get("policy"), e.get("source_sha256")) for e in events
                if e.get("event") == "kernel_module_import"}
    mismatches = [e for e in events if
                  (e.get("policy") not in POLICIES
                   or not isinstance(e.get("source"), str)
                   or not isinstance(e.get("source_sha256"), str)
                   or len(e.get("source_sha256", "")) != 64
                   or e.get("event") not in known_events
                   or (e.get("pid"), e.get("policy"), e.get("source_sha256")) not in imported
                   or (expected_policy is not None and e.get("policy") != expected_policy)
                   or (expected_sha is not None and e.get("source_sha256") != expected_sha)
                   or (expected_epsilon is not None and e.get("epsilon") != expected_epsilon))]
    kernels = sorted({e["kernel"] for e in events if e.get("event") == "first_python_kernel_launch"})
    drivers = sorted({e["driver"] for e in events if e.get("event") == "first_python_driver_shape"})
    return {"schema": "kvbench.residual.dispatch.v1", "events": events,
            "kernel_launch_names": kernels, "driver_names": drivers,
            "source_verified": bool(imported) and not mismatches,
            "fused_launch_observed": any(k in kernels for k in ["_kvarn_fused_decode_kernel", "_kvarn_fused_decode_stage1"]),
            "decode_driver_observed": "kvarn_decode_attention" in drivers,
            "identity_mismatches": len(mismatches),
            "formal_traffic_counters": None,
            "traffic_scope": "Formal traffic is not measured. Use separate microbench logical-byte diagnostics; they are not DRAM counters.",
            "scope": "Python imports/launches/capture only. Warmup included. No inference-token or CUDA-graph replay counts."}
