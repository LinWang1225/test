"""Versioned, non-pickle trace files and deterministic report utilities."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

SCHEMA = "kvbench.residual.real_trace.v1"
ARRAYS = ("query", "cache", "block_table", "seq_lens", "block_to_slot", "pool_k", "pool_v", "req_rows", "reference_gpu_rot")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def new_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Output must not already exist: {path}")
    path.mkdir(parents=True)
    return path


def trace_paths(directory, limit=0):
    paths = sorted(Path(directory).glob("trace_*.json"))
    if not paths:
        raise ValueError(f"No committed real traces under {directory}; do not substitute synthetic data")
    if limit < 0:
        raise ValueError("Trace limit cannot be negative")
    # Spread a bounded diagnostic across acquisition order, not just the first layer/case.
    if limit and len(paths) > limit:
        positions = np.linspace(0, len(paths)-1, limit).round().astype(int)
        paths = [paths[i] for i in positions]
    return paths


def load_trace(path):
    """Read locally produced uint8/fp16 snapshots, never pickle or torch.load."""
    from kvbench_residual.fixtures import Case, Layout
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA or meta.get("source_kind") != "real_vllm_decode":
        raise ValueError("Expected a real, versioned vLLM trace")
    name = meta["arrays_file"]
    if Path(name).name != name or not name.endswith(".npz"):
        raise ValueError("Unsafe trace array path")
    af = path.parent / name
    if sha_file(af) != meta["arrays_sha256"]:
        raise ValueError(f"Trace checksum mismatch: {af}")
    with np.load(af, allow_pickle=False) as z:
        if set(z.files) != set(ARRAYS):
            raise ValueError("Trace array schema mismatch")
        data = {k: z[k] for k in ARRAYS}
    layout = Layout(**meta["layout"])
    lengths = tuple(int(x) for x in data["seq_lens"])
    if len(lengths) != 1 or min(lengths) <= 0:
        raise ValueError("This version intentionally captures one request per trace")
    if meta["hq"] != 32 or meta["hk"] != 8 or layout.group != 128:
        raise ValueError("Only the validated Qwen3-4B K4V4/G128 shape is supported")
    if data["cache"].dtype != np.uint8 or data["cache"].shape[1:] != (8, layout.tile_bytes):
        raise ValueError("Invalid native packed cache")
    if data["query"].shape != (1, 32, 128) or data["query"].dtype != np.float16:
        raise ValueError("Expected actual rotated FP16 query, not an HF activation substitute")
    if data["reference_gpu_rot"].shape != data["query"].shape:
        raise ValueError("Invalid live attention output shape")
    for key in ("seq_lens", "block_table", "block_to_slot", "req_rows"):
        if data[key].dtype != np.int32:
            raise ValueError("Invalid integer metadata dtype: " + key)
    for key in ("pool_k", "pool_v"):
        if data[key].dtype != np.float16 or data[key].shape[1:] != (128, 8, 128):
            raise ValueError("Invalid FP16 sink/tail pool: " + key)
    if data["pool_k"].shape != data["pool_v"].shape:
        raise ValueError("K/V pool shape mismatch")
    blocks = (lengths[0]+127)//128
    if data["block_table"].shape[0] != 1 or data["block_table"].shape[1] < blocks:
        raise ValueError("Incomplete active block table")
    ids = data["block_table"][0, :blocks]
    if np.any(ids < 0) or np.any(ids >= len(data["cache"])):
        raise ValueError("Invalid compacted page mapping")
    if data["block_to_slot"].shape != (len(data["cache"]),):
        raise ValueError("Invalid pool mapping")
    slots = data["block_to_slot"][ids]
    if np.any(slots < -1) or np.any(slots >= len(data["pool_k"])):
        raise ValueError("Invalid active pool index")
    if data["req_rows"].tolist() != [0]:
        raise ValueError("Only direct decode snapshots are supported")
    if meta["block_n"] not in (16, 32, 64) or meta["splits"] < 1:
        raise ValueError("Invalid captured launch geometry")
    if not np.isfinite(meta["scale"]) or meta["scale"] <= 0:
        raise ValueError("Invalid attention scale")
    for key in ("query", "reference_gpu_rot"):
        if not np.isfinite(data[key]).all():
            raise ValueError("Nonfinite actual activation: " + key)
    data["case"] = Case(path.stem, lengths, layout, window=meta["window"], hq=32, hk=8)
    data["scale"] = float(meta["scale"])
    return data, meta


def output_metrics(output, reference, vref=None):
    o = np.asarray(output, np.float64)
    r = np.asarray(reference, np.float64)
    if o.shape != r.shape or not np.isfinite(o).all() or not np.isfinite(r).all():
        raise ValueError("Nonfinite or incompatible attention output")
    err = np.linalg.norm(o-r, axis=-1)
    norms = np.linalg.norm(r, axis=-1)
    rel = np.divide(err, norms, out=np.full_like(err, np.nan), where=norms > 1e-12)
    den = float(np.linalg.norm(r))
    result = {"abs_l2_mean": float(err.mean()), "abs_l2_max": float(err.max()),
              "relative_l2_fro": float(np.linalg.norm(o-r)/den) if den > 1e-12 else None,
              "relative_l2_max_head": float(np.nanmax(rel)) if np.any(np.isfinite(rel)) else None,
              "relative_undefined_heads": int((norms <= 1e-12).sum()),
              "max_abs_element": float(np.max(np.abs(o-r)))}
    if vref is not None:
        result["error_over_vref_max"] = float(err.max()/vref) if vref > 0 else (0.0 if err.max() == 0 else None)
    return result
