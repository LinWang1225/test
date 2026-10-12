from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_COMMIT = "3cf8dd9345e8e8181b0069b6b39c3355ae463b67"
LONGBENCH_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
HF_REPO = "zai-org/LongBench"
HF_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
DATA_ZIP_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
TASKS = json.loads((HERE / "config/tasks.json").read_text(encoding="utf-8"))
CORE_TASKS = list(TASKS)[:6]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical(row) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def tokenizer_identity(model):
    model = Path(model).resolve()
    if not model.is_dir():
        raise ValueError("Use the same existing local model snapshot for prepare and serve: " + str(model))
    files = {p.name: sha_file(p) for p in sorted(model.iterdir()) if p.is_file()
             and p.suffix in (".json", ".jinja", ".txt", ".model") and p.stat().st_size < 100_000_000}
    if not files:
        raise ValueError("No tokenizer/model configuration files found")
    return {"model_path": str(model), "files_sha256": files, "fingerprint": digest(files)}


def quantile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def describe(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {"n": len(values), "mean": sum(values) / len(values) if values else None,
            "p50": quantile(values, .5), "p95": quantile(values, .95),
            "p99": quantile(values, .99), "max": max(values) if values else None}


def checked_data(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "kvbench.longcontext.data.v1":
        raise ValueError("Unknown frozen data schema")
    policy = manifest.get("generation_budget_policy")
    if policy not in ("remaining-context", "reference", "fixed") or manifest.get("settings", {}).get("generation_budget_policy") != policy:
        raise ValueError("Frozen data have no matching explicit generation budget policy; prepare with the current version")
    window = manifest["settings"]["max_model_len"]
    datasets = {}
    for name, info in manifest["datasets"].items():
        path = folder / info["file"]
        if path.parent.resolve() != folder.resolve() or sha_file(path) != info["sha256"]:
            raise ValueError("Frozen request data changed: " + name)
        rows = read_jsonl(path)
        if len(rows) != info["selected_count"] or len({r["case_id"] for r in rows}) != len(rows):
            raise ValueError("Frozen count or case IDs do not match: " + name)
        if [r["case_id"] for r in rows] != info["selected_case_ids"]:
            raise ValueError("Frozen order changed: " + name)
        for row in rows:
            remaining = window - row["input_tokens"]
            cap = row["max_tokens"]
            valid_cap = cap is None if policy == "remaining-context" else isinstance(cap, int) and not isinstance(cap, bool) and cap > 0
            if (row.get("generation_budget_policy") != policy or row.get("max_model_len") != window
                    or row.get("remaining_context_tokens") != remaining or remaining < 1 or not valid_cap
                    or cap is not None and cap > remaining
                    or row.get("generation_budget_tokens") != (remaining if cap is None else cap)):
                raise ValueError("Invalid frozen generation/context budget: " + row["case_id"])
        datasets[name] = rows
    return manifest, datasets
