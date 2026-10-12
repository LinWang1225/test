"""Build an isolated vLLM overlay for Anchor--Residual KV experiments.

No torch/vLLM import or CUDA initialization occurs here. Source mutations are
limited to a newly copied private runtime. Installed binary targets are reused
by symlink; they are not OS-enforced read-only mounts, and this code never writes
them. Every reviewed representation writer/reader is pinned and recorded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

from .policy import DEFAULT_EPSILON, POLICIES, POLICY_DESCRIPTIONS, validate_epsilon

PARENT_COMMIT = "7586257f1c632e63187bfacbbe21ccb51540f7b3"
TARGET = Path("v1/attention/ops/triton_kvarn_decode.py")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_identity(package):
    def git(*args):
        try:
            p = subprocess.run(["git", "-C", str(package), *args], text=True,
                               capture_output=True, timeout=30)
            return {"exit_code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr[-1000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"error": str(exc)}
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def _copy_package(source, destination):
    binaries = []

    def copy_file(src, dst):
        src, dst = Path(src), Path(dst)
        if src.name.endswith((".so", ".dll", ".dylib")) or ".so." in src.name:
            real = src.resolve(strict=True)
            dst.symlink_to(real)
            stat = real.stat()
            binaries.append({"relative_path": str(dst.relative_to(destination)),
                             "source": str(real), "size": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns})
        else:
            shutil.copy2(src, dst)
        return str(dst)

    shutil.copytree(source, destination, copy_function=copy_file,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".git"))
    return binaries


def trace_footer(policy):
    return ("\n\n# AR-KV provenance hook: first Python launch receipts, not per-token counters.\n"
            "from kvbench_residual.trace import install as _ar_install_trace\n"
            "_ar_install_trace(globals(), __file__, " + repr(policy) + ")\n")


def reviewed_sources(package):
    from .transform import get_expected_pins
    pins = dict(get_expected_pins())
    if str(TARGET) not in pins or not pins:
        raise ValueError("Reviewed pin set must include the decode target")
    source = {}
    for name, expected in pins.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or not name.endswith(".py"):
            raise ValueError("Unsafe reviewed source path: " + name)
        path = package / rel
        actual = sha_file(path)
        if actual != expected:
            raise ValueError("Unsupported KVarN source: " + name + " (" + actual + "); expected " + expected
                             + ". The installed server is unchanged. Port the implementation explicitly before use.")
        source[name] = path.read_text(encoding="utf-8")
    return pins, source


def build_runtime(out, policy, package=None, epsilon=DEFAULT_EPSILON):
    from .transform import transform_sources
    out = Path(out).expanduser().resolve(strict=False)
    if policy not in POLICIES:
        raise ValueError("Unsupported Anchor--Residual policy")
    epsilon = validate_epsilon(epsilon)
    if package is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.origin:
            raise RuntimeError("Run runtime.py in the existing kvarn-serve environment; vLLM was not found")
        package = Path(spec.origin).parent
    package = Path(package).resolve(strict=True)
    if out == package or package in out.parents:
        raise ValueError("Place the runtime outside the installed vLLM package")
    if out.exists():
        raise ValueError("Runtime directory must not exist: " + str(out))
    pins, original = reviewed_sources(package)
    changed = transform_sources(original, policy, epsilon=epsilon)
    if not isinstance(changed, dict) or any(name not in original for name in changed):
        raise ValueError("Transformer must return reviewed source paths only")
    transformed = {**original, **changed}
    for name, text in transformed.items():
        if not isinstance(text, str):
            raise ValueError("Transformer returned non-text: " + name)
        compile(text, str(package / name) + ":" + policy, "exec")
    implementation = Path(__file__).resolve().parent
    implementation_sha = {str(p.relative_to(implementation)): sha_file(p)
                          for p in sorted(implementation.rglob("*.py"))}
    generated = {name: text_sha(text) for name, text in transformed.items()}
    out.mkdir(parents=True)
    try:
        binaries = _copy_package(package, out / "vllm")
        for name, expected in pins.items():
            if sha_file(out / "vllm" / name) != expected:
                raise ValueError("Parent source changed while copying the private runtime: " + name)
        shutil.copytree(implementation, out / "kvbench_residual",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        for name, expected in implementation_sha.items():
            if sha_file(out / "kvbench_residual" / name) != expected:
                raise ValueError("Residual implementation changed during runtime construction: " + name)
        for name, text in transformed.items():
            (out / "vllm" / name).write_text(text + (trace_footer(policy) if name == str(TARGET) else ""), encoding="utf-8")
        kernel = out / "vllm" / TARGET
        executed = {name: sha_file(out / "vllm" / name) for name in transformed}
        copied = {str(p.relative_to(out)): sha_file(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and not p.is_symlink()}
        manifest = {
            "schema": "kvbench.residual.runtime.v1", "policy": policy, "epsilon": epsilon,
            "policy_description": POLICY_DESCRIPTIONS[policy],
            "created": datetime.now(timezone.utc).isoformat(), "runtime_root": str(out),
            "parent_expected_commit": PARENT_COMMIT, "parent_package": str(package),
            "parent_source": str(package / TARGET), "parent_source_sha256": pins[str(TARGET)],
            "parent_pins": pins, "parent_git": _source_identity(package),
            "kernel_path": str(kernel), "generated_sha256": generated, "executed_files_sha256": executed,
            "transformed_source_sha256": generated[str(TARGET)], "executed_source_sha256": executed[str(TARGET)],
            "files_sha256": copied, "shared_binaries": binaries,
            "implementation_sha256": implementation_sha,
            "parent_control": POLICY_DESCRIPTIONS["parent"],
            "representation": "original KVarN packed nibble" if policy == "parent" else "independent high-2/low-2 bit planes; same full K4V4 codes",
            "cuda_initialized": False,
            "native_library_note": "Existing compiled libraries are symlinked; sizes/mtime are checked, not full content hashes.",
            "traffic_note": "Formal runtime has no per-token traffic counters. Separate diagnostic replay reports logical bytes, not measured DRAM transactions.",
        }
        atomic_json(out / "runtime.json", manifest)
        return manifest
    except BaseException:
        (out / "BUILD_FAILED.txt").write_text("Runtime construction failed. Use a new directory after fixing the error.\n", encoding="utf-8")
        raise


def verify_runtime(root, policy=None, full=False, epsilon=None):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "kvbench.residual.runtime.v1" or manifest.get("runtime_root") != str(root):
        raise ValueError("Unknown or relocated Anchor--Residual runtime")
    if policy is not None and manifest.get("policy") != policy:
        raise ValueError("Runtime policy differs from the requested method")
    if epsilon is not None and manifest.get("epsilon") != validate_epsilon(epsilon):
        raise ValueError("Runtime error budget differs from the requested method")
    pinned = set(manifest["parent_pins"])
    if str(TARGET) not in pinned or set(manifest["executed_files_sha256"]) != pinned:
        raise ValueError("Runtime is missing reviewed writer/reader evidence")
    for name, expected in manifest["executed_files_sha256"].items():
        if sha_file(root / "vllm" / name) != expected:
            raise ValueError("Executed residual writer/reader changed after runtime creation: " + name)
    for rel, expected in manifest["files_sha256"].items():
        if full or (rel.startswith("kvbench_residual/") and rel.endswith(".py")) or rel in {"vllm/" + name for name in pinned}:
            path = root / rel
            if not path.is_file() or path.is_symlink() or sha_file(path) != expected:
                raise ValueError("Runtime source changed: " + rel)
    for item in manifest["shared_binaries"]:
        path = root / "vllm" / item["relative_path"]
        stat = path.stat()
        if (not path.is_symlink() or str(path.resolve()) != item["source"]
                or stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]):
            raise ValueError("Shared backend binary changed: " + str(path))
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    args = p.parse_args(argv)
    result = build_runtime(args.out, args.policy, epsilon=args.epsilon)
    print(json.dumps({k: result[k] for k in ["runtime_root", "policy", "epsilon", "parent_source",
                     "parent_source_sha256", "generated_sha256", "executed_source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
