"""Explicit vLLM launcher: records requested and native cache configuration."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

from .common import atomic_json, package_versions, sha_file, tokenizer_identity

# g64 is a token group of 64, never a K6/V4 alias. Block size follows that group.
METHODS = {
    "fp16": ("auto", 128), "fp8": ("fp8_e4m3", 128),
    "tq4": ("turboquant_4bit_nc", 128), "tq34": ("turboquant_k3v4_nc", 128),
    "tq3": ("turboquant_3bit_nc", 128),
    "kvarn": ("kvarn_k4v2_g128", 128), "kvarn44": ("kvarn_k4v4_g128", 128),
    "kvarn64": ("kvarn_k4v2_g64", 64), "kvarn4464": ("kvarn_k4v4_g64", 64),
}


def http_text(url, timeout=5, headers=None):
    # Benchmark endpoint is local; never route model traffic through an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, headers=headers or {}), timeout=timeout) as response:
        return response.read().decode("utf-8")


def cache_configs(text):
    result = []
    for line in text.splitlines():
        if line.startswith("#") or "vllm:cache_config_info{" not in line:
            continue
        labels = {k: json.loads('"' + v + '"') for k, v in re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', line)}
        result.append(labels)
    return result


def verify_cache(text, expected):
    configs = cache_configs(text)
    if len(configs) != 1:
        raise ValueError("Expected exactly one native vllm:cache_config_info record (single GPU); cannot verify cache")
    actual = configs[0]
    checks = {"cache_dtype": expected["kv_dtype"], "block_size": str(expected["block_size"]),
              "enable_prefix_caching": "False"}
    if expected["kv_dtype"] == "fp8_e4m3":
        checks["calculate_kv_scales"] = "True"
    mismatches = {k: {"expected": v, "actual": actual.get(k)} for k, v in checks.items() if actual.get(k) != v}
    if mismatches:
        raise ValueError("Actual server cache configuration differs: " + json.dumps(mismatches))
    blocks = actual.get("num_gpu_blocks", "")
    if not re.fullmatch(r"[0-9]+", blocks) or int(blocks) < 1:
        raise ValueError("Native cache num_gpu_blocks must be a positive integer")
    return actual


def verify_model(models, expected):
    matching = [m for m in models.get("data", []) if m.get("id") == expected["model_name"]]
    if len(matching) != 1:
        raise ValueError("Live server model identity does not match this launch")
    actual = matching[0].get("max_model_len")
    if not isinstance(actual, int) or isinstance(actual, bool) or actual != expected["max_model_len"]:
        raise ValueError(f"Native /v1/models max_model_len={actual!r}, expected {expected['max_model_len']}; context budget not verified")
    return matching[0]


def launch_command(args, model_name):
    dtype, block = METHODS[args.method]
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(args.model.resolve()), "--tokenizer", str(args.model.resolve()),
        "--served-model-name", model_name, "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "float16", "--kv-cache-dtype", dtype, "--block-size", str(block),
        "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_seqs),
        "--max-num-batched-tokens", str(args.batch_tokens), "--gpu-memory-utilization", str(args.gpu_util),
        "--seed", str(args.server_seed), "--no-enable-prefix-caching", "--generation-config", "vllm"]
    if dtype == "fp8_e4m3":
        command.append("--calculate-kv-scales")
    if args.kv_bytes is not None:
        command.extend(["--kv-cache-memory-bytes", str(args.kv_bytes)])
    if args.eager:
        command.append("--enforce-eager")
    return command


def run_text(command, timeout=15):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--method", choices=list(METHODS), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-seqs", type=int, default=16)
    p.add_argument("--batch-tokens", type=int, default=1024)
    p.add_argument("--gpu-util", type=float, default=.85)
    p.add_argument("--kv-bytes", type=int)
    p.add_argument("--server-seed", type=int, default=20260917)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--startup-timeout", type=int, default=1800)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not 0 < args.gpu_util < 1 or min(args.max_model_len, args.max_seqs, args.batch_tokens, args.startup_timeout) < 1:
        raise ValueError("Invalid server budget")
    if args.kv_bytes is not None and args.kv_bytes < 1:
        raise ValueError("--kv-bytes must be positive")
    model_name = "kvlong-" + args.method + "-" + uuid.uuid4().hex[:10]
    dtype, block = METHODS[args.method]
    command = launch_command(args, model_name)
    config = {"method": args.method, "kv_dtype": dtype, "block_size": block,
              "model_name": model_name, "max_model_len": args.max_model_len, "max_seqs": args.max_seqs,
              "batch_tokens": args.batch_tokens, "gpu_util": args.gpu_util, "kv_bytes": args.kv_bytes,
              "eager": args.eager, "server_seed": args.server_seed, "apc": False,
              "generation_config": "vllm", "override_generation_config": {}}
    if args.dry_run:
        print(json.dumps({"command": command, "config": config, "status": "plan-only; no GPU or server inspected"}, indent=2))
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Server output directory must be new/empty")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("Port is already in use; existing processes are not stopped")
    gpu = run_text(["nvidia-smi", "-i", args.gpu, "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"])
    if gpu.get("returncode") != 0:
        raise RuntimeError("Unable to inspect target GPU: " + str(gpu))
    if gpu["stdout"] and "No running processes found" not in gpu["stdout"]:
        raise RuntimeError("Target GPU already has compute processes: " + gpu["stdout"])
    args.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Run serve in the existing kvarn-serve environment")
    module_dir = str(Path(spec.origin).parent)
    source_commit = run_text(["git", "-C", module_dir, "rev-parse", "HEAD"])
    source_status = run_text(["git", "-C", module_dir, "status", "--porcelain"])
    diff = run_text(["git", "-C", module_dir, "diff", "HEAD"], timeout=60)
    (args.out / "engine_source.diff").write_text(diff.get("stdout", ""), encoding="utf-8")
    launch = {"schema": "kvbench.longcontext.server.v1", "command": command, "config": config,
              "base_url": f"http://127.0.0.1:{args.port}", "tokenizer": tokenizer_identity(args.model),
              "packages": package_versions(["vllm", "torch", "triton", "transformers", "tokenizers"]),
              "vllm_module": spec.origin, "source_commit": source_commit, "source_status": source_status,
              "source_diff_sha256": sha_file(args.out / "engine_source.diff"), "python": sys.version,
              "gpu": run_text(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid,name,driver_version,memory.total,power.limit",
                               "--format=csv,noheader,nounits"]),
              "weight_file_sizes_mtime_ns": {f.name: [f.stat().st_size, f.stat().st_mtime_ns]
                                             for f in args.model.glob("*.safetensors")},
              "environment": {k: os.environ.get(k) for k in ["CUDA_HOME", "TRITON_PTXAS_PATH", "KVARN_SINK_TOKENS",
                              "KVARN_SINKHORN_ITERS", "VLLM_ATTENTION_BACKEND", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"]},
              "cuda_visible_devices": args.gpu, "native_verified": False}
    atomic_json(args.out / "launch.json", launch)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE="1")
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    proc = None
    try:
        with (args.out / "server.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("vLLM exited; inspect " + str(args.out / "server.log"))
                try:
                    models = json.loads(http_text(launch["base_url"] + "/v1/models"))
                    metrics = http_text(launch["base_url"] + "/metrics")
                except (OSError, ValueError):
                    time.sleep(1)
                    continue
                live_model = verify_model(models, config)
                native = verify_cache(metrics, config)
                (args.out / "metrics.ready.txt").write_text(metrics, encoding="utf-8")
                launch.update(native_verified=True, native_cache_config=native, models=models,
                              native_max_model_len=live_model["max_model_len"], pid=proc.pid,
                              native_kv_capacity_tokens=int(native["num_gpu_blocks"]) * int(native["block_size"]),
                              ready_unix_time=time.time())
                atomic_json(args.out / "launch.json", launch)
                print("Ready; native dtype/block verified. Client manifest: " + str((args.out / "launch.json").resolve()), flush=True)
                returncode = proc.wait()
                if returncode:
                    raise RuntimeError("vLLM exited with code " + str(returncode))
                return
            raise TimeoutError("Server startup/config verification timed out; inspect server.log")
    except KeyboardInterrupt:
        print("Stopping this launch's process group.", flush=True)
    finally:
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
