"""Launch one AR policy from an isolated runtime with a V3-compatible manifest."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from kvbench_longcontext.common import atomic_json, package_versions, sha_file, tokenizer_identity
from kvbench_longcontext.serve import http_text, launch_command, run_text, verify_cache, verify_model
from .runtime import verify_runtime
from .policy import DEFAULT_EPSILON, POLICIES, method_name, validate_epsilon
from .trace import summarize

METHODS = {"kvarn44": ("kvarn_k4v4_g128", "k4v4")}
ENV_KEYS = ("KVARN_SINK_TOKENS", "KVARN_SINKHORN_ITERS", "KVARN_POOL_SLOTS", "KVARN_FAST_FLUSH",
            "KVARN_FUSED_DECODE", "KVARN_SPLIT_K", "KVARN_NUM_KV_SPLITS", "KVARN_FUSED_VERIFY",
            "KVARN_SHARED_VERIFY", "VLLM_ATTENTION_BACKEND", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
            "VLLM_WORKER_MULTIPROC_METHOD", "CUDA_HOME", "TRITON_PTXAS_PATH",
            "NVIDIA_TF32_OVERRIDE", "TRITON_F32_DEFAULT")


def experiment_name(method, policy):
    if method not in METHODS:
        raise ValueError("The current representation is verified for K4V4 only")
    return method_name(policy)


def command_for(args, model_name):
    # Reuse the exact V3 model/engine CLI so generation defaults do not drift.
    original = launch_command(args, model_name)
    return [sys.executable, "-m", "kvbench_residual.bootstrap", "--runtime", str(args.runtime.resolve()),
            "--policy", args.policy, "--trace-dir", str((args.out / "dispatch").resolve()), "--", *original[3:]]


def validate_architecture(model):
    cfg = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    heads = cfg.get("num_attention_heads")
    kv_heads = cfg.get("num_key_value_heads")
    dim = cfg.get("head_dim")
    if dim is None and heads:
        dim = cfg.get("hidden_size", 0) // heads
    if (cfg.get("model_type"), heads, kv_heads, dim) != ("qwen3", 32, 8, 128):
        raise ValueError("This full-suite validation is scoped to Qwen3-4B (32 query heads, 8 KV heads, head_dim=128). "
                         "Extend the GPU validation matrix before serving another architecture.")
    return {"model_type": cfg["model_type"], "heads": heads, "kv_heads": kv_heads, "head_dim": dim,
            "num_hidden_layers": cfg.get("num_hidden_layers"), "config_sha256": sha_file(Path(model) / "config.json")}


def stop_owned_group(proc):
    """Clean the group created by this launcher, including orphaned workers."""
    def group_exists():
        try:
            os.killpg(proc.pid, 0)
            return True
        except ProcessLookupError:
            return False

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 20
    while group_exists() and time.monotonic() < deadline:
        proc.poll()
        time.sleep(.2)
    if group_exists():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=10)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
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
    args.out, args.runtime, args.model = args.out.resolve(), args.runtime.resolve(), args.model.resolve()
    if not 0 < args.gpu_util < 1 or min(args.max_model_len, args.max_seqs, args.batch_tokens, args.startup_timeout) < 1:
        raise ValueError("Invalid server budget")
    if not 1 <= args.port <= 65535 or not args.gpu.isdigit() or args.kv_bytes is not None and args.kv_bytes < 1:
        raise ValueError("Use one numeric physical GPU, a valid port, and a positive optional KV byte budget")
    args.epsilon = validate_epsilon(args.epsilon)
    name = experiment_name(args.method, args.policy)
    model_name = "kvlong-" + name + "-" + uuid.uuid4().hex[:10]
    dtype, _ = METHODS[args.method]
    command = command_for(args, model_name)
    config = {"method": name, "base_method": args.method, "residual_policy": args.policy, "residual_epsilon": args.epsilon,
              "kv_dtype": dtype, "block_size": 128, "model_name": model_name,
              "max_model_len": args.max_model_len, "max_seqs": args.max_seqs,
              "batch_tokens": args.batch_tokens, "gpu_util": args.gpu_util, "kv_bytes": args.kv_bytes,
              "eager": args.eager, "server_seed": args.server_seed, "apc": False,
              "generation_config": "vllm", "override_generation_config": {}}
    if args.dry_run:
        print(json.dumps({"command": command, "config": config,
                          "status": "plan-only; no GPU or server inspected"}, indent=2))
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Server output directory must be new/empty")
    runtime = verify_runtime(args.runtime, args.policy, full=True, epsilon=args.epsilon)
    architecture = validate_architecture(args.model)
    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("Port is already in use; existing processes are not stopped")
    gpu = run_text(["nvidia-smi", "-i", args.gpu, "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"])
    if gpu.get("returncode") != 0:
        raise RuntimeError("Unable to inspect target GPU: " + str(gpu))
    if gpu["stdout"] and "No running processes found" not in gpu["stdout"]:
        raise RuntimeError("Target GPU already has compute processes: " + gpu["stdout"])
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE="1", PYTHONPATH=str(args.runtime),
               AR_RUNTIME_ROOT=str(args.runtime), AR_POLICY=args.policy, AR_TRACE_DIR=str(args.out / "dispatch"),
               KVARN_FUSED_DECODE="1", TRITON_CACHE_DIR=str(args.out / "cache/triton"),
               TORCHINDUCTOR_CACHE_DIR=str(args.out / "cache/torchinductor"),
               VLLM_CACHE_ROOT=str(args.out / "cache/vllm"))
    launch = {"schema": "kvbench.longcontext.server.v1", "command": command, "config": config,
              "base_url": f"http://127.0.0.1:{args.port}", "tokenizer": tokenizer_identity(args.model),
              "packages": package_versions(["vllm", "torch", "triton", "transformers", "tokenizers"]),
              "vllm_module": str(args.runtime / "vllm/__init__.py"), "python": sys.version,
              "source_commit": runtime["parent_git"]["commit"], "source_status": runtime["parent_git"]["status"],
              "source_diff_sha256": None,
              "gpu": run_text(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid,name,driver_version,memory.total,power.limit",
                               "--format=csv,noheader,nounits"]),
              "weight_file_sizes_mtime_ns": {f.name: [f.stat().st_size, f.stat().st_mtime_ns]
                                             for f in args.model.glob("*.safetensors")},
              "environment": {k: env.get(k) for k in ENV_KEYS},
              "cuda_visible_devices": args.gpu, "native_verified": False,
              "residual": {"policy": args.policy, "epsilon": args.epsilon, "runtime_manifest": str(args.runtime / "runtime.json"),
                         "runtime_manifest_sha256": sha_file(args.runtime / "runtime.json"),
                         "parent_source_sha256": runtime["parent_source_sha256"],
                         "parent_pins": runtime["parent_pins"],
                         "generated_sha256": runtime["generated_sha256"],
                         "executed_files_sha256": runtime["executed_files_sha256"],
                         "representation": runtime["representation"],
                         "formal_traffic_counters": None,
                         "traffic_scope": runtime["traffic_note"],
                         "transformed_source_sha256": runtime["transformed_source_sha256"],
                         "executed_source_sha256": runtime["executed_source_sha256"],
                         "architecture": architecture, "parent_control": runtime["parent_control"],
                         "dispatch_directory": str(args.out / "dispatch"),
                         "fused_decode_requested_before_override": os.environ.get("KVARN_FUSED_DECODE"),
                         "fused_decode_effective": "1",
                         "launcher_files_sha256": {p.name: sha_file(p) for p in Path(__file__).parent.glob("*.py")},
                         "runtime_implementation_files_sha256": {k: v for k, v in runtime["files_sha256"].items()
                                                                 if k.startswith("kvbench_residual/") and k.endswith(".py")}}}
    atomic_json(args.out / "launch.json", launch)
    proc = None
    stopping = False

    def stop_requested(signum, frame):
        if not stopping:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    try:
        with (args.out / "server.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=args.runtime, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
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
                live_model, native = verify_model(models, config), verify_cache(metrics, config)
                receipts = summarize(args.out / "dispatch", args.policy, runtime["executed_source_sha256"], args.epsilon)
                if not receipts["source_verified"] or not receipts["fused_launch_observed"]:
                    raise RuntimeError("Server is online but the expected AR source and fused kernel launch were not verified")
                atomic_json(args.out / "dispatch.ready.json", receipts)
                (args.out / "metrics.ready.txt").write_text(metrics, encoding="utf-8")
                launch.update(native_verified=True, native_cache_config=native, models=models,
                              native_max_model_len=live_model["max_model_len"], pid=proc.pid,
                              native_kv_capacity_tokens=int(native["num_gpu_blocks"]) * 128,
                              ready_unix_time=time.time())
                launch["residual"]["dispatch_ready_sha256"] = sha_file(args.out / "dispatch.ready.json")
                atomic_json(args.out / "launch.json", launch)
                print("Ready: " + name + "; native cache and isolated fused kernel verified; "
                      + str(args.out / "launch.json"), flush=True)
                code = proc.wait()
                if code:
                    raise RuntimeError("vLLM exited with code " + str(code))
                return
            raise TimeoutError("Server startup/config verification timed out; inspect server.log")
    except KeyboardInterrupt:
        print("Stopping this AR launch's process group.", flush=True)
    finally:
        stopping = True
        try:
            if proc is not None:
                stop_owned_group(proc)
        finally:
            atomic_json(args.out / "dispatch_summary.json",
                        summarize(args.out / "dispatch", args.policy, runtime["executed_source_sha256"], args.epsilon))


if __name__ == "__main__":
    main()
