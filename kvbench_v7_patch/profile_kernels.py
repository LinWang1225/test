#!/usr/bin/env python3
"""
Short, separate CUDA-kernel profiling step for the KVarN/TurboQuant vLLM fork.

This is NOT a latency benchmark.  It enables vLLM's PyTorch profiler only for a
small decode slice, then groups CUDA kernel time into:
  - kv_quant_store
  - kv_decode_dequant_fused
  - fp16_attention_decode
  - sampling
  - other

Important: TurboQuant/KVarN dequantization is fused into decode-attention
kernels in the current implementation.  Therefore "kv_decode_dequant_fused"
cannot be split into pure attention vs pure dequant time without changing the
kernel/doing an ablation.  The main run's TPOT remains the end-to-end decode
metric; this script explains where CUDA kernel time goes.
"""
from __future__ import annotations
import argparse, csv, gzip, json, os, signal, sqlite3, subprocess, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import conda_command, ensure_gpu_idle, http_text, port_busy, stop_process_group

def post(url: str, timeout: int = 600):
    req = urllib.request.Request(url, data=b"", method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return r.read()

def wait_server(base: str, proc: subprocess.Popen, timeout: int):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("vLLM server exited during startup")
        try:
            http_text(base + "/health", 2)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("server startup timeout")

def classify(name: str) -> str:
    n = name.lower()

    # Quantize/store/flush work.  These are the parts that can reasonably be
    # called "quantization overhead" from an actual-kernel trace.
    quant_store = (
        "_tq_fused_store", "triton_turboquant_store",
        "_kvarn_scatter_store", "kvarn_store_tile", "_kvarn_flush",
        "kvarn_flush", "sinkhorn", "variance_normalize",
    )
    if any(x in n for x in quant_store):
        return "kv_quant_store"

    # Actual quantized decode paths.  Dequant is fused with attention / packing,
    # so this category is deliberately named fused rather than "dequant only".
    fused = (
        "_tq_decode_stage1", "_fwd_kernel_stage2", "_tq_full_dequant",
        "turboquant_decode",
        "_kvarn_fused_decode", "_kvarn_build_packed_kv",
        "_kvarn_dequant_blocks", "kvarn_grouped_stage1",
        "kvarn_splitk_stage1", "kvarn_splitk_stage2", "kvarn_decode",
    )
    if any(x in n for x in fused):
        return "kv_decode_dequant_fused"

    # FP16 / generic attention kernels.  In a decode-only capture this is the
    # baseline attention-decode bucket; raw names are retained for auditing.
    attention = ("flashattn", "flash_attn", "flash_fwd", "fmha", "flash::")
    if any(x in n for x in attention):
        return "fp16_attention_decode"

    if any(x in n for x in ("topk", "top_k", "topp", "top_p", "sampling", "multinomial")):
        return "sampling"
    return "other"

def load_trace_events(profile_dir: Path):
    files = sorted(profile_dir.rglob("*.pt.trace.json.gz")) + sorted(profile_dir.rglob("*.pt.trace.json"))
    if not files:
        # TensorBoard handler names can vary by torch version.
        files = sorted(profile_dir.rglob("*.json.gz")) + sorted(profile_dir.rglob("*.json"))
    events = []
    for path in files:
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    obj = json.load(f)
            else:
                obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for e in obj.get("traceEvents", []):
            if e.get("ph") != "X" or e.get("dur") is None:
                continue
            cat = str(e.get("cat", "")).lower()
            # PyTorch profiler CUDA kernel events normally use cat="kernel".
            if "kernel" not in cat:
                continue
            try:
                dur_us = float(e["dur"])
            except Exception:
                continue
            events.append((str(e.get("name", "unknown")), dur_us, str(path)))
    return files, events

def summarize_profile(profile_dir: Path, out_dir: Path, method: str, profile_iterations: int):
    files, events = load_trace_events(profile_dir)
    if not events:
        raise RuntimeError(
            f"No CUDA kernel events found under {profile_dir}. "
            "Raw traces are preserved; inspect profiler_out_0.txt / trace format."
        )
    cat_us = defaultdict(float)
    cat_calls = defaultdict(int)
    kernel_us = defaultdict(float)
    kernel_calls = defaultdict(int)
    for name, dur_us, _ in events:
        c = classify(name)
        cat_us[c] += dur_us
        cat_calls[c] += 1
        kernel_us[(c, name)] += dur_us
        kernel_calls[(c, name)] += 1
    total = sum(cat_us.values())
    cats = {}
    for c in sorted(cat_us):
        cats[c] = {
            "cuda_time_ms": cat_us[c] / 1000.0,
            "fraction_of_captured_cuda_kernel_time": cat_us[c] / total if total else None,
            "cuda_time_ms_per_requested_worker_iteration": (
                (cat_us[c] / 1000.0) / profile_iterations if profile_iterations else None
            ),
            "kernel_instances": cat_calls[c],
        }
    top = []
    for (c, name), us in sorted(kernel_us.items(), key=lambda x: x[1], reverse=True)[:80]:
        top.append({
            "category": c, "kernel": name, "cuda_time_ms": us / 1000.0,
            "instances": kernel_calls[(c, name)],
        })
    result = {
        "method": method,
        "profiler": "torch",
        "profile_iterations_requested": profile_iterations,
        "trace_files": [str(p) for p in files],
        "captured_cuda_kernel_time_ms": total / 1000.0,
        "categories": cats,
        "top_kernels": top,
        "interpretation": {
            "main_decode_metric": "Use non-profiled TPOT from the normal benchmark.",
            "kv_quant_store": "Separately launched KV quantization/store/flush kernels.",
            "kv_decode_dequant_fused": (
                "Quantized decode kernels containing BOTH attention/decode and "
                "dequantization/packing. Not pure dequant time."
            ),
            "pure_dequant_time": (
                "Not identifiable from the current fused implementation without "
                "a kernel ablation/split-kernel microbenchmark."
            ),
        },
    }
    (out_dir / "kernel_profile_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "kernel_profile_categories.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category", "cuda_time_ms", "fraction", "cuda_time_ms_per_requested_worker_iteration", "kernel_instances"])
        for c, v in cats.items():
            w.writerow([c, v["cuda_time_ms"], v["fraction_of_captured_cuda_kernel_time"],
                        v["cuda_time_ms_per_requested_worker_iteration"], v["kernel_instances"]])
    with (out_dir / "kernel_profile_top.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["category", "kernel", "cuda_time_ms", "instances"])
        w.writeheader(); w.writerows(top)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fp16","tq4","tq3","kvarn","kvarn44","kvarn64"])
    ap.add_argument("--requests-jsonl", type=Path, required=True,
                    help="Use a validated v7 measure.requests.jsonl; max_tokens:null sentinel should be present.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--number", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--profile-iterations", type=int, default=160,
                    help="GPU worker iterations to capture after one delayed step; 160 usually crosses a 128-token KVarN tile boundary.")
    ap.add_argument("--warmup-number", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--max-seqs", type=int, default=16)
    ap.add_argument("--batch-tokens", type=int, default=1024)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--startup-timeout", type=int, default=1800)
    args = ap.parse_args()

    root = Path(os.environ.get("KVBENCH_ROOT", str(Path.home() / "kvbench-3090-cu126"))).resolve()
    kit = Path(os.environ["KIT"]).resolve()
    model = os.environ["MODEL"]
    serve_env = os.environ.get("KVBENCH_SERVE_ENV", "kvarn-serve")
    eval_env = os.environ.get("KVBENCH_EVAL_ENV", "evalscope-client")
    conda_sh = os.environ["CONDA_SH"]

    args.out.mkdir(parents=True, exist_ok=False)
    profile_dir = (args.out / "torch_profile").resolve()
    profile_dir.mkdir()

    # Validate the null sentinel so EvalScope's own default 2048 cannot refill it.
    lines = [json.loads(x) for x in args.requests_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        raise ValueError("empty requests file")
    for i, body in enumerate(lines[:args.number]):
        if "max_tokens" not in body or body["max_tokens"] is not None:
            raise ValueError(
                f"request {i} must contain JSON max_tokens:null for unbounded transport; got {body.get('max_tokens','<missing>')!r}"
            )

    if port_busy(args.port):
        raise RuntimeError("port is busy")
    ensure_gpu_idle(args.gpu)

    env = os.environ.copy()
    env.update(
        KVBENCH_ROOT=str(root), MODEL=model, PORT=str(args.port),
        CUDA_VISIBLE_DEVICES=args.gpu, APC="0", MAXLEN=str(args.max_len),
        MAXSEQ=str(args.max_seqs), BATCHED_TOKENS=str(args.batch_tokens),
        GPU_UTIL=str(args.gpu_util), KV_BYTES="", PYTHONNOUSERSITE="1",
        KVBENCH_SERVE_ENV=serve_env,
    )
    env.pop("PYTHONHOME", None); env.pop("PYTHONPATH", None)

    profiler_cfg = json.dumps({
        "profiler": "torch",
        "torch_profiler_dir": str(profile_dir),
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_memory": False,
        "torch_profiler_dump_cuda_time_total": True,
        "ignore_frontend": True,
        "delay_iterations": 1,
        "max_iterations": args.profile_iterations,
    }, separators=(",", ":"))

    server_log = (args.out / "server.log").open("w")
    cmd = conda_command(serve_env, [
        "bash", str(kit / "serve.sh"), args.method,
        "--profiler-config", profiler_cfg,
    ])
    (args.out / "server.command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, cwd=kit, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)

    def evalscope_cmd(n: int, out: Path):
        return conda_command(eval_env, [
            "evalscope", "perf",
            "--url", f"http://127.0.0.1:{args.port}/v1/chat/completions",
            "--api", "openai", "--api-key", "EMPTY", "--model", "qwen-kvbench",
            "--dataset", "line_by_line", "--dataset-path", str(args.requests_jsonl),
            "--tokenizer-path", model, "--parallel", str(min(args.concurrency, n)),
            "--number", str(n), "--stream", "--no-test-connection", "--warmup-num", "0",
            "--outputs-dir", str(out),
        ])

    try:
        wait_server(f"http://127.0.0.1:{args.port}", proc, args.startup_timeout)

        # First cover JIT/autotune without profiler.
        warm_out = args.out / "warmup"
        warm_out.mkdir()
        with (args.out / "warmup.log").open("w") as f:
            subprocess.run(evalscope_cmd(args.warmup_number, warm_out), check=True, env=env, stdout=f, stderr=subprocess.STDOUT)

        post(f"http://127.0.0.1:{args.port}/start_profile")
        prof_out = args.out / "profile_run"
        prof_out.mkdir()
        try:
            with (args.out / "profile_run.log").open("w") as f:
                subprocess.run(evalscope_cmd(args.number, prof_out), check=True, env=env, stdout=f, stderr=subprocess.STDOUT)
        finally:
            # stop_profile flush can be slow; wait for it.
            post(f"http://127.0.0.1:{args.port}/stop_profile", timeout=1800)

        result = summarize_profile(profile_dir, args.out, args.method, args.profile_iterations)
        print(json.dumps(result["categories"], indent=2, ensure_ascii=False))
        print(f"saved: {args.out / 'kernel_profile_summary.json'}")
    finally:
        stop_process_group(proc)
        server_log.close()

if __name__ == "__main__":
    main()
