"""Optional Nsight Compute replay: real DRAM counters, separate from full runs.

Requires an already finished residual suite with a passed GPU validation.
Missing ncu/counter permissions produce explicit unavailable (NA) diagnostics;
this optional command never changes full-run scores or its success status.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

from .policy import POLICIES
from .runtime import atomic_json, sha_file, verify_runtime
from .suite import read_object, resolve_environment, validate_gpu_result

METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum")
DEFAULT_POLICIES = ("parent", "repacked_full", "bound_kv")
NCU_DOC = "https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html"


def parse_ncu_csv(text):
    """Parse base-unit raw CSV and reject missing/duplicate/nonfinite metrics."""
    lines = text.splitlines()
    header_at = None
    for i, line in enumerate(lines):
        fields = next(csv.reader([line]), [])
        if "Metric Name" in fields and "Metric Value" in fields and "ID" in fields:
            header_at = i
            break
    if header_at is None:
        raise ValueError("No Nsight Compute metric table was exported")
    groups = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[header_at:]))):
        metric = row.get("Metric Name")
        if metric not in METRICS:
            continue
        key = (row.get("Process ID", ""), row.get("ID", ""))
        if not key[1]:
            raise ValueError("Metric row has no kernel launch ID")
        group = groups.setdefault(key, {"process_id": key[0], "launch_id": key[1],
                                       "kernel_name": row.get("Kernel Name"), "metrics": {}})
        if metric in group["metrics"]:
            raise ValueError("Repeated metric for one kernel launch: " + metric)
        raw = (row.get("Metric Value") or "").replace(",", "").strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError("Unavailable numeric profiler metric: " + metric + "=" + raw) from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid profiler metric: " + metric)
        unit = (row.get("Metric Unit") or "").strip().lower()
        if metric.startswith("dram__"):
            if unit not in {"byte", "bytes"}:
                raise ValueError("Expected unscaled byte unit, got " + unit)
            base_value = value
        else:
            units = {"nsecond": 1e-9, "ns": 1e-9, "nanosecond": 1e-9,
                     "usecond": 1e-6, "us": 1e-6, "µs": 1e-6,
                     "msecond": 1e-3, "ms": 1e-3, "second": 1.0, "s": 1.0}
            if unit not in units:
                raise ValueError("Unknown duration unit: " + unit)
            base_value = value * units[unit]
        group["metrics"][metric] = {"value": value, "unit": unit, "base_value": base_value}
    if not groups or any(set(g["metrics"]) != set(METRICS) for g in groups.values()):
        raise ValueError("Not all requested counters were collected for every profiled launch")
    kernels = list(groups.values())
    return {
        "kernels": kernels,
        "kernel_launches": len(kernels),
        "dram_bytes_read": sum(k["metrics"][METRICS[0]]["base_value"] for k in kernels),
        "dram_bytes_write": sum(k["metrics"][METRICS[1]]["base_value"] for k in kernels),
        "profiled_kernel_duration_s": sum(k["metrics"][METRICS[2]]["base_value"] for k in kernels),
    }


def validate_profile_case(path, suite, policy, shape):
    receipt = read_object(path)
    if (receipt.get("schema") != "kvbench.residual.profile_case.v1"
            or receipt.get("status") != "completed" or receipt.get("gpu_executed") is not True):
        raise ValueError("The profiling workload did not complete on GPU")
    expected = {"policy": policy, "epsilon": suite["epsilon"], "parent_pins": suite["parent_pins"],
                "generated_sha256": suite["generated_sha256"], **shape}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError("Profiling workload receipt mismatch: " + key)
    if Path(receipt.get("source_root", "")).resolve() != Path(suite["kernel_parent_source_root"]).resolve():
        raise ValueError("Profiling workload used a different source root")
    expected_launches = 1 if shape["mode"] == "single" else 2
    if receipt.get("profiled_launches") != expected_launches:
        raise ValueError("Profiling workload launch count does not match the selected attention mode")
    return receipt


def error_class(text, returncode):
    if "ERR_NVGPUCTRPERM" in text or "permission" in text.lower() and "counter" in text.lower():
        return "counter_permission_unavailable"
    if "Failed to find metric" in text or "Unknown metric" in text or "not supported" in text.lower():
        return "metric_or_device_unsupported"
    if returncode:
        return "profiler_or_replay_failed"
    return "metrics_or_workload_receipt_unavailable"


def gpu_idle(cfg, env):
    result = subprocess.run(["nvidia-smi", "-i", str(cfg["gpu"]), "--query-compute-apps=pid,process_name",
                             "--format=csv,noheader,nounits"], env=env, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError("Cannot verify target GPU is free: " + result.stderr[-1000:])
    if result.stdout.strip() and "No running processes found" not in result.stdout:
        raise RuntimeError("Target GPU is occupied; no process was stopped: " + result.stdout.strip())
    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex(("127.0.0.1", int(cfg["port"]))) == 0:
            raise RuntimeError("The experiment port is occupied; no existing service was stopped")


def run_owned(command, cwd, env, logfile):
    with Path(logfile).open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    return proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("Nsight Compute replay is still running; log: " + str(logfile), flush=True)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
            raise


def write_report(out, report):
    atomic_json(out / "dram_profile.json", report)
    fields = ["policy", "status", "reason", "dram_bytes_read", "dram_bytes_write", "profiled_kernel_duration_s",
              "kernel_launches", "profiler_exit_code", "raw_csv"]
    with (out / "dram_profile.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["policies"])
    lines = ["# 独立 attention 重放的真实 DRAM 计数", "", "状态：`" + report["status"] + "`。", "",
             "本结果来自 Nsight Compute 性能计数器，独立于 microbench 的逻辑字节估计。",
             "范围是固定 B/L 的一次 attention 路径（split 模式含 combine）；不含量化写入、rotation、整层、HTTP 或完整生成。",
             "`cache-control=all` 每次 profiler replay 前清缓存，`clock-control=none` 不请求调整时钟。",
             "性能计数器会重放并影响执行时间；这里的 duration 不能替代正常 CUDA-event 微基准或 TPOT。", "",
             "| 策略 | 状态 | DRAM 读 bytes | DRAM 写 bytes | profiler kernel 秒 | 原始 CSV |",
             "|---|---|---:|---:|---:|---|"]
    def display(value):
        return "NA" if value is None else str(value)
    for row in report["policies"]:
        lines.append("| " + " | ".join(display(row.get(k)) for k in
                     ["policy", "status", "dram_bytes_read", "dram_bytes_write", "profiled_kernel_duration_s", "raw_csv"]) + " |")
        if row.get("reason"):
            lines.extend(["", row["policy"] + "：" + row["reason"]])
    lines += ["", "缺少 ncu、指标不支持、计数器权限不足或无有效导出时保留 NA，不能解释为 0 字节。",
              "此独立结果不修改已有完整评测状态。", "", "官方 CLI 说明：" + NCU_DOC, ""]
    (out / "dram_profile.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True, help="Finished residual suite.json with passed GPU correctness")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(DEFAULT_POLICIES))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--mode", choices=["single", "split"], default="split")
    parser.add_argument("--block-n", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--splits", type=int, default=32)
    args = parser.parse_args(argv)
    if min(args.batch, args.length, args.splits) < 1 or len(set(args.policies)) != len(args.policies) or args.splits & (args.splits - 1):
        raise ValueError("Use positive replay dimensions, power-of-two splits and unique policies")
    if args.mode == "single":
        args.splits = 1
    suite_path = args.suite.expanduser().resolve(strict=True)
    meta = read_object(suite_path)
    if meta.get("schema") != "kvbench.residual.full_suite.v1" or not meta.get("finished"):
        raise ValueError("Use a finished residual suite; profiling is not started while its full runs are active")
    cfg = meta["settings"]
    repo = Path(__file__).resolve().parents[1]
    if Path(cfg["repo"]).resolve() != repo:
        raise ValueError("Run profiling from the repository recorded in the residual suite")
    validation = meta.get("validation", {})
    if validation.get("passed") is not True:
        raise ValueError("The residual suite has no passed GPU correctness receipt")
    validation_path = Path(validation["path"])
    if sha_file(validation_path) != validation["sha256"]:
        raise ValueError("GPU validation receipt changed")
    validate_gpu_result(validation_path, meta["kernel_parent_source_root"], meta["parent_pins"],
                        meta["generated_sha256"], meta["epsilon"])
    for rel, expected in meta["new_source_sha256"].items():
        if sha_file(repo / rel) != expected:
            raise ValueError("Experiment implementation changed after full-run freeze: " + rel)
    for rel, expected in meta["parent_pins"].items():
        if sha_file(Path(meta["kernel_parent_source_root"]) / rel) != expected:
            raise ValueError("Installed parent source changed: " + rel)
    for policy in args.policies:
        frozen = verify_runtime(meta["runtimes"][policy], policy, full=True, epsilon=meta["epsilon"])
        if frozen["generated_sha256"] != meta["generated_sha256"][policy]:
            raise ValueError("Frozen runtime differs from the validated source: " + policy)
    suffix = "dram_{}_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"), os.getpid())
    out = (args.out.expanduser() if args.out else suite_path.parent / "profiles" / suffix).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "kvbench.residual.dram_profile.v1", "status": "unavailable", "source_suite": str(suite_path),
              "suite_sha256": sha_file(suite_path), "epsilon": meta["epsilon"], "batch": args.batch,
              "length": args.length, "mode": args.mode, "block_n": args.block_n, "splits": args.splits,
              "metrics": METRICS, "cache_control": "all", "clock_control": "none", "policies": [],
              "measurement_scope": "Hardware DRAM counters for one synthetic attention replay; not formal inference traffic or uninstrumented latency."}
    server = resolve_environment(cfg, "server")
    ncu = shutil.which("ncu", path=server["env"].get("PATH"))
    report["ncu"] = ncu
    for policy in args.policies:
        report["policies"].append({"policy": policy, "status": "unavailable", "reason": "not_started",
                                    "dram_bytes_read": None, "dram_bytes_write": None,
                                    "profiled_kernel_duration_s": None, "kernel_launches": None,
                                    "profiler_exit_code": None, "raw_csv": None})
    if not ncu:
        for row in report["policies"]:
            row["reason"] = "ncu_not_found_in_server_environment"
        write_report(out, report)
        print("NCU unavailable: recorded NA, no GPU work started. " + str(out / "dram_profile.md"))
        return 0
    version = subprocess.run([ncu, "--version"], capture_output=True, text=True, env=server["env"], timeout=30)
    report["ncu_version"] = {"returncode": version.returncode, "stdout": version.stdout, "stderr": version.stderr}
    env = server["env"].copy()
    env.update(CUDA_VISIBLE_DEVICES=str(cfg["gpu"]), PYTHONUNBUFFERED="1",
               TRITON_CACHE_DIR=str(out / "cache/triton"), TORCHINDUCTOR_CACHE_DIR=str(out / "cache/torchinductor"))
    try:
        for row in report["policies"]:
            gpu_idle(cfg, env)
            dest = out / row["policy"]
            dest.mkdir()
            csvpath, logpath = dest / "ncu.raw.csv", dest / "profiler.log"
            replay_out = dest / "replay"
            command = [ncu, "--target-processes", "all", "--profile-from-start", "off",
                       "--metrics", ",".join(METRICS), "--csv", "--page", "raw",
                       "--print-units", "base", "--print-metric-name", "name",
                       "--cache-control", "all", "--clock-control", "none", "--log-file", str(csvpath),
                       server["python"], "-m", "kvbench_residual.microbench", "--source-root", meta["kernel_parent_source_root"],
                       "--epsilon", str(meta["epsilon"]), "--out", str(replay_out), "--device", "cuda:0",
                       "--profile-case", "--batch", str(args.batch), "--length", str(args.length),
                       "--policy", row["policy"], "--mode", args.mode, "--block-n", str(args.block_n),
                       "--splits", str(args.splits)]
            row.update(status="profiling", reason="", command=command, raw_csv=str(csvpath), replay_output=str(replay_out))
            write_report(out, report)
            code = run_owned(command, repo, env, logpath)
            row["profiler_exit_code"] = code
            raw = csvpath.read_text(encoding="utf-8", errors="replace") if csvpath.exists() else ""
            logs = logpath.read_text(encoding="utf-8", errors="replace")
            try:
                if code:
                    raise ValueError("Profiler exited with code " + str(code))
                receipt_path = replay_out / "profile_case.json"
                receipt = validate_profile_case(receipt_path, meta, row["policy"], {
                    "batch": args.batch, "length": args.length, "mode": args.mode,
                    "block_n": args.block_n, "splits": args.splits})
                counters = parse_ncu_csv(raw)
                if counters["kernel_launches"] != receipt["profiled_launches"]:
                    raise ValueError("Collected kernel count differs from the profiled attention workload receipt")
                row.update(counters, status="measured", reason="", raw_csv_sha256=sha_file(csvpath),
                           profile_case_path=str(receipt_path), profile_case_sha256=sha_file(receipt_path))
            except (ValueError, OSError, KeyError) as exc:
                row.update(status="unavailable", reason=error_class(raw + "\n" + logs, code) + ": " + str(exc))
            write_report(out, report)
        report["status"] = "measured" if all(r["status"] == "measured" for r in report["policies"]) else "partial" if any(r["status"] == "measured" for r in report["policies"]) else "unavailable"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "stopped"
        report["error"] = str(exc)
        for row in report["policies"]:
            if row["status"] == "profiling":
                row.update(status="unavailable", reason="profiler_interrupted_or_failed")
        write_report(out, report)
        raise
    write_report(out, report)
    print(json.dumps({"status": report["status"], "readout": str(out / "dram_profile.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
