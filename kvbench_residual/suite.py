"""Queue after the explicit V3 suite, then validate and run six K4V4 residual policies."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

from kvbench_longcontext.common import atomic_json, checked_data, sha_file
from .policy import DEFAULT_EPSILON, POLICIES, POLICY_DESCRIPTIONS, method_name, validate_epsilon


TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "passage_retrieval_en"]
OLD_METHODS = ["fp16", "fp8", "tq4", "tq34", "tq3", "kvarn", "kvarn44", "kvarn64", "kvarn4464"]
METHODS = [
    {"method": method_name(policy), "base_method": "kvarn44", "precision": "k4v4", "policy": policy,
     "description": POLICY_DESCRIPTIONS[policy]} for policy in POLICIES
]
PROTOCOL = {
    "concurrency": 8, "repeats": 1, "max_model_len": 32768, "max_seqs": 16,
    "batch_tokens": 1024, "gpu_util": 0.85, "thinking": True,
    "generation_budget_policy": "remaining-context", "max_tokens": None,
    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0,
    "request_timeout": 0, "data_seed": 42, "request_seed": 20261009,
    "server_seed": 20260917,
}


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def upstream_identity(meta, path):
    if meta.get("schema") != "kvbench.longcontext.full_suite.v1":
        raise ValueError("--after-suite must be a V3 nine-method suite.json")
    cfg = meta.get("settings", {})
    if Path(cfg.get("batch", "")).resolve() != Path(path).parent.resolve():
        raise ValueError("The upstream suite's batch path does not match --after-suite")
    if cfg.get("methods") != OLD_METHODS or cfg.get("tasks") != TASKS:
        raise ValueError("Expected the original nine-method, six-task V3 suite")
    if not meta.get("started"):
        raise ValueError("Upstream suite has no stable started identity")
    return meta["started"], str(Path(cfg["batch"]).resolve())


def upstream_complete(meta):
    """A gap between two GPU jobs is not a completed suite."""
    if "finished" not in meta or "exit_code" not in meta:
        return False
    code = meta["exit_code"]
    if isinstance(code, bool) or not isinstance(code, int) or not meta["finished"]:
        raise ValueError("Invalid upstream finished/exit_code fields")
    if code != 0:
        raise RuntimeError(
            f"The upstream V3 suite finished with exit_code={code}. New GPU work was not started. "
            "Inspect its suite_status.md and repair or complete that batch first."
        )
    rows = meta.get("methods", [])
    if [r.get("method") for r in rows] != OLD_METHODS or any(r.get("state") != "completed" for r in rows):
        raise ValueError("Upstream reports success but not all nine methods are completed")
    return True


def validate_protocol(meta, manifest, datasets):
    cfg = meta["settings"]
    mismatches = {k: {"expected": v, "actual": cfg.get(k)} for k, v in PROTOCOL.items() if cfg.get(k) != v}
    if mismatches:
        raise ValueError("Upstream settings differ from the supported C8 once protocol: " + json.dumps(mismatches))
    expected = {"thinking": True, "sample_size": 0, "min_input_tokens": 0,
                "max_input_tokens": None, "max_new_tokens": None, "max_model_len": 32768,
                "overflow": "skip", "generation_budget_policy": "remaining-context", "data_seed": 42}
    settings = manifest.get("settings", {})
    mismatch = {k: {"expected": v, "actual": settings.get(k)} for k, v in expected.items() if settings.get(k) != v}
    if mismatch:
        raise ValueError("Frozen data do not match the full null-budget cohort: " + json.dumps(mismatch))
    if (manifest.get("variant") != cfg.get("variant")
            or manifest.get("max_tokens_transport") != "explicit-json-null"
            or set(datasets) != set(TASKS)):
        raise ValueError("Data variant, transport, or task set differs from the upstream suite")
    for task in TASKS:
        info = manifest["datasets"][task]
        if (info["selected_count"] != info["eligible_count"] or info["max_tokens"] is not None
                or info["truncated_count"] != 0 or len(datasets[task]) < cfg["concurrency"]):
            raise ValueError("Expected all eligible untruncated cases and at least C8 cases: " + task)


def validate_gpu_result(path, source_root=None, parent_pins=None, policy_sha=None, epsilon=DEFAULT_EPSILON):
    data = read_object(path)
    if (data.get("schema") != "kvbench.residual.validation.v1"
            or data.get("status") != "passed" or data.get("passed") is not True
            or data.get("gpu_executed") is not True
            or set(data.get("tested_policies", [])) != set(POLICIES)
            or set(data.get("tested_value_bits", [])) != {4}):
        raise RuntimeError("GPU correctness did not pass all six K4V4 policies: " + str(path))
    required = ["autotune_executed", "cuda_graph_replay_executed", "graph_updated_input_checked",
                "graph_updated_demand_checked", "packed_consumers_checked", "packed_writers_checked"]
    if (any(data.get(key) is not True for key in required)
            or set(data.get("tested_groups", [])) != {64, 128}
            or set(data.get("tested_modes", [])) != {"single", "split"}
            or set(data.get("tested_block_ns", [])) != {16, 32, 64}):
        raise RuntimeError("GPU correctness receipt is missing writer/reader, dynamic graph or shape coverage")
    if data.get("epsilon") != validate_epsilon(epsilon):
        raise RuntimeError("GPU validation used a different residual error budget")
    if source_root is not None and Path(data.get("source_root", "")).resolve() != Path(source_root).resolve():
        raise RuntimeError("GPU validation used a different parent source root")
    if parent_pins is not None and data.get("parent_pins") != parent_pins:
        raise RuntimeError("GPU validation used different parent writers/readers")
    if policy_sha is not None and data.get("generated_sha256") != policy_sha:
        raise RuntimeError("GPU validation's generated writer/reader policies differ from the isolated runtimes")
    return data


def validate_microbench_result(path, source_root, parent_pins, policy_sha, epsilon):
    data = read_object(path)
    if (data.get("schema") != "kvbench.residual.microbenchmark.v1"
            or data.get("status") != "completed" or data.get("passed") is not True
            or data.get("gpu_executed") is not True
            or Path(data.get("source_root", "")).resolve() != Path(source_root).resolve()
            or data.get("parent_pins") != parent_pins or data.get("generated_sha256") != policy_sha
            or data.get("epsilon") != epsilon):
        raise RuntimeError("Microbenchmark did not complete on GPU for the same validated writers/readers")
    expected = {(b, length, mode, policy) for b in (1, 8) for length in (1024, 4096, 8192, 16384, 32768)
                for mode in ("single", "split") for policy in POLICIES}
    rows = data.get("rows", [])
    actual = {(r.get("batch"), r.get("context_tokens"), r.get("mode"), r.get("policy")) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise RuntimeError("Microbenchmark is not the full fixed 120-row matrix")
    for row in rows:
        latency = row.get("latency_us_median")
        if (row.get("group") != 128 or row.get("value_bits") != 4
                or row.get("fixture_kind") != "random"
                or not isinstance(latency, (int, float)) or isinstance(latency, bool)
                or not math.isfinite(latency) or latency <= 0):
            raise RuntimeError("Microbenchmark has missing/invalid timing or the wrong precision/fixture")
    packers = data.get("writer_packing", [])
    if len(packers) != 2 or {r.get("policy") for r in packers} != {"native_int4", "tile_high2_low2"}:
        raise RuntimeError("Microbenchmark is missing separate packing-cost measurements")
    # No latency ratio, bandwidth reduction or quality-result threshold is used.
    return data


def resolve_environment(cfg, role):
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise RuntimeError("Conda was not found; launch this script from an initialized conda shell")
    query = subprocess.run([conda, "info", "--base"], check=True, capture_output=True, text=True)
    env = os.environ.copy()
    env.update(AR_CONDA_SH=str(Path(query.stdout.strip()) / "etc/profile.d/conda.sh"),
               AR_ROLE=role, AR_ENV_NAME=cfg[role + "_env"], KIT=cfg["kit"],
               KVBENCH_ROOT=cfg["root"], MODEL=cfg["model"])
    shell = '''
set -e
source "$AR_CONDA_SH"
conda activate "$AR_ENV_NAME"
if [ "$AR_ROLE" = server ]; then source "$KIT/env.sh"; fi
export MODEL PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
python -c 'import os,json,sys; print("AR_ENV_JSON="+json.dumps({"python":sys.executable,"env":dict(os.environ)}))'
'''
    result = subprocess.run(["bash", "--noprofile", "--norc", "-c", shell], env=env, capture_output=True, text=True)
    lines = [line[len("AR_ENV_JSON="):] for line in result.stdout.splitlines() if line.startswith("AR_ENV_JSON=")]
    if result.returncode or len(lines) != 1:
        raise RuntimeError("Cannot initialize " + role + " environment. Check conda and KIT/env.sh.\n" + result.stderr[-4000:])
    runtime = json.loads(lines[0])
    if Path(runtime["env"].get("MODEL", "")).resolve() != Path(cfg["model"]).resolve():
        raise ValueError("Environment setup changed the frozen MODEL path")
    runtime["env"].update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
    return runtime


class CleanupFailure(RuntimeError):
    pass


class Suite:
    def __init__(self, after_suite, out=None, epsilon=DEFAULT_EPSILON):
        self.epsilon = validate_epsilon(epsilon)
        supplied = Path(after_suite).expanduser()
        if not supplied.is_absolute():
            raise ValueError("--after-suite must be an absolute path to the running V3 suite.json")
        self.after_suite = supplied.resolve(strict=True)
        initial = read_object(self.after_suite)
        self.identity = upstream_identity(initial, self.after_suite)
        self.repo = Path(__file__).resolve().parents[1]
        if Path(initial["settings"]["repo"]).resolve() != self.repo:
            raise ValueError("Run this patch from the same repository recorded in the upstream V3 suite")
        suffix = "full_{}_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"), os.getpid())
        self.out = (Path(out).expanduser() if out else Path(initial["settings"]["root"]) / "results/residual_v1" / suffix).resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "logs").mkdir()
        self.cfg = None
        self.client = self.server = None
        self.server_proc = self.step_proc = None
        self.server_log = None
        self.cleaning = False
        self.cancelled = False
        self.source_hashes = {}
        self.meta = {"schema": "kvbench.residual.full_suite.v1", "started": timestamp(), "phase": "waiting_for_v3",
                     "after_suite": str(self.after_suite), "output": str(self.out), "epsilon": self.epsilon, "fatal_error": None,
                     "comparison_exit_code": None, "effects_exit_code": None, "methods": [
                         {**m, "state": "pending", "planned_points": len(TASKS), "expected_tasks": None,
                          "measurement_complete_points": 0, "evaluation_complete_points": 0,
                          "valid_responses": 0, "scored_tasks": 0, "length_capped": 0,
                          "budget_inconsistent_points": 0, "run_exit_code": None,
                          "score_exit_code": None, "report_exit_code": None,
                          "implementation_verified": False,
                          "run_dir": str(self.out / "runs" / m["method"]), "error": ""} for m in METHODS]}
        self.save()

    def save(self):
        self.meta["updated"] = timestamp()
        atomic_json(self.out / "suite.json", self.meta)
        lines = ["# AR-KV 全量实验状态", "", "阶段：`" + self.meta["phase"] + "`", "",
                 "上游批次：`" + str(self.after_suite) + "`", "",
                 "| 方法 | 状态 | 实现证据 | 测量完整点 | 评分完整点 | 有效/应有 | 已评分 | 截断 | 预算异常点 |",
                 "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in self.meta["methods"]:
            lines.append("| {method} | {state} | {implementation_verified} | {measurement_complete_points}/{planned_points} | "
                         "{evaluation_complete_points}/{planned_points} | {valid_responses}/{expected_tasks} | "
                         "{scored_tasks} | {length_capped} | {budget_inconsistent_points} |".format(**row))
        lines += ["", "总览步骤 exit=" + str(self.meta.get("comparison_exit_code"))
                  + "；差值步骤 exit=" + str(self.meta.get("effects_exit_code")) + "（null/None 表示尚未执行，不是成功）。",
                  "全部正式请求保持 thinking、C8、相同采样、max_tokens:null；完整截断响应仍评分。",
                  "GPU 正确性通过后才启动全量实验；微基准速度不决定保留哪种策略。",
                  "六组共享空 mask 数值保护；parent 保留原布局/算式但不是旧源字节副本。repacked_full 验证拆分表示本身。",
                  "当前全量仅支持 K4V4；error budget 不等于任务准确率保证。正式流量计数为 NA，诊断 logical bytes 不是真实 DRAM 事务。",
                  "先看 validation/validation.json，再看 microbench/benchmark.json 和 comparison/overview.md。",
                  "等待旧批次时不会加载模型、调用 GPU 核函数或停止其他进程。"]
        if self.meta["fatal_error"]:
            lines += ["", "错误：" + self.meta["fatal_error"]]
        (self.out / "suite_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def log(self, message):
        line = "[{}] {}".format(timestamp(), message)
        print(line, flush=True)
        with (self.out / "suite.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def phase(self, value):
        self.meta["phase"] = value
        self.save()

    def on_signal(self, signum, frame):
        self.cancelled = True
        if not self.cleaning:
            raise KeyboardInterrupt

    def wait_for_upstream(self):
        self.log("等待明确的旧批次 finished/exit_code；方法间 GPU 空闲不会触发新实验。")
        last = 0
        while True:
            meta = read_object(self.after_suite)
            if upstream_identity(meta, self.after_suite) != self.identity:
                raise ValueError("The upstream suite identity changed while waiting")
            if upstream_complete(meta):
                self.log("旧九方法批次成功结束；开始只读冻结校验。")
                return meta
            if time.monotonic() - last >= 60:
                active = [r["method"] + ":" + r["state"] for r in meta.get("methods", []) if r.get("state") in ("starting", "running")]
                self.log("仍在等待旧批次；当前活动方法：" + (", ".join(active) or "暂无，等待调度或收尾完成"))
                last = time.monotonic()
            time.sleep(5)

    def freeze_upstream(self, upstream):
        self.cfg = dict(upstream["settings"])
        data_dir = Path(self.cfg["data"]).resolve()
        if data_dir != self.after_suite.parent / "data":
            raise ValueError("Expected the upstream suite's own frozen data directory")
        if sha_file(data_dir / "manifest.json") != upstream.get("data_manifest_sha256"):
            raise ValueError("Upstream frozen data manifest hash changed")
        manifest, datasets = checked_data(data_dir)
        validate_protocol(upstream, manifest, datasets)
        if not Path(self.cfg["model"]).is_dir():
            raise ValueError("The upstream local MODEL directory is no longer available")
        self.meta.update(settings={**self.cfg, "batch": str(self.out), "methods": [m["method"] for m in METHODS],
                                   "epsilon": self.epsilon}, upstream_settings=self.cfg,
                         data_dir=str(data_dir), data_manifest_sha256=upstream["data_manifest_sha256"],
                         upstream_exit_code=upstream["exit_code"], upstream_finished=upstream["finished"])
        inventory = [self.after_suite, data_dir / "manifest.json"]
        inventory += [data_dir / info["file"] for info in manifest["datasets"].values()]
        dest = self.out / "upstream"
        dest.mkdir()
        shutil.copy2(self.after_suite, dest / "suite.json")
        shutil.copy2(data_dir / "manifest.json", dest / "data.manifest.json")
        old_runs = []
        for item in upstream["methods"]:
            folder = Path(item["run_dir"]).resolve()
            run = read_object(folder / "run.json")
            if run.get("method") != item["method"] or run.get("data_manifest_sha256") != upstream["data_manifest_sha256"]:
                raise ValueError("Upstream method identity or data hash mismatch: " + item["method"])
            for name, expected_sha in run.get("scripts_sha256", {}).items():
                local = self.repo / "kvbench_longcontext" / name
                if Path(name).name != name or not local.is_file() or sha_file(local) != expected_sha:
                    raise ValueError("Current V3 source differs from the completed upstream run: " + name)
            record_dest = dest / "runs" / item["method"]
            record_dest.mkdir(parents=True)
            for name in ["run.json", "server.launch.json", "grading_protocol.json"]:
                path = folder / name
                if path.is_file():
                    shutil.copy2(path, record_dest / name)
                    inventory.append(path)
            old_runs.append(str(folder))
        self.source_hashes = {str(path): sha_file(path) for path in inventory}
        atomic_json(dest / "source_hashes.json", self.source_hashes)
        self.meta["upstream_runs"] = old_runs
        self.meta["new_source_sha256"] = {str(p.relative_to(self.repo)): sha_file(p)
                                             for p in (self.repo / "kvbench_residual").rglob("*.py")}
        total = sum(len(rows) for rows in datasets.values())
        self.meta["data_coverage"] = [{"dataset": name, **{k: manifest["datasets"][name][k] for k in
                                      ["source_count", "eligible_count", "selected_count", "excluded_count", "truncated_count"]}} for name in TASKS]
        with (self.out / "data_coverage.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.meta["data_coverage"][0]))
            writer.writeheader()
            writer.writerows(self.meta["data_coverage"])
        for row in self.meta["methods"]:
            row["expected_tasks"] = total
        self.save()
        self.log(f"复用同一冻结数据：每组 {total} 题，新增 6 组共 {6 * total} 条正式请求；不重新下载或抽样。")

    def assert_upstream_unchanged(self):
        for path, expected in self.source_hashes.items():
            if not Path(path).is_file() or sha_file(path) != expected:
                raise ValueError("Read-only upstream artifact changed: " + path)
        for relative, expected in self.meta.get("new_source_sha256", {}).items():
            if sha_file(self.repo / relative) != expected:
                raise ValueError("The new experiment source changed during this batch: " + relative)

    def call(self, runtime, args, label):
        logpath = self.out / "logs" / (label + ".log")
        self.log("开始 " + label + "；日志 " + str(logpath))
        with logpath.open("w", encoding="utf-8") as out:
            proc = subprocess.Popen([runtime["python"], *map(str, args)], cwd=self.repo, env=runtime["env"],
                                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            self.step_proc = proc
            while True:
                try:
                    code = proc.wait(timeout=60)
                    self.step_proc = None
                    self.log(f"结束 {label}；exit={code}")
                    return code
                except subprocess.TimeoutExpired:
                    self.log(label + " 仍在运行；不施加正式生成长度或时间上限。")

    def require(self, runtime, args, label):
        if self.call(runtime, args, label) != 0:
            raise RuntimeError(label + " failed; inspect logs/" + label + ".log")

    def preflight(self):
        self.client = resolve_environment(self.cfg, "client")
        self.server = resolve_environment(self.cfg, "server")
        self.meta["python"] = {"client": self.client["python"], "server": self.server["python"]}
        self.save()
        check = "from kvbench_longcontext.scoring import check_dependencies; import kvbench_longcontext; assert kvbench_longcontext.__version__ == '3.0.0'; check_dependencies({'qa_f1','retrieval','rouge_l'}); print('V3 client scoring dependencies verified')"
        self.require(self.client, ["-c", check], "preflight_client")
        self.require(self.server, ["-c", "from kvbench_residual.serve import validate_architecture; import sys; print(validate_architecture(sys.argv[1]))",
                                   self.cfg["model"]], "preflight_architecture")
        self.require(self.client, ["-m", "kvbench_longcontext.run", "--data-dir", self.cfg["data"],
                     "--out", self.out / "plan_only", "--concurrency", "8", "--repeats", "1",
                     "--request-timeout", "0", "--dry-run"], "preflight_plan")

    def gpu_idle(self):
        self.assert_upstream_unchanged()
        result = subprocess.run(["nvidia-smi", "-i", str(self.cfg["gpu"]), "--query-compute-apps=pid,process_name",
                                 "--format=csv,noheader,nounits"], env=self.server["env"], capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise RuntimeError("Cannot verify that the target GPU is free: " + result.stderr[-1000:])
        if result.stdout.strip() and "No running processes found" not in result.stdout:
            raise RuntimeError("The target GPU is occupied; no process was stopped: " + result.stdout.strip())
        with socket.socket() as sock:
            sock.settimeout(2)
            if sock.connect_ex(("127.0.0.1", int(self.cfg["port"]))) == 0:
                raise RuntimeError("The target port is occupied; no existing service was stopped")

    def build_runtimes(self):
        self.runtimes = {}
        self.policy_hashes = {}
        sources = []
        for policy in POLICIES:
            dest = self.out / "runtimes" / policy
            self.require(self.server, ["-m", "kvbench_residual.runtime", "--out", dest, "--policy", policy,
                                       "--epsilon", str(self.epsilon)], "runtime_" + policy)
            info = read_object(dest / "runtime.json")
            source = Path(info["parent_source"]).resolve(strict=True)
            source_sha = sha_file(source)
            if (info.get("schema") != "kvbench.residual.runtime.v1" or info.get("policy") != policy
                    or Path(info["runtime_root"]).resolve() != dest.resolve()
                    or info.get("parent_source_sha256") != source_sha
                    or sha_file(info["kernel_path"]) != info.get("executed_source_sha256")
                    or not isinstance(info.get("transformed_source_sha256"), str)):
                raise ValueError("The isolated runtime manifest does not match its source and policy")
            self.source_hashes[str(dest / "runtime.json")] = sha_file(dest / "runtime.json")
            self.source_hashes[str(Path(info["kernel_path"]).resolve())] = info["executed_source_sha256"]
            sources.append((str(source), source_sha))
            self.runtimes[policy] = str(dest)
            self.policy_hashes[policy] = info["generated_sha256"]
            if info.get("epsilon") != self.epsilon:
                raise ValueError("Isolated runtime error budget does not match this suite")
            for relative, digest in info["parent_pins"].items():
                self.source_hashes[str(Path(info["parent_package"]) / relative)] = digest
            for relative, digest in info["executed_files_sha256"].items():
                self.source_hashes[str(dest / "vllm" / relative)] = digest
            sources[-1] += (str(Path(info["parent_package"]).resolve()), json.dumps(info["parent_pins"], sort_keys=True))
        if len(set(sources)) != 1:
            raise ValueError("All isolated policies must be built from exactly the same parent source")
        self.kernel_source, source_sha, self.kernel_source_root, parent_pins = sources[0]
        self.parent_pins = json.loads(parent_pins)
        self.source_hashes[self.kernel_source] = source_sha
        atomic_json(self.out / "upstream/source_hashes.json", self.source_hashes)
        self.meta.update(runtimes=self.runtimes, kernel_parent_source=self.kernel_source, kernel_parent_source_sha256=source_sha,
                         kernel_parent_source_root=self.kernel_source_root, parent_pins=self.parent_pins,
                         generated_sha256=self.policy_hashes)
        self.save()

    def kernel_checks(self):
        runtime = {"python": self.server["python"], "env": self.server["env"].copy()}
        cache = self.out / "kernel_cache"
        runtime["env"].update(CUDA_VISIBLE_DEVICES=str(self.cfg["gpu"]),
                              TRITON_CACHE_DIR=str(cache / "triton"),
                              TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
                              VLLM_CACHE_ROOT=str(cache / "vllm"))
        common = ["--source-root", self.kernel_source_root, "--device", "cuda:0", "--epsilon", str(self.epsilon)]
        self.phase("gpu_correctness")
        self.gpu_idle()
        self.require(runtime, ["-m", "kvbench_residual.validate", *common, "--out", self.out / "validation"], "gpu_validate")
        checked = validate_gpu_result(self.out / "validation/validation.json", self.kernel_source_root,
                                      self.parent_pins, self.policy_hashes, self.epsilon)
        self.meta["validation"] = {"path": str(self.out / "validation/validation.json"),
                                   "sha256": sha_file(self.out / "validation/validation.json"), "passed": True,
                                   "tested_policies": checked["tested_policies"], "tested_value_bits": checked["tested_value_bits"]}
        self.save()
        self.phase("gpu_microbench")
        self.gpu_idle()
        self.require(runtime, ["-m", "kvbench_residual.microbench", *common, "--out", self.out / "microbench"], "gpu_microbench")
        path = self.out / "microbench/benchmark.json"
        validate_microbench_result(path, self.kernel_source_root, self.parent_pins, self.policy_hashes, self.epsilon)
        self.meta["microbench"] = {"path": str(path), "sha256": sha_file(path), "speed_threshold_used": False}
        self.save()

    def start_server(self, row):
        self.gpu_idle()
        dest = self.out / "servers" / row["method"]
        args = ["-m", "kvbench_residual.serve", "--model", self.cfg["model"], "--method", row["base_method"],
                "--policy", row["policy"], "--epsilon", str(self.epsilon), "--runtime", self.runtimes[row["policy"]], "--out", str(dest),
                "--port", str(self.cfg["port"]), "--gpu", str(self.cfg["gpu"]), "--max-model-len", "32768",
                "--max-seqs", "16", "--batch-tokens", "1024", "--gpu-util", "0.85",
                "--server-seed", str(self.cfg["server_seed"]), "--startup-timeout", str(self.cfg["startup_timeout"])]
        self.server_log = (self.out / "logs" / (row["method"] + ".launcher.log")).open("w", encoding="utf-8")
        self.server_proc = subprocess.Popen([self.server["python"], *args], cwd=self.repo, env=self.server["env"],
                                            stdout=self.server_log, stderr=subprocess.STDOUT, start_new_session=True)
        path = dest / "launch.json"
        deadline = time.monotonic() + self.cfg["startup_timeout"] + 180
        last = time.monotonic()
        while time.monotonic() < deadline:
            if self.server_proc.poll() is not None:
                raise RuntimeError("Launcher exited; inspect logs/" + row["method"] + ".launcher.log and servers/*/server.log")
            if path.is_file():
                launch = read_object(path)
                if launch.get("native_verified") is True:
                    if launch.get("config", {}).get("method") != row["method"]:
                        raise ValueError("The residual launch method does not match this planned policy")
                    self.log(row["method"] + " 服务已就绪。")
                    return path
            if time.monotonic() - last >= 60:
                self.log(row["method"] + " 正在启动或编译；等待本次服务。")
                last = time.monotonic()
            time.sleep(1)
        raise TimeoutError("Residual service startup timed out")

    def stop_server(self):
        proc = self.server_proc
        if proc is not None and proc.poll() is None:
            self.log(f"停止本批次自己的 launcher PID={proc.pid}，等待其清理引擎。")
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired as exc:
                raise CleanupFailure("Own launcher did not finish cleanup; no further methods will start") from exc
        self.server_proc = None
        if self.server_log is not None:
            self.server_log.close()
            self.server_log = None

    def cleanup(self):
        self.cleaning = True
        try:
            self.stop_server()
            if self.step_proc is not None and self.step_proc.poll() is None:
                try:
                    self.step_proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.step_proc.terminate()
                    try:
                        self.step_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.step_proc.kill()
                        self.step_proc.wait(timeout=10)
            self.step_proc = None
        finally:
            self.cleaning = False

    def inspect_results(self, row):
        folder = Path(row["run_dir"])
        run = read_object(folder / "run.json")
        if run.get("method") != row["method"] or run.get("data_manifest_sha256") != self.meta["data_manifest_sha256"]:
            raise ValueError("Run identity or frozen data hash differs")
        wanted = {(task, 1, 8) for task in TASKS}
        if {(p["dataset"], p["repeat"], p["concurrency"]) for p in run["points"]} != wanted or len(run["points"]) != len(TASKS):
            raise ValueError("Run does not contain exactly the six planned full-data C8 points")
        for key in ["measurement_complete_points", "evaluation_complete_points", "valid_responses", "scored_tasks", "length_capped", "budget_inconsistent_points"]:
            row[key] = 0
        for point in run["points"]:
            dest = folder / point["id"]
            perf = read_object(dest / "performance.summary.json") if (dest / "performance.summary.json").is_file() else {}
            quality = read_object(dest / "scores.summary.json") if (dest / "scores.summary.json").is_file() else {}
            row["measurement_complete_points"] += perf.get("measurement_complete") is True
            row["evaluation_complete_points"] += quality.get("evaluation_complete") is True
            row["budget_inconsistent_points"] += perf.get("generation_budget_consistent") is False
            row["valid_responses"] += perf.get("valid_responses", 0) or 0
            row["length_capped"] += perf.get("length_capped", 0) or 0
            row["scored_tasks"] += quality.get("scored_tasks", 0) or 0

    def implementation_evidence(self, row):
        from .trace import summarize
        folder = self.out / "servers" / row["method"]
        path = folder / "dispatch_summary.json"
        saved = read_object(path)
        runtime = read_object(Path(self.runtimes[row["policy"]]) / "runtime.json")
        current = summarize(folder / "dispatch", row["policy"], runtime["executed_source_sha256"], self.epsilon)
        fields = ["source_verified", "fused_launch_observed", "decode_driver_observed"]
        if (saved.get("schema") != "kvbench.residual.dispatch.v1"
                or any(saved.get(k) is not True or current.get(k) is not True for k in fields)
                or saved.get("events") != current.get("events")):
            raise RuntimeError("Final dispatch receipts did not verify this policy's source, fused kernel and decode driver")
        return {"verified": True, "dispatch_summary": str(path), "dispatch_summary_sha256": sha_file(path),
                "executed_source_sha256": runtime["executed_source_sha256"],
                "scope": "Final import/Python launch/capture evidence; warmup included; not a count or proof of every formal token or CUDA graph replay."}

    def run_method(self, row):
        method, folder = row["method"], Path(row["run_dir"])
        row.update(state="starting", started=timestamp())
        self.phase("full_runs")
        self.log(method + "：六任务全量 C8，各一次。")
        try:
            launch = self.start_server(row)
            row["state"] = "running"
            self.save()
            args = ["-m", "kvbench_longcontext.run", "--data-dir", self.cfg["data"], "--out", folder,
                    "--server-manifest", launch, "--concurrency", "8", "--repeats", "1", "--temperature", "0.6",
                    "--top-p", "0.95", "--top-k", "20", "--min-p", "0", "--seed", str(self.cfg["request_seed"]),
                    "--request-timeout", "0", "--gpu", str(self.cfg["gpu"]), "--continue-on-error"]
            row["run_exit_code"] = self.call(self.client, args, method + ".run")
        except Exception as exc:
            row.update(state="startup_failed" if row["state"] == "starting" else "run_failed", error=str(exc))
            self.log(method + " 失败，保留记录：" + str(exc))
        finally:
            try:
                self.cleanup()
            except CleanupFailure as exc:
                row.update(state="cleanup_failed", error=str(exc))
                self.save()
                raise
        if self.cancelled:
            raise KeyboardInterrupt
        if (folder / "run.json").is_file():
            for key, suffix, module in [("score_exit_code", ".score", "scoring"), ("report_exit_code", ".report", "report")]:
                args = ["-m", "kvbench_longcontext." + module, "--run-dir", folder]
                args += ["--data-dir", self.cfg["data"]] if module == "scoring" else ["--out", folder / "readout"]
                try:
                    row[key] = self.call(self.client, args, method + suffix)
                except Exception as exc:
                    row.update(state="postprocess_failed", error=(row["error"] + "\n" + str(exc)).strip())
                    self.cleanup()
            try:
                self.inspect_results(row)
                complete = row["measurement_complete_points"] == 6 and row["evaluation_complete_points"] == 6
                row["state"] = ("run_failed" if row["run_exit_code"] != 0 else "postprocess_failed"
                                if row["score_exit_code"] != 0 or row["report_exit_code"] != 0 else
                                "completed" if complete else "incomplete")
            except Exception as exc:
                row.update(state="result_validation_failed", error=str(exc))
            try:
                row["implementation"] = self.implementation_evidence(row)
                row["implementation_verified"] = True
            except Exception as exc:
                row["implementation_verified"] = False
                row["implementation_error"] = str(exc)
                if row["state"] in ("completed", "incomplete"):
                    row.update(state="implementation_unverified", error=str(exc))
        elif row["state"] == "running":
            row.update(state="run_failed", error="No run.json was created")
        row["finished"] = timestamp()
        self.save()
        self.log(method + "：" + row["state"])

    def comparison(self):
        self.phase("comparison")
        new = [row["run_dir"] for row in self.meta["methods"] if (Path(row["run_dir"]) / "run.json").is_file()]
        runs = self.meta["upstream_runs"] + new
        self.meta["comparison_exit_code"] = self.call(self.client, ["-m", "kvbench_longcontext.report", "--run-dir", *runs,
                                                                 "--out", self.out / "comparison"], "comparison")
        self.save()
        if self.meta["comparison_exit_code"] == 0:
            self.meta["effects_exit_code"] = self.call(self.client, ["-m", "kvbench_residual.effects", "--suite", self.out / "suite.json",
                                                                    "--out", self.out / "comparison"], "residual_effects")
            self.save()

    def execute(self):
        signal.signal(signal.SIGINT, self.on_signal)
        signal.signal(signal.SIGTERM, self.on_signal)
        self.log("AR-KV 新批次目录：" + str(self.out))
        code = 1
        try:
            upstream = self.wait_for_upstream()
            self.phase("freeze_upstream")
            self.freeze_upstream(upstream)
            self.phase("preflight")
            self.preflight()
            self.phase("build_isolated_runtimes")
            self.build_runtimes()
            self.kernel_checks()
            for row in self.meta["methods"]:
                self.run_method(row)
            self.comparison()
            code = 0 if (all(row["state"] == "completed" for row in self.meta["methods"])
                         and self.meta["comparison_exit_code"] == 0 and self.meta["effects_exit_code"] == 0) else 1
        except KeyboardInterrupt:
            code = 130
            self.meta["fatal_error"] = "This new batch was interrupted; upstream work was not stopped. Completed outputs were retained."
        except Exception as exc:
            self.meta["fatal_error"] = str(exc)
            (self.out / "fatal.log").write_text(traceback.format_exc(), encoding="utf-8")
            self.log("新批次停止：" + str(exc))
        finally:
            try:
                self.cleanup()
            except Exception as exc:
                code = 1
                self.meta["fatal_error"] = (self.meta["fatal_error"] or "") + "\nCleanup: " + str(exc)
            for row in self.meta["methods"]:
                if row["state"] == "pending":
                    row["state"] = "not_started"
                elif row["state"] in ("starting", "running"):
                    row["state"] = "interrupted"
            self.meta.update(phase="finished" if code == 0 else "stopped", finished=timestamp(), exit_code=code)
            self.save()
        complete = sum(row["state"] == "completed" for row in self.meta["methods"])
        self.log(f"新增完成 {complete}/6 组；exit={code}。先看 {self.out / 'suite_status.md'}。")
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-suite", type=Path, required=True, help="Absolute path to the currently running V3 suite.json; wait for successful completion")
    parser.add_argument("--out", type=Path, help="New output directory; default is results/residual_v1/full_<timestamp>_<pid>")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="Attention-output error budget divided by the conservative V_ref scale bound, not output-relative error")
    parser.add_argument("--plan", action="store_true", help="Read the named suite and print the plan; no files, environment setup or GPU work")
    args = parser.parse_args(argv)
    args.epsilon = validate_epsilon(args.epsilon)
    if args.plan:
        path = args.after_suite.expanduser()
        if not path.is_absolute():
            raise ValueError("--after-suite must be absolute")
        upstream = read_object(path)
        upstream_identity(upstream, path)
        print(json.dumps({"status": "plan-only", "after_suite": str(path.resolve()), "upstream_finished": upstream.get("finished"),
                          "upstream_exit_code": upstream.get("exit_code"), "data": upstream["settings"]["data"],
                          "methods": METHODS, "protocol": PROTOCOL, "epsilon": args.epsilon,
                          "gpu_gate": "successful upstream completion, free GPU/port, all GPU correctness checks passed",
                          "new_output": str(args.out) if args.out else str(Path(upstream["settings"]["root"]) / "results/residual_v1/full_<timestamp>_<pid>")},
                         ensure_ascii=False, indent=2))
        return 0
    return Suite(args.after_suite, args.out, args.epsilon).execute()


if __name__ == "__main__":
    raise SystemExit(main())
