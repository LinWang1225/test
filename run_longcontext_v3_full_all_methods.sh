#!/usr/bin/env bash
# Run all 9 V3 methods once, serially, with all eligible cases of the 6 core tasks.
# Requires the already supplied V3 patch and existing client/server conda envs.
# Usage: REPO=/home/wanglin/kvbench-3090-cu126 bash run_longcontext_v3_full_all_methods.sh
# Optional: LC_VARIANT=longbench-e LC_CONCURRENCY=8 LC_SOURCE=/absolute/data.zip
# No measured output-token cap, no generation timeout, no automatic retry.
set -euo pipefail
export KVFULL_SCRIPT_PATH
KVFULL_SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
exec "${LC_DRIVER_PYTHON:-python3}" - "$@" <<'PY_KVBENCH_FULL'
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

METHODS = ["fp16", "fp8", "tq4", "tq34", "tq3", "kvarn", "kvarn44", "kvarn64", "kvarn4464"]
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "passage_retrieval_en"]
HF_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
WINDOW = 32768


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def configuration():
    root = Path(os.environ.get("KVBENCH_ROOT", str(Path.home() / "kvbench-3090-cu126"))).expanduser().resolve()
    default_repo = Path.cwd() if (Path.cwd() / "kvbench_longcontext/run.py").is_file() else root
    repo = Path(os.environ.get("REPO", str(default_repo))).expanduser().resolve()
    variant = os.environ.get("LC_VARIANT", "longbench")
    concurrency = int(os.environ.get("LC_CONCURRENCY", "8"))
    if variant not in ("longbench", "longbench-e") or not 1 <= concurrency <= 16:
        raise ValueError("LC_VARIANT must be longbench/longbench-e; LC_CONCURRENCY must be 1..16")
    port = int(os.environ.get("LC_PORT", "8000"))
    gpu = os.environ.get("LC_GPU", "0")
    startup = int(os.environ.get("LC_STARTUP_TIMEOUT", "1800"))
    if not 1 <= port <= 65535 or not gpu.isdigit() or startup < 1:
        raise ValueError("Use a valid LC_PORT, one physical numeric LC_GPU, and positive LC_STARTUP_TIMEOUT")
    name = "full_{}_C{}_{}_{}".format(variant, concurrency, datetime.now().strftime("%Y%m%d_%H%M%S"), os.getpid())
    batch = Path(os.environ.get("LC_BATCH_DIR", str(root / "results/longcontext_v3" / name))).expanduser().resolve()
    source = os.environ.get("LC_SOURCE", "")
    return {"repo": str(repo), "root": str(root),
            "kit": str(Path(os.environ.get("KIT", str(Path.home() / "kv_evalscope_3090_cu126_v2"))).expanduser().resolve()),
            "client_env": os.environ.get("LC_CLIENT_ENV", "evalscope-client"),
            "server_env": os.environ.get("LC_SERVER_ENV", "kvarn-serve"),
            "variant": variant, "concurrency": concurrency, "repeats": 1,
            "max_model_len": WINDOW, "max_seqs": 16, "batch_tokens": 1024, "gpu_util": 0.85,
            "thinking": True, "generation_budget_policy": "remaining-context", "max_tokens": None,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0,
            "request_timeout": 0, "data_seed": 42, "request_seed": 20261009, "server_seed": 20260917,
            "gpu": gpu, "port": port, "startup_timeout": startup,
            "batch": str(batch), "data": str(batch / "data"),
            "source": str(Path(source).expanduser().resolve()) if source else None,
            "methods": list(METHODS), "tasks": list(TASKS)}


def resolve_environment(cfg, role):
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise RuntimeError("Conda was not found. Run this script from a shell where conda is initialized.")
    query = subprocess.run([conda, "info", "--base"], capture_output=True, text=True, check=True)
    init = Path(query.stdout.strip()) / "etc/profile.d/conda.sh"
    env = os.environ.copy()
    env.update(KVFULL_CONDA_SH=str(init), KVFULL_ROLE=role,
               KVFULL_ENV_NAME=cfg[role + "_env"], KVBENCH_ROOT=cfg["root"], KIT=cfg["kit"])
    # Capture environment in memory only. Never write the full environment to logs.
    shell = r'''
set -e
source "$KVFULL_CONDA_SH"
conda activate "$KVFULL_ENV_NAME"
if [ "$KVFULL_ROLE" = server ]; then source "$KIT/env.sh"; fi
if [ -z "${MODEL:-}" ]; then source "$KVBENCH_ROOT/model.env"; fi
export MODEL PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
python -c 'import os,json,sys; print("KVFULL_ENV_JSON="+json.dumps({"python":sys.executable,"env":dict(os.environ)}))'
'''
    result = subprocess.run(["bash", "--noprofile", "--norc", "-c", shell], env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Cannot initialize {} environment {}. Check conda, model.env and KIT/env.sh.\n{}".format(
            role, cfg[role + "_env"], result.stderr[-4000:]))
    lines = [x[len("KVFULL_ENV_JSON="):] for x in result.stdout.splitlines() if x.startswith("KVFULL_ENV_JSON=")]
    if len(lines) != 1:
        raise RuntimeError("Environment capture failed for " + role)
    runtime = json.loads(lines[0])
    runtime["env"].update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
    return runtime


class CleanupFailure(RuntimeError):
    pass


class Suite:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch = Path(cfg["batch"])
        self.batch.mkdir(parents=True, exist_ok=False)
        (self.batch / "logs").mkdir()
        self.client = self.server = None
        self.step_proc = self.server_proc = None
        self.server_output = None
        self.cleaning = False
        self.cancelled = False
        self.meta = {"schema": "kvbench.longcontext.full_suite.v1", "started": timestamp(),
                     "settings": cfg, "fatal_error": None, "comparison_exit_code": None,
                     "methods": [{"method": m, "state": "pending", "run_exit_code": None,
                                  "score_exit_code": None, "report_exit_code": None,
                                  "measurement_complete_points": 0, "evaluation_complete_points": 0,
                                  "planned_points": len(TASKS), "expected_tasks": None,
                                  "valid_responses": 0, "scored_tasks": 0,
                                  "length_capped": 0, "budget_inconsistent_points": 0,
                                  "unexpected_length_before_budget_count": 0,
                                  "run_dir": str(self.batch / "runs" / m), "error": ""} for m in METHODS]}
        original = os.environ.get("KVFULL_SCRIPT_PATH")
        if original and Path(original).is_file():
            shutil.copy2(original, self.batch / "run_full_all_methods.sh")
        self.save()

    def log(self, message):
        line = "[{}] {}".format(timestamp(), message)
        print(line, flush=True)
        with (self.batch / "suite.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save(self):
        self.meta["updated"] = timestamp()
        write_json(self.batch / "suite.json", self.meta)
        fields = list(self.meta["methods"][0])
        with (self.batch / "suite_status.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.meta["methods"])
        lines = ["# 九方法全量实验状态", "", "各方法只尝试一次；完整输出截断仍评分。completed表示测量/评分/报告完成，预算诊断另列。", "",
                 "| 方法 | 状态 | 测量完整点 | 评分完整点 | 有效响应/应有 | 已评分 | 截断数 | 预算异常点 |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in self.meta["methods"]:
            lines.append("| {method} | {state} | {measurement_complete_points}/{planned_points} | {evaluation_complete_points}/{planned_points} | {valid_responses}/{expected_tasks} | {scored_tasks} | {length_capped} | {budget_inconsistent_points} |".format(**r))
        lines += ["", "具体任务分数、延迟和吞吐见 comparison/overview.md 或 overview.csv。",
                  "启动失败而没有run.json的方法保留在本表，不会伪装成0分。",
                  "数据纳入/排除数量见 data_coverage.csv。日志见 logs/ 和各 servers/<method>/server.log。"]
        if self.meta["fatal_error"]:
            lines += ["", "批次错误：" + self.meta["fatal_error"]]
        (self.batch / "suite_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def on_signal(self, signum, frame):
        self.cancelled = True
        if not self.cleaning:
            raise KeyboardInterrupt

    def call(self, runtime, args, label):
        logpath = self.batch / "logs" / (label + ".log")
        self.log("开始 {}；日志 {}".format(label, logpath))
        with logpath.open("w", encoding="utf-8") as out:
            self.step_proc = subprocess.Popen([runtime["python"], *map(str, args)], cwd=self.cfg["repo"],
                                              env=runtime["env"], stdout=out, stderr=subprocess.STDOUT,
                                              start_new_session=True)
            proc = self.step_proc
            while True:
                try:
                    code = proc.wait(timeout=60)
                    self.step_proc = None
                    self.log("结束 {}；exit={}".format(label, code))
                    return code
                except subprocess.TimeoutExpired:
                    # Heartbeat only. Never cancel or cap a generation request.
                    self.log("{} 仍在运行；正式生成不设输出/耗时上限。".format(label))

    def require(self, runtime, args, label):
        if self.call(runtime, args, label):
            raise RuntimeError(label + " failed; inspect logs/" + label + ".log")

    def preflight(self):
        if not (Path(self.cfg["repo"]) / "kvbench_longcontext/report.py").is_file():
            raise RuntimeError("Apply the full V3 patch to REPO before running this script.")
        self.client = resolve_environment(self.cfg, "client")
        self.server = resolve_environment(self.cfg, "server")
        client_model = self.client["env"].get("MODEL")
        server_model = self.server["env"].get("MODEL")
        if not client_model or not server_model:
            raise RuntimeError("MODEL is empty. Set MODEL explicitly or check KVBENCH_ROOT/model.env.")
        model = Path(client_model).expanduser().resolve()
        other = Path(server_model).expanduser().resolve()
        if not model.is_dir() or model != other:
            raise RuntimeError("Client and server must use the same existing MODEL directory.")
        self.cfg["model"] = str(model)
        self.meta["python"] = {"client": self.client["python"], "server": self.server["python"]}
        self.save()
        self.log("依赖检查失败时，请在客户端环境安装 V3 requirements-client.txt；尚未启动GPU服务。")
        client_check = '''
import transformers, tokenizers, huggingface_hub, rouge
import kvbench_longcontext
from kvbench_longcontext.scoring import check_dependencies
from kvbench_longcontext.serve import METHODS
assert kvbench_longcontext.__version__ == '3.0.0', 'Use the supplied complete V3 patch'
assert set(METHODS) == set(%r), 'Method list differs from the supplied V3 patch'
check_dependencies({'qa_f1', 'retrieval', 'rouge_l'})
assert rouge.Rouge().get_scores('alpha beta', 'alpha beta')[0]['rouge-l']['f'] > .99
print('V3 client dependencies and all nine method names verified.')
''' % METHODS
        self.require(self.client, ["-c", client_check], "preflight_client")
        self.require(self.server, ["-c", "import vllm,torch; from kvbench_longcontext.serve import METHODS; print('Server imports OK')"], "preflight_server")

    def prepare(self):
        args = ["-m", "kvbench_longcontext.prepare", "--model", self.cfg["model"], "--out", self.cfg["data"],
                "--datasets", *TASKS, "--variant", self.cfg["variant"], "--sample-size", "0", "--data-seed", "42",
                "--min-input-tokens", "0", "--max-model-len", str(WINDOW), "--overflow", "skip",
                "--thinking", "--generation-budget-policy", "remaining-context"]
        if self.cfg["source"]:
            if not Path(self.cfg["source"]).is_file():
                raise ValueError("LC_SOURCE must name the official local data.zip file")
            args += ["--source", self.cfg["source"], "--source-revision", HF_REVISION]
        code = self.call(self.client, args, "prepare_all_data")
        if code:
            raise RuntimeError("Dataset preparation failed. Check logs/prepare_all_data.log; if download is blocked, set LC_SOURCE to the pinned official data.zip and check HF_HUB_OFFLINE.")
        path = Path(self.cfg["data"]) / "manifest.json"
        data = read_json(path)
        settings = data["settings"]
        expected = {"thinking": True, "sample_size": 0, "min_input_tokens": 0,
                    "max_input_tokens": None, "max_new_tokens": None, "max_model_len": WINDOW,
                    "overflow": "skip", "generation_budget_policy": "remaining-context"}
        if any(settings.get(k) != v for k, v in expected.items()) or set(data["datasets"]) != set(TASKS):
            raise ValueError("Prepared cohort does not match the full-data null-budget protocol")
        if data["variant"] != self.cfg["variant"] or data["max_tokens_transport"] != "explicit-json-null":
            raise ValueError("Unexpected data variant or max_tokens transport")
        coverage = []
        for task in TASKS:
            info = data["datasets"][task]
            if info["selected_count"] != info["eligible_count"] or info["max_tokens"] is not None or info["truncated_count"]:
                raise ValueError("Not the entire eligible untruncated cohort: " + task)
            coverage.append({"dataset": task, **{k: info[k] for k in ["source_count", "eligible_count", "selected_count", "excluded_count", "truncated_count"]}})
        with (self.batch / "data_coverage.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(coverage[0]))
            writer.writeheader()
            writer.writerows(coverage)
        self.meta["data_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.meta["data_coverage"] = coverage
        count = sum(r["selected_count"] for r in coverage)
        for row in self.meta["methods"]:
            row["expected_tasks"] = count
        self.save()
        self.log("数据已冻结：每方法{}题，9方法共{}条正式请求；排除数量见data_coverage.csv。".format(count, count * len(METHODS)))
        self.require(self.client, ["-m", "kvbench_longcontext.run", "--data-dir", self.cfg["data"],
                     "--out", str(self.batch / "dry_run_no_output"), "--concurrency", str(self.cfg["concurrency"]),
                     "--repeats", "1", "--request-timeout", "0", "--dry-run"], "preflight_plan")

    def start_server(self, method):
        dest = self.batch / "servers" / method
        args = ["-m", "kvbench_longcontext.serve", "--model", self.cfg["model"], "--method", method,
                "--out", str(dest), "--port", str(self.cfg["port"]), "--gpu", self.cfg["gpu"],
                "--max-model-len", str(WINDOW), "--max-seqs", "16", "--batch-tokens", "1024",
                "--gpu-util", "0.85", "--server-seed", str(self.cfg["server_seed"]),
                "--startup-timeout", str(self.cfg["startup_timeout"])]
        self.server_output = (self.batch / "logs" / (method + ".launcher.log")).open("w", encoding="utf-8")
        self.server_proc = subprocess.Popen([self.server["python"], *args], cwd=self.cfg["repo"],
                                            env=self.server["env"], stdout=self.server_output,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + self.cfg["startup_timeout"] + 180
        last_notice = time.monotonic()
        manifest = dest / "launch.json"
        while time.monotonic() < deadline:
            code = self.server_proc.poll()
            if code is not None:
                raise RuntimeError("{} startup exited {}. See logs/{}.launcher.log and servers/{}/server.log".format(method, code, method, method))
            if manifest.is_file() and read_json(manifest).get("native_verified") is True:
                self.log(method + " 服务已就绪，原生配置已验证。")
                return manifest
            if time.monotonic() - last_notice >= 60:
                self.log(method + " 正在启动/编译；这是服务启动等待，不是生成时间上限。")
                last_notice = time.monotonic()
            time.sleep(1)
        raise RuntimeError(method + " server startup did not become ready")

    def stop_server(self):
        proc = self.server_proc
        if proc is not None and proc.poll() is None:
            self.log("停止本脚本的服务launcher PID={}，等待其清理自身引擎。".format(proc.pid))
            # Python launcher handles SIGINT and cleans its engine's separate PGID.
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired as exc:
                raise CleanupFailure("Own launcher PID={} did not finish cleanup. Remaining methods will not start; inspect its server log. No unrelated GPU process was killed.".format(proc.pid)) from exc
        self.server_proc = None
        if self.server_output is not None:
            self.server_output.close()
            self.server_output = None

    def cleanup(self):
        self.cleaning = True
        try:
            # Closing our server first lets timeout=0 client streams finish as partial.
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
        run = read_json(folder / "run.json")
        if run.get("method") != row["method"] or run.get("data_manifest_sha256") != self.meta["data_manifest_sha256"]:
            raise ValueError("Run identity or data hash mismatch")
        actual = {(p["dataset"], p["repeat"], p["concurrency"]) for p in run["points"]}
        wanted = {(task, 1, self.cfg["concurrency"]) for task in TASKS}
        if actual != wanted or len(run["points"]) != len(TASKS):
            raise ValueError("Run does not contain exactly the six planned C/repeat points")
        for key in ["measurement_complete_points", "evaluation_complete_points", "valid_responses", "scored_tasks", "length_capped", "budget_inconsistent_points", "unexpected_length_before_budget_count"]:
            row[key] = 0
        for point in run["points"]:
            dest = folder / point["id"]
            perf = read_json(dest / "performance.summary.json") if (dest / "performance.summary.json").is_file() else {}
            quality = read_json(dest / "scores.summary.json") if (dest / "scores.summary.json").is_file() else {}
            row["measurement_complete_points"] += perf.get("measurement_complete") is True
            row["evaluation_complete_points"] += quality.get("evaluation_complete") is True
            row["budget_inconsistent_points"] += perf.get("generation_budget_consistent") is False
            for key in ("valid_responses", "length_capped", "unexpected_length_before_budget_count"):
                row[key] += perf.get(key, 0) or 0
            row["scored_tasks"] += quality.get("scored_tasks", 0) or 0

    def run_method(self, row):
        method, folder = row["method"], Path(row["run_dir"])
        row.update(state="starting", started=timestamp())
        self.save()
        self.log("===== {}/{} {}：只运行一次 =====".format(METHODS.index(method) + 1, len(METHODS), method))
        try:
            manifest = self.start_server(method)
            row["state"] = "running"
            self.save()
            args = ["-m", "kvbench_longcontext.run", "--data-dir", self.cfg["data"], "--out", str(folder),
                    "--server-manifest", str(manifest), "--concurrency", str(self.cfg["concurrency"]),
                    "--repeats", "1", "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
                    "--min-p", "0", "--seed", str(self.cfg["request_seed"]), "--request-timeout", "0",
                    "--gpu", self.cfg["gpu"], "--continue-on-error"]
            row["run_exit_code"] = self.call(self.client, args, method + ".run")
        except Exception as exc:
            row["state"] = "startup_failed" if row["state"] == "starting" else "run_failed"
            row["error"] = str(exc)
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
            phases = [
                ("score_exit_code", ".score", ["-m", "kvbench_longcontext.scoring", "--data-dir", self.cfg["data"], "--run-dir", str(folder)]),
                ("report_exit_code", ".report", ["-m", "kvbench_longcontext.report", "--run-dir", str(folder), "--out", str(folder / "readout")]),
            ]
            for key, suffix, args in phases:
                try:
                    row[key] = self.call(self.client, args, method + suffix)
                except Exception as exc:
                    row[key] = None
                    row["state"] = "postprocess_failed"
                    row["error"] = (row["error"] + "\n" + suffix.lstrip(".") + ": " + str(exc)).strip()
                    self.log(method + suffix + " 失败，保留本方法状态并继续：" + str(exc))
                    self.cleanup()
                if self.cancelled:
                    raise KeyboardInterrupt
            try:
                self.inspect_results(row)
                complete = row["measurement_complete_points"] == len(TASKS) and row["evaluation_complete_points"] == len(TASKS)
                if row["run_exit_code"] != 0:
                    row["state"] = "run_failed"
                elif row["score_exit_code"] != 0 or row["report_exit_code"] != 0:
                    row["state"] = "postprocess_failed"
                else:
                    row["state"] = "completed" if complete else "incomplete"
            except Exception as exc:
                row.update(state="result_validation_failed", error=str(exc))
        elif row["state"] == "running":
            row.update(state="run_failed", error="No run.json was created")
        row["finished"] = timestamp()
        self.save()
        self.log("{}：{}；测量完整{}/6，评分完整{}/6。".format(method, row["state"], row["measurement_complete_points"], row["evaluation_complete_points"]))

    def comparison(self):
        runs = [r["run_dir"] for r in self.meta["methods"] if (Path(r["run_dir"]) / "run.json").is_file()]
        if runs and self.client:
            self.meta["comparison_exit_code"] = self.call(self.client, ["-m", "kvbench_longcontext.report", "--run-dir", *runs,
                "--out", str(self.batch / "comparison")], "comparison")
        self.save()

    def execute(self):
        # Reinstall handlers explicitly: nohup/background shells can inherit SIG_IGN.
        signal.signal(signal.SIGINT, self.on_signal)
        signal.signal(signal.SIGTERM, self.on_signal)
        self.log("批次目录：" + str(self.batch))
        self.log("全部9方法各一次；六任务全量合格样本；C{}；thinking=true；max_tokens=null；request-timeout=0。".format(self.cfg["concurrency"]))
        exit_code = 1
        try:
            self.preflight()
            self.prepare()
            for row in self.meta["methods"]:
                self.run_method(row)
            self.comparison()
            exit_code = 0 if all(r["state"] == "completed" for r in self.meta["methods"]) and self.meta["comparison_exit_code"] == 0 else 1
        except KeyboardInterrupt:
            exit_code = 130
            self.meta["fatal_error"] = "User interrupted the batch; completed outputs were retained."
            self.log("收到停止请求；先关闭自己的服务，再收尾客户端。")
        except Exception as exc:
            self.meta["fatal_error"] = str(exc)
            (self.batch / "fatal.log").write_text(traceback.format_exc(), encoding="utf-8")
            self.log("批次停止：" + str(exc))
        finally:
            try:
                self.cleanup()
            except Exception as exc:
                self.meta["fatal_error"] = (self.meta["fatal_error"] or "") + "\nCleanup: " + str(exc)
                self.log(str(exc))
            for row in self.meta["methods"]:
                if row["state"] == "pending":
                    row["state"] = "not_started"
                elif row["state"] in ("starting", "running"):
                    row["state"] = "interrupted"
            self.meta.update(finished=timestamp(), exit_code=exit_code)
            self.save()
        complete = sum(r["state"] == "completed" for r in self.meta["methods"])
        self.log("完成 {}/9 方法；exit={}。先看 {}，再看 comparison/overview.md。".format(complete, exit_code, self.batch / "suite_status.md"))
        return exit_code


def main():
    parser = argparse.ArgumentParser(description="V3 full eligible LongBench cohort, 9 KV methods once each, serial C8 by default.")
    parser.add_argument("--plan", action="store_true", help="Print settings without creating files, loading environments or starting GPU work")
    args = parser.parse_args()
    cfg = configuration()
    if args.plan:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return 0
    return Suite(cfg).execute()


if __name__ == "__main__":
    raise SystemExit(main())
PY_KVBENCH_FULL
