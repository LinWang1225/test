"""Read fixed AR comparisons from saved V3 results; no inference or regrading."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from kvbench_longcontext.common import atomic_json
from kvbench_longcontext.report import collect_run, markdown_table
from .suite import METHODS, TASKS


PAIRS = [
    ("main_net", "bound_kv", "parent", "净效果：K/V 误差界选择与原 KVarN K4V4"),
    ("representation", "repacked_full", "parent", "表示开销：拆分平面全读与原布局"),
    ("anchor_limit", "anchor", "repacked_full", "Anchor 极限：全部量化 token 保留 Anchor，但跳过全部残差"),
    ("key_selection", "bound_k", "repacked_full", "K 选择：仅 K 按误差界跳读，V 残差全读"),
    ("value_selection", "bound_v", "repacked_full", "V 选择：K 残差全读，仅 V 按误差界跳读"),
    ("joint_selection", "bound_kv", "repacked_full", "K/V 联合选择：相对同表示全读，预算各分一半"),
]

SETTING_KEYS = ["thinking", "temperature", "top_p", "top_k", "min_p", "generation_budget_policy",
                "max_tokens_transport", "request_seed_base", "request_timeout", "warmup",
                "warmup_per_point", "warmup_output_cap", "retries"]
SERVER_KEYS = ["max_model_len", "max_seqs", "batch_tokens", "gpu_util", "kv_bytes", "eager",
               "server_seed", "apc", "generation_config", "override_generation_config", "kv_dtype", "block_size", "residual_epsilon"]
ENV_KEYS = ["KVARN_SINK_TOKENS", "KVARN_SINKHORN_ITERS", "KVARN_POOL_SLOTS", "KVARN_FAST_FLUSH",
            "KVARN_FUSED_DECODE", "KVARN_SPLIT_K", "KVARN_NUM_KV_SPLITS", "KVARN_FUSED_VERIFY", "KVARN_SHARED_VERIFY",
            "VLLM_ATTENTION_BACKEND", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "VLLM_WORKER_MULTIPROC_METHOD",
            "CUDA_HOME", "TRITON_PTXAS_PATH", "NVIDIA_TF32_OVERRIDE", "TRITON_F32_DEFAULT"]
FIELDS = ["comparison", "precision", "dataset", "repeat", "concurrency", "target_method", "baseline_method",
          "target_state", "baseline_state", "target_measurement_complete", "baseline_measurement_complete",
          "target_evaluation_complete", "baseline_evaluation_complete", "target_implementation_verified",
          "baseline_implementation_verified", "data_hash_match", "protocol_match", "comparison_complete",
          "target_budget_consistent", "baseline_budget_consistent", "score_name", "score_delta_pp",
          "tpot_improvement_percent", "output_tps_gain_percent", "output_length_change_percent", "e2e_change_percent",
          "length_rate_delta_pp", "empty_final_delta", "empty_final_rate_delta_pp", "preemptions_delta",
          "target_score_0_100", "baseline_score_0_100", "target_tpot_ms", "baseline_tpot_ms",
          "data_manifest_sha256", "target_run_dir", "baseline_run_dir", "residual_epsilon",
          "formal_dram_bytes", "formal_traffic_scope", "unavailable_reason"]


def read_object(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return obj


def finite(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def difference(target, baseline):
    a, b = finite(target), finite(baseline)
    return a - b if a is not None and b is not None else None


def relative_change(target, baseline):
    a, b = finite(target), finite(baseline)
    return 100 * (a / b - 1) if a is not None and b is not None and b > 0 else None


def rate(count, n):
    count, n = finite(count), finite(n)
    return 100 * count / n if count is not None and n is not None and n > 0 else None


def load_method(info, output):
    path = Path(info.get("run_dir", Path(output) / "runs" / info["method"])).resolve()
    result = {"info": info, "folder": str(path), "run": None, "server": None, "points": {}, "error": ""}
    if not (path / "run.json").is_file():
        result["error"] = "run.json missing"
        return result
    try:
        run = read_object(path / "run.json")
        if run.get("method") != info["method"]:
            raise ValueError("Saved run method differs from the planned method")
        server = read_object(path / "server.launch.json")
        if server.get("config", {}).get("residual_policy") != info.get("policy"):
            raise ValueError("Saved runtime policy differs from the planned method")
        rows = collect_run(path)
        points = {(row["dataset"], row["repeat"], row["concurrency"]): row for row in rows}
        if len(points) != len(rows):
            raise ValueError("Duplicate task/repeat/concurrency points")
        result.update(run=run, server=server, points=points)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def protocol_matches(a, b, row_a, row_b):
    if a["run"] is None or b["run"] is None or a["server"] is None or b["server"] is None or row_a is None or row_b is None:
        return None, ["protocol unavailable"]
    reasons = []
    for key in SETTING_KEYS:
        settings_a, settings_b = a["run"].get("settings", {}), b["run"].get("settings", {})
        if key not in settings_a or key not in settings_b or settings_a[key] != settings_b[key]:
            reasons.append("settings." + key)
    for key in SERVER_KEYS:
        config_a, config_b = a["server"].get("config", {}), b["server"].get("config", {})
        if key not in config_a or key not in config_b or config_a[key] != config_b[key]:
            reasons.append("server." + key)
    if a["server"].get("tokenizer", {}).get("fingerprint") != b["server"].get("tokenizer", {}).get("fingerprint"):
        reasons.append("tokenizer fingerprint")
    if not a["server"].get("tokenizer", {}).get("fingerprint"):
        reasons.append("tokenizer fingerprint missing")
    for key in ENV_KEYS:
        if a["server"].get("environment", {}).get(key) != b["server"].get("environment", {}).get(key):
            reasons.append("environment." + key)
    for key in ["packages", "cuda_visible_devices", "gpu", "weight_file_sizes_mtime_ns"]:
        if a["server"].get(key) != b["server"].get(key):
            reasons.append("server_identity." + key)
    pins_a, pins_b = a["server"].get("residual", {}).get("parent_pins"), b["server"].get("residual", {}).get("parent_pins")
    if not pins_a or pins_a != pins_b:
        reasons.append("parent writer/reader source pins")
    for key in ["dataset", "repeat", "concurrency", "expected_tasks", "score_name"]:
        if row_a.get(key) != row_b.get(key):
            reasons.append("point." + key)
    return not reasons, reasons


def compare_one(kind, precision, task, target, baseline, expected_hash):
    ra, rb = target["points"].get((task, 1, 8)), baseline["points"].get((task, 1, 8))
    row = {key: None for key in FIELDS}
    row.update(comparison=kind, precision=precision, dataset=task, repeat=1, concurrency=8,
               target_method=target["info"]["method"], baseline_method=baseline["info"]["method"],
               target_state=target["info"].get("state", "not_started"), baseline_state=baseline["info"].get("state", "not_started"),
               target_implementation_verified=target["info"].get("implementation_verified") is True,
               baseline_implementation_verified=baseline["info"].get("implementation_verified") is True,
               target_run_dir=target["folder"], baseline_run_dir=baseline["folder"], data_manifest_sha256=expected_hash,
               residual_epsilon=target["server"].get("config", {}).get("residual_epsilon") if target["server"] else None,
               formal_dram_bytes=None,
               formal_traffic_scope="Not measured in formal generation. Microbench logical bytes are separate replay estimates, not DRAM transactions.")
    for prefix, saved in [("target", ra), ("baseline", rb)]:
        row[prefix + "_measurement_complete"] = bool(saved and saved.get("measurement_complete") is True)
        row[prefix + "_evaluation_complete"] = bool(saved and saved.get("evaluation_complete") is True)
        row[prefix + "_budget_consistent"] = saved.get("generation_budget_consistent") if saved else None
    if target["run"] is not None and baseline["run"] is not None and expected_hash:
        row["data_hash_match"] = (target["run"].get("data_manifest_sha256") == expected_hash
                                  and baseline["run"].get("data_manifest_sha256") == expected_hash)
    row["protocol_match"], protocol_reasons = protocol_matches(target, baseline, ra, rb)
    reasons = []
    for label, saved, point in [("target", target, ra), ("baseline", baseline, rb)]:
        if saved["error"]:
            reasons.append(label + ": " + saved["error"])
        if point is None:
            reasons.append(label + ": planned point unavailable")
        for suffix in ["measurement_complete", "evaluation_complete", "implementation_verified"]:
            if row[label + "_" + suffix] is not True:
                reasons.append(label + ": " + suffix + " is false")
    if row["data_hash_match"] is not True:
        reasons.append("frozen data hash unavailable or different")
    if row["protocol_match"] is not True:
        reasons.append("protocol unavailable or different: " + ", ".join(protocol_reasons))
    row["comparison_complete"] = not reasons
    row["unavailable_reason"] = "; ".join(reasons)
    row["score_name"] = ra.get("score_name") if ra is not None else rb.get("score_name") if rb is not None else None
    if not row["comparison_complete"]:
        return row
    row.update(target_score_0_100=finite(ra["task_score_0_100"]), baseline_score_0_100=finite(rb["task_score_0_100"]),
               target_tpot_ms=finite(ra["tpot_ms_mean"]), baseline_tpot_ms=finite(rb["tpot_ms_mean"]),
               score_delta_pp=difference(ra["task_score_0_100"], rb["task_score_0_100"]),
               output_tps_gain_percent=relative_change(ra["output_tokens_per_s"], rb["output_tokens_per_s"]),
               output_length_change_percent=relative_change(ra["output_tokens_mean"], rb["output_tokens_mean"]),
               e2e_change_percent=relative_change(ra["e2e_s_mean"], rb["e2e_s_mean"]),
               length_rate_delta_pp=difference(ra["length_capped_percent_of_valid"], rb["length_capped_percent_of_valid"]),
               empty_final_delta=difference(ra["empty_final_answers"], rb["empty_final_answers"]),
               empty_final_rate_delta_pp=difference(rate(ra["empty_final_answers"], ra["valid_responses"]),
                                                   rate(rb["empty_final_answers"], rb["valid_responses"])),
               preemptions_delta=difference(ra["preemptions_delta"], rb["preemptions_delta"]))
    change = relative_change(ra["tpot_ms_mean"], rb["tpot_ms_mean"])
    row["tpot_improvement_percent"] = -change if change is not None else None
    return row


def collect_effects(suite_path):
    meta = read_object(suite_path)
    if meta.get("schema") != "kvbench.residual.full_suite.v1":
        raise ValueError("Expected an AR full-suite manifest")
    info = {row["method"]: row for row in meta.get("methods", [])}
    if len(info) != len(meta.get("methods", [])):
        raise ValueError("Duplicate method rows in the suite")
    loaded = {m["method"]: load_method(info.get(m["method"], m), meta["output"]) for m in METHODS}
    rows = []
    for kind, target, baseline, _ in PAIRS:
        for precision in ["k4v4"]:
            for task in TASKS:
                rows.append(compare_one(kind, precision, task, loaded[f"ar_{precision}_{target}"],
                                        loaded[f"ar_{precision}_{baseline}"], meta.get("data_manifest_sha256")))
    return meta, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    meta, rows = collect_effects(args.suite)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "residual_effects.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(args.out / "residual_effects.json", {"schema": "kvbench.residual.effects.v1", "source_suite": str(args.suite.resolve()),
                                                 "planned_comparisons": 36, "complete_comparisons": sum(r["comparison_complete"] for r in rows),
                                                 "rows": rows})
    text = "# AR-KV 效果与消融读出\n\n"
    text += "固定列出 K4V4 × 6 个任务 × 6 个预先定义的对照，共 36 行。只读取已保存的 V3 评分和性能，不重新评分，不按结果选择策略。\n\n"
    text += "主差值仅在两侧测量、评分、最终实现来源证据完整，且冻结数据与实验协议一致时计算。失败或缺失保留为 NA，不能当成 0 分或 0 延迟。单项分母为 0 或指标不可计算时，该差值也为 NA。\n\n"
    text += "得分变化是 0–100 任务分的直接差，记为 pp；它不是所有任务通用的二元正确率。TPOT 改善=100×(1−新/对照)，正数表示每 token 时间下降；输出吞吐增益=100×(新/对照−1)，正数表示提高。\n\n"
    text += "E2E 和输出长度变化=100×(新/对照−1)，负数表示减少。生成更短不等于 kernel 更快，应一起阅读质量、TPOT、吞吐、输出长度、截断和空答案。\n\n"
    for kind, _, _, title in PAIRS:
        group = [r for r in rows if r["comparison"] == kind]
        text += "## " + title + "\n\n"
        text += markdown_table(["精度", "任务", "得分Δ pp", "TPOT改善%", "输出吞吐增益%", "E2E变化%", "输出长度变化%", "主对照可用"],
                               [[r["precision"], r["dataset"], r["score_delta_pp"], r["tpot_improvement_percent"],
                                 r["output_tps_gain_percent"], r["e2e_change_percent"], r["output_length_change_percent"], r["comparison_complete"]] for r in group]) + "\n\n"
    text += "## 截断、空答案与抢占\n\n"
    text += markdown_table(["对照", "精度", "任务", "截断率Δ pp", "空答案数Δ", "空答案率Δ pp", "抢占次数Δ"],
                           [[r["comparison"], r["precision"], r["dataset"], r["length_rate_delta_pp"], r["empty_final_delta"],
                             r["empty_final_rate_delta_pp"], r["preemptions_delta"]] for r in rows]) + "\n\n"
    text += "完整 length 截断照常评分；预算一致性诊断也保存在 CSV 中，不因为截断而删除题目。实现证据来自最终 Python 导入/launch/capture 记录，包含预热；它不表示对每个正式 token 或 CUDA graph replay 逐个计数。\n\n"
    text += "parent 使用原 KVarN K4V4 布局；repacked_full 使用拆分平面但读取全部残差。main_net 检查整个方法净效果，joint_selection 用同表示全读对照隔离选择收益，representation 单独检查拆分布局的成本。已有九方法结果在 overview.md/csv 中供参考。\n\n"
    text += "epsilon 是单步 attention 输出误差控制参数，不是任务得分损失百分比或生成序列一致性保证。正式推理不加入流量原子计数；本表 formal_dram_bytes 保留 NA。独立 microbench 的逻辑字节估计只能解释该重放工作负载，不能冒充正式推理 DRAM 实测。\n\n这是每组一次的全量对照，不做跨任务总体准确率、自动排名或随机种子置信区间。需要重复种子实验才能进一步评估采样波动。\n\n"
    unavailable = [r for r in rows if not r["comparison_complete"]]
    if unavailable:
        text += "## 尚不可作为主对照的行\n\n"
        text += markdown_table(["对照", "精度", "任务", "原因"], [[r["comparison"], r["precision"], r["dataset"], r["unavailable_reason"]] for r in unavailable]) + "\n\n"
    text += "来源：`" + str(args.suite.resolve()) + "`；data_manifest_sha256=`" + str(meta.get("data_manifest_sha256")) + "`。\n"
    (args.out / "residual_effects.md").write_text(text, encoding="utf-8")
    print(json.dumps({"planned_comparisons": len(rows), "complete_comparisons": sum(r["comparison_complete"] for r in rows),
                      "markdown": str((args.out / "residual_effects.md").resolve()),
                      "csv": str((args.out / "residual_effects.csv").resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
