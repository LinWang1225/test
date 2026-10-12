"""Create human-readable tables from saved results; never run inference or grading."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FIELDS = [
    "run_name", "method", "dataset", "repeat", "concurrency", "generation_budget_policy",
    "thinking", "temperature", "top_p", "top_k", "min_p", "data_manifest_sha256",
    "expected_tasks", "valid_responses", "scored_tasks", "measurement_complete", "evaluation_complete",
    "generation_budget_consistent", "unexpected_length_before_budget_count",
    "score_name", "task_score_0_100", "input_tokens_mean", "output_tokens_mean",
    "ttft_ms_mean", "ttft_ms_p95", "tpot_ms_mean", "tpot_samples", "e2e_s_mean", "e2e_s_p95",
    "output_tokens_per_s", "valid_requests_per_s", "length_capped", "length_capped_percent_of_valid",
    "empty_final_answers", "preemptions_delta", "http_failed", "incomplete_streams",
    "missing_native_usage", "prompt_token_mismatches", "missing_tasks", "invalid_response_tasks",
    "unscored_tasks", "source_point_dir",
]


def read_object(path, required=False):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def number(value, factor=1):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value * factor
    return None


def statistic(summary, name, key="mean", factor=1):
    return number(summary.get(name, {}).get(key), factor)


def collect_run(folder):
    folder = Path(folder).resolve()
    run = read_object(folder / "run.json", required=True)
    if run.get("schema") != "kvbench.longcontext.run.v1":
        raise ValueError("Unsupported run schema: " + str(folder))
    settings, rows, seen = run["settings"], [], set()
    for point in run["points"]:
        pid = point["id"]
        if not isinstance(pid, str) or Path(pid).name != pid or pid in (".", "..") or pid in seen:
            raise ValueError("Invalid or duplicate point ID")
        seen.add(pid)
        dest = folder / pid
        perf = read_object(dest / "performance.summary.json")
        quality = read_object(dest / "scores.summary.json")
        for summary in (perf, quality):
            for field in ("id", "dataset", "repeat", "concurrency", "expected_tasks"):
                if field in summary and summary[field] != point[field]:
                    raise ValueError(f"Summary/plan mismatch: {pid} {field}")
        if perf.get("method", run["method"]) != run["method"]:
            raise ValueError("Performance method differs from run method")
        complete = quality.get("evaluation_complete") is True
        row = {k: None for k in FIELDS}
        row.update(run_name=folder.name, method=run["method"], dataset=point["dataset"],
                   repeat=point["repeat"], concurrency=point["concurrency"],
                   generation_budget_policy=settings.get("generation_budget_policy", "not-recorded"),
                   data_manifest_sha256=run.get("data_manifest_sha256"),
                   expected_tasks=point["expected_tasks"],
                   measurement_complete=perf.get("measurement_complete") is True,
                   evaluation_complete=complete, source_point_dir=str(dest),
                   task_score_0_100=number(quality.get("task_score_0_100")) if complete else None,
                   score_name=quality.get("score_name"),
                   input_tokens_mean=statistic(perf, "prompt_tokens"),
                   output_tokens_mean=statistic(perf, "completion_tokens"),
                   ttft_ms_mean=statistic(perf, "ttft_s", factor=1000),
                   ttft_ms_p95=statistic(perf, "ttft_s", "p95", 1000),
                   tpot_ms_mean=statistic(perf, "tpot_s", factor=1000),
                   tpot_samples=statistic(perf, "tpot_s", "n"),
                   e2e_s_mean=statistic(perf, "e2e_s"),
                   e2e_s_p95=statistic(perf, "e2e_s", "p95"))
        for key in ("thinking", "temperature", "top_p", "top_k", "min_p"):
            row[key] = settings.get(key)
        for key in ("valid_responses", "output_tokens_per_s", "valid_requests_per_s", "length_capped",
                    "empty_final_answers", "preemptions_delta", "http_failed", "incomplete_streams",
                    "missing_native_usage", "prompt_token_mismatches", "generation_budget_consistent",
                    "unexpected_length_before_budget_count"):
            row[key] = perf.get(key)
        for key in ("scored_tasks", "missing_tasks", "invalid_response_tasks", "unscored_tasks"):
            row[key] = quality.get(key)
        valid, capped = number(row["valid_responses"]), number(row["length_capped"])
        row["length_capped_percent_of_valid"] = 100 * capped / valid if valid and capped is not None else None
        rows.append(row)
    return rows


def display(value, digits=2):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |"] +
                      ["| " + " | ".join(display(x) for x in row) + " |" for row in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="Directory for overview.md and overview.csv")
    args = parser.parse_args(argv)
    if len({str(p.resolve()) for p in args.run_dir}) != len(args.run_dir):
        raise ValueError("Duplicate run directories")
    rows = [row for folder in args.run_dir for row in collect_run(folder)]
    args.out.mkdir(parents=True, exist_ok=True)
    csvpath, mdpath = args.out / "overview.csv", args.out / "overview.md"
    with csvpath.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    quality_rows = [[r["run_name"], r["dataset"], f'r{r["repeat"]}/C{r["concurrency"]}',
                     f'{display(r["valid_responses"])}/{r["expected_tasks"]}', r["scored_tasks"],
                     r["task_score_0_100"], r["length_capped"], r["empty_final_answers"],
                     r["measurement_complete"], r["evaluation_complete"], r["generation_budget_consistent"]] for r in rows]
    performance_rows = [[r["run_name"], r["dataset"], f'r{r["repeat"]}/C{r["concurrency"]}',
                         r["input_tokens_mean"], r["output_tokens_mean"], r["ttft_ms_mean"],
                         r["tpot_ms_mean"], r["e2e_s_mean"], r["e2e_s_p95"],
                         r["output_tokens_per_s"], r["preemptions_delta"]] for r in rows]
    text = "# 长输入实验结果速读\n\n"
    text += "本报告只读取已保存的结果，不运行推理或重新评分。先检查测量完整/评分完整，再比较同一任务、同一数据清单与协议。\n\n"
    text += "NA 表示尚无完整结果或指标不可计算；真实 0 分保留为 0。CSV 的空白数值对应 NA。未完成点仍按运行计划列出。\n\n"
    text += "## 质量与完整性\n\n" + markdown_table(
        ["运行", "任务", "重复/并发", "有效/应有", "已评分", "质量分0–100", "截断数", "空答案", "测量完整", "评分完整", "预算一致"], quality_rows)
    text += "\n\n## 性能\n\n" + markdown_table(
        ["运行", "任务", "重复/并发", "输入均值token", "输出均值token", "TTFT均值ms", "TPOT均值ms", "E2E均值s", "E2E p95 s", "输出token/s", "抢占增量"], performance_rows)
    text += "\n\n思考与最终答案的全部生成 token 计入输出和 TPOT；TTFT 是首个 reasoning/content 片段，非最终答案首 token 时间。\n\n"
    text += "截断完整响应仍评分，不保证为0；思考未结束且无最终答案按空答案计分。基础设施失败不会伪装成完整0分。\n\n"
    text += "预算一致为否表示原生usage与推导预算存在差异，例如length早于剩余上下文边界。提前length的完整响应仍评分；应核对服务端预算和token计数，不能仅凭该标记断定有隐藏cap。\n\n"
    text += "测量不完整时，表中性能可能来自已收到的部分响应，不能当作完整主对照。质量分仅在评分完整时展示。\n\n"
    text += "不计算跨任务总体准确率，不自动按分数或吞吐判定方法优胜。少量 smoke 样本的p95只作流程诊断。\n\n## 来源与协议\n\n"
    for folder in args.run_dir:
        run = read_object(folder / "run.json", required=True)
        text += f"- `{folder.resolve()}`：method={run['method']}；data_manifest_sha256={run.get('data_manifest_sha256')}；settings=`{json.dumps(run['settings'], ensure_ascii=False, sort_keys=True)}`\n"
    mdpath.write_text(text, encoding="utf-8")
    print(json.dumps({"points": len(rows), "overview_markdown": str(mdpath.resolve()),
                      "overview_csv": str(csvpath.resolve()),
                      "complete_quality_points": sum(r["evaluation_complete"] for r in rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
