"""English LongBench metrics adapted from the pinned official implementation.

See LICENSE.LongBench and SOURCES.md. These are task scores, not pass@1.
No generated code is executed. Missing metric dependencies fail explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import string
from pathlib import Path

from .common import (LONGBENCH_COMMIT, TASKS, atomic_json, checked_data, digest,
                     package_versions, read_jsonl, sha_file, write_jsonl)


def normalize_answer(text):
    text = "".join(c for c in text.lower() if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_f1(prediction, reference):
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def retrieval(prediction, reference):
    matches = re.findall(r"Paragraph (\d+)", reference)
    if not matches:
        raise ValueError("Malformed official retrieval reference: " + reference)
    numbers = re.findall(r"\d+", prediction)
    return sum(x == matches[0] for x in numbers) / len(numbers) if numbers else 0.0


def rouge_l(prediction, reference):
    from rouge import Rouge
    try:
        return Rouge().get_scores([prediction], [reference], avg=True)["rouge-l"]["f"]
    except (ValueError, RecursionError):
        # Official LongBench maps empty/invalid ROUGE inputs to zero.
        return 0.0


def first_code_line(prediction):
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            return line
    return ""


def code_sim(prediction, reference):
    # Fix the optional backend explicitly: fuzzywuzzy otherwise changes to difflib
    # when python-Levenshtein is absent, which can change edit-similarity scores.
    import Levenshtein  # noqa: F401
    from fuzzywuzzy import fuzz
    return fuzz.ratio(first_code_line(prediction), reference) / 100.0


METRICS = {"qa_f1": qa_f1, "retrieval": retrieval, "rouge_l": rouge_l, "code_sim": code_sim}


def check_dependencies(metrics):
    versions = package_versions(["rouge", "fuzzywuzzy", "python-Levenshtein", "Levenshtein", "rapidfuzz"])
    if "rouge_l" in metrics and versions["rouge"] != "1.0.1":
        raise RuntimeError("Scoring requires rouge==1.0.1 in the client environment")
    if "code_sim" in metrics and (versions["fuzzywuzzy"] != "0.18.0" or versions["Levenshtein"] != "0.25.1"):
        raise RuntimeError("LCC scoring requires fuzzywuzzy==0.18.0 and python-Levenshtein==0.25.1")
    return versions


def score_one(row, prediction):
    return max(METRICS[row["metric"]](prediction, ref) for ref in row["answers"])


def score_records(expected, responses):
    wanted = {r["case_id"]: r for r in expected}
    observed = {}
    for response in responses:
        cid = response.get("case_id")
        if cid not in wanted:
            raise ValueError("Unexpected output case: " + str(cid))
        if cid in observed:
            raise ValueError("Duplicate output case: " + str(cid))
        observed[cid] = response
    output, missing, invalid = [], [], []
    for cid, row in wanted.items():
        response = observed.get(cid)
        valid = bool(response and response.get("http_success") and response.get("response_complete")
                     and response.get("protocol_valid") is True)
        if response is None:
            missing.append(cid)
        elif not valid:
            invalid.append(cid)
        prediction = response.get("final_answer", "") if valid else ""
        score = score_one(row, prediction) if valid else None
        output.append({"dataset": row["dataset"], "case_id": cid, "metric": row["metric"],
                       "score": score, "scored": valid, "missing_response": response is None,
                       "finish_reason": response.get("finish_reason") if response else None,
                       "empty_final_answer": not prediction.strip() if valid else None,
                       "input_tokens": row["input_tokens"], "source_length": row.get("source_length"),
                       "was_truncated": row["was_truncated"]})
    values = [r["score"] for r in output if r["scored"]]
    complete = not missing and not invalid
    summary = {"expected_tasks": len(expected), "observed_tasks": len(observed), "scored_tasks": len(values),
               "missing_tasks": len(missing), "invalid_response_tasks": len(invalid),
               "unscored_tasks": len(expected) - len(values), "missing_case_ids": missing,
               "invalid_case_ids": invalid, "evaluation_complete": complete,
               "task_score_0_100": 100 * sum(values) / len(expected) if complete else None,
               "partial_available_score_0_100": 100 * sum(values) / len(values) if values else None,
               "all_expected_zero_filled_score_0_100": 100 * sum(values) / len(expected) if expected else None,
               "score_name": expected[0]["metric"] if expected else None,
               "score_is_binary_accuracy": False, "aggregation": "mean of per-case max over references"}
    bins = {}
    for label, test in [("0-4k", lambda x: x < 4000), ("4-8k", lambda x: 4000 <= x < 8000),
                        ("8k+", lambda x: x >= 8000)]:
        members = [r for r in output if isinstance(r.get("source_length"), (int, float)) and test(r["source_length"])]
        good = [r["score"] for r in members if r["scored"]]
        bins[label] = {"expected": len(members), "scored": len(good),
                       "score_0_100": 100 * sum(good) / len(members) if members and len(good) == len(members) else None}
    summary["official_word_length_bins"] = bins
    summary["bin_unit"] = "original LongBench English words, not local tokenizer tokens"
    token_bins = {}
    for lower, upper in [(0, 4096), (4096, 8192), (8192, 16384), (16384, 24576), (24576, None)]:
        label = f"{lower}-{upper}" if upper else f"{lower}+"
        members = [r for r in output if lower <= r["input_tokens"] and (upper is None or r["input_tokens"] < upper)]
        good = [r["score"] for r in members if r["scored"]]
        token_bins[label] = {"expected": len(members), "scored": len(good),
            "score_0_100": 100 * sum(good) / len(members) if members and len(good) == len(members) else None,
            "mean_effective_input_tokens": sum(r["input_tokens"] for r in members) / len(members) if members else None,
            "truncated_tasks": sum(r["was_truncated"] for r in members)}
    summary["local_input_token_bins"] = token_bins
    summary["local_bin_definition"] = "lower inclusive, upper exclusive; effective prompt tokens including the actual chat template"
    return output, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    a = p.parse_args(argv)
    manifest, datasets = checked_data(a.data_dir)
    run = json.loads((a.run_dir / "run.json").read_text(encoding="utf-8"))
    if run["data_manifest_sha256"] != sha_file(a.data_dir / "manifest.json"):
        raise ValueError("Run and scoring data manifests do not match")
    versions = check_dependencies({r["metric"] for rows in datasets.values() for r in rows})
    all_summaries = []
    for point in run["points"]:
        folder = a.run_dir / point["id"]
        response_path = folder / "requests.jsonl"
        responses = read_jsonl(response_path) if response_path.is_file() else []
        rows, summary = score_records(datasets[point["dataset"]], responses)
        summary.update(point)
        summary["response_file_sha256"] = sha_file(response_path) if response_path.is_file() else None
        write_jsonl(folder / "scores.jsonl", rows)
        atomic_json(folder / "scores.summary.json", summary)
        all_summaries.append(summary)
    atomic_json(a.run_dir / "grading_protocol.json", {"upstream_code_commit": LONGBENCH_COMMIT,
                "scorer_file_sha256": sha_file(__file__), "dependencies": versions,
                "data_manifest_sha256": sha_file(a.data_dir / "manifest.json"),
                "variant": manifest.get("variant"), "policy": "max over references; capped outputs retained; missing outputs never hidden"})
    atomic_json(a.run_dir / "quality.summary.json", all_summaries)
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
