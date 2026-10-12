import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from kvbench_residual.effects import collect_effects, main
from kvbench_residual.suite import METHODS
from kvbench_longcontext.common import atomic_json


def make_results(root):
    settings = {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0,
                "generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null",
                "request_seed_base": 20261009, "request_timeout": 0, "warmup": 1, "warmup_per_point": -1,
                "warmup_output_cap": 32, "retries": 0}
    config = {"max_model_len": 32768, "max_seqs": 16, "batch_tokens": 1024, "gpu_util": 0.85,
              "kv_bytes": None, "eager": False, "server_seed": 20260917, "apc": False,
              "generation_config": "vllm", "override_generation_config": {}, "kv_dtype": "kvarn_k4v4_g128", "block_size": 128, "residual_epsilon": 0.02}
    metadata = {"schema": "kvbench.residual.full_suite.v1", "output": str(root), "data_manifest_sha256": "frozen-hash", "methods": []}
    point = {"id": "r01_qasper_C8", "dataset": "qasper", "repeat": 1, "concurrency": 8, "expected_tasks": 10}
    for method in METHODS:
        name = method["method"]
        folder = root / "runs" / name
        exists = name in ["ar_k4v4_parent", "ar_k4v4_bound_kv"]
        metadata["methods"].append({**method, "run_dir": str(folder), "state": "completed" if exists else "startup_failed",
                                    "implementation_verified": exists})
        if not exists:
            continue
        target = name.endswith("bound_kv")
        atomic_json(folder / "run.json", {"schema": "kvbench.longcontext.run.v1", "method": name,
                                          "settings": settings, "data_manifest_sha256": "frozen-hash", "points": [point]})
        atomic_json(folder / "server.launch.json", {"config": {**config, "residual_policy": method["policy"]}, "tokenizer": {"fingerprint": "same-model"}, "environment": {},
                                                        "residual": {"parent_pins": {"decode.py": "same-reviewed-source"}}})
        perf = {**point, "method": name, "measurement_complete": True, "valid_responses": 10,
                "generation_budget_consistent": True, "length_capped": 1 if target else 2,
                "empty_final_answers": 1 if target else 0, "preemptions_delta": 0 if target else 2,
                "prompt_tokens": {"mean": 5000}, "completion_tokens": {"mean": 80 if target else 100},
                "tpot_s": {"n": 10, "mean": .008 if target else .010}, "ttft_s": {"mean": 1, "p95": 2},
                "e2e_s": {"mean": 4.5 if target else 5, "p95": 7}, "output_tokens_per_s": 125 if target else 100}
        quality = {**point, "evaluation_complete": True, "scored_tasks": 10, "score_name": "qa_f1",
                   "task_score_0_100": 5 if target else 0}
        atomic_json(folder / point["id"] / "performance.summary.json", perf)
        atomic_json(folder / point["id"] / "scores.summary.json", quality)
    path = root / "suite.json"
    atomic_json(path, metadata)
    return path


def main_row(rows):
    return next(row for row in rows if (row["comparison"], row["precision"], row["dataset"]) == ("main_net", "k4v4", "qasper"))


class EffectsTests(unittest.TestCase):
    def test_fixed_plan_zero_score_and_ratio_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_results(root)
            _, rows = collect_effects(path)
            self.assertEqual(len(rows), 36)
            row = main_row(rows)
            self.assertTrue(row["comparison_complete"])
            self.assertEqual(row["baseline_score_0_100"], 0)
            self.assertEqual(row["score_delta_pp"], 5)
            self.assertAlmostEqual(row["tpot_improvement_percent"], 20)
            self.assertAlmostEqual(row["output_tps_gain_percent"], 25)
            self.assertAlmostEqual(row["output_length_change_percent"], -20)
            self.assertAlmostEqual(row["e2e_change_percent"], -10)
            self.assertEqual(row["length_rate_delta_pp"], -10)
            self.assertEqual(row["empty_final_delta"], 1)
            self.assertEqual(row["preemptions_delta"], -2)
            missing = next(r for r in rows if r["comparison"] == "key_selection")
            self.assertFalse(missing["comparison_complete"])
            self.assertIsNone(missing["score_delta_pp"])
            self.assertIsNone(missing["tpot_improvement_percent"])
            with redirect_stdout(io.StringIO()):
                main(["--suite", str(path), "--out", str(root / "comparison")])
            with (root / "comparison/residual_effects.csv").open(encoding="utf-8-sig", newline="") as f:
                csvrows = list(csv.DictReader(f))
            self.assertEqual(len(csvrows), 36)
            self.assertEqual(csvrows[0]["baseline_score_0_100"], "0")
            self.assertEqual(csvrows[1]["score_delta_pp"], "")

    def test_partial_or_protocol_mismatch_never_becomes_main_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_results(root)
            folder = root / "runs/ar_k4v4_bound_kv"
            perfpath = folder / "r01_qasper_C8/performance.summary.json"
            perf = json.loads(perfpath.read_text())
            perf["measurement_complete"] = False
            atomic_json(perfpath, perf)
            row = main_row(collect_effects(path)[1])
            self.assertFalse(row["comparison_complete"])
            self.assertIsNone(row["score_delta_pp"])
            self.assertIsNone(row["output_tps_gain_percent"])
            perf["measurement_complete"] = True
            atomic_json(perfpath, perf)
            runpath = folder / "run.json"
            run = json.loads(runpath.read_text())
            run["settings"]["request_seed_base"] += 1
            atomic_json(runpath, run)
            row = main_row(collect_effects(path)[1])
            self.assertFalse(row["protocol_match"])
            self.assertIsNone(row["tpot_improvement_percent"])
            self.assertIn("request_seed_base", row["unavailable_reason"])


if __name__ == "__main__":
    unittest.main()
