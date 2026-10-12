"""Reporting must not hide planned missing points or conflate zero and missing."""
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from kvbench_longcontext import report


class ReportTests(unittest.TestCase):
    def test_zero_score_capped_output_and_missing_points_remain_distinct(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "fp16_test"
            run_dir.mkdir()
            points = [{"id": "r01_qasper_C1", "dataset": "qasper", "repeat": 1,
                       "concurrency": 1, "expected_tasks": 1},
                      {"id": "r01_hotpotqa_C1", "dataset": "hotpotqa", "repeat": 1,
                       "concurrency": 1, "expected_tasks": 1}]
            (run_dir / "run.json").write_text(json.dumps({"schema": "kvbench.longcontext.run.v1",
                "method": "fp16", "points": points, "data_manifest_sha256": "fixture-data",
                "settings": {"thinking": True, "generation_budget_policy": "remaining-context",
                             "temperature": .6, "top_p": .95, "top_k": 20, "min_p": 0}}))
            dest = run_dir / points[0]["id"]
            dest.mkdir()
            (dest / "performance.summary.json").write_text(json.dumps({**points[0], "method": "fp16",
                "measurement_complete": True, "valid_responses": 1, "length_capped": 1,
                "generation_budget_consistent": False, "unexpected_length_before_budget_count": 1,
                "empty_final_answers": 1, "tpot_s": {"n": 1, "mean": .012},
                "ttft_s": {"mean": .25, "p95": .25}}))
            (dest / "scores.summary.json").write_text(json.dumps({**points[0], "evaluation_complete": True,
                "scored_tasks": 1, "task_score_0_100": 0.0, "score_name": "qa_f1",
                "missing_tasks": 0, "invalid_response_tasks": 0, "unscored_tasks": 0}))
            rows = report.collect_run(run_dir)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["task_score_0_100"], 0)
            self.assertEqual(rows[0]["length_capped_percent_of_valid"], 100)
            self.assertEqual(rows[0]["tpot_ms_mean"], 12)
            self.assertEqual(rows[0]["ttft_ms_mean"], 250)
            self.assertEqual(rows[0]["unexpected_length_before_budget_count"], 1)
            self.assertIs(rows[0]["generation_budget_consistent"], False)
            self.assertIsNone(rows[1]["task_score_0_100"])
            self.assertFalse(rows[1]["measurement_complete"])
            self.assertFalse(rows[1]["evaluation_complete"])
            with redirect_stdout(io.StringIO()):
                report.main(["--run-dir", str(run_dir), "--out", str(root / "readout")])
            with (root / "readout/overview.csv").open(encoding="utf-8-sig", newline="") as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(saved[0]["task_score_0_100"], "0.0")
            self.assertEqual(saved[0]["unexpected_length_before_budget_count"], "1")
            self.assertEqual(saved[1]["task_score_0_100"], "")
            self.assertIn("hotpotqa", (root / "readout/overview.md").read_text())
            # A wrongly matched result must not be attached to the planned task.
            (dest / "scores.summary.json").write_text(json.dumps({**points[0], "dataset": "other"}))
            with self.assertRaises(ValueError):
                report.collect_run(run_dir)


if __name__ == "__main__":
    unittest.main()
