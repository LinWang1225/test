"""CPU tests for orchestration gates; these never claim GPU validation."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvbench_residual.suite import (
    METHODS, OLD_METHODS, POLICIES, PROTOCOL, TASKS, Suite,
    upstream_complete, validate_gpu_result, validate_protocol, main, validate_microbench_result,
)
from kvbench_longcontext.common import atomic_json, checked_data, sha_file, write_jsonl


REPO = Path(__file__).resolve().parents[2]


def fixture(root, exit_code=0):
    old = root / "old"
    data_dir = old / "data"
    data_dir.mkdir(parents=True)
    model = root / "model"
    model.mkdir()
    settings = {"max_model_len": 32768, "max_input_tokens": None, "min_input_tokens": 0,
                "max_new_tokens": None, "generation_budget_policy": "remaining-context",
                "sample_size": 0, "data_seed": 42, "overflow": "skip", "thinking": True}
    manifest = {"schema": "kvbench.longcontext.data.v1", "variant": "longbench", "settings": settings,
                "generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "datasets": {}}
    for task in TASKS:
        rows = [{"case_id": f"{task}/{i}", "input_tokens": 128, "max_tokens": None,
                 "max_model_len": 32768, "generation_budget_policy": "remaining-context",
                 "remaining_context_tokens": 32640, "generation_budget_tokens": 32640} for i in range(8)]
        file = data_dir / (task + ".jsonl")
        write_jsonl(file, rows)
        manifest["datasets"][task] = {"file": file.name, "sha256": sha_file(file), "selected_count": 8,
                                     "eligible_count": 8, "source_count": 8, "excluded_count": 0, "truncated_count": 0,
                                     "max_tokens": None, "selected_case_ids": [r["case_id"] for r in rows]}
    atomic_json(data_dir / "manifest.json", manifest)
    data_sha = sha_file(data_dir / "manifest.json")
    cfg = {**PROTOCOL, "batch": str(old), "data": str(data_dir), "root": str(root), "repo": str(REPO),
           "model": str(model), "methods": OLD_METHODS, "tasks": TASKS, "variant": "longbench", "gpu": "0", "port": 8000,
           "client_env": "evalscope-client", "server_env": "kvarn-serve", "kit": str(root / "kit"), "startup_timeout": 1}
    methods = []
    for method in OLD_METHODS:
        folder = old / "runs" / method
        atomic_json(folder / "run.json", {"method": method, "data_manifest_sha256": data_sha,
                                          "scripts_sha256": {"run.py": sha_file(REPO / "kvbench_longcontext/run.py")}})
        atomic_json(folder / "server.launch.json", {})
        atomic_json(folder / "grading_protocol.json", {})
        methods.append({"method": method, "state": "completed", "run_dir": str(folder)})
    upstream = {"schema": "kvbench.longcontext.full_suite.v1", "started": "2026-10-09T00:00:00+00:00",
                "finished": "2026-10-09T01:00:00+00:00", "exit_code": exit_code,
                "settings": cfg, "methods": methods, "data_manifest_sha256": data_sha}
    path = old / "suite.json"
    atomic_json(path, upstream)
    return path, upstream


PASSED = {"schema": "kvbench.residual.validation.v1", "status": "passed", "passed": True, "gpu_executed": True,
          "tested_policies": POLICIES, "tested_value_bits": [4], "epsilon": 0.02,
          "tested_groups": [64, 128], "tested_modes": ["single", "split"], "tested_block_ns": [16, 32, 64],
          **{key: True for key in ["autotune_executed", "cuda_graph_replay_executed", "graph_updated_input_checked",
                                  "graph_updated_demand_checked", "packed_consumers_checked", "packed_writers_checked"]}}


class SimulatedSuite(Suite):
    """Drive the real phase/method scheduler without environments or GPU work."""
    def __init__(self, *args, validation_pass=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []
        self.validation_pass = validation_pass

    def preflight(self):
        self.client = self.server = {"python": "unused", "env": {}}
        self.calls.append("preflight")

    def build_runtimes(self):
        self.runtimes = {policy: "unused" for policy in POLICIES}
        self.calls.append("runtime")

    def kernel_checks(self):
        self.calls.append("gpu_correctness")
        value = dict(PASSED)
        value["passed"] = self.validation_pass
        path = self.out / "validation/validation.json"
        atomic_json(path, value)
        validate_gpu_result(path)
        self.calls.append("microbench")

    def start_server(self, row):
        self.calls.append("start:" + row["method"])
        path = self.out / "servers" / row["method"] / "launch.json"
        atomic_json(path, {"config": {"method": row["method"]}})
        return path

    def stop_server(self):
        self.calls.append("stop_own")

    def implementation_evidence(self, row):
        return {"verified": True, "scope": "CPU simulation fixture, no GPU claim"}

    def call(self, runtime, args, label):
        args = list(map(str, args))
        self.calls.append((label, args))
        if args[1] == "kvbench_longcontext.run":
            folder = Path(args[args.index("--out") + 1])
            points = [{"id": "r01_" + task + "_C8", "dataset": task, "repeat": 1,
                       "concurrency": 8, "expected_tasks": 8} for task in TASKS]
            atomic_json(folder / "run.json", {"method": folder.name, "data_manifest_sha256": self.meta["data_manifest_sha256"], "points": points})
            for point in points:
                dest = folder / point["id"]
                atomic_json(dest / "performance.summary.json", {"measurement_complete": True, "valid_responses": 8,
                                                                 "length_capped": 1, "generation_budget_consistent": True})
                atomic_json(dest / "scores.summary.json", {"evaluation_complete": True, "scored_tasks": 8})
        return 0


class SuiteTests(unittest.TestCase):
    def test_plan_reads_identity_but_creates_no_outputs_or_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            output = io.StringIO()
            with patch("kvbench_residual.suite.resolve_environment") as resolve, redirect_stdout(output):
                self.assertEqual(main(["--after-suite", str(path), "--out", str(root / "new"), "--plan"]), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual([m["method"] for m in plan["methods"]], [m["method"] for m in METHODS])
            self.assertEqual(plan["epsilon"], .02)
            self.assertFalse((root / "new").exists())
            resolve.assert_not_called()

    def test_explicit_upstream_terminal_gate(self):
        value = {"methods": [{"method": m, "state": "completed"} for m in OLD_METHODS]}
        self.assertFalse(upstream_complete(value))
        value["finished"] = "now"
        self.assertFalse(upstream_complete(value))
        value["exit_code"] = 0
        self.assertTrue(upstream_complete(value))
        value["methods"][-1]["state"] = "running"
        with self.assertRaises(ValueError):
            upstream_complete(value)
        value["exit_code"] = 1
        with self.assertRaisesRegex(RuntimeError, "no|not", msg="Failed old suite must not authorize GPU work"):
            upstream_complete(value)

    def test_gpu_only_and_full_coverage_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.json"
            atomic_json(path, PASSED)
            self.assertTrue(validate_gpu_result(path)["passed"])
            for change in [{"gpu_executed": False}, {"passed": False}, {"status": "skipped"},
                           {"tested_value_bits": [2]}, {"tested_policies": ["parent", "bound_kv"]}, {"epsilon": 0.5},
                           {"packed_consumers_checked": False}, {"graph_updated_demand_checked": False}]:
                atomic_json(path, {**PASSED, **change})
                with self.assertRaises(RuntimeError):
                    validate_gpu_result(path)
            hashes = {policy: {"decode.py": policy + "-hash"} for policy in POLICIES}
            source = Path(tmp) / "parent"
            pins = {"decode.py": "parent-hash"}
            evidence = {**PASSED, "source_root": str(source), "parent_pins": pins,
                        "generated_sha256": hashes}
            atomic_json(path, evidence)
            validate_gpu_result(path, source, pins, hashes)
            with self.assertRaisesRegex(RuntimeError, "policies differ"):
                validate_gpu_result(path, source, pins, {**hashes, "bound_kv": {"decode.py": "another-hash"}})

    def test_microbench_fixed_matrix_has_no_speed_selection_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark.json"
            rows = [{"batch": batch, "context_tokens": length, "mode": mode, "policy": policy,
                     "group": 128, "value_bits": 4, "fixture_kind": "random", "latency_us_median":
                     100 if policy == "parent" else 1000, "speedup_vs_parent": 1 if policy == "parent" else .1}
                    for batch in [1, 8] for length in [1024, 4096, 8192, 16384, 32768]
                    for mode in ["single", "split"] for policy in POLICIES]
            value = {"schema": "kvbench.residual.microbenchmark.v1", "status": "completed", "passed": True,
                     "gpu_executed": True, "source_root": tmp, "parent_pins": {"decode": "a"},
                     "generated_sha256": {"parent": {"decode": "b"}}, "epsilon": .02, "rows": rows,
                     "writer_packing": [{"policy": "native_int4"}, {"policy": "tile_high2_low2"}]}
            atomic_json(path, value)
            # A deliberately ten-times-slower candidate is still a complete experiment.
            validate_microbench_result(path, tmp, value["parent_pins"], value["generated_sha256"], .02)
            for change in [{"rows": rows[:-1]}, {"gpu_executed": False}, {"status": "profile_case_only"}]:
                atomic_json(path, {**value, **change})
                with self.assertRaises(RuntimeError):
                    validate_microbench_result(path, tmp, value["parent_pins"], value["generated_sha256"], .02)

    def test_freeze_rejects_mutation_and_protocol_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, upstream = fixture(root)
            manifest, data = checked_data(root / "old/data")
            validate_protocol(upstream, manifest, data)
            altered = copy.deepcopy(manifest)
            altered["settings"]["max_new_tokens"] = 8192
            with self.assertRaises(ValueError):
                validate_protocol(upstream, altered, data)
            with redirect_stdout(io.StringIO()):
                suite = Suite(path, root / "new")
                suite.freeze_upstream(upstream)
            target = root / "old/data/qasper.jsonl"
            target.write_text(target.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                suite.assert_upstream_unchanged()

    def test_upstream_failure_never_enters_preflight(self):
        with tempfile.TemporaryDirectory() as tmp, patch("kvbench_residual.suite.signal.signal"), redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path, _ = fixture(root, exit_code=1)
            suite = SimulatedSuite(path, root / "new")
            self.assertEqual(suite.execute(), 1)
            self.assertNotIn("preflight", suite.calls)
            self.assertNotIn("gpu_correctness", suite.calls)
            self.assertTrue(all(row["state"] == "not_started" for row in suite.meta["methods"]))

    def test_failed_correctness_blocks_all_full_runs(self):
        with tempfile.TemporaryDirectory() as tmp, patch("kvbench_residual.suite.signal.signal"), redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path, _ = fixture(root)
            suite = SimulatedSuite(path, root / "new", validation_pass=False)
            self.assertEqual(suite.execute(), 1)
            self.assertIn("gpu_correctness", suite.calls)
            self.assertNotIn("microbench", suite.calls)
            self.assertFalse(any(isinstance(x, str) and x.startswith("start:") for x in suite.calls))
            self.assertTrue(all(row["state"] == "not_started" for row in suite.meta["methods"]))

    def test_all_six_groups_once_same_data_and_null_protocol(self):
        with tempfile.TemporaryDirectory() as tmp, patch("kvbench_residual.suite.signal.signal"), redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path, upstream = fixture(root)
            suite = SimulatedSuite(path, root / "new")
            self.assertEqual(suite.execute(), 0)
            started = [x.removeprefix("start:") for x in suite.calls if isinstance(x, str) and x.startswith("start:")]
            self.assertEqual(started, [m["method"] for m in METHODS])
            runs = [(label, args) for call in suite.calls if isinstance(call, tuple)
                    for label, args in [call] if label.endswith(".run")]
            self.assertEqual(len(runs), 6)
            for label, args in runs:
                self.assertEqual(args[args.index("--data-dir") + 1], upstream["settings"]["data"])
                for flag, value in {"--concurrency": "8", "--repeats": "1", "--temperature": "0.6",
                                    "--top-p": "0.95", "--top-k": "20", "--min-p": "0",
                                    "--seed": "20261009", "--request-timeout": "0"}.items():
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertNotIn("--max-new-tokens", args)
            comparison = next(args for call in suite.calls if isinstance(call, tuple)
                              for label, args in [call] if label == "comparison")
            sources = comparison[comparison.index("--run-dir") + 1:comparison.index("--out")]
            self.assertEqual(len(sources), 15)
            self.assertEqual(suite.meta["methods"][0]["length_capped"], 6)
            self.assertTrue(all(row["state"] == "completed" and row["scored_tasks"] == 48 for row in suite.meta["methods"]))

    def test_missing_dispatch_evidence_preserves_scores_but_not_verified_state(self):
        class MissingEvidenceSuite(SimulatedSuite):
            def implementation_evidence(self, row):
                if row["method"] == "ar_k4v4_bound_kv":
                    raise RuntimeError("Synthetic missing decode-driver receipt")
                return super().implementation_evidence(row)

        with tempfile.TemporaryDirectory() as tmp, patch("kvbench_residual.suite.signal.signal"), redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path, _ = fixture(root)
            suite = MissingEvidenceSuite(path, root / "new")
            self.assertEqual(suite.execute(), 1)
            row = next(r for r in suite.meta["methods"] if r["method"] == "ar_k4v4_bound_kv")
            self.assertEqual(row["state"], "implementation_unverified")
            self.assertFalse(row["implementation_verified"])
            self.assertEqual(row["scored_tasks"], 48)
            self.assertEqual(row["length_capped"], 6)
            self.assertEqual(row["evaluation_complete_points"], 6)


if __name__ == "__main__":
    unittest.main()
