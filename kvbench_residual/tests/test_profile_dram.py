"""CPU tests of hardware-counter parsing; fixture numbers are not measurements."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from kvbench_residual.profile_dram import METRICS, error_class, gpu_idle, parse_ncu_csv, validate_profile_case


def counter_fixture(missing=None, duplicate=False):
    stream = io.StringIO()
    stream.write("==PROF== Synthetic parser fixture; never executed on GPU\n")
    writer = csv.writer(stream)
    writer.writerow(["ID", "Process ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    for launch in ["0", "1"]:
        for metric, unit, value in zip(METRICS, ["byte", "byte", "nsecond"], ["1,024", "0", "2,000"]):
            if metric == missing:
                continue
            row = [launch, "123", "fake_kernel_" + launch, metric, unit, value]
            writer.writerow(row)
            if duplicate and launch == "0" and metric == METRICS[0]:
                writer.writerow(row)
    return stream.getvalue()


class DrAMParserTests(unittest.TestCase):
    def test_counters_are_summed_by_launch_and_real_zero_is_kept(self):
        result = parse_ncu_csv(counter_fixture())
        self.assertEqual(result["kernel_launches"], 2)
        self.assertEqual(result["dram_bytes_read"], 2048)
        self.assertEqual(result["dram_bytes_write"], 0)
        self.assertAlmostEqual(result["profiled_kernel_duration_s"], 4e-6)

    def test_missing_duplicate_or_non_numeric_counters_are_not_silently_zero(self):
        for raw in ["==ERROR== ERR_NVGPUCTRPERM", counter_fixture(missing=METRICS[1]),
                    counter_fixture(duplicate=True), counter_fixture().replace("2,000", "N/A")]:
            with self.subTest(raw=raw[:60]), self.assertRaises(ValueError):
                parse_ncu_csv(raw)
        self.assertEqual(error_class("ERR_NVGPUCTRPERM", 1), "counter_permission_unavailable")

    def test_profile_receipt_requires_validated_source_policy_shape_and_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile_case.json"
            suite = {"epsilon": .02, "parent_pins": {"decode": "parent"},
                     "generated_sha256": {"bound_kv": {"decode": "transformed"}},
                     "kernel_parent_source_root": str(Path(tmp) / "source")}
            shape = {"batch": 8, "length": 8192, "mode": "split", "block_n": 32, "splits": 32}
            receipt = {"schema": "kvbench.residual.profile_case.v1", "status": "completed", "gpu_executed": True, "policy": "bound_kv",
                       "source_root": suite["kernel_parent_source_root"], "profiled_launches": 2,
                       **{k: suite[k] for k in ["epsilon", "parent_pins", "generated_sha256"]}, **shape}
            path.write_text(json.dumps(receipt))
            validate_profile_case(path, suite, "bound_kv", shape)
            for change in [{"policy": "anchor"}, {"gpu_executed": False}, {"epsilon": .5},
                           {"length": 4096}, {"profiled_launches": 1}, {"generated_sha256": {}}]:
                path.write_text(json.dumps({**receipt, **change}))
                with self.assertRaises(ValueError):
                    validate_profile_case(path, suite, "bound_kv", shape)

    def test_occupied_gpu_is_not_used_or_stopped(self):
        result = SimpleNamespace(returncode=0, stdout="991, old_vllm\n", stderr="")
        with patch("kvbench_residual.profile_dram.subprocess.run", return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                gpu_idle({"gpu": "0", "port": 8000}, {})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "nvidia-smi")


if __name__ == "__main__":
    unittest.main()
