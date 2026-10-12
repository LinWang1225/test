"""CPU-only isolation/provenance tests, including a real spawn subprocess.

The fake vLLM package contains no torch, Triton or CUDA code. Its fake native
library is never loaded; it only exercises the runtime's binary-link policy.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from kvbench_residual import runtime, trace


FAKE_KERNEL = '''"""CPU stand-in for the pinned attention module."""
class FakeKernel:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return "cpu-only-fake-launch"
        return launch

_kvarn_fused_decode_stage1 = FakeKernel()

def kvarn_decode_attention(query):
    _kvarn_fused_decode_stage1[(1, 8, 32)](
        D=128, GROUP=128, Q_PER_KV=4, K_BITS=4, V_BITS=4,
        NUM_KV_SPLITS=32)
    return query
'''


FAKE_API_SERVER = '''"""No HTTP or GPU: validate bootstrap's real spawn contract."""
import json
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace

def child_entry(result_path):
    import vllm
    import kvbench_residual.trace as receipts
    from vllm.v1.attention.ops import triton_kvarn_decode as attention
    query = SimpleNamespace(shape=(1, 32, 128))
    attention.kvarn_decode_attention(query)
    attention.kvarn_decode_attention(query)
    Path(result_path).write_text(json.dumps({
        "pid": os.getpid(),
        "vllm_origin": vllm.__file__,
        "kernel_origin": attention.__file__,
        "trace_origin": receipts.__file__,
        "policy": attention.AR_POLICY,
        "environment_policy": os.environ.get("AR_POLICY"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "cwd": os.getcwd(),
        "torch_imported": "torch" in sys.modules,
    }), encoding="utf-8")

if __name__ == "__main__":
    if sys.argv[1:] != ["--forwarded", "value with spaces"]:
        raise RuntimeError("bootstrap did not forward API arguments exactly")
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=child_entry, args=(os.environ["AR_TEST_RESULT"],))
    child.start()
    child.join(15)
    if child.is_alive():
        child.terminate()
        child.join(5)
        raise RuntimeError("fake spawned worker did not finish")
    if child.exitcode != 0:
        raise RuntimeError("fake spawned worker failed: " + str(child.exitcode))
    print("isolated spawn worker completed", flush=True)
'''


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ar_runtime_test_", dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.package = self.root / "installed" / "vllm"
        self.package.mkdir(parents=True)
        self.write("__init__.py", "INIT_EXECUTED = True\n")
        self.write(str(runtime.TARGET), FAKE_KERNEL)
        self.write("entrypoints/openai/api_server.py", FAKE_API_SERVER)
        self.companions = {}
        for index, name in enumerate(["v1/attention/ops/kvarn_store.py", "v1/attention/backends/kvarn_attn.py"]):
            self.write(name, f"COMPANION = {index}\n")
            self.companions[name] = runtime.sha_file(self.package / name)
        self.binary = self.package / "native_fixture.so"
        self.binary.write_bytes(b"NOT_AN_ELF: never imported by these tests")
        self.digest = hashlib.sha256(FAKE_KERNEL.encode()).hexdigest()
        self.out = self.root / "runtime"

    def write(self, name, content):
        path = self.package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        directory = path.parent
        while directory != self.package.parent:
            init = directory / "__init__.py"
            if not init.exists():
                init.write_text("", encoding="utf-8")
            directory = directory.parent
        path.write_text(content, encoding="utf-8")

    def build(self, policy="bound_k", out=None, copy_hook=None, discover=False):
        def transform(sources, selected, epsilon=0.02):
            return {k: value + "\nAR_POLICY = " + repr(selected) + "\n" for k, value in sources.items()}

        with ExitStack() as stack:
            stack.enter_context(patch("kvbench_residual.transform.get_expected_pins", return_value={str(runtime.TARGET): self.digest, **self.companions}))
            stack.enter_context(patch("kvbench_residual.transform.transform_sources", transform))
            stack.enter_context(patch.object(runtime, "_source_identity", return_value={
                "commit": {"exit_code": 0, "stdout": runtime.PARENT_COMMIT, "stderr": ""},
                "status": {"exit_code": 0, "stdout": "", "stderr": ""},
            }))
            if copy_hook:
                stack.enter_context(patch.object(runtime, "_copy_package", copy_hook))
            if discover:
                spec = types.SimpleNamespace(origin=str(self.package / "__init__.py"))
                stack.enter_context(patch.object(runtime.importlib.util, "find_spec", return_value=spec))
            return runtime.build_runtime(out or self.out, policy,
                                         package=None if discover else self.package)

    def snapshot(self):
        return {str(p.relative_to(self.package)): p.read_bytes()
                for p in self.package.rglob("*") if p.is_file()}

    def verify(self, root=None, policy="bound_k", full=False):
        return runtime.verify_runtime(root or self.out, policy, full=full)


class RuntimeIsolationTests(RuntimeFixture):
    def test_build_freezes_sources_without_importing_or_writing_installed_package(self):
        before = self.snapshot()
        imported_before = set(sys.modules)
        result = self.build(discover=True)
        self.assertEqual(before, self.snapshot())
        self.assertFalse(result["cuda_initialized"])
        self.assertFalse({"torch", "vllm"} & (set(sys.modules) - imported_before))
        copied_kernel = self.out / "vllm" / runtime.TARGET
        self.assertFalse(copied_kernel.is_symlink())
        self.assertNotEqual(copied_kernel.read_bytes(), (self.package / runtime.TARGET).read_bytes())
        copied_binary = self.out / "vllm" / self.binary.name
        self.assertTrue(copied_binary.is_symlink())
        self.assertEqual(copied_binary.resolve(), self.binary)
        self.assertEqual(result["executed_source_sha256"], runtime.sha_file(copied_kernel))
        self.verify(full=True)

        # A later checkout/source edit does not change an already frozen
        # Python module. Native code is intentionally shared and checked below.
        frozen = (self.out / "vllm/__init__.py").read_bytes()
        self.write("__init__.py", "INIT_EXECUTED = 'edited later'\n")
        self.assertEqual((self.out / "vllm/__init__.py").read_bytes(), frozen)
        self.verify(full=True)

    def test_unknown_kernel_or_companion_rejected_before_destination_creation(self):
        original = (self.package / runtime.TARGET).read_bytes()
        (self.package / runtime.TARGET).write_bytes(original + b"\n# unreviewed change\n")
        with self.assertRaisesRegex(ValueError, "Unsupported KVarN source"):
            self.build()
        self.assertFalse(self.out.exists())
        (self.package / runtime.TARGET).write_bytes(original)
        companion = self.package / next(iter(self.companions))
        companion.write_text("COMPANION = 'unreviewed'\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_alias_into_source_and_reuse_of_destination_are_rejected(self):
        alias = self.root / "source_alias"
        alias.symlink_to(self.package, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.build(out=alias / "nested_runtime")
        self.assertFalse((self.package / "nested_runtime").exists())
        self.build()
        with self.assertRaisesRegex(ValueError, "must not exist"):
            self.build()

    def test_kernel_helper_companion_and_shared_binary_changes_are_detected(self):
        self.build()
        paths = [self.out / "vllm" / runtime.TARGET,
                 self.out / "kvbench_residual/transform.py",
                 self.out / "vllm" / next(iter(self.companions))]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"\n# runtime modified after build\n")
                with self.assertRaises(ValueError):
                    self.verify(full=False)
                path.write_bytes(original)
        old = self.binary.stat()
        os.utime(self.binary, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(ValueError, "binary changed"):
            self.verify()

    def test_policy_mismatch_and_relocated_runtime_rejected(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "policy"):
            self.verify(policy="bound_kv")
        moved = self.root / "moved_runtime"
        shutil.move(str(self.out), moved)
        with self.assertRaisesRegex(ValueError, "relocated"):
            self.verify(root=moved)

    def test_failed_copy_is_marked_and_has_no_ready_manifest(self):
        def failing_copy(source, destination):
            destination.mkdir(parents=True)
            raise OSError("simulated interrupted construction")

        before = self.snapshot()
        with self.assertRaisesRegex(OSError, "interrupted"):
            self.build(copy_hook=failing_copy)
        self.assertTrue((self.out / "BUILD_FAILED.txt").is_file())
        self.assertFalse((self.out / "runtime.json").exists())
        self.assertEqual(self.snapshot(), before)

    def test_bootstrap_and_real_spawn_worker_use_same_frozen_overlay(self):
        result = self.build()
        # The child uses the actual frozen verification module and the
        # manifest's copied-source hash table. The tests above separately
        # exercise strict source pin checking at construction time.
        marker = self.root / "spawn_result.json"
        receipt_dir = self.root / "receipts"
        env = os.environ.copy()
        env.update(PYTHONPATH=str(self.out), PYTHONNOUSERSITE="1",
                   PYTHONDONTWRITEBYTECODE="1", AR_TEST_RESULT=str(marker))
        env.pop("PYTHONHOME", None)
        command = [sys.executable, "-m", "kvbench_residual.bootstrap",
                   "--runtime", str(self.out), "--policy", "bound_k",
                   "--trace-dir", str(receipt_dir), "--",
                   "--forwarded", "value with spaces"]
        completed = subprocess.run(command, cwd=self.out, env=env, text=True,
                                   capture_output=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        child = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(Path(child["vllm_origin"]).resolve(), self.out / "vllm/__init__.py")
        self.assertEqual(Path(child["kernel_origin"]).resolve(), self.out / "vllm" / runtime.TARGET)
        self.assertEqual(Path(child["trace_origin"]).resolve(), self.out / "kvbench_residual/trace.py")
        self.assertEqual(child["pythonpath"], str(self.out))
        self.assertEqual(child["cwd"], str(self.out))
        self.assertEqual((child["policy"], child["environment_policy"]), ("bound_k", "bound_k"))
        self.assertFalse(child["torch_imported"])
        report = trace.summarize(receipt_dir, "bound_k", result["executed_source_sha256"])
        self.assertTrue(report["source_verified"])
        self.assertTrue(report["fused_launch_observed"])
        self.assertTrue(report["decode_driver_observed"])
        self.assertEqual(sum(e["event"] == "first_python_kernel_launch" for e in report["events"]), 1)
        self.assertEqual(sum(e["event"] == "first_python_driver_shape" for e in report["events"]), 1)
        self.assertTrue(all(e["pid"] == child["pid"] for e in report["events"]))
        self.assertIn("No inference-token", report["scope"])
        self.assertNotIn("gpu_token_count", report)


class TraceEvidenceTests(RuntimeFixture):
    def test_failed_launch_has_no_success_receipt_and_success_is_sparse(self):
        manifest = self.build()
        directory = self.root / "trace"

        class FakeKernel:
            fail = True
            calls = 0

            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    self.calls += 1
                    if self.fail:
                        raise RuntimeError("synchronous fake launch failure")
                    return "ok"
                return run

        inner = FakeKernel()
        namespace = {"_kvarn_fused_decode_stage1": inner}

        def driver(query):
            namespace["_kvarn_fused_decode_stage1"][(1, 8, 32)](D=128, GROUP=128)
            return query

        namespace["kvarn_decode_attention"] = driver
        with patch.dict(os.environ, AR_TRACE_DIR=str(directory), AR_RUNTIME_ROOT=str(self.out)):
            trace.install(namespace, self.out / "vllm" / runtime.TARGET, "bound_k")
            query = types.SimpleNamespace(shape=(1, 32, 128))
            with self.assertRaisesRegex(RuntimeError, "synchronous"):
                namespace["kvarn_decode_attention"](query)
            initial = trace.summarize(directory, "bound_k", manifest["executed_source_sha256"])
            self.assertTrue(initial["source_verified"])
            self.assertFalse(initial["fused_launch_observed"])
            self.assertFalse(initial["decode_driver_observed"])
            inner.fail = False
            for _ in range(3):
                namespace["kvarn_decode_attention"](query)
        report = trace.summarize(directory, "bound_k", manifest["executed_source_sha256"])
        self.assertEqual(inner.calls, 4)
        self.assertEqual(sum(e["event"] == "first_python_kernel_launch" for e in report["events"]), 1)
        self.assertEqual(sum(e["event"] == "first_python_driver_shape" for e in report["events"]), 1)
        self.assertIn("Warmup included", report["scope"])
        launch_event = next(e for e in report["events"] if e["event"] == "first_python_kernel_launch")
        self.assertIn("Not a GPU replay count", launch_event["meaning"])

    def test_identityless_and_orphan_events_do_not_verify_a_source(self):
        directory = self.root / "trace"
        directory.mkdir()
        path = directory / "999.jsonl"
        path.write_text(json.dumps({"event": "first_python_kernel_launch",
                                    "kernel": "_kvarn_fused_decode_stage1"}) + "\n", encoding="utf-8")
        report = trace.summarize(directory, "bound_k", "a" * 64)
        self.assertFalse(report["source_verified"])
        self.assertGreater(report["identity_mismatches"], 0)
        orphan = {"event": "first_python_kernel_launch", "pid": 999,
                  "policy": "bound_k", "source": "/unobserved/kernel.py",
                  "source_sha256": "a" * 64, "transformed_source_sha256": "b" * 64,
                  "kernel": "_kvarn_fused_decode_stage1"}
        path.write_text(json.dumps(orphan) + "\n", encoding="utf-8")
        report = trace.summarize(directory, "bound_k", "a" * 64)
        self.assertFalse(report["source_verified"])
        self.assertGreater(report["identity_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
