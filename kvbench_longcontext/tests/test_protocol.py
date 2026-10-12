"""Offline protocol tests. Synthetic fixtures are not benchmark results."""
from __future__ import annotations

from contextlib import redirect_stdout
from contextlib import suppress
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
import zipfile

from kvbench_longcontext import prepare, run, scoring, serve
from kvbench_longcontext.common import TASKS, atomic_json, checked_data, package_versions, sha_file, tokenizer_identity


class FakeTokenizer:
    """Character tokens and 4/6 non-thinking/thinking wrapper tokens for boundary tests."""
    chat_template = "test-template"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("ascii"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode("ascii")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=False):
        return [1, 2] + self.encode(messages[0]["content"]) + ([3, 4, 5, 6] if enable_thinking else [3, 4])


def settings(**kw):
    return {"max_model_len": 4096, "max_input_tokens": None, "min_input_tokens": 0,
            "max_new_tokens": None, "generation_budget_policy": "reference", "sample_size": 0, "data_seed": 42,
            "overflow": "error", "thinking": False, **kw}


def source_record(idx, answer="alpha"):
    return {"_id": str(idx), "context": "A report about long inputs. " * 12,
            "input": "Which keyword is described?", "answers": [answer],
            "length": 4000 if idx else 3999, "all_classes": []}


METRICS = ('# TYPE vllm:cache_config_info gauge\n'
           'vllm:cache_config_info{cache_dtype="auto",block_size="128",calculate_kv_scales="False",'
           'enable_prefix_caching="False",num_gpu_blocks="1000"} 1\n'
           'vllm:kv_cache_usage_perc{engine="0"} 0.1\n'
           'vllm:num_preemptions_total{engine="0"} 0\n')


class Handler(BaseHTTPRequestHandler):
    mode = "ok"
    received = []
    model_max_len = 4096

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "mock-longcontext", "max_model_len": self.model_max_len}]}).encode() if self.path == "/v1/models" else METRICS.encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        request = json.loads(body)
        self.received.append({"path": self.path, "request": request, "authorization": self.headers.get("Authorization"),
                              "content_type": self.headers.get("Content-Type"), "raw_body": body.decode()})
        if self.mode == "http_error":
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"busy")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.mode == "slow":
            time.sleep(.05)
        is_chat = self.path.endswith("chat/completions")
        thinking = is_chat and request.get("chat_template_kwargs", {}).get("enable_thinking", False)
        prompt = request["messages"][0]["content"] if is_chat else request["prompt"]
        count = len(prompt) + ((6 if thinking else 4) if is_chat else 0)
        if self.mode == "mismatch":
            count += 1
        pieces = ["<think>synthetic reasoning", "</think>alpha"] if thinking else ["al", "pha"]
        events = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                  {"choices": [{"index": 0, "delta": {"content": pieces[0], "reasoning_content": None}, "text": pieces[0]}]},
                  {"choices": [{"index": 0, "delta": {"content": pieces[1]}, "text": pieces[1]}]}]
        budget = self.model_max_len - count if request["max_tokens"] is None else request["max_tokens"]
        completion_tokens = min(2, budget)
        if self.mode == "length_budget":
            completion_tokens = budget
            if budget == 1:
                events[1]["choices"][0]["delta"]["content"] = "<think>unfinished"
                events[2]["choices"][0]["delta"]["content"] = ""
        if self.mode != "partial":
            events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "length" if self.mode in ("length_budget", "early_length") else "stop"}]})
        if self.mode != "missing_usage":
            events.append({"choices": [], "usage": {"prompt_tokens": count, "completion_tokens": completion_tokens, "total_tokens": count + completion_tokens}})
        with suppress(BrokenPipeError, ConnectionResetError):
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            if self.mode != "partial":
                self.wfile.write(b"data: [DONE]\n\n")


class ProtocolTests(unittest.TestCase):
    def test_input_budget_and_explicit_truncation(self):
        tok = FakeTokenizer()
        self.assertEqual(prepare.fit_prompt(tok, "abcdefghij", "chat", False, 14, "error"),
                         ("abcdefghij", 14, 14, False))
        with self.assertRaises(ValueError):
            prepare.fit_prompt(tok, "abcdefghij", "chat", False, 13, "error")
        self.assertIsNone(prepare.fit_prompt(tok, "abcdefghij", "chat", False, 13, "skip")[0])
        result = prepare.fit_prompt(tok, "abcdefghij", "chat", False, 10, "head-tail")
        self.assertEqual(result, ("abchij", 14, 10, True))

    def test_no_answer_injection_and_fixed_sample(self):
        data = [source_record(i, "SECRET_REFERENCE_SENTINEL") for i in range(10)]
        rows, excluded, count = prepare.prepare_rows(data, "qasper", FakeTokenizer(), settings(sample_size=3))
        reversed_rows, _, _ = prepare.prepare_rows(list(reversed(data)), "qasper", FakeTokenizer(), settings(sample_size=3))
        self.assertEqual([x["case_id"] for x in rows], [x["case_id"] for x in reversed_rows])
        self.assertEqual(count, 10)
        self.assertFalse(excluded)
        for row in rows:
            self.assertNotIn("SECRET_REFERENCE_SENTINEL", row["prompt"])
            self.assertLessEqual(row["input_tokens"] + row["max_tokens"], 4096)
            request = run.make_request(row, "test", thinking=False, temperature=0, top_p=1, top_k=-1)
            self.assertNotIn("answers", request)
            self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(request["max_tokens"], 128)
            self.assertEqual((request["temperature"], request["top_p"], request["top_k"], request["min_p"]),
                             (0.0, 1.0, -1, 0.0))

    def test_sampling_parameter_validation(self):
        for top_k in (-1, 1, 20):
            for min_p in (0.0, 1.0):
                run.validate_sampling(0.6, .95, top_k, min_p)
        for top_k in (0, -2, 1.5, True):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                run.validate_sampling(0.6, .95, top_k, 0.0)
        for min_p in (-.1, 1.1, float("nan"), float("inf")):
            with self.subTest(min_p=min_p), self.assertRaises(ValueError):
                run.validate_sampling(0.6, .95, 20, min_p)

    def test_generation_budget_mode_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "frozen"
            # This must fail before tokenizer import, data download or output write.
            base = ["--model", str(Path(temp) / "unused-model"), "--out", str(out), "--datasets", "qasper"]
            for flags, error in [(["--generation-budget-policy", "fixed"], "requires an explicit"),
                                 (["--generation-budget-policy", "reference"], "also select --no-thinking"),
                                 (["--max-new-tokens", "8192"], "only used with")]:
                with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, error):
                    prepare.main(base + flags)
            self.assertFalse(out.exists())

    def test_official_task_caps_and_code_route(self):
        self.assertEqual({k: TASKS[k]["max_new_tokens"] for k in list(TASKS)[:6]},
                         {"qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32,
                          "2wikimqa": 32, "gov_report": 512, "passage_retrieval_en": 32})
        rows, _, _ = prepare.prepare_rows([source_record(1)], "lcc", FakeTokenizer(), settings())
        req = run.make_request(rows[0], "test")
        self.assertFalse(req["add_special_tokens"])
        self.assertIn("prompt", req)
        self.assertNotIn("messages", req)
        self.assertEqual(rows[0]["input_tokens"], len(rows[0]["prompt"]))
        code = "    if text == '</think>':\n"
        self.assertEqual(run.extract_final(code, "", False, "completion")[0], code)
        self.assertEqual(run.extract_final("<think> unfinished literal", "", False)[0], "<think> unfinished literal")
        self.assertEqual(run.extract_final("<think>reason</think>the literal </think>", "", True)[0], "the literal </think>")
        self.assertEqual(run.extract_final("literal </think>", "reason", True, reasoning_channel=True)[0], "literal </think>")

    def test_qa_retrieval_code_cleaning_and_reference_max(self):
        self.assertEqual(scoring.qa_f1("The red, car!", "red car"), 1)
        self.assertAlmostEqual(scoring.qa_f1("red car", "red"), 2 / 3)
        self.assertEqual(scoring.qa_f1("", ""), 0)
        self.assertEqual(scoring.retrieval("Paragraph 3 or 7", "Paragraph 7"), .5)
        self.assertEqual(scoring.first_code_line("```python\n# comment\nx = 1\n```"), "x = 1")
        self.assertEqual(scoring.score_one({"metric": "qa_f1", "answers": ["wrong", "alpha"]}, "alpha"), 1)

    @unittest.skipUnless(package_versions(["rouge"])["rouge"] == "1.0.1", "real rouge==1.0.1 backend not installed")
    def test_real_rouge_backend_if_installed(self):
        self.assertAlmostEqual(scoring.rouge_l("alpha beta", "alpha beta"), 1, places=6)
        self.assertGreater(scoring.rouge_l("alpha beta", "alpha gamma"), 0)
        self.assertLess(scoring.rouge_l("alpha beta", "alpha gamma"), 1)
        self.assertEqual(scoring.rouge_l("", "alpha beta"), 0)

    @unittest.skipUnless(package_versions(["fuzzywuzzy", "Levenshtein"]) == {"fuzzywuzzy": "0.18.0", "Levenshtein": "0.25.1"},
                         "real fuzzywuzzy==0.18.0 / Levenshtein==0.25.1 backend not installed")
    def test_real_lcc_backend_if_installed(self):
        self.assertEqual(scoring.code_sim("    return x", "    return x"), 1)
        self.assertLess(scoring.code_sim("    return x", "    raise ValueError()"), 1)
        self.assertEqual(scoring.code_sim("```python\n# comment\n    return x\n```", "    return x"), 1)

    def test_missing_partial_duplicates_and_e_bins(self):
        rows, _, _ = prepare.prepare_rows([source_record(i) for i in range(4)], "qasper", FakeTokenizer(), settings())
        for r, length in zip(rows, [3999, 4000, 7999, 8000]):
            r["source_length"] = length
        responses = [{"case_id": rows[0]["case_id"], "http_success": True, "response_complete": True,
                      "protocol_valid": True, "final_answer": "alpha", "finish_reason": "length"},
                     {"case_id": rows[1]["case_id"], "http_success": True, "response_complete": False}]
        output, summary = scoring.score_records(rows, responses)
        self.assertEqual(summary["missing_tasks"], 2)
        self.assertEqual(summary["unscored_tasks"], 3)
        self.assertEqual(summary["invalid_response_tasks"], 1)
        self.assertIsNone(summary["task_score_0_100"])
        self.assertEqual(summary["partial_available_score_0_100"], 100)
        self.assertEqual(summary["all_expected_zero_filled_score_0_100"], 25)
        self.assertEqual(summary["official_word_length_bins"]["4-8k"]["expected"], 2)
        self.assertEqual(summary["local_input_token_bins"]["0-4096"]["expected"], 4)
        self.assertEqual(output[0]["score"], 1)  # capped responses remain scored
        with self.assertRaises(ValueError):
            scoring.score_records(rows, responses + responses[:1])
        with self.assertRaises(ValueError):
            scoring.score_records(rows, [{"case_id": "unknown"}])

    def test_native_cache_validation_and_g64(self):
        config = {"kv_dtype": "auto", "block_size": 128}
        self.assertEqual(serve.verify_cache(METRICS, config)["num_gpu_blocks"], "1000")
        with self.assertRaises(ValueError):
            serve.verify_cache(METRICS, {"kv_dtype": "kvarn_k4v4_g64", "block_size": 64})
        self.assertEqual(serve.METHODS["kvarn4464"], ("kvarn_k4v4_g64", 64))
        with self.assertRaisesRegex(ValueError, "max_model_len"):
            serve.verify_model({"data": [{"id": "test", "max_model_len": 8192}]},
                               {"model_name": "test", "max_model_len": 32768})
        with self.assertRaises(ValueError):
            serve.verify_cache(METRICS.replace('cache_dtype="auto"', 'cache_dtype="fp8_e4m3"'),
                               {"kv_dtype": "fp8_e4m3", "block_size": 128})
        with self.assertRaisesRegex(ValueError, "num_gpu_blocks"):
            serve.verify_cache(METRICS.replace('num_gpu_blocks="1000"', 'num_gpu_blocks="None"'), config)


class HTTPIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Handler.received = []
        Handler.mode = "ok"
        Handler.model_max_len = 4096
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def call_one(self, mode="ok", kind="chat", timeout=3, max_tokens=32, native_max_model_len=None):
        Handler.mode = mode
        row = {"case_id": "mock/0", "dataset": "mock", "prompt": "question", "request_kind": kind,
               "input_tokens": 12 if kind == "chat" else 8, "max_tokens": max_tokens}
        req = run.make_request(row, "mock-longcontext", thinking=False, temperature=0, top_p=1, top_k=-1)
        result = run.stream_request(self.base, row, req, self.root / "response.json.gz", timeout,
                                    {"Authorization": "Bearer offline-test-key"}, native_max_model_len=native_max_model_len)
        return result

    def test_usage_only_chunk_auth_headers_and_content(self):
        result = self.call_one()
        self.assertTrue(result["response_complete"])
        self.assertTrue(result["protocol_valid"])
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["final_answer"], "alpha")
        self.assertEqual(result["token_count_source"], "server_usage")
        with gzip.open(self.root / "response.json.gz", "rt") as f:
            chunks = json.load(f)
        self.assertEqual(chunks[-1]["data"]["choices"], [])
        self.assertEqual(Handler.received[0]["authorization"], "Bearer offline-test-key")
        self.assertEqual(Handler.received[0]["content_type"], "application/json")
        self.assertEqual(Handler.received[0]["request"]["model"], "mock-longcontext")
        self.assertGreaterEqual(result["e2e_s"], result["ttft_s"])
        self.assertIsNotNone(result["tpot_s"])

    def test_completion_route(self):
        result = self.call_one(kind="completion")
        self.assertTrue(result["protocol_valid"])
        self.assertEqual(Handler.received[0]["path"], "/v1/completions")
        self.assertFalse(Handler.received[0]["request"]["add_special_tokens"])
        natural = self.call_one(kind="completion", timeout=0, max_tokens=None, native_max_model_len=4096)
        self.assertTrue(natural["protocol_valid"])
        self.assertIn('"max_tokens":null', Handler.received[-1]["raw_body"])
        self.assertNotIn("max_completion_tokens", Handler.received[-1]["request"])

    def test_zero_and_positive_time_limits_and_early_length_diagnostic(self):
        self.assertTrue(self.call_one(timeout=0)["response_complete"])
        timed = self.call_one(mode="slow", timeout=.01)
        self.assertFalse(timed["response_complete"])
        self.assertIsNone(timed["finish_reason"])
        self.assertIn("timed out", timed["error"])
        with self.assertRaises(TimeoutError):
            list(run.sse_events(iter([b"data: {}\n", b"\n"]), time.monotonic() - 1))
        self.assertEqual(list(run.sse_events(iter([b"data: {}\n", b"\n"]), None)), ["{}"])
        early = self.call_one(mode="early_length", timeout=0, max_tokens=None, native_max_model_len=4096)
        self.assertTrue(early["response_complete"])
        self.assertTrue(early["unexpected_length_before_budget"])
        self.assertFalse(early["generation_budget_consistent"])
        self.assertTrue(early["protocol_valid"])
        expected = {"case_id": "mock/0", "dataset": "mock", "metric": "qa_f1", "answers": ["alpha"],
                    "input_tokens": 12, "source_length": 500, "was_truncated": False}
        _, quality = scoring.score_records([expected], [early])
        self.assertEqual(quality["task_score_0_100"], 100)
        self.assertTrue(run.summarize([early], 1)["measurement_complete"])

    def test_partial_http_error_usage_missing_count_mismatch(self):
        for mode in ["partial", "http_error", "missing_usage", "mismatch"]:
            result = self.call_one(mode)
            self.assertFalse(result["protocol_valid"], mode)
            summary = run.summarize([result], 1)
            self.assertFalse(summary["measurement_complete"], mode)
            if mode == "partial":
                self.assertEqual(result["raw_content"], "alpha")
                self.assertFalse(result["done_marker_received"])
            elif mode == "http_error":
                self.assertEqual(result["http_status"], 503)
            elif mode == "missing_usage":
                self.assertIsNone(result["completion_tokens"])
            else:
                self.assertFalse(result["prompt_token_match"])

    def test_offline_prepare_run_score_roundtrip_and_dry_run(self):
        model = self.root / "model"
        model.mkdir()
        (model / "tokenizer_config.json").write_text('{"test_fixture":true}')
        source = self.root / "data.zip"
        with zipfile.ZipFile(source, "w") as z:
            z.writestr("data/qasper_e.jsonl", "\n".join(json.dumps(source_record(i)) for i in range(2)))
        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kw: FakeTokenizer())
        frozen = self.root / "frozen"
        with patch.dict("sys.modules", {"transformers": fake}), redirect_stdout(io.StringIO()):
            prepare.main(["--source", str(source), "--source-revision", "offline-synthetic-fixture",
                          "--model", str(model), "--out", str(frozen), "--datasets", "qasper",
                          "--variant", "longbench-e", "--max-model-len", "4096", "--no-thinking",
                          "--generation-budget-policy", "reference"])
        manifest, data = checked_data(frozen)
        self.assertEqual(manifest["variant"], "longbench-e")
        self.assertNotIn("greedy", manifest["protocol"])
        self.assertIn("sampling specified by run.json", manifest["protocol"])
        self.assertEqual(manifest["datasets"]["qasper"]["source_member"], "data/qasper_e.jsonl")
        launch_path = self.root / "launch.json"
        atomic_json(launch_path, {"schema": "kvbench.longcontext.server.v1", "native_verified": True,
            "config": {"method": "fp16", "kv_dtype": "auto", "block_size": 128, "model_name": "mock-longcontext",
                       "max_model_len": 4096, "max_seqs": 2, "generation_config": "vllm", "override_generation_config": {}}, "base_url": self.base,
            "tokenizer": tokenizer_identity(model)})
        output = self.root / "run"
        with redirect_stdout(io.StringIO()):
            run.main(["--data-dir", str(frozen), "--out", str(output), "--concurrency", "1", "2", "--dry-run"])
        self.assertFalse(output.exists())
        self.assertEqual(Handler.received, [])
        with patch.dict(os.environ, {"MOCK_LONGCONTEXT_KEY": "offline-test-key"}), redirect_stdout(io.StringIO()):
            run.main(["--data-dir", str(frozen), "--out", str(output), "--server-manifest", str(launch_path),
                      "--concurrency", "1", "2", "--warmup", "0", "--api-key-env", "MOCK_LONGCONTEXT_KEY",
                      "--temperature", "0", "--top-p", "1", "--top-k", "-1"])
            scoring.main(["--data-dir", str(frozen), "--run-dir", str(output)])
        stats = json.loads((output / "performance.summary.json").read_text())
        quality = json.loads((output / "quality.summary.json").read_text())
        self.assertEqual(len(stats), 2)
        self.assertTrue(all(s["measurement_complete"] for s in stats))
        self.assertEqual([s["unmeasured_long_warmup_requests"] for s in stats], [1, 2])
        self.assertEqual([s["recorded_tasks"] for s in stats], [2, 2])
        self.assertEqual(len(Handler.received), 7)  # 3 warmups excluded from 4 measured requests
        self.assertTrue(all(s["task_score_0_100"] == 100 for s in quality))
        for path in output.rglob("*.json*"):
            if path.suffix != ".gz":
                self.assertNotIn("offline-test-key", path.read_text())
        # A modified frozen file must be detected before any new request can run.
        (frozen / "qasper.jsonl").write_text("{}\n")
        with self.assertRaises(ValueError):
            checked_data(frozen)

    def test_thinking_sampling_reaches_measured_and_long_warmup_requests(self):
        model = self.root / "thinking_model"
        model.mkdir()
        (model / "tokenizer_config.json").write_text('{"test_fixture":true}')
        source = self.root / "thinking_data.zip"
        with zipfile.ZipFile(source, "w") as z:
            z.writestr("data/qasper.jsonl", "\n".join(json.dumps(source_record(i)) for i in range(2)))
        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kw: FakeTokenizer())
        frozen = self.root / "thinking_frozen"
        with patch.dict("sys.modules", {"transformers": fake}), redirect_stdout(io.StringIO()):
            prepare.main(["--source", str(source), "--source-revision", "offline-synthetic-fixture",
                          "--model", str(model), "--out", str(frozen), "--datasets", "qasper",
                          "--thinking", "--generation-budget-policy", "fixed", "--max-new-tokens", "1024", "--max-model-len", "4096"])
        manifest, data = checked_data(frozen)
        self.assertTrue(manifest["settings"]["thinking"])
        self.assertEqual(manifest["settings"]["max_new_tokens"], 1024)
        self.assertTrue(all(r["max_tokens"] == 1024 and r["input_tokens"] == len(r["prompt"]) + 6 for r in data["qasper"]))
        launch_path = self.root / "thinking_launch.json"
        atomic_json(launch_path, {"schema": "kvbench.longcontext.server.v1", "native_verified": True,
            "config": {"method": "fp16", "kv_dtype": "auto", "block_size": 128, "model_name": "mock-longcontext",
                       "max_model_len": 4096, "max_seqs": 2, "generation_config": "vllm", "override_generation_config": {}}, "base_url": self.base,
            "tokenizer": tokenizer_identity(model)})
        # min_p=0 matches V8; a nonzero value additionally detects accidental hardcoding.
        for number, min_p in enumerate((0.0, .05)):
            with self.subTest(min_p=min_p):
                output = self.root / f"thinking_run_{number}"
                flags = ["--data-dir", str(frozen), "--out", str(output), "--server-manifest", str(launch_path),
                         "--concurrency", "2", "--temperature", "0.6", "--top-p", "0.95",
                         "--top-k", "20", "--min-p", str(min_p)]
                expected = {"thinking": True, "temperature": .6, "top_p": .95, "top_k": 20, "min_p": min_p}
                count_before = len(Handler.received)
                preview = io.StringIO()
                with redirect_stdout(preview):
                    run.main(flags + ["--dry-run"])
                plan = json.loads(preview.getvalue())
                self.assertEqual({k: plan[k] for k in expected}, expected)
                self.assertFalse(output.exists())
                self.assertEqual(len(Handler.received), count_before)
                with redirect_stdout(io.StringIO()):
                    run.main(flags)
                    scoring.main(["--data-dir", str(frozen), "--run-dir", str(output)])
                saved = json.loads((output / "run.json").read_text())["settings"]
                self.assertEqual({k: saved[k] for k in expected}, expected)
                observed = [r["request"] for r in Handler.received[count_before:]]
                self.assertEqual(len(observed), 5)  # one health check, two long warmups, two measurements
                health = [r for r in observed if r["max_tokens"] == 16]
                self.assertEqual(len(health), 1)
                self.assertFalse(health[0]["chat_template_kwargs"]["enable_thinking"])
                active = [r for r in observed if r["max_tokens"] != 16]
                self.assertEqual(sorted(r["max_tokens"] for r in active), [32, 32, 1024, 1024])
                for request in active:
                    self.assertTrue(request["chat_template_kwargs"]["enable_thinking"])
                    self.assertEqual({k: request[k] for k in ["temperature", "top_p", "top_k", "min_p"]},
                                     {k: expected[k] for k in ["temperature", "top_p", "top_k", "min_p"]})
                point = output / "r01_qasper_C2"
                for filename in ["measure.requests.jsonl", "warmup/requests.sent.jsonl"]:
                    sent = [json.loads(x) for x in (point / filename).read_text().splitlines()]
                    self.assertTrue(all(x["top_k"] == 20 and x["min_p"] == min_p for x in sent))
                performance = json.loads((output / "performance.summary.json").read_text())[0]
                self.assertTrue(performance["measurement_complete"])
                self.assertEqual(performance["recorded_tasks"], 2)
                self.assertEqual(performance["unmeasured_long_warmup_requests"], 2)
                quality = json.loads((output / "quality.summary.json").read_text())[0]
                self.assertEqual(quality["task_score_0_100"], 100)
                whitespace = json.loads((point / "requests.jsonl").read_text().splitlines()[0])
                whitespace["final_answer"] = run.extract_final("reason</think>\n\n", "", True)[0]
                expected_row = next(r for r in data["qasper"] if r["case_id"] == whitespace["case_id"])
                score_rows, _ = scoring.score_records([expected_row], [whitespace])
                self.assertEqual((whitespace["final_answer"], run.summarize([whitespace], 1)["empty_final_answers"],
                                  score_rows[0]["empty_final_answer"]), ("\n\n", 1, True))

    def test_default_remaining_context_null_length_scoring_and_warmup_boundary(self):
        Handler.model_max_len = 32768
        Handler.mode = "length_budget"
        model = self.root / "natural_model"
        model.mkdir()
        (model / "tokenizer_config.json").write_text('{"test_fixture":true}')
        records = []
        for idx, target in enumerate((1024, 24000, 32767)):
            record = source_record(idx)
            record["context"] = ""
            empty_prompt = TASKS["qasper"]["prompt"].format(context="", input=record["input"])
            record["context"] = "x" * (target - len(empty_prompt) - 6)
            records.append(record)
        source = self.root / "natural_data.zip"
        with zipfile.ZipFile(source, "w") as z:
            z.writestr("data/qasper.jsonl", "\n".join(json.dumps(r) for r in records))
        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kw: FakeTokenizer())
        frozen = self.root / "natural_frozen"
        with patch.dict("sys.modules", {"transformers": fake}), redirect_stdout(io.StringIO()):
            # No --thinking, no output-cap flags: these are the actual v3 defaults.
            prepare.main(["--source", str(source), "--source-revision", "offline-synthetic-fixture",
                          "--model", str(model), "--out", str(frozen), "--datasets", "qasper"])
        manifest, data = checked_data(frozen)
        self.assertTrue(manifest["settings"]["thinking"])
        self.assertEqual(manifest["generation_budget_policy"], "remaining-context")
        self.assertEqual(sorted(r["remaining_context_tokens"] for r in data["qasper"]), [1, 8768, 31744])
        self.assertTrue(all(r["max_tokens"] is None and r["minimum_generation_reserve"] == 1 for r in data["qasper"]))
        self.assertTrue(all(r["generation_budget_tokens"] == r["remaining_context_tokens"] for r in data["qasper"]))
        too_long = {**records[-1], "context": records[-1]["context"] + "x"}
        with self.assertRaisesRegex(ValueError, "32768 tokens"):
            prepare.prepare_rows([too_long], "qasper", FakeTokenizer(), settings(
                max_model_len=32768, generation_budget_policy="remaining-context", thinking=True))
        launch_path = self.root / "natural_launch.json"
        atomic_json(launch_path, {"schema": "kvbench.longcontext.server.v1", "native_verified": True,
            "config": {"method": "fp16", "kv_dtype": "auto", "block_size": 128, "model_name": "mock-longcontext",
                       "max_model_len": 32768, "max_seqs": 16, "generation_config": "vllm", "override_generation_config": {}},
            "base_url": self.base, "tokenizer": tokenizer_identity(model)})
        output = self.root / "natural_run"
        flags = ["--data-dir", str(frozen), "--out", str(output), "--server-manifest", str(launch_path),
                 "--warmup-per-point", "3"]
        for bad in ("-1", "nan", "inf"):
            with self.subTest(timeout=bad), self.assertRaises(ValueError):
                run.main(flags + ["--request-timeout", bad, "--dry-run"])
        preview = io.StringIO()
        with redirect_stdout(preview):
            run.main(flags + ["--dry-run"])
        plan = json.loads(preview.getvalue())
        self.assertEqual((plan["thinking"], plan["temperature"], plan["top_p"], plan["top_k"], plan["min_p"], plan["request_timeout"]),
                         (True, .6, .95, 20, 0.0, 0))
        self.assertEqual(plan["max_tokens"], {"qasper": None})
        self.assertEqual(Handler.received, [])
        with redirect_stdout(io.StringIO()):
            run.main(flags)
            scoring.main(["--data-dir", str(frozen), "--run-dir", str(output)])
        point = output / "r01_qasper_C1"
        measured = [json.loads(line) for line in (point / "requests.jsonl").read_text().splitlines()]
        warmed = [json.loads(line) for line in (point / "warmup/responses.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(r["request"]["max_tokens"] for r in warmed), [1, 32, 32])
        self.assertTrue(all(r["protocol_valid"] for r in warmed))
        actual_formal = [r for r in Handler.received if r["request"]["max_tokens"] is None]
        self.assertEqual(len(actual_formal), 3)
        for request in actual_formal:
            self.assertIn('"max_tokens":null', request["raw_body"])
            self.assertNotIn("max_completion_tokens", request["request"])
            self.assertEqual((request["request"]["temperature"], request["request"]["top_p"], request["request"]["top_k"]), (.6, .95, 20))
        self.assertTrue(all(r["finish_reason"] == "length" and r["protocol_valid"] for r in measured))
        self.assertTrue(all(r["remaining_context_tokens_local"] == r["remaining_context_tokens_native"] == r["completion_tokens"] for r in measured))
        perf = json.loads((output / "performance.summary.json").read_text())[0]
        self.assertTrue(perf["measurement_complete"])
        self.assertEqual((perf["recorded_tasks"], perf["length_limit_terminations"], perf["empty_final_answers"]), (3, 3, 1))
        self.assertIsNone(perf["request_max_tokens"])
        quality = json.loads((output / "quality.summary.json").read_text())[0]
        self.assertTrue(quality["evaluation_complete"])
        self.assertAlmostEqual(quality["task_score_0_100"], 200 / 3)
        scored = [json.loads(line) for line in (point / "scores.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(r["score"] for r in scored), [0, 1, 1])
        _, incomplete = scoring.score_records(data["qasper"], measured[:2])
        self.assertEqual(incomplete["missing_tasks"], 1)
        self.assertIsNone(incomplete["task_score_0_100"])


if __name__ == "__main__":
    unittest.main()
