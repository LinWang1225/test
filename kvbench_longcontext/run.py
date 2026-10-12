"""Bounded, auditable local OpenAI-compatible SSE client for frozen long inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import (atomic_json, canonical, checked_data, describe, digest, package_versions,
                     sha_file, write_jsonl)
from .serve import http_text, run_text, verify_cache, verify_model


def validate_sampling(temperature, top_p, top_k, min_p):
    if (not math.isfinite(temperature) or temperature < 0
            or not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError("temperature must be finite and nonnegative; top_p must be in (0, 1]")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k != -1 and top_k < 1:
        raise ValueError("top_k must be -1 (disabled) or a positive integer")
    if not math.isfinite(min_p) or not 0 <= min_p <= 1:
        raise ValueError("min_p must be finite and in [0, 1]")


def make_request(row, model, thinking=True, temperature=0.6, top_p=0.95, seed=42, top_k=20, min_p=0.0):
    validate_sampling(temperature, top_p, top_k, min_p)
    request = {"model": model, "stream": True, "stream_options": {"include_usage": True},
               "n": 1, "max_tokens": row["max_tokens"], "temperature": temperature,
               "top_p": top_p, "top_k": top_k, "min_p": min_p, "seed": seed,
               "repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}
    if row["request_kind"] == "chat":
        request["messages"] = [{"role": "user", "content": row["prompt"]}]
        request["chat_template_kwargs"] = {"enable_thinking": thinking}
    else:
        request["prompt"] = row["prompt"]
        request["add_special_tokens"] = False
    return request


def sse_events(response, deadline):
    """Check elapsed time between SSE lines; socket timeout applies while blocked."""
    lines = []
    for raw in response:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("Elapsed-time limit exceeded between SSE lines")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
        # SSE comments, event/id/retry fields are not model content.
    if lines:
        yield "\n".join(lines)


def extract_final(content, reasoning, thinking, kind="chat", reasoning_channel=False):
    if reasoning_channel:
        return content, reasoning, "server-reasoning-channel"
    if not thinking:
        # Code-completion whitespace and literal <think> tags can be answer data.
        # Preserve them. Chat may have a leading EMPTY Qwen thinking wrapper only.
        if kind == "chat":
            match = re.match(r"^\s*<think>\s*</think>\s*", content)
            if match:
                return content[match.end():], reasoning, "leading-empty-think-wrapper"
        return content, reasoning, "non-thinking-raw-content"
    if "</think>" in content:
        before, final = content.split("</think>", 1)
        return final, reasoning or before.split("<think>", 1)[-1], "first-closed-think-delimiter"
    return "", reasoning or content, "unfinished-thinking-or-no-final-channel"


def stream_request(base_url, row, request, destination, timeout, headers=None, thinking=False, native_max_model_len=None):
    started = time.monotonic()
    unix_started = time.time()
    chunks, content, reasoning, error = [], [], [], None
    usage, finish, first, done, http_success = None, None, None, False, False
    reasoning_channel = False
    endpoint = "/v1/chat/completions" if row["request_kind"] == "chat" else "/v1/completions"
    reqheaders = {"Content-Type": "application/json", "Accept": "text/event-stream", **(headers or {})}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base_url + endpoint, data=canonical(request).encode("utf-8"), headers=reqheaders, method="POST")
    status = None
    try:
        with opener.open(req, timeout=timeout if timeout > 0 else None) as response:
            status = response.status
            http_success = 200 <= status < 300
            if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("Expected text/event-stream response")
            for data in sse_events(response, started + timeout if timeout > 0 else None):
                received = time.monotonic()
                if data == "[DONE]":
                    done = True
                    break
                chunks.append({"elapsed_s": received - started, "raw_data": data})
                chunk = json.loads(data)
                chunks[-1]["data"] = chunk
                if chunk.get("error") is not None:
                    raise ValueError("Server stream error: " + str(chunk["error"]))
                # Usage-only final chunks have choices=[], and must still be retained.
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    if choice.get("index", 0) != 0:
                        raise ValueError("Only n=1 / choice index 0 is supported")
                    delta = choice.get("delta", {})
                    reasoning_channel |= isinstance(delta.get("reasoning_content"), str) or isinstance(delta.get("reasoning"), str)
                    piece = delta.get("content", "") if row["request_kind"] == "chat" else choice.get("text", "")
                    rp = delta.get("reasoning_content")
                    if rp is None:
                        rp = delta.get("reasoning", "")
                    if piece is not None and not isinstance(piece, str) or rp is not None and not isinstance(rp, str):
                        raise ValueError("Unknown response content type")
                    if piece or rp:
                        if first is None:
                            first = received
                        content.append(piece or "")
                        reasoning.append(rp or "")
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        status = exc.code
        http_success = False
        error = f"HTTP {exc.code}: " + exc.read(4096).decode("utf-8", errors="replace")
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    ended = time.monotonic()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    raw, raw_reasoning = "".join(content), "".join(reasoning)
    final, full_reasoning, extraction = extract_final(raw, raw_reasoning, thinking, row["request_kind"], reasoning_channel)
    native = bool(usage and all(isinstance(usage.get(k), int) and not isinstance(usage[k], bool) and usage[k] >= 0
                              for k in ["prompt_tokens", "completion_tokens"]))
    input_count = usage["prompt_tokens"] if native else None
    output_count = usage["completion_tokens"] if native else None
    remaining_native = native_max_model_len - input_count if native and native_max_model_len is not None else None
    native_budget = (remaining_native if request["max_tokens"] is None else min(remaining_native, request["max_tokens"])) if remaining_native is not None else None
    prompt_match = input_count == row.get("input_tokens") if native and row.get("input_tokens") is not None else None
    complete = http_success and done and error is None and finish in ("stop", "length")
    early_length = bool(complete and request["max_tokens"] is None and native_budget is not None
                        and finish == "length" and output_count < native_budget)
    budget_match = None if native_budget is None else native_budget > 0 and output_count <= native_budget
    tpot = (ended - first) / (output_count - 1) if first is not None and native and output_count > 1 else None
    return {"dataset": row["dataset"], "case_id": row["case_id"], "request": request,
        "http_status": status, "http_success": http_success, "response_complete": complete,
        "protocol_valid": complete and native and prompt_match is not False and budget_match is not False,
        "error": error, "done_marker_received": done, "finish_reason": finish, "length_capped": finish == "length",
        "raw_content": raw, "reasoning_content": full_reasoning, "final_answer": final,
        "answer_extraction": extraction, "usage": usage, "native_usage_complete": native,
        "input_tokens_expected": row.get("input_tokens"), "prompt_tokens": input_count,
        "prompt_token_match": prompt_match, "completion_tokens": output_count,
        "generation_budget_policy": row.get("generation_budget_policy"), "request_max_tokens": request["max_tokens"],
        "remaining_context_tokens_local": row.get("remaining_context_tokens"),
        "remaining_context_tokens_native": remaining_native, "generation_budget_tokens_native": native_budget,
        "native_remaining_source": "computed_from_native_prompt_tokens_and_verified_model_window",
        "native_max_model_len": native_max_model_len, "generation_budget_valid": budget_match,
        "unexpected_length_before_budget": early_length,
        "generation_budget_consistent": None if budget_match is None else budget_match and not early_length,
        "token_count_source": "server_usage" if native else "missing-no-fallback",
        "started_monotonic": started, "ended_monotonic": ended, "started_unix_time": unix_started,
        "e2e_s": ended - started, "ttft_s": first - started if first else None, "tpot_s": tpot,
        "raw_chunks_file": ("raw/" if destination.parent.name == "raw" else "") + destination.name,
        "raw_chunks_sha256": sha_file(destination)}


def metric_values(text, suffix):
    values = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r"([^ {]+)(?:\{.*\})?\s+([^ ]+)", line)
        if match and match[1].endswith(suffix):
            try:
                values.append(float(match[2]))
            except ValueError:
                pass
    return values


class Observer:
    def __init__(self, base_url, path, gpu, interval, headers):
        self.base, self.path, self.gpu, self.interval, self.headers = base_url, Path(path), gpu, interval, headers
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        with self.path.open("w", encoding="utf-8") as target:
            while not self.stop_event.is_set():
                record = {"unix_time": time.time()}
                try:
                    text = http_text(self.base + "/metrics", timeout=3, headers=self.headers)
                    record["kv_usage_fraction"] = metric_values(text, "kv_cache_usage_perc")
                    record["preemptions_total"] = metric_values(text, "num_preemptions_total")
                    record["num_requests_running"] = metric_values(text, "num_requests_running")
                except Exception as exc:
                    record["metrics_error"] = str(exc)
                if self.gpu is not None:
                    record["gpu"] = run_text(["nvidia-smi", "-i", self.gpu,
                        "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                        "--format=csv,noheader,nounits"], timeout=4)
                target.write(canonical(record) + "\n")
                target.flush()
                self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("Observer did not stop; refusing overlapping measurement")


def summarize(records, expected):
    good = [r for r in records if r["response_complete"] and r["protocol_valid"]]
    window = max(r["ended_monotonic"] for r in records) - min(r["started_monotonic"] for r in records) if records else None
    usage_complete = bool(good) and all(r["native_usage_complete"] for r in good)
    return {"expected_tasks": expected, "recorded_tasks": len(records), "valid_responses": len(good),
        "measurement_complete": len(records) == expected and len(good) == expected,
        "http_failed": sum(not r["http_success"] for r in records),
        "incomplete_streams": sum(not r["response_complete"] for r in records),
        "missing_native_usage": sum(not r["native_usage_complete"] for r in records),
        "prompt_token_mismatches": sum(r["prompt_token_match"] is False for r in records),
        "generation_budget_violations": sum(r.get("generation_budget_valid") is False for r in records),
        "unexpected_length_before_budget_count": sum(r.get("unexpected_length_before_budget", False) for r in records),
        "generation_budget_consistent": (all(r["generation_budget_consistent"] for r in records)
            if records and all(r.get("generation_budget_consistent") is not None for r in records) else None),
        "length_capped": sum(r["length_capped"] for r in good),
        "length_limit_terminations": sum(r["length_capped"] for r in good),
        "empty_final_answers": sum(not r["final_answer"].strip() for r in good),
        "request_window_s": window, "valid_requests_per_s": len(good) / window if window else None,
        "output_tokens_per_s": sum(r["completion_tokens"] for r in good) / window if window and usage_complete else None,
        "prompt_tokens": describe([r["prompt_tokens"] for r in good]),
        "completion_tokens": describe([r["completion_tokens"] for r in good]),
        "ttft_s": describe([r["ttft_s"] for r in good]), "tpot_s": describe([r["tpot_s"] for r in good]),
        "e2e_s": describe([r["e2e_s"] for r in good]),
        "timing_definition": "TTFT=first nonempty content/reasoning chunk; TPOT=(E2E-TTFT)/(native completion_tokens-1); request-equal mean",
        "timing_scope": "HTTP client service latency; includes scheduling/network, not a pure GPU kernel timer",
        "quality": "Score separately with kvbench_longcontext.scoring; caps remain included"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--server-manifest", type=Path)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20, help="-1 disables top-k; otherwise a positive integer")
    p.add_argument("--min-p", type=float, default=0.0, help="Minimum relative token probability in [0, 1]")
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--request-timeout", type=float, default=0,
                   help="0 disables request socket/elapsed limits; positive seconds enable socket timeout and checks between SSE lines")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--warmup-per-point", type=int, default=-1,
                   help="Unmeasured real long prompts per point; -1 uses that point's concurrency, 0 disables; each output capped at 32")
    p.add_argument("--gpu", help="Optional physical GPU index for nvidia-smi telemetry")
    p.add_argument("--sample-interval", type=float, default=2.0)
    p.add_argument("--api-key-env", help="Optional environment variable containing API key; key is never written to artifacts")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    manifest, datasets = checked_data(args.data_dir)
    if (min(args.concurrency + [args.repeats]) < 1 or not math.isfinite(args.request_timeout) or args.request_timeout < 0 or args.sample_interval <= 0
            or args.warmup < 0 or args.warmup_per_point < -1):
        raise ValueError("Invalid count or timeout")
    validate_sampling(args.temperature, args.top_p, args.top_k, args.min_p)
    points = [{"id": f"r{rep:02d}_{name}_C{c}", "dataset": name, "repeat": rep, "concurrency": c,
               "expected_tasks": len(rows)} for rep in range(1, args.repeats + 1)
              for name, rows in datasets.items() for c in args.concurrency]
    if any(p["concurrency"] > p["expected_tasks"] for p in points):
        raise ValueError("Concurrency exceeds selected cases; prepare a larger cohort")
    if any(args.warmup_per_point > p["expected_tasks"] for p in points):
        raise ValueError("Requested per-point warmup exceeds selected cases")
    if len({p["id"] for p in points}) != len(points):
        raise ValueError("Duplicate concurrency levels")
    if args.dry_run:
        print(json.dumps({"status": "plan-only; no HTTP or GPU accessed", "points": points,
                         "data_manifest_sha256": sha_file(args.data_dir / "manifest.json"),
                         "thinking": manifest["settings"]["thinking"], "temperature": args.temperature,
                         "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
                         "generation_budget_policy": manifest["generation_budget_policy"],
                         "max_tokens_transport": manifest["max_tokens_transport"],
                         "request_timeout": args.request_timeout,
                         "warmup_per_point": args.warmup_per_point,
                         "max_tokens": {k: v["max_tokens"] for k, v in manifest["datasets"].items()}}, indent=2))
        return
    if args.server_manifest is None:
        raise ValueError("--server-manifest from the current serve launch is required")
    launch = json.loads(args.server_manifest.read_text(encoding="utf-8"))
    if launch.get("schema") != "kvbench.longcontext.server.v1" or not launch.get("native_verified"):
        raise ValueError("Launcher has not verified its native cache configuration")
    config, base = launch["config"], launch["base_url"].rstrip("/")
    url = urllib.parse.urlsplit(base)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or url.path:
        raise ValueError("This runner supports only an explicitly launched local HTTP server")
    if launch["tokenizer"]["fingerprint"] != manifest["tokenizer"]["fingerprint"]:
        raise ValueError("Server model/tokenizer fingerprint differs from frozen preparation")
    if config["max_model_len"] != manifest["settings"]["max_model_len"] or max(args.concurrency) > config["max_seqs"]:
        raise ValueError("Server context window or max-seqs does not match planned data/concurrency")
    headers = {}
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            raise ValueError("API key environment variable is empty")
        headers["Authorization"] = "Bearer " + key
    models = json.loads(http_text(base + "/v1/models", headers=headers))
    live_model = verify_model(models, config)
    if config.get("generation_config") != "vllm" or config.get("override_generation_config") != {}:
        raise ValueError("This protocol requires --generation-config vllm without generation overrides")
    before = http_text(base + "/metrics", headers=headers)
    actual = verify_cache(before, config)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Run output directory must be new/empty; no automatic resume or overwrite")
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "server.launch.json", launch)
    run = {"schema": "kvbench.longcontext.run.v1", "data_manifest_sha256": sha_file(args.data_dir / "manifest.json"),
           "data_dir": str(args.data_dir.resolve()), "server_manifest_sha256": sha_file(args.server_manifest),
           "method": config["method"], "native_cache_config": actual, "points": points,
           "native_max_model_len": live_model["max_model_len"],
           "native_kv_capacity_tokens": int(actual["num_gpu_blocks"]) * int(actual["block_size"]),
           "settings": {"thinking": manifest["settings"]["thinking"], "temperature": args.temperature,
                        "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
                        "generation_budget_policy": manifest["generation_budget_policy"],
                        "max_tokens_transport": manifest["max_tokens_transport"],
                        "request_seed_base": args.seed,
                        "request_timeout": args.request_timeout, "warmup": args.warmup,
                        "warmup_per_point": args.warmup_per_point, "warmup_output_cap": 32, "retries": 0},
           "client_packages": package_versions(["transformers", "tokenizers"]),
           "scripts_sha256": {x.name: sha_file(x) for x in Path(__file__).parent.glob("*.py")}}
    atomic_json(args.out / "run.json", run)
    # Short, separate warmup. It does not contain benchmark context or answers.
    for i in range(args.warmup):
        row = {"dataset": "warmup", "case_id": "warmup/" + str(i), "request_kind": "chat",
               "prompt": "Reply with the word ready.", "max_tokens": 16, "input_tokens": None,
               "generation_budget_policy": "unmeasured-health-warmup-fixed"}
        req = make_request(row, config["model_name"], thinking=False, temperature=0.0, top_p=1.0,
                           top_k=-1, min_p=0.0, seed=args.seed + i)
        result = stream_request(base, row, req, args.out / f"warmup.{i}.chunks.json.gz", args.request_timeout, headers,
                                native_max_model_len=live_model["max_model_len"])
        atomic_json(args.out / f"warmup.{i}.json", result)
        if not result["protocol_valid"]:
            raise RuntimeError("Warmup stream/native-usage protocol failed; measurement was not started")
    summaries = []
    for point in points:
        dest = args.out / point["id"]
        dest.mkdir()
        rows = datasets[point["dataset"]]
        requests = [make_request(r, config["model_name"], manifest["settings"]["thinking"], args.temperature,
                                args.top_p, int(digest([args.seed, point["repeat"], r["case_id"]])[:8], 16) % (2**31 - 1),
                                top_k=args.top_k, min_p=args.min_p) for r in rows]
        write_jsonl(dest / "measure.requests.jsonl", requests)
        atomic_json(dest / "point.json", point)
        warm_count = point["concurrency"] if args.warmup_per_point == -1 else args.warmup_per_point
        if warm_count:
            warm_dir = dest / "warmup"
            warm_dir.mkdir()
            warm_rows = [{**r, "max_tokens": min(32, r["remaining_context_tokens"],
                          r["max_tokens"] if r["max_tokens"] is not None else r["remaining_context_tokens"]),
                          "generation_budget_policy": "unmeasured-point-warmup-fixed"} for r in rows[:warm_count]]
            warm_requests = [make_request(r, config["model_name"], manifest["settings"]["thinking"],
                args.temperature, args.top_p, int(digest(["warmup", args.seed, point["repeat"], r["case_id"]])[:8], 16) % (2**31 - 1),
                top_k=args.top_k, min_p=args.min_p)
                for r in warm_rows]
            write_jsonl(warm_dir / "requests.sent.jsonl", warm_requests)
            with ThreadPoolExecutor(max_workers=min(point["concurrency"], warm_count)) as pool:
                futures = [pool.submit(stream_request, base, row, request, warm_dir / "raw" / f"{i:05d}.json.gz",
                    args.request_timeout, headers, manifest["settings"]["thinking"], live_model["max_model_len"])
                    for i, (row, request) in enumerate(zip(warm_rows, warm_requests))]
                warm_records = [f.result() for f in futures]
            write_jsonl(warm_dir / "responses.jsonl", warm_records)
            if not all(r["protocol_valid"] for r in warm_records):
                raise RuntimeError("Long-input per-point warmup failed; measurement for this point was not started")
        # APC is verified off. Real long-prompt warmup covers this concurrency's
        # prefill/decode paths, with a short cap; only subsequent work is measured.
        before = http_text(base + "/metrics", headers=headers)
        (dest / "metrics.before.txt").write_text(before, encoding="utf-8")
        verify_cache(before, config)
        observer = Observer(base, dest / "telemetry.jsonl", args.gpu, args.sample_interval, headers)
        observer.start()
        records = []
        try:
            with (dest / "requests.jsonl").open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=point["concurrency"]) as pool:
                futures = [pool.submit(stream_request, base, row, request, dest / "raw" / f"{i:05d}.json.gz",
                                       args.request_timeout, headers, manifest["settings"]["thinking"], live_model["max_model_len"])
                           for i, (row, request) in enumerate(zip(rows, requests))]
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    out.write(canonical(record) + "\n")
                    out.flush()
        finally:
            observer.stop()
        after = http_text(base + "/metrics", headers=headers)
        (dest / "metrics.after.txt").write_text(after, encoding="utf-8")
        summary = summarize(records, len(rows))
        a, b = metric_values(before, "num_preemptions_total"), metric_values(after, "num_preemptions_total")
        summary.update(point, method=config["method"],
                       generation_budget_policy=manifest["generation_budget_policy"],
                       request_max_tokens=manifest["datasets"][point["dataset"]]["max_tokens"],
                       remaining_context_tokens_local=manifest["datasets"][point["dataset"]]["remaining_context_tokens"],
                       remaining_context_tokens_native=describe([r["remaining_context_tokens_native"] for r in records]),
                       native_max_model_len=live_model["max_model_len"],
                       native_kv_capacity_tokens=int(actual["num_gpu_blocks"]) * int(actual["block_size"]),
                       request_timeout=args.request_timeout,
                       unmeasured_long_warmup_requests=warm_count, unmeasured_long_warmup_output_cap=32,
                       warmup_note="same frozen long prompts and point concurrency; APC off; a 32-token warmup is not proof of full steady state",
                       preemptions_delta=sum(b) - sum(a) if len(a) == len(b) and a and sum(b) >= sum(a) else None)
        atomic_json(dest / "performance.summary.json", summary)
        summaries.append(summary)
        atomic_json(args.out / "performance.summary.json", summaries)
        print(json.dumps({"point": point["id"], "measurement_complete": summary["measurement_complete"],
                          "request_window_s": summary["request_window_s"], "length_capped": summary["length_capped"]}), flush=True)
        if not summary["measurement_complete"] and not args.continue_on_error:
            raise RuntimeError("Incomplete/protocol-invalid point; raw outputs preserved. Score saved outputs and inspect diagnostics before rerunning")


if __name__ == "__main__":
    main()
