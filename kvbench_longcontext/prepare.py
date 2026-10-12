"""Freeze official LongBench v1 inputs with exact local tokenizer accounting."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

from .common import (CORE_TASKS, DATA_ZIP_SHA256, HF_REPO, HF_REVISION, LONGBENCH_COMMIT,
                     TASKS, atomic_json, canonical, describe, digest, package_versions,
                     sha_file, tokenizer_identity, write_jsonl)


def get_source(args):
    if args.source:
        source = Path(args.source).resolve()
        if not source.exists():
            raise ValueError("Source does not exist: " + str(source))
        revision = args.source_revision or "unknown-local-export"
        return source, {"source": "local", "path": str(source), "revision": revision}
    from huggingface_hub import hf_hub_download
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("--revision must be an immutable 40-character dataset commit")
    path = hf_hub_download(HF_REPO, "data.zip", repo_type="dataset", revision=args.revision,
                           cache_dir=args.cache_dir, local_files_only=args.offline)
    actual = sha_file(path)
    if args.revision == HF_REVISION and actual != DATA_ZIP_SHA256:
        raise ValueError("Official data.zip SHA256 mismatch; refusing changed bytes")
    return Path(path), {"source": "huggingface-data-zip", "dataset_id": HF_REPO,
                        "revision": args.revision, "filename": "data.zip", "sha256": actual}


def source_bytes(source, dataset):
    if source.is_dir():
        candidates = [source / (dataset + ".jsonl"), source / "data" / (dataset + ".jsonl")]
        found = [p for p in candidates if p.is_file()]
        if len(found) != 1:
            raise ValueError("Expected exactly one official JSONL for " + dataset)
        return found[0].read_bytes(), str(found[0])
    if not zipfile.is_zipfile(source):
        raise ValueError("--source must be an official data.zip or a directory of original JSONL files")
    with zipfile.ZipFile(source) as z:
        members = [n for n in z.namelist() if n in (dataset + ".jsonl", "data/" + dataset + ".jsonl")]
        if len(members) != 1:
            raise ValueError("Missing or ambiguous JSONL member for " + dataset)
        # Read named entries only; never extract archive paths or execute dataset code.
        return z.read(members[0]), members[0]


def count_input(tokenizer, prompt, kind, thinking):
    if kind == "completion":
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    return len(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=thinking))


def fit_prompt(tokenizer, prompt, kind, thinking, limit, policy):
    original = count_input(tokenizer, prompt, kind, thinking)
    if original <= limit:
        return prompt, original, original, False
    if policy == "error":
        raise ValueError(f"Prompt has {original} tokens, budget is {limit}. Select --overflow skip or explicit head-tail.")
    if policy == "skip":
        return None, original, None, False
    # Explicit LongBench-style middle removal on the formatted prompt, then re-count
    # the actual Qwen chat wrapper. The full prompt remains in raw source records.
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    keep = min(len(ids) - 1, limit)
    while keep > 0:
        left = (keep + 1) // 2
        right = keep // 2
        candidate = (tokenizer.decode(ids[:left], skip_special_tokens=True)
                     + (tokenizer.decode(ids[-right:], skip_special_tokens=True) if right else ""))
        actual = count_input(tokenizer, candidate, kind, thinking)
        if actual <= limit:
            return candidate, original, actual, True
        keep -= max(1, actual - limit)
    raise ValueError("Even the template cannot fit the requested input budget")


def prepare_rows(records, name, tokenizer, settings):
    task = TASKS[name]
    policy = settings["generation_budget_policy"]
    explicit_cap = settings["max_new_tokens"]
    if policy == "remaining-context":
        if explicit_cap is not None:
            raise ValueError("remaining-context sends max_tokens:null; use fixed for --max-new-tokens")
        cap, reserve = None, 1
    elif policy == "reference":
        if explicit_cap is not None or settings["thinking"]:
            raise ValueError("reference uses official short caps: select --no-thinking and omit --max-new-tokens")
        cap = reserve = task["max_new_tokens"]
    elif policy == "fixed":
        if not isinstance(explicit_cap, int) or isinstance(explicit_cap, bool) or explicit_cap < 1:
            raise ValueError("fixed requires a positive --max-new-tokens")
        cap = reserve = explicit_cap
    else:
        raise ValueError("Unknown generation budget policy: " + str(policy))
    limit = min(settings["max_input_tokens"] or settings["max_model_len"], settings["max_model_len"] - reserve)
    if limit <= 0 or settings["min_input_tokens"] > limit:
        raise ValueError("Input budget cannot fit generation reserve or minimum length")
    kept, excluded, seen = [], [], set()
    for i, record in enumerate(records):
        if not isinstance(record.get("context"), str) or not isinstance(record.get("input"), str):
            raise ValueError(f"{name} row {i}: official context/input strings required")
        answers = record.get("answers")
        if not isinstance(answers, list) or not answers or not all(isinstance(x, str) for x in answers):
            raise ValueError(f"{name} row {i}: nonempty original answers list required")
        length = record.get("length")
        if (not isinstance(length, (int, float)) or isinstance(length, bool)
                or not math.isfinite(length) or length < 0):
            raise ValueError(f"{name} row {i}: original finite nonnegative LongBench length required; never replace it with model tokens")
        raw_id = str(record.get("_id", record.get("id", i)))
        cid = name + "/" + raw_id
        if cid in seen:
            raise ValueError("Duplicate source case ID: " + cid)
        seen.add(cid)
        # Only context/input interpolate into the official template; references never do.
        prompt = task["prompt"].format(context=record["context"], input=record["input"])
        try:
            prompt, original, effective, truncated = fit_prompt(tokenizer, prompt, task["request_kind"],
                settings["thinking"], limit, settings["overflow"])
        except ValueError as exc:
            raise ValueError(cid + ": " + str(exc)) from exc
        if prompt is None or effective < settings["min_input_tokens"]:
            excluded.append({"case_id": cid, "source_index": i, "original_input_tokens": original,
                             "reason": "over_budget" if prompt is None else "below_minimum"})
            continue
        row = {"dataset": name, "case_id": cid, "source_index": i, "source_record_sha256": digest(record),
               "prompt": prompt, "request_kind": task["request_kind"], "answers": answers,
               "metric": task["metric"], "all_classes": record.get("all_classes", []),
               "source_length": record.get("length"), "source_length_unit": "LongBench English words (not model tokens)",
               "original_input_tokens": original, "input_tokens": effective, "input_token_budget": limit,
               "was_truncated": truncated, "truncation_policy": settings["overflow"], "max_tokens": cap,
               "generation_budget_policy": policy, "minimum_generation_reserve": reserve,
               "max_model_len": settings["max_model_len"],
               "remaining_context_tokens": settings["max_model_len"] - effective,
               "generation_budget_tokens": settings["max_model_len"] - effective if cap is None else cap,
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        kept.append(row)
    kept.sort(key=lambda r: digest([settings["data_seed"], r["case_id"]]))
    eligible_count = len(kept)
    size = settings["sample_size"]
    if size and len(kept) < size:
        raise ValueError(f"{name}: only {len(kept)} eligible records for requested sample {size}; choose a smaller sample explicitly")
    if not kept:
        raise ValueError(name + ": no eligible records")
    return kept[:size] if size else kept, excluded, eligible_count


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="Existing local model/tokenizer snapshot")
    p.add_argument("--datasets", nargs="+", choices=list(TASKS), default=CORE_TASKS)
    p.add_argument("--variant", choices=["longbench", "longbench-e"], default="longbench")
    p.add_argument("--source", type=Path, help="Official local data.zip or original JSONL directory")
    p.add_argument("--source-revision", help="Immutable revision of a local source when known")
    p.add_argument("--revision", default=HF_REVISION)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-input-tokens", type=int, default=None)
    p.add_argument("--min-input-tokens", type=int, default=0)
    p.add_argument("--generation-budget-policy", choices=["remaining-context", "reference", "fixed"], default="remaining-context",
                   help="Default: send max_tokens:null and let the server use the remaining context; reference uses official task caps")
    p.add_argument("--max-new-tokens", type=int, help="Positive explicit output cap; only for --generation-budget-policy fixed")
    p.add_argument("--sample-size", type=int, default=0, help="0 = every eligible case; otherwise deterministic sample")
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--overflow", choices=["error", "skip", "head-tail"], default="error")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args(argv)
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError("Output must be a new/empty directory; frozen data are not overwritten")
    if (a.sample_size < 0 or a.min_input_tokens < 0 or a.max_model_len < 1
            or a.max_input_tokens is not None and a.max_input_tokens < 1
            or a.max_new_tokens is not None and a.max_new_tokens < 1):
        raise ValueError("Invalid size or token budget")
    if a.generation_budget_policy == "fixed" and a.max_new_tokens is None:
        raise ValueError("fixed requires an explicit positive --max-new-tokens")
    if a.generation_budget_policy != "fixed" and a.max_new_tokens is not None:
        raise ValueError("--max-new-tokens is only used with --generation-budget-policy fixed")
    if a.generation_budget_policy == "reference" and a.thinking:
        raise ValueError("reference uses official short output caps; also select --no-thinking")
    if a.thinking and "lcc" in a.datasets:
        raise ValueError("LCC uses raw completion without chat thinking controls; prepare it separately with --no-thinking")
    if a.variant == "longbench-e" and "qmsum" in a.datasets:
        raise ValueError("Official LongBench-E has no qmsum split; prepare it separately as LongBench v1")
    from transformers import AutoTokenizer
    identity = tokenizer_identity(a.model)
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Model tokenizer has no chat template")
    source, provenance = get_source(a)
    if source.is_file():
        provenance["file_sha256"] = sha_file(source)
        if provenance.get("revision") == HF_REVISION and provenance["file_sha256"] != DATA_ZIP_SHA256:
            raise ValueError("Local archive does not match the SHA256 of the claimed official dataset revision")
    settings = {k: getattr(a, k) for k in ["max_model_len", "max_input_tokens", "min_input_tokens",
                "max_new_tokens", "generation_budget_policy", "sample_size", "data_seed", "overflow", "thinking"]}
    prepared = {}
    for name in dict.fromkeys(a.datasets):
        raw, member = source_bytes(source, name + ("_e" if a.variant == "longbench-e" else ""))
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        rows, excluded, eligible = prepare_rows(records, name, tokenizer, settings)
        prepared[name] = (raw, member, rows, excluded, len(records), eligible)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "kvbench.longcontext.data.v1", "benchmark": "LongBench-v1", "variant": a.variant,
        "generation_budget_policy": a.generation_budget_policy,
        "max_tokens_transport": "explicit-json-null" if a.generation_budget_policy == "remaining-context" else "explicit-positive-integer",
        "upstream_code_commit": LONGBENCH_COMMIT, "source": provenance, "settings": settings,
        "tokenizer": identity, "tokenizer_packages": package_versions(["transformers", "tokenizers", "huggingface_hub"]),
        "protocol": ("Qwen non-thinking inputs; sampling specified by run.json" if not a.thinking
                     else "experimental Qwen thinking inputs; sampling specified by run.json"),
        "source_config_sha256": sha_file(Path(__file__).parent / "config/tasks.json"), "datasets": {}}
    for name, (raw, member, rows, excluded, total, eligible) in prepared.items():
        rawpath = a.out / "raw" / (name + ".jsonl")
        rawpath.parent.mkdir(parents=True, exist_ok=True)
        rawpath.write_bytes(raw)
        path = a.out / (name + ".jsonl")
        write_jsonl(path, rows)
        write_jsonl(a.out / (name + ".excluded.jsonl"), excluded)
        manifest["datasets"][name] = {"file": path.name, "sha256": sha_file(path), "source_member": member,
            "raw_file": str(rawpath.relative_to(a.out)), "raw_sha256": sha_file(rawpath), "source_count": total,
            "eligible_count": eligible, "excluded_count": len(excluded), "selected_count": len(rows),
            "selected_case_ids": [r["case_id"] for r in rows], "truncated_count": sum(r["was_truncated"] for r in rows),
            "input_tokens": describe([r["input_tokens"] for r in rows]), "max_tokens": rows[0]["max_tokens"],
            "generation_budget_policy": a.generation_budget_policy,
            "remaining_context_tokens": describe([r["remaining_context_tokens"] for r in rows]),
            "generation_budget_tokens": describe([r["generation_budget_tokens"] for r in rows]),
            "cohort": "entire-source" if len(rows) == total and not any(r["was_truncated"] for r in rows) else "selected-or-truncated-cohort"}
    atomic_json(a.out / "manifest.json", manifest)
    print(json.dumps({"out": str(a.out), "datasets": {k: {x: v[x] for x in ["source_count", "selected_count", "excluded_count", "truncated_count", "input_tokens"]}
                                                  for k, v in manifest["datasets"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
