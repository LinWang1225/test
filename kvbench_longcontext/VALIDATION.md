# Validation record — v3, 2026-10-09

## Executed here

```bash
python -m unittest discover -s kvbench_longcontext/tests -v
```

Final suite: **18 tests discovered, 16 passed, 2 explicitly skipped**. The two
skips require real `rouge==1.0.1` and LCC `fuzzywuzzy==0.18.0` /
`Levenshtein==0.25.1`, which are not installed in this delivery environment.
Installing requirements-client.txt enables the ROUGE test; installing the
additional requirements-lcc.txt enables the optional LCC test. A skipped backend
test is not evidence that its real backend has been run.

The suite covers:

- Frozen official task prompts, multi-reference answers, no answer injection,
  deterministic sampling, archive member selection and source/data hashes.
- Synthetic tokenizer accounting after the selected chat wrapper, explicit
  skip/error/head-tail handling, and separate source-word / local-token bins.
- QA F1, retrieval, reference-max scoring, code-line extraction, missing/duplicate/
  unexpected response handling, and the distinction between real zero and NA.
- Cache dtype, block size, APC, FP8 scales, positive native num_gpu_blocks and
  verified model window. K4/V4/g64 has its distinct block64 mapping.
- Local HTTP mock chat and raw-completion requests, actual JSON bodies, headers,
  usage-only SSE chunks, reasoning/content extraction, HTTP errors, interrupted
  streams, absent native usage and local/native prompt-token mismatches.
- A synthetic end-to-end prepare → frozen data → dry-run → concurrent mock HTTP
  → response records → offline scoring sequence. Dry-run sends no HTTP and creates
  no run directory; measured rows exclude all independent warmups.
- Default thinking=true and sampling .6/.95/20/0 reaching both measured and long
  warmup requests; explicit raw and chat max_tokens:null without
  max_completion_tokens; optional reference/fixed mode guards.
- Inputs of 1024/24000/32767 tokens retaining 31744/8768/1 generated-token capacity
  under a verified 32768 window, warmup caps32/32/1 and formal requests null.
  A 32768-token input without any generation space is rejected.
- request-timeout=0 passing timeout=None to the HTTP connection and disabling the
  elapsed stream guard; positive-timeout failures remaining incomplete responses.
- Complete normal/early length responses retained for scoring and performance;
  unfinished thinking with no final scoring zero, and early length producing only
  a separate budget-consistency diagnostic.
- Reporting every planned point, including missing points; preserving real zero,
  converting milliseconds correctly, rejecting mismatched summaries, and carrying
  unexpected_length_before_budget_count through to CSV.

An independent review also exercised timeout=0 under a deliberately advanced mock
clock and checked that an early-length response can receive full quality credit,
whereas an interrupted version has a missing primary score. These checks are not
additional GPU tests and are not added to the 18-test count above.

## Source audit

Read-only inspection of the fixed KVarN/vLLM source confirms explicit-null chat
and completion semantics, the completion endpoint omitted-field default16,
the remaining-context budget merge, the effect of generation_config=vllm and
empty override, and the CUDA platform lack of a smaller output bound.
The source-linked explanation is BUDGET_SEMANTICS.zh-CN.md.

Official LongBench task prompts and metric definitions were compared with the
pinned upstream version. Prior uploaded native metrics and request logs were
used to verify cache labels and the old short-input null request protocol.

## Delivery verification

The complete v3 patch was applied in an isolated baseline checkout, checked for
reverse application, and all applied package files were compared byte-for-byte.
Only new kvbench_longcontext/ paths are added. No earlier patch is needed, and
existing short-input benchmark files are unchanged. SHA256SUMS covers all other
package files. Interpreter caches, temporary fixtures, model weights and datasets
are not included. This delivery is one full patch, not an incremental update.

## Not executed here

- No official dataset archive or model weights were downloaded for a real run.
- No real Qwen tokenizer was available. Offline tests use an explicitly synthetic
  fixture; prepare and native prompt_tokens comparison must verify the actual
  tokenizer on the target host.
- No real ROUGE/LCC backend score was computed here; missing packages were not
  silently replaced by substitute metrics.
- No target RTX3090, vLLM startup, CUDA compilation, real generation, latency,
  memory, task-quality or new-algorithm experiment was executed.
- The new direct HTTP client is not the old EvalScope runner. Generation settings
  match the selected short-input protocol, but performance comparisons should use
  the same client/protocol batch; old and new numbers should not be merged as if
  client overhead, dataset and concurrency were identical.

On the target host, install the documented client dependencies, run the suite,
then the four-cases-per-task FP16/FP8 C1 smoke. Preserve the data, launch, run and
grading manifests with real outputs before expanding to a full experiment.
