# Source and provenance ledger: patch v3

This v3 extension is based on [LinWang1225/test at 3cf8dd9345e8e8181b0069b6b39c3355ae463b67](https://github.com/LinWang1225/test/tree/3cf8dd9345e8e8181b0069b6b39c3355ae463b67). It provides LongBench v1/E data preparation, a local streaming client, offline scoring and a launcher for the installed KVarN/vLLM backend.

**The default v3 measurement protocol uses Qwen3 thinking=true and remaining-context generation. Every measured request explicitly sends max_tokens:null. Official LongBench short output caps are historical reference settings and are not the v3 default.**

## Fixed sources

| Component | Pinned identity |
|---|---|
| LongBench prompt, task and scorer source | `THUDM/LongBench@2e00731f8d0bff23dc4325161044d0ed8af94c1e` |
| Official Hugging Face dataset | `zai-org/LongBench@5e628be450b7e67fb7ae6e201bd6d8f7056f7672` |
| Official data.zip SHA256 | `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64` |
| Audited inference backend | `huawei-csl/KVarN@7586257f1c632e63187bfacbbe21ccb51540f7b3` |

## LongBench v1 and LongBench-E

The original benchmark's files are now under the official repository's `LongBench/` directory; the repository root also contains the separate LongBench v2 benchmark.

- [Official prompt strings](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2prompt.json).
- [Historical reference output caps](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2maxlen.json): used only when the reference policy is explicitly selected.
- [Official task metrics](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py).
- [Max-over-references aggregation and original E length bins](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/eval.py).
- [Task descriptions, counts and E subsets](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/task.md).
- [Dataset schema](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/README.md).
- [Historical model-specific prediction workflow](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/pred.py).

Eight English prompt strings and their historical caps are copied into `config/tasks.json`. Supported English metrics and the max-over-references rule are adapted in `scoring.py`. The MIT license is included as `LICENSE.LongBench`; the source copyright is THU-KEG & Zhipu AI, 2023. This software license does not replace the original datasets' source attribution or licenses.

The old Hugging Face identifier `THUDM/LongBench` redirects to `zai-org/LongBench`. The default download contains data, without executing the dataset's remote Python loading script:

- [Pinned official data.zip](https://huggingface.co/datasets/zai-org/LongBench/resolve/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip).
- [Dataset revision](https://huggingface.co/datasets/zai-org/LongBench/commit/5e628be450b7e67fb7ae6e201bd6d8f7056f7672).
- [Official archive pointer and checksum](https://huggingface.co/datasets/zai-org/LongBench/blob/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip).

The default archive hash is verified by preparation. Prepared original JSONL and request JSONL files are also hashed. Local imports record raw byte hashes and a supplied revision; an unspecified revision is recorded as `unknown-local-export`. Dataset text is not bundled with the patch.

The source `length` is an English word count, not the local model's token count. It is retained for `official_word_length_bins`. Actual prompt token counts, including the chat template where applicable, are recorded separately for `local_input_token_bins` and remaining-context checks. The six core tasks contain 1,150 original v1 examples or 1,574 E examples before filtering; Qasper-E has 224, MultiFieldQA-en-E has 150, and the other four core E datasets have 300 each. QMSum has no E split.

## v3 generation budget: backend evidence

The following sources are from the exact audited backend commit, not a generic assumption about all OpenAI-compatible servers.

| Question | Source and observed logic |
|---|---|
| Chat omission versus null | [ChatCompletionRequest](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/entrypoints/openai/chat_completion/protocol.py): both max_tokens and max_completion_tokens default to None. |
| Chat field precedence | [Chat serving](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/entrypoints/openai/chat_completion/serving.py): a non-None max_completion_tokens takes precedence over max_tokens. |
| Raw completion omission versus null | [CompletionRequest](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/entrypoints/openai/completion/protocol.py): omitted max_tokens defaults to 16; explicit null produces None. |
| Effective generation budget | [get_max_tokens](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/entrypoints/serve/utils/api_utils.py): takes the minimum non-None value among remaining model context, request/default budget, server override and platform limit. |
| Model generation defaults | [ModelConfig.get_diff_sampling_param](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/config/model.py): generation_config=vllm begins with an empty sampling-config dictionary; explicit override_generation_config is still applied. max_new_tokens is mapped to max_tokens. |
| CUDA platform limit | [Platform interface](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/platforms/interface.py) returns sys.maxsize; [CUDA classes](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/platforms/cuda.py) do not override that method. |

Consequently, with the shipped `--generation-config vllm` launcher and no explicit `max_new_tokens` server override, v3's null request resolves to:

~~~text
effective_max_tokens = max_model_len - backend_actual_input_tokens
~~~

The implementation sends null explicitly for both chat and raw completion. No integer max_completion_tokens is added. This avoids the raw completion endpoint's default of 16. EOS and stop conditions can still end generation earlier; remaining-context is a maximum, not a forced output length.

The old short-input experiment also used a null request sentinel. This is directly visible in [V7's measured request JSONL](https://github.com/LinWang1225/test/blob/7801b642d03a27aca0486341f4cde6ab4ce960d5/results/real/full_v7_C8/points/r01_fp16_math_500_apc0_C8/attempt001/measure.requests.jsonl). Its [runner](https://github.com/LinWang1225/test/blob/7801b642d03a27aca0486341f4cde6ab4ce960d5/kvbench_real_v6/run_real.py) explains that null prevented EvalScope's missing-key default of 2048 from being inserted.

## Model and measurement protocol

The [Qwen3-4B official model card](https://huggingface.co/Qwen/Qwen3-4B#best-practices) documents the thinking switch and recommends temperature=0.6, top_p=0.95, top_k=20 and min_p=0 for thinking. These are the default v3 sampling values for the chat tasks. LCC remains a raw completion task and is prepared separately with `--no-thinking`.

The official LongBench reference workflow used greedy decoding and task-specific short caps. V3 changes generation to thinking plus remaining context, while retaining source prompts and task metrics. Even selecting E does not make the v3 run a numerical reproduction of the TurboQuant or LongBench paper tables. Explicit `reference` and `fixed` policies are separate experiments; their settings must be reported with their results.

The SSE client calls the locally launched API directly and retains usage-only chunks. It records native token usage, response completion, prompt-count consistency, raw content, reasoning content and final-answer extraction. Offline scoring uses the extracted final answer. An unfinished thinking response with no final answer is distinguished from a missing or invalid response.

TTFT includes time to the first nonempty content or reasoning chunk. TPOT uses native completion tokens, including thinking tokens, with a request-equal mean. These are client measurements, not isolated GPU kernel timings. Compare methods within the same new batch; do not splice new client timings into old EvalScope measurements as though their measurement paths were identical.

## Verified relevance to KV-cache papers

- [KIVI, arXiv:2402.02750v2](https://arxiv.org/html/2402.02750v2), section 4.1 and appendix Tables 8–10: includes the six core tasks and optional QMSum/LCC among its LongBench evaluations.
- [TurboQuant, arXiv:2504.19874v1](https://arxiv.org/html/2504.19874v1), section 4.3: explicitly uses LongBench-E. The default v1 split differs.
- [GEAR, arXiv:2403.05527v4](https://arxiv.org/pdf/2403.05527v4), section 4.1/Table 2 and appendix: evaluates the 21-task LongBench suite.
- [RotateKV, IJCAI 2025](https://www.ijcai.org/proceedings/2025/0690.pdf), Table 6: its LongBench portion contains QMSum, LCC, SAMSum and TriviaQA.
- [KVarN, arXiv:2606.03458v1](https://arxiv.org/html/2606.03458v1): evaluates reasoning/generation tasks, Line Retrieval and NIAH; LongBench is mentioned as background, not presented as its experimental suite.
- [KVLinC, arXiv:2510.05373v1](https://arxiv.org/pdf/2510.05373v1), Tables 2/6: RULER at 4K/8K, with LongBench results also covering Qwen3-4B.
- [More Tokens, Lower Precision, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.429.pdf), Table 2/Appendix A: RULER at 4K/8K/16K in a pruning-plus-quantization study.

RULER is a documented future extension, not an implemented entry point in this patch. The inspected official source is [NVIDIA/RULER at c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a). Its README now points users from the deprecated main pipeline to NS branches. A future adapter must pin its actual generator, template, answer-prefix handling and export schema. Synthetic filler or LongBench retrieval results are not official RULER results.

Task descriptions, supported counts and interpretation guidance are in [DATASETS_RESEARCH.zh-CN.md](DATASETS_RESEARCH.zh-CN.md). Source inspection and offline software checks do not establish target-GPU benchmark results; those measurements must be run in the user's fixed inference environment.
