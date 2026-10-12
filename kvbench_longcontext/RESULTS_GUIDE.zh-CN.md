# v3 结果位置与解读

本说明对应 README 的默认协议：thinking=true、自然采样、正式请求 max_tokens:null、只受剩余上下文约束、默认不设请求总时限。有效完整的输出截断仍评分。

## 1. 本补丁完成什么、为了测什么

| 模块 | 完成的工作 | 要回答的问题 |
|---|---|---|
| prepare | 固定官方数据版本、题目顺序、提示与多参考答案，记录真实输入 token 和剩余空间 | 方法之间是否使用同一题集、同一答案和预算规则？ |
| serve | 启动已有 KV 方法，核对原生 dtype、block、APC、窗口和容量 | 实际运行的是哪个配置，KV 池有多少 token 槽位？ |
| run | 逐任务、并发、repeat 测量自然生成，保留 reasoning/final/usage/时延 | 长输入下速度、生成长度、截断、空答案和抢占怎样变化？ |
| scoring | 按 LongBench 对应指标离线评分，保留完整截断响应 | KV 资源节约是否伴随任务质量损失？ |
| report | 生成 Markdown/CSV 质量与性能表，保留计划中未完成的点 | 是否已有完整可比证据，应该检查哪个任务或样本？ |

核心六任务覆盖单文档问答、多跳问答、摘要和检索。本补丁提供评测流程，没有实现新的 KV 算法，也不包含已运行的真实 GPU 实验结果。

## 2. 结果在哪里

README 默认 LC_RESULTS 为 `/home/wanglin/kvbench-3090-cu126/results/longcontext_v3`。FP16 smoke 的 LC_RUN 为 `$LC_RESULTS/fp16_smoke_C1`；FP8 改为 fp8_smoke_C1。

| 路径 | 内容与用途 |
|---|---|
| `$LC_DATA/manifest.json` | 数据版本、采样 seed、入选 ID/hash、输入与剩余空间分布、排除数量 |
| `$LC_DATA/<task>.jsonl` | 冻结后的逐题输入、答案、输入 token 与 null 预算；不要手改后复用原清单 |
| `$LC_DATA/<task>.excluded.jsonl` | 因输入范围或窗口被排除的题及原因 |
| `$LC_RESULTS/servers/<method>_smoke/launch.json` | 启动配置、实际窗口、KV 容量、模型与引擎源码身份 |
| 同一服务目录下 `server.log` | 服务日志，启动或运行失败时查看 |
| `$LC_RUN/run.json` | 全部计划点、方法、并发/repeat、采样、数据 hash、时间限制、容量 |
| `$LC_RUN/server.launch.json` | 本次运行使用的服务清单副本 |
| `$LC_RUN/readout/overview.md` / `overview.csv` | 先看这里：质量、完整性、性能；CSV 空白数值为 NA |
| `$LC_RUN/quality.summary.json` | 全部计划点的质量汇总，包含未运行/缺失点 |
| `$LC_RUN/performance.summary.json` | 已完成测量并落盘的点的性能汇总，未落盘点可能不在此文件中 |
| `$LC_RUN/grading_protocol.json` | 评分源码 hash、依赖版本、数据 hash、上游版本 |

每个点直接在 LC_RUN 下。例如 `r01_qasper_C1` 表示第1次重复、Qasper、请求并发1，路径是 `$LC_RUN/r01_qasper_C1/`，没有额外 points 层。

| 每点文件 | 内容 |
|---|---|
| `point.json` | 本点计划与请求顺序 |
| `performance.summary.json` | 延迟、吞吐、截断、空答案、原生计数及预算诊断 |
| `scores.summary.json` | 质量主分、分母、缺失/无效数量、长度分箱 |
| `scores.jsonl` | 逐题得分，以 case_id 与其他记录对应 |
| `requests.jsonl` | 逐题实际请求、reasoning、final_answer、usage、finish_reason、延迟、error |
| `measure.requests.jsonl` | 实际正式请求体，用来确认 max_tokens:null 等参数 |
| `raw/*.json.gz` | 原始流式事件，用于解析或传输问题排查 |
| `metrics.before.txt` / `metrics.after.txt` | 测量前后原生服务指标 |
| `telemetry.jsonl` | 测量期间服务与 GPU 指标采样，部分值可能不可用 |
| `warmup/` | 独立短预算预热，完全排除于正式分数和性能 |

## 3. 先看完整性，再看分数

每任务4题的 smoke 共24题/方法/repeat/C。题集在 prepare 时固定，不能因为结果不好而减少分母。

| 字段 | 含义与检查方法 |
|---|---|
| expected_tasks | 应测题数，应等于冻结数据入选数量 |
| valid_responses | 完整且通过传输/原生计数/预算校验的响应数；包括完整 length 响应 |
| scored_tasks | 正常完成评分的题数；真实0分仍计入 |
| measurement_complete | 全部计划响应完整有效；false 时性能可能只是部分响应统计 |
| evaluation_complete | 全部计划题均完成评分；false 时主质量分为 null/NA |
| generation_budget_consistent | 原生 usage 与推导预算是否一致；false 需排查，null 表示无法判断 |
| unexpected_length_before_budget_count | 完整 length 响应的输出 token 少于推导剩余空间的题数；这些题仍评分 |

理想状态为 expected_tasks=valid_responses=scored_tasks，两个完整性标记都为 true。预算一致性独立于评分完整性：提前 length 的完整答案仍可正常评分，但需要核对服务预算和 token 计数。这个标记本身不能证明一定有隐藏 cap，更不能据此删除样本。

错误计数可能重叠：HTTP 错误也可能同时被计为不完整流和缺 usage。不要简单相加各错误列；优先看 expected_tasks−valid_responses，再按 case_id 排查。

## 4. 质量分代表什么

逐题 score 为0–1，task_score_0_100 为逐题均值乘100。多参考答案按题取最高匹配分。评分只用 final_answer；reasoning 单独保留，仍计入全部生成 token 与性能。

| 任务 | 指标 | 含义 |
|---|---|---|
| Qasper / MultiFieldQA-en | qa_f1 | 归一化词级答案 F1，额外解释也可能影响匹配 |
| HotpotQA / 2WikiMQA | qa_f1 | 多文档问答的答案 F1；没有另测 supporting-fact 指标 |
| GovReport / QMSum | rouge_l | 摘要与参考摘要的 ROUGE-L F1，不是事实正确率 |
| PassageRetrieval-en | retrieval | 输出段落编号的匹配比例；多个编号可能稀释分数 |
| LCC | code_sim | 按官方规则提取代码行后的编辑相似度，不是 HumanEval pass@1 |

这些是连续任务分，不是统一的二元 accuracy。跨任务不宜直接比较分值大小，方法对照应先在同一任务内进行。

### 输出截断、真实0分与缺结果

| 响应情况 | 处理 |
|---|---|
| 完整 stop，有 final | 正常评分 |
| 完整 length，有全部或部分 final | 对实际 final 正常评分，可以0、部分分或满分，保留分母和性能 |
| 完整 length，但 thinking 未结束、没有 final | 空答案0分，保留分母和性能 |
| 完整 stop，但 final 为空 | 同样是空答案0分 |
| 完整 early length，原生计数有效 | 正常评分，另记预算差异 |
| HTTP错误、断流、缺失、缺usage、prompt计数不匹配 | 不完整主分为NA，不伪装成完整0分 |

partial_available_score_0_100 只平均可评分部分；all_expected_zero_filled_score_0_100 将未评分部分暂当0后除以全部应有题数。二者是缺失影响诊断，不能替代完整主分。

## 5. 性能字段代表什么

| 字段 | 单位 | 解读 |
|---|---|---|
| input_tokens_mean | token/请求 | 原生 prompt token 均值，含模板 |
| output_tokens_mean | token/请求 | 原生 completion token 均值，包含思考和最终答案 |
| ttft_ms_mean / ttft_ms_p95 | 毫秒 | 发出请求至首个非空 reasoning/content 片段；含排队/网络，不是纯 prefill kernel 时间 |
| tpot_ms_mean | 毫秒/token | 各请求 `(E2E−TTFT)/(completion_tokens−1)` 的等权均值；越低表示该口径下生成越快 |
| tpot_samples | 请求数 | 可以计算TPOT的请求数；输出≤1 token时不可算，不能填0 |
| e2e_s_mean / e2e_s_p95 | 秒/请求 | 请求发出至流接收结束；同时受排队、输入和生成长度影响 |
| output_tokens_per_s | token/秒 | 有效响应生成token总和除以本点请求测量窗口，是服务吞吐 |
| valid_requests_per_s | 请求/秒 | 有效响应数除以同一窗口 |
| length_capped | 请求数 | 有效完整响应中 finish_reason=length 的数量 |
| length_capped_percent_of_valid | % | length_capped / valid_responses ×100 |
| empty_final_answers | 请求数 | 有效响应中 final 为空或纯空白的题数，不是基础设施失败数 |
| preemptions_delta | 事件数 | 测量前后原生累计抢占计数的增量，反映本轮调度压力；NA表示无法可靠计算 |

TPOT 来自 HTTP 流式片段与原生 token 数，是平均估计，不是每个 GPU decode step 的直接计时。每点只有4题时，p95/p99几乎等于最大值，只用于 smoke 诊断。

不能单看 E2E 宣称更快：生成更少、提前截断或没有 final 都可能缩短请求。也不能因某方法思考更长导致 E2E 上升，就推断每 token 更慢。至少同时比较任务分、TPOT、输出长度、E2E、吞吐和截断率。

## 6. 长度与容量如何解释

remaining_context_tokens_native 是“已验证服务窗口−原生 usage.prompt_tokens”的派生值，不是服务端原样返回的同名字段；native_remaining_source 明确记录来源。默认 null 模式下，这也是推导的最大生成空间。

source_length 是原始 LongBench 英文词数，用于 `<4000`、`4000–7999`、`>=8000` 分箱。local_input_token_bins 使用模板后的实际输入 token，区间为 `[0,4096)`、`[4096,8192)`、`[8192,16384)`、`[16384,24576)`、`[24576,+∞)`。两者不能互换。

was_truncated 表示 prepare 的 head-tail **输入截断**；length_capped / finish_reason=length 表示**输出截断**。README 默认 overflow=skip，不会静默裁输入，排除题列在 excluded 文件。

native_kv_capacity_tokens = num_gpu_blocks × block_size，表示KV池的token槽位容量，不是当前活跃token数或显存字节数。nvidia-smi 的预分配大块显存不能直接用来比较每token成本；应结合原生池容量、cache usage、抢占和并发解释。

相同32768窗口下，长输入会留下更少生成空间；边界输入可能只剩1个token。质量下降可能同时涉及输入难度和思考空间，需结合 remaining_context 分布、终止原因和空答案分析。不同数据集分数差也不能单独证明纯长度效应。

## 7. 推荐顺序

1. 同一份每任务4题数据，依次完成 FP16 C1、FP8 C1 的 run→scoring→report，确认接口、原生计数、各任务评分完整。
2. 先核对完整性、实际null请求、窗口、截断/空答案、预算诊断，再比较质量和性能。
3. 用同一冻结数据、模型、窗口、thinking、采样、并发和请求seed对照目标KV方法；质量差用百分点表示，速度差注明口径。
4. 扩至每任务32题或 sample-size=0，正式并发可统一用C8。每任务样本数至少等于最大C。C1不能直接与旧C8作性能对照。
5. 正式结论补足题量、重复/种子和配对分析。repeats不是服务冷启动，当前报告不自动给出统计显著性。

smoke通过后可以开展新方法的探索性实验；真正的新算法仍需在推理后端实现并接入配置。补丁测试通过不等于新方法质量或速度已获验证。

排查坏样本时，先从 scores.jsonl 取 case_id，再查 requests.jsonl 的同一ID，必要时查 raw 和 server.log。请求按完成顺序落盘，不要仅按行号与输入配对。

交付环境没有执行真实GPU推理，已完成和未完成的验证见 VALIDATION.md。
