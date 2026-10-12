# 长输入数据集与结果解读（补丁 v3）

本补丁默认采用 **LongBench v1 六任务、Qwen3 thinking=true、剩余上下文生成预算**。正式测量请求显式发送 `max_tokens: null`，不设置整数 `max_completion_tokens`，也不使用官方 LongBench 的 32/64/128/512 token 短输出上限。模型可以自然结束；硬长度边界由服务端实际输入长度和模型窗口决定。运行步骤见 [README.zh-CN.md](README.zh-CN.md)，版本与来源见 [SOURCES.md](SOURCES.md)。

这是使用 LongBench 数据和任务指标开展的思考模式研究协议。它改变了官方参考流程的生成策略，报告时应写清数据子集、thinking、采样参数和生成预算；选择同一数据集不代表直接复现相关论文表格。

## 1. 六个核心任务与两个扩展任务

以下样本数与平均长度来自[官方任务统计](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/task.md)，指原始 v1 数据。实际纳入数量由准备阶段的长度检查、筛选和抽样共同决定。

| 任务 key | 任务内容与本次用途 | 官方平均长度：英文词数 | v1 样本数 | 任务指标 |
|---|---|---:|---:|---|
| `qasper` | 依据 NLP 研究论文回答问题，含 yes/no/unanswerable；检查长文证据读取和答案限定 | 3,619 | 200 | 归一化 token F1 |
| `multifieldqa_en` | 多领域长文章问答，覆盖论文、司法和政府文档等来源；补充不同文体 | 4,559 | 150 | 归一化 token F1 |
| `hotpotqa` | 根据证据段落和干扰文档进行多跳问答；检查跨文档信息结合 | 9,151 | 200 | 归一化 token F1 |
| `2wikimqa` | 另一种多文档多跳问答分布；检验结果是否只适用于 HotpotQA | 4,887 | 200 | 归一化 token F1 |
| `gov_report` | 对长政府报告生成一页摘要；同时观察长输入利用与较长答案生成 | 8,734 | 200 | ROUGE-L F |
| `passage_retrieval_en` | 在 30 个 Wikipedia 段落中定位给定摘要的来源；检索目标明确，方便诊断 | 9,289 | 200 | 官方检索分数 |
| `qmsum`，扩展 | 按用户查询总结长会议记录；补充会议、对话和摘要场景 | 10,614 | 200 | ROUGE-L F |
| `lcc`，扩展 | 根据代码上下文补下一行；Python/C#/Java，检查代码上下文利用 | 1,235 | 500 | 编辑相似度 |

核心六项原始合计 **1,150 条**；加上两个扩展为 1,850 条。QMSum 可以加入思考模式测量。LCC 使用原始代码 completion，不接受 chat 的 thinking 控制，须单独以 `--no-thinking` 准备；它仍可采用默认的剩余上下文预算，正式请求同样发送 null。

这组核心任务覆盖单文档、多文档、摘要与检索，优先补充现有短输入 MATH-500/HumanEval 未覆盖的长输入负载。LCC 的官方平均输入较短，因此作为代码扩展，不作为首轮长输入压力的主要依据。这是本次实验设计选择。

## 2. v1、LongBench-E 与两种长度

可通过 `--variant longbench-e` 选择 E 数据。六个核心任务都有 E，但并非每项 300 条：

| E 数据 key | 原始 length <4,000 | 4,000 ≤ length <8,000 | length ≥8,000 | 合计 |
|---|---:|---:|---:|---:|
| `qasper_e` | 100 | 100 | 24 | 224 |
| `multifieldqa_en_e` | 67 | 70 | 13 | 150 |
| `hotpotqa_e` | 100 | 100 | 100 | 300 |
| `2wikimqa_e` | 100 | 100 | 100 | 300 |
| `gov_report_e` | 100 | 100 | 100 | 300 |
| `passage_retrieval_en_e` | 100 | 100 | 100 | 300 |

核心 E 合计 **1,574 条**。`qmsum` 没有 E，`lcc_e` 有 300 条。以上是原始分布，筛选或抽样后的分桶数量应以本次 manifest 为准。[官方 E 统计与构造说明](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/task.md)

**官方 length 不是模型 token 数。** 官方 schema 将它定义为 input、context、answers 三项总长；英文和代码用 Python split 计词数，中文计字符数。官方 E 计分按该字段的 4,000/8,000 阈值分桶。[官方 schema](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/README.md) · [官方 E 计分](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/eval.py)

补丁同时保留：

- `source_length`：原始词数，供 `official_word_length_bins` 使用。
- `original_input_tokens` 与 `input_tokens`：本地固定 tokenizer 计算的原始和实际请求长度；chat 包含实际模板和特殊 tokens。
- `local_input_token_bins`：按实际输入 token 分桶，用于分析上下文长度与运行成本。
- `remaining_context_tokens`：模型窗口减去实际输入 tokens，是默认协议下可用的生成空间；不是另加的固定输出 cap。

在同一 32K 窗口下，输入越长，可用于思考与最终答案的空间越少。解释长度分桶结果时，应一起看 FP16/FP8 基线、剩余空间、输出长度和 `finish_reason`，不能只把长输入下的质量下降归因于量化。

## 3. 生成协议与官方历史参考值

v3 默认设置为 `--generation-budget-policy remaining-context` 和 `--thinking`。正式测量的 `max_tokens` 为 null；输入必须严格短于模型窗口。准备阶段至少留出一个 token 只是合法性检查，**不表示输出只允许一个 token**。服务以 `--generation-config vllm` 启动，预算语义已按固定后端源码核验，详见 [SOURCES.md](SOURCES.md)。

**下表仅记录官方历史参考上限，不是本补丁默认，也不会在默认测量中发送：**

| 任务 | 官方参考 max_new_tokens |
|---|---:|
| Qasper | 128 |
| MultiFieldQA-en | 64 |
| HotpotQA / 2WikiMQA / PassageRetrieval-en | 32 |
| GovReport / QMSum | 512 |
| LCC | 64 |

这些数值来自[官方配置](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2maxlen.json)，保留在 `config/tasks.json` 供显式 `reference` 策略使用。选择该策略需同时 `--no-thinking`；若要对照历史参考流程，还须另行匹配其 greedy sampling 等设置。不要把不同生成策略的分数或延迟合并成同一组结果。

## 4. 如何读取任务分数与性能结果

补丁按最终答案计分，保存原始输出、思考内容和最终答案提取方式。思考未结束而没有最终答案，与网络失败或响应不完整属于不同情况，应分别查看。

| 指标 | 计算口径与解释 |
|---|---|
| QA F1 | 英文小写、去标点/冠词、合并空白后计算 token 多重集合 F1；每题取多个参考答案中的最高分 |
| ROUGE-L F | 使用 `rouge==1.0.1` 的 ROUGE-L F，再对参考答案取最高分；不能直接换成 rougeLsum 或另一套 stemming 配置 |
| 检索分数 | 参考答案指定 Paragraph N；预测里正确数字出现次数除以所有数字出现次数，无数字为 0 |
| LCC 编辑相似度 | 选第一个不含反引号、#、// 的输出行，用固定实现的 fuzzywuzzy 相似度比较参考下一行；不是执行测试 pass@1 |
| `task_score_0_100` | 全部预期样本获得可计分完整响应时的平均任务分；缺失或无效响应时为空，部分可用分与零填充诊断另列 |
| TTFT / TPOT / E2E | 客户端端到端时延；TTFT 可以是首个非空思考片段，TPOT 不是纯 GPU attention 或解量化时间 |
| 输出 tokens 与 length 结束率 | 输出 tokens 包含思考和最终答案；默认 null 协议下，length 结束应结合实际剩余上下文检查 |

任务算法来自[官方 metrics.py](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py)和[参考答案聚合逻辑](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/eval.py)。本补丁另加响应完整性检查、思考/答案提取和运行信息记录。

这些多数是连续任务分数，应称“F1、ROUGE-L、检索分数或编辑相似度”，不要统一称准确率。六任务均分应标为“本次六任务平均分”，不能标为全套 LongBench overall。若选择 skip、head-tail 或抽样，应同时报告纳入、排除、截断和完整计分的样本数。

## 5. 与相关 KV-cache 论文的关系

| 论文 | 一手实验依据 | 本次使用方式 |
|---|---|---|
| [KIVI](https://arxiv.org/html/2402.02750v2) | 4.1 节及附录 Tables 8–10 包含核心六项及 QMSum/LCC 等 | 六项均有量化论文中的实验先例 |
| [GEAR](https://arxiv.org/pdf/2403.05527v4) | 4.1 节评测 21 个 LongBench 任务，Table 2 突出 QMSum/SAMSum/GovReport | 支持真实长文与摘要评测 |
| [TurboQuant](https://arxiv.org/html/2504.19874v1) | 4.3 节明确使用 LongBench-E | 可选 E 数据进行相同长度分布的研究；默认 v1 是另一 split |
| [RotateKV](https://www.ijcai.org/proceedings/2025/0690.pdf) | Table 6 的 LongBench 部分是 QMSum、LCC、SAMSum、TriviaQA | QMSum/LCC 的扩展依据；该表不是本次六任务 |
| [KVarN](https://arxiv.org/html/2606.03458v1) | 实验含 MATH/AIME/HumanEval/IFEval、Line Retrieval、NIAH；LongBench 是引言举例 | 长输入评测补充现有持续解码任务，不能声称复现其 LongBench 结果 |

上述文献使用的模型、量化配置、生成预算和推理实现各不相同。v3 的首要比较对象应是同批、同一冻结输入和生成协议下运行的 FP16、FP8 与候选方法。

## 6. RULER：有论文依据的后续扩展

本次可执行补丁包含 LongBench v1/E，**未实现 RULER 运行入口**。[KVLinC](https://arxiv.org/pdf/2510.05373v1) 的 Tables 2/6 使用 RULER 4K/8K；[More Tokens, Lower Precision](https://aclanthology.org/2025.findings-emnlp.429.pdf) 的 Table 2/附录 A 使用 4K/8K/16K，可作为后续规则化长度实验的依据。

建议后续先接入官方 `niah_single_1`、`niah_multikey_1`、`vt` 三项，分别覆盖单目标检索、多 key 干扰和变量链追踪。须冻结官方生成器版本、tokenizer、配置、seed 与真实导出，保留 `input`、`outputs`、`answer_prefix`；本次核验的官方旧版推理器会拼接 input 与 answer_prefix。[官方 RULER 配置](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/synthetic.yaml) · [官方推理器](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/pred/call_api.py)

RULER 当前 README 已将 main 的旧流程标为 deprecated，并指向新的 NS 分支；后续适配须明确支持的导出版本。三项子集、自己生成的检索题和 LongBench PassageRetrieval 都不能标成全套官方 RULER 分数。

## 7. 本次推荐的测量顺序

1. 先用固定的小样本在 C1 跑 FP16、FP8，核验请求 null、真实输入 tokens、最终答案提取和离线计分，再加入候选方法。
2. 在相同输入与生成策略下比较任务分、思考/输出长度、TTFT、TPOT 和 E2E。默认自然生成允许不同方法产生不同输出长度，端到端时延变化不能全部归为单 token 执行变化。
3. 再扩大样本并做 C4/C8，观察原生 KV 容量、利用率与 preemption。每个请求最终都可能接近完整模型窗口，并发压力应按真实常驻上下文评估。
4. 保留 MATH-500/HumanEval 作为持续生成测试。长文 QA、摘要与原有短输入长输出结果共同用于确定新方法有效的负载区间。

所有方法复用一次准备得到的 manifest，不根据某方法的得分重新挑样本。实际 GPU 测量尚需在用户目标环境完成；本文件给出数据、协议与解读口径，不宣称新增 benchmark 成绩。
