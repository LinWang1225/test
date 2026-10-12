# 长输入 KV Cache 评测补丁 v3：自然生成，不设额外输出上限

本版是完整补丁，直接应用到 `LinWang1225/test`，不需要先应用 v1/v2。
核验基点：`3cf8dd9345e8e8181b0069b6b39c3355ae463b67`。新增 `kvbench_longcontext/`；既有短输入实验由原 runner 继续负责。

## 1. 本版默认协议

| 设置 | 默认值/行为 |
|---|---|
| 模型 | 复用现有本地 Qwen3-4B snapshot |
| Thinking | true |
| Sampling | temperature=0.6, top_p=0.95, top_k=20, min_p=0 |
| 正式请求输出预算 | 显式 JSON `max_tokens: null`；不发送整数 `max_completion_tokens` |
| 服务生成配置 | `--generation-config vllm`，不设置生成长度 override |
| 硬上下文窗口 | 32768；输入+生成总 token 受此窗口限制 |
| 默认请求耗时限制 | `--request-timeout 0`，不设置整请求 elapsed/阻塞读取超时 |
| EOS/stop | 正常自然结束；不强制生成满窗口，不使用 ignore_eos |
| 输出截断 | 完整 `finish_reason=length` 响应照常计性能、评分、分母 |
| APC | 关闭 |
| 服务配置 | TP1、float16 weights、max-seqs16、batch-tokens1024、gpu-util0.85 |
| 并发 | 首轮 C1；正式对照可显式选择 C8 |

在已核验的 KVarN/vLLM CUDA 后端上，这个配置使生成长度硬限制为：

`剩余上下文 = 服务端实际 max_model_len - 服务端实际 prompt_tokens`。

例如同样 32768 窗口，输入 8192 的题可生成至最多约24576，输入24576的题最多约8192。数字由输入决定，不是补丁另设了8192输出cap。实际仍可提前EOS结束。详见 `BUDGET_SEMANTICS.zh-CN.md`。

短连接预检、每点长 prompt 预热仍用至多16/32 token的短预算，且不超过该题剩余空间。它们独立保存、完全排除于测量分数/吞吐/延迟；正式请求全部采用上面的null策略。

## 2. 应用完整新补丁

```bash
cd /path/to/test
git apply --check /path/to/kvbench_longcontext_v3_natural_20261009.patch
git apply /path/to/kvbench_longcontext_v3_natural_20261009.patch
```

把路径换成实际 Git checkout 和下载文件。你尚未应用旧版，直接执行本节即可。

## 3. 环境与依赖

复用现有 `evalscope-client` 和 `kvarn-serve` 环境。以下变量在两个终端中分别设置；`REPO` 是 Git checkout，`KVBENCH_ROOT` 是现有模型/数据/实验目录。

客户端终端：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate evalscope-client
export REPO=/path/to/test
export KVBENCH_ROOT=/home/wanglin/kvbench-3090-cu126
source "$KVBENCH_ROOT/model.env"
cd "$REPO"

python -m pip install -c "$KVBENCH_ROOT/constraints/tokenizer-stack.txt" \
  -r kvbench_longcontext/requirements-client.txt
python -m pip check
python -m unittest discover -s kvbench_longcontext/tests -v
```

这些依赖安装在客户端，沿用已上传V8的tokenizer版本。LCC另需`requirements-lcc.txt`；核心六项不需要它。

## 4. 准备数据：先每任务4题

默认六项：qasper、multifieldqa_en、hotpotqa、2wikimqa、gov_report、passage_retrieval_en。默认下载固定官方数据archive并核验SHA，不运行远程数据集Python代码。

```bash
export LC_DATA="$KVBENCH_ROOT/data/longbench_v3_smoke_8k"
python -m kvbench_longcontext.prepare \
  --model "$MODEL" --out "$LC_DATA" \
  --variant longbench \
  --sample-size 4 --data-seed 42 \
  --min-input-tokens 1024 --max-input-tokens 8192 \
  --max-model-len 32768 --overflow skip \
  --thinking --generation-budget-policy remaining-context
```

没有 `--max-new-tokens`。`--max-input-tokens 8192` 只筛选实际输入范围，不限制输出为8192。所有token预算都包含真实聊天模板。

若使用LongBench-E，改为`--variant longbench-e`并使用新目录。QMSum只有v1；LCC是raw completion，须单独加`--no-thinking`准备，仍显式发送max_tokens:null。

数据目录含`manifest.json`、各任务JSONL、原始`raw/*.jsonl`、`*.excluded.jsonl`。相同方法对照必须复用同一份数据目录；不同seed或输入范围重新prepare可能改变题集。

`--overflow skip`明列排除样本；这些结果应标为该输入范围内的固定子集。默认error会拒绝超长输入；head-tail是显式裁掉中间的输入截断变体，可能删去证据，须单独报告。正常remaining模式至少保留1个可生成token，但接近32768的输入可能没有充分思考空间，应查看remaining_context分布。

网络受限时可以先取得官方data.zip，用以下选项替换在线来源：

```bash
--source /absolute/path/data.zip \
--source-revision 5e628be450b7e67fb7ae6e201bd6d8f7056f7672
```

已有缓存可加`--offline`；如果现有shell设置了HF_HUB_OFFLINE，在线下载也会受该环境设置影响。数据本身与模型权重不包含在补丁内。

## 5. 终端A：启动一个方法

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kvarn-serve
export REPO=/path/to/test
export KVBENCH_ROOT=/home/wanglin/kvbench-3090-cu126
export KIT=/home/wanglin/kv_evalscope_3090_cu126_v2
source "$KIT/env.sh"
source "$KVBENCH_ROOT/model.env"
cd "$REPO"

export LC_RESULTS="$KVBENCH_ROOT/results/longcontext_v3"
export METHOD=fp16
python -m kvbench_longcontext.serve \
  --model "$MODEL" --method "$METHOD" \
  --out "$LC_RESULTS/servers/${METHOD}_smoke" \
  --max-model-len 32768 --max-seqs 16 --batch-tokens 1024 --gpu-util 0.85
```

看到`Ready; native dtype/block verified`后运行客户端。启动器记录原生窗口、KV block与容量、模型/tokenizer身份、命令和引擎源码身份。它只管理自己启动的服务，发现已有GPU进程或端口占用则报错。

支持fp16、fp8、kvarn、kvarn44、kvarn64、kvarn4464、tq4、tq34、tq3。g64明确block64；FP8启用calculate-kv-scales。TQ84未纳入健康路径列表。

## 6. 终端B：运行、评分、生成速读表

在客户端终端，保持第3/4节变量，并与服务端使用相同METHOD：

```bash
export LC_RESULTS="$KVBENCH_ROOT/results/longcontext_v3"
export METHOD=fp16
export LC_RUN="$LC_RESULTS/${METHOD}_smoke_C1"

python -m kvbench_longcontext.run \
  --data-dir "$LC_DATA" --out "$LC_RUN" \
  --concurrency 1 --dry-run

python -m kvbench_longcontext.run \
  --data-dir "$LC_DATA" --out "$LC_RUN" \
  --server-manifest "$LC_RESULTS/servers/${METHOD}_smoke/launch.json" \
  --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
  --concurrency 1 --repeats 1 --request-timeout 0 --gpu 0

python -m kvbench_longcontext.scoring --data-dir "$LC_DATA" --run-dir "$LC_RUN"
python -m kvbench_longcontext.report --run-dir "$LC_RUN" --out "$LC_RUN/readout"
```

`dry-run`只检查冻结数据和显示计划，既不发请求也不创建结果目录。真实运行逐请求核对native prompt_tokens。数据目录、服务目录、推理目录都使用新路径，避免混入旧结果。

每个方法要依次完成run→scoring→report。FP16结束后，在终端A按Ctrl-C退出本次服务；两端METHOD改fp8并改用新服务目录/结果目录，继续用同一LC_DATA。再根据需要测试目标低比特方法。

## 7. 正式扩展

每任务32题：重新prepare到新数据目录，将sample-size改32；可扩输入上限到24576。此时实际输入为8192的样本依然保留24576生成空间，不是固定8192输出。

全部合格样本：将sample-size改0。要尽量覆盖所有可放入窗口的原样本，可以省略max-input-tokens/min-input-tokens，使用remaining-context与overflow skip；仍须报告排除数量及remaining_context分布，不能把只剩很少生成空间的题当成同样资源条件。

并发统一时，run可指定`--concurrency 8`或`--concurrency 1 4 8`，每任务样本数至少达到最大C。run默认C1只用于先跑通；C1与旧C8不是同一性能对照。长输入+C8可能触及FP16池容量，须同时看preemptions。没有输出cap时，不应再用固定512生成量估算并发容量上界。

重复运行可设`--repeats 3`，每个repeat保存独立point与请求seed。它不是三次服务冷启动，也不会自动计算配对统计显著性。

多个方法已有结果后：

```bash
python -m kvbench_longcontext.report \
  --run-dir "$LC_RESULTS/fp16_smoke_C1" "$LC_RESULTS/fp8_smoke_C1" \
  --out "$LC_RESULTS/compare_smoke_C1"
```

优先打开`compare_smoke_C1/overview.md`或CSV，按同一任务、同一清单、同一并发比较。核心六任务的分数不是一个跨任务总体accuracy。

## 8. 结果在哪里与怎样看

完整说明见`RESULTS_GUIDE.zh-CN.md`，最常用路径如下：

| 位置 | 内容 |
|---|---|
| `$LC_DATA/manifest.json` | 数据版本、题目清单、输入/剩余空间分布、预算策略 |
| `$LC_RESULTS/servers/<method>_smoke/launch.json` | 服务实际配置与原生窗口/容量 |
| `$LC_RUN/run.json` | 运行计划、方法、thinking、采样、null传输策略、时间限制 |
| `$LC_RUN/readout/overview.md` / `overview.csv` | 质量、完整性、性能总览，先看这里 |
| `$LC_RUN/quality.summary.json` | 每任务/并发/repeat的质量汇总 |
| `$LC_RUN/performance.summary.json` | 已完成测量点的性能汇总 |
| `$LC_RUN/r01_qasper_C1/scores.jsonl` | Qasper逐题得分、终止原因、空答案标记 |
| `$LC_RUN/r01_qasper_C1/requests.jsonl` | 逐题实际请求、reasoning、final、usage、时延 |
| 同一point下`measure.requests.jsonl` | 实际正式请求体，max_tokens应为null |
| 同一point下`metrics.before.txt`、`metrics.after.txt`、`telemetry.jsonl` | 原生指标和运行时采样 |

quality summary按计划包含未运行点，performance顶层只包含已落盘点；report按run.json的计划列出全部point，不会隐藏缺失点。

## 9. 截断评分标准

完整且通过传输/原生计数校验的`finish_reason=length`响应永远保留：有final就对该final正常评分，可能为0、部分分或满分；最终答案为空时按空答案计0分，包括thinking未结束而没有final。所有这些样本都进入正式样本分母和性能统计。

即使出现“length终止却早于按原生输入推导的预算”的异常，完整响应仍评分；逐题记录unexpected_length_before_budget，每点汇总unexpected_length_before_budget_count，并记录generation_budget_consistent。应排查服务端预算与token计数差异，不能仅凭这个标记断定有隐藏上限，也不能跳过排查就称这批已满足无额外cap协议。

HTTP错误、断流、缺原生usage或prompt计数不匹配属于基础设施/协议不完整，不能和正常输出截断混为一谈。主质量分为null/NA，另给partial与zero-filled诊断分，不能代替完整主分。

## 10. 停止运行与验证范围

默认不设置请求总时间上限，模型可能运行很久。需要诊断时间界限时可显式设正数request-timeout；它产生的中断按不完整响应处理，不是正常length终止。

需要提前停止时，先结束自己在终端A启动的服务，使连接关闭；客户端可保存已收到的断流记录。已完成记录逐行flush。强制杀死客户端不保证正在内存中接收的全部片段都能落盘。

本补丁提供数据准备、测量、评分与速读流程；没有实现新KV算法、固定生成步数微基准或RULER。真实3090/vLLM和官方数据集推理未在交付环境执行。实际离线测试与未执行范围见`VALIDATION.md`。
