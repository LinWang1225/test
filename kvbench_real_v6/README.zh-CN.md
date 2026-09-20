# KVBench v5：MATH-500 / HumanEval 自然生成、原生缓存统计与逐请求存档

本包替代 v4 的“随机输入 + 固定输出长度”主流程。只复用现有 Conda 环境和原来的 `serve.sh` / `env.sh`；不重新安装 vLLM，不改变 Torch/CUDA 版本，不覆盖旧实验包。

测量数据来自真实题目。保持原题长度，不拼接无意义长文本，不强制生成固定 token 数。生成结束后，对**同一份已保存输出**离线评分，不再次调用模型。

## 0. 验证范围与边界

开发环境没有你的 GPU、已安装的 EvalScope/vLLM，也不能下载数据或启动 Docker。本包完成 Python/Bash 语法检查、15 项协议/数据库测试和 1 项带模拟边界的流程集成测试。该集成测试不等于实际 EvalScope、Conda、Docker 或 GPU 的端到端验证。真实数据下载、模型推理与评分需在实验机上验证。

接口根据 EvalScope v1.12.0 源码与现有 Conda v3 `serve.sh` 核对。模板在准备数据时从**实际安装的 EvalScope registry**读取；输出解析只接受已核对的 SQLite schema，发现格式变化会报错并保留原数据，不猜测或丢弃。

## 1. 已明确的实验协议

- 数据集：MATH-500 全部 500 题；HumanEval 全部 164 题。pilot 固定抽取子集，full 使用全部题目；0-shot 模板来自现有 EvalScope，题号与内容 SHA256 固定。
- 方法：`fp16 / tq4 / kvarn`。分别对应当前 serve.sh 中的 FP16、TurboQuant 4-bit 工程预设、KVarN K4V2 g128。工程配置比较不代表同位宽比较。
- 生成：默认 thinking=true、temperature=0.6、top_p=0.95、top_k=20、min_p=0、n=1、repetition_penalty=1、frequency/presence penalty=0。每题 seed 由题号和基础 seed 确定，各方法使用相同题目与采样参数；这不是跨调度逐 token 一致性保证。
- 自然停止：默认**不发送请求级 `max_tokens`**，也不发送 `min_tokens`、`ignore_eos` 或人为 `stop`。现有 `serve.sh` 使用 `--generation-config vllm`，因此不会继承模型仓库里的 `max_new_tokens` 作为服务器级上限；生成通常由 EOS 停止，硬边界是服务端 `max_model_len` 的剩余上下文（以及底层平台限制）。`finish_reason=length` 单列为长度/上下文边界终止，不伪称自然完成。
- 引擎起始配置：单卡、FP16、TP=1、max_model_len=32768、max_num_seqs=16、max_num_batched_tokens=1024、gpu_memory_utilization=0.85。只是当前 Qwen 实验的起始配置，不是任何 3090 都能运行的保证。
- 先 APC=0。需要前缀实验时明确设置 APC=1；APC=1 每个点重启服务，不预先运行正式题目，不假造命中。
- 不做轨迹分叉定位，不记录 logits，不强制固定生成长度。

**MATH-500/HumanEval 是数学/代码任务，不等于长输入测评集。**长输出可能形成较大 KV，但它们不能替代长文档问答。不要为制造高缓存命中而默认给原基准灌入长前缀。以后再添加真正长输入的独立任务。

## 2. 路径与 Conda

把本包解压到已存在的工作目录，例如：

```bash
export KVBENCH_ROOT="$HOME/kvbench-3090-cu126"
export ROOT="$KVBENCH_ROOT"
cd "$ROOT"
unzip kvbench_real_v5.zip
export REAL="$ROOT/kvbench_real_v5"

# 保留已经跑通的旧 KIT 路径；只在未设置时填写实际目录。
export KIT="${KIT:-$HOME/kv_evalscope_3090_cu126_v2}"
ls "$KIT/serve.sh" "$KIT/env.sh"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate evalscope-client
source "$ROOT/model.env"
export MODEL

export KVBENCH_SERVE_ENV=kvarn-serve
export KVBENCH_EVAL_ENV=evalscope-client
```

若 `conda` 找不到，先 source 已安装 Conda 的 conda.sh；也可以设置 `CONDA_SH=/实际路径/etc/profile.d/conda.sh`。不需要 uv，不需要新建虚拟环境。

原来的冒烟服务应手动停止，端口 8000 和目标 GPU 应空闲。控制器不会接管已启动服务，也不会终止其他人的 GPU 进程。

## 3. 一次性冻结真实数据

在 `evalscope-client` 环境中运行：

```bash
python "$REAL/prepare_data.py" \
  --source modelscope \
  --out "$ROOT/data/real_v5"
```

ModelScope 数据源使用现有 EvalScope 声明的 `AI-ModelScope/MATH-500`、`opencompass/humaneval`。已有缓存由对应库复用。源站版本不能解析成具体 commit 时不伪造 revision：manifest 记录请求版本与数据来源，以本地冻结 JSONL 的 SHA256 为实验依据。

可选择 Hugging Face（新输出目录，不覆盖已有数据）：

```bash
python "$REAL/prepare_data.py" \
  --source huggingface \
  --out "$ROOT/data/real_v5_hf"
```

该路径使用 `HuggingFaceH4/MATH-500` 与 `openai/openai_humaneval`，先解析 dataset revision，再加载 test split。网络受限时，使用本地原始 JSON/JSONL：

```bash
python "$REAL/prepare_data.py" \
  --source local \
  --math-file /实际位置/math500_original.jsonl \
  --human-file /实际位置/humaneval_original.jsonl \
  --out "$ROOT/data/real_v5_local"
```

本地文件需要原始字段：MATH 的 `problem,answer`，HumanEval 的 `task_id,prompt,canonical_solution,test,entry_point`。不是已经生成的答案文件。准备脚本检查全集条数，拒绝悄悄把子集标成全集。

生成：

```text
real_v5/
  manifest.json       # 来源、EvalScope 版本、模板、数据 SHA256
  math_500.jsonl       # 500 条，有题号、原始记录与未泄露答案的 messages
  humaneval.jsonl     # 164 条
```

参考答案和 HumanEval 测试代码只在本地 manifest/结果中保留，**不发送给模型**。

## 3.1 一个 Shell 自动跑完第一轮

数据准备完毕，且旧服务已停止后，可以直接运行：

```bash
bash "$REAL/run_pipeline.sh" \
  "$ROOT/data/real_v5" \
  "$ROOT/results/real/pipeline01"
```

它顺序完成：真实题 pilot → 选择各方法共同可用的并发 → 三种方法的 MATH-500/HumanEval 全集自然生成 → 对保存输出进行 MATH 评分。无可用并发建议时停止，不自动改精度或缓存配置。

已准备好并获准使用 Docker 时，可以同时评分 HumanEval：

```bash
SCORE_HUMAN_DOCKER=1 \
bash "$REAL/run_pipeline.sh" \
  "$ROOT/data/real_v5" \
  "$ROOT/results/real/pipeline_with_code_grade01"
```

此开关不会自动拉取镜像。请预先准备 `python:3.11-slim` 或通过 `DOCKER_IMAGE` 指定批准的本地镜像。未开启时 HumanEval 的输出照常全部保存，只是代码分数尚未计算。

可覆盖的管线起始参数：`PILOT_CONCURRENCY='8'`（当前 3090 首选；保守复核可设 `'4 8'`）、`PILOT_SIZE=32`、`MAXSEQ=16`、`MAXLEN=32768`、`APC=0`、`GPU=0`、`KV_BYTES=...`。默认不设置 `MATH_MAX_TOKENS/HUMAN_MAX_TOKENS`；只有调试时显式设置这两个环境变量才会重新引入请求级上限。pipeline 不覆盖已有结果目录；中断后按下面的分阶段命令加 `--resume` 继续。

以下分阶段命令与一键管线是两种操作方式，不需要把同一套实验重复运行。

## 4. pilot：用真实题测并发，不猜测“3090 最大并发”

默认用每个数据集固定 32 题、并发 2/4/8，三种方法共 18 个点。

```bash
bash "$REAL/run_real.sh" \
  --stage pilot \
  --data-dir "$ROOT/data/real_v5" \
  --run-dir "$ROOT/results/real/pilot01" \
  --concurrency 2 4 8 \
  --pilot-size 32 \
  --apc 0 \
  --continue-on-error
```

加 `--dry-run` 只预览计划，不启动模型。要先检查新协议，使用新的目录，缩小成：

```bash
bash "$REAL/run_real.sh" \
  --stage pilot \
  --data-dir "$ROOT/data/real_v5" \
  --run-dir "$ROOT/results/real/check01" \
  --datasets math_500 --methods fp16 --concurrency 2 --pilot-size 8
```

pilot 会记录请求/s、输出 token/s、TTFT/TPOT/端到端延迟、输出长度、自然停止/截断/错误、原生缓存指标与 GPU 采样。

`recommendations.json` 按数据集和 APC 状态给出：
- 每个方法在测试过的可用并发中，成功请求/s 最大的选择。
- 所有方法共同通过的并发档位，用于同并发对比。默认 full 使用共同档位，而不是把不同方法各自最优并发混为同并发结果。

默认 pilot 候选要求：正式请求齐全且成功、原生 usage 与 finish reason 齐全，并满足用户显式设置的 SLO/抢占条件。`finish_reason=length` 在默认无请求级上限时视为模型/工作负载的长度或上下文边界结果，**不会用于淘汰并发档位**。只有显式设置请求级 `max_tokens` 并同时设置 `--max-pilot-cap-fraction` 时，才将其作为筛查条件。

还可以指定实际公司的延迟条件：`--max-p95-ttft-ms`、`--max-p95-tpot-ms`。没有填写条件时，本包不会将结果称为“满足公司 SLO”。抢占次数默认只记录；`--max-preemptions 0` 可增加零抢占筛查条件，但必须有对应原生计数器才能判定。

若并发 8 仍有明确收益，可在**新目录**扩展为 `--concurrency 2 4 8 16 --pilot-size 32`。max-seqs 默认 16；并发 32 需要显式设置 `--max-seqs 32`，并作为新引擎配置重新对比。并发与显存并非简单线性对应，不能用 nvidia-smi 已分配显存或启动日志里的理论 Maximum concurrency 直接代替实测。

遇到失败时保存证据；开启 continue-on-error 后，跳过同组更高并发，继续其他方法/数据集。不会为了跑完而自动改位宽、上下文窗口、显存预算或 offload。pilot 建议是小样本经验结果，正式长尾请求仍可能失败，需要以新目录显式降低并发再测。

## 5. full：对两份全集自然生成

按 pilot 选出的**共同并发**：

```bash
bash "$REAL/run_real.sh" \
  --stage full \
  --data-dir "$ROOT/data/real_v5" \
  --recommendations "$ROOT/results/real/pilot01/recommendations.json" \
  --selection common \
  --apc 0 \
  --run-dir "$ROOT/results/real/full01"
```

检查模型、数据、硬件、包、源码、输出边界策略、采样与缓存预算是否与 pilot 一致；不一致时拒绝套用建议。没有可用建议时也不会擅自选并发。

也可以明确固定同一并发（必须是你已经验证过的档位）：

```bash
bash "$REAL/run_real.sh" \
  --stage full \
  --data-dir "$ROOT/data/real_v5" \
  --concurrency 4 \
  --apc 0 \
  --run-dir "$ROOT/results/real/full_C4"
```

每种方法：MATH-500 500 次生成，HumanEval 164 次生成。默认一题一次，一轮三种方法共 1992 次正式生成。输出长度自然变化，不重写为固定 256/512。

每点单独预热，不计入正式统计；预热使用真实题目，但添加不匹配的 leading system 标识，避免预载正式前缀。预热和正式请求采用相同的“无请求级 max_tokens、由上下文窗口兜底”策略；完整预热输出另外保存。默认 2 个预热请求并不保证所有动态批形状都已编译，可增加 `--warmup-requests`。因此正式比较要查看首批延迟并对关键点复测。

`finish_reason=length` 并不使测量结果消失，状态仍可为测量完成；`length_limit_terminations` 单列。默认协议下它通常意味着剩余上下文/平台长度边界，而不是人为请求上限。请求失败/缺失、生成参数被覆盖、输出无法导出不会标为完成。没有 reasoning parser 时，Qwen3 thinking 输出应以闭合 `</think>` 后的内容供评分；没有闭合且没有服务端 reasoning channel 时保留全文，并将“最终答案未提取”单列，而不是把未完成推理中的一个答案当最终答案。

## 6. Prefix cache：使用 vLLM 原生统计与资源参数

### EvalScope 不能设置独立 prefix-cache 显存配额

本包使用外部 vLLM HTTP 服务：
- `--kv-cache-memory-bytes` 是 **vLLM 每 GPU 的整个 KV 池预算**；本控制器的对应入口是 `--kv-bytes`。
- `--enable-prefix-caching` 控制 vLLM 是否复用前缀；入口是 `--apc 0/1`。
- EvalScope `--prefix-length` 是 random 数据集共享输入前缀的 token 数，不是缓存池大小；本真实题 JSONL 流程不使用它。
- EvalScope `dataset_args.prefix_file` 是输入文本注入，也不是显存配额；完整 JSON request body 不走该文本长度转换路径。

例如比较相同 6 GiB **总 KV 池**预算：

```bash
bash "$REAL/run_real.sh" \
  --stage full \
  --data-dir "$ROOT/data/real_v5" \
  --concurrency 4 --apc 0 1 \
  --kv-bytes 6442450944 \
  --run-dir "$ROOT/results/real/cache_6GiB_C4"
```

6 GiB 和 C=4 仅为示例，不保证模型/后端能够启动或运行。正式使用前可用相同预算先 pilot。显式指定 kv-bytes 后，KV 池不再由 gpu-util 自动推算；仍同时记录这两个请求参数及真实启动日志，避免混淆模型/工作空间与缓存池显存。

本轮 APC=1 是从未预载正式题目的状态跑真实题目，过程中允许自然复用。不是强制热缓存；不保证命中率高。不同题目共有指令很短，特别是缓存块较大时可能几乎没有可复用块。低命中率是工作负载结果，不应为了图好看去修改 benchmark。

### 原生统计，不在客户端推断命中

保留完整 vLLM 日志、每点的 `server.measurement.log`、`metrics.before.txt`、`metrics.after.txt` 和周期采样。提取原生日志 `Prefix cache hit rate`、原生查询/命中 Counter、`cache_config_info` labels 等字段。

**没有从输入字符串相似度估算命中，也不重新生成一个“自定义命中率”。** CSV 中 `prefix_native_last_percent` 仅是本点日志最后报告的原生百分比，`prefix_native_last_is_run_aggregate=false` 明确说明它不是整个实验窗口的平均值。日志窗口语义依当前 fork 而定，不对多个百分比取算术平均。以后需要严格窗口汇总，应在 Prometheus 中对 vLLM 原生 Counter 使用一致窗口；本包默认只留原始证据，不混入另一套口径。

### 日后确实需要共享长前缀

本控制器提供可选 `--shared-prefix-file /实际文本.txt`。这是**控制器自己的输入构造选项**，不是 EvalScope 的缓存容量开关。它会增加 system 消息，保存文件路径、SHA256 与本地 tokenizer 统计的 token 数，结果标记 `augmented-system-prefix`。不要把这种修改后的任务分数与原版 0-shot benchmark 分数混在一起。默认不用。

## 7. 对同一份输出评分：不再生成一次

生成结束后，先确保没有另一轮性能测量正在使用同一主机。评分阶段与推理时延分开；准确率与性能来自同一批已保存输出。

MATH-500 使用当前 EvalScope 的答案提取与 `Accuracy(numeric=True)` 规则：

```bash
conda activate evalscope-client
python "$REAL/score_saved.py" \
  --suite "$ROOT/results/real/full01" \
  --datasets math_500
```

HumanEval 必须执行单元测试才能得到 pass@1；仅保存文本和请求成功率不等于 pass@1。本包不在宿主执行生成代码。需要已批准的 Docker/隔离环境，示例镜像为官方 Python 3.11：

```bash
# 只在你获准使用 Docker 且镜像尚未准备时执行；不在压测过程中拉镜像。
docker pull python:3.11-slim

python "$REAL/score_saved.py" \
  --suite "$ROOT/results/real/full01" \
  --datasets humaneval \
  --human-sandbox docker \
  --docker-image python:3.11-slim
```

HumanEval：调用现有 EvalScope `_postprocess`，拼接原始 prompt、生成代码和原始测试；在禁网、无宿主目录挂载、只读根文件系统、非 root 用户、限 CPU/内存/进程数的容器中执行。记录具体 image ID，每题代码执行超时默认 4 秒。Docker 不是绝对安全保证，公司有更强沙箱时应优先使用公司方案。没有 Docker 时保留输出，分数记为未评分，绝不回退到宿主 exec。

MATH 符号匹配超时、Docker 基础设施错误记为 unknown；有未评分任务时不伪造完整准确率。截断样本保留并按现有输出评分。生成请求失败与判题失败分别记录。一题一次时报告 pass@1，不把三次实验重复称为 pass@3。

评分结果写入原点目录，并更新 summary.csv。已评分的点默认跳过；添加 `--force` 会备份旧分数后对同一份输出重新评分，仍不会调用模型。

## 8. 保存哪些内容

```text
full01/
  suite.json                         # 配置、代码/包/模型/数据标识和计划
  summary.csv / summary.json
  servers/.../
    launch.json / launcher.log       # 完整启动与运行日志
    server/.../command.txt           # 原 serve.sh 写下的真实 vLLM 参数
    server/.../environment.txt
    server/.../server.log
  points/r01_fp16_math_500_apc0_C4/attempt001/
    state.json
    measure.manifest.json            # 每题 ID、参考答案、原始记录、预期请求参数
    measure.requests.jsonl           # 实际送入 EvalScope 的 JSON 请求列表
    measured/.../benchmark_data.db   # EvalScope 原始结果库与报告
    requests.jsonl                   # 逐请求全文、reasoning、最终答案、usage、结束原因、时延
    requests.csv                     # 便于查看的轻量指标表
    raw_responses/*.json.gz          # 原始完整响应块，包括所有输出与原始 usage
    stats.json / output_export.json
    server_capacity.json             # 原生日志 KV token 容量、可用 KV 内存及证据行
    server.measurement.log
    native_prefix_stats.json
    metrics.before.txt / metrics.after.txt
    metrics.samples/*.prom.gz
    telemetry.jsonl                  # GPU/原生服务状态采样
    warmup/ + warmup_export/          # 与正式测量分开的预热数据和完整输出
    scores.jsonl                     # 离线评分后才生成
    scores.summary.json
    grading_protocol.json
```

结构化字段区分请求配置与原生日志值：
- 配置：方法、KV dtype、block_size、APC 开关、kv_cache_memory_bytes_requested、gpu_memory_utilization_requested、引擎参数、实际客户端并发、`request_max_tokens`（默认 unset）、服务端 `max_model_len`、输出边界策略和采样参数。
- 日志：`GPU KV cache size` token 容量、`Available KV cache memory`（可用预算，不伪称精确分配字节）、理论 Maximum concurrency 的原始行；字段未出现则留空。
- 原生 CacheConfig 信息保存在 stats.json 的 native_cache_config_labels；完整原始指标保留 HELP 与标签。
- `independent_prefix_cache_bytes=null` 表示没有独立 prefix 配额；`evalscope_random_prefix_length=null` 表示当前真实请求没有使用 random 前缀参数。null 不是 0 字节或 0% 命中率。
- TTFT/TPOT/E2E avg/P50/P95/P99、完整输出 tokens 分布、自然停止/截断/错误、请求/s、输出 token/s。原始请求间隔保留，但不把流式块间隔称作纯 GPU decode 时间。
- 输入和完整输出，包括 reasoning，原生 usage 可用时优先标记 server_usage；缺失时明确标记 EvalScope fallback，不冒充服务端计数。
- 保存完整原始响应而不只有评分后的答案；不会重新 tokenize 最终答案来替代完整生成工作量。

没有为逐请求精确拆分 GPU kernel 时间，也没有将并发请求延迟相加冒充总 GPU 时间。需要时后续再做 profiling。

## 9. 续跑、停止、失败检查

相同命令增加 `--resume`：已完成点跳过，失败点创建新 attempt，不覆盖旧输出。参数、数据、代码、模型、包或 GPU 标识变化时拒绝混入旧目录。

```bash
# 与最初 full01 命令保持一致
bash "$REAL/run_real.sh" \
  --stage full --data-dir "$ROOT/data/real_v5" \
  --recommendations "$ROOT/results/real/pilot01/recommendations.json" \
  --selection common --apc 0 \
  --run-dir "$ROOT/results/real/full01" --resume
```

Ctrl-C 或向 `controller.pid` 中的进程发送 TERM，会清理本控制器启动的服务；不使用全局 pkill/GPU reset。中断前 EvalScope 已写入的 DB 保留。需要将失败 attempt 中的已完成响应重新导出为明文：

```bash
python "$REAL/export_saved.py" \
  "$ROOT/results/real/full01/points/实际点名/attempt001"
```

不要把失败点剩余未发出的题目默默当成成功；新的 attempt 会重新跑完整该点，两个 attempt 分开保存。

## 10. 第一份报告怎么读

先看数据/请求是否齐全、EOS/stop 自然结束比例、`finish_reason=length` 的长度/上下文边界终止比例，再看质量与时延。相同任务/采样/并发的结果放在一张表；每种方法各自较优并发的结果另做部署容量表。

比吞吐时同时看请求/s、输出 token/s、任务正确率，以及离线评分后得到的正确任务/s。模型生成更短却答错，不算任务效率提升。先收集这些结果，再决定要不要展开层级量化、内核或前缀管理消融。
