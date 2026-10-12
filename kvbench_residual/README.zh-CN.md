# Anchor–Residual KV：K4V4 全量实验补丁 v1

本补丁实现的是 **低比特 Anchor 全量参与 attention、按当前 query 的误差界选择读取低位 Residual**。它增加独立的 `kvbench_residual/` 包，只依赖已经使用的 `kvbench_longcontext` V3 和原有 KVarN 服务环境；不要求先应用之前的 AFA 补丁。

该版本是待用户 GPU 验证的研究原型。交付端只完成了 CPU 数学、源码变换、隔离和调度检查，没有 RTX 3090 的实测速度或新方法任务分数。脚本先在服务器上执行 GPU 正确性检查，成功后才开展六组正式全量评测。微基准是否提速不决定保留哪些策略。

## 1. 这次实现到哪一步

| 内容 | 实际实现与范围 |
|---|---|
| 表示 | 将原 K4V4 的每个 INT4 码 `c` 写为 `4h+l`，`h,l` 各 2 bit；使用居中的 Anchor `4h+1.5` 与 Residual `l−1.5` |
| 物理布局 | 一个 K/V 字段的 high2 字节在前半部、low2 在后半部，支持独立读取；总码字节和量化参数大小不变 |
| 写入与全读路径 | 修改量化后的 pack、torch unpack 及所有相关 Triton packed-cache readers；读取全部 Residual 时先恢复同一 INT4 码 |
| 选择范围 | 量化 decode chunk；所有 token 的 Anchor 都进入最终 attention，FP16 sink/tail 不降精度 |
| K 选择 | 用已有 scale 与当前 query 计算 logit 残差上界，用本 split 已处理前缀构造分母下界，决定是否加载 low2 |
| V 选择 | 在本 chunk 的 K 决策之后，计算 V 修正的输出影响上界，决定是否加载 low2 |
| GQA | 同一 KV head 的全部真实 query heads 都允许跳过时才不读；统计按读取并集计算 |
| 调度 | 固定六组、各一次、六任务全部合格样本；复用旧 V3 冻结数据与原来的服务端/客户端环境 |
| GPU 统计 | 正式推理不加入每 token 统计；独立微基准用额外计数版本观察实际分支，输出的是逻辑字节估计 |
| 可选 DRAM 实测 | 提供独立 Nsight Compute 重放入口；缺少 ncu 或计数器权限时明确记 NA，不影响正式评分 |
| 兼容范围 | 当前全量只支持 Qwen3-4B、32 query heads / 8 KV heads / head_dim 128、`kvarn_k4v4_g128`；不声称已经实现 K4V2 全量 |

prefill、大批多 query materialization 和专用 verify 读取完整位平面。标准单 token decode 及共用该内核的 indirect verify 使用选择读取。完整数学定义、实现路径和误差边界见 `TECH_CONTRACT.md`。

## 2. 六组分别测什么

默认总预算参数 `--epsilon 0.02`。它表示相对保守 Value 范数上界 `V_ref` 的 attention 输出误差预算；**不是 2% 任务准确率损失上限，不是输出向量相对误差，也不是浮点形式化认证**。

理论界首先针对旋转空间的 attention 核输出。映射回模型空间时还需考虑实际 FP16 逆旋转矩阵的算子范数和该运算的舍入。每 tile 的完整码和量化参数字节不变；vLLM 最终分配的缓存块数仍以实际运行记录为准，临时张量与图捕获可能影响容量。

| 方法目录 | 配置 | 目的 |
|---|---|---|
| `ar_k4v4_parent` | 原 packed 布局和原重建算式 | 同批次净效果基线 |
| `ar_k4v4_repacked_full` | 新 high2/low2 布局，全部残差都读 | 测拆分布局、写入与读回本身的成本；与选择策略共用同一完整量化表示 |
| `ar_k4v4_anchor` | 量化 token 只用 K/V Anchor | 无预算限制的激进消融：观察最大跳读与质量代价 |
| `ar_k4v4_bound_k` | K 预算 0.02，V 残差全读 | 单独研究 K 选择 |
| `ar_k4v4_bound_v` | K 残差全读，V 预算 0.02 | 单独研究 V 选择 |
| `ar_k4v4_bound_kv` | K、V 各预算 0.01 | 研究联合选择的净效果和精度 |

六组共享空 mask/空 split 的数值保护，因此新 `parent` 与旧九方法源文件并非字节完全相同；它保持原 packed 布局和原仿射算式。主比较以这次六组为准，旧九方法放入总览提供参考。该补丁不重新跑旧九方法。

## 3. 应用补丁

在用户已有项目里执行，补丁路径替换为实际下载路径：

```bash
cd /home/wanglin/kvbench-3090-cu126
PATCH=/绝对路径/kvbench_residual_v1_full_20261010.patch

git apply --check "$PATCH"
git apply "$PATCH"
```

`git apply --check` 只校验能否应用。不要再叠加旧 AFA 补丁。本补丁不直接改安装环境的 vLLM Python 文件。

## 4. 一条全量命令与后台运行

已经上传的成功 V3 批次是：

```text
/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json
```

它于 2026-10-10 12:15:23（北京时间）以 `exit_code=0` 完成九方法；每种方法有 1,147 个合格样本。本次复用这份冻结数据，默认六组共 **6,882 条正式请求**。实际数量仍以读取到的 `data_coverage.csv` 为准。不要使用 19:52 开始、数据准备失败的旧批次。

先查看计划：

```bash
cd /home/wanglin/kvbench-3090-cu126
source /home/wanglin/miniconda3/etc/profile.d/conda.sh

AFTER_SUITE=/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json

bash kvbench_residual/run_residual_full.sh \
  --after-suite "$AFTER_SUITE" \
  --epsilon 0.02 \
  --plan
```

`--plan` 读取该 suite 并打印六组与协议，不创建结果目录、不初始化服务环境、不访问 GPU。

直接跑完全部新方法：

```bash
bash kvbench_residual/run_residual_full.sh \
  --after-suite "$AFTER_SUITE" \
  --epsilon 0.02
```

需要关闭终端后继续运行时，用下面完整脚本：

```bash
cd /home/wanglin/kvbench-3090-cu126
source /home/wanglin/miniconda3/etc/profile.d/conda.sh

AFTER_SUITE=/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json
AR_BATCH=/home/wanglin/kvbench-3090-cu126/results/residual_v1/full_$(date +%Y%m%d_%H%M%S)_$$

mkdir -p "$(dirname "$AR_BATCH")"
nohup bash kvbench_residual/run_residual_full.sh \
  --after-suite "$AFTER_SUITE" \
  --epsilon 0.02 \
  --out "$AR_BATCH" \
  >"${AR_BATCH}.driver.log" 2>&1 &

AR_PID=$!
printf '调度 PID=%s\n新批次=%s\n主日志=%s\n' \
  "$AR_PID" "$AR_BATCH" "${AR_BATCH}.driver.log"
```

查看进度：

```bash
tail -f "${AR_BATCH}.driver.log"
```

这里生成的是 `.driver.log`，位于新批次目录旁边；不会自动生成 `.console`。在 VS Code 打开 `results/residual_v1/` 并刷新即可找到。批次内部另有 `suite.log` 与各步骤日志。

如果你指定的是还在运行的新 V3 批次，脚本会等到该批次明确写入 `finished`、`exit_code=0` 且九方法全为 `completed` 后才继续；GPU 在两种方法之间短暂空闲不会被当成旧批次结束。已成功完成的批次会立即进入冻结检查。旧批次失败则停止新批次，不启动 GPU。

### 自动完成的顺序

1. 核对旧批次身份、数据 manifest、逐任务文件与 case 顺序、V3 代码哈希。
2. 自动激活 `evalscope-client` 和 `kvarn-serve`，服务环境自动加载原 `KIT/env.sh`。
3. 检查模型结构；为六策略各创建独立 vLLM runtime，核对全部六个关键 writer/reader 源文件。
4. 检查指定 GPU 与端口空闲，执行 GPU 数值/表示/路径验证。
5. 执行独立微基准。数值失败停止；微基准没有提速也不会自动删除实验组。
6. 逐组启动自己的服务器，等待真实模型/缓存配置与内核来源证据，再启动客户端全量请求。
7. 每组请求结束后停止本批次自己的服务器，执行评分和汇总，再跑下一组。
8. 生成旧九方法加新六组总览，以及固定六种对照的差值表。

不需要手动另开一个服务端。数据直接使用现有冻结文件，不调用 Hugging Face 下载；客户端未安装 PyTorch 的 tokenizer-only 提示不是这里的 GPU 环境错误。若只有仓库摘要而没有原机器的 `data/*.jsonl`，不能伪造冻结输入重跑，需要在保留完整原数据的机器执行。

## 5. 保持一致的正式评测协议

| 项目 | 本补丁使用的值 |
|---|---|
| 数据 | 同一个成功 V3 批次的六任务全部合格、未裁剪样本 |
| 并发/重复 | C8；每组一次 |
| thinking | `true` |
| 采样 | temperature 0.6；top_p 0.95；top_k 20；min_p 0 |
| 正式输出上限字段 | `max_tokens: null`，不另设 max_completion_tokens 或其他独立输出 token cap |
| 自然结束约束 | EOS/stop 或模型剩余上下文；max_model_len=32768 |
| 正式请求超时 | 0，无请求时间限制 |
| seed | request_seed=20261009；server_seed=20260917 |
| 服务参数 | max_seqs=16，batch_tokens=1024，gpu_util=0.85，APC 关闭 |
| CUDA Graph | 保持 V3 默认；动态选择发生在 Triton GPU 分支，未增加每步 host 同步 |
| length 响应 | 完整收到的 `finish_reason=length` 仍评分；不得因截断丢题 |
| 空答案/故障 | 真正空 final 按评分规则得实际 0；请求或评分基础设施失败保留 NA，不能当成正确率 0 |

V3 的健康检查与预热保留原来的短输出 cap，均不计入正式性能和质量结果。服务启动等待默认 1,800 秒，约束的是启动/编译，不是正式生成时间。

## 6. 结果在哪里，按什么顺序看

不指定 `--out` 时默认写到：

```text
/home/wanglin/kvbench-3090-cu126/results/residual_v1/full_<时间>_<PID>/
```

| 文件 | 看什么、代表什么 |
|---|---|
| `suite_status.md` / `suite.json` | 六组是否开始、完成、失败、评分完整；`implementation_verified` 是否有对应内核来源证据 |
| `data_coverage.csv` | 每任务原始、合格、实际选中、排除、裁剪数量；正式题数以 selected_count 为准 |
| `validation/validation.json` | GPU 是否真正执行并通过；`gpu_executed=false` 或 CPU-only 不能放行正式全量 |
| `microbench/benchmark.json`、CSV/读出 | 固定形状 attention 时延与独立诊断计数；不是正式端到端 TPOT |
| `servers/<方法>/launch.json` | 生效模型、KV dtype、缓存容量、epsilon、源代码哈希、私有 runtime、环境 |
| `servers/<方法>/server.log` | vLLM 启动、编译、OOM、运行异常 |
| `servers/<方法>/dispatch_summary.json` | 首次 Python 导入/launch/driver 回执；包含 warmup/图捕获，不是每个正式 token 的计数 |
| `runs/<方法>/r01_<任务>_C8/requests.jsonl` | **正式**请求响应与完成状态；V3 的 `responses.jsonl` 用于预热，不能拿它当正式全量答案 |
| 同目录 `scores.jsonl` | 每个正式样本得分、空答案、错误状态 |
| 同目录 `performance.summary.json` | 有效响应、TPOT、TTFT、E2E、吞吐、输出长度、截断、抢占 |
| 同目录 `scores.summary.json` | 任务得分、已评分数、是否完整；长任务得分不都等于二元准确率 |
| `comparison/overview.md` / CSV | 已有九方法与新六组保存结果的总览 |
| `comparison/residual_effects.md` / CSV / JSON | 六个预先固定比较 × 六任务，共 36 行差值；缺证据或协议不一致的主差值记 NA |
| `logs/` | preflight、GPU 验证、微基准、每组启动/请求/评分/报告的逐步日志 |

### 固定对照怎样读

- `main_net`：`bound_kv` 对新 `parent`，判断整个方案最终是否有净收益。
- `representation`：`repacked_full` 对 `parent`，判断拆分布局/重建本身的成本。
- `anchor_limit`：`anchor` 对 `repacked_full`，观察只读 Anchor 的极端质量代价。
- `key_selection` / `value_selection`：分别看只选 K 或只选 V。
- `joint_selection`：`bound_kv` 对 `repacked_full`，控制完整量化表示后看按需选择的收益。

得分变化 `score_delta_pp` 是 0–100 任务分的直接差，正数表示提高。`tpot_improvement_percent=100×(1−新/对照)`，正数表示每 token 时间下降；`output_tps_gain_percent=100×(新/对照−1)`，正数表示吞吐提高。E2E 和输出长度变化使用 `100×(新/对照−1)`，负数表示减少。输出更短本身不能证明内核更快。

F1、ROUGE-L 与 retrieval 得分各有任务含义，不把不同任务混成一项总体正确率。每组一次能做首轮筛选，但不能据此得到采样稳定性或统计非劣结论。

## 7. 可选：在 GPU 空闲时采集真实 DRAM 计数

正式六组结束后再运行。下面命令自动激活原服务环境，默认对 `parent`、`repacked_full`、`bound_kv` 各测固定 B8、L8192 的一个 attention 路径：

```bash
bash kvbench_residual/profile_dram.sh \
  --suite "$AR_BATCH/suite.json"
```

需要修改独立重放形状时：

```bash
bash kvbench_residual/profile_dram.sh \
  --suite "$AR_BATCH/suite.json" \
  --batch 8 --length 16384 --mode split --block-n 32 --splits 32
```

它采集 `dram__bytes_read.sum`、`dram__bytes_write.sum`、`gpu__time_duration.sum`，保留每组 `ncu.raw.csv`、`profiler.log`、重放 provenance 和汇总 `dram_profile.json/csv/md`。默认目录是该 AR 批次下的 `profiles/dram_<时间>_<PID>/`。

这是**真实 GPU DRAM 性能计数器在独立重放中的结果**，不等于整次正式生成的流量。`cache-control=all` 使 profiler 每次 kernel replay 前清缓存；`clock-control=none` 不请求修改 GPU 时钟。profiled kernel 时长受到 profiler 重放影响，不能替代常规微基准或正式 TPOT。计数范围不包含量化写入、rotation、整层或 HTTP。

缺 ncu、计数器权限不足或指标不支持时写 NA 与具体原因，不会自动安装工具、修改驱动权限，也不会改动已完成评分。RTX 3090 使用 GDDR6X；报告统一称 DRAM/显存字节，不把这台卡称为 HBM GPU。

官方 CLI 说明：<https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>。

## 8. 失败时如何判断

| 状态 | 意义与后续 |
|---|---|
| `waiting_for_v3` | 旧批次尚未明确成功结束；正常排队，没有新 GPU 工作 |
| upstream `exit_code != 0` | 指定了失败旧批次，先看其 `suite_status.md`；不要把空闲 GPU 当成实验完成 |
| source pin mismatch | 安装的 writer/reader 与审查版本不同；私有 runtime 未完成，不会猜测兼容并继续 |
| GPU validation failed | 新核尚不能运行正式全量；保留 validation 和 gpu_validate.log，不能跳过这个门槛 |
| `startup_failed` | 查看对应 launcher.log 与 server.log；数据和其他已完成组仍保留 |
| `incomplete` / `run_failed` | 正式请求或评分未全数完成；汇总不能当成完整实验 |
| `implementation_unverified` | 输出/分数保留，但来源或 decode launch 证据不足，不计算该行主差值 |
| `cleanup_failed` | 本批次自己的服务没有可靠退出；停止后续组，避免重叠启动 |
| profiler `unavailable` | 独立 DRAM 采集不可用；正式结果照常有效，不能以逻辑字节冒充实测 |

每次创建新批次目录；脚本不覆盖、不自动恢复、不自动重试未完成正式样本。若要主动停止后台 AR 批次，可向保存的 `AR_PID` 发送 `SIGINT`；它只清理本次创建的服务组，不停止旧实验服务。之后重跑使用新的输出目录，已完成结果仍然保留。
