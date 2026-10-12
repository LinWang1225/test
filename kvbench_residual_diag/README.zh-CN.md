# Anchor + Residual：两个真实 trace 诊断实验

版本：2026-10-12 / v1.0.0。

这是**诊断补丁**，不是加速版 v2，也不是重新执行六组正式任务评分。仅新增 `kvbench_residual_diag/`，依赖仓库已经存在的 `kvbench_residual/`、`kvbench_longcontext/` 和原 `kvarn-serve` 环境。

## 两个实验及其边界

| 实验 | 数据/执行方式 | 输出 | 不代表什么 |
|---|---|---|---|
| 实验一：真实跳读诊断 | 在独立 vLLM Parent 服务采集真正旋转后的 Query、原生 K4V4 缓存和 GPU 输出；离线用原 AR Triton 内核重放、计数 | GPU K/V skipped chunks、逻辑字节；CPU 误差界/真实误差比、预算比、GQA 阻断情况 | 不是随机 KV；也不是 bound 策略自然生成全程的跳读率。逻辑字节不是 DRAM 实测 |
| 实验二：Oracle 参考曲线 | 同一批真实 Q/KV，float64 反事实 attention；共享 GQA 读取掩码 | K-only/V-only/KV 的误差—读取预算曲线、可行选择掩码、V-only 有效下界、小规模精确枚举 | 不测任务准确率、TPOT、显存事务；离线贪心不保证全局最优 |

两个实验共用一次采集。实际驱动传入 `[B*Hq,D]` 的持久化旋转 Query 缓冲区；采集器读取后仅恢复为 `[1,Hq,D]` 存储，不把测试夹具的三维输入形状强加给实际服务。所有 token 的 Anchor 始终进入 attention；FP16 sink/tail 不降低精度。默认在 Parent 轨迹上做同输入反事实比较，排除了不同策略自然生成轨迹不一致的干扰；但仍需后续闭环实验验证真实生成效果。

**更正 Oracle 的术语：** `offline_greedy` 是用完整 Residual 做一步前瞻的可行参考；它的误差是最优误差的上界，而不是“最优误差本身”。只有 `exhaustive_exact_small` 的相应小问题才精确求解。V-only 另给有效但可能很松的误差下界。贪心效果不好、下界为零时，不能据此否定该方向。

## 1. 应用与预检

在已有项目根目录执行，PATCH 替换为补丁实际存放路径：

```bash
cd /home/wanglin/kvbench-3090-cu126
PATCH=/绝对路径/kvbench_residual_two_experiments_20261012.patch

git apply --check "$PATCH"
git apply "$PATCH"
```

源码 ZIP 与 patch 是**替代交付形式**，不需要叠加应用。ZIP 中只有新增包；选择 ZIP 时把 `kvbench_residual_diag/` 解压到项目根目录。

查看计划，不启动服务器或 GPU 推理，也不创建结果目录：

```bash
bash kvbench_residual_diag/run_two_experiments.sh --plan
```

默认读取以下成功旧批次：

```text
results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json
```

从这个 `suite.json` 读取原模型、冻结数据目录、KIT、环境、采样、thinking 和服务窗口。不要传 `residual_v1` 的 suite。只有 GitHub 摘要、缺少原机器冻结 `data/*.jsonl` 时会停止，不下载替代数据、不伪造输入。`checked_data` 检查全部任务文件哈希、case ID 顺序与预算信息。

默认 Miniconda 根目录为 `/home/wanglin/miniconda3`；变化时设置 `CONDA_BASE`。默认 BLAS 线程为 4，可通过 `AR_DIAG_BLAS_THREADS` 修改。

## 2. 默认一条命令执行两个实验

```bash
cd /home/wanglin/kvbench-3090-cu126
bash kvbench_residual_diag/run_two_experiments.sh
```

默认设置：

| 项目 | 值 |
|---|---|
| 采集任务 | qasper、hotpotqa、passage_retrieval_en |
| 每任务样本数 | 2；按冻结输入长度排序后均匀抽样，不改原 prompt |
| 总请求 | 6 个诊断请求；另有 1 个明确不采集的预热请求 |
| layer cache 采样 | 按本地模型层数取约 4 个 cache ordinal；对 36 层模型为 0、12、24、35 |
| decode 调用位置 | 0、15、63；是采集器观察的单 token decode 调用序号，不声称绝对生成 token 位置 |
| trace 上限 | 默认计划至多 72 个；EOS 早结束时少于此数，记录缺失，不强制续写 |
| 采集环境 | 独立 C1、eager、APC 关闭、真实原生 K4V4；端口 18080；GPU 0 |
| 诊断生成 cap | 96 tokens；只为采集短 decode 前缀。**不是正式任务输出上限，也不评分** |
| thinking / sampling | 复用旧 suite，不改 greedy，不人为强制输出到 96 |
| 总 trace guard | 至多 256 个、8 GiB 数组字节；上限不含私有 runtime/logs |
| 实验一 | 全部采集 trace，全部 8 KV heads，epsilon=0.02，bound_k / bound_v / bound_kv，原采集 BLOCK_N/splits |
| 实验二 | 最多 12 个按采集顺序均匀分布的 trace，KV heads 0/3/7；默认 V-only、KV；预算 0/10/25/50/75/100% |
| 输出隔离 | `results/residual_diag/run_<时间>_<PID>/`；存在即拒绝覆盖 |

采集使用 `--enforce-eager` 是为了 Python hook 能看见每次实际 decode。C1 和短前缀是**明确的诊断协议变化**，因此不会输出可与旧 C8 结果比较的 TPOT/吞吐。

`cache_ordinal` 是同一请求 forward 过程中观察到的不同缓存地址的顺序，不伪称已经验证了模型的语义层名。正式论文图表应标为 cache ordinal，或后续补充后端 layer-name 映射。

流程顺序：CPU 单元测试和原 v1 桥接检查 → 建立私有 Parent runtime → 原生服务配置检查 → 不采集的 eager 预热和来源检查 → arming 后采集实际请求 → **彻底停止本次服务器** → GPU 重放诊断 → CPU Oracle。

GPU 正忙、端口占用、source pin 不匹配、数值校验失败均停止；不停止其他实验进程。只清理本次创建的服务进程组。无需手动另开服务端。

## 3. 后台执行与日志路径

```bash
cd /home/wanglin/kvbench-3090-cu126
DIAG="$PWD/results/residual_diag/run_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$(dirname "$DIAG")"

nohup bash kvbench_residual_diag/run_two_experiments.sh \
  --out "$DIAG" \
  >"${DIAG}.driver.log" 2>&1 &

DIAG_PID=$!
printf 'PID=%s\n结果=%s\n主日志=%s\n' "$DIAG_PID" "$DIAG" "${DIAG}.driver.log"
```

```bash
tail -f "${DIAG}.driver.log"
```

主日志位于批次目录**旁边**，不是 `.console` 文件。GPU 重放和 Oracle 的逐 trace 进度分别在 `experiment1.log`、`experiment2.log`，服务端详细日志在 `server/server.log`。

## 4. 首次只做较小贯通检查

该命令仍然完成两个实验，只减少 trace 数：

```bash
bash kvbench_residual_diag/run_two_experiments.sh \
  --per-task 1 \
  --layer-ordinals 0 18 \
  --decode-calls 0 15 \
  --diagnostic-max-tokens 32 \
  --oracle-trace-limit 6
```

短输出可能在后续采集点之前结束，属于自然 EOS；查看 `capture_coverage.csv`。不因缺少晚期 trace 而重新选择更有利的题目。完全没有可用 trace 时会失败，不用合成数据替代。

## 5. 分开运行：同一 trace 不重复采集

先只采集：

```bash
bash kvbench_residual_diag/run_two_experiments.sh --capture-only
```

下面在已有 `kvarn-serve` 环境、项目根目录运行。DIAG 设置为实际已经生成的批次路径，不要填尚不存在的目录。

### 实验一：多预算 + split 保守性诊断

```bash
source /home/wanglin/miniconda3/etc/profile.d/conda.sh
conda activate kvarn-serve
source /home/wanglin/kv_evalscope_3090_cu126_v2/env.sh
cd /home/wanglin/kvbench-3090-cu126

DIAG=/绝对路径/已经完成采集的批次
CUDA_VISIBLE_DEVICES=0 python -m kvbench_residual_diag.diagnose \
  --traces "$DIAG/traces" \
  --out "$DIAG/experiment1_epsilon_split_$(date +%Y%m%d_%H%M%S)" \
  --epsilons 0.01 0.02 0.05 0.10 \
  --splits 1 32 \
  --trace-limit 12
```

`--splits 1 32` 在相同数据上检查 split 前缀下界的影响；不是部署性能结论。不传此参数时使用真实采集的 split 数。`--block-n 16/32/64` 是另一维消融，不默认大网格全扫。

只做 CPU 解释时必须显式添加 `--cpu-only`。该模式输出 `completed_cpu_only`，GPU skipped/bytes 保留 NA，不能用于声称已经在 GPU 上跳读。

GPU 校验包括：真实保存的 Parent 输出 vs float64 完整量化参考、Parent 重放 vs 保存的真实输出、选择策略 GPU 输出 vs CPU 对应策略。沿用原 v1 的经验数值门槛：整体 relative L2 和最大元素绝对误差均不超过 0.02。门槛只是实现检查，**不是任务非劣保证或浮点形式化认证**。

原始 autotune 的 BLOCK_N、warps、stages、maxnreg 均记录；重放固定 BLOCK_N/warps/stages。现有 GpuCase 不透传 maxnreg，因此它仅记录不复用；本补丁不计时，并通过原输出数值校验防止静默错误。

### 实验二：扩大 Oracle 覆盖

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m kvbench_residual_diag.oracle \
  --traces "$DIAG/traces" \
  --out "$DIAG/experiment2_all_$(date +%Y%m%d_%H%M%S)" \
  --trace-limit 0 \
  --kv-heads 0 1 2 3 4 5 6 7 \
  --modes k v kv \
  --fractions 0 0.10 0.25 0.50 0.75 1.0
```

Oracle 不需要 GPU，不加载模型。离线贪心计算量约随 chunk 数平方增长；先看默认子集结果，再扩大所有 trace/head。不会悄悄裁短 KV 来声称全上下文结果。

## 6. 实验一先看这些列

`experiment1/summary.csv`：

- `gpu_executed`：是否真的调用了原 AR Triton kernel。
- `gpu_key_skipped_chunks` / `gpu_value_skipped_chunks`：真实输入重放中的 GPU 分支统计。
- `gpu_logical_code_reduction_fraction`：量化码逻辑字节减少，不含 scale、FP16 pool，也不等于 DRAM。
- `k/v_threshold_ratio_p50/p95`：待比较界项 / 当前局部预算；大于 1 表示未通过。
- `k/v_gqa_blocked_chunks`：有部分 query head 可跳，但 GQA 共享组不能整体跳过。
- `cpu_gpu_count_match`：CPU 数学预测与 GPU 计数是否一致。边界差异保留，不以 CPU 数字覆盖 GPU。
- `gpu_vs_full_*`、`gpu_vs_cpu_policy_*`：相对完整量化参考和同策略 CPU 输出的误差。

`chunk_decisions.jsonl`：每个真实 chunk 的误差界、实际 logit 残差、真实完整 attention mass、前缀质量上界、通过的 query head 数。真实 mass/Residual 只用于**离线解释**，没有被偷偷用于在线选择。

`counts_*.npy`：每个 KV head/split 的计数，列顺序见原 `kvbench_residual.kernel.STATS_FIELDS`。

## 7. 实验二的算法、预算与下界

### 可行算法

`attention`：按 attention mass/字节排序。V-only 使用完整 K 得到的 attention；K/KV 使用 Anchor logits。`random_seed_*`：默认三条固定随机排序。`offline_greedy`：读取全部 Residual 做离线一步前瞻，每次选择单位字节收益最大的 K 或 V chunk。以上均共享同一 GQA mask。

残差贡献可能互相抵消，读取更多块不必严格改善误差。算法保留预算内最好的可行前缀，实际读取比例可能小于预算上限，必须读取对应 `read_*` 字段。

`best_feasible_envelope`：在相同预算下，从上述可行候选中选择误差最小者，并保存实际选中的算法名称。这仍然不是全局最优。

### V-only 的有效误差下界

K 固定全读时，chunk b 的实际输出修正是：

```text
delta_b,h = sum_{t in chunk b} p_full(h,t) * Residual_V(t)
error_h(S) = sum_b delta_b,h - sum_{b in selected S} delta_b,h
```

三角不等式给出 `||error_h(S)|| >= max(0, ||sum_b delta_b,h|| - sum_{b in S} ||delta_b,h||)`。

在字节预算下，用分数背包放宽“可移除的残差范数总和”，得到其上界；代回得到最优误差的有效下界。允许每个 query head 各自选择只会放宽问题；取各头下界最大值，仍是共享 GQA mask 的有效下界。相对误差分母接近零时保留 NA，并另外报告绝对误差。

下界为 0 时通常只是**没有判别力**。只有下界显著高于目标误差时，才可排除这一 V-only 读取预算。

### 精确枚举

完整 GQA 组的可选动作不超过 `--exact-max-actions`（默认 14，硬上限 20）时，枚举所有子集，得到该组、该预算、最大头相对 L2 目标的精确解。K/KV 大规模问题没有实现精确下界，字段为 NA。长 trace 通常超过阈值，这是正常状态，不是假装精确求解失败。选择目标用 `max(||误差_h|| / max(||参考_h||,1e-12))`，相应 floor 与目标值保存在曲线行里；正式相对误差报告对近零参考头保留 NA，不用 floor 伪装有定义的相对误差。

### 四个比例不要混淆

`requested_fraction`：可控侧的 Residual 预算上限。

`controllable_read_fraction`：实际可控侧读取比例。

`all_residual_read_fraction`：包含另一侧固定全读的低位读取比例。

`logical_code_reduction_fraction`：相对同位平面完整读取，K/V **量化码**字节减少比例。

在等大 K/V、2+2 位、完整 chunk 情况下，V-only 即使不读任何 V Residual，也只减少 K+V 总量化码的 25%；K/V 都跳过 Residual 才达到 50%。这不是全部 attention 流量，更不是端到端速度提升百分比。

## 8. 输出目录与最少回传文件

```text
<DIAG>/
  plan.json                    # 原协议、诊断改动、采样计划、数据哈希
  status.json
  cpu_preflight.log
  diagnostic_requests.json     # 诊断请求及 native usage；无性能比较指标
  capture_coverage.csv
  server/server.log
  server/launch.json
  server/dispatch_verified.json
  traces/trace_*.json           # provenance、真实 shape、kernel hash
  traces/trace_*.npz            # 当前请求活跃页、旋转 Q、原生码、pool、真实输出
  experiment1/summary.csv
  experiment1/validation.json
  experiment1/chunk_decisions.jsonl
  experiment1/readout.md
  experiment2/curves.csv
  experiment2/aggregate.csv
  experiment2/coverage.csv
  experiment2/readout.md
  experiment2/masks/*.npz
```

先回传 `status.json`、`capture_coverage.csv`、两个实验的 `readout.md` 和 CSV、实验一 `validation.json`。排错再补日志和少量 trace。无需为了结果分析先上传全部缓存快照；快照包含真实模型中间状态，应按自己的数据管理要求保存。

## 9. 验证状态与使用限制

交付环境：39 项 CPU/模拟 API 测试通过；2 项依赖原 `kvbench_residual` 的源码桥接测试因本地无该包而跳过；Python 编译及 shell 语法检查通过。**没有运行 RTX 3090、GPU 编译或真实模型采集。** 参见 `TEST_RESULTS.txt`。

运行脚本会在原仓库自动执行全部测试，此时源码桥接测试不能因为找不到依赖而成为研究证据。私有 runtime 仍使用原 v1 的精确 SHA pin；未知版本停止，不篡改已安装 vLLM。真正 GPU 跳读结论必须等实验一的 GPU 检查通过。

不涉及模型微调、不构造拟真的假结果、不复用随机 KV 作为真实证据、不自动更改驱动权限、不修改旧结果。
