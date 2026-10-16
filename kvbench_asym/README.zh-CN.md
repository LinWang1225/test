# 等逻辑读取量的 K/V 非对称全样本实验 v1

## 1. 目的与边界

本补丁回答：在**同一个真实 Query/KV、同一组可控 chunk、完全相同的量化码读取字节数**下，把 Residual 预算偏向 K、平均分配、偏向 V，哪种注意力输出误差更小？

这是全样本**机制/敏感性实验**。每题只运行一次原生 KVarN K4V4 Parent，采集指定位置的真实 Q/KV；然后在 CPU float64 上比较所有读取掩码。选择器没有注入模型生成。

因此：

- 有真实模型激活，不使用随机 KV 代替全量数据。
- 默认 LongBench v1 **21 个子任务的全部记录**，不是旧批次六任务的 1,147 个合格样本。
- 可增加已有冻结的 MATH-500 全部 500 题、HumanEval 全部 164 题。
- 不产生非对称策略的任务准确率、HumanEval pass@1、TPOT 或吞吐。
- 等的是**选中量化码的逻辑载荷**，不是实测 DRAM 事务，也不是包含排序、QK 重算、临时张量后的总内存流量。
- 使用 Parent 轨迹作同输入反事实，不代表各策略自由生成后的轨迹分布。
- 仅支持已验证接口范围：Qwen3-4B、32 Query heads / 8 KV heads、head_dim=128、K4V4/G128。

补丁只新增 `kvbench_asym/`，不覆盖 `kvbench_residual/`、`kvbench_residual_diag/` 或旧结果。

## 2. 前置条件

以仓库提交 `5702570bb3c182bc9168c5d8434f7f95bcc7da6d` 为核对基线。

需要已有的两个包：

```text
kvbench_residual/
kvbench_residual_diag/
```

运行脚本读取已有成功的 LongContext V3 suite：

```text
results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json
```

从中复用本地模型路径、server_env、KIT、上下文窗口、采样和 GPU 服务参数。不重新下载模型/数据，不安装新 CUDA/vLLM，不停止其他服务。

源码哈希不匹配时明确失败；不要删除 pin 检查来强行运行。新包不会应用之前的源码补丁，也不会把未安装依赖静默替换成近似版本。

## 3. 应用与检查

```bash
cd /home/wanglin/kvbench-3090-cu126
PATCH="$PWD/kvbench_asym_equal_bytes_full_20261012.patch"
git apply --check "$PATCH"
git apply "$PATCH"

bash kvbench_asym/run_equal_bytes.sh --plan
```

`--plan`读取源数据数量/哈希和模型配置，不加载模型权重、不创建结果目录、不启动 GPU。实际题数以计划和 `inputs/coverage.csv` 为准。

Shell 使用现有 `/home/wanglin/miniconda3`；路径不同时可设置 `CONDA_BASE`。它自动激活 suite 指定的服务环境并 source 原 `KIT/env.sh`。BLAS 线程默认 4，可在启动前用 `AR_ASYM_BLAS_THREADS` 显式设置。

### 首次建议做贯通检查

```bash
bash kvbench_asym/run_equal_bytes.sh \
  --out results/asym_equal_bytes/pilot01 \
  --case-limit 3 \
  --layers 0 35 \
  --decode-calls 0 15 \
  --diagnostic-max-tokens 32
```

`--case-limit`是**所有任务合并后的总上限**，不是每任务抽 3 题。它只用于贯通检查，可能只涉及最先的任务；正式实验省略此参数，默认 0，表示全部记录。

## 4. 正式扩大到 LongBench v1 全部 21 子任务

```bash
cd /home/wanglin/kvbench-3090-cu126
OUT="$PWD/results/asym_equal_bytes/longbench21_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$(dirname "$OUT")"

nohup bash kvbench_asym/run_equal_bytes.sh \
  --datasets longbench_all \
  --out "$OUT" \
  >"${OUT}.driver.log" 2>&1 &

printf 'PID=%s\n结果=%s\n日志=%s\n' "$!" "$OUT" "${OUT}.driver.log"
```

不需要后台运行时直接执行同一个 `bash ...` 命令即可。

查看：

```bash
tail -f "${OUT}.driver.log"
cat "$OUT/status.json"
```

原始源路径默认使用 V3 suite 的 `settings.source`。这必须是完整 LongBench v1 `data.zip`，或者含 21 个原始 JSONL 的目录。需要改路径时加：

```bash
--longbench-source /绝对路径/data.zip
```

缺少任意一个子任务（包括 lcc、repobench-p）都会失败，**不会退回六任务并声称全量**。仅显式选择 `--datasets longbench_frozen` 才复用旧冻结清单的全部任务/全部记录；它与 LongBench-21 的覆盖范围分别标注，不可混称。

### “全量”和超长输入的明确约定

默认 `--overflow head-tail`：处理全部原始记录；超过 `max_model_len - diagnostic_max_tokens` 的输入进行首尾保留/中间移除，并记录原始长度、有效长度、截断标记和提示词哈希。

**全量记录不等于所有原文未经截断。** 这与旧六任务“不裁剪、排除超窗输入”的协议不同。该实验不是官方 LongBench 成绩复现；中文、few-shot、代码任务也不因指标不支持被删掉，因为这里只测 attention 误差。

要求不改变任意输入时使用 `--overflow error`。发现超窗会在准备阶段报错，不会暗中排除该题或冒称全部完成。扩大上下文窗口还需模型和服务真正支持，不能只改输入长度计数。

默认所有样本都进入生成，**不设置最短输入阈值**。没有可控量化历史的 trace 会保留覆盖信息，但不作为“零误差、方法成功”的样本混入非对称比较。

## 5. MATH-500 与 HumanEval

推荐另开批次，为短提示词保留更长的真实生成前缀：

```bash
bash kvbench_asym/run_equal_bytes.sh \
  --datasets math_500 humaneval \
  --diagnostic-max-tokens 1024 \
  --decode-calls 0 63 255 511 \
  --out results/asym_equal_bytes/math_humaneval01
```

默认读取 `data/real_v5/manifest.json` 和两份 JSONL，核对哈希与 500/164 的完整数量。更换文件夹使用 `--real-data /绝对路径/real_v5`。

只使用已冻结的用户 messages，不把 reference、solution、测试程序拼接到输入。**不执行模型生成的 Python 代码**，不需要 Docker/unshare 评分环境，也不声称获得 HumanEval pass@1。

1024 是本次诊断的输出上限，不强制生成 1024 tokens；自然 EOS 之前没有达到的采集位置明确记为缺失。不会为了让短题拥有 KV 历史而添加无关前缀或强制续写。

可以在同一批次覆盖三者：

```bash
bash kvbench_asym/run_equal_bytes.sh \
  --datasets longbench_all math_500 humaneval \
  --diagnostic-max-tokens 1024 \
  --decode-calls 0 63 255 511 \
  --out results/asym_equal_bytes/all_datasets01
```

注意该命令也把 LongBench 的生成预留改为 1024，可能使更多长输入被 head-tail 裁剪。分批运行更便于控制协议和阅读结果。不要在同一 GPU 上并行启动这两个模型服务。

## 6. 默认实验矩阵

| 配置项 | 默认值 |
|---|---|
| 数据 | LongBench v1 21 子任务，全部原始记录 |
| 本地模型 | V3 suite 指定的 Qwen3-4B |
| 生成轨迹 | 原生 KVarN K4V4 Parent，C1、eager、APC 关闭 |
| 采样/thinking | 复用 suite；生成前缀 cap 默认为 96 |
| 缓存层位 | 根据本地层数选 4 个 cache ordinal；36 层时 0/12/24/35 |
| decode 调用 | 0/15/63；第一个生成 token 来自 prefill，decode 0 产生第二个 token |
| KV heads | 全部 8 个；每组处理共享的 4 个真实 Query heads |
| 分析 chunk | 32 tokens；可显式选择 16/32/64 |
| 总残差预算 | rho_K+rho_V = 1.0、1.1 |
| K 分配优先级 | 0 / 0.25 / 0.5 / 0.75 / 1 |
| 选择策略 | anchor_mass、staged_mass、random |
| random 掩码种子 | 42/43/44；不是三个独立生成种子 |
| 控制组 | 全读、可控块 Anchor-only |
| GPU/端口 | 0 / 18080 |

`cache_ordinal`是原有观察钩子按照首次出现的缓存张量分配的序号，不冒称已经验证其等于模型层名称。记录中保留该限制。

`--layers`可传全部层序号，但默认“全样本”不是“所有层所有生成 token 全采集”。存储上界和每题抓取点数随采集矩阵变化，`--plan`会打印。

## 7. 严格等逻辑字节如何实现

对于 K4V4、2+2 表示，每个完整 chunk 的 K Residual 和 V Residual 各有：

```text
packet_bytes = BLOCK_N * head_dim / 4
```

默认 32×128/4 = 1024 bytes。只把完整且 K/V 单次读取字节相同的 quantized chunks 作为可控候选。部分边界 chunk 的 K/V 全读；FP16 sink/tail 始终全读。它们没有被删除，成本对所有配对策略一致。

设候选完整 chunks 数 N，总预算参数 t = rho_K+rho_V。整数动作预算：

```text
M = floor(t*N)
K_min = max(0, M-N)
K_max = min(M, N)
K_count = round_half_up(K_min + share*(K_max-K_min))
V_count = M-K_count
```

因此，不同 K priority 的 K/V 残差合计始终是 **M 个 packet**。对称预算的 M 为奇数时，K/V 可能相差一个 packet，保留实际值，绝不伪造“完全相同的小数比例”。

N 足够且整除时，t=1.1 对应：

| K priority | K 残差读率 | V 残差读率 | K/V 码合计平均读取 |
|---:|---:|---:|---:|
| 0 | 10% | 100% | 6.2 bit / 原 8 bit |
| 0.25 | 32.5% | 77.5% | 6.2 bit / 原 8 bit |
| 0.5 | 55% | 55% | 6.2 bit / 原 8 bit |
| 0.75 | 77.5% | 32.5% | 6.2 bit / 原 8 bit |
| 1 | 100% | 10% | 6.2 bit / 原 8 bit |

这里的 bit 指“一对 K/V 标量的合计码位”，不是每个标量都是 6.2 bit。

分析器和汇总器都会再次检查配对字节相等；不相等直接失败。scale/pool 的固定逻辑成本单列。没有引入真实非对称 GPU kernel，因此这不是 GPU DRAM 流量相等的证明。

## 8. 选择器的信息边界

- `anchor_mass`：用当前 Query 与所有 K Anchor 得到的块 attention mass 排序，GQA 取真实 query heads 的最大块质量，K/V 各选固定数量。
- `staged_mass`：先同样选 K；再用已选择的完整 K 与其他 K Anchor 形成注意力，排序选择 V。未被选择的 K Residual 不用于 V 排序。
- `random`：固定随机排序，仍严格读取预算规定的块数。

没有 `best-prefix`、没有 `oracle_stop`，没有用完整参考输出决定“再读一个是否更好”。完整输出仅在掩码确定后用于误差评估。

离线分析为计算输出误差会读取原始完整快照；这里验证的是**选择规则的信息约束与载荷预算**。它不是一个已经通过性能验证的在线 top-k 实现。全局排序/第二阶段的缓存访问开销尚未计入。

## 9. 大样本磁盘与续跑

默认对每题执行：

```text
arm 钩子 → 原生 Parent 生成/抓取 → 关闭 arm
→ CPU 分析全部 trace 和 8 个 KV heads
→ 保存指标、掩码、验证、文件哈希
→ 提交该题 done.json
→ 清理本次该题新采集的原始 .npz
```

**默认删除的是成功分析后的新快照，不是旧实验文件。** 一旦默认清理，就不能用这些快照重新更换选择器分析；只能阅读保存的指标/掩码，或者重新采集。

需要保留原始 Query/KV 时，从第一次运行起加 `--keep-traces`，并准备足够磁盘空间；计划会给出保守未压缩上界。该参数也是续跑身份的一部分。

每题快照默认硬上限 `--scratch-gib 4`。增加层/位置导致上界超限时停止，必须显式提高。失败的未提交 attempt 保留，恢复时创建新的 attempt，不暗中删除失败证据。

续跑使用原来的参数和**同一个输出目录**：

```bash
bash kvbench_asym/run_equal_bytes.sh \
  --datasets longbench_all \
  --out "$OUT" \
  --resume
```

已经提交的题会核验结果哈希后跳过，不重复生成。未提交的题重新执行。数据、模型/tokenizer 元数据、配置、依赖和新包代码哈希发生变化则拒绝续跑。测试日志、说明文档不影响代码身份。

准备阶段单独执行可加 `--prepare-only`；之后省略该选项并加 `--resume`。

## 10. 结果与判读

### 10.1 离线 Query-aware Controller

在不改变 Parent 生成轨迹的前提下，可显式开启 Anchor + Residual 的离线 Controller：

```bash
bash kvbench_asym/run_equal_bytes.sh \
  --datasets longbench_all \
  --out results/asym_equal_bytes/controller_pilot01 \
  --case-limit 3 \
  --controller-tolerances 0.01 0.02 0.05 \
  --controller-margin 1.25
```

Controller 只使用 Anchor logits 和 block metadata；每次按“估计误差下降 / Residual bytes”选择收益最高的 block-side，达到 tolerance 后停止。`controller_case_level.csv` 和 `controller_aggregate.csv` 同时记录估计误差、真实 Full-KV 误差、`rho(q)`、步数和误停率。`underread_rate` 定义为真实误差超过目标 tolerance 的比例。

这仍是 Parent 轨迹上的离线反事实，不代表 GPU 已跳过相同的显存 load，也不报告 TPOT。只有在 `rho(q)` 有足够 query 间差异且误差估计的误停率可接受后，才进入 reader/kernel 接入。

```text
<OUT>/
  plan.json
  status.json
  selftest.log
  inputs/
    manifest.json
    cases.jsonl
    coverage.csv
  server_<本次启动ID>/server.log
  cases/<题目ID>/
    done.json
    attempt_<ID>/
      request.json
      capture_coverage.csv
      traces/trace_*.json
      analysis/
        rows.csv.gz
        masks.npz
        validation.json
        trace_metadata.json
        receipt.json
  comparison/
    case_level.csv
    case_coverage.csv
    aggregate.csv
    paired_vs_symmetric.csv
    controller_case_level.csv
    controller_aggregate.csv
    report.json
    readout.md
```

先看 `status.json`、`inputs/coverage.csv`，确认整个输入覆盖和截断情况；再看采集覆盖。自然 EOS 导致后期位置缺失不算服务故障，但必须体现覆盖偏差；按输出 token 数应该出现的点缺失则直接失败。

`validation.json`要求完整量化 float64 输出与采集的真实 GPU 输出相符，门槛为 relative Frobenius L2 ≤ 0.02 且最大元素绝对差 ≤ 0.02。此门槛只验证数据/计算路径，不是任务质量非劣阈值，更不是严格浮点认证。

`paired_vs_symmetric.csv`中：

- `mean_error_difference < 0`：比同载荷对称分配误差更低。
- `bootstrap_ci95_high < 0`：该探索性、请求级重采样区间位于零以下。
- 区间跨零：当前数据/统计尚不足以支持稳定优胜。

先在每道题内平均已采集的层/head/位置，再按题等权汇总；不把几千个 head 当成几千道独立题。多个 profile 的区间未作多重比较校正，不自动宣布论文显著性。

`p95_case_mean_error`是“逐题平均误差”的 P95，不是全体 head 原始误差的 P95；`mean_case_p95_error`是逐题内部 P95 的平均。逐组明细保留在各题压缩 CSV。

无可控量化历史的组不进入非对称优劣比较，仍保存控制和覆盖信息。该情况下应解释为“没有测试到可选择的量化历史”，不是“所有策略都达到完美保真”。

中断后生成已提交部分的汇总：

```bash
# 在原服务环境、项目根目录执行
python -m kvbench_asym.report --run "$OUT"
```

汇总报告会同时显示计划题数和完成题数，不将部分结果标为全量。

## 11. 复用已经保存的旧 trace

原始 `.npz` 仍存在时，可以先不跑新模型：

```bash
python -m kvbench_asym.analyze \
  --traces /绝对路径/旧批次/traces \
  --out /绝对路径/新的等载荷分析目录
```

不限制为 12 个 trace 或 3 个 KV heads；使用该目录全部 trace、全部 8 个 KV heads。这种单独分析不会自动生成全数据集请求级汇总；全量 runner 会生成。

## 12. 测试和未验证项

交付时实际运行 56 项 CPU/模拟接口测试，覆盖等字节整数预算、边界 chunk、GQA 共享、无参考输出泄漏、全 21 子任务覆盖、输入裁剪、真实数据数量/哈希、掩码回读、断点恢复、仅清理新快照、请求级统计和模拟完整流程。

未执行：RTX 3090、真实 vLLM 启动、CUDA/Triton 编译、真实模型采集、GPU 非对称选择 kernel、DRAM 性能计数器和任务评分。模拟服务测试不是 GPU 实测。

完整本地测试命令：

```bash
python -m unittest discover -s kvbench_asym/tests -v
```

第一次真实运行仍需要看实际服务日志和 `validation.json`；不能跳过源码哈希或数值检查。
