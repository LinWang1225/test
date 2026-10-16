# Progressive-KV Online v1：Qasper × 20 条，真实 vLLM + GPU Attention

> **这是一次小规模真实在线主实验，不是已出结果的论文结论。** 代码尚未在目标 RTX 3090 上实际编译/运行；若未得到 `status.json: completed` 和三组 `scores.summary.json`，不能声称测得 TPOT 或模型质量。报告中绝不填假数字。

## 1. 一句话运行

```bash
cd /home/wanglin/kvbench-3090-cu126
bash kvbench_progressive_online/run_qasper20.sh \
  --upstream-suite "$PWD/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json" \
  --task qasper --count 20 --concurrency 8 --gpu 0
```

运行脚本复用原 `kvarn-serve`、`evalscope-client`、KIT/env.sh、Qwen3-4B 模型和已冻结 LongBench Qasper 数据。不安装依赖、不下载数据，不重新采集 Trace。`--plan` 可以检查选中的 20 个 case ID（无 GPU 工作，但不是强制阶段）。

输出位于新目录 `results/progressive_online/qasper20_<日期>_<PID>`；已有目录不会覆盖。依次运行三组独立 vLLM 服务，绝不与原服务并行；不会清除旧结果。

## 2. 本次只测一个数据集、三种方法

| 实验组 | KV 存储 | GPU Decode | 用途 |
|---|---|---|---|
| `parent` | 原生 KVarN K4V4，每个标量 INT4 | 原生 Attention | **真实性能和质量基线** |
| `p211_full` | 原始 INT4 无损拆成 High2/Bit1/Bit0 | GPU 对每块完整读取 4bit | 衡量拆分/拼回/额外访存的**表示成本** |
| `p211_adaptive` | 与 `p211_full` **同样的完整 4bit 存储** | 按 Query 和当前 Chunk 的 Attention 代理分数决定读 2/3/4bit | 衡量多级选择的**实际净收益** |

20 条 Qasper 来自**成功的 2026-10-09 V3 已冻结长上下文数据**，使用其已确定的随机次序前 20 条，无重分词、无隐式裁剪。Three arms 使用完全相同的 20 个 case ID、请求种子、thinking=true、temperature=0.6、top_p=0.95、top_k=20，原主实验协议的 `max_tokens:null` / 剩余上下文预算。默认 C8 并发、显存使用率 0.85、APC 关闭；vLLM 正式路径默认**启用 CUDA Graph（不加 `--enforce-eager`）**。

每组有独立 Runtime copy，基于原 `kvbench_residual.runtime` 的审核过的 KVarN 固定 Commit `7586257f...` 和源码 SHA pins；在独立副本里修改 KVarN 的 torch Writer、torch Unpacker、完整量化 materializer、Verify full-reader、Triton fused Decode single/split。GPU 真实 Attention Kernel 中加载的是独立位平面。不是 HF 输出伪装 GPU 结果，不把 CPU 反事实当 TPOT。

## 3. 2+1+1 的精确编码

原 KVarN K4V4 码为整数 `c`，范围 0..15：

```
完整 INT4: b3 b2 b1 b0

High2:       b3 b2     -> 每 tile 2bit × N，连续地址
Bit1:              b1  -> 每 tile 1bit × N，连续地址
Bit0:                 b0 -> 每 tile 1bit × N，连续地址
```

重建值：

\[
Q_2(c)=4(c\gg2)+1.5,\quad
Q_3(c)=4(c\gg2)+2[(c\gg1)\&1]+0.5,\quad
Q_4(c)=c.
\]

完整 Q4 **逐码字、逐字节数值完全等于原 INT4**，不需要重新校准，也不需要 Gaussian Lloyd Tree 的二次量化。原 KVarN 的 Scale、Zero Point、FP16 Sink/Tail Pool 保持不变。

**真实 3bit 读取不会暗中读取完整的 2bit Low Plane**：`High2` 与后两条 1bit plane 在内存中物理分开，各有不同 `tl.load`；K/V 的精度选择路径由 Triton 统一标量分支处理，读取 B1 后才决定是否进一步加载 B0。自变量 `D` 和 `GROUP` 必须同时能被 8 整除，当前明确限定 Qwen3-4B、GQA 32/8、head_dim 128、K4V4/G128。

## 4. 在线 Controller（没有固定每 Query 读取预算）

对于每个当前真实 Decode Query 与当前 KV Chunk，统一处理共享该物理 KV Head 的 4 个 GQA Query Heads。先读 K 高两位，得到近似 Score `s2`，并依靠已有的 K Scale、Q 计算查询敏感度：

\[
 r_K(q)=|\mathrm{attn\_scale}|\max_{j\in b}|r_{K,j}|
   \sqrt{\sum_c(q_cs_{K,c})^2}.
\]

根据当前 Chunk 的粗精度 Logit 和**已处理前缀**在线维护的 LSE，计算一个近似重要性 `mass_proxy`。这里不是完整 Attention 归一化概率，也**不是有保证的误差上界**。

- 若 `mass_proxy × 1.5 × r_K <= risk-k2`，K 选 2bit。
- 否则加载 K Bit1，重算 3bit Logit；若 `mass_proxy3 × 0.5 × r_K <= risk-k3`，选 3bit。
- 否则加载 K Bit0，选完整 4bit。

得到已选 K 的 Score 后，在当前 Chunk 内立刻决定 V 精度（无需先扫完全局 K）：

\[
r_V=\frac{\max_{j\in b}|s_{V,j}|}{\max_{j\in b}\max(|z_{V,j}|,|z_{V,j}+15s_{V,j}|)}.
\]

当 `mass_proxy × 1.5 × rV <= risk-v2`，V 选 2bit；否则读 Bit1，当 `mass_proxy × 0.5 × rV <= risk-v3`，选择 3bit；否则读 Bit0 得到 4bit。全部 Query Heads 取 max risk，确保一个 GQA 物理 Chunk 统一读取深度。

默认阈值 `risk-k2=.08, risk-k3=.04, risk-v2=.12, risk-v3=.08` **属于未校准的探索性参数**，不是能够保证 Attention Error<epsilon 的置信界。每个 Decode step 读取多少 Residual 完全由实际 Query、Chunk 和阈值决定，既没有全局固定预算 B，也没有按总预算排序的 Controller。这种流式近似可能受历史访问顺序影响，不能直接声称 ReadKV 的分配最优性。

**注意：** 这个 Controller 的 FP16/TF32 舍入效果、在 CUDA Graph 下的 Triton 分支及质量都需要目标 GPU 实测。它可能让推理变慢或产生明显任务质量下降，必须保留并报告负面结果。

## 5. 与 ReadKV 的区别（论文工作要明确）

| 维度 | ReadKV | 本 Progressive-KV Online v1 |
|---|---|---|
| 码本 | Gaussian Lloyd 嵌套量化树、校准均值方差 | **原 KVarN INT4 码字无损分级拆分**，无需新量化器 |
| 完整位数 | 主要使用 W=8 | **W=4（2+1+1）**，控制变量匹配已有 Qwen3-4B 基线 |
| 读取预算 | **固定 R**，按各元素边际失真贪心分配 | **无固定总预算**，在线根据当前 Chunk 的粗 Attention 风险阈值选择深度 |
| K/V 粒度 | Key 按通道，Value 按历史 Token | **K/V 都按 GPU Token Chunk**，一个物理 GQA 组统一深度 |
| 依赖 | K 精度决定后计算近似 Attention，再对 V 全局分配 | **流式处理当前 Chunk 的 K → V → Online Softmax 继续**；不建立完整 Attention 概率矩阵 |
| 精度理论 | 固定预算内校准失真代理最优 | **暂不保证误差阈值**，需要在线 Quality/TPOT 实证和后续严格风险研究 |
| 主要目标 | 读比特预算的理论失真质量 | 在现有 vLLM K4V4 GPU fused 路径上验证**实际 TPOT/吞吐与质量权衡** |

二者都基于 Progressive Precision 和 Query-aware Reading，**不应把这两个已有概念本身声称为原创**。这版的区分点是兼容已有量化缓存 + 块级融合实时判断 + 非固定每 Query 读取量。但是否具有足够新的论文贡献，仍取决于质量/延迟实验的真实性能提升和后续系统/理论完善。

## 6. TPOT 与输出文件

每组重用 `kvbench_longcontext.run`（服务端 OpenAI SSE 真正解码）以及 `kvbench_longcontext.scoring`（原 LongBench QA F1），不重新定义评分。正式结果直接输出：

```
results/progressive_online/qasper20_.../
  plan.json                   # 20个固定case ID、源套件/模型 SHA
  status.json                 # 完成/失败状态，真实完成前绝不报告成功
  settings.json
  data/qasper.jsonl           # 选中请求、Prompt、答案及原生预算
  data/manifest.json          # 新数据集Hash + 来源
  runtimes/{parent,p211_full,p211_adaptive}/runtime.json
  servers/<arm>/progressive.launch.json
  servers/<arm>/server.log
  servers/<arm>/dispatch_summary.json
  runs/<arm>/r01_qasper_C8/requests.jsonl
  runs/<arm>/r01_qasper_C8/performance.summary.json
  runs/<arm>/r01_qasper_C8/scores.jsonl
  runs/<arm>/r01_qasper_C8/scores.summary.json
  runs/<arm>/r01_qasper_C8/telemetry.jsonl
  logs/*.log
  comparison.csv
  comparison.json
  paired_quality.csv
  readout.md                  # 最终中文三组对照表
```

### 指标定义

- **TPOT**：每请求 `(E2E - 首个内容/思考片段时间)/(native completion_tokens-1)`，按请求等权平均；报告 P95。这是包含 HTTP/调度的在线 TPOT，不是一个纯 Triton Kernel 时间。
- **TTFT**：首个内容或 reasoning 片段的延迟；不是“最终验证完成的第一个回答 token”专用指标。
- **Output Tokens/s**：并发请求总完成 Token / 真实测量窗口。
- **LongBench Score**：Qasper 官方 QA F1，转换为 0~100 分；20 个 case 逐一配对，不丢弃截断响应。
- **GPU 使用量**：nvidia-smi 周期采样的显存使用峰值，是 GPU-wide 采样值，不等于算子的 Peak Allocated 或 HBM 物理读取计数。
- **输出长度/E2E/Preemptions**：报告辅助解读 TPOT，避免把生成更短误当作 Kernel 加速。

本补丁**不**提供真实 `dram__bytes_read`、逐 Chunk 的动态精度占比或逐 token CUDA events；如以后要核实逻辑节省是否变成物理 DRAM 节省，可再单独启用 ncu。但这不是运行此次 20 条在线主实验的前置条件。

## 7. 补丁使用步骤

方案 A（Git Apply）：

```bash
cd /home/wanglin/kvbench-3090-cu126
PATCH=/绝对路径/kvbench_progressive_online_qasper20_20261015.patch
git apply --check "$PATCH"
git apply "$PATCH"
# 可选的轻量正确性测试，CPU 9 项
source /home/wanglin/miniconda3/etc/profile.d/conda.sh
conda activate kvarn-serve
python -m unittest discover -s kvbench_progressive_online/tests -v
# 真正的三组 GPU 在线实验
bash kvbench_progressive_online/run_qasper20.sh \
 --upstream-suite "$PWD/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json" \
 --task qasper --count 20 --concurrency 8 --gpu 0
```

方案 B：源码 ZIP 直接解压到项目根目录，跳过 `git apply`，后续执行完全相同。不要既解压又重复应用补丁。

如需要先看选择了哪 20 条（非必须）：

```bash
bash kvbench_progressive_online/run_qasper20.sh --plan
```

若 CUDA 编译失败，先读 `status.json`、`logs/p211_*.server_launcher.log`、`servers/p211_*/server.log`，保留实际 Triton traceback；不要用其他生成结果代替失败实验。源码固定 SHA 不匹配时明确停止，不强行修改旧版 vLLM。

### 运行环境与验证边界

依赖仓库已有 `kvbench_residual/`、`kvbench_longcontext/` 和已运行的 Qwen3-4B/KVarN fork。没有使用之前的离线 `kvbench_progressive` 包，因此可直接在未应用上一版补丁时使用。

已经在交付环境验证：Python 编译、Shell 语法、NumPy/Torch CPU 编码无损往返、物理 bit-plane 地址、源码变换局部测试。**未在交付环境运行目标 RTX 3090、Triton Kernel 编译、在线生成、LongBench 打分或 TPOT**。实际结果应从本补丁的用户侧运行中产生。
