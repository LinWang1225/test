# AR-KV 效果与消融读出

固定列出 K4V4 × 6 个任务 × 6 个预先定义的对照，共 36 行。只读取已保存的 V3 评分和性能，不重新评分，不按结果选择策略。

主差值仅在两侧测量、评分、最终实现来源证据完整，且冻结数据与实验协议一致时计算。失败或缺失保留为 NA，不能当成 0 分或 0 延迟。单项分母为 0 或指标不可计算时，该差值也为 NA。

得分变化是 0–100 任务分的直接差，记为 pp；它不是所有任务通用的二元正确率。TPOT 改善=100×(1−新/对照)，正数表示每 token 时间下降；输出吞吐增益=100×(新/对照−1)，正数表示提高。

E2E 和输出长度变化=100×(新/对照−1)，负数表示减少。生成更短不等于 kernel 更快，应一起阅读质量、TPOT、吞吐、输出长度、截断和空答案。

## 净效果：K/V 误差界选择与原 KVarN K4V4

| 精度 | 任务 | 得分Δ pp | TPOT改善% | 输出吞吐增益% | E2E变化% | 输出长度变化% | 主对照可用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k4v4 | qasper | -0.52 | -30.43 | -22.29 | 32.27 | 2.84 | 是 |
| k4v4 | multifieldqa_en | -0.91 | -21.77 | -19.19 | 33.88 | 8.48 | 是 |
| k4v4 | hotpotqa | 0.01 | -29.89 | -26.28 | 22.01 | -7.46 | 是 |
| k4v4 | 2wikimqa | 2.56 | -41.37 | -27.32 | 37.93 | -3.01 | 是 |
| k4v4 | gov_report | 0.29 | -46.82 | -31.24 | 46.33 | 0.44 | 是 |
| k4v4 | passage_retrieval_en | 0.33 | -33.47 | -24.36 | 30.36 | -1.50 | 是 |

## 表示开销：拆分平面全读与原布局

| 精度 | 任务 | 得分Δ pp | TPOT改善% | 输出吞吐增益% | E2E变化% | 输出长度变化% | 主对照可用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k4v4 | qasper | -0.95 | -16.14 | -30.72 | 32.15 | 33.00 | 是 |
| k4v4 | multifieldqa_en | -1.57 | -16.23 | -12.91 | 15.15 | 2.27 | 是 |
| k4v4 | hotpotqa | 1.02 | -16.65 | -14.24 | -3.18 | -23.94 | 是 |
| k4v4 | 2wikimqa | -1.60 | -18.64 | -19.32 | 15.90 | 5.30 | 是 |
| k4v4 | gov_report | 0.20 | -19.68 | -15.89 | 21.32 | 1.94 | 是 |
| k4v4 | passage_retrieval_en | 0.50 | -15.63 | -14.32 | 13.95 | -2.40 | 是 |

## Anchor 极限：全部量化 token 保留 Anchor，但跳过全部残差

| 精度 | 任务 | 得分Δ pp | TPOT改善% | 输出吞吐增益% | E2E变化% | 输出长度变化% | 主对照可用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k4v4 | qasper | 2.02 | 10.52 | 41.14 | -16.79 | -19.88 | 是 |
| k4v4 | multifieldqa_en | 0.29 | 7.15 | 6.34 | -5.21 | 1.58 | 是 |
| k4v4 | hotpotqa | 0.81 | 19.12 | 22.05 | -5.75 | 15.42 | 是 |
| k4v4 | 2wikimqa | 0.91 | 16.97 | 43.72 | -7.60 | -1.80 | 是 |
| k4v4 | gov_report | -0.63 | 10.83 | 9.42 | -9.44 | -0.91 | 是 |
| k4v4 | passage_retrieval_en | -3.83 | 12.29 | 11.66 | 5.28 | 25.55 | 是 |

## K 选择：仅 K 按误差界跳读，V 残差全读

| 精度 | 任务 | 得分Δ pp | TPOT改善% | 输出吞吐增益% | E2E变化% | 输出长度变化% | 主对照可用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k4v4 | qasper | -0.13 | -11.70 | 14.29 | 1.08 | -21.10 | 是 |
| k4v4 | multifieldqa_en | 0.14 | -7.44 | -8.86 | 16.39 | 3.78 | 是 |
| k4v4 | hotpotqa | -1.08 | -12.22 | -12.34 | 25.71 | 16.11 | 是 |
| k4v4 | 2wikimqa | 1.85 | -12.72 | 8.13 | 3.56 | -17.29 | 是 |
| k4v4 | gov_report | -0.09 | -18.55 | -15.40 | 16.41 | -1.47 | 是 |
| k4v4 | passage_retrieval_en | -0.33 | -13.61 | -10.30 | 13.35 | 1.76 | 是 |

## V 选择：K 残差全读，仅 V 按误差界跳读

| 精度 | 任务 | 得分Δ pp | TPOT改善% | 输出吞吐增益% | E2E变化% | 输出长度变化% | 主对照可用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k4v4 | qasper | -0.58 | -4.49 | 20.35 | -2.52 | -19.62 | 是 |
| k4v4 | multifieldqa_en | -1.78 | -7.08 | -7.15 | 2.17 | -4.93 | 是 |
| k4v4 | hotpotqa | -0.10 | -4.76 | -3.24 | 3.84 | -0.34 | 是 |
| k4v4 | 2wikimqa | 1.34 | -5.41 | 13.53 | -4.06 | -18.37 | 是 |
| k4v4 | gov_report | -0.14 | -8.66 | -8.25 | 7.77 | -0.99 | 是 |
| k4v4 | passage_retrieval_en | 0.00 | -4.22 | -3.58 | 7.33 | 3.19 | 是 |

## K/V 联合选择：相对同表示全读，预算各分一半

| 精度 | 任务 | 得分Δ pp | TPOT改善% | 输出吞吐增益% | E2E变化% | 输出长度变化% | 主对照可用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k4v4 | qasper | 0.42 | -12.30 | 12.16 | 0.09 | -22.68 | 是 |
| k4v4 | multifieldqa_en | 0.65 | -4.77 | -7.20 | 16.27 | 6.07 | 是 |
| k4v4 | hotpotqa | -1.00 | -11.35 | -14.04 | 26.02 | 21.67 | 是 |
| k4v4 | 2wikimqa | 4.16 | -19.16 | -9.91 | 19.01 | -7.89 | 是 |
| k4v4 | gov_report | 0.09 | -22.67 | -18.25 | 20.61 | -1.47 | 是 |
| k4v4 | passage_retrieval_en | -0.17 | -15.42 | -11.72 | 14.40 | 0.92 | 是 |

## 截断、空答案与抢占

| 对照 | 精度 | 任务 | 截断率Δ pp | 空答案数Δ | 空答案率Δ pp | 抢占次数Δ |
| --- | --- | --- | --- | --- | --- | --- |
| main_net | k4v4 | qasper | 0.00 | 0 | 0.00 | 0.00 |
| main_net | k4v4 | multifieldqa_en | 0.00 | 0 | 0.00 | 0.00 |
| main_net | k4v4 | hotpotqa | -0.50 | -1 | -0.50 | 0.00 |
| main_net | k4v4 | 2wikimqa | 0.00 | 0 | 0.00 | 0.00 |
| main_net | k4v4 | gov_report | 0.00 | 0 | 0.00 | 0.00 |
| main_net | k4v4 | passage_retrieval_en | 0.00 | 0 | 0.00 | 0.00 |
| representation | k4v4 | qasper | 0.50 | 1 | 0.50 | 0.00 |
| representation | k4v4 | multifieldqa_en | 0.00 | 0 | 0.00 | 0.00 |
| representation | k4v4 | hotpotqa | -1.00 | -2 | -1.00 | 0.00 |
| representation | k4v4 | 2wikimqa | 0.00 | 0 | 0.00 | 0.00 |
| representation | k4v4 | gov_report | 0.00 | 0 | 0.00 | 0.00 |
| representation | k4v4 | passage_retrieval_en | 0.00 | 0 | 0.00 | 0.00 |
| anchor_limit | k4v4 | qasper | -0.50 | -1 | -0.50 | 0.00 |
| anchor_limit | k4v4 | multifieldqa_en | 0.00 | 0 | 0.00 | 0.00 |
| anchor_limit | k4v4 | hotpotqa | 0.00 | 0 | 0.00 | 0.00 |
| anchor_limit | k4v4 | 2wikimqa | -0.50 | -1 | -0.50 | 0.00 |
| anchor_limit | k4v4 | gov_report | 0.00 | 0 | 0.00 | 0.00 |
| anchor_limit | k4v4 | passage_retrieval_en | 0.50 | 1 | 0.50 | 0.00 |
| key_selection | k4v4 | qasper | -0.50 | -1 | -0.50 | 0.00 |
| key_selection | k4v4 | multifieldqa_en | 0.00 | 0 | 0.00 | 0.00 |
| key_selection | k4v4 | hotpotqa | 0.50 | 1 | 0.50 | 0.00 |
| key_selection | k4v4 | 2wikimqa | -0.50 | -1 | -0.50 | 0.00 |
| key_selection | k4v4 | gov_report | 0.00 | 0 | 0.00 | 0.00 |
| key_selection | k4v4 | passage_retrieval_en | 0.00 | 0 | 0.00 | 0.00 |
| value_selection | k4v4 | qasper | -0.50 | -1 | -0.50 | 0.00 |
| value_selection | k4v4 | multifieldqa_en | 0.00 | 0 | 0.00 | 0.00 |
| value_selection | k4v4 | hotpotqa | 0.00 | 0 | 0.00 | 0.00 |
| value_selection | k4v4 | 2wikimqa | -0.50 | -1 | -0.50 | 0.00 |
| value_selection | k4v4 | gov_report | 0.00 | 0 | 0.00 | 0.00 |
| value_selection | k4v4 | passage_retrieval_en | 0.00 | 0 | 0.00 | 0.00 |
| joint_selection | k4v4 | qasper | -0.50 | -1 | -0.50 | 0.00 |
| joint_selection | k4v4 | multifieldqa_en | 0.00 | 0 | 0.00 | 0.00 |
| joint_selection | k4v4 | hotpotqa | 0.50 | 1 | 0.50 | 0.00 |
| joint_selection | k4v4 | 2wikimqa | 0.00 | 0 | 0.00 | 0.00 |
| joint_selection | k4v4 | gov_report | 0.00 | 0 | 0.00 | 0.00 |
| joint_selection | k4v4 | passage_retrieval_en | 0.00 | 0 | 0.00 | 0.00 |

完整 length 截断照常评分；预算一致性诊断也保存在 CSV 中，不因为截断而删除题目。实现证据来自最终 Python 导入/launch/capture 记录，包含预热；它不表示对每个正式 token 或 CUDA graph replay 逐个计数。

parent 使用原 KVarN K4V4 布局；repacked_full 使用拆分平面但读取全部残差。main_net 检查整个方法净效果，joint_selection 用同表示全读对照隔离选择收益，representation 单独检查拆分布局的成本。已有九方法结果在 overview.md/csv 中供参考。

epsilon 是单步 attention 输出误差控制参数，不是任务得分损失百分比或生成序列一致性保证。正式推理不加入流量原子计数；本表 formal_dram_bytes 保留 NA。独立 microbench 的逻辑字节估计只能解释该重放工作负载，不能冒充正式推理 DRAM 实测。

这是每组一次的全量对照，不做跨任务总体准确率、自动排名或随机种子置信区间。需要重复种子实验才能进一步评估采样波动。

来源：`/home/wanglin/kvbench-3090-cu126/results/residual_v1/full_20261010_160834_3993496/suite.json`；data_manifest_sha256=`609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522`。
