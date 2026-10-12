# 实验一：真实 decode 跳读诊断

状态：completed；真实 trace：72；结果行：216。

这些是 Parent 轨迹上相同 Query/KV 的反事实重放，不是 bound 策略自然生成全程的跳读率。
没有 TPOT、吞吐或 DRAM 实测。GPU 字节数来自实际分支逻辑地址计数。

## 读数方法

`summary.csv`：先看 gpu_executed，再看 gpu_key/value_skipped_chunks 与 gpu_logical_code_reduction_fraction。
`chunk_decisions.jsonl`：threshold_ratio > 1 表示该头/组没有通过预算；查看 bound/actual、mass 上界和 GQA 阻断。
`counts_*.npy`：列名顺序是 kvbench_residual.kernel.STATS_FIELDS，保存每个 KV head/split 的 GPU 计数。
`validation.json`：完整量化 float64 输出与真实 GPU 输出、Parent 重放对照。CPU/GPU 分支不同必须检查边界，不能悄悄覆盖计数。

建议用 --splits 1 32 在相同 trace 比较 split 保守性的影响；这是数学/分支消融，不是调度性能结论。
