# 实验二：真实 trace 的读取量—attention 输出误差

Trace：12；KV heads：[0, 3, 7]；结果行：2664。默认只是显式子样本，不代表所有层/请求。

`curves.csv` 保留每个 trace/GQA 组，`aggregate.csv` 为组等权统计，不是模型总体准确率。
K-only 固定 V 全读；V-only 固定 K 全读；KV 联合决定两类低位读取。所有 token 的 Anchor 始终保留，sink/tail 始终完整。
`controllable_read_fraction` 仅指本实验可控制的 Residual；`all_residual_read_fraction` 包含固定全读的另一侧。
`logical_code_reduction_fraction` 只统计量化码；scale/FP16 pool 不变且未含其中。不是 DRAM 节省。

## Oracle 的准确含义

offline_greedy 使用完整 Residual 做离线一步前瞻，不可部署，也不保证最优；记录 best-feasible-prefix，以处理残差抵消导致的非单调。
best_feasible_envelope 是各可行算法在相同预算下的最佳误差，因而是最优误差的上界，不是信息论下界。
V-only 的 certified_v_error_lower_bound 使用三角不等式和分数背包松弛；为 0 时通常没有判别力。
exhaustive_exact_small 只在动作数不超过阈值的完整 GQA 组枚举所有子集。不裁短序列伪装原序列最优。
K/KV 的大规模精确下界本补丁没有实现；相应字段为 NA。

## 判读

可行方案在低读取比例达到低误差：说明存在可实现选择空间，但还须设计廉价在线控制器。
有效下界仍明显高于目标误差：可排除该 V-only 读取预算。
贪心表现差、下界为 0：证据不足，不能据此否定方法。
误差相对完整量化 attention 输出计算；不等同 FP16 模型误差、任务得分、生成长度或 TPOT。

| 模式 | 策略 | 预算比例 | 组数 | 平均实际可控残差读取 | 平均码字节减少 | 最大头相对误差中位数 |
|---|---|---:|---:|---:|---:|---:|
| v | best_feasible_envelope | 0.00000 | 36 | 0.00000 | 0.25000 | 0.09966 |
| v | best_feasible_envelope | 0.10000 | 36 | 0.09921 | 0.22520 | 0.00490 |
| v | best_feasible_envelope | 0.25000 | 36 | 0.24980 | 0.18755 | 0.00251 |
| v | best_feasible_envelope | 0.50000 | 36 | 0.49835 | 0.12541 | 0.00130 |
| v | best_feasible_envelope | 0.75000 | 36 | 0.74926 | 0.06268 | 0.00037 |
| v | best_feasible_envelope | 1.00000 | 36 | 1.00000 | 0.00000 | 0.00000 |
| v | current_bound_cpu | NA | 36 | 0.99137 | 0.00216 | 0.00000 |
| kv | best_feasible_envelope | 0.00000 | 36 | 0.00000 | 0.50000 | 0.13599 |
| kv | best_feasible_envelope | 0.10000 | 36 | 0.09905 | 0.45047 | 0.01364 |
| kv | best_feasible_envelope | 0.25000 | 36 | 0.24940 | 0.37530 | 0.00608 |
| kv | best_feasible_envelope | 0.50000 | 36 | 0.49874 | 0.25063 | 0.00282 |
| kv | best_feasible_envelope | 0.75000 | 36 | 0.74692 | 0.12654 | 0.00078 |
| kv | best_feasible_envelope | 1.00000 | 36 | 1.00000 | 0.00000 | 0.00000 |
| kv | current_bound_cpu | NA | 36 | 0.99703 | 0.00149 | 0.00000 |
