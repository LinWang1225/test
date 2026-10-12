# AR-KV 全量实验状态

阶段：`finished`

上游批次：`/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json`

| 方法 | 状态 | 实现证据 | 测量完整点 | 评分完整点 | 有效/应有 | 已评分 | 截断 | 预算异常点 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ar_k4v4_parent | completed | True | 6/6 | 6/6 | 1147/1147 | 1147 | 3 | 0 |
| ar_k4v4_repacked_full | completed | True | 6/6 | 6/6 | 1147/1147 | 1147 | 2 | 0 |
| ar_k4v4_anchor | completed | True | 6/6 | 6/6 | 1147/1147 | 1147 | 1 | 0 |
| ar_k4v4_bound_k | completed | True | 6/6 | 6/6 | 1147/1147 | 1147 | 1 | 0 |
| ar_k4v4_bound_v | completed | True | 6/6 | 6/6 | 1147/1147 | 1147 | 0 | 0 |
| ar_k4v4_bound_kv | completed | True | 6/6 | 6/6 | 1147/1147 | 1147 | 2 | 0 |

总览步骤 exit=0；差值步骤 exit=0（null/None 表示尚未执行，不是成功）。
全部正式请求保持 thinking、C8、相同采样、max_tokens:null；完整截断响应仍评分。
GPU 正确性通过后才启动全量实验；微基准速度不决定保留哪种策略。
六组共享空 mask 数值保护；parent 保留原布局/算式但不是旧源字节副本。repacked_full 验证拆分表示本身。
当前全量仅支持 K4V4；error budget 不等于任务准确率保证。正式流量计数为 NA，诊断 logical bytes 不是真实 DRAM 事务。
先看 validation/validation.json，再看 microbench/benchmark.json 和 comparison/overview.md。
等待旧批次时不会加载模型、调用 GPU 核函数或停止其他进程。
