# 长输入实验结果速读

本报告只读取已保存的结果，不运行推理或重新评分。先检查测量完整/评分完整，再比较同一任务、同一数据清单与协议。

NA 表示尚无完整结果或指标不可计算；真实 0 分保留为 0。CSV 的空白数值对应 NA。未完成点仍按运行计划列出。

## 质量与完整性

| 运行 | 任务 | 重复/并发 | 有效/应有 | 已评分 | 质量分0–100 | 截断数 | 空答案 | 测量完整 | 评分完整 | 预算一致 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | qasper | r1/C8 | 200/200 | 200 | 44.04 | 1 | 1 | 是 | 是 | 是 |
| fp16 | multifieldqa_en | r1/C8 | 150/150 | 150 | 49.46 | 0 | 0 | 是 | 是 | 是 |
| fp16 | hotpotqa | r1/C8 | 200/200 | 200 | 64.04 | 2 | 2 | 是 | 是 | 是 |
| fp16 | 2wikimqa | r1/C8 | 200/200 | 200 | 73.25 | 0 | 0 | 是 | 是 | 是 |
| fp16 | gov_report | r1/C8 | 197/197 | 197 | 29.02 | 0 | 0 | 是 | 是 | 是 |
| fp16 | passage_retrieval_en | r1/C8 | 200/200 | 200 | 99.00 | 0 | 0 | 是 | 是 | 是 |
| fp8 | qasper | r1/C8 | 200/200 | 200 | 42.23 | 1 | 1 | 是 | 是 | 是 |
| fp8 | multifieldqa_en | r1/C8 | 150/150 | 150 | 47.38 | 0 | 0 | 是 | 是 | 是 |
| fp8 | hotpotqa | r1/C8 | 200/200 | 200 | 62.97 | 1 | 0 | 是 | 是 | 是 |
| fp8 | 2wikimqa | r1/C8 | 200/200 | 200 | 73.10 | 0 | 0 | 是 | 是 | 是 |
| fp8 | gov_report | r1/C8 | 197/197 | 197 | 29.41 | 0 | 0 | 是 | 是 | 是 |
| fp8 | passage_retrieval_en | r1/C8 | 200/200 | 200 | 98.42 | 0 | 0 | 是 | 是 | 是 |
| tq4 | qasper | r1/C8 | 200/200 | 200 | 42.10 | 0 | 0 | 是 | 是 | 是 |
| tq4 | multifieldqa_en | r1/C8 | 150/150 | 150 | 48.77 | 0 | 0 | 是 | 是 | 是 |
| tq4 | hotpotqa | r1/C8 | 200/200 | 200 | 65.41 | 0 | 0 | 是 | 是 | 是 |
| tq4 | 2wikimqa | r1/C8 | 200/200 | 200 | 72.97 | 0 | 0 | 是 | 是 | 是 |
| tq4 | gov_report | r1/C8 | 197/197 | 197 | 29.24 | 0 | 0 | 是 | 是 | 是 |
| tq4 | passage_retrieval_en | r1/C8 | 200/200 | 200 | 99.01 | 0 | 0 | 是 | 是 | 是 |
| tq34 | qasper | r1/C8 | 200/200 | 200 | 41.70 | 1 | 1 | 是 | 是 | 是 |
| tq34 | multifieldqa_en | r1/C8 | 150/150 | 150 | 47.91 | 0 | 0 | 是 | 是 | 是 |
| tq34 | hotpotqa | r1/C8 | 200/200 | 200 | 61.78 | 0 | 0 | 是 | 是 | 是 |
| tq34 | 2wikimqa | r1/C8 | 200/200 | 200 | 70.74 | 0 | 0 | 是 | 是 | 是 |
| tq34 | gov_report | r1/C8 | 197/197 | 197 | 29.03 | 0 | 0 | 是 | 是 | 是 |
| tq34 | passage_retrieval_en | r1/C8 | 200/200 | 200 | 93.88 | 1 | 1 | 是 | 是 | 是 |
| tq3 | qasper | r1/C8 | 200/200 | 200 | 41.34 | 0 | 0 | 是 | 是 | 是 |
| tq3 | multifieldqa_en | r1/C8 | 150/150 | 150 | 46.67 | 0 | 0 | 是 | 是 | 是 |
| tq3 | hotpotqa | r1/C8 | 200/200 | 200 | 63.82 | 0 | 0 | 是 | 是 | 是 |
| tq3 | 2wikimqa | r1/C8 | 200/200 | 200 | 72.01 | 1 | 1 | 是 | 是 | 是 |
| tq3 | gov_report | r1/C8 | 197/197 | 197 | 28.52 | 0 | 0 | 是 | 是 | 是 |
| tq3 | passage_retrieval_en | r1/C8 | 200/200 | 200 | 94.88 | 0 | 0 | 是 | 是 | 是 |
| kvarn | qasper | r1/C8 | 200/200 | 200 | 43.06 | 0 | 0 | 是 | 是 | 是 |
| kvarn | multifieldqa_en | r1/C8 | 150/150 | 150 | 50.30 | 0 | 0 | 是 | 是 | 是 |
| kvarn | hotpotqa | r1/C8 | 200/200 | 200 | 62.82 | 1 | 1 | 是 | 是 | 是 |
| kvarn | 2wikimqa | r1/C8 | 200/200 | 200 | 75.16 | 0 | 0 | 是 | 是 | 是 |
| kvarn | gov_report | r1/C8 | 197/197 | 197 | 29.33 | 0 | 0 | 是 | 是 | 是 |
| kvarn | passage_retrieval_en | r1/C8 | 200/200 | 200 | 96.54 | 1 | 1 | 是 | 是 | 是 |
| kvarn44 | qasper | r1/C8 | 200/200 | 200 | 41.35 | 0 | 0 | 是 | 是 | 是 |
| kvarn44 | multifieldqa_en | r1/C8 | 150/150 | 150 | 50.70 | 0 | 0 | 是 | 是 | 是 |
| kvarn44 | hotpotqa | r1/C8 | 200/200 | 200 | 65.99 | 2 | 1 | 是 | 是 | 是 |
| kvarn44 | 2wikimqa | r1/C8 | 200/200 | 200 | 73.89 | 0 | 0 | 是 | 是 | 是 |
| kvarn44 | gov_report | r1/C8 | 197/197 | 197 | 29.17 | 0 | 0 | 是 | 是 | 是 |
| kvarn44 | passage_retrieval_en | r1/C8 | 200/200 | 200 | 99.17 | 0 | 0 | 是 | 是 | 是 |
| kvarn64 | qasper | r1/C8 | 200/200 | 200 | 41.37 | 0 | 0 | 是 | 是 | 是 |
| kvarn64 | multifieldqa_en | r1/C8 | 150/150 | 150 | 49.41 | 0 | 0 | 是 | 是 | 是 |
| kvarn64 | hotpotqa | r1/C8 | 200/200 | 200 | 65.28 | 0 | 0 | 是 | 是 | 是 |
| kvarn64 | 2wikimqa | r1/C8 | 200/200 | 200 | 74.50 | 0 | 0 | 是 | 是 | 是 |
| kvarn64 | gov_report | r1/C8 | 197/197 | 197 | 29.55 | 0 | 0 | 是 | 是 | 是 |
| kvarn64 | passage_retrieval_en | r1/C8 | 200/200 | 200 | 98.17 | 0 | 0 | 是 | 是 | 是 |
| kvarn4464 | qasper | r1/C8 | 200/200 | 200 | 42.21 | 2 | 1 | 是 | 是 | 是 |
| kvarn4464 | multifieldqa_en | r1/C8 | 150/150 | 150 | 50.11 | 0 | 0 | 是 | 是 | 是 |
| kvarn4464 | hotpotqa | r1/C8 | 200/200 | 200 | 64.60 | 2 | 2 | 是 | 是 | 是 |
| kvarn4464 | 2wikimqa | r1/C8 | 200/200 | 200 | 70.87 | 0 | 0 | 是 | 是 | 是 |
| kvarn4464 | gov_report | r1/C8 | 197/197 | 197 | 29.47 | 0 | 0 | 是 | 是 | 是 |
| kvarn4464 | passage_retrieval_en | r1/C8 | 200/200 | 200 | 99.00 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_parent | qasper | r1/C8 | 200/200 | 200 | 43.10 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_parent | multifieldqa_en | r1/C8 | 150/150 | 150 | 50.25 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_parent | hotpotqa | r1/C8 | 200/200 | 200 | 64.39 | 2 | 2 | 是 | 是 | 是 |
| ar_k4v4_parent | 2wikimqa | r1/C8 | 200/200 | 200 | 73.13 | 1 | 1 | 是 | 是 | 是 |
| ar_k4v4_parent | gov_report | r1/C8 | 197/197 | 197 | 29.12 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_parent | passage_retrieval_en | r1/C8 | 200/200 | 200 | 99.33 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_repacked_full | qasper | r1/C8 | 200/200 | 200 | 42.16 | 1 | 1 | 是 | 是 | 是 |
| ar_k4v4_repacked_full | multifieldqa_en | r1/C8 | 150/150 | 150 | 48.68 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_repacked_full | hotpotqa | r1/C8 | 200/200 | 200 | 65.40 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_repacked_full | 2wikimqa | r1/C8 | 200/200 | 200 | 71.54 | 1 | 1 | 是 | 是 | 是 |
| ar_k4v4_repacked_full | gov_report | r1/C8 | 197/197 | 197 | 29.33 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_repacked_full | passage_retrieval_en | r1/C8 | 200/200 | 200 | 99.83 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_anchor | qasper | r1/C8 | 200/200 | 200 | 44.18 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_anchor | multifieldqa_en | r1/C8 | 150/150 | 150 | 48.98 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_anchor | hotpotqa | r1/C8 | 200/200 | 200 | 66.21 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_anchor | 2wikimqa | r1/C8 | 200/200 | 200 | 72.45 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_anchor | gov_report | r1/C8 | 197/197 | 197 | 28.70 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_anchor | passage_retrieval_en | r1/C8 | 200/200 | 200 | 96.00 | 1 | 1 | 是 | 是 | 是 |
| ar_k4v4_bound_k | qasper | r1/C8 | 200/200 | 200 | 42.03 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_k | multifieldqa_en | r1/C8 | 150/150 | 150 | 48.82 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_k | hotpotqa | r1/C8 | 200/200 | 200 | 64.32 | 1 | 1 | 是 | 是 | 是 |
| ar_k4v4_bound_k | 2wikimqa | r1/C8 | 200/200 | 200 | 73.38 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_k | gov_report | r1/C8 | 197/197 | 197 | 29.24 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_k | passage_retrieval_en | r1/C8 | 200/200 | 200 | 99.50 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_v | qasper | r1/C8 | 200/200 | 200 | 41.58 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_v | multifieldqa_en | r1/C8 | 150/150 | 150 | 46.90 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_v | hotpotqa | r1/C8 | 200/200 | 200 | 65.30 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_v | 2wikimqa | r1/C8 | 200/200 | 200 | 72.88 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_v | gov_report | r1/C8 | 197/197 | 197 | 29.18 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_v | passage_retrieval_en | r1/C8 | 200/200 | 200 | 99.83 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_kv | qasper | r1/C8 | 200/200 | 200 | 42.58 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_kv | multifieldqa_en | r1/C8 | 150/150 | 150 | 49.34 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_kv | hotpotqa | r1/C8 | 200/200 | 200 | 64.40 | 1 | 1 | 是 | 是 | 是 |
| ar_k4v4_bound_kv | 2wikimqa | r1/C8 | 200/200 | 200 | 75.70 | 1 | 1 | 是 | 是 | 是 |
| ar_k4v4_bound_kv | gov_report | r1/C8 | 197/197 | 197 | 29.41 | 0 | 0 | 是 | 是 | 是 |
| ar_k4v4_bound_kv | passage_retrieval_en | r1/C8 | 200/200 | 200 | 99.67 | 0 | 0 | 是 | 是 | 是 |

## 性能

| 运行 | 任务 | 重复/并发 | 输入均值token | 输出均值token | TTFT均值ms | TPOT均值ms | E2E均值s | E2E p95 s | 输出token/s | 抢占增量 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | qasper | r1/C8 | 5200.42 | 597.48 | 1332.52 | 33.38 | 19.14 | 55.86 | 142.81 | 0.00 |
| fp16 | multifieldqa_en | r1/C8 | 7177.52 | 341.32 | 2083.07 | 49.73 | 18.96 | 47.39 | 136.21 | 0.00 |
| fp16 | hotpotqa | r1/C8 | 13454.45 | 740.37 | 25073.03 | 48.99 | 59.66 | 135.30 | 97.58 | 16.00 |
| fp16 | 2wikimqa | r1/C8 | 7554.02 | 496.94 | 2158.19 | 45.69 | 24.07 | 83.36 | 153.59 | 1.00 |
| fp16 | gov_report | r1/C8 | 10081.48 | 975.61 | 6861.76 | 41.11 | 47.03 | 68.27 | 164.59 | 8.00 |
| fp16 | passage_retrieval_en | r1/C8 | 12923.12 | 513.72 | 13359.99 | 54.71 | 41.63 | 65.59 | 97.89 | 8.00 |
| fp8 | qasper | r1/C8 | 5200.42 | 597.26 | 1225.81 | 27.85 | 16.42 | 38.72 | 176.31 | 0.00 |
| fp8 | multifieldqa_en | r1/C8 | 7177.52 | 338.45 | 1890.65 | 41.77 | 16.03 | 31.81 | 167.01 | 0.00 |
| fp8 | hotpotqa | r1/C8 | 13454.45 | 690.87 | 4154.43 | 51.27 | 38.48 | 110.62 | 140.42 | 0.00 |
| fp8 | 2wikimqa | r1/C8 | 7554.02 | 484.69 | 2056.25 | 36.45 | 19.61 | 52.32 | 194.68 | 0.00 |
| fp8 | gov_report | r1/C8 | 10081.48 | 985.93 | 2634.17 | 33.56 | 35.82 | 48.81 | 217.61 | 0.00 |
| fp8 | passage_retrieval_en | r1/C8 | 12923.12 | 483.10 | 3974.52 | 56.79 | 31.08 | 52.18 | 123.19 | 0.00 |
| tq4 | qasper | r1/C8 | 5200.42 | 437.86 | 1375.22 | 47.92 | 22.20 | 62.26 | 150.26 | 0.00 |
| tq4 | multifieldqa_en | r1/C8 | 7177.52 | 337.89 | 2035.37 | 64.70 | 23.88 | 49.30 | 111.71 | 0.00 |
| tq4 | hotpotqa | r1/C8 | 13454.45 | 581.79 | 4708.15 | 94.06 | 58.44 | 185.67 | 77.37 | 0.00 |
| tq4 | 2wikimqa | r1/C8 | 7554.02 | 536.43 | 2136.33 | 62.75 | 35.35 | 119.88 | 119.01 | 0.00 |
| tq4 | gov_report | r1/C8 | 10081.48 | 958.52 | 2917.61 | 66.52 | 66.81 | 96.84 | 114.14 | 0.00 |
| tq4 | passage_retrieval_en | r1/C8 | 12923.12 | 494.38 | 4206.86 | 93.65 | 50.04 | 82.29 | 78.07 | 0.00 |
| tq34 | qasper | r1/C8 | 5200.42 | 570.04 | 1396.36 | 52.66 | 28.68 | 75.26 | 93.52 | 0.00 |
| tq34 | multifieldqa_en | r1/C8 | 7177.52 | 361.19 | 2187.15 | 71.06 | 28.20 | 50.01 | 101.14 | 0.00 |
| tq34 | hotpotqa | r1/C8 | 13454.45 | 549.15 | 4884.19 | 106.83 | 62.70 | 165.74 | 68.86 | 0.00 |
| tq34 | 2wikimqa | r1/C8 | 7554.02 | 573.20 | 2068.92 | 68.33 | 41.10 | 131.63 | 110.15 | 0.00 |
| tq34 | gov_report | r1/C8 | 10081.48 | 1003.77 | 2975.71 | 74.70 | 78.06 | 110.43 | 101.83 | 0.00 |
| tq34 | passage_retrieval_en | r1/C8 | 12923.12 | 582.25 | 4360.45 | 101.60 | 57.16 | 93.59 | 62.89 | 0.00 |
| tq3 | qasper | r1/C8 | 5200.42 | 450.83 | 1446.26 | 54.40 | 25.91 | 72.50 | 135.39 | 0.00 |
| tq3 | multifieldqa_en | r1/C8 | 7177.52 | 324.41 | 2183.75 | 75.60 | 26.90 | 70.41 | 95.49 | 0.00 |
| tq3 | hotpotqa | r1/C8 | 13454.45 | 601.32 | 4943.83 | 107.95 | 69.05 | 214.01 | 68.57 | 0.00 |
| tq3 | 2wikimqa | r1/C8 | 7554.02 | 648.35 | 2193.43 | 72.87 | 48.18 | 141.75 | 86.78 | 0.00 |
| tq3 | gov_report | r1/C8 | 10081.48 | 957.42 | 3113.14 | 78.42 | 78.22 | 113.48 | 97.06 | 0.00 |
| tq3 | passage_retrieval_en | r1/C8 | 12923.12 | 420.11 | 4468.90 | 112.11 | 51.05 | 93.72 | 65.03 | 0.00 |
| kvarn | qasper | r1/C8 | 5200.42 | 446.27 | 2423.20 | 49.73 | 24.51 | 65.19 | 143.40 | 0.00 |
| kvarn | multifieldqa_en | r1/C8 | 7177.52 | 333.38 | 3279.63 | 62.13 | 24.35 | 51.31 | 108.35 | 0.00 |
| kvarn | hotpotqa | r1/C8 | 13454.45 | 688.90 | 5597.61 | 75.26 | 56.71 | 177.12 | 93.34 | 0.00 |
| kvarn | 2wikimqa | r1/C8 | 7554.02 | 521.06 | 2586.87 | 52.35 | 29.47 | 94.97 | 139.52 | 0.00 |
| kvarn | gov_report | r1/C8 | 10081.48 | 992.56 | 3678.91 | 50.11 | 53.45 | 74.80 | 147.07 | 0.00 |
| kvarn | passage_retrieval_en | r1/C8 | 12923.12 | 636.50 | 4587.19 | 70.94 | 48.33 | 68.58 | 97.77 | 0.00 |
| kvarn44 | qasper | r1/C8 | 5200.42 | 441.69 | 2395.29 | 49.60 | 24.36 | 65.94 | 142.95 | 0.00 |
| kvarn44 | multifieldqa_en | r1/C8 | 7177.52 | 337.28 | 2939.93 | 63.47 | 24.63 | 61.53 | 108.48 | 0.00 |
| kvarn44 | hotpotqa | r1/C8 | 13454.45 | 791.61 | 5355.31 | 75.36 | 62.46 | 162.14 | 93.26 | 0.00 |
| kvarn44 | 2wikimqa | r1/C8 | 7554.02 | 521.39 | 2481.48 | 53.28 | 29.66 | 83.37 | 136.41 | 0.00 |
| kvarn44 | gov_report | r1/C8 | 10081.48 | 976.80 | 3416.16 | 50.76 | 52.97 | 69.42 | 146.40 | 0.00 |
| kvarn44 | passage_retrieval_en | r1/C8 | 12923.12 | 505.85 | 4522.65 | 76.72 | 42.91 | 68.49 | 93.24 | 0.00 |
| kvarn64 | qasper | r1/C8 | 5200.42 | 499.46 | 2396.71 | 48.46 | 26.38 | 67.83 | 148.53 | 0.00 |
| kvarn64 | multifieldqa_en | r1/C8 | 7177.52 | 336.60 | 2837.43 | 64.39 | 24.59 | 58.38 | 108.34 | 0.00 |
| kvarn64 | hotpotqa | r1/C8 | 13454.45 | 577.11 | 5086.07 | 79.06 | 49.22 | 153.19 | 92.13 | 0.00 |
| kvarn64 | 2wikimqa | r1/C8 | 7554.02 | 537.05 | 2759.41 | 53.16 | 30.74 | 72.36 | 133.71 | 0.00 |
| kvarn64 | gov_report | r1/C8 | 10081.48 | 1002.16 | 3592.19 | 51.88 | 55.62 | 75.39 | 142.80 | 0.00 |
| kvarn64 | passage_retrieval_en | r1/C8 | 12923.12 | 505.64 | 4591.69 | 76.48 | 42.51 | 67.75 | 94.03 | 0.00 |
| kvarn4464 | qasper | r1/C8 | 5200.42 | 741.80 | 2307.95 | 48.06 | 32.96 | 83.77 | 115.77 | 0.00 |
| kvarn4464 | multifieldqa_en | r1/C8 | 7177.52 | 335.25 | 2672.62 | 62.68 | 23.29 | 48.87 | 111.32 | 0.00 |
| kvarn4464 | hotpotqa | r1/C8 | 13454.45 | 763.43 | 5448.21 | 74.17 | 62.34 | 183.87 | 95.17 | 0.00 |
| kvarn4464 | 2wikimqa | r1/C8 | 7554.02 | 504.13 | 2627.39 | 54.80 | 29.49 | 87.63 | 128.73 | 0.00 |
| kvarn4464 | gov_report | r1/C8 | 10081.48 | 961.74 | 3357.12 | 52.24 | 53.45 | 75.17 | 142.76 | 0.00 |
| kvarn4464 | passage_retrieval_en | r1/C8 | 12923.12 | 507.68 | 4586.38 | 76.01 | 43.00 | 67.27 | 93.68 | 0.00 |
| ar_k4v4_parent | qasper | r1/C8 | 5200.42 | 427.43 | 2728.26 | 51.62 | 24.64 | 61.36 | 136.27 | 0.00 |
| ar_k4v4_parent | multifieldqa_en | r1/C8 | 7177.52 | 342.31 | 2855.35 | 62.89 | 24.55 | 55.62 | 110.41 | 0.00 |
| ar_k4v4_parent | hotpotqa | r1/C8 | 13454.45 | 724.98 | 5281.35 | 81.23 | 59.02 | 165.18 | 88.17 | 0.00 |
| ar_k4v4_parent | 2wikimqa | r1/C8 | 7554.02 | 618.42 | 2580.70 | 52.32 | 32.87 | 92.55 | 121.59 | 0.00 |
| ar_k4v4_parent | gov_report | r1/C8 | 10081.48 | 967.84 | 3363.88 | 51.06 | 52.71 | 70.93 | 145.50 | 0.00 |
| ar_k4v4_parent | passage_retrieval_en | r1/C8 | 12923.12 | 504.18 | 4508.89 | 77.57 | 43.00 | 68.66 | 92.87 | 0.00 |
| ar_k4v4_repacked_full | qasper | r1/C8 | 5200.42 | 568.50 | 3142.44 | 59.95 | 32.57 | 78.67 | 94.42 | 0.00 |
| ar_k4v4_repacked_full | multifieldqa_en | r1/C8 | 7177.52 | 350.09 | 3031.94 | 73.09 | 28.27 | 64.20 | 96.15 | 0.00 |
| ar_k4v4_repacked_full | hotpotqa | r1/C8 | 13454.45 | 551.40 | 5852.65 | 94.76 | 57.14 | 169.95 | 75.62 | 0.00 |
| ar_k4v4_repacked_full | 2wikimqa | r1/C8 | 7554.02 | 651.22 | 2982.37 | 62.07 | 38.10 | 94.42 | 98.09 | 0.00 |
| ar_k4v4_repacked_full | gov_report | r1/C8 | 10081.48 | 986.60 | 3826.99 | 61.11 | 63.95 | 87.91 | 122.38 | 0.00 |
| ar_k4v4_repacked_full | passage_retrieval_en | r1/C8 | 12923.12 | 492.07 | 5039.82 | 89.69 | 49.00 | 80.29 | 79.57 | 0.00 |
| ar_k4v4_anchor | qasper | r1/C8 | 5200.42 | 455.51 | 2804.61 | 53.64 | 27.10 | 83.66 | 133.26 | 0.00 |
| ar_k4v4_anchor | multifieldqa_en | r1/C8 | 7177.52 | 355.61 | 3129.83 | 67.87 | 26.80 | 57.93 | 102.24 | 0.00 |
| ar_k4v4_anchor | hotpotqa | r1/C8 | 13454.45 | 636.45 | 5451.03 | 76.64 | 53.86 | 156.05 | 92.29 | 0.00 |
| ar_k4v4_anchor | 2wikimqa | r1/C8 | 7554.02 | 639.49 | 2662.81 | 51.54 | 35.20 | 134.80 | 140.99 | 0.00 |
| ar_k4v4_anchor | gov_report | r1/C8 | 10081.48 | 977.61 | 4576.58 | 54.50 | 57.91 | 86.85 | 133.91 | 0.00 |
| ar_k4v4_anchor | passage_retrieval_en | r1/C8 | 12923.12 | 617.82 | 5083.87 | 78.67 | 51.59 | 73.91 | 88.85 | 0.00 |
| ar_k4v4_bound_k | qasper | r1/C8 | 5200.42 | 448.56 | 2918.09 | 66.96 | 32.92 | 88.50 | 107.91 | 0.00 |
| ar_k4v4_bound_k | multifieldqa_en | r1/C8 | 7177.52 | 363.31 | 3516.54 | 78.53 | 32.90 | 96.95 | 87.63 | 0.00 |
| ar_k4v4_bound_k | hotpotqa | r1/C8 | 13454.45 | 640.21 | 5922.12 | 106.34 | 71.84 | 217.09 | 66.29 | 0.00 |
| ar_k4v4_bound_k | 2wikimqa | r1/C8 | 7554.02 | 538.62 | 2843.30 | 69.97 | 39.45 | 135.76 | 106.07 | 0.00 |
| ar_k4v4_bound_k | gov_report | r1/C8 | 10081.48 | 972.08 | 4035.66 | 72.45 | 74.45 | 101.66 | 103.54 | 0.00 |
| ar_k4v4_bound_k | passage_retrieval_en | r1/C8 | 12923.12 | 500.74 | 5235.76 | 101.90 | 55.54 | 90.14 | 71.37 | 0.00 |
| ar_k4v4_bound_v | qasper | r1/C8 | 5200.42 | 456.97 | 2941.37 | 62.64 | 31.75 | 79.42 | 113.63 | 0.00 |
| ar_k4v4_bound_v | multifieldqa_en | r1/C8 | 7177.52 | 332.83 | 2996.90 | 78.27 | 28.88 | 56.05 | 89.28 | 0.00 |
| ar_k4v4_bound_v | hotpotqa | r1/C8 | 13454.45 | 549.53 | 6026.38 | 99.27 | 59.34 | 176.53 | 73.17 | 0.00 |
| ar_k4v4_bound_v | 2wikimqa | r1/C8 | 7554.02 | 531.58 | 2733.35 | 65.43 | 36.55 | 113.25 | 111.37 | 0.00 |
| ar_k4v4_bound_v | gov_report | r1/C8 | 10081.48 | 976.88 | 3987.08 | 66.41 | 68.92 | 99.72 | 112.28 | 0.00 |
| ar_k4v4_bound_v | passage_retrieval_en | r1/C8 | 12923.12 | 507.79 | 5363.45 | 93.47 | 52.59 | 85.75 | 76.72 | 0.00 |
| ar_k4v4_bound_kv | qasper | r1/C8 | 5200.42 | 439.56 | 3002.98 | 67.32 | 32.60 | 88.36 | 105.90 | 0.00 |
| ar_k4v4_bound_kv | multifieldqa_en | r1/C8 | 7177.52 | 371.34 | 3629.40 | 76.58 | 32.87 | 79.87 | 89.23 | 0.00 |
| ar_k4v4_bound_kv | hotpotqa | r1/C8 | 13454.45 | 670.87 | 5802.97 | 105.52 | 72.02 | 207.58 | 65.00 | 0.00 |
| ar_k4v4_bound_kv | 2wikimqa | r1/C8 | 7554.02 | 599.84 | 2860.36 | 73.96 | 45.34 | 116.98 | 88.37 | 0.00 |
| ar_k4v4_bound_kv | gov_report | r1/C8 | 10081.48 | 972.11 | 4260.78 | 74.97 | 77.13 | 105.59 | 100.05 | 0.00 |
| ar_k4v4_bound_kv | passage_retrieval_en | r1/C8 | 12923.12 | 496.61 | 5239.93 | 103.53 | 56.06 | 90.40 | 70.24 | 0.00 |

思考与最终答案的全部生成 token 计入输出和 TPOT；TTFT 是首个 reasoning/content 片段，非最终答案首 token 时间。

截断完整响应仍评分，不保证为0；思考未结束且无最终答案按空答案计分。基础设施失败不会伪装成完整0分。

预算一致为否表示原生usage与推导预算存在差异，例如length早于剩余上下文边界。提前length的完整响应仍评分；应核对服务端预算和token计数，不能仅凭该标记断定有隐藏cap。

测量不完整时，表中性能可能来自已收到的部分响应，不能当作完整主对照。质量分仅在评分完整时展示。

不计算跨任务总体准确率，不自动按分数或吞吐判定方法优胜。少量 smoke 样本的p95只作流程诊断。

## 来源与协议

- `/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/runs/fp16`：method=fp16；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/runs/fp8`：method=fp8；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/runs/tq4`：method=tq4；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/runs/tq34`：method=tq34；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/runs/tq3`：method=tq3；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/runs/kvarn`：method=kvarn；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/runs/kvarn44`：method=kvarn44；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/runs/kvarn64`：method=kvarn64；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/runs/kvarn4464`：method=kvarn4464；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/residual_v1/full_20261010_160834_3993496/runs/ar_k4v4_parent`：method=ar_k4v4_parent；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/residual_v1/full_20261010_160834_3993496/runs/ar_k4v4_repacked_full`：method=ar_k4v4_repacked_full；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/residual_v1/full_20261010_160834_3993496/runs/ar_k4v4_anchor`：method=ar_k4v4_anchor；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/residual_v1/full_20261010_160834_3993496/runs/ar_k4v4_bound_k`：method=ar_k4v4_bound_k；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/residual_v1/full_20261010_160834_3993496/runs/ar_k4v4_bound_v`：method=ar_k4v4_bound_v；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
- `/home/wanglin/kvbench-3090-cu126/results/residual_v1/full_20261010_160834_3993496/runs/ar_k4v4_bound_kv`：method=ar_k4v4_bound_kv；data_manifest_sha256=609a9bcf70f1bf7156f0e9370b09d1ae32309023e38f883f8d9f2713d1e2c522；settings=`{"generation_budget_policy": "remaining-context", "max_tokens_transport": "explicit-json-null", "min_p": 0.0, "request_seed_base": 20261009, "request_timeout": 0.0, "retries": 0, "temperature": 0.6, "thinking": true, "top_k": 20, "top_p": 0.95, "warmup": 1, "warmup_output_cap": 32, "warmup_per_point": -1}`
