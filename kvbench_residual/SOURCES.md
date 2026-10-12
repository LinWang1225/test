# 来源、版本与交付范围

## 实验基础

- 现有结果固定于 [LinWang1225/test@be60b3a413687a75ce8e8928a1b222ac4e0ed46d](https://github.com/LinWang1225/test/tree/be60b3a413687a75ce8e8928a1b222ac4e0ed46d)。
- 全量入口复用用户原机器的成功批次 `full_longbench_C8_20261009_201156_2217646`，普通 LongBench 六任务，每方法 1,147 题。
- 依赖已经应用并跑通的 LongContext V3；新增包 `kvbench_residual/` 不需要 AFA 补丁，也不包含其代码目录。
- 交付仅增加实验扩展。没有把当前新方法的实验写入历史结果，没有启动用户 GPU，没有修改 GitHub 远程仓库。

## KVarN 后端

来源：[huawei-csl/KVarN@7586257f1c632e63187bfacbbe21ccb51540f7b3](https://github.com/huawei-csl/KVarN/tree/7586257f1c632e63187bfacbbe21ccb51540f7b3)。

以原 vLLM 包目录为根，固定六个源码文件。SHA256 由运行时再次验证，未知版本不执行源码变换。

| 文件 | SHA256 |
|---|---|
| `v1/attention/ops/triton_kvarn_decode.py` | `dbc73619e34f564438343bc13f6ca07d9ad2980a016a3fe23230e0c29e9d130a` |
| `v1/attention/ops/kvarn_store.py` | `7bc98ba130ad84c6e9bf482d03fd9faf3d6595ce55b7a0b6ecf09f40ccf0ed5d` |
| `v1/attention/ops/kvarn_decode.py` | `cfb2bf1a70883b0df6d4fbdde13c82afeb98415bd54bf8a2aad0b7425f355653` |
| `v1/attention/backends/kvarn_attn.py` | `1d31d9839fb57da096c8f8a1f42a485470c0b1b4f9990059adbdd7e376450575` |
| `model_executor/layers/quantization/kvarn/config.py` | `ece82a7721276a5c11fc3c56fb5d88dda39580f310cdff32bfa1aee857e7b7d3` |
| `v1/attention/ops/triton_kvarn_sinkhorn.py` | `122e7b2dc206413877b0d622ce3892ec65d3ec27b00625a02d9726bf1832b8a3` |

运行时创建私有 Python 覆盖目录，保留原二进制来源并记录链接、大小和 mtime。原始安装源码不被写入。
六个策略都包含同一组空 mask / 空 split / padding 保护，以及旧 dequant helper 的位宽默认值修正。
因此同批新 `parent` 是主因果控制组，历史九方法只作已有结果参考。

## 研究近邻与创新边界

- [SpAtten](https://arxiv.org/abs/2012.09852)：已经提出渐进式 MSB/LSB 读取。
- [Faster Than Flash / FFD](https://arxiv.org/abs/2609.00097v1)：低比特扫描与稀疏 attention 融合。
- [Runtime-Certified Bounded-Error Quantized Attention](https://arxiv.org/abs/2605.20868v1)：局部 K/V 输出误差分解与精度回退。
- [Fathom](https://arxiv.org/abs/2609.17652v3)：当前 query 决定位平面读取深度，主要针对 offloaded K 索引。
- [AnchorKV](https://arxiv.org/abs/2608.02901v1)：共享 anchor 向量和 prefill 时残差存储分配。

本补丁研究完整量化码在 GPU 驻留时的 K/V 精度按需读取，所有 token 的 Anchor 保留，采用单遍 prefix 下界和分块预算。
不宣称“Anchor+Residual”“按 query 选择精度”或 attention 误差上界本身从未出现。
方法细节、数学适用范围和 GPU 未验证事项见 `TECH_CONTRACT.md`、`VALIDATION.md`。

可选硬件性能计数接口依据 [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。
`cache-control=all` 与 `clock-control=none` 在结果中固定记录。逻辑码字节不能替代真实 DRAM 指标。

## 完整性

`SHA256SUMS` 列出包内交付文件（不含该清单自身）的 SHA256。应用补丁后可从项目根运行：

```bash
sha256sum -c kvbench_residual/SHA256SUMS
```

这只验证本地文件完整性，不替代运行时来源检查、GPU 数值门槛或正式实验。
