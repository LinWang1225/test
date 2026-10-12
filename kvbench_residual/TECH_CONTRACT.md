# Anchor–Residual v1 实现与数学边界

## 表示与存储

主实验仅支持 `kvarn_k4v4_g128`，head_dim=128，Qwen3 的 GQA 路径。
每个原 INT4 码 c∈{0,…,15} 写成 c=4h+l，h,l∈{0,…,3}。
Anchor 码为 4h+1.5，Residual 码为 l−1.5。
每个 K/V tile 的全部 high2 字节连续存储在字段前半部，全部 low2
字节在后半部。每个 tile 的码/参数字节及理论 tokens-per-byte 不变。
vLLM 实际分配的 num_gpu_blocks 和可用容量仍以 launch/metrics 为准；
packer 临时张量、CUDA graph 内存及运行时 profiling 可能改变分配结果。
读取全部 Residual 后先恢复整数 c，再执行原重建表达式；没有二次量化。
写时原量化器产生 c 后使用独立位平面 packer，额外打包开销属于方法成本。
没有第二份原缓存、shadow cache、host offload 或误差摘要 sidecar。

## 每个 decode chunk 的选择

chunk 大小由 parent Triton autotune 的 BLOCK_N∈{16,32,64} 决定；
group=128 是量化/分页粒度。所有 token 的 K/V Anchor 均进入最终 attention。
完整 fp16 sink/tail 不降精度。只在块内 quantized chunk 上判断读不读低位。
同一 KV head 的所有真实 query heads 都通过才跳过（即读取需求的并集）。

K 参考重建为 K_dt=(c_dt s_d+z_d)r_t。当前 q 的 logit 残差界：

    ε_b(q)=1.5 |attention_scale| max_t |r_t| ∑_d |q_d s_d|.

不需要读取 Residual 本体，全部摘要来自原已有 scale。
记 chunk 的 Anchor logsumexp 为 L_b，当前 split 已处理前缀的真实/近似
logits 下界质量为 Z_lower。则任意真实与近似 K 连线上的 chunk 总概率有上界：

    M_b ≤ min(1, exp(L_b+ε_b)/(Z_lower+exp(L_b−ε_b))).

每个 chunk 处理后把实际使用 logits 减去其尚未补读的 K 误差界，
对应的 exp 和计入前缀下界。pool 与全读 K 的误差为 0。
每个 split 独立从 0 前缀开始，因此它最初的界更松，但仍是全序列分母下界。
不依赖上一 decode token 的选择，不读取未来 chunk，不二扫完整 Anchor。

对该 query 实际注意的 L 个 token，chunk 内合法 token 数 n_b：

    K 跳过条件：2 M_b ε_b ≤ τ_K n_b/L。

V 参考为 V_td=(c_td u_t+z_t)v_d，定义该 chunk 的 scale 范数界：

    B_b=||v||₂ max_t max(|z_t|, |z_t+15u_t|),
    η_b=1.5 ||v||₂ max_t|u_t|。

V 判断发生在当前 chunk 的 K 决策之后，使用更新后的 M_b：

    V 跳过条件：M_b η_b/B_b ≤ τ_V n_b/L。

比例中 ||v||₂ 可约去。B_b=0 且完整 V=0 时额外误差为 0。
令 V_ref 为全序列所有 quantized chunk 的 B_b 与 fp16 pool token
的 V 向量范数的最大值。在实数运算与有效输入假设下：

    ||O_complete_quantized_rot − O_selective_rot||₂ ≤ (τ_K+τ_V) V_ref。

推导：softmax 关于 logits 的导数给 K 项 ≤2 V_ref ∑ M_b ε_b；
V 项 ≤∑ M_b η_b≤V_ref ∑M_bη_b/B_b；所有 chunk 的 n_b/L 总和≤1。
未来 chunk 只会增加真实分母，因此使用本 split 的前缀下界仍有效。
标准滑窗按其合法 token 数分配预算，未计入注意范围的 token 不消耗预算。

## 不能扩大的结论

这里 O_rot 是 Hadamard 旋转坐标系中的 attention 输出、尚未做输出舍入
和逆旋转。实数逆旋转后的界需乘以实际 H16 矩阵的算子范数；不能将
fp16 Hadamard 自动当成精确正交矩阵。GPU舍入仍属于下述额外数值问题。
预算相对的是 scale 推导的保守 V_ref，**不是输出向量相对误差**；
也不是 FP16 模型误差、生成序列不变或任务准确率不变保证。
此 CPU 实数推导没有涵盖 GPU 的 TF32、fp16、FMA、exp 和归约舍入。
实现给局部界加很小余量，并要求服务器 GPU 数值验证，但不称为
带定向舍入的形式化浮点认证。全量任务还须实际评测质量。
τ=.02 是显式实验参数，bound_k 独占 τ_K，bound_v 独占 τ_V，
bound_kv 两者各 .01。anchor 是无误差预算约束的激进消融。

## 路径覆盖

修改 KVarN torch pack/unpack 和全部五个 Triton packed-cache readers：
dequant_blocks、build_packed_kv、fused_decode、decode_stage1、fused_verify_stage1。
标准单 token decode 及共用该核的 indirect verify 实施选择读取；
prefill/大批多 query materialization 和 dedicated verify 读取全部位平面。
公共控制还为旧 standalone dequant helper 补上原函数遗漏的 K_BITS/V_BITS
constexpr 默认值（均为4），避免其直接测试入口引用未定义名字。
该范围写入 provenance，不把 prefill/full-read fallback 宣称为选择读取。
只在新的独立 vLLM runtime 中改变 writer/reader，旧服务不被改动。

## 观测范围

正式核不带每 token 统计。独立 microbench 的额外 instrumentation
记录 GPU 实际执行分支的 logical code bytes（含 GQA 读取并集），不计为
DRAM/HBM 计数。真实显存流量须用 Nsight Compute 的 dram 指标另测。
量化码读取减少不能直接推导 TPOT 提升；Anchor 扫描、补读后的 QK 重算、
尺度读取、误差计算、动态分支和写时 repack 都须计入实验时间。
