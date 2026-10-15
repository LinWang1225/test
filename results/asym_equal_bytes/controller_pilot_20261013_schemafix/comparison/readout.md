# 等逻辑载荷 K/V 非对称对照

完成请求：3 / 3；真实 trace：12。

先检查 inputs/coverage.csv 的全部子任务、原始题数和截断数；再检查 case_coverage.csv 的自然 EOS/缺失采集点。
case_level.csv 先在每道题内平均层/head/位置；aggregate.csv 再在任务内对题等权统计。
paired_vs_symmetric.csv：误差差值 < 0 表示比同流量对称分配更好。95% bootstrap 按请求重采样；是探索性区间，未作多重比较校正。
mask_seed 是随机读取掩码种子，不是三个独立的生成随机种子。
无可控量化历史的短 trace 不作为零误差成功样本进入非对称比较。
K priority=0 为 V 优先，0.5 为对称，1 为 K 优先；整数取整可能使对称组相差一个 packet。
所有配对组均校验相同逻辑码字节；这不等于 DRAM 事务相同，不含排序/中间张量流量，也不报告 TPOT。
这是 Parent 轨迹上的 float64 注意力反事实；没有把选择策略写入生成过程，不提供任务准确率。
controller_case_level.csv/controller_aggregate.csv 记录 Query-aware 估计、真实误差、误停率和 rho(q)。
误差估计是 Anchor+metadata 的离线代理；underread_rate 以真实 Full-KV 误差超过 tolerance 定义。
