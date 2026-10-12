# 验证记录与运行门槛

## 本次交付环境实际完成

- 整包 34 项 CPU 单元测试通过，包含位平面字节恢复、真实数值误差预算、
  GQA 读取并集、零预算全读、source pin 与分支位置检查、runtime 隔离、
  子进程/worker 路由、结果缺失保留 NA 和 profiler receipt 验证。
- 默认 epsilon=0.02 的 96 项 float64 验证通过。
- epsilon=0 的 96 项 float64 验证通过，确认零预算不要求必有精度省略。
- 所有生成的 Python 源均能被 AST 解析；未导入 torch 即可校验和生成 runtime。
- 六个候选 runtime 的源变换由精确 SHA256 pin 约束，未知 writer/reader 拒绝运行。

交付环境没有 CUDA、PyTorch、Triton，**未执行 GPU 编译、真实模型推理、
吞吐或显存流量测量**。CPU-only 的 validation.json 明确为
`status=cpu_only, passed=false, gpu_executed=false`；它不能让全量任务越过GPU门槛。

## 在用户服务器执行的 GPU 验证

全量脚本先自动完成以下检查，任何失败都会阻止六组正式推理：

1. 真实 Torch tile packer 与 batched RTN writer：full planes 恢复相同 INT4 码，
   原有 scale/zero 字节保持一致，g64/g128 均验证。
2. 全量读取兼容性：prefill build_packed_kv、旧 standalone dequant、
   dedicated multi-query verify 在所有六策略下恢复一致的完整缓存。
3. 单核和 split-K decode，BLOCK_N=16/32/64，非连续分页、sink/tail、
   滑窗、空 split、padding、间接 verify、Qwen3 的 32 query/8 KV heads。
4. 六策略都实际运行原 Triton autotuner，核对其最终 chunk 配置。
5. 六策略在实际声明 epsilon 下执行 CUDA graph replay；随后原地更新 query，
   输出与对应 float64 oracle 比较。
6. 另以固定 epsilon=0.02 的、明确标记的 synthetic dominant-chunk fixture
   检查图重放中 query 改变能实际改变 K/V Residual demand 分支。
   该诊断的生成源哈希与预算单独记录，不能冒充正式 epsilon 的速度结果。

GPU数值误差限默认：对各自 float64 policy 输出，整体 relative L2≤0.02
且最大元素绝对误差≤0.02；parent 与 repacked_full 另有更紧的相互检查。
这些是实现检查门槛，不是模型准确率的非劣性标准。

对有界策略，GPU结果还保存逐 query head 的“相对 GPU parent 的 L2误差”、
“epsilon×V_ref”、“不加浮点余量的严格超额值”，以及两个实现各自相对
float64 的数值偏差。经验数值偏差没有被当作数学定理的容许误差。

## 性能与流量

默认 microbench 含 120 条固定形状 attention 计时：
B∈{1,8}，L∈{1024,4096,8192,16384,32768}，single/split，两种模式各六策略。
同源 full-plane 控制组用于分离 repack/表示开销与选择精度的影响。
另独立计时原 INT4 packer 和新位平面 packer；完整写时成本仍包含在正式服务里。

计时和统计分开。logical code bytes 是GPU真实执行的分支所对应的逻辑地址
字节，**不是** DRAM 事务。Nsight Compute 入口在独立固定工作负载上测量
真实 `dram__bytes_read.sum` 等指标；工具或权限不可用时保留 NA。
没有因为 kernel 变慢或残差跳过比例小而删除某个策略。

## 仍需真实实验回答

保守界可能使大部分请求退化为读取全部残差，特别是 split 首部或平坦注意力。
若选择开销超过节省的读取成本，该方案可能不提速甚至回退。synthetic
dominant fixture 用于检查分支正确性，不代表真实模型的注意力统计。
六任务分数、TPOT、吞吐、输出长度、截断率与preemption必须共同解释；
首轮每组一次用于筛查，不能代替多次运行或多种种子的稳定性论证。
