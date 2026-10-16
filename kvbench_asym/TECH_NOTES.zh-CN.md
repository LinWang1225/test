# 当前 Anchor+Residual 的源码接入、页内布局与 ReadKV 定位

## 1. “行为改变”和“编辑安装文件”是两个维度

改变行为意味着：生成进程实际调用新的 writer、reader、反量化、attention 或选择内核。改变安装文件意味着：把这些实现写进原 site-packages/vllm 或原 KVarN checkout。前者是方法运行所需的实现变化；后者只是部署方式之一。

三种部署路线：

| 路线 | 实际执行代码改变 | 原安装目录改变 |
|---|---|---|
| 直接修改已安装文件/原分叉源码 | 是 | 是 |
| 私有 runtime/overlay + bootstrap 导入 | 是，仅本次服务 | 否 |
| 独立插件注册自定义 backend、内核 | 是，选用插件的服务 | 否 |

本项目目前的 AR v1 是第二种：`kvbench_residual.runtime.build_runtime`复制私有 Python runtime，对固定哈希的 writer/readers 做变换；bootstrap/PYTHONPATH 使服务和子 worker 导入私有副本。原来的安装目录不被覆盖。已编译的底层库通过受检查的符号链接复用。

原环境本身已经是 KVarN 的 vLLM 分叉；不是纯官方 pip vLLM 已经原生支持这一 INT4 表示。AR v1 也不是已经完成的标准第三方插件。

独立插件可以注册新的 attention backend，由该 backend 调用包内的 writer 和 Triton 内核。它仍须保证所有缓存路径一致，也要适配 cache dtype 校验、缓存规格、分配和 worker 初始化；并非挂一个 Python 回调即可改变任意内存格式。若新功能超出既有扩展接口，可能还需小范围核心适配。Triton 内核可独立 JIT；使用新的 C++/CUDA extension 时需编译对应扩展，但不等于必须重编整个 vLLM。

仅在原 attention 输出上后处理、或者已经读取全 KV 后再做 mask，并没有减少 KV 从显存的访问。

## 2. page 与 block table

这里的 page 是 vLLM 管理的逻辑 token block，不是操作系统的 4 KB 页。

当前范围下，一个逻辑 block 包含 128 个 token。`block_table[request, logical_block]` 给出该逻辑 block 对应的物理 KV 缓存块编号；同一请求的逻辑块可以映射到非连续物理块。

保持 page 大小不变，是指 token 数、每层每块的记录字节数、整体外部张量约定不变。保持 block table 不变，是指逻辑块到物理块编号的映射语义不变。保持生命周期不变，是指仍按原 KVarN/vLLM 的分配、写入、FP16 tail、量化 flush、复用读取、请求结束释放规则管理；Anchor/Residual 不引入两套独立的分配、驱逐或 offload。

只修改页内表示，相当于同一个物理块里有相同大小的记录，但字段内部的 bit 打包顺序不同。地址映射不需要知道这些 bit 的语义，读取/写入内核需要知道。

## 3. 当前 K4V4/G128/head_dim128 的真实记录

以下是一个 `(物理block, KV head)` tile；外部张量为 `[num_blocks, num_kv_heads, tile_bytes] uint8`。

| 字段 | 字节偏移（含端点） | 长度 |
|---|---:|---:|
| K high2 位平面 | 0–4095 | 4096 B |
| K low2 位平面 | 4096–8191 | 4096 B |
| K s_col | 8192–8447 | 256 B |
| K zp | 8448–8703 | 256 B |
| K s_row | 8704–8959 | 256 B |
| V high2 位平面 | 8960–13055 | 4096 B |
| V low2 位平面 | 13056–17151 | 4096 B |
| V s_col | 17152–17407 | 256 B |
| V s_row | 17408–17663 | 256 B |
| V zp | 17664–17919 | 256 B |

每 head tile 总计 17,920 B。8 个 KV heads 在一层的一个 block 合计 143,360 B（140 KiB）；不包含 FP16 pool、临时缓冲或其他运行时开销。

原生 INT4 在 K/V 码字段内每 byte 放两个 4-bit code。AR 先取同一原始码 c 的高两位 h、低两位 l；每 byte 打包四个 2-bit 值，每个字段先连续放完 high2，再连续放完 low2。它不是把 high/low 在每一行里交错存放。

```text
c = 4*h + l
Anchor 码 = 4*h + 1.5
Residual 码 = l - 1.5
```

实际存储的是 h/l 的整数位平面，不是另存两份 FP16 Anchor/Residual 张量。

对 K：`K = (c*s_col_K + zp_K)*s_row_K`。
对 V：`V = (c*s_row_V + zp_V)*s_col_V`。

只读 high2 时，使用 Anchor 码；读取 low2 后先恢复原整数 c，再执行原反量化表达式。理论上恢复同一完整量化表示，不额外做一次残差再量化。

K/V high2 都始终参加 attention。未读取的 low2 仍保存在该页内，以后其他 Query 可以再读取。FP16 sink/tail 保持原有存储。

所以该表示与 Parent 都是完整 K4V4：它改变的是每次读取量，不是把存储容量再压缩一半。可用 num_gpu_blocks 仍可能受临时张量、图捕获和运行时显存测量影响，应以服务记录为准。

## 4. 三轮实验使用布局的方式不同

| 实验 | 生成服务中的缓存 | 选择和补读位置 |
|---|---|---|
| AR v1 六组正式实验 | Parent 为原格式；其余策略为 high2/low2 新布局 | 非 Parent 的实际 GPU attention 内核 |
| 上一轮真实 trace 诊断 | 原生 Parent | 采集后独立 GPU 重放；验证 fixture 可同时持有 native/planar 两份 |
| 本次等载荷全样本实验 | 原生 Parent | CPU float64 反事实，保存固定预算掩码；不部署新非对称 GPU 内核 |

验证/诊断里临时产生两种格式，不代表正式 AR v1 维护双份 shadow cache。新补丁的 mask 数学选择等载荷，也不代表 Parent 生成服务已经真的按这些掩码跳过 GPU load。

## 5. 与 ReadKV 的关系

ReadKV 已经讨论存储/读取预算解耦、渐进量化、GQA 和 K/V 不同分配方式。其编码为渐进 Gaussian Lloyd tree，K 以 channel 为精度分配单位，V 以 token 为单位；当前 AR 是既有 INT4 的两级 2+2 表示，以 chunk 补读。

ReadKV 的 K/V split 作为 allocator 输入，理论分配最优性针对给定每阶段预算下的代理失真，不等于全模型任务最优。附录 F.8 讨论了 GPU reader 强制等 K/V 读率的质量代价。

因此“发现非对称可能更好”是本项目的验证与决策依据，但不足以单独作为相对 ReadKV 的新贡献。更有意义的后续目标是：在原生低比特缓存、分页/GQA 服务约束下，用低开销的补读和融合内核，取得同一在线策略的质量、实际流量和端到端延迟收益。

ReadKV GPU 结果与灵活质量 reader 的约束不同；也不应把其单层加速直接等同 vLLM 服务 TPOT。当前 AR 尚未超过 Parent，所以也不能宣称实践效果已胜过 ReadKV。

参考来源见 `SOURCES.md`。本次补丁没有实现 ReadKV 基线。
