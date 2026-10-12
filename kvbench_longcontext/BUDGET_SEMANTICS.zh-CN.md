# v3 的“无额外输出上限”如何实现

本版对齐已有短输入测量中的显式 JSON max_tokens:null，并固定服务生成配置。不是改成另一个大整数，也不强制生成到窗口末尾。

## 1. 默认协议

prepare 默认 thinking=true、generation-budget-policy=remaining-context。正式 chat 和 raw completion 请求均保留 max_tokens 键且值为 JSON null，不发送 max_completion_tokens。服务固定 --generation-config vllm，generation override 为空。

采样为 temperature=0.6、top_p=0.95、top_k=20、min_p=0。EOS/stop正常生效。request-timeout=0同时取消生成请求的整请求elapsed截止与阻塞读取超时。健康检查和独立预热不计入测量；预热短预算不会传给正式请求。

在已核验的后端中，硬生成预算等于“实际模型窗口−实际原生输入token数”。

| 含模板输入token | 窗口 | 剩余生成空间 | 正式max_tokens |
|---|---|---|---|
| 1024 | 32768 | 31744 | null |
| 8192 | 32768 | 24576 | null |
| 24000 | 32768 | 8768 | null |
| 32767 | 32768 | 1 | null |

以上是预算示例，不是真实模型结果。生成仍可自然提前结束。输入占满窗口时拒绝或按显式skip排除，不会偷偷扩窗或裁输入。

## 2. 为什么必须同时核验服务配置

固定核验后端为 [huawei-csl/KVarN commit 7586257](https://github.com/huawei-csl/KVarN/tree/7586257f1c632e63187bfacbbe21ccb51540f7b3)。

[get_max_tokens](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/entrypoints/serve/utils/api_utils.py) 将非空的剩余上下文、请求或默认生成限制、服务override限制、平台限制取最小。因此不能对任意服务泛称“null一定只受上下文约束”。

[模型配置逻辑](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/config/model.py) 在generation_config=vllm时不载入模型sampling defaults；空override不引入max_new_tokens。已核验的 [CUDA平台](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/platforms/cuda.py) 沿用 [平台接口](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/platforms/interface.py) 的极大输出限制，不比32768窗口更小。模型EOS等结束token仍会工作。

启动器和runner保存并核验窗口、生成配置、模型身份。源码身份被记录不等于未来任何版本都自动得到验证；升级后端后需重新检查语义。

## 3. null与省略字段为什么不同

[chat协议](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/entrypoints/openai/chat_completion/protocol.py) 的两个长度字段默认None，但非空max_completion_tokens优先。

[raw completion协议](https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/entrypoints/openai/completion/protocol.py) 省略max_tokens默认16，显式null才得到None。因此LCC原始补全也必须发null。serving将解析后的具体预算传给采样参数，不会再被采样参数类的默认16覆盖。

已有短输入 [V7实际请求](https://github.com/LinWang1225/test/blob/7801b642d03a27aca0486341f4cde6ab4ce960d5/results/real/full_v7_C8/points/r01_fp16_math_500_apc0_C8/attempt001/measure.requests.jsonl) 可直接见到thinking=true、temperature=.6、top_p=.95、top_k=20、max_tokens:null。旧 [run_real.py](https://github.com/LinWang1225/test/blob/7801b642d03a27aca0486341f4cde6ab4ce960d5/kvbench_real_v6/run_real.py) 用显式null避免客户端补上缺省整数。v3对齐这一真实请求行为。

## 4. 从结果检查

1. manifest.json与run.json记录generation_budget_policy=remaining-context及max_tokens_transport=explicit-json-null。
2. measure.requests.jsonl中的正式max_tokens必须存在且为null，不应有max_completion_tokens整数。
3. launch.json应为generation_config=vllm、override_generation_config={}，原生窗口与prepare一致。
4. 对照本地输入token与原生usage.prompt_tokens；不一致响应不算完整主测量。
5. remaining_context_tokens_native由原生输入和已验证窗口推导。完整length早于此预算仍评分，另记unexpected_length_before_budget、每点计数和generation_budget_consistent。

提前length是排查信号，可能涉及服务预算、版本或token计数语义。不能据此删除结果或直接断定有隐藏cap。HTTP断流或缺usage则是不完整响应，不伪装成正常length截断。

## 5. 与官方LongBench的关系

本版使用官方任务数据、提示、评分指标；默认生成协议为自然输出thinking，未使用官方dataset2maxlen整数，也不强制greedy。应标注“LongBench tasks under Qwen thinking / remaining-context protocol”，不可当作官方生成协议复现与论文原表混排。

CLI保留显式reference（要求no-thinking）和fixed策略以供专门复现；默认与README推荐命令都不使用。此次实验只需remaining-context，不传max-new-tokens。
