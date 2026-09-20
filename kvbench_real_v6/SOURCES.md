# 核对的主要接口来源

- EvalScope v1.12.0 registry / template：
  https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/api/registry.py
  https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/benchmarks/math_500/math_500_adapter.py
  https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/benchmarks/humaneval/humaneval_adapter.py
- EvalScope 数值评分：
  https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/metrics/nlp/metrics.py
- 完整 JSON request body 按已有字段优先合并，支持 stream_options include_usage：
  https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/perf/plugin/api/openai_api.py
  https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/perf/plugin/datasets/line_by_line.py
- EvalScope SQLite / response_messages 编码：
  https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/perf/utils/db_util.py
- vLLM 自然输出边界与 generation config：
  https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/entrypoints/serve/utils/api_utils.py
  https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/config/model.py
- vLLM 原生日志中 Prefix cache hit rate：
  https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/v1/metrics/loggers.py
- 配置参数及数据集说明：
  https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/parameters.html
  https://evalscope.readthedocs.io/en/latest/benchmarks/math_500.html
  https://evalscope.readthedocs.io/en/latest/benchmarks/humaneval.html
  https://docs.vllm.ai/en/latest/configuration/engine_args/
  https://docs.vllm.ai/en/latest/design/metrics/
- Docker 限制参数：
  https://docs.docker.com/engine/containers/run/

通用 latest 文档只用于解释参数含义；不把主线最新新增参数强加给旧 KVarN fork。启动仍由已运行过的本地 serve.sh 负责。
