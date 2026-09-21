# kvbench v7 补丁：自然生成 null sentinel + 独立 kernel profiling

## 1. 为什么用 `"max_tokens": null`

EvalScope perf 1.12.0 的 CLI `max_tokens` 默认是 2048。对于 `line_by_line` 的完整 JSON 请求，
缺失字段会由 CLI 默认值补齐；但已存在的 key 使用 `setdefault` 语义，不会覆盖。

所以自然生成请求应包含：

```json
"max_tokens": null
```

这不是“把上限设置成某个数”。它只是传输层 sentinel：
- EvalScope 不再补 2048；
- vLLM 的 ChatCompletion 协议接受 `max_tokens=None`；
- vLLM 再按 `max_model_len - prompt_len` 和平台限制决定真实硬边界；
- 因而换模型时无需在客户端知道模型 context 上限。

不要使用一个很大的整数冒充“无限”，也不要修改 site-packages 中的 EvalScope 源码。

## 2. 应用补丁

```bash
export ROOT="$HOME/kvbench-3090-cu126"
export REAL="$ROOT/kvbench_real_v6"

python apply_v7_patch.py "$REAL"
cp profile_kernels.py "$REAL/"
python -m py_compile "$REAL/run_real.py" "$REAL/profile_kernels.py"
```

补丁会备份原 `run_real.py`。

主要变化：
1. 无显式 cap 时，请求 JSON 写 `"max_tokens": null`。
2. `measure.manifest.json` 会记录 `max_tokens_transport=json_null_sentinel`。
3. `missing_native_usage` 默认只记录、不再自动淘汰 pilot；若确实要强制，可加 `--require-native-usage`。
4. 不改 EvalScope 安装环境，不改 vLLM 服务端模型上限。

## 3. 先做协议验证

```bash
bash "$REAL/run_real.sh" \
  --stage pilot \
  --data-dir "$ROOT/data/real_v5" \
  --run-dir "$ROOT/results/real/protocol_v7" \
  --datasets math_500 \
  --methods fp16 \
  --concurrency 2 \
  --pilot-size 8 \
  --warmup-requests 8 \
  --apc 0
```

检查发送给 EvalScope 的完整 request：

```bash
grep -n '"max_tokens": null' \
  "$ROOT/results/real/protocol_v7/points/"*/attempt001/measure.requests.jsonl | head
```

注意：`measurement.log` 的 EvalScope 全局 Arguments 仍可能打印 `"max_tokens": 2048`，
因为那是 CLI 默认对象；关键是完整 JSON body 已存在 `max_tokens:null`，因此 OpenAI plugin 的
`setdefault` 不会覆盖它。最终 `benchmark_data.db` / `requests.jsonl` 中保存的实际 request
也应保持 null。`outputs.py` 已经会把参数被改写记录为 `request_parameter_mismatches`。

最直观验证：原先撞 2048 的题应能出现 `completion_tokens > 2048`。

## 4. decode 与量化/反量化怎么分

主实验继续使用**不启 profiler**的结果：
- `TPOT`：端到端 decode 每输出 token 的服务性能；
- `E2E/RPS/output tok/s`：任务层性能。

另做短 profiling，不把 profiler 延迟混入主性能结果。

当前 TurboQuant/KVarN 的 dequant 是融合在 decode-attention kernel 内的。因此只能可靠拆成：

- `kv_quant_store`：独立的量化/写 KV / tile flush kernel；
- `kv_decode_dequant_fused`：融合了 attention decode + dequant/packing 的量化 decode kernel；
- `fp16_attention_decode`：FP16/通用 attention kernel；
- `sampling`；
- `other`。

**不能把 `kv_decode_dequant_fused` 全部叫做“反量化时间”。**
若要得到纯 dequant 时间，需要后续做 split-kernel/disable-dequant ablation 或专门 microbenchmark，
那会改变执行路径，不能混入主 serving benchmark。

## 5. 运行独立 kernel profile

先让 v7 pilot 产生已验证的 `measure.requests.jsonl`，然后分别跑：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
export KVBENCH_ROOT="$HOME/kvbench-3090-cu126"
export ROOT="$KVBENCH_ROOT"
export REAL="$ROOT/kvbench_real_v6"
export KIT="$HOME/kv_evalscope_3090_cu126_v2"
source "$ROOT/model.env"

REQ="$ROOT/results/real/pilot_v7/points/r01_fp16_math_500_apc0_C8/attempt001/measure.requests.jsonl"

python "$REAL/profile_kernels.py" \
  --method fp16 \
  --requests-jsonl "$REQ" \
  --out "$ROOT/results/profile/fp16_math_c8" \
  --concurrency 8 \
  --number 8 \
  --warmup-number 8 \
  --profile-iterations 160
```

TurboQuant/KVarN 使用对应方法和其请求文件分别运行。

`profile-iterations=160` 的目的不是完整跑完推理，而是采一段 decode：
- profiler 在 `/start_profile` 后延迟 1 个 worker iteration，尽量绕开首个 prefill；
- 采 160 个 worker iterations；
- 160 通常足以跨过至少一个 128-token KVarN tile boundary，从而捕获 flush/quantize。

输出：

```text
kernel_profile_summary.json
kernel_profile_categories.csv
kernel_profile_top.csv
torch_profile/...
server.log
profile_run.log
```

Torch profiler 本身会明显降低推理速度，所以这些 profiling run 的 TTFT/TPOT **不能**拿来和主实验性能比较。
它只用于解释主实验 TPOT 差异来自哪些 kernel 类别。

## 6. 后续建议

主全量结果和 kernel profile 分两张表：

### 服务性能
`Accuracy / output tokens / TTFT / TPOT / E2E / RPS / KV capacity`

### 内核解释
`kv_quant_store CUDA ms(%), kv_decode_dequant_fused CUDA ms(%), attention CUDA ms(%), top kernels`

这样能区分：
- “量化方法的 overall decode 变慢/变快”（TPOT）
- “额外 KV quant/store 占了多少 GPU 时间”
- “量化 decode+dequant fused path 占了多少 GPU 时间”

而不会错误地把融合 kernel 的全部时间都归因于反量化。

补充：`kernel_profile_categories.csv` 同时给出每个类别的 `cuda_time_ms_per_requested_worker_iteration`；它是按请求的 profiler worker-iteration 数归一化，不等于单 token 时延。
