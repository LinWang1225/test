# KVCache precision sweep v8

This patch adds precision-sweep aliases while leaving the tested `serve.sh`
unchanged.

## Method matrix

### Primary precision comparison

| Alias | KV dtype / preset | Interpretation |
|---|---|---|
| `fp16` | `auto` | existing FP16 baseline |
| `fp8` | `fp8_e4m3` + dynamic KV scales | standard 8-bit FP8 K/V |
| `tq84` | `turboquant_k8v4` | TQ: FP8 K + 4-bit V |
| `tq4` | `turboquant_4bit_nc` | TQ: 4-bit K + 4-bit V |
| `tq34` | `turboquant_k3v4_nc` | TQ: 3-bit K + 4-bit V |
| `tq3` | `turboquant_3bit_nc` | TQ: 3-bit K + 3-bit V |
| `kvarn44` | `kvarn_k4v4_g128` | KVarN 4-bit K + 4-bit V |
| `kvarn` | `kvarn_k4v2_g128` | KVarN 4-bit K + 2-bit V |

### KVarN group-size ablation (not a precision ladder)

| Alias | Preset |
|---|---|
| `kvarn44` | K4V4, group 128 |
| `kvarn4464` | K4V4, group 64 |
| `kvarn` | K4V2, group 128 |
| `kvarn64` | K4V2, group 64 |

Do not mix `g64 vs g128` into the main “precision” conclusion: group size
changes quantization granularity and metadata overhead, not just precision.

## Why no serve.sh edit?

The existing engine already passes `--enforce-eager` through `serve.sh`, so
the launcher supports extra vLLM CLI arguments. New aliases reuse an existing
base case with the correct block size:

- `fp8` -> `fp16` base (block 128), then overrides KV dtype.
- `tq84/tq34` -> `tq4` base (block 128), then overrides KV dtype.
- `kvarn4464` -> `kvarn64` base (block 64), then overrides KV dtype.

The final `--kv-cache-dtype` wins in vLLM argparse. Launch metadata records both
the logical method and the actual base/extra args.

## FP8 scaling

Standard vLLM FP8 KV uses `--calculate-kv-scales`. If scales are absent and this
is disabled, vLLM falls back to 1.0. This patch enables dynamic scales for the
`fp8` alias.

On RTX 3090 (SM80), standard FP8 KV is a storage format rather than native
Hopper FP8 hardware acceleration. Also, FlashAttention in this fork accepts
FP8 KV only on SM90 + FA3; on a 3090 the automatic attention selector can fall
back to another backend. Therefore:

- the normal `fp8` run is a realistic deployment comparison;
- its latency difference versus the existing FP16 result can include an
  attention-backend change;
- an optional controlled `TRITON_ATTN` FP16-vs-FP8 run is provided below.

## Apply patch

```bash
export ROOT="$HOME/kvbench-3090-cu126"
export REAL="$ROOT/kvbench_real_v6"

python apply_precision_patch.py "$REAL"

python -m py_compile \
  "$REAL/run_real.py" \
  "$REAL/engine.py" \
  "$REAL/profile_kernels.py"
```

Check aliases:

```bash
python "$REAL/run_real.py" --help | grep -A2 methods
```

## Stage 1: 32-task pilot before any new full run

Use both datasets because HumanEval exposed behavior that MATH did not.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate evalscope-client

export ROOT="$HOME/kvbench-3090-cu126"
export REAL="$ROOT/kvbench_real_v6"
export KIT="$HOME/kv_evalscope_3090_cu126_v2"
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
export KVBENCH_SERVE_ENV=kvarn-serve
export KVBENCH_EVAL_ENV=evalscope-client
source "$ROOT/model.env"
export MODEL

bash "$REAL/run_real.sh" \
  --stage pilot \
  --data-dir "$ROOT/data/real_v5" \
  --run-dir "$ROOT/results/real/precision_v8_pilot" \
  --datasets math_500 humaneval \
  --methods fp8 tq84 tq34 tq3 kvarn44 kvarn64 kvarn4464 \
  --concurrency 8 \
  --pilot-size 32 \
  --warmup-requests 16 \
  --apc 0 \
  --continue-on-error
```

For every point verify:
- `measurement_complete=true`
- no OOM / service exit
- `waiting=0`, `preemptions=0` at C8 unless the method genuinely hits pressure
- inspect `server.measurement.log` for method-specific JIT warnings
- startup log reports the expected KV dtype and KV capacity.

## Stage 2: full new-method sweep at C8

The existing `full_v7_C8` already contains `fp16`, `tq4`, and `kvarn`.
Do not spend time rerunning those unless you deliberately want a fresh repeat.

```bash
bash "$REAL/run_real.sh" \
  --stage full \
  --data-dir "$ROOT/data/real_v5" \
  --run-dir "$ROOT/results/real/precision_v8_C8" \
  --datasets math_500 humaneval \
  --methods fp8 tq84 tq34 tq3 kvarn44 kvarn64 kvarn4464 \
  --concurrency 8 \
  --warmup-requests 16 \
  --apc 0 \
  --continue-on-error
```

If interrupted, rerun the same command with:

```bash
--resume
```

## Score MATH

```bash
python "$REAL/score_saved.py" \
  --suite "$ROOT/results/real/precision_v8_C8" \
  --datasets math_500
```

## Score HumanEval with the existing unshare scorer

```bash
python "$REAL/score_humaneval_unshare.py" \
  --suite "$ROOT/results/real/precision_v8_C8"
```

## Optional controlled FP16-vs-FP8 backend experiment

On RTX 3090 the default FP16 run normally uses FlashAttention, whereas FP8 can
fall back because this fork's FlashAttention backend accepts FP8 KV only on
SM90/FA3. To isolate *KV dtype* from *attention backend*, run both under Triton:

```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
bash "$REAL/run_real.sh" \
  --stage pilot \
  --data-dir "$ROOT/data/real_v5" \
  --run-dir "$ROOT/results/real/fp8_triton_control" \
  --datasets math_500 \
  --methods fp16 fp8 \
  --concurrency 8 \
  --pilot-size 64 \
  --warmup-requests 16 \
  --apc 0
```

If stable, repeat with `--stage full`.

This control should be reported separately from the normal production-like
auto-backend results.

## What to compare

For each preset collect:

- MATH accuracy / HumanEval pass@1
- correct tasks/s
- RPS and output tok/s
- TPOT P50/P95
- TTFT P50/P95
- E2E P50/P95
- output-token mean/P50/P95/P99
- context-limit fraction
- KV capacity tokens and capacity ratio vs FP16
- peak KV usage
- waiting / preemptions
- GPU utilization
- later: `kv_quant_store` and `kv_decode_dequant_fused` profile time

The most useful plots are:
1. Accuracy/pass@1 vs KV capacity.
2. Correct tasks/s vs KV capacity.
3. TPOT vs KV capacity.
4. Quality vs nominal bits/preset.
5. Within TurboQuant: K8V4 -> K4V4 -> K3V4 -> K3V3.
6. Within KVarN (g128 only): K4V4 -> K4V2.
7. Separate KVarN group ablation: g128 vs g64 at fixed K/V bits.
