# Reviewed integration sources

User repository: https://github.com/LinWang1225/test

Existing integration contracts inspected on 2026-10-12:

- `kvbench_residual/runtime.py`: isolated overlay and exact source pins.
- `kvbench_residual/bootstrap.py`: private package import ordering and worker propagation.
- `kvbench_residual/trace.py`: launch proxy and source provenance verification.
- `kvbench_residual/fixtures.py`: native packed K4V4 layout, real-arithmetic policy reference, and GpuCase launch/statistics API.
- `kvbench_residual/kernel.py`, `TECH_CONTRACT.md`: chunk budgets, prefix mass lower bounds, GQA union semantics.
- `kvbench_residual/serve.py`: architecture validation and owned-process cleanup.
- `kvbench_longcontext/common.py`, `run.py`, `serve.py`: frozen dataset checks, request construction, streaming protocol, and native model/cache checks.
- Frozen source suite: `results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json`.

KVarN parent package commit: `7586257f1c632e63187bfacbbe21ccb51540f7b3`.
https://github.com/huawei-csl/KVarN/blob/7586257f1c632e63187bfacbbe21ccb51540f7b3/vllm/v1/attention/ops/triton_kvarn_decode.py

Runtime construction reuses the existing six-file SHA256 verification. This package adds a separate observer and counterfactual analysis; it does not claim GPU validation in the delivery environment.

The offline V-only lower bound is derived directly from the triangle inequality and a fractional-knapsack relaxation. It is not attributed to an external paper and is not advertised as a tight optimum. K/KV greedy selection is an achievable reference only.
