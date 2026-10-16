# Sources, compatibility and scope

## Existing project code

Repository baseline:
https://github.com/LinWang1225/test/commit/5702570bb3c182bc9168c5d8434f7f95bcc7da6d

This additive package imports the pre-existing `kvbench_residual` and
`kvbench_residual_diag` modules. It does not redistribute or replace them.
Four reused diagnostic files are SHA256-pinned in `dependency_pins.py`.
The original runtime builder separately pins the six KVarN writer/readers at:
https://github.com/huawei-csl/KVarN/tree/7586257f1c632e63187bfacbbe21ccb51540f7b3

The existing mounted two-experiment package was checked against the repository
blob identities for common.py, math_ops.py, capture.py, run.py, runtime.py,
diagnose.py and oracle.py before developing this additive experiment.

## LongBench prompts and source

`longbench_prompts.json` reproduces the 21 original LongBench v1 prompt
strings (JSON key ordering/whitespace normalized) from:
https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2prompt.json
Git blob observed: 1c85f6bc0f0df4e42131aa49a867797ee7043ecf

Upstream license retained as `LICENSE.LongBench`:
https://github.com/THUDM/LongBench/blob/main/LICENSE

Original data are NOT bundled or downloaded. The runner reads the user's
existing local official data.zip/JSONL and records each source file hash.
Known existing dataset zip SHA256:
cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64
Existing frozen dataset revision:
5e628be450b7e67fb7ae6e201bd6d8f7056f7672 (zai-org/LongBench)
A supplied different local source is explicitly identified by its bytes and
not silently treated as matching that known archive.

## Related method, not implemented as a baseline here

Read What Matters: Query-Adaptive Quantization for KV Caches (ReadKV),
arXiv:2610.11245v1, 2026-10-08.
https://arxiv.org/abs/2610.11245
https://arxiv.org/html/2610.11245v1

ReadKV already separates retained/read budgets and uses asymmetric K/V
allocation units; Appendix F.8 studies GPU reader constraints including equal
K/V rates. This package does not implement or benchmark ReadKV and does not
claim novelty merely from unequal K/V read rates.

vLLM plugin system documentation (implementation packaging context):
https://docs.vllm.ai/en/latest/design/plugin_system/

## Measurement distinction

The new package compares logical selected KV-code payload at fixed captured
Query/KV. It measures float64 attention errors, not GPU selective kernel
latency, total physical traffic, model task scores, or a globally optimal
allocation. The existing service is changed only inside a new private Parent
runtime by an eager diagnostic observation hook.
