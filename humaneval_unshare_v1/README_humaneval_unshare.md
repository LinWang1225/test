# HumanEval scorer without Docker

This scorer uses Linux namespaces + an ephemeral chroot. It needs no sudo,
Docker, Podman, Apptainer, or bubblewrap.

## Install

```bash
export ROOT="$HOME/kvbench-3090-cu126"
export REAL="$ROOT/kvbench_real_v6"

cp score_humaneval_unshare.py "$REAL/"
python -m py_compile "$REAL/score_humaneval_unshare.py"
```

## Self-test first

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate evalscope-client

python "$REAL/score_humaneval_unshare.py" \
  --suite "$ROOT/results/real/full_v7_C8" \
  --self-test-only
```

Expected:

```text
sandbox self-test: PASS
```

The self-test checks that `/home` is hidden, the chroot root is not writable,
external network access is unavailable, and ordinary Python execution works.

## Score HumanEval

```bash
python "$REAL/score_humaneval_unshare.py" \
  --suite "$ROOT/results/real/full_v7_C8"
```

This uses only the saved HumanEval generations. It does not rerun inference.
It writes the same files as the existing scorer:

```text
points/*humaneval*/attempt001/
  scores.jsonl
  scores.summary.json
  grading_protocol.json
```

and refreshes root-level `summary.csv` / `summary.json`.

If score files already exist, add `--force`; old scoring files are backed up.

## Security note

This is materially safer than host `exec` or a venv: user/network/mount/PID/
IPC/UTS namespaces, chroot, read-only runtime mounts, private `/tmp`, dropped
capabilities, `no_new_privs`, resource limits, and a wall timeout are used.
It is still not equivalent to a hardened VM/gVisor/Firecracker sandbox against
kernel exploits; use it for controlled offline HumanEval benchmarking, not an
open untrusted-code service.
