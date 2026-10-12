#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
SUITE="results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json"
args=("$@")
for ((i=0;i<${#args[@]};i++)); do
  if [[ ${args[i]} == --upstream-suite ]]; then
    if (( i+1>=${#args[@]} )); then echo '--upstream-suite needs a path' >&2; exit 2; fi
    SUITE=${args[i+1]}
  elif [[ ${args[i]} == --upstream-suite=* ]]; then
    SUITE=${args[i]#*=}
  fi
done
BASE=${CONDA_BASE:-/home/wanglin/miniconda3}
if [[ ! -f "$BASE/etc/profile.d/conda.sh" ]]; then
  echo 'Set CONDA_BASE to the existing Miniconda root; nothing is installed automatically.' >&2
  exit 2
fi
# Bootstrap only reads the known local suite. No network/model/GPU access.
BOOT_PY="$BASE/bin/python"
mapfile -t SETUP < <("$BOOT_PY" - "$SUITE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))['settings']
print(s['server_env'])
print(s['kit'])
PY
)
if [[ ${#SETUP[@]} -ne 2 ]]; then echo 'Unable to read server environment and KIT from suite' >&2; exit 2; fi
set +u
source "$BASE/etc/profile.d/conda.sh"
conda activate "${SETUP[0]}"
if [[ ! -f "${SETUP[1]}/env.sh" ]]; then echo 'The original KIT/env.sh is missing; refusing guessed CUDA settings' >&2; exit 2; fi
export KIT="${SETUP[1]}"
source "$KIT/env.sh"
set -u
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Keep BLAS thread counts explicit so offline greedy diagnostics do not oversubscribe CPUs.
export OPENBLAS_NUM_THREADS=${AR_DIAG_BLAS_THREADS:-4}
export OMP_NUM_THREADS=${AR_DIAG_BLAS_THREADS:-4}
export MKL_NUM_THREADS=${AR_DIAG_BLAS_THREADS:-4}
exec python -m kvbench_residual_diag.run "$@"
