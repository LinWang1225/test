#!/usr/bin/env bash
# One-command real GPU online LongBench run (20 cases per dataset).
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
SUITE="$ROOT/results/longcontext_v3/full_longbench_C8_20261009_201156_2217646/suite.json"
ARGS=("$@")
for ((i=0;i<${#ARGS[@]};i++));do
  if [[ ${ARGS[i]} == --upstream-suite && $((i+1)) -lt ${#ARGS[@]} ]];then
    SUITE=${ARGS[i+1]}
  elif [[ ${ARGS[i]} == --upstream-suite=* ]];then
    SUITE=${ARGS[i]#*=}
  fi
done
BASE=${CONDA_BASE:-/home/wanglin/miniconda3}
if [[ ! -f "$BASE/etc/profile.d/conda.sh" ]];then
  echo "Set CONDA_BASE to the existing Miniconda root" >&2; exit 2
fi
mapfile -t PREP < <("$BASE/bin/python" - "$SUITE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]).expanduser().resolve(strict=True)
s=json.loads(p.read_text())['settings']
print(s['server_env'])
print(s['kit'])
PY
)
if [[ ${#PREP[@]} -ne 2 ]];then echo "Invalid upstream V3 suite" >&2;exit 2;fi
set +u
source "$BASE/etc/profile.d/conda.sh"
conda activate "${PREP[0]}"
source "${PREP[1]}/env.sh"
set -u
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python -m kvbench_progressive_online.run "$@"
