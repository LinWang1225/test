#!/usr/bin/env bash
# Uses existing Conda environments. Installs nothing and does not modify v3/v4.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KVBENCH_ROOT="${KVBENCH_ROOT:-$HOME/kvbench-3090-cu126}"
export KVBENCH_SERVE_ENV="${KVBENCH_SERVE_ENV:-kvarn-serve}"
export KVBENCH_EVAL_ENV="${KVBENCH_EVAL_ENV:-evalscope-client}"
if [[ -z "${CONDA_SH:-}" ]]; then
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    BASE="$("$CONDA_EXE" info --base)"
  elif command -v conda >/dev/null 2>&1; then
    BASE="$(conda info --base)"
  else
    echo '找不到 Conda，请设置 CONDA_SH=/实际位置/etc/profile.d/conda.sh' >&2; exit 2
  fi
  export CONDA_SH="$BASE/etc/profile.d/conda.sh"
fi
source "$CONDA_SH"
conda activate "$KVBENCH_EVAL_ENV"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
if [[ -z "${MODEL:-}" && -f "$KVBENCH_ROOT/model.env" ]]; then
  source "$KVBENCH_ROOT/model.env"
  export MODEL
fi
exec python -u "$HERE/run_real.py" "$@"
