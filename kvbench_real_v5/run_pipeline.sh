#!/usr/bin/env bash
# Real-data pilot -> common-concurrency full run -> offline scoring of saved outputs.
# Usage: bash run_pipeline.sh DATA_DIR RESULT_ROOT
set -eo pipefail
[[ $# -eq 2 ]] || { echo '用法：bash run_pipeline.sh 冻结数据目录 新结果根目录' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$1"; RUN="$2"
[[ -f "$DATA/manifest.json" ]] || { echo "缺少 $DATA/manifest.json，请先 prepare_data.py" >&2; exit 2; }
if [[ -f "$RUN/pilot/suite.json" || -f "$RUN/full/suite.json" ]]; then
  echo '结果目录已有实验。续跑请使用 README 的分阶段 --resume 命令，不自动覆盖。' >&2; exit 2
fi
# Check an explicitly requested code sandbox before spending GPU time.
if [[ "${SCORE_HUMAN_DOCKER:-0}" == 1 ]]; then
  docker image inspect "${DOCKER_IMAGE:-python:3.11-slim}" --format '{{.Id}}' >/dev/null || {
    echo 'HumanEval Docker 镜像/权限未就绪；请先准备，或不启用 SCORE_HUMAN_DOCKER，仅保存生成结果。' >&2; exit 2;
  }
fi
read -r -a LEVELS <<< "${PILOT_CONCURRENCY:-2 4 8}"
COMMON=(--data-dir "$DATA" --apc "${APC:-0}" --gpu "${GPU:-0}"
        --max-seqs "${MAXSEQ:-16}" --max-len "${MAXLEN:-32768}"
        --math-max-tokens "${MATH_MAX_TOKENS:-16384}" --human-max-tokens "${HUMAN_MAX_TOKENS:-8192}")
[[ -z "${KV_BYTES:-}" ]] || COMMON+=(--kv-bytes "$KV_BYTES")
RC=0
bash "$HERE/run_real.sh" --stage pilot "${COMMON[@]}" --concurrency "${LEVELS[@]}" \
  --pilot-size "${PILOT_SIZE:-32}" --continue-on-error --run-dir "$RUN/pilot" || RC=$?
# Recoverable individual point failures may still leave usable lower-C recommendations.
# Interruptions/configuration errors must stop the pipeline.
if ((RC>1)); then exit "$RC"; fi
[[ -f "$RUN/pilot/recommendations.json" ]] || { echo '未生成并发建议，停止。' >&2; exit 2; }
bash "$HERE/run_real.sh" --stage full "${COMMON[@]}" \
  --recommendations "$RUN/pilot/recommendations.json" --selection common --run-dir "$RUN/full"

# Activate the existing client only for offline grading; never execute model code on host.
if [[ -z "${CONDA_SH:-}" ]]; then
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then BASE="$("$CONDA_EXE" info --base)";
  else BASE="$(conda info --base)"; fi
  export CONDA_SH="$BASE/etc/profile.d/conda.sh"
fi
source "$CONDA_SH"
conda activate "${KVBENCH_EVAL_ENV:-evalscope-client}"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
if [[ "${SCORE_HUMAN_DOCKER:-0}" == 1 ]]; then
  python "$HERE/score_saved.py" --suite "$RUN/full" --datasets math_500 humaneval \
    --human-sandbox docker --docker-image "${DOCKER_IMAGE:-python:3.11-slim}"
else
  python "$HERE/score_saved.py" --suite "$RUN/full" --datasets math_500
  echo 'HumanEval 生成已保存，未执行代码评分。准备好隔离环境后，单独运行 score_saved.py --datasets humaneval --human-sandbox docker。'
fi
printf '结果：%s/full/summary.csv\n' "$RUN"
