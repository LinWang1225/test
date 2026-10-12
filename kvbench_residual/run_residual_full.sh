#!/usr/bin/env bash
# May start now: waits for the explicitly named V3 suite before any GPU work.
# Usage: bash kvbench_residual/run_residual_full.sh --after-suite /absolute/old_batch/suite.json
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${AR_DRIVER_PYTHON:-python3}" -m kvbench_residual.suite "$@"
