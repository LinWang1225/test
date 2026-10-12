#!/usr/bin/env bash
# Optional, after a residual suite has finished. Never installs drivers or changes counter permissions.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${AR_DRIVER_PYTHON:-python3}" -m kvbench_residual.profile_dram "$@"
