#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-config_L4_pilot.json}"
ED_CSV="${2:-}"
MODEL_CSV="${3:-}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CONFIG_PATH="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${CONFIG_PATH}")"
RESULT_ROOT="$("${PYTHON_BIN}" covariance_compensation.py root --config "${CONFIG_PATH}")"
mkdir -p "${RESULT_ROOT}/logs"

"${PYTHON_BIN}" covariance_compensation.py status --config "${CONFIG_PATH}"
qsub -o "${RESULT_ROOT}/logs/analyze.out" -v "CONFIG_PATH=${CONFIG_PATH},PROJECT_DIR=${PROJECT_DIR},PYTHON_BIN=${PYTHON_BIN},ED_CSV=${ED_CSV},MODEL_CSV=${MODEL_CSV}" pbs_analyze.sh
