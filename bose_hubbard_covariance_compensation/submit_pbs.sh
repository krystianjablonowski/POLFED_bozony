#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-config_L4_pilot.json}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CONFIG_PATH="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${CONFIG_PATH}")"

"${PYTHON_BIN}" covariance_compensation.py validate --config "${CONFIG_PATH}"
"${PYTHON_BIN}" covariance_compensation.py init --config "${CONFIG_PATH}"
TASK_COUNT="$("${PYTHON_BIN}" covariance_compensation.py task-count --config "${CONFIG_PATH}")"
LAST_TASK="$((TASK_COUNT - 1))"
RESULT_ROOT="$("${PYTHON_BIN}" covariance_compensation.py root --config "${CONFIG_PATH}")"
mkdir -p "${RESULT_ROOT}/logs"

set +e
JOB_ID="$(qsub -J "0-${LAST_TASK}" -o "${RESULT_ROOT}/logs/" -v "CONFIG_PATH=${CONFIG_PATH},PROJECT_DIR=${PROJECT_DIR},PYTHON_BIN=${PYTHON_BIN}" pbs_worker.sh 2>&1)"
ARRAY_STATUS=$?
set -e
if [[ ${ARRAY_STATUS} -eq 0 ]]; then
  echo "Submitted ${TASK_COUNT} worker tasks as ${JOB_ID} (no concurrency limit)."
else
  echo "PBS array unavailable (${JOB_ID}); submitting ${TASK_COUNT} ordinary jobs."
  for TASK_ID in $(seq 0 "${LAST_TASK}"); do
    qsub -o "${RESULT_ROOT}/logs/" -v "CONFIG_PATH=${CONFIG_PATH},PROJECT_DIR=${PROJECT_DIR},PYTHON_BIN=${PYTHON_BIN},TASK_ID=${TASK_ID}" pbs_worker.sh
  done
fi
echo "Check: qstat -u $USER"
echo "Progress: python covariance_compensation.py status --config ${CONFIG_PATH}"
