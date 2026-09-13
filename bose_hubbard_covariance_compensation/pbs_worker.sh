#!/bin/bash
#PBS -N bh_cov_comp
#PBS -l walltime=04:00:00
#PBS -l mem=6gb
#PBS -l nodes=1:ppn=1
#PBS -j oe

set -euo pipefail

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH is required}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_ID="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-${TASK_ID:-}}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "PBS array index is missing" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -u covariance_compensation.py worker --config "${CONFIG_PATH}" --task-id "${TASK_ID}"
