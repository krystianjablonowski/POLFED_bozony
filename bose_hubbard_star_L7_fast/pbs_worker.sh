#!/bin/bash
#PBS -N bh_star_L7
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -V

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/config_L7_corrected.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_ID="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-${TASK_ID:-}}}"
if [[ -z "$TASK_ID" ]]; then
  echo "Missing PBS array index" >&2
  exit 2
fi
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
"$PYTHON_BIN" -u run_l7_star.py worker --config "$CONFIG_PATH" --task-id "$TASK_ID"
