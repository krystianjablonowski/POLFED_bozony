#!/bin/bash
#PBS -N bh_cov_plot
#PBS -l walltime=04:00:00
#PBS -l mem=12gb
#PBS -l nodes=1:ppn=1
#PBS -j oe

set -euo pipefail

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH is required}"
ED_CSV="${ED_CSV:-}"
MODEL_CSV="${MODEL_CSV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -u covariance_compensation.py analyze --config "${CONFIG_PATH}" --ed-csv "${ED_CSV}" --model-csv "${MODEL_CSV}"
