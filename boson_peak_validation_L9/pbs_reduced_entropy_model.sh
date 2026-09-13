#!/bin/bash
#PBS -N bh_entropy_reduced
#PBS -q batch
#PBS -l nodes=1:ppn=1
#PBS -l mem=8gb
#PBS -l walltime=02:00:00

set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${VERIFICATION_L7:?VERIFICATION_L7 is required}"
: "${VERIFICATION_L8:?VERIFICATION_L8 is required}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/reduced_entropy_model_L7_L8}"

export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

cd "$PROJECT_DIR"
"$PYTHON_BIN" analyze_reduced_entropy_model.py \
  "$VERIFICATION_L7" \
  "$VERIFICATION_L8" \
  --output-dir "$OUTPUT_DIR"
