#!/bin/bash
#PBS -N bh_entropy_verify
#PBS -l walltime=06:00:00
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
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
ONLY_L="${ONLY_L:-7}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -u verify_entropy_mechanism.py "${RUN_DIR}" \
  --L "${ONLY_L}" \
  --min-W 0.8 \
  --max-W 2.5 \
  --fit-W-minima 0.8,1.0,1.2 \
  --bootstrap "${BOOTSTRAP}"
