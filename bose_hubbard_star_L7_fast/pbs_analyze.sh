#!/bin/bash
#PBS -N bh_star_L7_plot
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -V

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/config_L7_pilot.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PROJECT_DIR"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" -u run_l7_star.py analyze --config "$CONFIG_PATH"
