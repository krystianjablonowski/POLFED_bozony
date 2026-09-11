#!/usr/bin/env bash
#PBS -N bh_star_agg
#PBS -l nodes=1:ppn=1
#PBS -l walltime=04:00:00
#PBS -l mem=8gb
#PBS -j oe
#PBS -V
set -euo pipefail
project_dir="${PROJECT_DIR:-${PBS_O_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}}"
cd "$project_dir"
config_path="${CONFIG_PATH:-config_L7_L8.json}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
extra_args=()
if [[ "${FAST_ANALYSIS:-0}" == "1" ]]; then
  extra_args+=(--fast)
fi
python3 star_model.py aggregate --config "$config_path" "${extra_args[@]}"
