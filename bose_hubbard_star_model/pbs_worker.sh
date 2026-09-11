#!/usr/bin/env bash
#PBS -N bh_star
#PBS -l nodes=1:ppn=1
#PBS -j oe
#PBS -V
set -euo pipefail

project_dir="${PROJECT_DIR:-${PBS_O_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}}"
cd "$project_dir"

config_path="${CONFIG_PATH:-config_L7_L8.json}"
task_id="${TASK_ID:-${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}}"
if [[ -z "$task_id" ]]; then
  echo "TASK_ID/PBS_ARRAY_INDEX/PBS_ARRAYID is not set" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 star_model.py worker --config "$config_path" --task-id "$task_id"
