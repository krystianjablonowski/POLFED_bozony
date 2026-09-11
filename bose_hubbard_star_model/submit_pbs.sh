#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$project_dir"
config_path="${1:-config_L7_L8.json}"
concurrency="${2:-}"
config_path="$(python3 - "$config_path" <<'PY'
import pathlib, sys
print(pathlib.Path(sys.argv[1]).resolve())
PY
)"

python3 star_model.py validate --config "$config_path"
if [[ ! -f "$(python3 - "$config_path" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]).resolve()
c=json.loads(p.read_text())
r=pathlib.Path(c['paths']['results_dir'])
print(r if r.is_absolute() else p.parent/r)
PY
)/tasks.csv" ]]; then
  python3 star_model.py init --config "$config_path"
fi

task_count="$(python3 - "$config_path" <<'PY'
import csv, json, pathlib, sys
p=pathlib.Path(sys.argv[1]).resolve()
c=json.loads(p.read_text())
r=pathlib.Path(c['paths']['results_dir'])
r=r if r.is_absolute() else p.parent/r
print(sum(1 for _ in csv.DictReader((r/'tasks.csv').open())))
PY
)"
read -r walltime memory config_concurrency < <(python3 - "$config_path" <<'PY'
import json, pathlib, sys
c=json.loads(pathlib.Path(sys.argv[1]).read_text())
print(c['cluster']['walltime'], c['cluster']['memory'], c['cluster']['array_concurrency'])
PY
)
concurrency="${concurrency:-$config_concurrency}"
last=$((task_count - 1))
if [[ "$concurrency" == "all" ]] || (( concurrency >= task_count )); then
  array_spec="0-${last}"
  echo "Submitting all $task_count tasks without a client-side concurrency limit"
else
  array_spec="0-${last}%${concurrency}"
  echo "Submitting $task_count tasks with concurrency $concurrency"
fi
mkdir -p logs

set +e
array_output="$(qsub -l "walltime=$walltime,mem=$memory" -o logs -v CONFIG_PATH="$config_path",PROJECT_DIR="$project_dir" -J "$array_spec" pbs_worker.sh 2>&1)"
array_status=$?
set -e
if [[ $array_status -eq 0 ]]; then
  echo "PBS array submitted: $array_output"
else
  echo "PBS array unavailable ($array_output); submitting ordinary jobs."
  for task_id in $(seq 0 "$last"); do
    qsub -l "walltime=$walltime,mem=$memory" -o logs -v CONFIG_PATH="$config_path",PROJECT_DIR="$project_dir",TASK_ID="$task_id" pbs_worker.sh
  done
fi
echo "After workers finish: qsub -v CONFIG_PATH=$config_path pbs_aggregate.sh"
