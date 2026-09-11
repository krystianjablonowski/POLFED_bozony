#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="${1:-$PROJECT_DIR/config_L7_corrected.json}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONNOUSERSITE=1

CONFIG_PATH="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
"$PYTHON_BIN" run_l7_star.py validate --config "$CONFIG_PATH"

MANIFEST="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
from run_l7_star import load_config, root_path
print(root_path(load_config(Path(sys.argv[1]))) / "tasks.csv")
PY
)"
if [[ ! -f "$MANIFEST" ]]; then
  "$PYTHON_BIN" run_l7_star.py init --config "$CONFIG_PATH"
fi

TASK_COUNT="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
from run_l7_star import load_config, tasks
print(len(tasks(load_config(Path(sys.argv[1])))))
PY
)"
read -r WALLTIME MEMORY < <("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
from run_l7_star import load_config
c = load_config(Path(sys.argv[1]))
print(c["cluster"]["walltime"], c["cluster"]["memory"])
PY
)
LAST_TASK=$((TASK_COUNT - 1))
echo "Submitting all $TASK_COUNT independent tasks (array 0-$LAST_TASK)."

set +e
ARRAY_OUTPUT="$(qsub -l "walltime=$WALLTIME,mem=$MEMORY" -J "0-$LAST_TASK" -o "$PROJECT_DIR/logs/" \
  -v "CONFIG_PATH=$CONFIG_PATH,PROJECT_DIR=$PROJECT_DIR,PYTHON_BIN=$PYTHON_BIN" "$PROJECT_DIR/pbs_worker.sh" 2>&1)"
ARRAY_STATUS=$?
set -e
if [[ $ARRAY_STATUS -eq 0 ]]; then
  echo "PBS array submitted: $ARRAY_OUTPUT"
else
  echo "PBS array unavailable ($ARRAY_OUTPUT); submitting all $TASK_COUNT ordinary jobs."
  for TASK_ID in $(seq 0 "$LAST_TASK"); do
    qsub -l "walltime=$WALLTIME,mem=$MEMORY" -o "$PROJECT_DIR/logs/" \
      -v "CONFIG_PATH=$CONFIG_PATH,PROJECT_DIR=$PROJECT_DIR,PYTHON_BIN=$PYTHON_BIN,TASK_ID=$TASK_ID" "$PROJECT_DIR/pbs_worker.sh"
  done
fi

echo "After the workers finish, submit plotting with:"
echo "qsub -l walltime=01:00:00,mem=8gb -o $PROJECT_DIR/logs/analyze_corrected.out -v CONFIG_PATH=$CONFIG_PATH,PROJECT_DIR=$PROJECT_DIR,PYTHON_BIN=$PYTHON_BIN $PROJECT_DIR/pbs_analyze.sh"
