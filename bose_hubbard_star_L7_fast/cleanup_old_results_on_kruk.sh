#!/bin/bash
set -euo pipefail

# This removes only outputs of the superseded implementation, never its source.
OLD_RESULTS="/home/2/kj405942/POLFED_bosons/bose_hubbard_star_model/results_L7_L8"
OLD_LOGS="/home/2/kj405942/POLFED_bosons/bose_hubbard_star_model/logs"
EXPECTED_RESULTS="$OLD_RESULTS"

if [[ ! -d "$OLD_RESULTS" ]]; then
  echo "Old results directory does not exist: $OLD_RESULTS"
  exit 0
fi
RESOLVED="$(realpath "$OLD_RESULTS")"
if [[ "$RESOLVED" != "$EXPECTED_RESULTS" ]]; then
  echo "Refusing to delete unexpected path: $RESOLVED" >&2
  exit 2
fi
du -sh "$OLD_RESULTS" "$OLD_LOGS" 2>/dev/null || true
printf 'Type DELETE to permanently remove the old results and logs: '
read -r CONFIRMATION
if [[ "$CONFIRMATION" != "DELETE" ]]; then
  echo "Nothing removed."
  exit 1
fi
rm -rf -- "$OLD_RESULTS"
if [[ -d "$OLD_LOGS" && "$(realpath "$OLD_LOGS")" == "$OLD_LOGS" ]]; then
  rm -rf -- "$OLD_LOGS"
fi
echo "Removed old generated results and logs. The old source code was preserved."
