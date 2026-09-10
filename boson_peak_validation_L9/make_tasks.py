#!/usr/bin/env python3
"""Create resumable PBS/SLURM task arrays for the bosonic ED pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import boson_peak_core as core
import boson_peak_cli as cli
import boson_peak_hpc as engine


def task_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    realization_count = int(config["ed"]["realizations"])
    block_size = int(config["ed"].get("realization_block_size", realization_count))
    index = 0
    for L, N, nmax in core.ed_sectors(config):
        for W_over_t in sorted(set(float(x) for x in config["ed"]["W_values"])):
            U_values = engine.ed_U_grid(config, L, N, nmax, W_over_t)
            digest = engine.core_hash(config, L, N, nmax, W_over_t)
            for start in range(0, realization_count, block_size):
                stop = min(realization_count, start + block_size)
                rows.append(
                    {
                        "task_index": index,
                        "L": L,
                        "N": N,
                        "nmax": nmax,
                        "W_over_t": W_over_t,
                        "realization_start": start,
                        "realization_stop": stop,
                        "realizations_in_block": stop - start,
                        "U_points": int(U_values.size),
                        "U_values_over_t_json": json.dumps([float(x) for x in U_values], separators=(",", ":")),
                        "core_hash": digest,
                    }
                )
                index += 1
    return rows


def basis_metadata(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = []
    for L, N, nmax in core.ed_sectors(config):
        structure = core.build_structure(L, N, nmax, float(config["model"].get("t", 1.0)))
        output.append(
            {
                "L": L,
                "N": N,
                "nmax": nmax,
                "D": structure.dim,
                "hopping_nnz": int(structure.hopping.nnz),
                "hopping_density": float(structure.hopping.nnz / structure.dim**2),
                "Q_min": int(np.min(structure.interaction_q, initial=0)),
                "Q_max": int(np.max(structure.interaction_q, initial=0)),
                "estimated_peak_memory_MB": engine.estimated_memory_mb(structure, config),
            }
        )
    return output


def shell_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "'\"'\"'") + "'"


def write_scripts(output: Path, config_snapshot: Path, task_count: int, config: Dict[str, Any]) -> None:
    cluster = config["cluster"]
    source_dir = Path(__file__).resolve().parent
    python_command = str(cluster.get("python", "python"))
    concurrency = max(1, int(cluster.get("array_concurrency", 4)))
    cpus = max(1, int(cluster.get("cpus_per_task", 1)))
    memory = str(cluster.get("memory", "32gb"))
    walltime = str(cluster.get("walltime", "24:00:00"))
    tasks_path = output / "tasks.csv"
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    pbs = """#!/bin/bash
#PBS -N boson_peak_pilot
#PBS -l walltime={walltime}
#PBS -l mem={memory}
#PBS -l nodes=1:ppn={cpus}
#PBS -o {logs}
#PBS -e {logs}
#PBS -V

set -euo pipefail
cd {source_dir}
export OMP_NUM_THREADS={cpus}
export OPENBLAS_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}
TASK_INDEX="${{TASK_INDEX_OVERRIDE:-}}"
if [[ -z "$TASK_INDEX" ]]; then
  ARRAY_INDEX="${{PBS_ARRAY_INDEX:-${{PBS_ARRAYID:-}}}}"
  if [[ -z "$ARRAY_INDEX" ]]; then
    echo "None of TASK_INDEX_OVERRIDE, PBS_ARRAY_INDEX, or PBS_ARRAYID is set" >&2
    exit 2
  fi
  TASK_INDEX=$((ARRAY_INDEX - 1))
fi
{python_command} boson_peak_hpc.py ed --config {config_path} --task-table {tasks_path} --task-index "$TASK_INDEX" --resume
""".format(
        walltime=walltime, memory=memory, cpus=cpus, task_count=task_count,
        concurrency=concurrency, logs=str(logs.resolve()),
        source_dir=shell_quote(source_dir), python_command=python_command,
        config_path=shell_quote(config_snapshot.resolve()), tasks_path=shell_quote(tasks_path.resolve()),
    )
    slurm_memory = memory.lower().replace("gb", "G").replace("mb", "M")
    slurm = """#!/bin/bash
#SBATCH --job-name=boson_peak_pilot
#SBATCH --time={walltime}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus}
#SBATCH --array=0-{last_task}%{concurrency}
#SBATCH --output={logs}/slurm_%A_%a.out
#SBATCH --error={logs}/slurm_%A_%a.err

set -euo pipefail
cd {source_dir}
export OMP_NUM_THREADS={cpus}
export OPENBLAS_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}
{python_command} boson_peak_hpc.py ed --config {config_path} --task-table {tasks_path} --task-index "$SLURM_ARRAY_TASK_ID" --resume
""".format(
        walltime=walltime, memory=slurm_memory, cpus=cpus, last_task=task_count - 1,
        concurrency=concurrency, logs=str(logs.resolve()), source_dir=shell_quote(source_dir),
        python_command=python_command, config_path=shell_quote(config_snapshot.resolve()),
        tasks_path=shell_quote(tasks_path.resolve()),
    )
    (output / "job_array.pbs").write_text(pbs, encoding="utf-8")
    (output / "job_array.slurm").write_text(slurm, encoding="utf-8")

    launcher = """#!/bin/bash
# Kruk may reject throttled Torque arrays (for example 1-N%limit).  Try the
# compact array first, then preserve the concurrency limit with dependency chains.
set -euo pipefail

WORKER={worker}
TASK_COUNT={task_count}
CONCURRENCY={concurrency}
ARRAY_SPEC="1-${{TASK_COUNT}}%${{CONCURRENCY}}"

echo "Trying PBS array: $ARRAY_SPEC"
set +e
ARRAY_JOB=$(qsub -t "$ARRAY_SPEC" "$WORKER" 2>&1)
ARRAY_STATUS=$?
set -e
if [[ $ARRAY_STATUS -eq 0 ]]; then
  echo "Submitted array job: $ARRAY_JOB"
  exit 0
fi

echo "PBS rejected the throttled array: $ARRAY_JOB" >&2
echo "Falling back to $CONCURRENCY serialized dependency chains." >&2
MANIFEST={manifest}
printf 'task_index\tchain\tjob_id\tdepends_on\n' >"$MANIFEST"
declare -a CHAIN_TAILS=()
for ((TASK_INDEX=0; TASK_INDEX<TASK_COUNT; TASK_INDEX++)); do
  CHAIN=$((TASK_INDEX % CONCURRENCY))
  PREVIOUS="${{CHAIN_TAILS[$CHAIN]:-}}"
  QSUB_ARGS=(-v "TASK_INDEX_OVERRIDE=$TASK_INDEX")
  if [[ -n "$PREVIOUS" ]]; then
    QSUB_ARGS+=(-W "depend=afterany:$PREVIOUS")
  fi
  JOB_ID=$(qsub "${{QSUB_ARGS[@]}}" "$WORKER")
  CHAIN_TAILS[$CHAIN]="$JOB_ID"
  printf '%s\t%s\t%s\t%s\n' "$TASK_INDEX" "$CHAIN" "$JOB_ID" "${{PREVIOUS:--}}" >>"$MANIFEST"
  if (( (TASK_INDEX + 1) % 20 == 0 || TASK_INDEX + 1 == TASK_COUNT )); then
    echo "Submitted $((TASK_INDEX + 1))/$TASK_COUNT worker jobs..."
  fi
done
echo "Submission manifest: $MANIFEST"
echo "At most $CONCURRENCY workers can run simultaneously."
""".format(
        worker=shell_quote((output / "job_array.pbs").resolve()),
        task_count=task_count,
        concurrency=concurrency,
        manifest=shell_quote((logs / "pbs_submission.tsv").resolve()),
    )
    (output / "submit_pbs.sh").write_text(launcher, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace task metadata, never raw realization files")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override a dotted configuration key")
    cli.add_scan_arguments(parser, include_cluster=True)
    args = parser.parse_args()

    config = cli.apply_scan_arguments(
        core.apply_config_overrides(core.load_config(args.config), args.set), args
    )
    output = args.output.resolve()
    tasks_file = output / "tasks.csv"
    if tasks_file.exists() and not args.force:
        raise SystemExit("{} already exists; use --force only to regenerate task metadata".format(tasks_file))
    for folder in (output, output / "raw", output / "progress", output / "failed", output / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    rows = task_rows(config)
    core.atomic_write_csv(output / "tasks.csv", rows)
    snapshot = output / "config_resolved.json"
    core.atomic_write_json(snapshot, config)
    core.atomic_write_csv(output / "basis_metadata.csv", basis_metadata(config))
    theory_tables = engine.build_theory_tables(
        config,
        sectors=core.theory_sectors(config),
        W_values=sorted(set(core.theory_W_values(config)) | set(float(x) for x in config["ed"]["W_values"])),
    )
    engine.write_theory_tables(output, theory_tables)
    core.atomic_write_json(output / "run_manifest.json", core.run_manifest(config, len(rows), Path(__file__).resolve().parent))
    write_scripts(output, snapshot, len(rows), config)

    diagonalizations = sum(int(row["realizations_in_block"]) * int(row["U_points"]) for row in rows)
    print("Prepared tasks: {}".format(len(rows)))
    print("Planned diagonalizations: {}".format(diagonalizations))
    print("Task table: {}".format(tasks_file))
    print("PBS submit: bash {}".format(output / "submit_pbs.sh"))
    print("SLURM array: sbatch {}".format(output / "job_array.slurm"))
    print("Run the dry-run before submission:")
    print("python boson_peak_hpc.py ed --config {} --task-table {} --dry-run".format(snapshot, tasks_file))


if __name__ == "__main__":
    main()
