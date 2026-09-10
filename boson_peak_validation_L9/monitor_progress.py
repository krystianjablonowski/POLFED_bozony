#!/usr/bin/env python3
"""Aggregate durable progress records from all PBS/SLURM array blocks."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boson_peak_hpc as engine


def duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    value = int(round(seconds))
    return "{:02d}:{:02d}:{:02d}".format(value // 3600, (value % 3600) // 60, value % 60)


def read_progress(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "error", "completed": 0, "failed": 1, "elapsed_s": 0.0}


def snapshot(run_dir: Path) -> Dict[str, Any]:
    tasks = engine.load_tasks(run_dir / "tasks.csv")
    task_states = defaultdict(int)
    groups: Dict[Tuple[int, int, int, float], Dict[str, float]] = defaultdict(
        lambda: {"complete": 0.0, "total": 0.0, "failed": 0.0, "elapsed": 0.0}
    )
    last_file = None
    last_mtime = -1.0
    all_completed = 0
    all_total = 0
    all_failed = 0
    total_elapsed = 0.0
    for task in tasks:
        target = engine.progress_path(run_dir, task["task_index"])
        if target.exists():
            progress = read_progress(target)
            state = str(progress.get("state", "running"))
            mtime = target.stat().st_mtime
            if mtime > last_mtime:
                last_mtime, last_file = mtime, target
        else:
            progress = {"state": "pending", "completed": 0, "failed": 0, "elapsed_s": 0.0}
            state = "pending"
        if state not in ("pending", "running", "complete", "error"):
            state = "error"
        task_states[state] += 1
        total = task["realization_stop"] - task["realization_start"]
        complete = min(int(progress.get("completed", 0)), total)
        failed = int(progress.get("failed", 0))
        elapsed = float(progress.get("elapsed_s", 0.0))
        group = groups[(task["L"], task["N"], task["nmax"], task["W_over_t"])]
        group["complete"] += complete
        group["total"] += total
        group["failed"] += failed
        group["elapsed"] += elapsed
        all_completed += complete
        all_total += total
        all_failed += failed
        total_elapsed += elapsed
    failure_files = list((run_dir / "failed").glob("*.json")) if (run_dir / "failed").exists() else []
    per_realization = total_elapsed / max(all_completed + all_failed, 1)
    concurrency = max(1, task_states["running"])
    remaining = max(all_total - all_completed - all_failed, 0)
    overall_eta = per_realization * remaining / concurrency
    return {
        "tasks": dict(task_states), "groups": groups,
        "completed": all_completed, "total": all_total, "failed": all_failed,
        "failed_diagonalization_records": len(failure_files),
        "last_file": last_file, "last_mtime": last_mtime, "eta_s": overall_eta,
    }


def print_snapshot(run_dir: Path) -> bool:
    report = snapshot(run_dir)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print("Progress {}  {}".format(stamp, run_dir), flush=True)
    print("L  N  nmax  W/t    complete/total   percent   failures", flush=True)
    for (L, N, nmax, W), values in sorted(report["groups"].items()):
        percent = 100.0 * values["complete"] / values["total"] if values["total"] else 0.0
        print(
            "{:<2d} {:<2d} {:<5d} {:<6g} {:>7.0f}/{:<7.0f} {:>7.2f}% {:>8.0f}".format(
                L, N, nmax, W, values["complete"], values["total"], percent, values["failed"]
            ),
            flush=True,
        )
    states = report["tasks"]
    print(
        "Tasks pending={} running={} complete={} error={}".format(
            states.get("pending", 0), states.get("running", 0), states.get("complete", 0), states.get("error", 0)
        ),
        flush=True,
    )
    print(
        "Realizations {}/{} complete; failed={}; solver failure records={}; projected ETA={}".format(
            report["completed"], report["total"], report["failed"],
            report["failed_diagonalization_records"], duration(report["eta_s"]),
        ),
        flush=True,
    )
    if report["last_file"] is not None:
        age = max(0.0, time.time() - report["last_mtime"])
        print("Last update: {} ({:.0f}s ago)".format(report["last_file"], age), flush=True)
    print("", flush=True)
    return states.get("pending", 0) == 0 and states.get("running", 0) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--watch", type=float, default=0.0, metavar="SECONDS")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "tasks.csv").exists():
        raise SystemExit("Missing {}".format(run_dir / "tasks.csv"))
    while True:
        finished = print_snapshot(run_dir)
        if args.watch <= 0 or finished:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
