#!/usr/bin/env python3
"""Validate and merge atomic realization files without diagonalizing again."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import boson_peak_core as core
import boson_peak_hpc as engine


def atomic_dataframe_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent), text=True)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_dataframe_parquet(path: Path, frame: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, str(path))
        return True
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        notice = path.with_suffix(path.suffix + ".UNAVAILABLE.txt")
        notice.write_text(
            "Parquet engine unavailable: {}\nInstall pyarrow and rerun merge_results.py.\n".format(exc),
            encoding="utf-8",
        )
        return False


def load_npz(path: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    with np.load(str(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {name: np.array(data[name], copy=True) for name in data.files if name != "metadata_json"}
    return metadata, arrays


def expected_realizations(tasks: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int, int, float, int], Dict[str, Any]]:
    output: Dict[Tuple[int, int, int, float, int], Dict[str, Any]] = {}
    for task in tasks:
        for realization in range(task["realization_start"], task["realization_stop"]):
            key = (task["L"], task["N"], task["nmax"], task["W_over_t"], realization)
            if key in output:
                raise RuntimeError("Overlapping realization blocks in tasks.csv: {}".format(key))
            output[key] = task
    return output


def read_failures(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "failed").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            row["failure_file"] = str(path.relative_to(run_dir))
            row.pop("traceback", None)
            rows.append(row)
        except (OSError, json.JSONDecodeError):
            rows.append({"failure_file": str(path.relative_to(run_dir)), "error": "unreadable failure record"})
    return rows


def merge(run_dir: Path, allow_incomplete: bool) -> None:
    task_file = run_dir / "tasks.csv"
    if not task_file.exists():
        raise SystemExit("Missing task table: {}".format(task_file))
    tasks = engine.load_tasks(task_file)
    expected = expected_realizations(tasks)
    scalar_rows: List[Dict[str, Any]] = []
    mixing_rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []
    recovered_retry_rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    seen_keys = set()

    for key, task in sorted(expected.items()):
        L, N, nmax, W_over_t, realization = key
        path = engine.result_path(run_dir, task, realization)
        expected_hash = task["core_hash"]
        if not path.exists():
            missing.append({"L": L, "N": N, "nmax": nmax, "W_over_t": W_over_t, "realization": realization, "reason": "missing"})
            continue
        try:
            metadata, arrays = load_npz(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            missing.append({"L": L, "N": N, "nmax": nmax, "W_over_t": W_over_t, "realization": realization, "reason": "corrupt: " + str(exc)})
            continue
        if metadata.get("core_hash") != expected_hash:
            missing.append({"L": L, "N": N, "nmax": nmax, "W_over_t": W_over_t, "realization": realization, "reason": "incompatible hash"})
            continue
        available = np.asarray(arrays["U_over_t"], dtype=float)
        required = np.asarray(task["U_values_over_t"], dtype=float)
        absent = [float(U) for U in required if not np.any(np.isclose(available, U, atol=1.0e-10, rtol=0.0))]
        if absent:
            missing.append(
                {"L": L, "N": N, "nmax": nmax, "W_over_t": W_over_t, "realization": realization, "reason": "missing U: " + json.dumps(absent)}
            )
            continue
        baseline_candidates = np.flatnonzero(np.isclose(available, 0.0, atol=1.0e-10, rtol=0.0))
        baseline = float(arrays["entropy_norm"][baseline_candidates[0]]) if baseline_candidates.size else float("nan")
        for row_index, U_over_t in enumerate(available):
            scalar_key = (L, N, nmax, W_over_t, realization, round(float(U_over_t), 10))
            if scalar_key in seen_keys:
                raise RuntimeError("Duplicate merged key: {}".format(scalar_key))
            seen_keys.add(scalar_key)
            common = {
                "L": L, "N": N, "nmax": nmax, "filling": N / L,
                "W_over_t": W_over_t, "realization": realization, "U_over_t": float(U_over_t),
            }
            scalar_rows.append(
                {
                    **common,
                    "entropy": float(arrays["entropy"][row_index]),
                    "entropy_norm": float(arrays["entropy_norm"][row_index]),
                    "delta_entropy_norm": float(arrays["entropy_norm"][row_index]) - baseline,
                    "entropy2": float(arrays["entropy2"][row_index]),
                    "entropy2_norm": float(arrays["entropy2_norm"][row_index]),
                    "IPR": float(arrays["IPR"][row_index]),
                    "gap_ratio": float(arrays["gap_ratio"][row_index]),
                    "gap_ratio_count": int(arrays["gap_ratio_count"][row_index]),
                    "Q_mean": float(arrays["Q_mean"][row_index]),
                    "Q_variance": float(arrays["Q_variance"][row_index]),
                    "H_Q": float(arrays["H_Q"][row_index]),
                    "S_intra_Q": float(arrays["S_intra_Q"][row_index]),
                    "entropy_identity_error": float(arrays["entropy_identity_error"][row_index]),
                    "selected_states": int(arrays["selected_states"][row_index]),
                }
            )
            mixing_rows.append({**common, "sample_mixing": float(arrays["sample_mixing"][row_index])})
            timing_rows.append(
                {
                    **common,
                    "solver_backend": str(arrays["solver_backend"][row_index]),
                    "solver_time_s": float(arrays["solver_time_s"][row_index]),
                    "peak_memory_MB": float(arrays["peak_memory_MB"][row_index]),
                    "residual_max": float(arrays["residual_max"][row_index]),
                    "orthogonality_error": float(arrays["orthogonality_error"][row_index]),
                    "solver_attempts": int(arrays["solver_attempts"][row_index]),
                    "E_min_over_t": float(arrays["E_min_over_t"][row_index]),
                    "E_max_over_t": float(arrays["E_max_over_t"][row_index]),
                    "sigma_over_t": float(arrays["sigma_over_t"][row_index]),
                    "solver_retry_log_json": (
                        str(arrays["solver_retry_log_json"][row_index])
                        if "solver_retry_log_json" in arrays else "[]"
                    ),
                }
            )
            if "solver_retry_log_json" in arrays:
                try:
                    retry_log = json.loads(str(arrays["solver_retry_log_json"][row_index]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    retry_log = []
                for retry in retry_log:
                    recovered_retry_rows.append(
                        {
                            **common,
                            "status": "recovered_retry",
                            "attempt": retry.get("attempt"),
                            "sigma": retry.get("sigma"),
                            "ncv": retry.get("ncv"),
                            "error_type": retry.get("error_type"),
                            "error": retry.get("error"),
                        }
                    )

    core.atomic_write_csv(run_dir / "incomplete_realizations.csv", missing)
    failures = read_failures(run_dir) + recovered_retry_rows
    core.atomic_write_csv(run_dir / "failed_tasks.csv", failures)
    if missing and not allow_incomplete:
        raise SystemExit(
            "Merge stopped: {} incomplete realization(s). Inspect {} or rerun with --allow-incomplete.".format(
                len(missing), run_dir / "incomplete_realizations.csv"
            )
        )
    if not scalar_rows:
        raise SystemExit("No complete realization rows found")
    scalar_frame = pd.DataFrame(scalar_rows).sort_values(["L", "N", "nmax", "W_over_t", "realization", "U_over_t"])
    mixing_frame = pd.DataFrame(mixing_rows).sort_values(["L", "N", "nmax", "W_over_t", "realization", "U_over_t"])
    timing_frame = pd.DataFrame(timing_rows).sort_values(["L", "N", "nmax", "W_over_t", "realization", "U_over_t"])
    atomic_dataframe_csv(run_dir / "ed_observables_by_realization.csv", scalar_frame)
    atomic_dataframe_csv(run_dir / "sample_mixing_curves.csv", mixing_frame)
    atomic_dataframe_csv(run_dir / "timing_and_memory.csv", timing_frame)
    parquet_ed = atomic_dataframe_parquet(run_dir / "ed_observables_by_realization.parquet", scalar_frame)
    parquet_mixing = atomic_dataframe_parquet(run_dir / "sample_mixing_curves.parquet", mixing_frame)
    print("Merged realization-U rows: {}".format(len(scalar_frame)))
    print("Incomplete realizations: {}".format(len(missing)))
    print("Recorded solver failures: {}".format(len(failures)))
    print("Parquet outputs: {}".format("OK" if parquet_ed and parquet_mixing else "pyarrow unavailable; CSV fallback saved"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    merge(args.run_dir.resolve(), args.allow_incomplete)


if __name__ == "__main__":
    main()
