#!/usr/bin/env python3
"""HPC engine for the bosonic entropy-maximum validation.

Subcommands
-----------
theory
    Evaluate the independent compensating-channel prediction without ED.
ed
    Run one task-table block or print a complete resource dry-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import boson_peak_core as core
import boson_peak_cli as cli


def number_tag(value: float) -> str:
    text = "{:.12g}".format(float(value))
    return text.replace("-", "m").replace(".", "p")


def group_name(L: int, N: int, nmax: int, W_over_t: float) -> str:
    return "L{}_N{}_nmax{}_W{}".format(L, N, nmax, number_tag(W_over_t))


def result_path(run_dir: Path, task: Dict[str, Any], realization: int) -> Path:
    return run_dir / "raw" / group_name(
        int(task["L"]), int(task["N"]), int(task["nmax"]), float(task["W_over_t"])
    ) / "realization_{:06d}.npz".format(realization)


def progress_path(run_dir: Path, task_index: int) -> Path:
    return run_dir / "progress" / "task_{:06d}.json".format(task_index)


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tasks: List[Dict[str, Any]] = []
    for row in rows:
        tasks.append(
            {
                "task_index": int(row["task_index"]),
                "L": int(row["L"]),
                "N": int(row["N"]),
                "nmax": int(row["nmax"]),
                "W_over_t": float(row["W_over_t"]),
                "realization_start": int(row["realization_start"]),
                "realization_stop": int(row["realization_stop"]),
                "U_values_over_t": [float(x) for x in json.loads(row["U_values_over_t_json"])],
                "core_hash": row["core_hash"],
            }
        )
    return tasks


def core_hash(config: Dict[str, Any], L: int, N: int, nmax: int, W_over_t: float) -> str:
    payload = {
        "program_version": core.PROGRAM_VERSION,
        "schema_version": core.SCHEMA_VERSION,
        "model": config["model"],
        "eigensolver": config["eigensolver"],
        "master_seed": int(config["ed"].get("master_seed", 20260907)),
        "L": L,
        "N": N,
        "nmax": nmax,
        "W_over_t": W_over_t,
    }
    return core.sha256_json(payload)


def theory_for_sector_W(
    config: Dict[str, Any], L: int, N: int, nmax: int, W_over_t: float
) -> Tuple[List[core.Channel], Dict[str, float], Dict[str, float]]:
    t = float(config["model"].get("t", 1.0))
    channels = core.channel_table(L, N, nmax, t)
    coefficients = core.theory_coefficients(channels)
    peak = core.global_theory_peak(W_over_t * t, channels, t, config.get("theory", {}))
    return channels, coefficients, peak


def build_theory_tables(
    config: Dict[str, Any], sectors: Optional[Sequence[Tuple[int, int, int]]] = None,
    W_values: Optional[Sequence[float]] = None, report_progress: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sectors = list(sectors if sectors is not None else core.theory_sectors(config))
    W_values = list(W_values if W_values is not None else core.theory_W_values(config))
    channels_output: List[Dict[str, Any]] = []
    coefficients_output: List[Dict[str, Any]] = []
    peaks_output: List[Dict[str, Any]] = []
    total = len(sectors) * len(W_values)
    completed = 0
    for L, N, nmax in sectors:
        channels = core.channel_table(L, N, nmax, float(config["model"].get("t", 1.0)))
        channels_output.extend(core.channel_rows(channels))
        coefficients = core.theory_coefficients(channels)
        coefficients_output.append({"L": L, "N": N, "nmax": nmax, **coefficients})
        sector_rows: List[Dict[str, Any]] = []
        for W_over_t in W_values:
            peak = core.global_theory_peak(
                W_over_t * float(config["model"].get("t", 1.0)),
                channels,
                float(config["model"].get("t", 1.0)),
                config.get("theory", {}),
            )
            row = {
                "L": L,
                "N": N,
                "nmax": nmax,
                "W_over_t": W_over_t,
                **peak,
                "U_small_W_over_t": coefficients["A"] * W_over_t + coefficients["B"] * W_over_t**3,
                "U_large_W_over_t": coefficients["C"] * math.sqrt(W_over_t),
                "small_W_regime": W_over_t / coefficients["Gamma_min_over_t"] < 0.7,
                "large_W_regime": coefficients["Gamma_max_over_t"] / W_over_t < 0.3 if W_over_t > 0 else False,
            }
            sector_rows.append(row)
            completed += 1
            if report_progress and (completed == total or completed % max(1, total // 20) == 0):
                print("Theory: {:.1f}% ({}/{})".format(100.0 * completed / total, completed, total), flush=True)
        positive_rows = [row for row in sector_rows if row["W_over_t"] > 0 and row["U_M_star_over_t"] > 0]
        if len(positive_rows) >= 3:
            logW = np.log([row["W_over_t"] for row in positive_rows])
            logU = np.log([row["U_M_star_over_t"] for row in positive_rows])
            beta = np.gradient(logU, logW)
            for row, value in zip(positive_rows, beta):
                row["beta_eff"] = float(value)
        for row in sector_rows:
            row.setdefault("beta_eff", float("nan"))
        peaks_output.extend(sector_rows)
    return channels_output, coefficients_output, peaks_output


def write_theory_tables(output_dir: Path, tables: Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    channels, coefficients, peaks = tables
    core.atomic_write_csv(output_dir / "channels.csv", channels)
    core.atomic_write_csv(output_dir / "theory_coefficients.csv", coefficients)
    core.atomic_write_csv(output_dir / "theory_peaks.csv", peaks)


def command_theory(args: argparse.Namespace) -> None:
    config = core.apply_config_overrides(core.load_config(args.config), args.set)
    output = args.output_dir
    if output is None:
        output = Path(config.get("paths", {}).get("theory_output_dir", "theory_results"))
        if not output.is_absolute():
            output = args.config.resolve().parent / output
    tables = build_theory_tables(config)
    write_theory_tables(output, tables)
    print("Saved theory tables: {}".format(output.resolve()))


def ed_U_grid(config: Dict[str, Any], L: int, N: int, nmax: int, W_over_t: float) -> np.ndarray:
    _, _, peak = theory_for_sector_W(config, L, N, nmax, W_over_t)
    return core.adaptive_U_grid(
        peak["U_M_star_over_t"], float(config["model"].get("t", 1.0)), config["ed"].get("U_grid", {})
    ) / float(config["model"].get("t", 1.0))


def estimated_memory_mb(structure: core.BosonStructure, config: Dict[str, Any]) -> float:
    dim, nnz = structure.dim, int(structure.hopping.nnz)
    solver = config["eigensolver"]
    nev = min(int(solver.get("nev", 36)), dim)
    csr_bytes = nnz * (8 + 4) + (dim + 1) * 4
    vectors_bytes = dim * nev * 8
    backend = solver.get("backend", "auto")
    dense = backend == "dense" or (backend == "auto" and dim <= int(config["model"].get("dense_max", 5000)))
    if dense:
        estimate = 3.5 * dim * dim * 8 + vectors_bytes + csr_bytes
    else:
        estimate = float(solver.get("shift_invert_memory_factor", 18.0)) * csr_bytes + 3.0 * vectors_bytes
    return estimate / 1024.0**2


def dry_run(config: Dict[str, Any], task_table: Optional[Path]) -> None:
    if task_table is not None:
        tasks = load_tasks(task_table)
        combinations = sorted(set((t["L"], t["N"], t["nmax"], t["W_over_t"]) for t in tasks))
        block_lookup = {}
        for task in tasks:
            key = (task["L"], task["N"], task["nmax"], task["W_over_t"])
            block_lookup.setdefault(key, []).append(task)
    else:
        combinations = [
            (L, N, nmax, W)
            for L, N, nmax in core.ed_sectors(config)
            for W in sorted(set(float(x) for x in config["ed"]["W_values"]))
        ]
        block_lookup = {}
    cached: Dict[Tuple[int, int, int], core.BosonStructure] = {}
    grand_diagonalizations = 0
    print("L  N  nmax  W/t       D       nnz   R  nU  diagonalizations  estimated_MB")
    print("-" * 88)
    metadata_rows = []
    for L, N, nmax, W in combinations:
        key = (L, N, nmax)
        if key not in cached:
            cached[key] = core.build_structure(L, N, nmax, float(config["model"].get("t", 1.0)))
        structure = cached[key]
        U_values = ed_U_grid(config, L, N, nmax, W)
        if block_lookup:
            realization_count = sum(t["realization_stop"] - t["realization_start"] for t in block_lookup[(L, N, nmax, W)])
        else:
            realization_count = int(config["ed"]["realizations"])
        diagonalizations = realization_count * U_values.size
        grand_diagonalizations += diagonalizations
        memory = estimated_memory_mb(structure, config)
        print(
            "{:<2d} {:<2d} {:<5d} {:<6g} {:>7d} {:>9d} {:>3d} {:>3d} {:>17d} {:>13.1f}".format(
                L, N, nmax, W, structure.dim, structure.hopping.nnz,
                realization_count, U_values.size, diagonalizations, memory,
            )
        )
        metadata_rows.append(
            {
                "L": L, "N": N, "nmax": nmax, "W_over_t": W,
                "D": structure.dim, "hopping_nnz": int(structure.hopping.nnz),
                "realizations": realization_count, "U_points": int(U_values.size),
                "diagonalizations": diagonalizations, "estimated_peak_memory_MB": memory,
            }
        )
    print("Total diagonalizations: {}".format(grand_diagonalizations))
    timing_file = task_table.parent / "timing_and_memory.csv" if task_table else None
    if timing_file is not None and timing_file.exists():
        try:
            with timing_file.open(newline="", encoding="utf-8") as handle:
                timings = [float(row["solver_time_s"]) for row in csv.DictReader(handle) if row.get("solver_time_s")]
            if timings:
                print("Existing median solver time: {:.2f} s".format(float(np.median(timings))))
                print("Projected solver time: {:.2f} h".format(float(np.median(timings)) * grand_diagonalizations / 3600.0))
        except (OSError, ValueError, KeyError):
            pass


def empty_result_arrays(structure: core.BosonStructure, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    nev = min(int(config["eigensolver"].get("nev", 36)), structure.dim)
    nq = int(np.max(structure.interaction_q, initial=0)) + 1
    return {
        "U_over_t": np.empty(0, dtype=float),
        "energies_over_t": np.empty((0, nev), dtype=float),
        "relative_residuals": np.empty((0, nev), dtype=float),
        "entropy": np.empty(0), "entropy_norm": np.empty(0),
        "entropy2": np.empty(0), "entropy2_norm": np.empty(0),
        "IPR": np.empty(0), "gap_ratio": np.empty(0), "gap_ratio_count": np.empty(0, dtype=int),
        "Q_mean": np.empty(0), "Q_variance": np.empty(0),
        "H_Q": np.empty(0), "S_intra_Q": np.empty(0), "entropy_identity_error": np.empty(0),
        "P_Q": np.empty((0, nq)), "sample_mixing": np.empty(0),
        "E_min_over_t": np.empty(0), "E_max_over_t": np.empty(0), "sigma_over_t": np.empty(0),
        "solver_time_s": np.empty(0), "peak_memory_MB": np.empty(0),
        "residual_max": np.empty(0), "orthogonality_error": np.empty(0),
        "solver_attempts": np.empty(0, dtype=int), "selected_states": np.empty(0, dtype=int),
        "solver_backend": np.empty(0, dtype="U32"),
        "solver_retry_log_json": np.empty(0, dtype="U8192"),
    }


def load_existing_result(path: Path, expected_hash: str, structure: core.BosonStructure, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    if not path.exists():
        return {}, empty_result_arrays(structure, config)
    with np.load(str(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("core_hash") != expected_hash:
            raise RuntimeError("Existing result has an incompatible core hash: {}".format(path))
        arrays = {name: np.array(data[name], copy=True) for name in data.files if name != "metadata_json" and name != "eta"}
        arrays["eta"] = np.array(data["eta"], copy=True)
    return metadata, arrays


def append_row(arrays: Dict[str, np.ndarray], row: Dict[str, Any]) -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    for key, existing in arrays.items():
        if key == "eta":
            output[key] = existing
            continue
        value = np.asarray(row[key])
        if existing.ndim == 1:
            output[key] = np.concatenate((existing, value.reshape(1)))
        else:
            output[key] = np.concatenate((existing, value.reshape((1,) + existing.shape[1:])), axis=0)
    return output


def sort_result_arrays(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    order = np.argsort(arrays["U_over_t"], kind="stable")
    output = {"eta": arrays["eta"]}
    for key, value in arrays.items():
        if key == "eta":
            continue
        output[key] = value[order]
    return output


def solve_one_U(
    structure: core.BosonStructure,
    channels: Sequence[core.Channel],
    epsilon: np.ndarray,
    U_over_t: float,
    W_over_t: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    t = float(config["model"].get("t", 1.0))
    H = core.hamiltonian(structure, U_over_t * t, epsilon)
    solution = core.solve_central_spectrum(H, config["eigensolver"], int(config["model"].get("dense_max", 5000)))
    _, selected_vectors = core.select_central_states(solution, int(config["eigensolver"].get("n_states", 20)))
    observables = core.state_observables(selected_vectors, structure.interaction_q, structure.dim)
    tolerance = float(config["analysis"].get("entropy_identity_tol", 1.0e-9))
    if observables["identity_error"] > tolerance:
        raise RuntimeError("Entropy decomposition identity failed: {:.3e}".format(observables["identity_error"]))
    r_mean, r_count = core.adjacent_gap_ratio(
        solution["energies"], int(config["eigensolver"].get("gap_edge_discard", 2))
    )
    return {
        "U_over_t": U_over_t,
        "energies_over_t": np.asarray(solution["energies"]) / t,
        "relative_residuals": np.asarray(solution["residuals"]),
        "entropy": float(np.mean(observables["entropy"])),
        "entropy_norm": float(np.mean(observables["entropy_norm"])),
        "entropy2": float(np.mean(observables["entropy2"])),
        "entropy2_norm": float(np.mean(observables["entropy2_norm"])),
        "IPR": float(np.mean(observables["IPR"])),
        "gap_ratio": r_mean,
        "gap_ratio_count": r_count,
        "Q_mean": float(np.mean(observables["Q_mean"])),
        "Q_variance": float(np.mean(observables["Q_var"])),
        "H_Q": float(np.mean(observables["H_Q"])),
        "S_intra_Q": float(np.mean(observables["S_intra_Q"])),
        "entropy_identity_error": float(observables["identity_error"]),
        "P_Q": np.mean(observables["P_Q"], axis=1),
        "sample_mixing": float(core.sample_mixing(np.asarray([U_over_t * t]), epsilon, channels, t)[0]),
        "E_min_over_t": float(solution["E_min"]) / t,
        "E_max_over_t": float(solution["E_max"]) / t,
        "sigma_over_t": float(solution["sigma"]) / t,
        "solver_time_s": float(solution["elapsed_s"]),
        "peak_memory_MB": core.current_peak_memory_mb(),
        "residual_max": float(solution["residual_max"]),
        "orthogonality_error": float(solution["orthogonality_error"]),
        "solver_attempts": int(solution["solver_attempts"]),
        "selected_states": int(selected_vectors.shape[1]),
        "solver_backend": str(solution["solver_backend"]),
        "solver_retry_log_json": core.canonical_json(solution["solver_retry_log"]),
    }


def realization_is_complete(path: Path, expected_hash: str, U_values: Sequence[float]) -> bool:
    try:
        with np.load(str(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            existing = np.asarray(data["U_over_t"], dtype=float)
        return metadata.get("core_hash") == expected_hash and all(
            np.any(np.isclose(existing, U, atol=1.0e-10, rtol=0.0)) for U in U_values
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def process_realization(
    run_dir: Path,
    task: Dict[str, Any],
    realization: int,
    structure: core.BosonStructure,
    channels: Sequence[core.Channel],
    config: Dict[str, Any],
    resume: bool,
) -> str:
    target = result_path(run_dir, task, realization)
    expected_hash = core_hash(config, task["L"], task["N"], task["nmax"], task["W_over_t"])
    requested_U = sorted(set(float(x) for x in task["U_values_over_t"]))
    if resume and realization_is_complete(target, expected_hash, requested_U):
        return "SKIP"
    metadata, arrays = load_existing_result(target, expected_hash, structure, config)
    master_seed = int(config["ed"].get("master_seed", 20260907))
    expected_eta = core.disorder_eta(master_seed, int(task["L"]), realization)
    if "eta" in arrays and arrays["eta"].size:
        if not np.array_equal(arrays["eta"], expected_eta):
            raise RuntimeError("Stored eta is inconsistent with the deterministic seed")
    else:
        arrays["eta"] = expected_eta
    t = float(config["model"].get("t", 1.0))
    epsilon = task["W_over_t"] * t * expected_eta
    existing_U = np.asarray(arrays["U_over_t"], dtype=float)
    missing = [U for U in requested_U if not np.any(np.isclose(existing_U, U, atol=1.0e-10, rtol=0.0))]
    validation = metadata.get("dense_sparse_validation", {})
    if (
        missing and int(task["L"]) == 6 and realization == 0
        and bool(config["ed"].get("validate_L6_dense_sparse", True)) and not validation
    ):
        validation = core.validate_dense_sparse(structure, missing[0] * t, epsilon, config["eigensolver"])
        limits = config["ed"].get("L6_validation_tolerances", {})
        if validation["energy_error"] > float(limits.get("energy", 1.0e-8)):
            raise RuntimeError("L6 sparse/dense energy validation failed")
        if validation["entropy_error"] > float(limits.get("entropy", 1.0e-7)):
            raise RuntimeError("L6 sparse/dense entropy validation failed")
    for U in missing:
        row = solve_one_U(structure, channels, epsilon, U, task["W_over_t"], config)
        arrays = append_row(arrays, row)
        arrays = sort_result_arrays(arrays)
        completed_now = all(
            np.any(np.isclose(arrays["U_over_t"], wanted, atol=1.0e-10, rtol=0.0)) for wanted in requested_U
        )
        metadata = {
            "program_version": core.PROGRAM_VERSION,
            "schema_version": core.SCHEMA_VERSION,
            "core_hash": expected_hash,
            "full_config_hash": core.sha256_json(config),
            "status": "complete" if completed_now else "partial",
            "L": int(task["L"]), "N": int(task["N"]), "nmax": int(task["nmax"]),
            "W_over_t": float(task["W_over_t"]), "realization": int(realization),
            "master_seed": master_seed, "boundary": "open",
            "interaction_convention": "U_sum_n_n_minus_1",
            "disorder_convention": "epsilon_uniform_minus_W_to_W",
            "requested_U_values_over_t": requested_U,
            "dense_sparse_validation": validation,
            "updated_unix": time.time(),
        }
        core.atomic_save_npz(target, metadata_json=np.asarray(core.canonical_json(metadata)), **arrays)
        print(
            "  realization {:06d}, U/t={:.8g}, S/lnD={:.6f}, <r>={:.6f}, residual={:.2e}, {:.2f}s".format(
                realization, U, row["entropy_norm"], row["gap_ratio"], row["residual_max"], row["solver_time_s"]
            ),
            flush=True,
        )
    return "OK" if missing else "SKIP"


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_i = int(round(seconds))
    return "{:02d}:{:02d}:{:02d}".format(seconds_i // 3600, (seconds_i % 3600) // 60, seconds_i % 60)


def write_failure(run_dir: Path, task: Dict[str, Any], realization: int, exc: BaseException) -> None:
    target = run_dir / "failed" / "task_{:06d}_realization_{:06d}.json".format(task["task_index"], realization)
    core.atomic_write_json(
        target,
        {
            "task_index": task["task_index"], "realization": realization,
            "L": task["L"], "N": task["N"], "nmax": task["nmax"], "W_over_t": task["W_over_t"],
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "time_unix": time.time(),
        },
    )


def process_task(config: Dict[str, Any], task_table: Path, task_index: int, resume: bool) -> None:
    tasks = load_tasks(task_table)
    matching = [task for task in tasks if task["task_index"] == task_index]
    if len(matching) != 1:
        raise SystemExit("Task index {} not found exactly once".format(task_index))
    task = matching[0]
    expected_hash = core_hash(config, task["L"], task["N"], task["nmax"], task["W_over_t"])
    if task["core_hash"] != expected_hash:
        raise RuntimeError("Task table is incompatible with the configuration")
    run_dir = task_table.resolve().parent
    started = time.perf_counter()
    structure = core.build_structure(task["L"], task["N"], task["nmax"], float(config["model"].get("t", 1.0)))
    channels = core.channel_table(task["L"], task["N"], task["nmax"], float(config["model"].get("t", 1.0)))
    total = task["realization_stop"] - task["realization_start"]
    completed, skipped, failed = 0, 0, 0
    core.atomic_write_json(
        progress_path(run_dir, task_index),
        {"task_index": task_index, "state": "running", "completed": 0, "total": total, "failed": 0, "updated_unix": time.time()},
    )
    for realization in range(task["realization_start"], task["realization_stop"]):
        status = "ERROR"
        try:
            status = process_realization(run_dir, task, realization, structure, channels, config, resume)
            completed += 1
            skipped += int(status == "SKIP")
        except BaseException as exc:
            failed += 1
            write_failure(run_dir, task, realization, exc)
            print("ERROR realization {:06d}: {}".format(realization, exc), file=sys.stderr, flush=True)
        elapsed = time.perf_counter() - started
        done = completed + failed
        eta = elapsed / done * max(total - done, 0) if done else float("nan")
        progress = {
            "task_index": task_index, "state": "running", "completed": completed,
            "skipped": skipped, "failed": failed, "total": total,
            "L": task["L"], "N": task["N"], "nmax": task["nmax"], "W_over_t": task["W_over_t"],
            "elapsed_s": elapsed, "eta_s": eta, "last_realization": realization,
            "updated_unix": time.time(),
        }
        core.atomic_write_json(progress_path(run_dir, task_index), progress)
        print(
            "ED L={}, N={}, nmax={}, W/t={:g}: {:.2f}% ({}/{} realizations), time {}, ETA {}, last {:06d}, status {}".format(
                task["L"], task["N"], task["nmax"], task["W_over_t"],
                100.0 * done / total, completed, total, format_duration(elapsed), format_duration(eta), realization, status,
            ),
            flush=True,
        )
    final_state = "complete" if failed == 0 else "error"
    progress = json.loads(progress_path(run_dir, task_index).read_text(encoding="utf-8"))
    progress.update({"state": final_state, "updated_unix": time.time(), "elapsed_s": time.perf_counter() - started})
    core.atomic_write_json(progress_path(run_dir, task_index), progress)
    if failed:
        raise SystemExit("Task completed with {} failed realizations; inspect {}".format(failed, run_dir / "failed"))


def command_ed(args: argparse.Namespace) -> None:
    config = cli.apply_scan_arguments(
        core.apply_config_overrides(core.load_config(args.config), args.set), args
    )
    if args.dry_run:
        dry_run(config, args.task_table)
        return
    if args.task_table is None or args.task_index is None:
        raise SystemExit("ed requires --task-table and --task-index unless --dry-run is used")
    process_task(config, args.task_table, args.task_index, args.resume)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", action="version", version=core.PROGRAM_VERSION)
    commands = result.add_subparsers(dest="command", required=True)
    theory = commands.add_parser("theory", help="run independent channel theory")
    theory.add_argument("--config", type=Path, required=True)
    theory.add_argument("--output-dir", type=Path)
    theory.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override a dotted configuration key")
    ed = commands.add_parser("ed", help="run ED task or resource dry-run")
    ed.add_argument("--config", type=Path, required=True)
    ed.add_argument("--task-table", type=Path)
    ed.add_argument("--task-index", type=int)
    ed.add_argument("--resume", action="store_true")
    ed.add_argument("--dry-run", action="store_true")
    ed.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override a dotted configuration key")
    cli.add_scan_arguments(ed, include_cluster=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "theory":
        command_theory(args)
    elif args.command == "ed":
        command_ed(args)


if __name__ == "__main__":
    main()
