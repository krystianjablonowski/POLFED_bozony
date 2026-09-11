#!/usr/bin/env python3
"""Cluster CLI for the Bose-Hubbard depth-one star model."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from star_model_core import (
    VERSION,
    build_fock_graph,
    compute_observables,
    deterministic_seed,
    microcanonical_indices,
    verify_detuning,
)


OBSERVABLES = ("M_sum", "M_per_edge", "S2_sum", "S2_per_edge", "Sstar_mean")
PRIMARY_OBSERVABLES = ("M_sum", "S2_sum", "Sstar_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "init", "status", "aggregate", "merge-sectors"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name == "init":
            p.add_argument("--force", action="store_true")
        if name == "aggregate":
            p.add_argument("--sector-id", type=int, help="Aggregate only this zero-based sector into its own folder.")
            p.add_argument("--fast", action="store_true", help="Skip very large raw/channel/distribution CSV exports.")
    p = sub.add_parser("worker")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--task-id", type=int, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def grid(spec: dict) -> np.ndarray:
    start, stop, step = float(spec["start"]), float(spec["stop"]), float(spec["step"])
    if step <= 0 or stop < start:
        raise ValueError("Grid requires step>0 and stop>=start")
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1, dtype=float)
    if abs(values[-1] - stop) > 1e-9:
        raise ValueError(f"Grid interval [{start},{stop}] is not divisible by {step}")
    return values


def results_root(cfg: dict) -> Path:
    config_dir = Path(cfg["_config_path"]).parent
    root = Path(cfg["paths"]["results_dir"])
    return root if root.is_absolute() else config_dir / root


def task_rows(cfg: dict) -> list[dict[str, int | float]]:
    rows = []
    n_samples = int(cfg["sampling"]["n_disorder"])
    chunk = int(cfg["cluster"]["samples_per_task"])
    task_id = 0
    for sector_id, sector in enumerate(cfg["model"]["sectors"]):
        for W_id, W in enumerate(grid(cfg["grid"]["W"])):
            for first in range(0, n_samples, chunk):
                rows.append({
                    "task_id": task_id,
                    "sector_id": sector_id,
                    "L": int(sector["L"]),
                    "N": int(sector["N"]),
                    "nmax": int(sector["nmax"]),
                    "W_id": W_id,
                    "W": float(W),
                    "sample_first": first,
                    "sample_stop": min(n_samples, first + chunk),
                })
                task_id += 1
    return rows


def validate_config(cfg: dict) -> None:
    if cfg["model"].get("boundary") != "open":
        raise ValueError("This implementation currently requires open boundaries")
    if cfg["model"].get("interaction_convention") != "U_n_nminus1":
        raise ValueError("interaction_convention must be U_n_nminus1")
    if cfg["disorder"].get("distribution") != "uniform_minus_W_to_W":
        raise ValueError("distribution must be uniform_minus_W_to_W")
    if int(cfg["sampling"]["n_disorder"]) < 1 or int(cfg["sampling"]["n_configurations"]) < 1:
        raise ValueError("Sampling counts must be positive")
    grid(cfg["grid"]["U"])
    grid(cfg["grid"]["W"])
    for sector in cfg["model"]["sectors"]:
        L, N, nmax = int(sector["L"]), int(sector["N"]), int(sector["nmax"])
        graph = build_fock_graph(L, N, nmax, float(cfg["model"]["t"]))
        print(f"sector L={L} N={N} nmax={nmax}: dim={graph.dimension}, directed_edges={len(graph.edge_dst)}")
    print("Hamiltonian: -t hopping + U*sum_i n_i(n_i-1) + sum_i epsilon_i*n_i")
    print("Disorder: epsilon_i ~ Uniform[-W,W]; boundary=open")
    print(f"U points={len(grid(cfg['grid']['U']))}, W points={len(grid(cfg['grid']['W']))}, tasks={len(task_rows(cfg))}")


def atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def initialize(cfg: dict, force: bool) -> None:
    validate_config(cfg)
    root = results_root(cfg)
    tasks = task_rows(cfg)
    manifest = root / "tasks.csv"
    if manifest.exists() and not force:
        raise SystemExit(f"{manifest} already exists; use --force only to rewrite the manifest")
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "analysis" / "figures").mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)
    metadata = {key: value for key, value in cfg.items() if not key.startswith("_")}
    metadata.update({
        "program_version": VERSION,
        "star_branch_selection": "largest_central_overlap_average_ties",
        "full_many_body_diagonalization": False,
        "paired_disorder_note": "unit fields depend on sector and sample_id, not U or W",
    })
    (root / "configuration.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {manifest} with {len(tasks)} tasks")


def raw_path(root: Path, task_id: int) -> Path:
    return root / "raw" / f"task_{task_id:06d}.npz"


def run_worker(cfg: dict, task_id: int) -> None:
    tasks = task_rows(cfg)
    if not 0 <= task_id < len(tasks):
        raise SystemExit(f"task-id must be in [0,{len(tasks)-1}]")
    task = tasks[task_id]
    output = raw_path(results_root(cfg), task_id)
    if output.exists():
        print(f"SKIP complete output: {output}")
        return

    L, N, nmax = int(task["L"]), int(task["N"]), int(task["nmax"])
    W = float(task["W"])
    t_hop = float(cfg["model"]["t"])
    u_values = grid(cfg["grid"]["U"])
    sample_ids = np.arange(int(task["sample_first"]), int(task["sample_stop"]), dtype=np.int64)
    graph = build_fock_graph(L, N, nmax, t_hop)
    n_cfg = int(cfg["sampling"]["n_configurations"])
    energy_fraction = float(cfg["sampling"]["energy_window_fraction"])
    master_seed = int(cfg["disorder"]["master_seed"])
    tie_tol = float(cfg["numerics"]["tie_tolerance"])
    k_values = np.arange(-nmax, nmax + 1, dtype=np.int16)
    histogram_edges = np.linspace(
        0.0,
        math.log(2 * (L - 1) + 1),
        int(cfg["analysis"].get("star_histogram_bins", 40)) + 1,
    )
    shape = (len(sample_ids), len(u_values))
    data = {name: np.full(shape, np.nan) for name in OBSERVABLES}
    for name in ("Sstar_median", "Sstar_std", "Pstar_mean", "z_mean", "resonant_edges_mean", "number_of_configurations", "number_of_edges", "max_normalization_error"):
        data[name] = np.full(shape, np.nan)
    channel_M = np.zeros(shape + (len(k_values),), dtype=float)
    channel_S2 = np.zeros_like(channel_M)
    channel_fraction = np.zeros_like(channel_M)
    star_histogram = np.zeros(shape + (len(histogram_edges) - 1,), dtype=np.int32)
    started = time.monotonic()

    for sample_pos, sample_id in enumerate(sample_ids):
        seed = deterministic_seed(master_seed, L, N, nmax, int(sample_id))
        rng = np.random.default_rng(seed)
        unit_fields = rng.uniform(-1.0, 1.0, size=L)
        epsilon = W * unit_fields
        priority = rng.permutation(graph.dimension)
        priority_rank = np.empty(graph.dimension, dtype=np.int64)
        priority_rank[priority] = np.arange(graph.dimension)
        detuning_error = verify_detuning(graph, float(u_values[len(u_values) // 2]), epsilon, seed=seed & 0xFFFFFFFF)
        if detuning_error > float(cfg["numerics"]["detuning_tolerance"]):
            raise RuntimeError(f"detuning identity failed: error={detuning_error:.3e}")
        for u_pos, U in enumerate(u_values):
            energies = U * graph.interaction_count + graph.basis @ epsilon
            central = microcanonical_indices(energies, energy_fraction, n_cfg, priority_rank)
            obs = compute_observables(graph, float(U), epsilon, central, tie_tol)
            for name in data:
                data[name][sample_pos, u_pos] = float(obs[name])
            edge_total = max(1, int(obs["number_of_edges"]))
            star_histogram[sample_pos, u_pos], _ = np.histogram(obs["Sstar_local"], bins=histogram_edges)
            for key, value in obs["channel_M_sum"].items():
                k_pos = int(key) + nmax
                if 0 <= k_pos < len(k_values):
                    channel_M[sample_pos, u_pos, k_pos] = value / len(central)
            for key, value in obs["channel_S2_sum"].items():
                k_pos = int(key) + nmax
                if 0 <= k_pos < len(k_values):
                    channel_S2[sample_pos, u_pos, k_pos] = value / len(central)
            for key, value in obs["channel_count"].items():
                k_pos = int(key) + nmax
                if 0 <= k_pos < len(k_values):
                    channel_fraction[sample_pos, u_pos, k_pos] = value / edge_total
        elapsed = time.monotonic() - started
        done = sample_pos + 1
        remaining = elapsed / done * (len(sample_ids) - done)
        print(f"task {task_id}: sample {done}/{len(sample_ids)}, elapsed={elapsed:.1f}s, ETA={remaining:.1f}s", flush=True)

    payload: dict[str, object] = {
        "program_version": np.asarray(VERSION), "task_id": task_id,
        "L": L, "N": N, "nmax": nmax, "W": W, "W_id": int(task["W_id"]),
        "sample_ids": sample_ids, "U_values": u_values, "k_values": k_values,
        "channel_M_sum": channel_M, "channel_S2_sum": channel_S2, "channel_fraction": channel_fraction,
        "star_histogram": star_histogram, "star_histogram_edges": histogram_edges,
    }
    payload.update(data)
    atomic_savez(output, **payload)
    print(f"Saved {output}")


def status(cfg: dict) -> None:
    root = results_root(cfg)
    tasks = task_rows(cfg)
    missing = [int(row["task_id"]) for row in tasks if not raw_path(root, int(row["task_id"])).exists()]
    print(f"Complete: {len(tasks)-len(missing)}/{len(tasks)}; missing: {len(missing)}")
    if missing:
        ranges = []
        start = previous = missing[0]
        for value in missing[1:]:
            if value != previous + 1:
                ranges.append(str(start) if start == previous else f"{start}-{previous}")
                start = value
            previous = value
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        print("Missing task IDs: " + ",".join(ranges))


def peak_parabola(x: np.ndarray, y: np.ndarray, points: int = 5) -> dict[str, float | str]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3:
        return {"U_peak": math.nan, "peak_value": math.nan, "curvature": math.nan, "quality_flag": "too_few_points"}
    imax = int(np.argmax(y))
    if imax == 0 or imax == len(x) - 1:
        return {"U_peak": float(x[imax]), "peak_value": float(y[imax]), "curvature": math.nan, "quality_flag": "boundary"}
    count = min(max(3, points | 1), len(x))
    lo = max(0, imax - count // 2)
    hi = min(len(x), lo + count)
    lo = max(0, hi - count)
    a, b, c = np.polyfit(x[lo:hi], y[lo:hi], 2)
    if a >= 0:
        return {"U_peak": float(x[imax]), "peak_value": float(y[imax]), "curvature": float(2*a), "quality_flag": "nonnegative_curvature"}
    peak = float(-b / (2 * a))
    flag = "ok" if x[lo] <= peak <= x[hi - 1] else "vertex_outside_fit_window"
    return {"U_peak": peak, "peak_value": float(a*peak*peak+b*peak+c), "curvature": float(2*a), "quality_flag": flag}


def stable_peak_parabola(x: np.ndarray, y: np.ndarray, points: int, max_shift_steps: float) -> dict[str, float | str]:
    result = peak_parabola(x, y, points)
    if result["quality_flag"] != "ok" or len(x) < 4:
        return result
    step = float(np.median(np.diff(np.sort(np.unique(x)))))
    alternatives = [peak_parabola(x, y, count) for count in (3, 7) if count != points]
    valid = [alt for alt in alternatives if alt["quality_flag"] == "ok"]
    if valid and max(abs(float(alt["U_peak"]) - float(result["U_peak"])) for alt in valid) > max_shift_steps * step:
        result["quality_flag"] = "unstable_fit_window"
    return result


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else fieldnames
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def linear_statistics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {"intercept_b": math.nan, "slope_c": math.nan, "R2": math.nan, "RMSE": math.nan}
    if float(np.ptp(x)) <= 1e-14:
        return {"intercept_b": float(np.mean(y)), "slope_c": math.nan, "R2": math.nan,
                "RMSE": float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))}
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    residual = y - pred
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {"intercept_b": float(intercept), "slope_c": float(slope),
            "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
            "RMSE": float(np.sqrt(np.mean(residual * residual)))}


def setup_plots():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("WARNING matplotlib unavailable; skipping figures")
        return None


def sector_analysis_name(cfg: dict, sector_id: int) -> str:
    sector = cfg["model"]["sectors"][sector_id]
    return f"sector_{sector_id}_L{int(sector['L'])}_N{int(sector['N'])}_nmax{int(sector['nmax'])}"


def aggregate(cfg: dict, sector_id: int | None = None, fast: bool = False) -> None:
    root = results_root(cfg)
    tasks = task_rows(cfg)
    if sector_id is not None:
        if not 0 <= sector_id < len(cfg["model"]["sectors"]):
            raise SystemExit(f"sector-id must be in [0,{len(cfg['model']['sectors'])-1}]")
        tasks = [task for task in tasks if int(task["sector_id"]) == sector_id]
    missing = [row for row in tasks if not raw_path(root, int(row["task_id"])).exists()]
    if missing:
        raise SystemExit(f"Cannot aggregate: {len(missing)} task files are missing; run status")
    analysis = root / "analysis"
    if sector_id is not None:
        analysis = analysis / sector_analysis_name(cfg, sector_id)
    analysis.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[int, int, int, float], list[dict[str, object]]] = {}
    channel_groups: dict[tuple[int, int, int, float], dict[str, object]] = {}
    distribution_groups: dict[tuple[int, int, int, float], dict[str, object]] = {}
    raw_fields = ["L", "N", "nmax", "W_over_t", "U_over_t", "sample_id", "seed", *OBSERVABLES,
                  "Sstar_median", "Sstar_std", "Pstar_mean", "z_mean", "resonant_edges_mean",
                  "number_of_configurations", "energy_window"]
    raw_csv_path = Path(os.devnull) if fast else analysis / "raw_observables.csv.gz"
    with gzip.open(raw_csv_path, "wt", newline="", encoding="utf-8") as raw_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_fields)
        raw_writer.writeheader()
        for task in tasks:
            with np.load(raw_path(root, int(task["task_id"])), allow_pickle=False) as f:
                L, N, nmax, W = int(f["L"]), int(f["N"]), int(f["nmax"]), float(f["W"])
                U = f["U_values"].astype(float)
                samples = f["sample_ids"].astype(int)
                record = {name: f[name].copy() for name in OBSERVABLES}
                record["sample_ids"] = samples
                record["U"] = U
                groups.setdefault((L, N, nmax, W), []).append(record)
                if not fast:
                    group_key = (L, N, nmax, W)
                    if group_key not in channel_groups:
                        channel_groups[group_key] = {
                            "M": np.zeros_like(f["channel_M_sum"][0]),
                            "S2": np.zeros_like(f["channel_S2_sum"][0]),
                            "P": np.zeros_like(f["channel_fraction"][0]),
                            "count": 0, "k": f["k_values"].copy(), "U": U,
                        }
                        distribution_groups[group_key] = {
                            "counts": np.zeros_like(f["star_histogram"][0], dtype=np.int64),
                            "edges": f["star_histogram_edges"].copy(), "U": U,
                        }
                    channel_groups[group_key]["M"] += np.sum(f["channel_M_sum"], axis=0)
                    channel_groups[group_key]["S2"] += np.sum(f["channel_S2_sum"], axis=0)
                    channel_groups[group_key]["P"] += np.sum(f["channel_fraction"], axis=0)
                    channel_groups[group_key]["count"] += len(samples)
                    distribution_groups[group_key]["counts"] += np.sum(f["star_histogram"], axis=0, dtype=np.int64)
                    for si, sample_id in enumerate(samples):
                        seed = deterministic_seed(int(cfg["disorder"]["master_seed"]), L, N, nmax, int(sample_id))
                        for ui, value_u in enumerate(U):
                            raw_writer.writerow({
                                "L": L, "N": N, "nmax": nmax, "W_over_t": W / float(cfg["model"]["t"]),
                                "U_over_t": value_u / float(cfg["model"]["t"]), "sample_id": int(sample_id), "seed": seed,
                                **{name: float(f[name][si, ui]) for name in OBSERVABLES},
                                "Sstar_median": float(f["Sstar_median"][si, ui]), "Sstar_std": float(f["Sstar_std"][si, ui]),
                                "Pstar_mean": float(f["Pstar_mean"][si, ui]), "z_mean": float(f["z_mean"][si, ui]),
                                "resonant_edges_mean": float(f["resonant_edges_mean"][si, ui]),
                                "number_of_configurations": int(f["number_of_configurations"][si, ui]),
                                "energy_window": float(cfg["sampling"]["energy_window_fraction"]),
                            })
    channel_fields = ["L", "N", "nmax", "W_over_t", "U_over_t", "k", "M_k", "S2_k", "P_k"]
    channel_path = Path(os.devnull) if fast else analysis / "channels.csv.gz"
    with gzip.open(channel_path, "wt", newline="", encoding="utf-8") as channel_handle:
        channel_writer = csv.DictWriter(channel_handle, fieldnames=channel_fields)
        channel_writer.writeheader()
        for (L, N, nmax, W), group in sorted(channel_groups.items()):
            U, k_values = group["U"], group["k"].astype(int)
            means = {name: group[name] / int(group["count"]) for name in ("M", "S2", "P")}
            for ui, value_u in enumerate(U):
                for ki, kval in enumerate(k_values):
                    channel_writer.writerow({"L": L, "N": N, "nmax": nmax,
                                             "W_over_t": W / float(cfg["model"]["t"]),
                                             "U_over_t": value_u / float(cfg["model"]["t"]), "k": int(kval),
                                             "M_k": float(means["M"][ui, ki]), "S2_k": float(means["S2"][ui, ki]),
                                             "P_k": float(means["P"][ui, ki])})
    distribution_fields = ["L", "N", "nmax", "W_over_t", "U_over_t", "bin_left", "bin_right", "probability"]
    distribution_path = Path(os.devnull) if fast else analysis / "Sstar_distribution.csv.gz"
    with gzip.open(distribution_path, "wt", newline="", encoding="utf-8") as distribution_handle:
        distribution_writer = csv.DictWriter(distribution_handle, fieldnames=distribution_fields)
        distribution_writer.writeheader()
        for (L, N, nmax, W), group in sorted(distribution_groups.items()):
            U, edges = group["U"], group["edges"]
            counts = group["counts"]
            probabilities = counts / np.maximum(1, np.sum(counts, axis=1, keepdims=True))
            for ui, value_u in enumerate(U):
                for bi in range(len(edges) - 1):
                    distribution_writer.writerow({"L": L, "N": N, "nmax": nmax,
                                                  "W_over_t": W / float(cfg["model"]["t"]),
                                                  "U_over_t": value_u / float(cfg["model"]["t"]),
                                                  "bin_left": float(edges[bi]), "bin_right": float(edges[bi + 1]),
                                                  "probability": float(probabilities[ui, bi])})

    bootstrap_n = int(cfg["analysis"]["bootstrap_replicates"])
    confidence = float(cfg["analysis"]["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    bootstrap_seed = int(cfg["analysis"]["bootstrap_seed"])
    peak_points = int(cfg["analysis"]["peak_fit_points"])
    bootstrap_radius = int(cfg["analysis"].get("bootstrap_peak_radius_points", 50))
    max_shift_steps = float(cfg["analysis"].get("max_peak_window_shift_steps", 3.0))
    max_ci_fraction = float(cfg["analysis"].get("max_peak_ci_range_fraction", 0.5))
    peak_rows: list[dict] = []
    curves: dict[tuple[int, int, int, float, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    datasets: dict[tuple[int, int, int, float, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    bootstrap_weights: dict[tuple[int, int, int], np.ndarray] = {}
    peak_bootstrap: dict[tuple[int, int, int, float, str], np.ndarray] = {}
    for sector in sorted({key[:3] for key in groups}):
        example_key = next(key for key in groups if key[:3] == sector)
        pieces = sorted(groups[example_key], key=lambda x: int(np.min(x["sample_ids"])))
        ids = np.concatenate([p["sample_ids"] for p in pieces])
        sector_rng = np.random.default_rng(np.random.SeedSequence([bootstrap_seed, *sector]))
        indices = sector_rng.integers(0, len(ids), size=(bootstrap_n, len(ids)))
        weights = np.zeros((bootstrap_n, len(ids)), dtype=np.float64)
        for bootstrap_id in range(bootstrap_n):
            weights[bootstrap_id] = np.bincount(indices[bootstrap_id], minlength=len(ids)) / len(ids)
        bootstrap_weights[sector] = weights
    for key, pieces in groups.items():
        L, N, nmax, W = key
        pieces.sort(key=lambda x: int(np.min(x["sample_ids"])))
        sample_ids = np.concatenate([p["sample_ids"] for p in pieces])
        U = pieces[0]["U"]
        for observable in OBSERVABLES:
            values = np.concatenate([p[observable] for p in pieces], axis=0)
            order = np.argsort(sample_ids)
            values, ordered_samples = values[order], sample_ids[order]
            datasets[(L, N, nmax, W, observable)] = (ordered_samples, U.copy(), values)
            mean = np.mean(values, axis=0)
            central = stable_peak_parabola(U, mean, peak_points, max_shift_steps)
            boot_peaks = np.full(bootstrap_n, np.nan)
            if observable in PRIMARY_OBSERVABLES:
                center = int(np.argmax(mean))
                lo = max(0, center - bootstrap_radius)
                hi = min(len(U), center + bootstrap_radius + 1)
                boot_curves = bootstrap_weights[(L, N, nmax)] @ values[:, lo:hi]
                for bootstrap_id, boot_curve in enumerate(boot_curves):
                    result = stable_peak_parabola(U[lo:hi], boot_curve, peak_points, max_shift_steps)
                    if result["quality_flag"] == "ok":
                        boot_peaks[bootstrap_id] = float(result["U_peak"])
            peak_bootstrap[(L, N, nmax, W, observable)] = boot_peaks
            valid_boot = boot_peaks[np.isfinite(boot_peaks)]
            low, high = (np.quantile(valid_boot, [alpha, 1-alpha]) if len(valid_boot) else (math.nan, math.nan))
            if (central["quality_flag"] == "ok" and np.isfinite(low) and np.isfinite(high)
                    and high - low >= max_ci_fraction * (U[-1] - U[0])):
                central["quality_flag"] = "broad_confidence_interval"
            peak_rows.append({"L": L, "N": N, "nmax": nmax, "W_over_t": W / float(cfg["model"]["t"]),
                              "observable": observable, "U_peak": central["U_peak"], "U_peak_ci_low": low,
                              "U_peak_ci_high": high, "peak_value": central["peak_value"], "curvature": central["curvature"],
                              "fit_method": f"local_parabola_{peak_points}_points", "quality_flag": central["quality_flag"],
                              "bootstrap_valid": len(valid_boot), "n_samples": len(values)})
            curves[(L, N, nmax, W, observable)] = (U.copy(), mean, np.std(values, axis=0, ddof=1) / math.sqrt(len(values)))
    write_csv(analysis / "peaks.csv", peak_rows)

    convergence_rows: list[dict] = []
    convergence_counts = sorted({int(x) for x in cfg["analysis"].get("convergence_sample_counts", [])})
    for (L, N, nmax, W, observable), (sample_ids, U, values) in sorted(datasets.items()):
        for requested in convergence_counts:
            count = min(requested, len(sample_ids))
            if count < 2:
                continue
            result = stable_peak_parabola(U, np.mean(values[:count], axis=0), peak_points, max_shift_steps)
            convergence_rows.append({"L": L, "N": N, "nmax": nmax,
                                     "W_over_t": W / float(cfg["model"]["t"]),
                                     "observable": observable, "n_disorder": count,
                                     "U_peak": result["U_peak"], "peak_value": result["peak_value"],
                                     "quality_flag": result["quality_flag"]})
    write_csv(analysis / "convergence_peaks.csv", convergence_rows)

    fit_rows: list[dict] = []
    sector_keys = sorted({(r["L"], r["N"], r["nmax"]) for r in peak_rows})
    wmins = [float(x) for x in cfg["analysis"]["fit_Wmin_values"]]
    for L, N, nmax in sector_keys:
        for observable in PRIMARY_OBSERVABLES:
            for wmin in wmins:
                rows = sorted([r for r in peak_rows if (r["L"],r["N"],r["nmax"],r["observable"]) == (L,N,nmax,observable)
                               and r["W_over_t"] >= wmin and r["quality_flag"] == "ok"], key=lambda r: r["W_over_t"])
                if len(rows) < 3:
                    continue
                x = np.asarray([r["W_over_t"] for r in rows])
                y = np.asarray([r["U_peak"] / float(cfg["model"]["t"]) for r in rows])
                stats = linear_statistics(x, y)
                m_rows = {r["W_over_t"]: r for r in peak_rows if (r["L"],r["N"],r["nmax"],r["observable"]) == (L,N,nmax,"M_sum")}
                eta = 1.0 if observable == "M_sum" else math.nan
                direct = {"direct_intercept_q": 0.0, "direct_eta": 1.0, "direct_R2": 1.0}
                if observable != "M_sum" and all(float(xx) in m_rows for xx in x):
                    m_y = np.asarray([m_rows[float(xx)]["U_peak"] / float(cfg["model"]["t"]) for xx in x])
                    m_slope = linear_statistics(x, m_y)["slope_c"]
                    if np.isfinite(m_slope) and abs(m_slope) > 1e-14:
                        eta = stats["slope_c"] / m_slope
                    direct_stats = linear_statistics(m_y, y)
                    direct = {"direct_intercept_q": direct_stats["intercept_b"], "direct_eta": direct_stats["slope_c"],
                              "direct_R2": direct_stats["R2"]}
                slopes_boot, etas_boot, direct_boot = [], [], []
                actual_W = [float(r["W_over_t"]) * float(cfg["model"]["t"]) for r in rows]
                for bootstrap_id in range(bootstrap_n):
                    yb = np.asarray([peak_bootstrap[(L,N,nmax,W,observable)][bootstrap_id] / float(cfg["model"]["t"])
                                     for W in actual_W])
                    mb = np.asarray([peak_bootstrap[(L,N,nmax,W,"M_sum")][bootstrap_id] / float(cfg["model"]["t"])
                                     for W in actual_W])
                    if not np.all(np.isfinite(yb)):
                        continue
                    sb = linear_statistics(x, yb)["slope_c"]
                    slopes_boot.append(sb)
                    if observable == "M_sum":
                        etas_boot.append(1.0)
                        direct_boot.append(1.0)
                    elif np.all(np.isfinite(mb)):
                        sm = linear_statistics(x, mb)["slope_c"]
                        if abs(sm) > 1e-15:
                            etas_boot.append(sb / sm)
                        direct_boot.append(linear_statistics(mb, yb)["slope_c"])
                slope_ci = np.quantile(slopes_boot, [alpha, 1-alpha]) if slopes_boot else (math.nan, math.nan)
                eta_ci = np.quantile(etas_boot, [alpha, 1-alpha]) if etas_boot else (math.nan, math.nan)
                direct_ci = np.quantile(direct_boot, [alpha, 1-alpha]) if direct_boot else (math.nan, math.nan)
                fit_rows.append({"L": L, "N": N, "nmax": nmax, "observable": observable,
                                 "Wmin": wmin, "Wmax": float(np.max(x)), **stats,
                                 "slope_ci_low": slope_ci[0], "slope_ci_high": slope_ci[1],
                                 "eta_to_M": eta, "eta_ci_low": eta_ci[0], "eta_ci_high": eta_ci[1],
                                 **direct, "direct_eta_ci_low": direct_ci[0], "direct_eta_ci_high": direct_ci[1],
                                 "bootstrap_valid": len(slopes_boot)})
    fit_fields = ["L", "N", "nmax", "observable", "Wmin", "Wmax", "intercept_b", "slope_c", "R2", "RMSE",
                  "slope_ci_low", "slope_ci_high", "eta_to_M", "eta_ci_low", "eta_ci_high",
                  "direct_intercept_q", "direct_eta", "direct_R2", "direct_eta_ci_low", "direct_eta_ci_high",
                  "bootstrap_valid"]
    write_csv(analysis / "fits.csv", fit_rows, fieldnames=fit_fields)
    make_plots(cfg, analysis, curves, peak_rows)
    summary = summarize(peaks=peak_rows, fits=fit_rows)
    (analysis / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def merge_sector_analyses(cfg: dict) -> None:
    """Merge small per-sector tables; large CSV files stay sector-local."""
    analysis = results_root(cfg) / "analysis"
    table_names = ("peaks.csv", "fits.csv", "convergence_peaks.csv")
    for table_name in table_names:
        rows: list[dict] = []
        fields: list[str] | None = None
        for sector_id in range(len(cfg["model"]["sectors"])):
            path = analysis / sector_analysis_name(cfg, sector_id) / table_name
            if not path.exists():
                raise SystemExit(f"Missing sector analysis table: {path}")
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = fields or list(reader.fieldnames or [])
                rows.extend(reader)
        write_csv(analysis / table_name, rows, fieldnames=fields)
    summaries = []
    for sector_id in range(len(cfg["model"]["sectors"])):
        path = analysis / sector_analysis_name(cfg, sector_id) / "summary.txt"
        if path.exists():
            summaries.append(path.read_text(encoding="utf-8").strip())
    (analysis / "summary.txt").write_text("\n\n".join(summaries) + "\n", encoding="utf-8")
    print(f"Merged {len(cfg['model']['sectors'])} sector analyses into {analysis}")


def make_plots(cfg: dict, analysis: Path, curves: dict, peaks: list[dict]) -> None:
    plt = setup_plots()
    if plt is None:
        return
    figdir = analysis / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    colors = {"M_sum": "#0072B2", "S2_sum": "#D55E00", "Sstar_mean": "#009E73"}
    for L, N, nmax in sorted({key[:3] for key in curves}):
        W_values = sorted({key[3] for key in curves if key[:3] == (L,N,nmax)})
        chosen = W_values[::max(1, len(W_values)//4)][:5]
        for W in chosen:
            fig, ax = plt.subplots(figsize=(4.2, 3.0))
            for obs in ("M_sum", "S2_sum", "Sstar_mean"):
                U, mean, stderr = curves[(L,N,nmax,W,obs)]
                scale = np.nanmax(mean)
                ax.plot(U, mean/scale if scale > 0 else mean, label=obs, color=colors[obs])
            ax.set(xlabel=r"$U/t$", ylabel="curve / maximum", title=rf"$L={L},N={N},n_{{max}}={nmax},W/t={W:g}$")
            ax.legend(frameon=False)
            stem = figdir / f"curves_L{L}_N{N}_nmax{nmax}_W{W:g}".replace(".", "p")
            fig.savefig(stem.with_suffix(".png"), dpi=int(cfg["analysis"]["figure_dpi"]), bbox_inches="tight")
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for obs in ("M_sum", "S2_sum", "Sstar_mean"):
            rows = sorted([r for r in peaks if (r["L"],r["N"],r["nmax"],r["observable"]) == (L,N,nmax,obs)], key=lambda r:r["W_over_t"])
            lower = [max(0.0, r["U_peak"] - r["U_peak_ci_low"]) if np.isfinite(r["U_peak_ci_low"]) else 0.0 for r in rows]
            upper = [max(0.0, r["U_peak_ci_high"] - r["U_peak"]) if np.isfinite(r["U_peak_ci_high"]) else 0.0 for r in rows]
            ax.errorbar([r["W_over_t"] for r in rows], [r["U_peak"] for r in rows],
                        yerr=[lower, upper],
                        marker="o", ms=3, capsize=2, label=obs, color=colors[obs])
        ax.set(xlabel=r"$W/t$", ylabel=r"$U^*/t$", title=rf"$L={L},N={N},n_{{max}}={nmax}$")
        ax.legend(frameon=False)
        stem = figdir / f"peaks_L{L}_N{N}_nmax{nmax}"
        fig.savefig(stem.with_suffix(".png"), dpi=int(cfg["analysis"]["figure_dpi"]), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def summarize(peaks: list[dict], fits: list[dict]) -> str:
    ok = sum(r["quality_flag"] == "ok" for r in peaks)
    total = len(peaks)
    lines = ["Bose-Hubbard star-model analysis", f"Valid interior parabolic peaks: {ok}/{total}."]
    for row in fits:
        if row["observable"] in ("M_sum", "S2_sum", "Sstar_mean") and row["Wmin"] == min(r["Wmin"] for r in fits):
            lines.append(f"L={row['L']} N={row['N']} nmax={row['nmax']} {row['observable']}: c={row['slope_c']:.6g}, R2={row['R2']:.5g}, eta_to_M={row['eta_to_M']:.6g}")
    lines.append("This run contains no ED data; compare peaks.csv directly with the existing ED maxima.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.command == "validate": validate_config(cfg)
    elif args.command == "init": initialize(cfg, args.force)
    elif args.command == "worker": run_worker(cfg, args.task_id)
    elif args.command == "status": status(cfg)
    elif args.command == "aggregate": aggregate(cfg, args.sector_id, args.fast)
    elif args.command == "merge-sectors": merge_sector_analyses(cfg)


if __name__ == "__main__":
    main()
