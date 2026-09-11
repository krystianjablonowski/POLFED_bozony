#!/usr/bin/env python3
"""Correct, reproducible L=7 Bose-Hubbard star-model calculation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from star_l7_core import (
    VERSION, build_graph, central_configurations, compensating_channels,
    compensating_observables, configuration_priority, deterministic_seed,
    disorder_vector, observables_at_u, verify_detuning,
)


# The first two names preserve the pre-existing POLFED mixing-channel model.
# The graph versions implement the all-edge definitions in the supplied note.
PRIMARY_OBSERVABLES = ("M_existing", "S2_existing", "Sstar")
GRAPH_OBSERVABLES = ("M_graph_sum", "S2_graph_sum", "M_graph_per_edge", "S2_graph_per_edge")
OBSERVABLES = PRIMARY_OBSERVABLES + GRAPH_OBSERVABLES
STORED = OBSERVABLES + ("Sstar_median", "Sstar_std", "Pstar", "z_mean", "resonant_edges")
CHANNEL_STORED = ("M_k", "S2_k", "P_k")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "init", "status", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name == "init":
            p.add_argument("--force", action="store_true")
        if name == "analyze":
            p.add_argument("--no-plots", action="store_true")
    p = sub.add_parser("worker")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--task-id", type=int, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_path"] = str(path.resolve())
    return cfg


def computational_signature(cfg: dict) -> str:
    payload = {key: cfg[key] for key in ("model", "disorder", "grid", "sampling", "numerics")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def regular_grid(spec: dict) -> np.ndarray:
    if "values" in spec:
        values = np.asarray(spec["values"], dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Explicit grid must be a finite non-empty vector")
        if np.any(np.diff(values) <= 0):
            raise ValueError("Explicit grid must be strictly increasing")
        return values
    start, stop, step = map(float, (spec["start"], spec["stop"], spec["step"]))
    if step <= 0:
        raise ValueError("Grid step must be positive")
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1)
    if count < 0 or abs(values[-1] - stop) > 1e-10:
        raise ValueError("Grid interval must be an integer multiple of step")
    return values


def root_path(cfg: dict) -> Path:
    root = Path(cfg["paths"]["results"])
    return root if root.is_absolute() else Path(cfg["_path"]).parent / root


def tasks(cfg: dict) -> List[dict]:
    output = []
    chunk = int(cfg["sampling"]["samples_per_task"])
    count = int(cfg["sampling"]["n_disorder"])
    task_id = 0
    for sector_id, sector in enumerate(cfg["model"]["sectors"]):
        for W_id, W in enumerate(regular_grid(cfg["grid"]["W"])):
            for first in range(0, count, chunk):
                output.append({
                    "task_id": task_id, "sector_id": sector_id,
                    "N": int(sector["N"]), "nmax": int(sector["nmax"]),
                    "W_id": W_id, "W": float(W), "sample_first": first,
                    "sample_stop": min(first + chunk, count),
                })
                task_id += 1
    return output


def validate(cfg: dict) -> None:
    if int(cfg["model"]["L"]) != 7:
        raise ValueError("This focused implementation requires L=7")
    if cfg["model"]["boundary"] != "open":
        raise ValueError("The instruction requires open boundaries")
    if cfg["model"]["interaction"] != "U_n_nminus1":
        raise ValueError("Require H_int=U*sum_i n_i(n_i-1)")
    if cfg["disorder"]["distribution"] != "uniform_minus_W_plus_W":
        raise ValueError("Require independent epsilon_i uniform on [-W,W]")
    if int(cfg["sampling"]["n_disorder"]) <= 1 or int(cfg["sampling"]["samples_per_task"]) <= 0:
        raise ValueError("Invalid sample counts")
    selection = cfg["sampling"].get("central_selection")
    supported = ("closest_count", "normalized_half_width", "rank_fraction")
    if not isinstance(selection, dict) or selection.get("mode") not in supported:
        raise ValueError("sampling.central_selection must explicitly define a supported mode")
    U, W = regular_grid(cfg["grid"]["U"]), regular_grid(cfg["grid"]["W"])
    if len(U) < 7:
        raise ValueError("At least seven U points are required")
    for sector in cfg["model"]["sectors"]:
        graph = build_graph(7, int(sector["N"]), int(sector["nmax"]), float(cfg["model"]["t"]))
        print("L=7 N={} nmax={}: dim={}, zmax={}".format(
            sector["N"], sector["nmax"], graph.dim, graph.neighbors.shape[1]))
    print("U/t={}..{}, points={}; W/t={}..{}, points={}".format(U[0], U[-1], len(U), W[0], W[-1], len(W)))
    print("disorder={}, central_selection={}, tasks={}".format(
        cfg["sampling"]["n_disorder"], json.dumps(selection, sort_keys=True), len(tasks(cfg))))
    print("Hamiltonian: H_diag = U*sum_i n_i(n_i-1) + sum_i epsilon_i*n_i")
    print("Disorder: eta=epsilon/W ~ Uniform[-1,1]; SeedSequence([master_seed,L,sample_id])")
    print("M_existing/S2_existing: preserved positive-k compensating-channel model")
    print("M_graph_*/S2_graph_*: every directed one-hop edge, direct detuning")
    print("Sstar: all one-hop neighbours; maximum-overlap ties are averaged")
    print("Production workers perform no full many-body diagonalization.")


def initialize(cfg: dict, force: bool) -> None:
    validate(cfg)
    root = root_path(cfg)
    manifest = root / "tasks.csv"
    if manifest.exists() and not force:
        raise SystemExit("Manifest exists: {}".format(manifest))
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "analysis" / "figures").mkdir(parents=True, exist_ok=True)
    rows = tasks(cfg)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {key: value for key, value in cfg.items() if not key.startswith("_")}
    metadata.update({
        "version": VERSION, "config_sha256": computational_signature(cfg),
        "full_ED_in_workers": False,
        "star_branch_selection": "average_all_maximum-overlap_ties",
        "definitions": {
            "M_existing": "positive-k weighted compensating-channel model preserved from POLFED",
            "S2_existing": "binary entropy applied to the same preserved mixing channels",
            "M_graph_sum": "mean_alpha sum_edges m_edge",
            "M_graph_per_edge": "mean_alpha (sum_edges m_edge / z_alpha)",
            "Sstar": "mean_alpha Shannon entropy of selected local-star eigenvector",
        },
    })
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Initialized {} tasks in {}".format(len(rows), root))


def output_path(cfg: dict, task_id: int) -> Path:
    return root_path(cfg) / "raw" / "task_{:05d}.npz".format(task_id)


def valid_output(path: Path, task: dict, U: np.ndarray, cfg: dict) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = set(STORED + CHANNEL_STORED) | {
                "version", "config_sha256", "N", "nmax", "W", "U", "sample_ids",
                "seeds", "eta", "channel_k", "Sstar_hist", "Sstar_bin_edges",
                "number_of_configurations",
            }
            return (
                str(data["version"]) == VERSION
                and str(data["config_sha256"]) == computational_signature(cfg)
                and required.issubset(data.files)
                and int(data["N"]) == task["N"] and int(data["nmax"]) == task["nmax"]
                and abs(float(data["W"]) - task["W"]) < 1e-12
                and np.array_equal(data["U"], U)
                and np.array_equal(data["sample_ids"], np.arange(task["sample_first"], task["sample_stop"]))
            )
    except Exception:
        return False


def worker(cfg: dict, task_id: int) -> None:
    rows = tasks(cfg)
    if not 0 <= task_id < len(rows):
        raise SystemExit("task-id outside [0,{}]".format(len(rows) - 1))
    task = rows[task_id]
    U_values = regular_grid(cfg["grid"]["U"])
    path = output_path(cfg, task_id)
    if valid_output(path, task, U_values, cfg):
        print("SKIP {}".format(path))
        return

    N, nmax, W = task["N"], task["nmax"], task["W"]
    graph = build_graph(7, N, nmax, float(cfg["model"]["t"]))
    old_channels = compensating_channels(graph)
    valid_k = sorted({int(graph.channels[a, e]) for a in range(graph.dim) for e in range(int(graph.degree[a]))})
    channel_k = np.asarray(valid_k, dtype=np.int16)
    channel_index = {int(k): i for i, k in enumerate(channel_k)}
    sample_ids = np.arange(task["sample_first"], task["sample_stop"], dtype=np.int32)
    shape = (len(sample_ids), len(U_values))
    arrays = {name: np.empty(shape, dtype=np.float64) for name in STORED}
    channel_arrays = {name: np.zeros(shape + (len(channel_k),), dtype=np.float64) for name in CHANNEL_STORED}
    configuration_counts = np.empty(shape, dtype=np.int32)
    seeds = np.empty(len(sample_ids), dtype=np.uint64)
    eta_vectors = np.empty((len(sample_ids), 7), dtype=np.float64)
    bins = int(cfg["sampling"].get("star_entropy_histogram_bins", 40))
    bin_edges = np.linspace(0.0, math.log(graph.neighbors.shape[1] + 1.0), bins + 1)
    histograms = np.zeros(shape + (bins,), dtype=np.int32)
    maximum_cfg = int(cfg["sampling"].get("n_configurations", 0))
    selection = cfg["sampling"]["central_selection"]
    tie_tolerance = float(cfg["numerics"]["tie_tolerance"])
    master = int(cfg["disorder"]["master_seed"])
    started = time.monotonic()

    for si, sample_id in enumerate(sample_ids):
        seed = deterministic_seed(master, 7, int(sample_id))
        eta = disorder_vector(master, 7, int(sample_id))
        epsilon = W * eta
        seeds[si], eta_vectors[si] = seed, eta
        disorder_energy = graph.basis @ epsilon
        priority = configuration_priority(graph.basis, seed)
        error = verify_detuning(graph, float(U_values[len(U_values) // 2]), epsilon, seed & 0xFFFFFFFF)
        if error > float(cfg["numerics"]["detuning_tolerance"]):
            raise RuntimeError("detuning identity error {:.3e}".format(error))
        for ui, U in enumerate(U_values):
            energies = U * graph.interaction + disorder_energy
            central = central_configurations(energies, selection, maximum_cfg, priority)
            result = observables_at_u(graph, float(U), disorder_energy, central, tie_tolerance)
            if float(result["normalization_error"]) > float(cfg["numerics"]["normalization_tolerance"]):
                raise RuntimeError("normalization error {:.3e}".format(result["normalization_error"]))
            m_old, s2_old = compensating_observables(float(U), epsilon, graph, old_channels)
            mapped = {
                "M_existing": m_old, "S2_existing": s2_old,
                "M_graph_sum": result["M_sum"], "S2_graph_sum": result["S2_sum"],
                "M_graph_per_edge": result["M_per_edge"], "S2_graph_per_edge": result["S2_per_edge"],
                "Sstar": result["Sstar"], "Sstar_median": result["Sstar_median"],
                "Sstar_std": result["Sstar_std"], "Pstar": result["Pstar"],
                "z_mean": result["z_mean"], "resonant_edges": result["resonant_edges"],
            }
            for name in STORED:
                arrays[name][si, ui] = float(mapped[name])
            for name in CHANNEL_STORED:
                for k, value in result[name].items():
                    channel_arrays[name][si, ui, channel_index[int(k)]] = float(value)
            configuration_counts[si, ui] = len(central)
            histograms[si, ui] = np.histogram(result["Sstar_values"], bins=bin_edges)[0]
        elapsed = time.monotonic() - started
        remaining = elapsed / (si + 1) * (len(sample_ids) - si - 1)
        print("task={} sample={}/{} elapsed={:.2f}s eta={:.2f}s".format(
            task_id, si + 1, len(sample_ids), elapsed, remaining), flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.{}.npz".format(os.getpid()))
    np.savez_compressed(
        temporary, version=VERSION, config_sha256=computational_signature(cfg),
        L=7, N=N, nmax=nmax, W=W, U=U_values, sample_ids=sample_ids,
        seeds=seeds, eta=eta_vectors, channel_k=channel_k, Sstar_hist=histograms,
        Sstar_bin_edges=bin_edges, number_of_configurations=configuration_counts,
        **arrays, **channel_arrays
    )
    os.replace(temporary, path)
    print("Saved {}; total={:.2f}s".format(path, time.monotonic() - started))


def status(cfg: dict) -> None:
    U = regular_grid(cfg["grid"]["U"])
    rows = tasks(cfg)
    complete = [valid_output(output_path(cfg, row["task_id"]), row, U, cfg) for row in rows]
    print("Complete {}/{}; missing {}".format(sum(complete), len(rows), len(rows) - sum(complete)))
    missing = [str(row["task_id"]) for row, ok in zip(rows, complete) if not ok]
    if missing:
        print("Missing task IDs: " + ",".join(missing))


def smoothing_matrix(length: int, window: int, degree: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    radius = window // 2
    matrix = np.zeros((length, length), dtype=float)
    for i in range(length):
        lo, hi = max(0, i - radius), min(length, i + radius + 1)
        xx = np.arange(lo - i, hi - i, dtype=float)
        fit_degree = min(degree, len(xx) - 1)
        vandermonde = np.vander(xx, fit_degree + 1)
        evaluation = np.zeros(fit_degree + 1)
        evaluation[-1] = 1.0
        matrix[i, lo:hi] = evaluation @ np.linalg.pinv(vandermonde)
    return matrix


def smooth_curve(y: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix @ np.asarray(y, dtype=float)


def _quadratic_candidate(x: np.ndarray, y: np.ndarray, imax: int, points: int) -> Optional[Tuple[float, float, float]]:
    points = max(3, int(points) | 1)
    half = points // 2
    lo, hi = imax - half, imax + half + 1
    if lo < 0 or hi > len(x):
        return None
    a, b, c = np.polyfit(x[lo:hi], y[lo:hi], 2)
    if not np.isfinite(a + b + c) or a >= 0:
        return None
    peak = float(-b / (2 * a))
    if not x[lo] <= peak <= x[hi - 1]:
        return None
    return peak, float(a * peak * peak + b * peak + c), float(2 * a)


def locate_peak(x: np.ndarray, y: np.ndarray, cfg: dict, smoother: Optional[np.ndarray] = None) -> dict:
    if smoother is None:
        smoother = smoothing_matrix(len(y), int(cfg["analysis"]["smoothing_window"]),
                                    int(cfg["analysis"]["smoothing_degree"]))
    return locate_smoothed_peak(x, smooth_curve(y, smoother), cfg)


def locate_smoothed_peak(x: np.ndarray, smoothed: np.ndarray, cfg: dict) -> dict:
    imax = int(np.argmax(smoothed))
    margin = int(cfg["analysis"]["boundary_margin_points"])
    base = {"U_peak": math.nan, "peak_value": float(smoothed[imax]), "curvature": math.nan,
            "grid_argmax": float(x[imax]), "fit_window_spread": math.nan}
    if imax < margin or imax >= len(x) - margin:
        return dict(base, quality_flag="boundary")
    candidate = _quadratic_candidate(x, smoothed, imax, int(cfg["analysis"]["peak_fit_points"]))
    if candidate is None:
        return dict(base, quality_flag="invalid_quadratic")
    peak, height, curvature = candidate
    candidates = [_quadratic_candidate(x, smoothed, imax, int(width))
                  for width in cfg["analysis"].get("stability_fit_points", [3, 5, 7])]
    stable_peaks = [item[0] for item in candidates if item is not None]
    spread = float(max(stable_peaks) - min(stable_peaks)) if len(stable_peaks) >= 2 else math.inf
    quality = "ok" if np.isfinite(spread) and spread <= float(
        cfg["analysis"].get("max_fit_window_spread", 0.02)) else "fit_window_unstable"
    return {"U_peak": peak, "peak_value": height, "curvature": curvature,
            "grid_argmax": float(x[imax]), "fit_window_spread": spread, "quality_flag": quality}


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _linear_statistics(x: np.ndarray, y: np.ndarray) -> dict:
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    sse = float(np.sum(residual * residual))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    dof = len(x) - 2
    covariance = (sse / dof) * np.linalg.inv(design.T @ design) if dof > 0 else np.full((2, 2), math.nan)
    return {"intercept": float(coefficients[0]), "slope": float(coefficients[1]),
            "covariance": covariance, "R2": 1.0 - sse / sst if sst > 0 else math.nan,
            "RMSE": math.sqrt(sse / len(x))}


def _bootstrap_slopes(x: np.ndarray, peak_arrays: List[np.ndarray]) -> np.ndarray:
    if not peak_arrays:
        return np.empty(0)
    stacked = np.column_stack(peak_arrays)
    output = np.full(stacked.shape[0], math.nan)
    for bi, row in enumerate(stacked):
        finite = np.isfinite(row)
        if np.sum(finite) >= 3:
            output[bi] = _linear_statistics(x[finite], row[finite])["slope"]
    return output


def _load_ed_peaks(cfg: dict) -> List[dict]:
    configured = cfg.get("paths", {}).get("ed_peaks_csv")
    if not configured:
        return []
    path = Path(configured)
    if not path.is_absolute():
        path = Path(cfg["_path"]).parent / path
    if not path.exists():
        print("ED reference not found, skipping: {}".format(path))
        return []
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for source in csv.DictReader(handle):
            if "L" in source and int(float(source["L"])) != 7:
                continue
            peak_key = next((key for key in ("U_peak", "U_S_star_over_t", "Ustar", "U_star") if key in source), None)
            if peak_key is None:
                raise ValueError("ED CSV needs U_peak or U_S_star_over_t")
            rows.append({
                "L": 7, "N": int(float(source["N"])),
                "nmax": int(float(source.get("nmax", source["N"]))),
                "W_over_t": float(source.get("W_over_t", source.get("W", "nan"))),
                "observable": "S_ED", "U_peak": float(source[peak_key]),
                "peak_value": float(source.get("peak_value", "nan")),
                "curvature": float(source.get("curvature", "nan")),
                "grid_argmax": float(source[peak_key]),
                "fit_window_spread": float(source.get("fit_window_spread", "nan")),
                "quality_flag": source.get("quality_flag", "ok"),
                "fit_method": source.get("fit_method", "imported_ED"),
                "U_peak_ci_low": float(source.get("U_peak_ci_low", "nan")),
                "U_peak_ci_high": float(source.get("U_peak_ci_high", "nan")),
                "bootstrap_valid": source.get("bootstrap_valid", ""),
                "n_disorder": source.get("n_disorder", ""),
            })
    print("Loaded {} ED peak rows from {}".format(len(rows), path))
    return rows


def analyze(cfg: dict, no_plots: bool = False) -> None:
    task_rows = tasks(cfg)
    U = regular_grid(cfg["grid"]["U"])
    missing = [row for row in task_rows if not valid_output(output_path(cfg, row["task_id"]), row, U, cfg)]
    if missing:
        raise SystemExit("Cannot analyze: {} tasks missing or incompatible".format(len(missing)))
    groups = {}
    for task in task_rows:
        with np.load(output_path(cfg, task["task_id"]), allow_pickle=False) as data:
            record = {name: data[name].copy() for name in STORED + CHANNEL_STORED}
            for name in ("sample_ids", "seeds", "eta", "channel_k", "Sstar_hist",
                         "Sstar_bin_edges", "number_of_configurations"):
                record[name] = data[name].copy()
            groups.setdefault((task["N"], task["nmax"], task["W"]), []).append(record)

    analysis_dir = root_path(cfg) / "analysis"
    figure_dir = analysis_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    curve_rows, peak_rows, channel_rows, distribution_rows = [], [], [], []
    curves, channel_curves, peak_boot, group_records = {}, {}, {}, {}
    B = int(cfg["analysis"]["bootstrap_replicates"])
    alpha = (1.0 - float(cfg["analysis"]["confidence_level"])) / 2.0
    smoother = smoothing_matrix(len(U), int(cfg["analysis"]["smoothing_window"]),
                                int(cfg["analysis"]["smoothing_degree"]))
    bootstrap_indices = {}

    for (N, nmax, W), pieces in sorted(groups.items()):
        pieces.sort(key=lambda p: int(np.min(p["sample_ids"])))
        sample_ids = np.concatenate([p["sample_ids"] for p in pieces])
        order = np.argsort(sample_ids)
        expected = np.arange(int(cfg["sampling"]["n_disorder"]))
        if not np.array_equal(sample_ids[order], expected):
            raise RuntimeError("Incomplete or duplicate sample IDs for N={}, nmax={}, W={}".format(N, nmax, W))
        if (N, nmax) not in bootstrap_indices:
            rng = np.random.default_rng(np.random.SeedSequence([
                int(cfg["analysis"]["bootstrap_seed"]), 7, N, nmax]))
            bootstrap_indices[(N, nmax)] = (rng.integers(0, len(sample_ids), size=(B, len(sample_ids)))
                                                  if B else np.empty((0, len(sample_ids)), dtype=int))
        indices = bootstrap_indices[(N, nmax)]
        group_records[(N, nmax, W)] = (pieces, order)

        for observable in OBSERVABLES:
            values = np.concatenate([p[observable] for p in pieces], axis=0)[order]
            mean = np.mean(values, axis=0)
            stderr = np.std(values, axis=0, ddof=1) / math.sqrt(len(values))
            curves[(N, nmax, W, observable)] = (U, mean, stderr)
            for ui, value_u in enumerate(U):
                curve_rows.append({"L": 7, "N": N, "nmax": nmax, "W_over_t": W,
                                   "U_over_t": float(value_u), "observable": observable,
                                   "mean": float(mean[ui]), "stderr": float(stderr[ui]),
                                   "n_disorder": len(values)})
            central = locate_peak(U, mean, cfg, smoother)
            boot = np.full(B, math.nan)
            if central["quality_flag"] == "ok" and B:
                bootstrap_smoothed = np.mean(values[indices], axis=1) @ smoother.T
                for bi, smoothed in enumerate(bootstrap_smoothed):
                    result = locate_smoothed_peak(U, smoothed, cfg)
                    if result["quality_flag"] == "ok":
                        boot[bi] = result["U_peak"]
            finite_boot = boot[np.isfinite(boot)]
            low, high = (np.quantile(finite_boot, [alpha, 1.0 - alpha])
                         if len(finite_boot) else (math.nan, math.nan))
            quality = central["quality_flag"]
            if quality == "ok" and B and len(finite_boot) < float(
                    cfg["analysis"].get("min_bootstrap_valid_fraction", 0.8)) * B:
                quality = "unstable_bootstrap"
            if quality == "ok" and np.isfinite(low) and np.isfinite(high) and high - low > float(
                    cfg["analysis"].get("max_peak_ci_width", 0.12)):
                quality = "wide_ci"
            central["quality_flag"] = quality
            peak_boot[(N, nmax, W, observable)] = boot
            peak_rows.append({"L": 7, "N": N, "nmax": nmax, "W_over_t": W,
                              "observable": observable, **central,
                              "fit_method": "local_smooth_quadratic_3_5_7_stability",
                              "U_peak_ci_low": float(low), "U_peak_ci_high": float(high),
                              "bootstrap_valid": int(len(finite_boot)), "n_disorder": len(values)})

        channel_k = pieces[0]["channel_k"]
        if any(not np.array_equal(piece["channel_k"], channel_k) for piece in pieces):
            raise RuntimeError("Inconsistent channel axes")
        for quantity in CHANNEL_STORED:
            values = np.concatenate([p[quantity] for p in pieces], axis=0)[order]
            mean = np.mean(values, axis=0)
            stderr = np.std(values, axis=0, ddof=1) / math.sqrt(len(values))
            channel_curves[(N, nmax, W, quantity)] = (U, channel_k, mean, stderr)
            for ui, value_u in enumerate(U):
                for ki, k in enumerate(channel_k):
                    channel_rows.append({"L": 7, "N": N, "nmax": nmax, "W_over_t": W,
                                         "U_over_t": float(value_u), "quantity": quantity,
                                         "k": int(k), "mean": float(mean[ui, ki]),
                                         "stderr": float(stderr[ui, ki]), "n_disorder": len(values)})
        hist = np.sum(np.concatenate([p["Sstar_hist"] for p in pieces], axis=0)[order], axis=0)
        edges = pieces[0]["Sstar_bin_edges"]
        for ui, value_u in enumerate(U):
            total = int(np.sum(hist[ui]))
            for bi in range(len(edges) - 1):
                distribution_rows.append({"L": 7, "N": N, "nmax": nmax, "W_over_t": W,
                                          "U_over_t": float(value_u), "bin_left": float(edges[bi]),
                                          "bin_right": float(edges[bi + 1]), "count": int(hist[ui, bi]),
                                          "probability": float(hist[ui, bi] / total) if total else math.nan})

    imported_ed = _load_ed_peaks(cfg)
    peak_rows.extend(imported_ed)
    write_csv(analysis_dir / "curves.csv", curve_rows)
    write_csv(analysis_dir / "peaks.csv", peak_rows)
    write_csv(analysis_dir / "channels.csv", channel_rows)
    write_csv(analysis_dir / "star_entropy_distribution.csv", distribution_rows)

    quality_rows = []
    for N, nmax in sorted({(row["N"], row["nmax"]) for row in peak_rows}):
        names = sorted({row["observable"] for row in peak_rows if (row["N"], row["nmax"]) == (N, nmax)})
        for observable in names:
            selected = [row for row in peak_rows if (row["N"], row["nmax"], row["observable"]) == (N, nmax, observable)]
            for flag in sorted({row["quality_flag"] for row in selected}):
                quality_rows.append({"L": 7, "N": N, "nmax": nmax, "observable": observable,
                                     "quality_flag": flag,
                                     "count": sum(row["quality_flag"] == flag for row in selected),
                                     "total_W": len(selected)})
    write_csv(analysis_dir / "quality_counts.csv", quality_rows)

    fit_rows, direct_rows = _make_fits(cfg, curves, peak_rows, peak_boot, alpha)
    ed_fits, ed_direct = _make_imported_ed_fits(cfg, peak_rows, imported_ed)
    fit_rows.extend(ed_fits)
    direct_rows.extend(ed_direct)
    write_csv(analysis_dir / "fits.csv", fit_rows)
    write_csv(analysis_dir / "direct_fits.csv", direct_rows)
    comparison_rows = _compare_to_ed(peak_rows, imported_ed)
    write_csv(analysis_dir / "comparison_to_ED.csv", comparison_rows)
    if cfg["analysis"].get("write_raw_long_csv", True):
        _write_raw_long_csv(cfg, group_records, U, analysis_dir / "raw_observables.csv.gz")
    if not no_plots:
        make_figures(cfg, curves, channel_curves, peak_rows, fit_rows, comparison_rows, figure_dir)

    model_peaks = [row for row in peak_rows if row["observable"] != "S_ED"]
    flags = {flag: sum(row["quality_flag"] == flag for row in model_peaks)
             for flag in sorted({row["quality_flag"] for row in model_peaks})}
    lines = ["L=7 star model v{}".format(VERSION),
             "Valid model peaks: {}/{}".format(flags.get("ok", 0), len(model_peaks)),
             "Quality flags: {}".format(flags),
             "Central selection: {}".format(json.dumps(cfg["sampling"]["central_selection"], sort_keys=True)),
             "Paired bootstrap uses the same sample resampling for all W and observables."]
    if comparison_rows:
        rmse_by_observable = {}
        for observable in PRIMARY_OBSERVABLES:
            residuals = [r["residual"] for r in comparison_rows if r["observable"] == observable]
            if residuals:
                rmse_by_observable[observable] = math.sqrt(np.mean(np.square(residuals)))
                lines.append("RMSE({} vs ED)={:.6g}".format(observable, rmse_by_observable[observable]))
        if rmse_by_observable:
            best = min(rmse_by_observable, key=rmse_by_observable.get)
            lines.append("Best direct peak predictor among available primary estimators: {}".format(best))
            if "Sstar" in rmse_by_observable and "M_existing" in rmse_by_observable:
                verdict = "improves" if rmse_by_observable["Sstar"] < rmse_by_observable["M_existing"] else "does not improve"
                lines.append("Sstar {} RMSE relative to M_existing.".format(verdict))
    else:
        lines.append("No ED peaks CSV configured; ED comparison was not performed.")
    summary = "\n".join(lines) + "\n"
    (analysis_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def _make_fits(cfg: dict, curves: dict, peak_rows: List[dict], peak_boot: dict, alpha: float):
    lookup = {(r["N"], r["nmax"], float(r["W_over_t"]), r["observable"]): r
              for r in peak_rows if r["observable"] != "S_ED"}
    Wmins = [float(x) for x in cfg["analysis"].get("fit_Wmin_values", [0.8, 1.0, 1.2, 1.4])]
    Wmax = float(cfg["analysis"].get("fit_Wmax", np.max(regular_grid(cfg["grid"]["W"]))))
    fit_rows, direct_rows = [], []
    for N, nmax in sorted({key[:2] for key in curves}):
        all_w = sorted({key[2] for key in curves if key[:2] == (N, nmax)})
        for observable in OBSERVABLES:
            for Wmin in Wmins:
                valid_w = [W for W in all_w if Wmin <= W <= Wmax and lookup[(N, nmax, W, observable)]["quality_flag"] == "ok"]
                if len(valid_w) < 3:
                    continue
                x = np.asarray(valid_w)
                y = np.asarray([lookup[(N, nmax, W, observable)]["U_peak"] for W in valid_w])
                stats = _linear_statistics(x, y)
                slopes = _bootstrap_slopes(x, [peak_boot[(N, nmax, W, observable)] for W in valid_w])
                finite_slopes = slopes[np.isfinite(slopes)]
                slope_low, slope_high = (np.quantile(finite_slopes, [alpha, 1.0 - alpha])
                                         if len(finite_slopes) else (math.nan, math.nan))
                eta_value = eta_low = eta_high = math.nan
                if observable == "M_existing":
                    eta_value = eta_low = eta_high = 1.0
                else:
                    common = [W for W in valid_w if lookup[(N, nmax, W, "M_existing")]["quality_flag"] == "ok"]
                    if len(common) >= 3:
                        xx = np.asarray(common)
                        num = _linear_statistics(xx, np.asarray([lookup[(N, nmax, W, observable)]["U_peak"] for W in common]))["slope"]
                        den = _linear_statistics(xx, np.asarray([lookup[(N, nmax, W, "M_existing")]["U_peak"] for W in common]))["slope"]
                        eta_value = num / den if abs(den) > 1e-15 else math.nan
                        nb = _bootstrap_slopes(xx, [peak_boot[(N, nmax, W, observable)] for W in common])
                        db = _bootstrap_slopes(xx, [peak_boot[(N, nmax, W, "M_existing")] for W in common])
                        good = np.isfinite(nb) & np.isfinite(db) & (np.abs(db) > 1e-15)
                        ratios = nb[good] / db[good]
                        if len(ratios):
                            eta_low, eta_high = np.quantile(ratios, [alpha, 1.0 - alpha])
                        xm = np.asarray([lookup[(N, nmax, W, "M_existing")]["U_peak"] for W in common])
                        yy = np.asarray([lookup[(N, nmax, W, observable)]["U_peak"] for W in common])
                        direct = _linear_statistics(xm, yy)
                        direct_rows.append({"L": 7, "N": N, "nmax": nmax, "observable": observable,
                                            "Wmin": Wmin, "Wmax": Wmax, "n_points": len(common),
                                            "intercept_q": direct["intercept"], "eta_direct": direct["slope"],
                                            "R2": direct["R2"], "RMSE": direct["RMSE"]})
                cov = stats["covariance"]
                fit_rows.append({"L": 7, "N": N, "nmax": nmax, "observable": observable,
                                 "Wmin": Wmin, "Wmax": Wmax, "n_points": len(valid_w),
                                 "intercept_b": stats["intercept"], "slope_c": stats["slope"],
                                 "cov_bb": float(cov[0, 0]), "cov_bc": float(cov[0, 1]), "cov_cc": float(cov[1, 1]),
                                 "slope_ci_low": float(slope_low), "slope_ci_high": float(slope_high),
                                 "R2": stats["R2"], "RMSE": stats["RMSE"],
                                 "eta_to_M": eta_value, "eta_ci_low": eta_low,
                                 "eta_ci_high": eta_high, "bootstrap_valid": int(len(finite_slopes))})
    return fit_rows, direct_rows


def _compare_to_ed(peaks: List[dict], imported_ed: List[dict]) -> List[dict]:
    ed = {(r["N"], r["nmax"], round(float(r["W_over_t"]), 12)): r
          for r in imported_ed if r["quality_flag"] == "ok"}
    rows = []
    for row in peaks:
        key = (row["N"], row["nmax"], round(float(row["W_over_t"]), 12))
        if row["observable"] != "S_ED" and row["quality_flag"] == "ok" and key in ed:
            rows.append({"L": 7, "N": row["N"], "nmax": row["nmax"], "W_over_t": row["W_over_t"],
                         "observable": row["observable"], "U_model": row["U_peak"],
                         "U_ED": ed[key]["U_peak"], "residual": float(row["U_peak"] - ed[key]["U_peak"])})
    return rows


def _make_imported_ed_fits(cfg: dict, peaks: List[dict], imported_ed: List[dict]):
    """Fit imported ED peak positions without inventing unavailable bootstrap samples."""
    if not imported_ed:
        return [], []
    lookup = {(r["N"], r["nmax"], round(float(r["W_over_t"]), 12), r["observable"]): r for r in peaks}
    Wmins = [float(x) for x in cfg["analysis"].get("fit_Wmin_values", [0.8, 1.0, 1.2, 1.4])]
    Wmax = float(cfg["analysis"].get("fit_Wmax", 2.5))
    fit_rows, direct_rows = [], []
    for N, nmax in sorted({(r["N"], r["nmax"]) for r in imported_ed}):
        all_w = sorted({round(float(r["W_over_t"]), 12) for r in imported_ed
                        if (r["N"], r["nmax"], r["quality_flag"]) == (N, nmax, "ok")})
        for Wmin in Wmins:
            common = [W for W in all_w if Wmin <= W <= Wmax
                      and (N, nmax, W, "M_existing") in lookup
                      and lookup[(N, nmax, W, "M_existing")]["quality_flag"] == "ok"]
            if len(common) < 3:
                continue
            x = np.asarray(common)
            y = np.asarray([lookup[(N, nmax, W, "S_ED")]["U_peak"] for W in common])
            m = np.asarray([lookup[(N, nmax, W, "M_existing")]["U_peak"] for W in common])
            stats, mstats = _linear_statistics(x, y), _linear_statistics(x, m)
            eta = stats["slope"] / mstats["slope"] if abs(mstats["slope"]) > 1e-15 else math.nan
            covariance = stats["covariance"]
            fit_rows.append({"L": 7, "N": N, "nmax": nmax, "observable": "S_ED",
                             "Wmin": Wmin, "Wmax": Wmax, "n_points": len(common),
                             "intercept_b": stats["intercept"], "slope_c": stats["slope"],
                             "cov_bb": float(covariance[0, 0]), "cov_bc": float(covariance[0, 1]),
                             "cov_cc": float(covariance[1, 1]), "slope_ci_low": math.nan,
                             "slope_ci_high": math.nan, "R2": stats["R2"], "RMSE": stats["RMSE"],
                             "eta_to_M": eta, "eta_ci_low": math.nan, "eta_ci_high": math.nan,
                             "bootstrap_valid": 0})
            direct = _linear_statistics(m, y)
            direct_rows.append({"L": 7, "N": N, "nmax": nmax, "observable": "S_ED",
                                "Wmin": Wmin, "Wmax": Wmax, "n_points": len(common),
                                "intercept_q": direct["intercept"], "eta_direct": direct["slope"],
                                "R2": direct["R2"], "RMSE": direct["RMSE"]})
    return fit_rows, direct_rows


def _write_raw_long_csv(cfg: dict, groups: dict, U: np.ndarray, path: Path) -> None:
    fields = ["L", "N", "nmax", "W_over_t", "U_over_t", "sample_id", "seed",
              "M_existing", "S2_existing", "M_sum", "M_per_edge", "S2_sum", "S2_per_edge",
              "Sstar_mean", "Sstar_median", "Sstar_std", "Pstar_mean", "z_mean",
              "resonant_edges_mean", "number_of_configurations", "energy_window"]
    selection_text = json.dumps(cfg["sampling"]["central_selection"], sort_keys=True, separators=(",", ":"))
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (N, nmax, W), (pieces, order) in sorted(groups.items()):
            joined = {name: np.concatenate([p[name] for p in pieces], axis=0)[order] for name in STORED}
            sample_ids = np.concatenate([p["sample_ids"] for p in pieces])[order]
            seeds = np.concatenate([p["seeds"] for p in pieces])[order]
            counts = np.concatenate([p["number_of_configurations"] for p in pieces], axis=0)[order]
            for si, sample_id in enumerate(sample_ids):
                for ui, value_u in enumerate(U):
                    writer.writerow({"L": 7, "N": N, "nmax": nmax, "W_over_t": W,
                                     "U_over_t": float(value_u), "sample_id": int(sample_id), "seed": int(seeds[si]),
                                     "M_existing": joined["M_existing"][si, ui], "S2_existing": joined["S2_existing"][si, ui],
                                     "M_sum": joined["M_graph_sum"][si, ui], "M_per_edge": joined["M_graph_per_edge"][si, ui],
                                     "S2_sum": joined["S2_graph_sum"][si, ui], "S2_per_edge": joined["S2_graph_per_edge"][si, ui],
                                     "Sstar_mean": joined["Sstar"][si, ui], "Sstar_median": joined["Sstar_median"][si, ui],
                                     "Sstar_std": joined["Sstar_std"][si, ui], "Pstar_mean": joined["Pstar"][si, ui],
                                     "z_mean": joined["z_mean"][si, ui], "resonant_edges_mean": joined["resonant_edges"][si, ui],
                                     "number_of_configurations": int(counts[si, ui]), "energy_window": selection_text})


def make_figures(cfg: dict, curves: dict, channel_curves: dict, peaks: List[dict],
                 fits: List[dict], comparisons: List[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"M_existing": "#0072B2", "S2_existing": "#D55E00", "Sstar": "#009E73",
              "M_graph_sum": "#56B4E9", "S2_graph_sum": "#E69F00", "S_ED": "black"}
    W_values = regular_grid(cfg["grid"]["W"])
    selected_W = W_values[::max(1, len(W_values) // 5)]
    for N, nmax in sorted({key[:2] for key in curves}):
        fig, ax = plt.subplots(figsize=(5.2, 3.7))
        rejected_label = False
        names = PRIMARY_OBSERVABLES + (("S_ED",) if any(r["observable"] == "S_ED" for r in peaks) else ())
        for observable in names:
            valid = sorted([r for r in peaks if (r["N"], r["nmax"], r["observable"], r["quality_flag"])
                            == (N, nmax, observable, "ok")], key=lambda r: r["W_over_t"])
            if valid:
                low = [max(0.0, r["U_peak"] - r["U_peak_ci_low"]) if np.isfinite(float(r["U_peak_ci_low"])) else 0.0 for r in valid]
                high = [max(0.0, r["U_peak_ci_high"] - r["U_peak"]) if np.isfinite(float(r["U_peak_ci_high"])) else 0.0 for r in valid]
                ax.errorbar([r["W_over_t"] for r in valid], [r["U_peak"] for r in valid], yerr=[low, high],
                            fmt="o-" if observable == "S_ED" else "o", ms=4, capsize=2,
                            color=colors[observable], label=observable)
            rejected = sorted([r for r in peaks if (r["N"], r["nmax"], r["observable"])
                               == (N, nmax, observable) and r["quality_flag"] != "ok"], key=lambda r: r["W_over_t"])
            if rejected and observable != "S_ED":
                ax.plot([r["W_over_t"] for r in rejected],
                        [r["U_peak"] if np.isfinite(r["U_peak"]) else r["grid_argmax"] for r in rejected],
                        linestyle="none", marker="x", ms=4, alpha=0.35, color=colors[observable],
                        label="rejected candidate" if not rejected_label else None)
                rejected_label = True
        ax.set(xlabel=r"$W/t$", ylabel=r"$U^*/t$", title=rf"$L=7,N={N},n_{{max}}={nmax}$")
        ax.legend(frameon=False, fontsize=8)
        _save_both(fig, output / "peaks_L7_N{}_nmax{}".format(N, nmax))
        plt.close(fig)

        for W in selected_W:
            W = float(W)
            fig, ax = plt.subplots(figsize=(5.2, 3.7))
            for observable in PRIMARY_OBSERVABLES:
                x, mean, _ = curves[(N, nmax, W, observable)]
                if np.max(mean) > 0:
                    ax.plot(x, mean / np.max(mean), color=colors[observable], label=observable)
            ax.set(xlabel=r"$U/t$", ylabel="estimator / max",
                   title=rf"$L=7,N={N},n_{{max}}={nmax},W/t={W:g}$")
            ax.legend(frameon=False, fontsize=8)
            tag = str(round(W, 3)).replace(".", "p")
            _save_both(fig, output / "curves_L7_N{}_nmax{}_W{}".format(N, nmax, tag))
            plt.close(fig)

            fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharex=True)
            for axis, quantity in zip(axes, ("M_k", "S2_k")):
                x, channel_k, mean, _ = channel_curves[(N, nmax, W, quantity)]
                for ki, k in enumerate(channel_k):
                    axis.plot(x, mean[:, ki], label="k={}".format(int(k)))
                axis.set(xlabel=r"$U/t$", ylabel=quantity)
            axes[1].legend(frameon=False, fontsize=7, ncol=2)
            fig.suptitle(rf"channels: $L=7,N={N},n_{{max}}={nmax},W/t={W:g}$")
            _save_both(fig, output / "channels_L7_N{}_nmax{}_W{}".format(N, nmax, tag))
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        m = {float(r["W_over_t"]): r for r in peaks if (r["N"], r["nmax"], r["observable"], r["quality_flag"])
             == (N, nmax, "M_existing", "ok")}
        for observable in ("S2_existing", "Sstar"):
            other = {float(r["W_over_t"]): r for r in peaks if (r["N"], r["nmax"], r["observable"], r["quality_flag"])
                     == (N, nmax, observable, "ok")}
            common = sorted(set(m) & set(other))
            if common:
                ax.plot([m[w]["U_peak"] for w in common], [other[w]["U_peak"] for w in common], "o",
                        color=colors[observable], label=observable)
        limits = ax.get_xlim()
        ax.plot(limits, limits, "k--", lw=1)
        ax.set(xlabel=r"$U_M^*/t$", ylabel=r"$U_X^*/t$", title=rf"$L=7,N={N},n_{{max}}={nmax}$")
        ax.legend(frameon=False, fontsize=8)
        _save_both(fig, output / "direct_relation_L7_N{}_nmax{}".format(N, nmax))
        plt.close(fig)

    if comparisons:
        for N, nmax in sorted({(r["N"], r["nmax"]) for r in comparisons}):
            fig, ax = plt.subplots(figsize=(5.0, 3.4))
            for observable in PRIMARY_OBSERVABLES:
                selected = sorted([r for r in comparisons if (r["N"], r["nmax"], r["observable"])
                                   == (N, nmax, observable)], key=lambda r: r["W_over_t"])
                if selected:
                    ax.plot([r["W_over_t"] for r in selected], [r["residual"] for r in selected], "o-",
                            color=colors[observable], label=observable)
            ax.axhline(0.0, color="k", ls="--", lw=1)
            ax.set(xlabel=r"$W/t$", ylabel=r"$U_X^*-U_{ED}^*$")
            ax.legend(frameon=False, fontsize=8)
            _save_both(fig, output / "residuals_ED_L7_N{}_nmax{}".format(N, nmax))
            plt.close(fig)

    preferred_wmin = float(cfg["analysis"].get("fit_Wmin_values", [0.8])[0])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    eta_plotted = False
    for observable in ("S2_existing", "Sstar", "S_ED"):
        selected = sorted([r for r in fits if r["observable"] == observable
                           and abs(float(r["Wmin"]) - preferred_wmin) < 1e-12], key=lambda r: r["N"])
        if not selected:
            continue
        eta_plotted = True
        x = [r["N"] for r in selected]
        y = [r["eta_to_M"] for r in selected]
        if all(np.isfinite(r["eta_ci_low"]) and np.isfinite(r["eta_ci_high"]) for r in selected):
            low = [max(0.0, r["eta_to_M"] - r["eta_ci_low"]) for r in selected]
            high = [max(0.0, r["eta_ci_high"] - r["eta_to_M"]) for r in selected]
            ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3,
                        color=colors[observable], label=observable)
        else:
            ax.plot(x, y, "o-", color=colors[observable], label=observable)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set(xlabel=r"$N=n_{max}$", ylabel=r"$\eta_X=c_X/c_M$", title=rf"$L=7, W_{{min}}/t={preferred_wmin:g}$")
    if eta_plotted:
        ax.legend(frameon=False, fontsize=8)
    _save_both(fig, output / "eta_vs_N_L7")
    plt.close(fig)


def _save_both(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    args = arguments()
    cfg = load_config(args.config)
    if args.command == "validate":
        validate(cfg)
    elif args.command == "init":
        initialize(cfg, args.force)
    elif args.command == "worker":
        worker(cfg, args.task_id)
    elif args.command == "status":
        status(cfg)
    elif args.command == "analyze":
        analyze(cfg, args.no_plots)


if __name__ == "__main__":
    main()
