#!/usr/bin/env python3
"""Fast, focused L=7 star-model scan and peak analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from star_l7_core import (VERSION, build_graph, central_configurations, configuration_priority,
                          deterministic_seed, observables_at_u, verify_detuning)


OBSERVABLES = ("M_sum", "S2_sum", "Sstar")
STORED = OBSERVABLES + ("M_per_edge", "S2_per_edge", "Sstar_median", "Sstar_std",
                        "Pstar", "z_mean", "resonant_edges")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "init", "status", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name == "init":
            p.add_argument("--force", action="store_true")
        if name == "analyze":
            p.add_argument("--no-plots", action="store_true", help="write CSV/summary without importing Matplotlib")
    p = sub.add_parser("worker")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--task-id", type=int, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_path"] = str(path.resolve())
    return cfg


def regular_grid(spec: dict) -> np.ndarray:
    start, stop, step = map(float, (spec["start"], spec["stop"], spec["step"]))
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1)
    if step <= 0 or abs(values[-1] - stop) > 1e-10:
        raise ValueError(f"Invalid regular grid: {spec}")
    return values


def root_path(cfg: dict) -> Path:
    root = Path(cfg["paths"]["results"])
    return root if root.is_absolute() else Path(cfg["_path"]).parent / root


def tasks(cfg: dict) -> list[dict]:
    output = []
    chunk = int(cfg["sampling"]["samples_per_task"])
    count = int(cfg["sampling"]["n_disorder"])
    task_id = 0
    for sector_id, sector in enumerate(cfg["model"]["sectors"]):
        for W_id, W in enumerate(regular_grid(cfg["grid"]["W"])):
            for first in range(0, count, chunk):
                output.append({"task_id": task_id, "sector_id": sector_id, "N": int(sector["N"]),
                               "nmax": int(sector["nmax"]), "W_id": W_id, "W": float(W),
                               "sample_first": first, "sample_stop": min(first + chunk, count)})
                task_id += 1
    return output


def validate(cfg: dict) -> None:
    if int(cfg["model"]["L"]) != 7:
        raise ValueError("This focused implementation requires L=7")
    if cfg["model"]["boundary"] != "open" or cfg["model"]["interaction"] != "U_n_nminus1":
        raise ValueError("Require open boundaries and U_n_nminus1 interaction")
    if cfg["disorder"]["distribution"] != "uniform_minus_W_plus_W":
        raise ValueError("Require independent uniform disorder on [-W,W]")
    U, W = regular_grid(cfg["grid"]["U"]), regular_grid(cfg["grid"]["W"])
    for sector in cfg["model"]["sectors"]:
        graph = build_graph(7, int(sector["N"]), int(sector["nmax"]), float(cfg["model"]["t"]))
        print(f"L=7 N={sector['N']} nmax={sector['nmax']}: dim={graph.dim}, zmax={graph.neighbors.shape[1]}")
    print(f"U/t=[{U[0]:g},{U[-1]:g}], dU={U[1]-U[0]:g}, points={len(U)}")
    print(f"W/t=[{W[0]:g},{W[-1]:g}], dW={W[1]-W[0]:g}, points={len(W)}")
    print(f"disorder={cfg['sampling']['n_disorder']}, configurations={cfg['sampling']['n_configurations']}, tasks={len(tasks(cfg))}")
    print("Hamiltonian: H_diag = U*sum_i n_i(n_i-1) + sum_i epsilon_i*n_i")
    print("Disorder: independent epsilon_i/W uniform on [-1,1]")
    print("No full many-body diagonalization; batched local star matrices only.")


def initialize(cfg: dict, force: bool) -> None:
    validate(cfg)
    root = root_path(cfg)
    manifest = root / "tasks.csv"
    if manifest.exists() and not force:
        raise SystemExit(f"Manifest exists: {manifest}")
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "analysis" / "figures").mkdir(parents=True, exist_ok=True)
    rows = tasks(cfg)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {key: value for key, value in cfg.items() if not key.startswith("_")}
    metadata.update({"version": VERSION, "full_ED": False, "star_solver": "batched_numpy_eigh",
                     "branch_selection": "largest_central_overlap_average_ties"})
    (root / "configuration.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Initialized {len(rows)} tasks in {root}")


def output_path(cfg: dict, task_id: int) -> Path:
    return root_path(cfg) / "raw" / f"task_{task_id:04d}.npz"


def valid_output(path: Path, task: dict, U: np.ndarray) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (str(data["version"]) == VERSION
                    and all(name in data.files for name in STORED)
                    and int(data["N"]) == task["N"] and int(data["nmax"]) == task["nmax"]
                    and abs(float(data["W"]) - task["W"]) < 1e-12
                    and np.array_equal(data["U"], U)
                    and np.array_equal(data["sample_ids"], np.arange(task["sample_first"], task["sample_stop"])))
    except Exception:
        return False


def worker(cfg: dict, task_id: int) -> None:
    rows = tasks(cfg)
    if not 0 <= task_id < len(rows):
        raise SystemExit(f"task-id outside [0,{len(rows)-1}]")
    task = rows[task_id]
    U_values = regular_grid(cfg["grid"]["U"])
    path = output_path(cfg, task_id)
    if valid_output(path, task, U_values):
        print(f"SKIP {path}")
        return
    N, nmax, W = task["N"], task["nmax"], task["W"]
    graph = build_graph(7, N, nmax, float(cfg["model"]["t"]))
    sample_ids = np.arange(task["sample_first"], task["sample_stop"], dtype=np.int32)
    arrays = {name: np.empty((len(sample_ids), len(U_values)), dtype=np.float64) for name in STORED}
    maximum_cfg = int(cfg["sampling"]["n_configurations"])
    fraction = float(cfg["sampling"]["energy_window_fraction"])
    tie_tolerance = float(cfg["numerics"]["tie_tolerance"])
    master = int(cfg["disorder"]["master_seed"])
    started = time.monotonic()
    for si, sample_id in enumerate(sample_ids):
        seed = deterministic_seed(master, N, int(sample_id))
        rng = np.random.default_rng(seed)
        epsilon = W * rng.uniform(-1.0, 1.0, size=7)
        disorder_energy = graph.basis @ epsilon
        priority = configuration_priority(graph.basis, seed)
        error = verify_detuning(graph, float(U_values[len(U_values)//2]), epsilon, seed & 0xFFFFFFFF)
        if error > float(cfg["numerics"]["detuning_tolerance"]):
            raise RuntimeError(f"detuning identity error {error:.3e}")
        for ui, U in enumerate(U_values):
            energies = U * graph.interaction + disorder_energy
            central = central_configurations(energies, fraction, maximum_cfg, priority)
            result = observables_at_u(graph, float(U), disorder_energy, central, tie_tolerance)
            if result["normalization_error"] > float(cfg["numerics"]["normalization_tolerance"]):
                raise RuntimeError(f"normalization error {result['normalization_error']:.3e}")
            for name in STORED:
                arrays[name][si, ui] = result[name]
        elapsed = time.monotonic() - started
        eta = elapsed/(si+1)*(len(sample_ids)-si-1)
        print(f"task={task_id} sample={si+1}/{len(sample_ids)} elapsed={elapsed:.2f}s eta={eta:.2f}s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, version=VERSION, L=7, N=N, nmax=nmax, W=W, U=U_values,
                        sample_ids=sample_ids, **arrays)
    os.replace(temporary, path)
    print(f"Saved {path}; total={time.monotonic()-started:.2f}s")


def status(cfg: dict) -> None:
    U = regular_grid(cfg["grid"]["U"])
    rows = tasks(cfg)
    complete = [valid_output(output_path(cfg, row["task_id"]), row, U) for row in rows]
    print(f"Complete {sum(complete)}/{len(rows)}; missing {len(rows)-sum(complete)}")
    missing = [str(row["task_id"]) for row, ok in zip(rows, complete) if not ok]
    if missing:
        print("Missing task IDs: " + ",".join(missing))


def smoothing_matrix(length: int, window: int, degree: int) -> np.ndarray:
    """Linear local-polynomial smoother, precomputed once for all bootstraps."""
    window = max(3, int(window) | 1)
    radius = window // 2
    matrix = np.zeros((length, length), dtype=float)
    for i in range(length):
        lo, hi = max(0, i-radius), min(length, i+radius+1)
        xx = np.arange(lo-i, hi-i, dtype=float)
        fit_degree = min(degree, len(xx)-1)
        vandermonde = np.vander(xx, fit_degree + 1)
        evaluation = np.zeros(fit_degree + 1)
        evaluation[-1] = 1.0
        matrix[i, lo:hi] = evaluation @ np.linalg.pinv(vandermonde)
    return matrix


def smooth_curve(y: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix @ np.asarray(y, dtype=float)


def locate_peak(x: np.ndarray, y: np.ndarray, cfg: dict, smoother: np.ndarray | None = None) -> dict:
    if smoother is None:
        smoother = smoothing_matrix(len(y), int(cfg["analysis"]["smoothing_window"]),
                                    int(cfg["analysis"]["smoothing_degree"]))
    smoothed = smooth_curve(y, smoother)
    return locate_smoothed_peak(x, smoothed, cfg)


def locate_smoothed_peak(x: np.ndarray, smoothed: np.ndarray, cfg: dict) -> dict:
    imax = int(np.argmax(smoothed))
    margin = int(cfg["analysis"]["boundary_margin_points"])
    if imax < margin or imax >= len(x)-margin:
        return {"U_peak": math.nan, "peak_value": float(smoothed[imax]), "curvature": math.nan,
                "grid_argmax": float(x[imax]), "quality_flag": "boundary"}
    half = int(cfg["analysis"]["peak_fit_points"]) // 2
    lo, hi = imax-half, imax+half+1
    a, b, c = np.polyfit(x[lo:hi], smoothed[lo:hi], 2)
    if a >= 0:
        return {"U_peak": math.nan, "peak_value": float(smoothed[imax]), "curvature": float(2*a),
                "grid_argmax": float(x[imax]), "quality_flag": "nonnegative_curvature"}
    peak = float(-b/(2*a))
    if not x[lo] <= peak <= x[hi-1]:
        return {"U_peak": math.nan, "peak_value": float(smoothed[imax]), "curvature": float(2*a),
                "grid_argmax": float(x[imax]), "quality_flag": "vertex_outside"}
    return {"U_peak": peak, "peak_value": float(a*peak*peak+b*peak+c), "curvature": float(2*a),
            "grid_argmax": float(x[imax]), "quality_flag": "ok"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(cfg: dict, no_plots: bool = False) -> None:
    rows = tasks(cfg)
    U = regular_grid(cfg["grid"]["U"])
    missing = [row for row in rows if not valid_output(output_path(cfg, row["task_id"]), row, U)]
    if missing:
        raise SystemExit(f"Cannot analyze: {len(missing)} tasks missing")
    groups: dict[tuple[int, int, float], list[dict]] = {}
    for task in rows:
        with np.load(output_path(cfg, task["task_id"]), allow_pickle=False) as data:
            record = {name: data[name].copy() for name in STORED}
            record["sample_ids"] = data["sample_ids"].copy()
            groups.setdefault((task["N"], task["nmax"], task["W"]), []).append(record)
    analysis_dir = root_path(cfg) / "analysis"
    figure_dir = analysis_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    curve_rows: list[dict] = []
    peak_rows: list[dict] = []
    curves: dict[tuple[int,int,float,str], tuple[np.ndarray,np.ndarray,np.ndarray]] = {}
    B = int(cfg["analysis"]["bootstrap_replicates"])
    alpha = (1-float(cfg["analysis"]["confidence_level"]))/2
    smoother = smoothing_matrix(len(U), int(cfg["analysis"]["smoothing_window"]),
                                int(cfg["analysis"]["smoothing_degree"]))
    for (N, nmax, W), pieces in sorted(groups.items()):
        pieces.sort(key=lambda p: int(np.min(p["sample_ids"])))
        sample_ids = np.concatenate([p["sample_ids"] for p in pieces])
        order = np.argsort(sample_ids)
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise RuntimeError(f"Duplicate sample IDs for N={N}, nmax={nmax}, W={W}")
        expected = np.arange(int(cfg["sampling"]["n_disorder"]))
        if not np.array_equal(np.sort(sample_ids), expected):
            raise RuntimeError(f"Incomplete sample IDs for N={N}, nmax={nmax}, W={W}")
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg["analysis"]["bootstrap_seed"]), N, nmax, int(round(1000*W))]))
        indices = rng.integers(0, len(sample_ids), size=(B, len(sample_ids))) if B else np.empty((0,len(sample_ids)), dtype=int)
        for observable in OBSERVABLES:
            values = np.concatenate([p[observable] for p in pieces], axis=0)[order]
            mean = np.mean(values, axis=0)
            stderr = (np.std(values, axis=0, ddof=1)/math.sqrt(len(values))
                      if len(values) > 1 else np.zeros_like(mean))
            curves[(N,nmax,W,observable)] = (U, mean, stderr)
            for ui, value_u in enumerate(U):
                curve_rows.append({"L": 7, "N": N, "nmax": nmax, "W_over_t": W, "U_over_t": float(value_u),
                                   "observable": observable, "mean": float(mean[ui]), "stderr": float(stderr[ui]),
                                   "n_disorder": len(values)})
            central = locate_peak(U, mean, cfg, smoother)
            boot = []
            if central["quality_flag"] == "ok":
                bootstrap_means = np.mean(values[indices], axis=1)
                bootstrap_smoothed = bootstrap_means @ smoother.T
                for smoothed in bootstrap_smoothed:
                    result = locate_smoothed_peak(U, smoothed, cfg)
                    if result["quality_flag"] == "ok":
                        boot.append(result["U_peak"])
            low, high = np.quantile(boot, [alpha,1-alpha]) if boot else (math.nan,math.nan)
            quality = central["quality_flag"]
            if quality == "ok" and B and len(boot) < float(cfg["analysis"].get("min_bootstrap_valid_fraction", 0.8))*B:
                quality = "unstable_bootstrap"
            if (quality == "ok" and np.isfinite(low) and np.isfinite(high)
                    and high-low > float(cfg["analysis"].get("max_peak_ci_range_fraction", 0.5))*(U[-1]-U[0])):
                quality = "wide_ci"
            central["quality_flag"] = quality
            peak_rows.append({"L": 7, "N": N, "nmax": nmax, "W_over_t": W, "observable": observable,
                              **central, "fit_method": "local_polynomial_smoothing_then_quadratic",
                              "U_peak_ci_low": low, "U_peak_ci_high": high,
                              "bootstrap_valid": len(boot), "n_disorder": len(values)})
    write_csv(analysis_dir / "curves.csv", curve_rows)
    write_csv(analysis_dir / "peaks.csv", peak_rows)
    if not no_plots:
        make_figures(cfg, curves, peak_rows, figure_dir)
    valid = sum(row["quality_flag"] == "ok" for row in peak_rows)
    summary = (f"L=7 vectorized star model v{VERSION}\nValid peaks: {valid}/{len(peak_rows)}\n"
               f"U range: {U[0]:g}..{U[-1]:g}; W range: {regular_grid(cfg['grid']['W'])[0]:g}..{regular_grid(cfg['grid']['W'])[-1]:g}\n")
    (analysis_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def make_figures(cfg: dict, curves: dict, peaks: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"M_sum":"#0072B2", "S2_sum":"#D55E00", "Sstar":"#009E73"}
    W_values = regular_grid(cfg["grid"]["W"])
    selected_W = W_values[::max(1, len(W_values)//6)]
    for N, nmax in sorted({key[:2] for key in curves}):
        fig, ax = plt.subplots(figsize=(4.4,3.2))
        for observable in OBSERVABLES:
            valid = sorted([r for r in peaks if (r["N"],r["nmax"],r["observable"],r["quality_flag"]) == (N,nmax,observable,"ok")], key=lambda r:r["W_over_t"])
            if not valid:
                continue
            lower = [max(0,r["U_peak"]-r["U_peak_ci_low"]) if np.isfinite(r["U_peak_ci_low"]) else 0 for r in valid]
            upper = [max(0,r["U_peak_ci_high"]-r["U_peak"]) if np.isfinite(r["U_peak_ci_high"]) else 0 for r in valid]
            ax.errorbar([r["W_over_t"] for r in valid], [r["U_peak"] for r in valid], yerr=[lower,upper],
                        marker="o", ms=3, capsize=2, color=colors[observable], label=observable)
        ax.set(xlabel=r"$W/t$", ylabel=r"$U^*/t$", title=rf"$L=7,N=n_{{max}}={N}$")
        if ax.lines:
            ax.legend(frameon=False)
        for suffix in ("png","pdf"):
            fig.savefig(output/f"peaks_L7_N{N}_nmax{nmax}.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(fig)
        for W in selected_W:
            fig, ax = plt.subplots(figsize=(4.4,3.2))
            for observable in OBSERVABLES:
                x, mean, stderr = curves[(N,nmax,float(W),observable)]
                scale = float(np.max(mean))
                if scale > 0:
                    ax.plot(x, mean/scale, color=colors[observable], label=observable)
            ax.set(xlabel=r"$U/t$", ylabel="normalized estimator", title=rf"$L=7,N=n_{{max}}={N},W/t={W:g}$")
            ax.legend(frameon=False)
            tag = str(round(float(W),3)).replace(".","p")
            for suffix in ("png","pdf"):
                fig.savefig(output/f"curves_L7_N{N}_nmax{nmax}_W{tag}.{suffix}", dpi=220, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = arguments(); cfg = load_config(args.config)
    if args.command == "validate": validate(cfg)
    elif args.command == "init": initialize(cfg, args.force)
    elif args.command == "worker": worker(cfg, args.task_id)
    elif args.command == "status": status(cfg)
    elif args.command == "analyze": analyze(cfg, args.no_plots)


if __name__ == "__main__":
    main()
