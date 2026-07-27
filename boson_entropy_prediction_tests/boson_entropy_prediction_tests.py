#!/usr/bin/env python3
"""Bosonic entropy tests from existing observables_*.csv files.

This script implements a compact, data-driven version of the tests described in
``instrukcja_testy_bozony_entropy.tex``.  It reads realization-level observable
files produced by the bosonic filling studies and writes CSV summaries plus
PRB-style diagnostic figures.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PRB_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
PRB_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]


@dataclass(frozen=True)
class AggRow:
    L: int
    N: int
    nmax: int
    U: float
    W: float
    dim: int
    observable: str
    entropy_mean: float
    entropy_stderr: float
    pr_normalized_mean: float
    pr_normalized_stderr: float
    pair_density_mean: float
    pair_density_stderr: float
    n_realizations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("boson_entropy_prediction_results"))
    parser.add_argument(
        "--observable",
        choices=("fock_entropy_normalized", "ln_pr_normalized", "pr_normalized"),
        default="fock_entropy_normalized",
        help="Main entropy-like observable. ln_pr_normalized means ln(PR)/ln(D).",
    )
    parser.add_argument("--L-list", default="", help="Comma-separated L values to keep.")
    parser.add_argument("--N-list", default="", help="Comma-separated N values to keep.")
    parser.add_argument("--nmax-list", default="", help="Comma-separated nmax values to keep.")
    parser.add_argument("--min-U", type=float)
    parser.add_argument("--max-U", type=float)
    parser.add_argument("--min-W", type=float)
    parser.add_argument("--max-W", type=float)
    parser.add_argument("--tail-points", type=int, default=2)
    parser.add_argument("--half-level", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-grid", type=int, default=2001)
    parser.add_argument("--kappa-U-reference", default="min", choices=("min", "first"))
    parser.add_argument("--chi-U", type=float, default=2.0)
    parser.add_argument("--plot-W-list", default="1,3,5,7,10,15")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def finite(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def parse_int_set(text: str) -> set[int] | None:
    if not text.strip():
        return None
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def parse_float_list(text: str) -> list[float]:
    if not text.strip():
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def bounded_composition_dim(L: int, N: int, nmax: int) -> int:
    coeff = [0] * (N + 1)
    coeff[0] = 1
    for _ in range(L):
        new = [0] * (N + 1)
        for current in range(N + 1):
            value = coeff[current]
            if value == 0:
                continue
            for n in range(min(nmax, N - current) + 1):
                new[current + n] += value
        coeff = new
    return coeff[N]


def bounded_coeffs(L: int, Nmax: int, nmax: int) -> list[int]:
    coeff = [0] * (Nmax + 1)
    coeff[0] = 1
    for _ in range(L):
        new = [0] * (Nmax + 1)
        for current, value in enumerate(coeff):
            if value == 0:
                continue
            for n in range(min(nmax, Nmax - current) + 1):
                new[current + n] += value
        coeff = new
    return coeff


def mean_stderr(values) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan, 0
    stderr = float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else math.nan
    return float(np.mean(array)), stderr, int(len(array))


def read_rows(input_dirs: list[Path], observable: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    paths = sorted({path.resolve() for input_dir in input_dirs for path in input_dir.rglob("observables_*.csv")})
    for path in paths:
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                row = {key: finite(value) for key, value in raw.items()}
                if not all(math.isfinite(row.get(key, math.nan)) for key in ("L", "N", "nmax", "U", "W")):
                    continue
                L, N, nmax = int(row["L"]), int(row["N"]), int(row["nmax"])
                dim = int(round(row["dim"])) if math.isfinite(row.get("dim", math.nan)) else bounded_composition_dim(L, N, nmax)
                row["dim"] = dim
                if not math.isfinite(row.get("pr_normalized", math.nan)) and math.isfinite(row.get("pr", math.nan)):
                    row["pr_normalized"] = row["pr"] / dim if dim > 0 else math.nan
                if not math.isfinite(row.get("fock_entropy_normalized", math.nan)) and math.isfinite(row.get("fock_entropy", math.nan)):
                    row["fock_entropy_normalized"] = row["fock_entropy"] / math.log(dim) if dim > 1 else math.nan
                if math.isfinite(row.get("pr", math.nan)) and dim > 1:
                    row["ln_pr_normalized"] = math.log(max(row["pr"], 1e-300)) / math.log(dim)
                elif math.isfinite(row.get("pr_normalized", math.nan)) and dim > 1:
                    row["ln_pr_normalized"] = math.log(max(row["pr_normalized"] * dim, 1e-300)) / math.log(dim)
                else:
                    row["ln_pr_normalized"] = math.nan
                if math.isfinite(row.get(observable, math.nan)):
                    row["_path"] = str(path)
                    rows.append(row)
    return rows


def filter_rows(rows: list[dict[str, float]], args: argparse.Namespace) -> list[dict[str, float]]:
    keep_L = parse_int_set(args.L_list)
    keep_N = parse_int_set(args.N_list)
    keep_nmax = parse_int_set(args.nmax_list)
    output = []
    for row in rows:
        L, N, nmax = int(row["L"]), int(row["N"]), int(row["nmax"])
        U, W = row["U"], row["W"]
        if keep_L is not None and L not in keep_L:
            continue
        if keep_N is not None and N not in keep_N:
            continue
        if keep_nmax is not None and nmax not in keep_nmax:
            continue
        if args.min_U is not None and U < args.min_U:
            continue
        if args.max_U is not None and U > args.max_U:
            continue
        if args.min_W is not None and W < args.min_W:
            continue
        if args.max_W is not None and W > args.max_W:
            continue
        output.append(row)
    return output


def aggregate(rows: list[dict[str, float]], observable: str) -> list[AggRow]:
    groups: dict[tuple[int, int, int, float, float], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        key = (int(row["L"]), int(row["N"]), int(row["nmax"]), float(row["U"]), float(row["W"]))
        groups[key].append(row)

    output = []
    for (L, N, nmax, U, W), group in sorted(groups.items()):
        entropy_mean, entropy_stderr, nobs = mean_stderr(row.get(observable, math.nan) for row in group)
        pr_mean, pr_stderr, _ = mean_stderr(row.get("pr_normalized", math.nan) for row in group)
        pair_mean, pair_stderr, _ = mean_stderr(row.get("onsite_pair_density", math.nan) for row in group)
        dim = int(round(np.nanmedian([row["dim"] for row in group])))
        output.append(
            AggRow(
                L=L,
                N=N,
                nmax=nmax,
                U=U,
                W=W,
                dim=dim,
                observable=observable,
                entropy_mean=entropy_mean,
                entropy_stderr=entropy_stderr,
                pr_normalized_mean=pr_mean,
                pr_normalized_stderr=pr_stderr,
                pair_density_mean=pair_mean,
                pair_density_stderr=pair_stderr,
                n_realizations=nobs,
            )
        )
    return output


def asdict_row(row: AggRow) -> dict[str, float | int | str]:
    return {
        "L": row.L,
        "N": row.N,
        "nmax": row.nmax,
        "filling": row.N / row.L,
        "U": row.U,
        "W": row.W,
        "dim": row.dim,
        "observable": row.observable,
        "entropy_mean": row.entropy_mean,
        "entropy_stderr": row.entropy_stderr,
        "pr_normalized_mean": row.pr_normalized_mean,
        "pr_normalized_stderr": row.pr_normalized_stderr,
        "onsite_pair_density_mean": row.pair_density_mean,
        "onsite_pair_density_stderr": row.pair_density_stderr,
        "n_realizations": row.n_realizations,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grouped_by(rows: list[AggRow], attrs: tuple[str, ...]) -> dict[tuple, list[AggRow]]:
    groups: dict[tuple, list[AggRow]] = defaultdict(list)
    for row in rows:
        groups[tuple(getattr(row, attr) for attr in attrs)].append(row)
    return {key: sorted(value, key=lambda r: (r.U, r.W)) for key, value in groups.items()}


def descending_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if not (math.isfinite(y0) and math.isfinite(y1)):
            continue
        if y0 == threshold:
            return float(x[i])
        if (y0 - threshold) * (y1 - threshold) <= 0 and y1 != y0:
            t = (threshold - y0) / (y1 - y0)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return math.nan


def normalized_decay_curve(curve: list[AggRow], tail_points: int) -> tuple[np.ndarray, np.ndarray, float]:
    curve = sorted(curve, key=lambda row: row.U)
    U = np.array([row.U for row in curve], dtype=float)
    s = np.array([row.entropy_mean for row in curve], dtype=float)
    finite = np.isfinite(U) & np.isfinite(s)
    U, s = U[finite], s[finite]
    if len(U) < 3:
        return U, np.full_like(U, np.nan), math.nan
    tail = max(1, min(tail_points, len(s)))
    s_inf = float(np.mean(s[-tail:]))
    denom = s[0] - s_inf
    if abs(denom) < 1e-12:
        return U, np.full_like(U, np.nan), s_inf
    return U, (s - s_inf) / denom, s_inf


def linear_fit(x, y) -> tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return math.nan, math.nan, math.nan, int(len(x))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(slope), float(intercept), r2, int(len(x))


def channel_weights(L: int, N: int, nmax: int) -> dict[int, float]:
    dim = bounded_composition_dim(L, N, nmax)
    if dim <= 0 or L < 2:
        return {}
    remainder = bounded_coeffs(L - 2, N, nmax)
    weights: dict[int, float] = defaultdict(float)
    for n in range(nmax):
        for m in range(1, nmax + 1):
            rest_n = N - n - m
            if 0 <= rest_n < len(remainder):
                q = n - m + 1
                weights[q] += (n + 1) * m * remainder[rest_n] / dim
    return dict(sorted(weights.items()))


def rq_integral(q: int, U: float, W: float, gamma: float, grid: int) -> float:
    if gamma <= 0:
        gamma = 1e-12
    if W <= 0:
        return 1.0 / ((q * U) ** 2 + gamma * gamma)
    grid = max(101, int(grid) | 1)
    delta = np.linspace(-W, W, grid)
    density = (W - np.abs(delta)) / (W * W)
    values = density / ((delta + q * U) ** 2 + gamma * gamma)
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(values, delta))


def lambda_site(L: int, weights: dict[int, float], U: float, W: float, gamma: float, grid: int, t: float = 1.0) -> float:
    return float(2 * (L - 1) * t * t * sum(aq * rq_integral(q, U, W, gamma, grid) for q, aq in weights.items()))


def partition_Z(L: int, N: int, nmax: int, kappa: float) -> float:
    weights = [math.exp(-kappa * n * (n - 1) / 2.0) for n in range(nmax + 1)]
    coeff = [0.0] * (N + 1)
    coeff[0] = 1.0
    for _ in range(L):
        new = [0.0] * (N + 1)
        for current, value in enumerate(coeff):
            if value == 0.0:
                continue
            for n, weight in enumerate(weights):
                if current + n <= N:
                    new[current + n] += value * weight
        coeff = new
    return coeff[N]


def setup_matplotlib():
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "mathtext.fontset": "stix",
                "font.size": 8.5,
                "axes.labelsize": 9,
                "axes.titlesize": 9,
                "axes.linewidth": 0.9,
                "lines.linewidth": 1.15,
                "legend.fontsize": 7,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
                "xtick.minor.visible": True,
                "ytick.minor.visible": True,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def savefig(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")


def label_sector(L: int, N: int, nmax: int) -> str:
    return rf"$L={L},\,N={N},\,n_{{max}}={nmax}$"


def analyze_monotonicity(rows: list[AggRow]) -> list[dict]:
    output = []
    for (L, N, nmax, W), curve in grouped_by(rows, ("L", "N", "nmax", "W")).items():
        curve = sorted(curve, key=lambda row: row.U)
        U = np.array([row.U for row in curve])
        s = np.array([row.entropy_mean for row in curve])
        if len(U) < 2:
            continue
        ds = np.diff(s)
        violation_fraction = float(np.mean(ds > 0))
        slope, intercept, r2, n_fit = linear_fit(U, s)
        output.append(
            {
                "L": L,
                "N": N,
                "nmax": nmax,
                "W": W,
                "slope_vs_U": slope,
                "intercept": intercept,
                "r2": r2,
                "n_U": n_fit,
                "mean_positive_step_fraction": violation_fraction,
                "monotone_decreasing_strict": int(np.all(ds <= 0)),
            }
        )
    return output


def analyze_uhalf(rows: list[AggRow], tail_points: int, threshold: float) -> tuple[list[dict], list[dict]]:
    by_curve = grouped_by(rows, ("L", "N", "nmax", "W"))
    uhalf_rows = []
    for (L, N, nmax, W), curve in by_curve.items():
        U, stilde, s_inf = normalized_decay_curve(curve, tail_points)
        uhalf = descending_crossing(U, stilde, threshold)
        uhalf_rows.append(
            {
                "L": L,
                "N": N,
                "nmax": nmax,
                "W": W,
                "Uhalf": uhalf,
                "half_level": threshold,
                "s_infinity_estimate": s_inf,
                "n_U": int(len(U)),
            }
        )
    fit_rows = []
    for (L, N, nmax), group in grouped_by_dict(uhalf_rows, ("L", "N", "nmax")).items():
        W = [row["W"] for row in group]
        Uhalf = [row["Uhalf"] for row in group]
        slope, intercept, r2, n_fit = linear_fit(W, Uhalf)
        fit_rows.append(
            {
                "L": L,
                "N": N,
                "nmax": nmax,
                "c_half_slope": slope,
                "b_half_intercept": intercept,
                "r2": r2,
                "n_W": n_fit,
            }
        )
    return uhalf_rows, fit_rows


def grouped_by_dict(rows: list[dict], attrs: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[attr] for attr in attrs)].append(row)
    return groups


def analyze_lambda(rows: list[AggRow], gamma: float, grid: int, tail_points: int) -> tuple[list[dict], list[dict], list[dict]]:
    sectors = sorted({(row.L, row.N, row.nmax) for row in rows})
    weight_rows = []
    qeff_rows = []
    weights_by_sector = {}
    for L, N, nmax in sectors:
        weights = channel_weights(L, N, nmax)
        weights_by_sector[(L, N, nmax)] = weights
        denom = sum(weights.values())
        qeff2 = sum(aq * q * q for q, aq in weights.items()) / denom if denom > 0 else math.nan
        qmean = sum(aq * q for q, aq in weights.items()) / denom if denom > 0 else math.nan
        for q, aq in weights.items():
            weight_rows.append({"L": L, "N": N, "nmax": nmax, "q": q, "a_q": aq})
        qeff_rows.append({"L": L, "N": N, "nmax": nmax, "q_eff2": qeff2, "q_mean": qmean, "sum_aq": denom})

    comparison = []
    for (L, N, nmax, W), curve in grouped_by(rows, ("L", "N", "nmax", "W")).items():
        U, stilde, _ = normalized_decay_curve(curve, tail_points)
        if len(U) < 3 or not np.any(np.isfinite(stilde)):
            continue
        weights = weights_by_sector[(L, N, nmax)]
        lam = np.array([lambda_site(L, weights, u, W, gamma, grid) for u in U], dtype=float)
        tail = max(1, min(tail_points, len(lam)))
        lam_inf = float(np.mean(lam[-tail:]))
        denom = lam[0] - lam_inf
        f = (lam - lam_inf) / denom if abs(denom) > 1e-14 else np.full_like(lam, np.nan)
        mask = np.isfinite(stilde) & np.isfinite(f)
        if np.sum(mask) >= 2:
            rmse = float(np.sqrt(np.mean((stilde[mask] - f[mask]) ** 2)))
            corr = float(np.corrcoef(stilde[mask], f[mask])[0, 1]) if np.std(stilde[mask]) > 0 and np.std(f[mask]) > 0 else math.nan
        else:
            rmse = math.nan
            corr = math.nan
        comparison.append(
            {
                "L": L,
                "N": N,
                "nmax": nmax,
                "W": W,
                "gamma": gamma,
                "lambda_shape_rmse": rmse,
                "lambda_shape_corr": corr,
                "n_U": int(np.sum(mask)),
            }
        )
    return weight_rows, qeff_rows, comparison


def analyze_hardcore(rows: list[AggRow], tail_points: int) -> list[dict]:
    output = []
    for (L, N, nmax, W), curve in grouped_by(rows, ("L", "N", "nmax", "W")).items():
        U, _, s_inf = normalized_decay_curve(curve, tail_points)
        dim = bounded_composition_dim(L, N, nmax)
        dhc = math.comb(L, N) if 0 <= N <= L else 0
        bound = math.log(dhc) / math.log(dim) if dhc > 0 and dim > 1 else math.nan
        output.append(
            {
                "L": L,
                "N": N,
                "nmax": nmax,
                "W": W,
                "U_max_used": float(np.max(U)) if len(U) else math.nan,
                "s_infinity_estimate": s_inf,
                "D_full": dim,
                "D_hardcore": dhc,
                "lnDhc_over_lnD": bound,
                "ratio_to_hardcore_bound": s_inf / bound if math.isfinite(bound) and bound != 0 else math.nan,
            }
        )
    return output


def analyze_N_crossings(rows: list[AggRow]) -> tuple[list[dict], list[dict], list[dict]]:
    by_lnw = grouped_by(rows, ("L", "nmax", "U", "W"))
    slopes = []
    for (L, nmax, U, W), group in by_lnw.items():
        Ns = np.array([row.N for row in group], dtype=float)
        s = np.array([row.entropy_mean for row in group], dtype=float)
        if len(set(Ns)) < 2:
            continue
        N0 = float(np.mean(Ns))
        denom = float(np.sum((Ns - N0) ** 2))
        if denom <= 0:
            continue
        B = float(np.sum((Ns - N0) * s) / denom)
        A = float(np.mean(s))
        slopes.append({"L": L, "nmax": nmax, "U": U, "W": W, "N0": N0, "A": A, "B": B, "n_N": len(Ns)})

    crossings = []
    for (L, nmax, W), group in grouped_by_dict(slopes, ("L", "nmax", "W")).items():
        group = sorted(group, key=lambda row: row["U"])
        U = np.array([row["U"] for row in group], dtype=float)
        B = np.array([row["B"] for row in group], dtype=float)
        Ux = descending_crossing(U, B, 0.0)
        if not math.isfinite(Ux):
            Ux = descending_crossing(U, -B, 0.0)
        crossings.append({"L": L, "nmax": nmax, "W": W, "U_cross": Ux, "n_U": len(U)})

    fits = []
    for (L, nmax), group in grouped_by_dict(crossings, ("L", "nmax")).items():
        slope, intercept, r2, n_fit = linear_fit([row["W"] for row in group], [row["U_cross"] for row in group])
        fits.append({"L": L, "nmax": nmax, "c_cross_slope": slope, "b_cross_intercept": intercept, "r2": r2, "n_W": n_fit})
    return slopes, crossings, fits


def analyze_qeff_slope(rows: list[AggRow], qeff_rows: list[dict], chi_U: float) -> list[dict]:
    qeff_lookup = {(row["L"], row["N"], row["nmax"]): row["q_eff2"] for row in qeff_rows}
    slope_rows = []
    for (L, N, nmax, W), curve in grouped_by(rows, ("L", "N", "nmax", "W")).items():
        curve = sorted(curve, key=lambda row: row.U)
        U = np.array([row.U for row in curve], dtype=float)
        s = np.array([row.entropy_mean for row in curve], dtype=float)
        if len(U) < 2:
            continue
        idx = int(np.argmin(np.abs(U - chi_U)))
        if idx == 0:
            deriv = (s[1] - s[0]) / (U[1] - U[0])
        elif idx == len(U) - 1:
            deriv = (s[-1] - s[-2]) / (U[-1] - U[-2])
        else:
            deriv = (s[idx + 1] - s[idx - 1]) / (U[idx + 1] - U[idx - 1])
        slope_rows.append(
            {
                "L": L,
                "N": N,
                "nmax": nmax,
                "W": W,
                "U_for_chi": chi_U,
                "nearest_U": float(U[idx]),
                "chi_U": float(-deriv),
                "q_eff2": qeff_lookup.get((L, N, nmax), math.nan),
            }
        )
    return slope_rows


def analyze_kappa_filter(rows: list[AggRow], gamma: float, grid: int, tail_points: int) -> list[dict]:
    output = []
    weights_by_sector = {(L, N, nmax): channel_weights(L, N, nmax) for L, N, nmax in sorted({(r.L, r.N, r.nmax) for r in rows})}
    for (L, N, nmax, W), curve in grouped_by(rows, ("L", "N", "nmax", "W")).items():
        curve = sorted(curve, key=lambda row: row.U)
        if len(curve) < 2:
            continue
        weights = {q: aq for q, aq in weights_by_sector[(L, N, nmax)].items() if q != 0}
        if not weights:
            continue
        U_ref = curve[0].U
        ref = lambda_site(L, weights, U_ref, W, gamma, grid)
        for row in curve:
            lam = lambda_site(L, weights, row.U, W, gamma, grid)
            ratio = lam / ref if ref > 0 else math.nan
            kappa = -math.log(max(ratio, 1e-300)) if math.isfinite(ratio) else math.nan
            if math.isfinite(kappa):
                z1 = partition_Z(L, N, nmax, kappa)
                z2 = partition_Z(L, N, nmax, 2 * kappa)
                dim = bounded_composition_dim(L, N, nmax)
                s2_model = 2 * math.log(max(z1, 1e-300)) - math.log(max(z2, 1e-300))
                model_norm = s2_model / math.log(dim) if dim > 1 else math.nan
            else:
                model_norm = math.nan
            output.append(
                {
                    "L": L,
                    "N": N,
                    "nmax": nmax,
                    "U": row.U,
                    "W": W,
                    "gamma": gamma,
                    "kappa": kappa,
                    "S2_model_over_lnD": model_norm,
                    "data_entropy": row.entropy_mean,
                    "data_lnPR_over_lnD_available": row.observable == "ln_pr_normalized",
                }
            )
    return output


def plot_entropy_vs_U(rows: list[AggRow], output_dir: Path, plot_W: list[float], plt) -> None:
    for (L, N, nmax), sector_rows in grouped_by(rows, ("L", "N", "nmax")).items():
        by_w = grouped_by(sector_rows, ("W",))
        selected = [W for W in sorted(w[0] for w in by_w) if not plot_W or any(abs(W - wanted) < 1e-9 for wanted in plot_W)]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(3.45, 2.55))
        for i, W in enumerate(selected):
            curve = sorted(by_w[(W,)], key=lambda row: row.U)
            ax.errorbar(
                [row.U for row in curve],
                [row.entropy_mean for row in curve],
                yerr=[row.entropy_stderr if math.isfinite(row.entropy_stderr) else 0.0 for row in curve],
                color=PRB_COLORS[i % len(PRB_COLORS)],
                marker=PRB_MARKERS[i % len(PRB_MARKERS)],
                ms=3.0,
                lw=1.0,
                capsize=1.5,
                label=rf"$W={W:g}$",
            )
        ax.set_xlabel(r"$U$")
        ax.set_ylabel(r"$s$")
        ax.text(0.04, 0.94, label_sector(L, N, nmax), transform=ax.transAxes, va="top")
        ax.legend(loc="upper right", ncol=2)
        savefig(fig, output_dir / "entropy_vs_U", f"entropy_vs_U_L{L}_N{N}_nmax{nmax}")
        plt.close(fig)


def plot_collapse(rows: list[AggRow], output_dir: Path, tail_points: int, plot_W: list[float], plt) -> None:
    for (L, N, nmax), sector_rows in grouped_by(rows, ("L", "N", "nmax")).items():
        by_w = grouped_by(sector_rows, ("W",))
        fig, ax = plt.subplots(figsize=(3.45, 2.55))
        plotted = 0
        for i, (W_key,) in enumerate(sorted(by_w)):
            W = W_key
            if plot_W and not any(abs(W - wanted) < 1e-9 for wanted in plot_W):
                continue
            U, stilde, _ = normalized_decay_curve(by_w[(W,)], tail_points)
            mask = np.isfinite(stilde) & (W != 0)
            if np.sum(mask) < 2:
                continue
            ax.plot(U[mask] / W, stilde[mask], color=PRB_COLORS[i % len(PRB_COLORS)], marker=PRB_MARKERS[i % len(PRB_MARKERS)], ms=3.0, lw=1.0, label=rf"$W={W:g}$")
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            continue
        ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
        ax.set_xlabel(r"$U/W$")
        ax.set_ylabel(r"$\widetilde{s}$")
        ax.text(0.04, 0.94, label_sector(L, N, nmax), transform=ax.transAxes, va="top")
        ax.legend(loc="upper right", ncol=2)
        savefig(fig, output_dir / "collapse_U_over_W", f"collapse_U_over_W_L{L}_N{N}_nmax{nmax}")
        plt.close(fig)


def plot_simple_xy(rows: list[dict], output_dir: Path, filename: str, xkey: str, ykey: str, group_keys: tuple[str, ...], xlabel: str, ylabel: str, plt) -> None:
    groups = grouped_by_dict(rows, group_keys)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    plotted = 0
    for i, (key, group) in enumerate(sorted(groups.items())):
        group = sorted(group, key=lambda row: row[xkey])
        x = [row[xkey] for row in group if math.isfinite(row.get(ykey, math.nan))]
        y = [row[ykey] for row in group if math.isfinite(row.get(ykey, math.nan))]
        if len(x) < 1:
            continue
        label = ", ".join(f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}" for name, value in zip(group_keys, key))
        ax.plot(x, y, color=PRB_COLORS[i % len(PRB_COLORS)], marker=PRB_MARKERS[i % len(PRB_MARKERS)], ms=3.0, lw=1.0, label=label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=6)
    savefig(fig, output_dir, filename)
    plt.close(fig)


def plot_pair_correlation(rows: list[AggRow], output_dir: Path, plt) -> None:
    for (L, nmax), group in grouped_by(rows, ("L", "nmax")).items():
        fig, ax = plt.subplots(figsize=(3.45, 2.55))
        for i, (N,) in enumerate(sorted(grouped_by(group, ("N",)))):
            sub = grouped_by(group, ("N",))[(N,)]
            x = [row.pair_density_mean for row in sub]
            y = [row.entropy_mean for row in sub]
            ax.scatter(x, y, color=PRB_COLORS[i % len(PRB_COLORS)], marker=PRB_MARKERS[i % len(PRB_MARKERS)], s=14, label=rf"$N={N}$")
        ax.set_xlabel(r"$\langle\mathcal{D}\rangle/L$")
        ax.set_ylabel(r"$s$")
        ax.text(0.04, 0.94, rf"$L={L},\,n_{{max}}={nmax}$", transform=ax.transAxes, va="top")
        ax.legend(loc="best", fontsize=6)
        savefig(fig, output_dir / "pair_correlation", f"entropy_vs_pair_L{L}_nmax{nmax}")
        plt.close(fig)


def make_plots(rows: list[AggRow], output_dir: Path, plot_W: list[float], uhalf_rows: list[dict], lambda_rows: list[dict],
               hardcore_rows: list[dict], slopes: list[dict], crossings: list[dict], qeff_rows: list[dict],
               qeff_slope_rows: list[dict], kappa_rows: list[dict], tail_points: int) -> None:
    plt = setup_matplotlib()
    if plt is None:
        print("WARNING: matplotlib unavailable; skipping plots.")
        return
    plot_entropy_vs_U(rows, output_dir, plot_W, plt)
    plot_collapse(rows, output_dir, tail_points, plot_W, plt)
    plot_pair_correlation(rows, output_dir, plt)
    plot_simple_xy(uhalf_rows, output_dir / "Uhalf_vs_W", "Uhalf_vs_W", "W", "Uhalf", ("L", "N", "nmax"), r"$W$", r"$U_{1/2}$", plt)
    plot_simple_xy(hardcore_rows, output_dir / "hardcore_plateau", "hardcore_plateau", "W", "s_infinity_estimate", ("L", "N", "nmax"), r"$W$", r"$s_\infty$", plt)
    plot_simple_xy(crossings, output_dir / "N_crossing", "Ucross_vs_W", "W", "U_cross", ("L", "nmax"), r"$W$", r"$U_\times$", plt)
    plot_simple_xy(qeff_rows, output_dir / "qeff", "qeff_vs_nmax", "nmax", "q_eff2", ("L", "N"), r"$n_{\max}$", r"$q_{\mathrm{eff}}^2$", plt)
    plot_simple_xy(qeff_slope_rows, output_dir / "qeff", "chi_vs_qeff", "q_eff2", "chi_U", ("L", "W"), r"$q_{\mathrm{eff}}^2$", r"$\chi_U$", plt)
    plot_simple_xy(lambda_rows, output_dir / "lambda_comparison", "lambda_shape_corr_vs_W", "W", "lambda_shape_corr", ("L", "N", "nmax"), r"$W$", r"corr$(\widetilde{s},F_\Lambda)$", plt)
    plot_simple_xy(kappa_rows, output_dir / "kappa_filter", "kappa_model_scatter", "S2_model_over_lnD", "data_entropy", ("L", "N", "nmax"), r"$S_2^{model}/\ln D$", r"$s_{data}$", plt)

    for (L, nmax, W), group in grouped_by_dict(slopes, ("L", "nmax", "W")).items():
        fig, ax = plt.subplots(figsize=(3.45, 2.55))
        group = sorted(group, key=lambda row: row["U"])
        ax.axhline(0.0, color="0.4", ls="--", lw=0.8)
        ax.plot([row["U"] for row in group], [row["B"] for row in group], color="#0072B2", marker="o", ms=3.0)
        ax.set_xlabel(r"$U$")
        ax.set_ylabel(r"$B(U,W)$")
        ax.text(0.04, 0.94, rf"$L={L},\,n_{{max}}={nmax},\,W={W:g}$", transform=ax.transAxes, va="top")
        savefig(fig, output_dir / "N_crossing", f"B_vs_U_L{L}_nmax{nmax}_W{W:g}".replace(".", "p"))
        plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_dir, args.observable)
    rows = filter_rows(rows, args)
    if not rows:
        joined = ", ".join(str(path) for path in args.input_dir)
        raise SystemExit(f"No usable observables_*.csv rows found below: {joined}")

    aggregate_rows = aggregate(rows, args.observable)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    monotonic = analyze_monotonicity(aggregate_rows)
    uhalf_rows, uhalf_fits = analyze_uhalf(aggregate_rows, args.tail_points, args.half_level)
    channel_rows, qeff_rows, lambda_rows = analyze_lambda(aggregate_rows, args.gamma, args.lambda_grid, args.tail_points)
    hardcore_rows = analyze_hardcore(aggregate_rows, args.tail_points)
    slopes, crossings, crossing_fits = analyze_N_crossings(aggregate_rows)
    qeff_slope_rows = analyze_qeff_slope(aggregate_rows, qeff_rows, args.chi_U)
    kappa_rows = analyze_kappa_filter(aggregate_rows, args.gamma, args.lambda_grid, args.tail_points)

    write_csv(output_dir / "aggregate_observables.csv", [asdict_row(row) for row in aggregate_rows])
    write_csv(output_dir / "monotonicity_summary.csv", monotonic)
    write_csv(output_dir / "Uhalf_by_W.csv", uhalf_rows)
    write_csv(output_dir / "Uhalf_linear_fits.csv", uhalf_fits)
    write_csv(output_dir / "channel_weights.csv", channel_rows)
    write_csv(output_dir / "qeff_summary.csv", qeff_rows)
    write_csv(output_dir / "lambda_shape_comparison.csv", lambda_rows)
    write_csv(output_dir / "hardcore_plateau.csv", hardcore_rows)
    write_csv(output_dir / "N_crossing_slopes.csv", slopes)
    write_csv(output_dir / "N_crossings.csv", crossings)
    write_csv(output_dir / "N_crossing_linear_fits.csv", crossing_fits)
    write_csv(output_dir / "qeff_slope_correlation.csv", qeff_slope_rows)
    write_csv(output_dir / "kappa_filter_model.csv", kappa_rows)

    if not args.no_plots:
        make_plots(
            aggregate_rows,
            output_dir,
            parse_float_list(args.plot_W_list),
            uhalf_rows,
            lambda_rows,
            hardcore_rows,
            slopes,
            crossings,
            qeff_rows,
            qeff_slope_rows,
            kappa_rows,
            args.tail_points,
        )

    print(f"Read realization rows: {len(rows)}")
    print(f"Aggregated points: {len(aggregate_rows)}")
    print(f"Saved output: {output_dir}")


if __name__ == "__main__":
    main()
