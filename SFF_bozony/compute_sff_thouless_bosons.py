#!/usr/bin/env python3
"""
Compute the spectral form factor and Thouless time for bosonic POLFED spectra.

The expected inputs are energy files written by
run_bose_hubbard_polfed_N_fixed_nmax.jl. Each column is one disorder
realization and each row is an eigenvalue near the POLFED target.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Metadata:
    U: float
    W: float
    L: int
    N: int
    nmax: int
    boundary: str


@dataclass(frozen=True)
class SFFResult:
    path: Path
    metadata: Metadata
    tau: np.ndarray
    sff: np.ndarray
    sff_smoothed: np.ndarray
    goe: np.ndarray
    delta_k: np.ndarray
    tau_th: float
    mean_spacing: float
    t_h: float
    t_th: float
    g: float
    n_realizations_used: int
    n_realizations_total: int
    n_levels: int


def parse_metadata(path: Path) -> Metadata:
    match = re.search(
        r"energies_(?:polfed|full)_U(?P<U>[-+0-9.eE]+)_L(?P<L>\d+)"
        r"_N(?P<N>\d+)_nmax(?P<nmax>\d+)_W(?P<W>[-+0-9.eE]+)"
        r"_boundary(?P<boundary>[A-Za-z0-9_-]+)_nreal",
        path.name,
    )
    if not match:
        raise ValueError(f"Cannot parse bosonic metadata from filename: {path.name}")
    return Metadata(
        U=float(match.group("U")),
        W=float(match.group("W")),
        L=int(match.group("L")),
        N=int(match.group("N")),
        nmax=int(match.group("nmax")),
        boundary=match.group("boundary"),
    )


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    return values or None


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    return values or None


def float_in_list(value: float, wanted: list[float] | None) -> bool:
    if wanted is None:
        return True
    return any(math.isclose(value, item, rel_tol=0.0, abs_tol=1e-10) for item in wanted)


def metadata_matches(metadata: Metadata, args: argparse.Namespace) -> bool:
    return (
        (args.L_list is None or metadata.L in args.L_list)
        and (args.N is None or metadata.N == args.N)
        and (args.nmax is None or metadata.nmax == args.nmax)
        and (args.boundary is None or metadata.boundary == args.boundary)
        and float_in_list(metadata.U, args.U_list)
        and float_in_list(metadata.W, args.W_list)
        and (args.min_U is None or metadata.U >= args.min_U)
        and (args.max_U is None or metadata.U <= args.max_U)
        and (args.min_W is None or metadata.W >= args.min_W)
        and (args.max_W is None or metadata.W <= args.max_W)
    )


def nreal_hint(path: Path) -> int:
    match = re.search(r"_nreal(?P<nreal>\d+)(?:_|\.txt$)", path.name)
    return int(match.group("nreal")) if match else -1


def select_best_parameter_files(paths: list[Path]) -> list[Path]:
    """Keep the richest file for every (L, N, nmax, boundary, U, W) point."""
    best: dict[Metadata, Path] = {}
    for path in paths:
        metadata = parse_metadata(path)
        previous = best.get(metadata)
        if previous is None:
            best[metadata] = path
            continue
        previous_score = (nreal_hint(previous), previous.stat().st_size, previous.stat().st_mtime)
        current_score = (nreal_hint(path), path.stat().st_size, path.stat().st_mtime)
        if current_score > previous_score:
            best[metadata] = path
    return sorted(best.values())


def load_energy_matrix(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#", dtype=float)
    if data.size == 0:
        return np.empty((0, 0), dtype=float)
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data


def central_levels(values: np.ndarray, middle_count: int | None,
                   edge_fraction: float) -> np.ndarray:
    levels = np.asarray(values, dtype=float)
    levels = np.sort(levels[np.isfinite(levels)])
    if middle_count is not None:
        keep = min(middle_count, len(levels))
        start = (len(levels) - keep) // 2
        return levels[start:start + keep]
    drop = int(math.floor(edge_fraction * len(levels)))
    return levels[drop:len(levels) - drop] if drop else levels


def unfold_local(levels: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    """Polynomially unfold one locally sampled spectrum and return raw spacing."""
    levels = np.asarray(levels, dtype=float)
    if len(levels) < degree + 2:
        raise ValueError("Too few levels for polynomial unfolding.")
    mean_spacing = float(np.mean(np.diff(levels)))
    if not np.isfinite(mean_spacing) or mean_spacing <= 0:
        raise ValueError("Non-positive mean level spacing.")

    # Rescaling improves conditioning without changing the fitted staircase.
    centered = (levels - np.mean(levels)) / mean_spacing
    staircase = np.arange(len(levels), dtype=float)
    fit_degree = min(degree, len(levels) - 1)
    coefficients = np.polyfit(centered, staircase, deg=fit_degree)
    unfolded = np.polyval(coefficients, centered)
    unfolded = np.sort(unfolded)
    unfolded /= np.mean(np.diff(unfolded))
    return unfolded, mean_spacing


def gaussian_weights(levels: np.ndarray, sigma_fraction: float) -> np.ndarray:
    center = 0.5 * (levels[0] + levels[-1])
    sigma = sigma_fraction * (levels[-1] - levels[0])
    if sigma <= 0:
        raise ValueError("The unfolded spectrum has zero width.")
    weights = np.exp(-0.5 * ((levels - center) / sigma) ** 2)
    weights /= np.sum(weights)
    return weights


def spectral_form_factor(levels: np.ndarray, tau: np.ndarray,
                         sigma_fraction: float) -> np.ndarray:
    """
    Windowed SFF normalized to a unit plateau.

    For unfolded levels with mean spacing one, tau=t/t_H and the phase is
    exp(-2*pi*i*tau*epsilon_n). The analytical diagonal plateau is sum(w_n^2).
    """
    weights = gaussian_weights(levels, sigma_fraction)
    phase = np.exp(-2j * np.pi * levels[:, None] * tau[None, :])
    trace = np.sum(weights[:, None] * phase, axis=0)
    plateau = np.sum(weights**2)
    return np.abs(trace) ** 2 / plateau


def goe_connected(tau: np.ndarray) -> np.ndarray:
    """Connected GOE spectral form factor with a unit late-time plateau."""
    tau = np.asarray(tau, dtype=float)
    goe = np.zeros_like(tau)
    low = (tau > 0) & (tau <= 1)
    high = tau > 1
    goe[low] = 2 * tau[low] - tau[low] * np.log1p(2 * tau[low])
    goe[high] = 2 - tau[high] * np.log(
        (2 * tau[high] + 1) / (2 * tau[high] - 1)
    )
    return goe


def smooth_log_curve(values: np.ndarray, sigma: float) -> np.ndarray:
    tiny = np.finfo(float).tiny
    log_values = np.log(np.maximum(values, tiny))
    if sigma == 0:
        return np.exp(log_values)
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(log_values, (radius, radius), mode="edge")
    return np.exp(np.convolve(padded, kernel, mode="valid"))


def find_thouless_time(tau: np.ndarray, sff_smoothed: np.ndarray,
                       goe: np.ndarray, epsilon: float,
                       stable_points: int, dip_tau_max: float) -> tuple[float, np.ndarray]:
    """
    Find the first stable entry into the GOE ramp after the correlation hole.

    Delta K = abs(log10(K/K_GOE)); tau_Th is the first point after the SFF
    minimum for which stable_points consecutive values satisfy Delta K < epsilon.
    """
    valid = (tau > 0) & (sff_smoothed > 0) & (goe > 0)
    delta_k = np.full_like(tau, np.nan, dtype=float)
    delta_k[valid] = np.abs(np.log10(sff_smoothed[valid] / goe[valid]))
    if not np.any(valid):
        return float("nan"), delta_k

    dip_search = valid & (tau <= dip_tau_max)
    if not np.any(dip_search):
        dip_search = valid
    dip_index = int(np.nanargmin(np.where(dip_search, sff_smoothed, np.nan)))
    for index in range(dip_index, len(tau) - stable_points + 1):
        window = delta_k[index:index + stable_points]
        if np.all(np.isfinite(window)) and np.all(window < epsilon):
            return float(tau[index]), delta_k
    return float("nan"), delta_k


def analyze_file(path: Path, args: argparse.Namespace) -> SFFResult:
    metadata = parse_metadata(path)
    energy_matrix = load_energy_matrix(path)
    if energy_matrix.size == 0:
        raise ValueError("Empty energy file.")

    tau = np.logspace(args.log10_tau_min, args.log10_tau_max, args.num_tau)
    accumulated = np.zeros_like(tau)
    mean_spacings: list[float] = []
    n_levels: int | None = None
    n_used = 0

    for column in range(energy_matrix.shape[1]):
        levels = central_levels(
            energy_matrix[:, column], args.middle_count, args.edge_fraction
        )
        if len(levels) < args.min_levels:
            continue
        try:
            unfolded, mean_spacing = unfold_local(levels, args.unfold_degree)
            accumulated += spectral_form_factor(
                unfolded, tau, args.window_sigma_fraction
            )
        except ValueError:
            continue
        mean_spacings.append(mean_spacing)
        n_levels = len(levels) if n_levels is None else min(n_levels, len(levels))
        n_used += 1

    if n_used == 0:
        raise ValueError("No usable disorder realizations.")

    sff = accumulated / n_used
    sff_smoothed = smooth_log_curve(sff, args.smoothing_sigma)
    goe = goe_connected(tau)
    tau_th, delta_k = find_thouless_time(
        tau, sff_smoothed, goe, args.epsilon, args.stable_points, args.dip_tau_max
    )
    mean_spacing = float(np.mean(mean_spacings))
    t_h = 2 * np.pi / mean_spacing
    t_th = tau_th * t_h
    g = float(np.log10(t_h / t_th)) if np.isfinite(t_th) and t_th > 0 else float("nan")
    return SFFResult(
        path=path,
        metadata=metadata,
        tau=tau,
        sff=sff,
        sff_smoothed=sff_smoothed,
        goe=goe,
        delta_k=delta_k,
        tau_th=tau_th,
        mean_spacing=mean_spacing,
        t_h=t_h,
        t_th=t_th,
        g=g,
        n_realizations_used=n_used,
        n_realizations_total=energy_matrix.shape[1],
        n_levels=n_levels or 0,
    )


def result_stem(result: SFFResult) -> str:
    m = result.metadata
    stem = (
        f"SFF_U{m.U:g}_W{m.W:g}_L{m.L}_N{m.N}_nmax{m.nmax}"
        f"_boundary{m.boundary}"
    )
    hint = nreal_hint(result.path)
    return f"{stem}_nreal{hint}" if hint >= 0 else stem


def write_curve(result: SFFResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = np.column_stack(
        [result.tau, result.sff, result.sff_smoothed, result.goe, result.delta_k]
    )
    header = "tau=t/t_H SFF SFF_smoothed GOE_connected DeltaK_abs_log10_ratio"
    np.savetxt(output_dir / f"{result_stem(result)}.txt", table, header=header)


def style_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def plot_result(result: SFFResult, output_dir: Path, epsilon: float) -> None:
    import matplotlib.pyplot as plt

    style_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = result_stem(result)
    fig, (ax_sff, ax_delta) = plt.subplots(
        1, 2, figsize=(8.2, 3.35), constrained_layout=True
    )
    ax_sff.loglog(result.tau, result.sff, color="0.75", lw=0.8, label="SFF")
    ax_sff.loglog(result.tau, result.sff_smoothed, color="#0072B2", lw=1.5,
                  label="smoothed SFF")
    ax_sff.loglog(result.tau, result.goe, "k--", lw=1.2, label="GOE")
    if np.isfinite(result.tau_th):
        ax_sff.axvline(result.tau_th, color="#D55E00", ls=":", lw=1.3,
                      label=rf"$\tau_{{Th}}={result.tau_th:.3g}$")
    ax_sff.set_xlabel(r"$\tau=t/t_H$")
    ax_sff.set_ylabel(r"$K(\tau)$")
    ax_sff.legend(fontsize=8)

    ax_delta.semilogx(result.tau, result.delta_k, color="#0072B2", lw=1.25)
    ax_delta.axhline(epsilon, color="#D55E00", ls="--", lw=1.1,
                     label=rf"$\epsilon={epsilon:g}$")
    if np.isfinite(result.tau_th):
        ax_delta.axvline(result.tau_th, color="#D55E00", ls=":", lw=1.3)
    ax_delta.set_xlabel(r"$\tau=t/t_H$")
    ax_delta.set_ylabel(r"$|\log_{10}(K/K_{\rm GOE})|$")
    ax_delta.legend(fontsize=8)

    title = (
        rf"$L={result.metadata.L}$, $U={result.metadata.U:g}$, "
        rf"$W={result.metadata.W:g}$"
    )
    fig.suptitle(title, fontsize=11)
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def write_summary(results: list[SFFResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "thouless_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "U", "W", "L", "N", "nmax", "boundary", "tau_Th",
                "mean_spacing", "t_H", "t_Th", "g",
                "n_realizations_used", "n_realizations_total", "n_levels",
                "file",
            ]
        )
        for result in sorted(
            results,
            key=lambda r: (
                r.metadata.L, r.metadata.N, r.metadata.nmax, r.metadata.boundary,
                r.metadata.U, r.metadata.W, str(r.path),
            ),
        ):
            m = result.metadata
            writer.writerow(
                [
                    m.U, m.W, m.L, m.N, m.nmax, m.boundary, result.tau_th,
                    result.mean_spacing, result.t_h, result.t_th, result.g, result.n_realizations_used,
                    result.n_realizations_total, result.n_levels,
                    result.path.as_posix(),
                ]
            )
    return path


def plot_summary(results: list[SFFResult], output_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    style_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    sectors: dict[tuple[int, int, int, str], list[SFFResult]] = {}
    for result in results:
        m = result.metadata
        sectors.setdefault((m.L, m.N, m.nmax, m.boundary), []).append(result)

    for (L, N, nmax, boundary), sector_rows in sorted(sectors.items()):
        by_u: dict[float, list[SFFResult]] = {}
        for result in sector_rows:
            by_u.setdefault(result.metadata.U, []).append(result)

        fig, (ax_tau, ax_time) = plt.subplots(
            1, 2, figsize=(8.2, 3.35), constrained_layout=True
        )
        available_u = sorted(by_u)
        cmap = plt.get_cmap("viridis")
        if len(available_u) == 1:
            norm = mpl.colors.Normalize(vmin=available_u[0] - 0.5, vmax=available_u[0] + 0.5)
        else:
            norm = mpl.colors.Normalize(vmin=min(available_u), vmax=max(available_u))
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

        for index, U in enumerate(available_u):
            rows = sorted(by_u[U], key=lambda row: row.metadata.W)
            W = np.array([row.metadata.W for row in rows])
            tau_th = np.array([row.tau_th for row in rows])
            t_th = np.array([row.t_th for row in rows])
            kwargs = {
                "color": cmap(norm(U)),
                "marker": markers[index % len(markers)],
                "lw": 1.15,
                "ms": 3.5,
            }
            ax_tau.plot(W, tau_th, **kwargs)
            ax_time.plot(W, t_th, **kwargs)

        ax_tau.set_xlabel(r"$W$")
        ax_tau.set_ylabel(r"$\tau_{Th}=t_{Th}/t_H$")
        ax_time.set_xlabel(r"$W$")
        ax_time.set_ylabel(r"$t_{Th}$")
        ax_tau.set_yscale("log")
        ax_time.set_yscale("log")
        fig.suptitle(
            rf"$L={L}$, $N={N}$, $n_{{max}}={nmax}$, {boundary}",
            fontsize=11,
        )
        scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        colorbar = fig.colorbar(scalar_map, ax=[ax_tau, ax_time], pad=0.015, fraction=0.04)
        colorbar.set_label(r"$U$")

        stem = f"thouless_vs_W_L{L}_N{N}_nmax{nmax}_boundary{boundary}"
        fig.savefig(output_dir / f"{stem}.png", dpi=300)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)


def plot_g_vs_w(results: list[SFFResult], output_dir: Path) -> None:
    """Plot the article-style ergodicity indicator g(W), comparing system sizes."""
    import matplotlib.pyplot as plt

    style_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_u: dict[float, list[SFFResult]] = {}
    for result in results:
        by_u.setdefault(result.metadata.U, []).append(result)

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    colors = plt.get_cmap("tab10").colors
    for U, u_rows in sorted(by_u.items()):
        sectors: dict[tuple[int, int, int, str], list[SFFResult]] = {}
        for result in u_rows:
            m = result.metadata
            sectors.setdefault((m.L, m.N, m.nmax, m.boundary), []).append(result)

        fig, ax = plt.subplots(figsize=(3.65, 2.65), constrained_layout=True)
        for index, ((L, N, nmax, boundary), rows) in enumerate(sorted(sectors.items())):
            rows = sorted(rows, key=lambda row: row.metadata.W)
            W = np.array([row.metadata.W for row in rows], dtype=float)
            g = np.array([row.g for row in rows], dtype=float)
            finite = np.isfinite(g)
            if not np.any(finite):
                continue
            ax.plot(
                W[finite],
                g[finite],
                color=colors[index % len(colors)],
                marker=markers[index % len(markers)],
                lw=1.25,
                ms=3.8,
                label=rf"$L={L}$, $N={N}$, $n_{{max}}={nmax}$, {boundary}",
            )

        ax.axhline(0.0, color="0.45", lw=0.85, ls="--")
        ax.set_xlabel(r"$W$")
        ax.set_ylabel(r"$g=\log_{10}(t_H/t_{\rm Th})$")
        ax.set_title(rf"$U={U:g}$", pad=3)
        ax.legend(
            frameon=True,
            fancybox=False,
            framealpha=0.9,
            facecolor="white",
            edgecolor="0.55",
            loc="best",
        )
        stem = f"g_vs_W_U{U:g}"
        fig.savefig(output_dir / f"{stem}.png", dpi=600)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)


def find_energy_files(input_dir: Path, include_partial: bool) -> list[Path]:
    files = sorted(input_dir.rglob("energies_polfed_*.txt"))
    files += sorted(input_dir.rglob("energies_full_*.txt"))
    if not include_partial:
        files = [path for path in files if "_partial" not in path.name]
    return files


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compute bosonic SFF and extract the Thouless time."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("pbs_bose_polfed_grouped/data"))
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results")
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Analyze every matching file instead of keeping the richest nreal file per point.")
    parser.add_argument("--L-list", type=parse_int_list, default=None,
                        help="Comma-separated chain lengths, for example 6,7,8,9.")
    parser.add_argument("--N", type=int, default=None,
                        help="Total boson number.")
    parser.add_argument("--nmax", type=int, default=None,
                        help="Maximum on-site occupation.")
    parser.add_argument("--boundary", choices=["open", "periodic"], default=None)
    parser.add_argument("--U-list", type=parse_float_list, default=None,
                        help="Comma-separated interaction strengths.")
    parser.add_argument("--W-list", type=parse_float_list, default=None,
                        help="Comma-separated disorder strengths.")
    parser.add_argument("--min-U", type=float, default=None)
    parser.add_argument("--max-U", type=float, default=None)
    parser.add_argument("--min-W", type=float, default=None)
    parser.add_argument("--max-W", type=float, default=None)
    parser.add_argument("--middle-count", type=int, default=None)
    parser.add_argument("--edge-fraction", type=float, default=0.20)
    parser.add_argument("--min-levels", type=int, default=40)
    parser.add_argument("--unfold-degree", type=int, default=3)
    parser.add_argument("--window-sigma-fraction", type=float, default=0.30)
    parser.add_argument("--log10-tau-min", type=float, default=-4.0)
    parser.add_argument("--log10-tau-max", type=float, default=1.0)
    parser.add_argument("--num-tau", type=int, default=1000)
    parser.add_argument("--smoothing-sigma", type=float, default=5.0)
    parser.add_argument("--epsilon", type=float, default=0.08,
                        help="Maximum |log10(K/K_GOE)| used to identify tau_Th.")
    parser.add_argument("--stable-points", type=int, default=4)
    parser.add_argument("--dip-tau-max", type=float, default=1.0,
                        help="Search for the correlation-hole minimum only up to this tau.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-individual-plots", action="store_true",
                        help="Save summary plots but skip one PNG/PDF pair per parameter point.")
    parser.add_argument("--list-files-only", action="store_true",
                        help="Print selected input files and stop before analyzing spectra.")
    args = parser.parse_args()

    if not 0 <= args.edge_fraction < 0.5:
        raise SystemExit("--edge-fraction must satisfy 0 <= value < 0.5")
    if args.middle_count is not None and args.middle_count < args.min_levels:
        raise SystemExit("--middle-count must be at least --min-levels")
    if args.min_levels < 4:
        raise SystemExit("--min-levels must be at least 4")
    if args.unfold_degree < 1:
        raise SystemExit("--unfold-degree must be positive")
    if args.window_sigma_fraction <= 0:
        raise SystemExit("--window-sigma-fraction must be positive")
    if args.num_tau < 10:
        raise SystemExit("--num-tau must be at least 10")
    if args.log10_tau_min >= args.log10_tau_max:
        raise SystemExit("--log10-tau-min must be smaller than --log10-tau-max")
    if args.smoothing_sigma < 0:
        raise SystemExit("--smoothing-sigma must be nonnegative")
    if args.epsilon <= 0:
        raise SystemExit("--epsilon must be positive")
    if args.stable_points < 1:
        raise SystemExit("--stable-points must be positive")
    if args.dip_tau_max <= 0:
        raise SystemExit("--dip-tau-max must be positive")
    if args.min_U is not None and args.max_U is not None and args.min_U > args.max_U:
        raise SystemExit("--min-U cannot exceed --max-U")
    if args.min_W is not None and args.max_W is not None and args.min_W > args.max_W:
        raise SystemExit("--min-W cannot exceed --max-W")

    files = find_energy_files(args.input_dir, args.include_partial)
    if not files:
        raise SystemExit(f"No bosonic energy files found under {args.input_dir}")
    selected_files: list[Path] = []
    for path in files:
        try:
            metadata = parse_metadata(path)
        except ValueError as exc:
            print(f"WARNING: skipping {path}: {exc}")
            continue
        if metadata_matches(metadata, args):
            selected_files.append(path)
    if not selected_files:
        raise SystemExit("No bosonic energy files matched the requested filters.")
    if not args.keep_duplicates:
        before = len(selected_files)
        selected_files = select_best_parameter_files(selected_files)
        dropped = before - len(selected_files)
        if dropped:
            print(f"Dropped duplicate parameter files: {dropped}")
    files = selected_files
    print(f"Selected energy files: {len(files)}")
    if args.list_files_only:
        for path in files:
            print(path)
        return

    results: list[SFFResult] = []
    plots_enabled = not args.skip_plots
    for path in files:
        try:
            result = analyze_file(path, args)
        except Exception as exc:
            print(f"WARNING: skipping {path}: {exc}")
            continue
        results.append(result)
        write_curve(result, args.output_dir / "curves")
        if plots_enabled and not args.skip_individual_plots:
            try:
                plot_result(result, args.output_dir / "plots", args.epsilon)
            except ModuleNotFoundError as exc:
                if exc.name != "matplotlib":
                    raise
                print("WARNING: matplotlib is unavailable; skipping plots.")
                plots_enabled = False
        print(
            f"L={result.metadata.L} U={result.metadata.U:g} W={result.metadata.W:g} "
            f"tau_Th={result.tau_th:.6g} t_H={result.t_h:.6g} "
            f"t_Th={result.t_th:.6g} g={result.g:.6g} "
            f"realizations={result.n_realizations_used}/{result.n_realizations_total}"
        )

    if not results:
        raise SystemExit("No usable bosonic spectra found.")
    summary = write_summary(results, args.output_dir)
    if plots_enabled:
        try:
            plot_summary(results, args.output_dir / "plots")
            plot_g_vs_w(results, args.output_dir / "plots")
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("WARNING: matplotlib is unavailable; skipping summary plot.")
    print(f"Saved summary: {summary}")


if __name__ == "__main__":
    main()
