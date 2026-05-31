#!/usr/bin/env python3
"""
Compute the adjacent gap ratio <r> from POLFED energy files and plot <r>(U,W).

Expected input layout is the one produced by submit_polfed_pbs_grouped_fixed.pl
or submit_bose_polfed_pbs_grouped_nmax.pl:

  pbs_polfed_grouped/data/U_4p0/W_6p0/L_9/Nup_5_Ndown_4/
      energies_polfed_U4_L9_Nup5_Ndown4_W6_nreal400.txt

  pbs_bose_polfed_grouped/data/U_4p0/W_6p0/L_10/N_10_nmax_2/boundary_periodic/
      energies_polfed_U4_L10_N10_nmax2_W6_boundaryperiodic_nreal400.txt

Each energy file is interpreted as a matrix whose columns are disorder
realizations and whose rows are sorted eigenvalues near the requested target.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


POISSON_R = 2.0 * math.log(2.0) - 1.0
GOE_R = 0.5307


@dataclass(frozen=True)
class GapRatioResult:
    path: Path
    U: float
    W: float
    L: int | None
    Nup: int | None
    Ndown: int | None
    N: int | None
    nmax: int | None
    boundary: str | None
    mean_r: float
    stderr_r: float
    n_r_values: int
    n_realizations_used: int
    n_realizations_total: int
    n_levels_min: int
    n_levels_max: int
    zero_gap_fraction: float
    duplicate_levels_removed: int


def parse_float_tag(tag: str) -> float:
    return float(tag.replace("m", "-").replace("p", "."))


def parse_boundary(path: Path) -> str | None:
    text = path.as_posix()
    file_match = re.search(r"_boundary(?P<boundary>open|periodic)_", path.name)
    if file_match:
        return file_match.group("boundary")
    path_match = re.search(r"/boundary_(?P<boundary>open|periodic)/", text)
    return path_match.group("boundary") if path_match else None


def parse_metadata(path: Path) -> tuple[float, float, int | None, int | None, int | None, int | None, int | None, str | None]:
    text = path.as_posix()
    boundary = parse_boundary(path)

    fermion_file_match = re.search(
        r"energies_[^_]+_U(?P<U>[-+0-9.eE]+)_L(?P<L>\d+)_Nup(?P<Nup>\d+)_Ndown(?P<Ndown>\d+)_W(?P<W>[-+0-9.eE]+)_",
        path.name,
    )
    if fermion_file_match:
        return (
            float(fermion_file_match.group("U")),
            float(fermion_file_match.group("W")),
            int(fermion_file_match.group("L")),
            int(fermion_file_match.group("Nup")),
            int(fermion_file_match.group("Ndown")),
            None,
            None,
            boundary,
        )

    boson_file_match = re.search(
        r"energies_[^_]+_U(?P<U>[-+0-9.eE]+)_L(?P<L>\d+)_N(?P<N>\d+)_nmax(?P<nmax>\d+)_W(?P<W>[-+0-9.eE]+)_",
        path.name,
    )
    if boson_file_match:
        return (
            float(boson_file_match.group("U")),
            float(boson_file_match.group("W")),
            int(boson_file_match.group("L")),
            None,
            None,
            int(boson_file_match.group("N")),
            int(boson_file_match.group("nmax")),
            boundary,
        )

    u_match = re.search(r"/U_([^/]+)/", text)
    w_match = re.search(r"/W_([^/]+)/", text)
    l_match = re.search(r"/L_(\d+)/", text)
    fermion_sector_match = re.search(r"/Nup_(\d+)_Ndown_(\d+)/", text)
    boson_sector_match = re.search(r"/N_(\d+)_nmax_(\d+)/", text)

    if not u_match or not w_match:
        raise ValueError(f"Could not parse U/W from path: {path}")

    U = parse_float_tag(u_match.group(1))
    W = parse_float_tag(w_match.group(1))
    L = int(l_match.group(1)) if l_match else None
    Nup = int(fermion_sector_match.group(1)) if fermion_sector_match else None
    Ndown = int(fermion_sector_match.group(2)) if fermion_sector_match else None
    N = int(boson_sector_match.group(1)) if boson_sector_match else None
    nmax = int(boson_sector_match.group(2)) if boson_sector_match else None
    return U, W, L, Nup, Ndown, N, nmax, boundary


def load_energy_matrix(path: Path) -> np.ndarray:
    try:
        data = np.loadtxt(path, comments="#", dtype=float)
    except ValueError:
        return np.empty((0, 0), dtype=float)

    if data.size == 0:
        return np.empty((0, 0), dtype=float)
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data


def deduplicate_sorted_levels(values: np.ndarray, tol: float) -> tuple[np.ndarray, int]:
    if tol <= 0 or len(values) <= 1:
        return values, 0
    keep = np.ones(len(values), dtype=bool)
    last = values[0]
    removed = 0
    for i in range(1, len(values)):
        if abs(values[i] - last) <= tol:
            keep[i] = False
            removed += 1
        else:
            last = values[i]
    return values[keep], removed


def central_window(values: np.ndarray, middle_count: int | None, edge_fraction: float,
                   deduplicate_tol: float) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values.sort()
    values, removed = deduplicate_sorted_levels(values, deduplicate_tol)

    n = len(values)
    if n == 0:
        return values, removed

    if middle_count is not None:
        keep = min(middle_count, n)
        start = (n - keep) // 2
        return values[start : start + keep], removed

    drop = int(math.floor(edge_fraction * n))
    if 2 * drop >= n:
        return values, removed
    return values[drop : n - drop], removed


def adjacent_gap_ratios(energies: np.ndarray, gap_atol: float) -> np.ndarray:
    if len(energies) < 3:
        return np.empty(0, dtype=float)

    gaps = np.diff(energies)
    left = gaps[:-1]
    right = gaps[1:]
    denom = np.maximum(left, right)
    numer = np.minimum(left, right)
    mask = np.isfinite(denom) & np.isfinite(numer) & (denom > gap_atol) & (numer >= -gap_atol)
    if not np.any(mask):
        return np.empty(0, dtype=float)
    r = numer[mask] / denom[mask]
    return r[(r >= -gap_atol) & (r <= 1.0 + gap_atol)]


def zero_gap_fraction(energies: np.ndarray, gap_atol: float) -> float:
    if len(energies) < 2:
        return float("nan")
    gaps = np.diff(energies)
    finite = gaps[np.isfinite(gaps)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(np.abs(finite) <= gap_atol))


def analyze_file(path: Path, middle_count: int | None, edge_fraction: float,
                 min_levels: int, gap_atol: float, deduplicate_tol: float) -> GapRatioResult | None:
    U, W, L, Nup, Ndown, N, nmax, boundary = parse_metadata(path)
    E = load_energy_matrix(path)
    if E.size == 0:
        return None

    all_r: list[np.ndarray] = []
    n_used = 0
    level_counts: list[int] = []
    zero_gap_fractions: list[float] = []
    duplicate_levels_removed = 0

    for col in range(E.shape[1]):
        levels, removed = central_window(E[:, col], middle_count, edge_fraction, deduplicate_tol)
        duplicate_levels_removed += removed
        level_counts.append(len(levels))
        if len(levels) < min_levels:
            continue
        zero_gap_fractions.append(zero_gap_fraction(levels, gap_atol))
        r = adjacent_gap_ratios(levels, gap_atol)
        if len(r) == 0:
            continue
        all_r.append(r)
        n_used += 1

    if not all_r:
        mean_r = float("nan")
        stderr_r = float("nan")
        n_r_values = 0
    else:
        joined = np.concatenate(all_r)
        mean_r = float(np.mean(joined))
        stderr_r = float(np.std(joined, ddof=1) / math.sqrt(len(joined))) if len(joined) > 1 else float("nan")
        n_r_values = int(len(joined))

    return GapRatioResult(
        path=path,
        U=U,
        W=W,
        L=L,
        Nup=Nup,
        Ndown=Ndown,
        N=N,
        nmax=nmax,
        boundary=boundary,
        mean_r=mean_r,
        stderr_r=stderr_r,
        n_r_values=n_r_values,
        n_realizations_used=n_used,
        n_realizations_total=E.shape[1],
        n_levels_min=min(level_counts) if level_counts else 0,
        n_levels_max=max(level_counts) if level_counts else 0,
        zero_gap_fraction=float(np.nanmean(zero_gap_fractions)) if zero_gap_fractions else float("nan"),
        duplicate_levels_removed=duplicate_levels_removed,
    )


def find_energy_files(input_dir: Path, include_partial: bool) -> list[Path]:
    files = sorted(input_dir.rglob("energies_polfed_*.txt"))
    files += sorted(input_dir.rglob("energies_full_*.txt"))
    if not include_partial:
        files = [p for p in files if "_partial" not in p.name]
    return files


def write_csv(results: list[GapRatioResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "U",
            "W",
            "mean_r",
            "stderr_r",
            "n_r_values",
            "n_realizations_used",
            "n_realizations_total",
            "n_levels_min",
            "n_levels_max",
            "zero_gap_fraction",
            "duplicate_levels_removed",
            "L",
            "Nup",
            "Ndown",
            "N",
            "nmax",
            "boundary",
            "file",
        ])
        for r in sorted(results, key=lambda x: (x.W, x.U, str(x.path))):
            writer.writerow([
                r.U,
                r.W,
                r.mean_r,
                r.stderr_r,
                r.n_r_values,
                r.n_realizations_used,
                r.n_realizations_total,
                r.n_levels_min,
                r.n_levels_max,
                r.zero_gap_fraction,
                r.duplicate_levels_removed,
                r.L if r.L is not None else "",
                r.Nup if r.Nup is not None else "",
                r.Ndown if r.Ndown is not None else "",
                r.N if r.N is not None else "",
                r.nmax if r.nmax is not None else "",
                r.boundary if r.boundary is not None else "",
                r.path.as_posix(),
            ])


def apply_filters(results: list[GapRatioResult], L: int | None, Nup: int | None,
                  Ndown: int | None, N: int | None, nmax: int | None,
                  boundary: str | None) -> list[GapRatioResult]:
    filtered = results
    if L is not None:
        filtered = [r for r in filtered if r.L == L]
    if Nup is not None:
        filtered = [r for r in filtered if r.Nup == Nup]
    if Ndown is not None:
        filtered = [r for r in filtered if r.Ndown == Ndown]
    if N is not None:
        filtered = [r for r in filtered if r.N == N]
    if nmax is not None:
        filtered = [r for r in filtered if r.nmax == nmax]
    if boundary is not None:
        filtered = [r for r in filtered if r.boundary == boundary]
    return filtered


def keep_best_duplicate(results: list[GapRatioResult]) -> list[GapRatioResult]:
    best: dict[tuple[float, float, int | None, int | None, int | None, int | None, int | None, str | None], GapRatioResult] = {}
    for r in results:
        key = (r.U, r.W, r.L, r.Nup, r.Ndown, r.N, r.nmax, r.boundary)
        old = best.get(key)
        if old is None:
            best[key] = r
            continue
        old_score = (old.n_realizations_used, old.n_r_values, old.path.stat().st_mtime)
        new_score = (r.n_realizations_used, r.n_r_values, r.path.stat().st_mtime)
        if new_score > old_score:
            best[key] = r
    return list(best.values())


def pivot_results(results: list[GapRatioResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U_values = np.array(sorted({r.U for r in results}), dtype=float)
    W_values = np.array(sorted({r.W for r in results}), dtype=float)
    Z = np.full((len(W_values), len(U_values)), np.nan, dtype=float)
    for r in results:
        wi = int(np.where(W_values == r.W)[0][0])
        ui = int(np.where(U_values == r.U)[0][0])
        Z[wi, ui] = r.mean_r
    return U_values, W_values, Z


def plot_heatmap(results: list[GapRatioResult], output_png: Path, output_pdf: Path | None,
                 title: str, vmin: float | None, vmax: float | None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    U_values, W_values, Z = pivot_results(results)

    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e8e8e8")

    if len(U_values) > 1:
        du = np.min(np.diff(U_values))
    else:
        du = 1.0
    if len(W_values) > 1:
        dw = np.min(np.diff(W_values))
    else:
        dw = 1.0

    extent = [
        U_values[0] - 0.5 * du,
        U_values[-1] + 0.5 * du,
        W_values[0] - 0.5 * dw,
        W_values[-1] + 0.5 * dw,
    ]
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\langle r\rangle$")
    cbar.ax.axhline(POISSON_R, color="white", lw=1.2, alpha=0.9)
    cbar.ax.axhline(GOE_R, color="black", lw=1.2, alpha=0.75)

    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$W$")
    ax.set_title(title)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    if len(U_values) <= 25:
        ax.set_xticks(U_values)
    if len(W_values) <= 20:
        ax.set_yticks(W_values)

    ax.text(
        0.02,
        0.98,
        rf"Poisson $\approx {POISSON_R:.3f}$" + "\n" + rf"GOE $\approx {GOE_R:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.85),
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    if output_pdf is not None:
        fig.savefig(output_pdf)
    plt.close(fig)


def parse_curve_u_list(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    if value.strip().lower() == "all":
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def choose_default_curve_us(results: list[GapRatioResult]) -> list[float]:
    available = sorted({r.U for r in results})
    preferred = [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 50.0]
    chosen = [u for u in preferred if u in available]
    if chosen:
        return chosen
    if len(available) <= 7:
        return available
    idx = np.linspace(0, len(available) - 1, 7).round().astype(int)
    return [available[i] for i in sorted(set(idx))]


def plot_curves(results: list[GapRatioResult], output_png: Path, output_pdf: Path | None,
                title: str, curve_us: list[float] | None) -> None:
    import matplotlib.pyplot as plt

    available_by_u: dict[float, list[GapRatioResult]] = {}
    for r in results:
        if np.isfinite(r.mean_r):
            available_by_u.setdefault(r.U, []).append(r)

    if curve_us == []:
        curve_us = sorted(available_by_u)
    elif curve_us is None:
        curve_us = choose_default_curve_us(results)

    curve_us = [u for u in curve_us if u in available_by_u]
    if not curve_us:
        raise SystemExit("No requested U values are available for the curve plot.")

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)

    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(curve_us)))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]

    for i, U in enumerate(curve_us):
        rows = sorted(available_by_u[U], key=lambda r: r.W)
        W = np.array([r.W for r in rows], dtype=float)
        mean_r = np.array([r.mean_r for r in rows], dtype=float)
        err = np.array([r.stderr_r for r in rows], dtype=float)
        label = f"U = {U:g}"
        marker = markers[i % len(markers)]

        if np.all(np.isfinite(err)) and np.any(err > 0):
            ax.errorbar(W, mean_r, yerr=err, color=colors[i], marker=marker,
                        lw=1.8, ms=5.0, capsize=2.5, label=label)
        else:
            ax.plot(W, mean_r, color=colors[i], marker=marker,
                    lw=1.8, ms=5.0, label=label)

    ax.axhline(GOE_R, color="0.35", lw=1.2, ls="--", label="GOE")
    ax.axhline(POISSON_R, color="0.35", lw=1.2, ls=":", label="Poisson")

    ax.set_xlabel(r"$W$")
    ax.set_ylabel(r"$\langle r\rangle$")
    ax.set_title(title)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.set_xlim(left=min(r.W for r in results), right=max(r.W for r in results))
    finite_y = [r.mean_r for r in results if np.isfinite(r.mean_r)]
    if finite_y:
        ymin = min(min(finite_y), POISSON_R) - 0.015
        ymax = max(max(finite_y), GOE_R) + 0.015
        ax.set_ylim(ymin, ymax)

    ax.legend(frameon=True, fancybox=False, edgecolor="0.35", fontsize=9,
              ncol=2, loc="best")
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    if output_pdf is not None:
        fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute adjacent gap ratio <r> and plot a U-W heatmap."
    )
    parser.add_argument("--input-dir", default="pbs_polfed_grouped/data", type=Path)
    parser.add_argument("--output-dir", default="gap_ratio_results", type=Path)
    parser.add_argument("--middle-count", type=int, default=None,
                        help="Use exactly this many central levels from each realization.")
    parser.add_argument("--edge-fraction", type=float, default=0.20,
                        help="If --middle-count is not set, drop this fraction from each edge.")
    parser.add_argument("--min-levels", type=int, default=20,
                        help="Skip a realization if fewer central levels remain.")
    parser.add_argument("--gap-atol", type=float, default=1e-12)
    parser.add_argument("--deduplicate-tol", type=float, default=0.0,
                        help="Collapse adjacent levels closer than this tolerance before computing gaps.")
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--L", type=int, default=None, help="Analyze only this chain length.")
    parser.add_argument("--Nup", type=int, default=None, help="Analyze only this Nup sector.")
    parser.add_argument("--Ndown", type=int, default=None, help="Analyze only this Ndown sector.")
    parser.add_argument("--N", type=int, default=None, help="Analyze only this bosonic total-N sector.")
    parser.add_argument("--nmax", type=int, default=None, help="Analyze only this bosonic nmax sector.")
    parser.add_argument("--boundary", choices=["open", "periodic"], default=None,
                        help="Analyze only this boundary-condition sector.")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Keep duplicate U,W files instead of choosing the one with most data.")
    parser.add_argument("--plot", choices=["curves", "heatmap", "both"], default="curves",
                        help="Which plot type to generate.")
    parser.add_argument("--curve-U-list", default=None,
                        help="Comma-separated U values for the <r>(W) curve plot.")
    parser.add_argument("--title", default=r"Average adjacent gap ratio $\langle r\rangle$")
    parser.add_argument("--vmin", type=float, default=POISSON_R)
    parser.add_argument("--vmax", type=float, default=GOE_R)
    args = parser.parse_args()

    if args.edge_fraction < 0 or args.edge_fraction >= 0.5:
        raise SystemExit("--edge-fraction must satisfy 0 <= edge_fraction < 0.5")
    if args.middle_count is not None and args.middle_count < 3:
        raise SystemExit("--middle-count must be at least 3")

    files = find_energy_files(args.input_dir, args.include_partial)
    if not files:
        raise SystemExit(f"No energies_polfed_*.txt or energies_full_*.txt files found in {args.input_dir}")

    results: list[GapRatioResult] = []
    for path in files:
        try:
            result = analyze_file(path, args.middle_count, args.edge_fraction,
                                  args.min_levels, args.gap_atol, args.deduplicate_tol)
        except Exception as exc:
            print(f"WARNING: skipping {path}: {exc}")
            continue
        if result is not None:
            results.append(result)

    if not results:
        raise SystemExit("No usable energy files found.")

    results = apply_filters(results, args.L, args.Nup, args.Ndown, args.N, args.nmax, args.boundary)
    if not results:
        raise SystemExit("No files matched the requested L/Nup/Ndown/N/nmax/boundary filters.")
    if not args.keep_duplicates:
        before = len(results)
        results = keep_best_duplicate(results)
        dropped = before - len(results)
        if dropped:
            print(f"Dropped duplicate parameter files: {dropped}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "gap_ratio_summary.csv"

    write_csv(results, csv_path)
    if args.plot in ("heatmap", "both"):
        png_path = args.output_dir / "gap_ratio_heatmap.png"
        pdf_path = args.output_dir / "gap_ratio_heatmap.pdf"
        plot_heatmap(results, png_path, pdf_path, args.title, args.vmin, args.vmax)
        print(f"Saved heatmap: {png_path}")
        print(f"Saved heatmap: {pdf_path}")
    if args.plot in ("curves", "both"):
        curve_png_path = args.output_dir / "gap_ratio_curves.png"
        curve_pdf_path = args.output_dir / "gap_ratio_curves.pdf"
        curve_us = parse_curve_u_list(args.curve_U_list)
        plot_curves(results, curve_png_path, curve_pdf_path, args.title, curve_us)
        print(f"Saved curves: {curve_png_path}")
        print(f"Saved curves: {curve_pdf_path}")

    n_files = len(results)
    n_real_used = sum(r.n_realizations_used for r in results)
    n_real_total = sum(r.n_realizations_total for r in results)
    print(f"Analyzed files: {n_files}")
    print(f"Used realizations: {n_real_used} / {n_real_total}")
    print(f"Saved table: {csv_path}")


if __name__ == "__main__":
    main()
