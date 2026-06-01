#!/usr/bin/env python3
"""
Plot the finite-size Thouless-time scaling indicators used in Fig. 7.

The script reads existing thouless_summary.csv files. It does not recompute
spectral form factors or diagonalize the Hamiltonian.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ThoulessPoint:
    U: float
    W: float
    L: int
    N: int
    nmax: int
    boundary: str
    t_th: float
    n_realizations_used: int
    source: Path


@dataclass(frozen=True)
class ScalingPoint:
    U: float
    W: float
    L1: int
    L2: int
    N_L1: int
    N_L2: int
    nmax: int
    boundary: str
    filling: float
    L_average: float
    z: float
    xi_th: float
    t_th_L1: float
    t_th_L2: float


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


def safe_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {key!r} column.") from exc


def safe_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Invalid {key!r} column.") from exc


def safe_str(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing or invalid {key!r} column.")
    return value


def discover_summaries(values: list[Path]) -> list[Path]:
    files: list[Path] = []
    for value in values:
        if value.is_dir():
            files.extend(sorted(value.rglob("thouless_summary.csv")))
        elif value.is_file():
            files.append(value)
        else:
            raise FileNotFoundError(f"Summary path does not exist: {value}")
    unique = sorted({path.resolve() for path in files})
    if not unique:
        raise FileNotFoundError("No thouless_summary.csv files found.")
    return unique


def read_summaries(paths: list[Path]) -> list[ThoulessPoint]:
    points: list[ThoulessPoint] = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                try:
                    points.append(
                        ThoulessPoint(
                            U=safe_float(row, "U"),
                            W=safe_float(row, "W"),
                            L=safe_int(row, "L"),
                            N=safe_int(row, "N"),
                            nmax=safe_int(row, "nmax"),
                            boundary=safe_str(row, "boundary"),
                            t_th=safe_float(row, "t_Th"),
                            n_realizations_used=safe_int(row, "n_realizations_used"),
                            source=path,
                        )
                    )
                except ValueError as exc:
                    print(f"WARNING: skipping {path}:{row_number}: {exc}")
    return points


def point_matches(point: ThoulessPoint, args: argparse.Namespace) -> bool:
    return (
        (args.L_list is None or point.L in args.L_list)
        and (args.nmax_list is None or point.nmax in args.nmax_list)
        and (args.boundary is None or point.boundary == args.boundary)
        and float_in_list(point.U, args.U_list)
        and float_in_list(point.W, args.W_list)
        and (args.min_U is None or point.U >= args.min_U)
        and (args.max_U is None or point.U <= args.max_U)
        and (args.min_W is None or point.W >= args.min_W)
        and (args.max_W is None or point.W <= args.max_W)
    )


def choose_best_points(points: list[ThoulessPoint]) -> list[ThoulessPoint]:
    """Keep the richest row for every bosonic parameter point."""
    best: dict[tuple[float, float, int, int, int, str], ThoulessPoint] = {}
    for point in points:
        key = (point.U, point.W, point.L, point.N, point.nmax, point.boundary)
        previous = best.get(key)
        if previous is None or point.n_realizations_used > previous.n_realizations_used:
            best[key] = point
    return sorted(best.values(), key=lambda point: (point.U, point.W, point.L))


def calculate_scaling(points: list[ThoulessPoint]) -> list[ScalingPoint]:
    by_sector: dict[tuple[float, float, int, str, Fraction], dict[int, ThoulessPoint]] = {}
    for point in points:
        if np.isfinite(point.t_th) and point.t_th > 0:
            filling = Fraction(point.N, point.L)
            by_sector.setdefault(
                (point.U, point.W, point.nmax, point.boundary, filling), {}
            )[point.L] = point

    scaling: list[ScalingPoint] = []
    for (U, W, nmax, boundary, filling), by_L in sorted(by_sector.items()):
        for L1, L2 in itertools.combinations(sorted(by_L), 2):
            t_th_L1 = by_L[L1].t_th
            t_th_L2 = by_L[L2].t_th
            log_ratio = math.log(t_th_L2 / t_th_L1)
            z = log_ratio / math.log(L2 / L1)
            xi_th = (L2 - L1) / log_ratio if not math.isclose(log_ratio, 0.0) else math.nan
            scaling.append(
                ScalingPoint(
                    U=U,
                    W=W,
                    L1=L1,
                    L2=L2,
                    N_L1=by_L[L1].N,
                    N_L2=by_L[L2].N,
                    nmax=nmax,
                    boundary=boundary,
                    filling=float(filling),
                    L_average=0.5 * (L1 + L2),
                    z=z,
                    xi_th=xi_th,
                    t_th_L1=t_th_L1,
                    t_th_L2=t_th_L2,
                )
            )
    return scaling


def write_scaling_csv(points: list[ScalingPoint], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "figure7_scaling_pairs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "U", "W", "L1", "L2", "N_L1", "N_L2", "nmax", "boundary",
                "filling", "L_average", "z", "xi_Th", "t_Th_L1", "t_Th_L2",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.U,
                    point.W,
                    point.L1,
                    point.L2,
                    point.N_L1,
                    point.N_L2,
                    point.nmax,
                    point.boundary,
                    point.filling,
                    point.L_average,
                    point.z,
                    point.xi_th,
                    point.t_th_L1,
                    point.t_th_L2,
                ]
            )
    return path


def style_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.linewidth": 1.15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": True,
            "legend.fancybox": False,
        }
    )


def safe_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def plot_for_sector(points: list[ScalingPoint], U: float, nmax: int,
                    boundary: str, filling: float, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        point for point in points
        if math.isclose(point.U, U, abs_tol=1e-10)
        and point.nmax == nmax
        and point.boundary == boundary
        and math.isclose(point.filling, filling, abs_tol=1e-10)
    ]
    pairs = sorted({(point.L1, point.L2) for point in selected})
    if not pairs:
        return

    style_matplotlib()
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.90, len(pairs)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    fig, (ax_z, ax_xi) = plt.subplots(2, 1, figsize=(6.0, 6.2), sharex=True)

    for index, pair in enumerate(pairs):
        pair_points = sorted(
            (point for point in selected if (point.L1, point.L2) == pair),
            key=lambda point: point.W,
        )
        W = np.array([point.W for point in pair_points])
        z = np.array([point.z for point in pair_points])
        xi = np.array([point.xi_th for point in pair_points])
        L_average = 0.5 * (pair[0] + pair[1])
        label = rf"$L_1={pair[0]},\,L_2={pair[1]}$  ($\bar{{L}}={L_average:g}$)"
        style = {
            "color": colors[index],
            "marker": markers[index % len(markers)],
            "ms": 4.2,
            "lw": 1.25,
        }
        finite_z = np.isfinite(z)
        finite_xi = np.isfinite(xi) & (xi > 0)
        ax_z.plot(W[finite_z], z[finite_z], label=label, **style)
        ax_xi.plot(W[finite_xi], xi[finite_xi], **style)

    ax_z.set_ylabel(r"$z(W)$")
    ax_xi.set_ylabel(r"$\xi_{\mathrm{Th}}(W)$")
    ax_xi.set_xlabel(r"$W$")
    ax_z.set_title(
        rf"Bosons, $U={U:g}$, $n_{{max}}={nmax}$, "
        rf"$N/L={filling:g}$, {boundary}"
    )
    ax_z.legend(loc="best", fontsize=8.0, edgecolor="0.55")
    for axis in (ax_z, ax_xi):
        axis.grid(False)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"figure7_thouless_scaling_U{safe_label(U)}_nmax{nmax}"
        f"_boundary{boundary}_filling{safe_label(filling)}"
    )
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot the Fig. 7 finite-size scaling indicators z(W) and xi_Th(W) "
            "from existing bosonic Thouless-time summaries."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        action="append",
        required=True,
        help=(
            "Existing thouless_summary.csv file or directory containing summaries. "
            "Repeat this option to combine multiple result folders."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "figure7_scaling")
    parser.add_argument("--L-list", type=parse_int_list, default=None)
    parser.add_argument("--nmax-list", type=parse_int_list, default=None)
    parser.add_argument("--boundary", choices=["open", "periodic"], default=None)
    parser.add_argument("--U-list", type=parse_float_list, default=None)
    parser.add_argument("--W-list", type=parse_float_list, default=None)
    parser.add_argument("--min-U", type=float, default=None)
    parser.add_argument("--max-U", type=float, default=None)
    parser.add_argument("--min-W", type=float, default=None)
    parser.add_argument("--max-W", type=float, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if args.min_U is not None and args.max_U is not None and args.min_U > args.max_U:
        raise SystemExit("--min-U cannot exceed --max-U")
    if args.min_W is not None and args.max_W is not None and args.min_W > args.max_W:
        raise SystemExit("--min-W cannot exceed --max-W")

    summaries = discover_summaries(args.summary)
    print(f"Summary files: {len(summaries)}")
    for path in summaries:
        print(f"  {path}")

    points = [point for point in read_summaries(summaries) if point_matches(point, args)]
    points = choose_best_points(points)
    if not points:
        raise SystemExit("No finite-size Thouless-time rows matched the requested filters.")
    print(f"Selected Thouless-time rows: {len(points)}")

    scaling = calculate_scaling(points)
    if not scaling:
        raise SystemExit(
            "No L pairs could be formed. Each plotted (U, W) point needs finite t_Th "
            "for at least two different system sizes."
        )
    table = write_scaling_csv(scaling, args.output_dir)
    print(f"Saved table: {table}")
    print(f"Scaling rows: {len(scaling)}")

    if not args.skip_plots:
        try:
            sectors = sorted(
                {(point.U, point.nmax, point.boundary, point.filling) for point in scaling}
            )
            for U, nmax, boundary, filling in sectors:
                plot_for_sector(
                    scaling, U, nmax, boundary, filling, args.output_dir / "plots"
                )
                print(
                    f"Saved Figure 7 plot for U={U:g}, nmax={nmax}, "
                    f"boundary={boundary}, filling={filling:g}"
                )
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("WARNING: matplotlib is unavailable; skipping plots.")


if __name__ == "__main__":
    main()
