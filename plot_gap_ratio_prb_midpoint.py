#!/usr/bin/env python3
"""
Make PRB-style <r>(W) curves and extract a midpoint crossing W*(U).

Input is gap_ratio_summary.csv produced by compute_gap_ratio_map.py.
The midpoint is, by default, the crossing of

    r_half = (r_GOE + r_Poisson) / 2.

This is a compact diagnostic for the horizontal displacement of the level
statistics crossover as U changes.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


POISSON_R = 2.0 * math.log(2.0) - 1.0
GOE_R = 0.5307
HALF_R = 0.5 * (POISSON_R + GOE_R)


@dataclass(frozen=True)
class Row:
    U: float
    W: float
    mean_r: float
    stderr_r: float
    n_realizations_used: int
    n_realizations_total: int


def read_summary(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            try:
                mean_r = float(rec["mean_r"])
            except ValueError:
                continue
            if not np.isfinite(mean_r):
                continue
            rows.append(
                Row(
                    U=float(rec["U"]),
                    W=float(rec["W"]),
                    mean_r=mean_r,
                    stderr_r=float(rec["stderr_r"]) if rec["stderr_r"] else float("nan"),
                    n_realizations_used=int(float(rec["n_realizations_used"])),
                    n_realizations_total=int(float(rec["n_realizations_total"])),
                )
            )
    return rows


def group_by_u(rows: list[Row]) -> dict[float, list[Row]]:
    grouped: dict[float, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.U, []).append(row)
    return {u: sorted(v, key=lambda x: x.W) for u, v in sorted(grouped.items())}


def parse_u_filter(value: str | None, available: list[float]) -> list[float]:
    if value is None or value.strip().lower() == "all":
        return available
    wanted = [float(x.strip()) for x in value.split(",") if x.strip()]
    available_set = set(available)
    return [u for u in wanted if u in available_set]


def first_descending_crossing(W: np.ndarray, R: np.ndarray, threshold: float) -> float:
    order = np.argsort(W)
    W = W[order]
    R = R[order]

    for i in range(len(W) - 1):
        r0, r1 = R[i], R[i + 1]
        if not (np.isfinite(r0) and np.isfinite(r1)):
            continue
        if (r0 - threshold) == 0:
            return float(W[i])
        if (r0 - threshold) * (r1 - threshold) <= 0:
            if r1 == r0:
                return float(0.5 * (W[i] + W[i + 1]))
            t = (threshold - r0) / (r1 - r0)
            return float(W[i] + t * (W[i + 1] - W[i]))
    return float("nan")


def adaptive_half_threshold(R: np.ndarray, low_points: int, high_points: int) -> float:
    finite = R[np.isfinite(R)]
    if len(finite) == 0:
        return float("nan")
    low_n = min(low_points, len(finite))
    high_n = min(high_points, len(finite))
    high_plateau = float(np.mean(finite[:low_n]))
    low_plateau = float(np.mean(finite[-high_n:]))
    return 0.5 * (high_plateau + low_plateau)


def style_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_curves(grouped: dict[float, list[Row]], selected_u: list[float], output: Path,
                panel_label: str, threshold: float, curve_labeling: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    style_matplotlib()
    fig, ax = plt.subplots(figsize=(3.65, 2.65), constrained_layout=True)

    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=min(selected_u), vmax=max(selected_u))
    colors = [cmap(norm(U)) for U in selected_u]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "p"]
    curve_handles = []

    for i, U in enumerate(selected_u):
        rows = grouped[U]
        W = np.array([r.W for r in rows], dtype=float)
        R = np.array([r.mean_r for r in rows], dtype=float)
        E = np.array([r.stderr_r for r in rows], dtype=float)
        marker = markers[i % len(markers)]
        label = rf"$U={U:g}$" if curve_labeling == "legend" else None
        (handle,) = ax.plot(
            W,
            R,
            color=colors[i],
            marker=marker,
            ms=3.1,
            lw=1.05,
            label=label,
            alpha=0.95,
        )
        curve_handles.append(handle)

    goe_line = ax.axhline(GOE_R, color="0.35", lw=0.95, ls="--", label="GOE")
    poisson_line = ax.axhline(POISSON_R, color="0.35", lw=1.05, ls=":", label="Poisson")
    half_line = ax.axhline(threshold, color="0.55", lw=0.75, ls="-.", label=r"$r_{1/2}$")
    ax.set_xlabel(r"$W$")
    ax.set_ylabel(r"$\langle r\rangle$")
    ax.text(0.03, 0.95, panel_label, transform=ax.transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold")
    ref_legend = ax.legend(
        handles=[goe_line, poisson_line, half_line],
        frameon=True,
        fancybox=False,
        framealpha=0.88,
        facecolor="white",
        edgecolor="0.55",
        loc="upper right",
        handlelength=1.9,
        borderaxespad=0.25,
    )
    ax.add_artist(ref_legend)

    if curve_labeling == "legend":
        ax.legend(frameon=False, ncol=2, loc="lower left", handlelength=1.7, columnspacing=0.9)
    elif curve_labeling == "colorbar":
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.055)
        cbar.set_label(r"$U$")
        preferred_ticks = [1, 5, 10, 15, 20, 30, 40, 50]
        ticks = [u for u in preferred_ticks if min(selected_u) <= u <= max(selected_u)]
        if not ticks:
            ticks = np.linspace(min(selected_u), max(selected_u), 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{u:g}" for u in ticks])

    ax.set_xlim(left=min(min(r.W for r in rows) for rows in grouped.values()))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=600)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_midpoints(midpoints: list[tuple[float, float, float]], output: Path,
                   panel_label: str) -> None:
    import matplotlib.pyplot as plt

    style_matplotlib()
    fig, ax = plt.subplots(figsize=(3.55, 2.45), constrained_layout=True)

    data = np.array(midpoints, dtype=float)
    finite = data[np.isfinite(data[:, 1])]
    ax.plot(finite[:, 0], finite[:, 1], "o-", color="#1f77b4", ms=4.0, lw=1.25)
    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$W^\ast$")
    ax.text(0.03, 0.95, panel_label, transform=ax.transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold")
    ax.set_title(r"$\langle r\rangle = r_{1/2}$ crossing", pad=3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=600)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def write_midpoints_csv(midpoints: list[tuple[float, float, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["U", "W_star", "threshold"])
        for row in midpoints:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("gap_ratio_prb"))
    parser.add_argument("--curve-U-list", default="all")
    parser.add_argument("--midpoint-U-list", default="all",
                        help="U values used for W*(U); defaults to every available U.")
    parser.add_argument("--min-W", type=float, default=None,
                        help="Discard points below this W, for example --min-W 0.001 to omit W=0.")
    parser.add_argument("--max-W", type=float, default=None,
                        help="Discard points above this W.")
    parser.add_argument("--curve-labeling", choices=["colorbar", "legend", "none"],
                        default="colorbar",
                        help="How to identify U curves in the PRB-style curve plot.")
    parser.add_argument("--threshold", default="goe-poisson-half",
                        help="goe-poisson-half, adaptive-half, or numeric value.")
    parser.add_argument("--adaptive-low-points", type=int, default=2)
    parser.add_argument("--adaptive-high-points", type=int, default=3)
    args = parser.parse_args()

    rows = read_summary(args.summary)
    if args.min_W is not None:
        rows = [row for row in rows if row.W >= args.min_W]
    if args.max_W is not None:
        rows = [row for row in rows if row.W <= args.max_W]
    if not rows:
        raise SystemExit(f"No usable rows in {args.summary}")

    grouped = group_by_u(rows)
    available_u = sorted(grouped)
    curve_u = parse_u_filter(args.curve_U_list, available_u)
    midpoint_u = parse_u_filter(args.midpoint_U_list, available_u)
    if not curve_u:
        raise SystemExit("No selected U values are present in the summary.")
    if not midpoint_u:
        raise SystemExit("No midpoint U values are present in the summary.")

    threshold_for_curves = HALF_R
    if args.threshold not in ("goe-poisson-half", "adaptive-half"):
        threshold_for_curves = float(args.threshold)

    midpoints: list[tuple[float, float, float]] = []
    for U in midpoint_u:
        group = grouped[U]
        W = np.array([r.W for r in group], dtype=float)
        R = np.array([r.mean_r for r in group], dtype=float)
        if args.threshold == "adaptive-half":
            thr = adaptive_half_threshold(R, args.adaptive_low_points, args.adaptive_high_points)
        elif args.threshold == "goe-poisson-half":
            thr = HALF_R
        else:
            thr = float(args.threshold)
        W_star = first_descending_crossing(W, R, thr)
        midpoints.append((U, W_star, thr))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(grouped, curve_u, args.output_dir / "gap_ratio_curves_prb",
                "(a)", threshold_for_curves, args.curve_labeling)
    plot_midpoints(midpoints, args.output_dir / "gap_ratio_midpoint_Wstar",
                   "(b)")
    write_midpoints_csv(midpoints, args.output_dir / "gap_ratio_midpoints.csv")

    print(f"Saved {args.output_dir / 'gap_ratio_curves_prb.png'}")
    print(f"Saved {args.output_dir / 'gap_ratio_midpoint_Wstar.png'}")
    print(f"Saved {args.output_dir / 'gap_ratio_midpoints.csv'}")
    print("Midpoints:")
    for U, W_star, thr in midpoints:
        print(f"U={U:g} W*={W_star:.6g} threshold={thr:.6g}")


if __name__ == "__main__":
    main()
