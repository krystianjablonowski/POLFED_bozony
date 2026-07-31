#!/usr/bin/env python3
"""PRB-style multi-panel <r>(W) plot for bosons at unit filling N=L.

The script reads one or more ``gap_ratio_summary.csv`` files produced by
``compute_gap_ratio_map.py`` and plots curves for different interaction
strengths U.  Each summary file becomes one panel, typically one system size L.
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


@dataclass(frozen=True)
class Row:
    U: float
    W: float
    mean_r: float
    stderr_r: float
    L: int | None
    N: int | None
    nmax: int | None
    boundary: str | None


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def read_summary(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            try:
                mean_r = parse_float(rec.get("mean_r"))
            except ValueError:
                continue
            if not np.isfinite(mean_r):
                continue
            rows.append(
                Row(
                    U=parse_float(rec.get("U")),
                    W=parse_float(rec.get("W")),
                    mean_r=mean_r,
                    stderr_r=parse_float(rec.get("stderr_r")),
                    L=parse_int(rec.get("L")),
                    N=parse_int(rec.get("N")),
                    nmax=parse_int(rec.get("nmax")),
                    boundary=rec.get("boundary") or None,
                )
            )
    return rows


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None or text.strip() == "":
        return None
    if text.strip().lower() == "all":
        return None
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def apply_filters(
    rows: list[Row],
    unit_filling: bool,
    nmax: int | None,
    min_u: float | None,
    max_u: float | None,
    min_w: float | None,
    max_w: float | None,
    boundary: str | None,
) -> list[Row]:
    filtered = rows
    if unit_filling:
        filtered = [row for row in filtered if row.L is not None and row.N == row.L]
    if nmax is not None:
        filtered = [row for row in filtered if row.nmax == nmax]
    if boundary is not None:
        filtered = [row for row in filtered if row.boundary == boundary]
    if min_u is not None:
        filtered = [row for row in filtered if row.U >= min_u]
    if max_u is not None:
        filtered = [row for row in filtered if row.U <= max_u]
    if min_w is not None:
        filtered = [row for row in filtered if row.W >= min_w]
    if max_w is not None:
        filtered = [row for row in filtered if row.W <= max_w]
    return filtered


def group_by_u(rows: list[Row]) -> dict[float, list[Row]]:
    grouped: dict[float, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.U, []).append(row)
    return {u: sorted(group, key=lambda item: item.W) for u, group in sorted(grouped.items())}


def panel_title(rows: list[Row], fallback: str) -> str:
    if not rows:
        return fallback
    L_values = sorted({row.L for row in rows if row.L is not None})
    N_values = sorted({row.N for row in rows if row.N is not None})
    nmax_values = sorted({row.nmax for row in rows if row.nmax is not None})
    parts = []
    if len(L_values) == 1:
        parts.append(rf"L={L_values[0]}")
    if len(N_values) == 1:
        parts.append(rf"N={N_values[0]}")
    if len(nmax_values) == 1:
        parts.append(rf"n_{{\max}}={nmax_values[0]}")
    return r"$" + r",\ ".join(parts) + r"$" if parts else fallback


def style_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 13,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.05,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.width": 1.05,
            "ytick.major.width": 1.05,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def choose_u_values(panels: list[list[Row]], requested: list[float] | None) -> list[float]:
    available = sorted({row.U for rows in panels for row in rows})
    if requested is None:
        return available
    available_set = set(available)
    return [u for u in requested if u in available_set]


def make_plot(
    panels: list[tuple[str, list[Row]]],
    output_base: Path,
    requested_u: list[float] | None,
    cmap_name: str,
    ncols: int,
    y_min: float | None,
    y_max: float | None,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    style_matplotlib()
    usable_panels = [(label, rows) for label, rows in panels if rows]
    if not usable_panels:
        raise SystemExit("No usable rows after filtering.")

    all_rows = [row for _, rows in usable_panels for row in rows]
    selected_u = choose_u_values([rows for _, rows in usable_panels], requested_u)
    if not selected_u:
        raise SystemExit("No requested U values are present in the input summaries.")

    ncols = max(1, ncols)
    nrows = int(math.ceil(len(usable_panels) / ncols))
    fig_width = 3.65 * ncols + 0.55
    fig_height = 2.85 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=min(selected_u), vmax=max(selected_u))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "P", "X", "*"]
    panel_letters = "abcdefghijklmnopqrstuvwxyz"

    for panel_index, (label, rows) in enumerate(usable_panels):
        ax = axes[panel_index // ncols][panel_index % ncols]
        grouped = group_by_u(rows)
        for u_index, U in enumerate(selected_u):
            if U not in grouped:
                continue
            group = grouped[U]
            W = np.array([row.W for row in group], dtype=float)
            R = np.array([row.mean_r for row in group], dtype=float)
            order = np.argsort(W)
            ax.plot(
                W[order],
                R[order],
                color=cmap(norm(U)),
                marker=markers[u_index % len(markers)],
                ms=3.0,
                lw=1.15,
                alpha=0.96,
            )

        ax.axhline(GOE_R, color="0.35", lw=1.05, ls="--")
        ax.axhline(POISSON_R, color="0.35", lw=1.15, ls=":")
        letter = panel_letters[panel_index] if panel_index < len(panel_letters) else str(panel_index + 1)
        ax.text(
            0.96,
            0.94,
            rf"$({letter})$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=17,
            fontweight="bold",
        )
        ax.text(
            0.53,
            0.83,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12.5,
        )
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.minorticks_on()

    for index in range(len(usable_panels), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel(r"$W$")
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel(r"$\langle r\rangle$")

    finite_r = np.array([row.mean_r for row in all_rows if np.isfinite(row.mean_r)])
    if y_min is None:
        y_min = min(float(np.min(finite_r)), POISSON_R) - 0.008
    if y_max is None:
        y_max = max(float(np.max(finite_r)), GOE_R) + 0.008
    for ax in axes.flat:
        if ax.has_data():
            ax.set_ylim(y_min, y_max)

    first_ax = axes[0][0]
    goe = mpl.lines.Line2D([0], [0], color="0.35", lw=1.05, ls="--", label="GOE")
    poisson = mpl.lines.Line2D([0], [0], color="0.35", lw=1.15, ls=":", label="Poisson")
    first_ax.legend(
        handles=[goe, poisson],
        loc="lower left",
        frameon=True,
        fancybox=False,
        framealpha=0.88,
        facecolor="white",
        edgecolor="0.55",
        handlelength=1.7,
        borderpad=0.35,
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    data_axes = [ax for ax in axes.flat if ax.has_data()]
    cbar = fig.colorbar(sm, ax=data_axes, pad=0.02, fraction=0.035)
    cbar.set_label(r"$U$")
    preferred_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 25, 30, 40, 50]
    ticks = [tick for tick in preferred_ticks if min(selected_u) <= tick <= max(selected_u)]
    if len(ticks) < 3:
        ticks = np.linspace(min(selected_u), max(selected_u), min(5, len(selected_u)))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{tick:g}" for tick in ticks])

    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.09, top=0.98, wspace=0.08, hspace=0.13)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600)
    fig.savefig(output_base.with_suffix(".pdf"))
    print(f"Saved: {output_base.with_suffix('.png')}")
    print(f"Saved: {output_base.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        action="append",
        required=True,
        help="Path to a gap_ratio_summary.csv file. Repeat once per panel.",
    )
    parser.add_argument("--output", type=Path, default=Path("gap_ratio_bose_unit_filling_panels/r_vs_W_unit_filling"))
    parser.add_argument("--U-list", default="all", help="Comma-separated U values, or 'all'.")
    parser.add_argument("--nmax", type=int, default=None, help="Keep only this nmax if summaries contain mixed data.")
    parser.add_argument("--unit-filling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--boundary", default=None, choices=["open", "periodic"])
    parser.add_argument("--min-U", type=float, default=None)
    parser.add_argument("--max-U", type=float, default=None)
    parser.add_argument("--min-W", type=float, default=1.0)
    parser.add_argument("--max-W", type=float, default=15.0)
    parser.add_argument("--y-min", type=float, default=0.375)
    parser.add_argument("--y-max", type=float, default=0.535)
    parser.add_argument("--cmap", default="plasma")
    parser.add_argument("--ncols", type=int, default=2)
    args = parser.parse_args()

    requested_u = parse_float_list(args.U_list)
    panels: list[tuple[str, list[Row]]] = []
    for path in args.summary:
        rows = read_summary(path)
        rows = apply_filters(
            rows,
            unit_filling=args.unit_filling,
            nmax=args.nmax,
            min_u=args.min_U,
            max_u=args.max_U,
            min_w=args.min_W,
            max_w=args.max_W,
            boundary=args.boundary,
        )
        panels.append((panel_title(rows, path.parent.name), rows))

    make_plot(
        panels=panels,
        output_base=args.output,
        requested_u=requested_u,
        cmap_name=args.cmap,
        ncols=args.ncols,
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()
