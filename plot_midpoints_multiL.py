#!/usr/bin/env python3
"""Plot W*(U) midpoint crossings for several system sizes on one figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_midpoints(path: Path) -> tuple[np.ndarray, np.ndarray]:
    U_vals = []
    W_vals = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                U = float(row["U"])
                W = float(row["W_star"])
            except ValueError:
                continue
            if np.isfinite(U) and np.isfinite(W):
                U_vals.append(U)
                W_vals.append(W)
    order = np.argsort(U_vals)
    return np.array(U_vals)[order], np.array(W_vals)[order]


def style_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Entry in the form L:path/to/gap_ratio_midpoints.csv",
    )
    parser.add_argument("--output", type=Path, default=Path("midpoints_multiL"))
    parser.add_argument("--title", default=r"$\langle r\rangle=r_{1/2}$ crossing")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    style_matplotlib()
    fig, ax = plt.subplots(figsize=(3.65, 2.65), constrained_layout=True)

    markers = ["o", "s", "^", "D", "v", "P"]
    colors = plt.get_cmap("tab10").colors

    for i, entry in enumerate(args.input):
        if ":" not in entry:
            raise SystemExit(f"Bad --input entry: {entry}")
        label, path_text = entry.split(":", 1)
        U, W = read_midpoints(Path(path_text))
        ax.plot(
            U,
            W,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            lw=1.35,
            ms=4.0,
            label=rf"$L={label}$",
        )

    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$W^\ast$")
    ax.set_title(args.title, pad=3)
    ax.legend(frameon=True, fancybox=False, framealpha=0.9, facecolor="white", edgecolor="0.55")
    ax.text(0.03, 0.95, "(b)", transform=ax.transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=600)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(f"Saved {args.output.with_suffix('.png')}")
    print(f"Saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
