#!/usr/bin/env python3
"""Create the complete PRB-style figure set from merged analysis tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import boson_peak_core as core


GOE = 0.5307
POISSON = 2.0 * math.log(2.0) - 1.0
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")


def prb_style() -> Dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / (stem + "." + suffix)
        fig.savefig(path, dpi=dpi if suffix == "png" else None)
        print("Saved: {}".format(path))
    plt.close(fig)


def panel_label(ax: mpl.axes.Axes, index: int) -> None:
    ax.text(0.965, 0.95, "({})".format(chr(ord("a") + index)), transform=ax.transAxes,
            ha="right", va="top", fontsize=12, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4})


def sector_text(ax: mpl.axes.Axes, L: int, N: int, nmax: int) -> None:
    ax.text(0.965, 0.82, r"$L={},\ N={},\ n_{{\max}}={}$".format(L, N, nmax),
            transform=ax.transAxes, ha="right", va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4})


def grid_axes(count: int, width: float = 7.0, sharex: bool = False, sharey: bool = False):
    columns = 2 if count > 1 else 1
    rows = int(math.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(width, 2.55 * rows), sharex=sharex, sharey=sharey, squeeze=False)
    flat = list(axes.flat)
    for ax in flat[count:]:
        ax.set_visible(False)
    return fig, flat[:count]


def bool_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = frame[name]
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    return values.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy(dtype=bool)


def optional_bool_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    return bool_column(frame, name)


def main_sectors(frame: pd.DataFrame) -> pd.DataFrame:
    unit = frame[frame["L"] == frame["N"]]
    if unit.empty:
        return frame
    max_L = int(unit["L"].max())
    return unit[unit["L"] == max_L]


def colors_for(values: Sequence[float], cmap_name: str = "plasma"):
    unique = np.asarray(sorted(set(float(x) for x in values)), dtype=float)
    normalization = mpl.colors.Normalize(vmin=float(unique.min()), vmax=float(unique.max()) if unique.size > 1 else float(unique.min() + 1.0))
    cmap = mpl.colormaps[cmap_name]
    return unique, normalization, cmap


def common_colorbar(fig: mpl.figure.Figure, axes: Sequence[mpl.axes.Axes], norm, cmap, label: str) -> None:
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=list(axes), fraction=0.035, pad=0.035)
    colorbar.set_label(label)


def plot_entropy_curves(curves: pd.DataFrame, peaks: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = main_sectors(curves)
    sectors = list(selected.groupby(["L", "N", "nmax"], sort=True))
    if not sectors:
        return
    fig, axes = grid_axes(len(sectors), sharex=False, sharey=True)
    W_values, norm, cmap = colors_for(selected["W_over_t"].unique())
    for index, ((L, N, nmax), group) in enumerate(sectors):
        ax = axes[index]
        for marker_index, (W, curve) in enumerate(group.groupby("W_over_t", sort=True)):
            curve = curve.sort_values("U_over_t")
            x = curve["U_over_t"].to_numpy(dtype=float)
            y = curve["delta_entropy_norm_mean"].to_numpy(dtype=float)
            error = curve["delta_entropy_norm_paired_sem"].to_numpy(dtype=float)
            color = cmap(norm(float(W)))
            ax.plot(x, y, color=color, marker=MARKERS[marker_index % len(MARKERS)], markevery=max(1, len(x) // 9))
            ax.fill_between(x, y - error, y + error, color=color, alpha=0.12, linewidth=0)
            row = peaks[(peaks["L"] == L) & (peaks["N"] == N) & (peaks["nmax"] == nmax) & np.isclose(peaks["W_over_t"], W)]
            if not row.empty:
                item = row.iloc[0]
                theory_x = float(item["U_M_star_over_t"])
                theory_y = float(np.interp(theory_x, x, y))
                ax.plot(theory_x, theory_y, marker="*", color=color, markeredgecolor="black", markeredgewidth=0.35, markersize=6.5, linestyle="none")
                resolved = bool_column(row, "peak_resolved")[0]
                ed_x = float(item["U_S_star_over_t"] if resolved else item["U_S_candidate_over_t"])
                ed_y = float(np.interp(ed_x, x, y))
                ax.plot(ed_x, ed_y, marker="D", mfc=color if resolved else "none", mec="black", mew=0.45,
                        color=color, markersize=4.5, linestyle="none")
                if resolved:
                    ax.hlines(ed_y, float(item["U_S_ci95_low"]), float(item["U_S_ci95_high"]), color=color, lw=1.0)
                else:
                    ax.hlines(ed_y, float(item["plateau_U_low_over_t"]), float(item["plateau_U_high_over_t"]), color=color, lw=1.0)
        ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
        ax.set_xlabel(r"$U/t$")
        if index % 2 == 0:
            ax.set_ylabel(r"$\Delta S_F/\ln D$")
        panel_label(ax, index)
        sector_text(ax, int(L), int(N), int(nmax))
    common_colorbar(fig, axes, norm, cmap, r"$W/t$")
    fig.subplots_adjust(wspace=0.16, hspace=0.18, right=0.88)
    save_figure(fig, output, "entropy_curves_L9", dpi)


def plot_peak_positions(peaks: pd.DataFrame, theory: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = main_sectors(peaks)
    sectors = list(selected.groupby(["L", "N", "nmax"], sort=True))
    if not sectors:
        return
    fig, axes = grid_axes(len(sectors), sharex=True, sharey=False)
    for index, ((L, N, nmax), group) in enumerate(sectors):
        ax = axes[index]
        smooth = theory[(theory["L"] == L) & (theory["N"] == N) & (theory["nmax"] == nmax)].sort_values("W_over_t")
        ax.plot(smooth["W_over_t"], smooth["U_M_star_over_t"], color="black", label=r"channel theory $U_M^*$")
        small = smooth.iloc[np.flatnonzero(optional_bool_column(smooth, "small_W_regime"))]
        large = smooth.iloc[np.flatnonzero(optional_bool_column(smooth, "large_W_regime"))]
        if not small.empty:
            ax.plot(small["W_over_t"], small["U_small_W_over_t"], color="0.45", ls=":", label=r"$AW+BW^3$")
        if not large.empty:
            ax.plot(large["W_over_t"], large["U_large_W_over_t"], color="0.45", ls="--", label=r"$C\sqrt{W}$")
        group = group.sort_values("W_over_t")
        same_lower = group["U_M_same_samples_over_t"] - group["U_M_same_samples_ci95_low"]
        same_upper = group["U_M_same_samples_ci95_high"] - group["U_M_same_samples_over_t"]
        ax.plot(group["W_over_t"], group["U_M_same_samples_over_t"], marker="x", color="#009E73",
                linestyle="none", label="same-sample theory")
        valid_same_ci = np.isfinite(same_lower.to_numpy(dtype=float)) & np.isfinite(same_upper.to_numpy(dtype=float))
        if np.any(valid_same_ci):
            ax.errorbar(
                group.loc[valid_same_ci, "W_over_t"], group.loc[valid_same_ci, "U_M_same_samples_over_t"],
                yerr=np.vstack((same_lower.loc[valid_same_ci].clip(lower=0.0),
                                same_upper.loc[valid_same_ci].clip(lower=0.0))),
                fmt="none", ecolor="#009E73", capsize=2.0,
            )
        resolved_mask = bool_column(group, "peak_resolved")
        resolved = group.iloc[np.flatnonzero(resolved_mask)]
        unresolved = group.iloc[np.flatnonzero(~resolved_mask)]
        if not resolved.empty:
            lower = resolved["U_S_star_over_t"] - resolved["U_S_ci95_low"]
            upper = resolved["U_S_ci95_high"] - resolved["U_S_star_over_t"]
            ax.errorbar(resolved["W_over_t"], resolved["U_S_star_over_t"], yerr=np.vstack((lower, upper)),
                        fmt="o", color="#0072B2", capsize=2.5, label=r"ED $U_S^*$")
        if not unresolved.empty:
            ax.plot(unresolved["W_over_t"], unresolved["U_S_candidate_over_t"], "o", mfc="none", mec="#D55E00",
                    label="unresolved ED")
            for _, row in unresolved.iterrows():
                ax.vlines(float(row["W_over_t"]), float(row["plateau_U_low_over_t"]),
                          float(row["plateau_U_high_over_t"]), color="#D55E00", lw=1.0)
        panel_label(ax, index)
        sector_text(ax, int(L), int(N), int(nmax))
        ax.set_xlabel(r"$W/t$")
        if index % 2 == 0:
            ax.set_ylabel(r"$U^*/t$")
        if index == 0:
            ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(wspace=0.20, hspace=0.18)
    save_figure(fig, output, "peak_positions_vs_W", dpi)


def plot_peak_difference(peaks: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = main_sectors(peaks)
    sectors = list(selected.groupby(["L", "N", "nmax"], sort=True))
    if not sectors:
        return
    fig, axes = grid_axes(len(sectors), sharex=True, sharey=True)
    for index, ((L, N, nmax), group) in enumerate(sectors):
        ax = axes[index]
        group = group.sort_values("W_over_t")
        mask = bool_column(group, "peak_resolved")
        resolved = group.iloc[np.flatnonzero(mask)]
        unresolved = group.iloc[np.flatnonzero(~mask)]
        if not resolved.empty:
            difference = resolved["U_S_star_over_t"] - resolved["U_M_star_over_t"]
            lower = resolved["U_S_star_over_t"] - resolved["U_S_ci95_low"]
            upper = resolved["U_S_ci95_high"] - resolved["U_S_star_over_t"]
            ax.errorbar(resolved["W_over_t"], difference, yerr=np.vstack((lower, upper)), fmt="o-", color="#0072B2", capsize=2.5)
        if not unresolved.empty:
            candidate = unresolved["U_S_candidate_over_t"] - unresolved["U_M_star_over_t"]
            ax.plot(unresolved["W_over_t"], candidate, "o", mfc="none", mec="#D55E00")
        ax.axhline(0.0, color="black", lw=0.9, ls="--")
        panel_label(ax, index)
        sector_text(ax, int(L), int(N), int(nmax))
        ax.set_xlabel(r"$W/t$")
        if index % 2 == 0:
            ax.set_ylabel(r"$(U_S^*-U_M^*)/t$")
    fig.subplots_adjust(wspace=0.16, hspace=0.18)
    save_figure(fig, output, "peak_difference_vs_W", dpi)


def plot_coefficients(coefficients: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.8))
    unit = coefficients[coefficients["L"] == coefficients["N"]]
    for line_index, (L, group) in enumerate(unit.groupby("L", sort=True)):
        group = group.sort_values("nmax")
        for ax, column in zip(axes[0], ("A", "B", "C")):
            ax.plot(group["nmax"], group[column], marker=MARKERS[line_index % len(MARKERS)], label=r"$L=N={}$".format(int(L)))
    L9 = coefficients[coefficients["L"] == 9]
    for line_index, (nmax, group) in enumerate(L9.groupby("nmax", sort=True)):
        group = group.sort_values("N")
        for ax, column in zip(axes[1], ("A", "B", "C")):
            ax.plot(group["N"], group[column], marker=MARKERS[line_index % len(MARKERS)],
                    label=r"$n_{{\max}}={}$".format(int(nmax)))
    for index, (ax, column) in enumerate(zip(axes[0], ("A", "B", "C"))):
        ax.set_xlabel(r"$n_{\max}$")
        ax.set_ylabel(r"${}$".format(column))
        panel_label(ax, index)
        if index == 0:
            if ax.lines:
                ax.legend(loc="upper left", frameon=False, ncol=2)
    for offset, (ax, column) in enumerate(zip(axes[1], ("A", "B", "C")), start=3):
        ax.set_xlabel(r"$N$ at $L=9$")
        ax.set_ylabel(r"${}$".format(column))
        panel_label(ax, offset)
        if offset == 3:
            if ax.lines:
                ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.subplots_adjust(wspace=0.38, hspace=0.30)
    save_figure(fig, output, "coefficients_vs_nmax_and_N", dpi)


def plot_effective_exponent(theory: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = main_sectors(theory)
    fig, ax = plt.subplots(figsize=(3.55, 2.75))
    for index, ((L, N, nmax), group) in enumerate(selected.groupby(["L", "N", "nmax"], sort=True)):
        group = group.sort_values("W_over_t")
        mask = np.isfinite(group["beta_eff"].to_numpy(dtype=float))
        ax.semilogx(group.loc[mask, "W_over_t"], group.loc[mask, "beta_eff"], marker=MARKERS[index % len(MARKERS)],
                    markevery=max(1, int(mask.sum()) // 8), label=r"$n_{{\max}}={}$".format(int(nmax)))
    ax.axhline(1.0, color="0.45", ls=":", label=r"$\beta=1$")
    ax.axhline(0.5, color="0.45", ls="--", label=r"$\beta=1/2$")
    ax.set_xlabel(r"$W/t$")
    ax.set_ylabel(r"$\beta_{\mathrm{eff}}=d\ln U_M^*/d\ln W$")
    ax.legend(loc="best", frameon=False, ncol=2)
    save_figure(fig, output, "effective_exponent", dpi)


def plot_peak_visibility(peaks: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = main_sectors(peaks)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    columns = (
        ("peak_height_delta_entropy_norm", r"$\Delta S_F^{\mathrm{peak}}/\ln D$"),
        ("peak_significance_z", r"peak significance $z$"),
        ("curvature", r"local curvature"),
        ("plateau_width_over_t", r"plateau width$/t$"),
    )
    for line_index, ((L, N, nmax), group) in enumerate(selected.groupby(["L", "N", "nmax"], sort=True)):
        group = group.sort_values("W_over_t")
        for ax, (column, _) in zip(axes.flat, columns):
            ax.plot(group["W_over_t"], group[column], marker=MARKERS[line_index % len(MARKERS)],
                    label=r"$n_{{\max}}={}$".format(int(nmax)))
    for index, (ax, (_, label)) in enumerate(zip(axes.flat, columns)):
        ax.set_ylabel(label)
        panel_label(ax, index)
        if index >= 2:
            ax.set_xlabel(r"$W/t$")
    axes.flat[0].legend(loc="upper left", frameon=False, ncol=2)
    fig.subplots_adjust(wspace=0.28, hspace=0.17)
    save_figure(fig, output, "peak_visibility", dpi)


def plot_finite_size(peaks: pd.DataFrame, output: Path, dpi: int) -> None:
    unit = peaks[peaks["L"] == peaks["N"]]
    fig, ax = plt.subplots(figsize=(3.7, 2.85))
    selected_W = sorted(unit["W_over_t"].unique())
    for index, W in enumerate(selected_W):
        group = unit[np.isclose(unit["W_over_t"], W)]
        preferred = 4 if np.any(group["nmax"] == 4) else int(group["nmax"].min())
        group = group[group["nmax"] == preferred].sort_values("L")
        resolved = group.iloc[np.flatnonzero(bool_column(group, "peak_resolved"))]
        if not resolved.empty:
            ax.plot(resolved["L"], resolved["U_S_star_over_t"], marker=MARKERS[index % len(MARKERS)], label=r"$W/t={:g}$".format(W))
    if not ax.lines:
        ax.text(0.5, 0.5, "Finite-size production sectors are not present", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.legend(loc="best", frameon=False, ncol=2)
    ax.set_xlabel(r"$L=N$")
    ax.set_ylabel(r"$U_S^*/t$")
    save_figure(fig, output, "finite_size_comparison", dpi)


def plot_nmax_comparison(peaks: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = main_sectors(peaks)
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    W_values, norm, cmap = colors_for(selected["W_over_t"].unique())
    for index, (W, group) in enumerate(selected.groupby("W_over_t", sort=True)):
        group = group.sort_values("nmax")
        mask = bool_column(group, "peak_resolved")
        color = cmap(norm(float(W)))
        resolved = group.iloc[np.flatnonzero(mask)]
        unresolved = group.iloc[np.flatnonzero(~mask)]
        if not resolved.empty:
            ax.plot(resolved["nmax"], resolved["U_S_star_over_t"], color=color, marker=MARKERS[index % len(MARKERS)])
        if not unresolved.empty:
            ax.plot(unresolved["nmax"], unresolved["U_S_candidate_over_t"], linestyle="none", marker="o", mfc="none", mec=color)
        nmax2 = group[group["nmax"] == 2]
        if not nmax2.empty:
            y = float(nmax2["U_S_star_over_t"].iloc[0] if bool_column(nmax2, "peak_resolved")[0] else nmax2["U_S_candidate_over_t"].iloc[0])
            ax.plot([2], [y], marker="o", mfc="white", mec=color, mew=1.2, linestyle="none")
    ax.set_xlabel(r"$n_{\max}$")
    ax.set_ylabel(r"$U_S^*/t$")
    common_colorbar(fig, [ax], norm, cmap, r"$W/t$")
    save_figure(fig, output, "nmax_comparison_L9", dpi)


def plot_entropy_decomposition(curves: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = main_sectors(curves)
    available = sorted(int(x) for x in selected["nmax"].unique())
    preferred = next((value for value in available if value > 2), available[0])
    selected = selected[selected["nmax"] == preferred]
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.5), sharex=True)
    fields = (("H_Q_mean", r"$H(P_Q)/\ln D$"), ("S_intra_Q_mean", r"$S_{\mathrm{intra},Q}/\ln D$"),
              ("entropy_norm_mean", r"$S_F/\ln D$"))
    dimension = core.bounded_composition_dim(int(selected["L"].iloc[0]), int(selected["N"].iloc[0]), preferred)
    log_dimension = math.log(dimension)
    W_values, norm, cmap = colors_for(selected["W_over_t"].unique())
    for line_index, (W, group) in enumerate(selected.groupby("W_over_t", sort=True)):
        group = group.sort_values("U_over_t")
        for field_index, (ax, (field, _)) in enumerate(zip(axes, fields)):
            values = group[field] / log_dimension if field_index < 2 else group[field]
            ax.plot(group["U_over_t"], values, color=cmap(norm(float(W))), marker=MARKERS[line_index % len(MARKERS)],
                    markevery=max(1, len(group) // 8))
    for index, (ax, (_, label)) in enumerate(zip(axes, fields)):
        ax.set_xlabel(r"$U/t$")
        ax.set_ylabel(label)
        panel_label(ax, index)
    common_colorbar(fig, axes, norm, cmap, r"$W/t$")
    fig.subplots_adjust(wspace=0.38, right=0.88)
    save_figure(fig, output, "entropy_decomposition_Q", dpi)


def load_table(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        raise SystemExit("Missing analysis table: {}. Run merge_results.py and analyze_peaks.py first.".format(path))
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else run_dir / "figures"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    dpi = int(manifest["config"].get("analysis", {}).get("figure_dpi", 300))
    curves = load_table(run_dir, "ed_averaged_curves.csv")
    peaks = load_table(run_dir, "peak_summary.csv")
    theory = load_table(run_dir, "theory_peaks.csv")
    coefficients = load_table(run_dir, "theory_coefficients.csv")
    with mpl.rc_context(prb_style()):
        plot_entropy_curves(curves, peaks, output, dpi)
        plot_peak_positions(peaks, theory, output, dpi)
        plot_peak_difference(peaks, output, dpi)
        plot_coefficients(coefficients, output, dpi)
        plot_effective_exponent(theory, output, dpi)
        plot_peak_visibility(peaks, output, dpi)
        plot_finite_size(peaks, output, dpi)
        plot_nmax_comparison(peaks, output, dpi)
        plot_entropy_decomposition(curves, output, dpi)


if __name__ == "__main__":
    main()
