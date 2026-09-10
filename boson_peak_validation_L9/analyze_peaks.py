#!/usr/bin/env python3
"""Paired bootstrap, entropy peaks, plateaus, and theory comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import boson_peak_core as core


GROUP = ["L", "N", "nmax", "W_over_t"]
SECTOR = ["L", "N", "nmax"]


def sem(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite, ddof=1) / math.sqrt(finite.size)) if finite.size > 1 else float("nan")


def bool_mask(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column]
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    return values.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy(dtype=bool)


def curve_peak(U: np.ndarray, values: np.ndarray, local_points: int = 5) -> Dict[str, Any]:
    U = np.asarray(U, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(U) & np.isfinite(values)
    U, values = U[finite], values[finite]
    order = np.argsort(U)
    U, values = U[order], values[order]
    if U.size < 3:
        return {"grid_index": -1, "grid_peak": float("nan"), "peak": float("nan"), "curvature": float("nan"), "accepted": False, "boundary": True}
    index = int(np.argmax(values))
    boundary = index == 0 or index == U.size - 1
    half = max(1, int(local_points) // 2)
    start = max(0, index - half)
    stop = min(U.size, index + half + 1)
    if stop - start < 3:
        if start == 0:
            stop = min(U.size, 3)
        else:
            start = max(0, U.size - 3)
    x, y = U[start:stop], values[start:stop]
    center, scale = float(np.mean(x)), float(np.ptp(x))
    accepted = False
    vertex = float(U[index])
    curvature = float("nan")
    condition = float("inf")
    if scale > 0:
        z = (x - center) / scale
        design = np.column_stack((z**2, z, np.ones_like(z)))
        condition = float(np.linalg.cond(design))
        if np.isfinite(condition) and condition < 1.0e10:
            a_z, b_z, _ = np.linalg.lstsq(design, y, rcond=None)[0]
            curvature = float(2.0 * a_z / scale**2)
            if a_z < 0:
                candidate = center - scale * b_z / (2.0 * a_z)
                if float(x[0]) <= candidate <= float(x[-1]):
                    vertex = float(candidate)
                    accepted = not boundary
    return {
        "grid_index": index,
        "grid_peak": float(U[index]),
        "peak": vertex,
        "curvature": curvature,
        "accepted": bool(accepted),
        "boundary": bool(boundary),
        "fit_condition": condition,
        "fit_U_min": float(x[0]),
        "fit_U_max": float(x[-1]),
    }


def pivot_group(frame: pd.DataFrame, value: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = frame.pivot_table(index="realization", columns="U_over_t", values=value, aggfunc="first")
    pivot = pivot.dropna(axis=0, how="any").sort_index(axis=1)
    return pivot.index.to_numpy(dtype=int), pivot.columns.to_numpy(dtype=float), pivot.to_numpy(dtype=float)


def plateau_interval(U: np.ndarray, matrix: np.ndarray, peak_index: int) -> Tuple[float, float, int, bool]:
    means = np.mean(matrix, axis=0)
    accepted = []
    for column in range(U.size):
        paired_difference = matrix[:, peak_index] - matrix[:, column]
        threshold = sem(paired_difference)
        if not math.isfinite(threshold):
            threshold = 0.0
        if means[peak_index] - means[column] <= threshold + 1.0e-14:
            accepted.append(column)
    if not accepted:
        accepted = [peak_index]
    low, high = float(U[min(accepted)]), float(U[max(accepted)])
    includes_zero = any(abs(float(U[index])) <= 1.0e-12 for index in accepted)
    return low, high, len(accepted), includes_zero


def averaged_curves(frame: pd.DataFrame, mixing: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output = []
    for key, group in frame.groupby(GROUP + ["U_over_t"], sort=True):
        values = group["entropy_norm"].to_numpy(dtype=float)
        delta = group["delta_entropy_norm"].to_numpy(dtype=float)
        row = dict(zip(GROUP + ["U_over_t"], key))
        row.update(
            {
                "realizations": int(values.size),
                "entropy_norm_mean": float(np.mean(values)),
                "entropy_norm_sem": sem(values),
                "delta_entropy_norm_mean": float(np.mean(delta)),
                "delta_entropy_norm_paired_sem": sem(delta),
            }
        )
        for column in ("entropy2_norm", "IPR", "gap_ratio", "Q_mean", "Q_variance", "H_Q", "S_intra_Q"):
            data = group[column].to_numpy(dtype=float)
            row[column + "_mean"] = float(np.mean(data))
            row[column + "_sem"] = sem(data)
        output.append(row)
    mixing_output = []
    for key, group in mixing.groupby(GROUP + ["U_over_t"], sort=True):
        values = group["sample_mixing"].to_numpy(dtype=float)
        mixing_output.append(
            {
                **dict(zip(GROUP + ["U_over_t"], key)),
                "realizations": int(values.size),
                "sample_mixing_mean": float(np.mean(values)),
                "sample_mixing_sem": sem(values),
            }
        )
    return pd.DataFrame(output), pd.DataFrame(mixing_output)


def theory_lookup(run_dir: Path) -> Dict[Tuple[int, int, int, float], Dict[str, Any]]:
    lookup = {}
    with (run_dir / "theory_peaks.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["L"]), int(row["N"]), int(row["nmax"]), round(float(row["W_over_t"]), 10))
            lookup[key] = row
    return lookup


def bootstrap_sector(
    sector_frame: pd.DataFrame, sector_mixing: pd.DataFrame, replicates: int, rng: np.random.Generator,
    local_points: int,
) -> Dict[float, Dict[str, np.ndarray]]:
    by_W = {float(W): group for W, group in sector_frame.groupby("W_over_t")}
    mixing_by_W = {float(W): group for W, group in sector_mixing.groupby("W_over_t")}
    common_ids: Optional[set] = None
    for group in by_W.values():
        ids = set(int(x) for x in group["realization"].unique())
        common_ids = ids if common_ids is None else common_ids & ids
    ids_array = np.asarray(sorted(common_ids or []), dtype=int)
    output = {
        W: {"entropy_peak": np.full(replicates, np.nan), "curvature": np.full(replicates, np.nan), "mixing_peak": np.full(replicates, np.nan)}
        for W in by_W
    }
    if ids_array.size < 2:
        return output
    entropy_data = {}
    mixing_data = {}
    for W, group in by_W.items():
        pivot = group.pivot_table(index="realization", columns="U_over_t", values="entropy_norm", aggfunc="first")
        pivot = pivot.reindex(ids_array).dropna(axis=0, how="any").sort_index(axis=1)
        entropy_data[W] = (pivot.columns.to_numpy(dtype=float), pivot.to_numpy(dtype=float))
        mgroup = mixing_by_W[W]
        mpivot = mgroup.pivot_table(index="realization", columns="U_over_t", values="sample_mixing", aggfunc="first")
        mpivot = mpivot.reindex(ids_array).dropna(axis=0, how="any").sort_index(axis=1)
        mixing_data[W] = (mpivot.columns.to_numpy(dtype=float), mpivot.to_numpy(dtype=float))
    for replicate in range(replicates):
        selected = rng.integers(0, ids_array.size, size=ids_array.size)
        for W in by_W:
            U, matrix = entropy_data[W]
            estimate = curve_peak(U, np.mean(matrix[selected], axis=0), local_points)
            if estimate["accepted"]:
                output[W]["entropy_peak"][replicate] = estimate["peak"]
                output[W]["curvature"][replicate] = estimate["curvature"]
            mU, mmatrix = mixing_data[W]
            mixing_estimate = curve_peak(mU, np.mean(mmatrix[selected], axis=0), local_points)
            output[W]["mixing_peak"][replicate] = mixing_estimate["peak"] if mixing_estimate["accepted"] else mixing_estimate["grid_peak"]
        if (replicate + 1) % max(1, replicates // 10) == 0 or replicate + 1 == replicates:
            print("Bootstrap sector: {:.0f}% ({}/{})".format(100.0 * (replicate + 1) / replicates, replicate + 1, replicates), flush=True)
    return output


def finite_quantiles(values: np.ndarray) -> Tuple[float, float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan"), 0.0
    low, median, high = np.quantile(finite, [0.025, 0.5, 0.975])
    return float(low), float(median), float(high), float(finite.size / values.size)


def analyze_group(
    group: pd.DataFrame,
    mixing_group: pd.DataFrame,
    bootstrap: Dict[str, np.ndarray],
    theory: Dict[str, Any],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    ids, U, matrix = pivot_group(group, "entropy_norm")
    _, mixing_U, mixing_matrix = pivot_group(mixing_group, "sample_mixing")
    means = np.mean(matrix, axis=0)
    estimate = curve_peak(U, means, int(settings.get("local_fit_points", 5)))
    peak_index = int(estimate["grid_index"])
    plateau_low, plateau_high, plateau_points, plateau_zero = plateau_interval(U, matrix, peak_index)
    zero_candidates = np.flatnonzero(np.isclose(U, 0.0, atol=1.0e-12, rtol=0.0))
    if zero_candidates.size:
        paired_peak = matrix[:, peak_index] - matrix[:, zero_candidates[0]]
        height = float(np.mean(paired_peak))
        height_sem = sem(paired_peak)
    else:
        height, height_sem = float("nan"), float("nan")
    if math.isfinite(height_sem) and height_sem > 0:
        significance = height / height_sem
    elif math.isfinite(height) and height > 0 and height_sem == 0:
        significance = float("inf")
    else:
        significance = float("nan")
    low, median, high, valid_fraction = finite_quantiles(bootstrap["entropy_peak"])
    curvature_low, _, curvature_high, curvature_valid = finite_quantiles(bootstrap["curvature"])
    mix_estimate = curve_peak(mixing_U, np.mean(mixing_matrix, axis=0), int(settings.get("local_fit_points", 5)))
    mix_low, mix_median, mix_high, mix_valid = finite_quantiles(bootstrap["mixing_peak"])
    grid_step = float(np.median(np.diff(U)) / 2.0) if U.size > 1 else float("nan")
    max_plateau_points = int(settings.get("max_plateau_points", 3))
    min_z = float(settings.get("min_peak_significance_z", 2.0))
    min_valid = float(settings.get("min_bootstrap_valid_fraction", 0.5))
    max_ci_fraction = float(settings.get("max_ci_width_fraction_scan", 0.6))
    max_required = int(settings.get("max_required_realizations", 500))
    scan_width = float(np.ptp(U))
    ci_width = high - low if math.isfinite(low) and math.isfinite(high) else float("nan")
    broad_ci = not math.isfinite(ci_width) or (scan_width > 0 and ci_width > max_ci_fraction * scan_width)
    required = float("nan")
    if math.isfinite(significance) and significance > 0:
        required = len(ids) * (3.0 / significance) ** 2
    elif math.isinf(significance):
        required = 0.0
    exceeds_realization_limit = math.isfinite(required) and required > max_required
    flat_curvature = math.isfinite(curvature_high) and curvature_high >= 0.0
    resolved = (
        bool(estimate["accepted"])
        and not bool(estimate["boundary"])
        and not plateau_zero
        and plateau_points <= max_plateau_points
        and significance >= min_z
        and valid_fraction >= min_valid
        and not broad_ci
        and not exceeds_realization_limit
        and not flat_curvature
    )
    U_S = median if resolved else float("nan")
    U_M = float(theory["U_M_star_over_t"])
    agreement = "unresolved"
    if resolved:
        agreement = "consistent" if low - grid_step <= U_M <= high + grid_step else "inconsistent"
    reasons = []
    if not bool(estimate["accepted"]):
        reasons.append("local_fit_rejected")
    if bool(estimate["boundary"]):
        reasons.append("scan_boundary")
    if plateau_zero:
        reasons.append("plateau_includes_U0")
    if plateau_points > max_plateau_points:
        reasons.append("wide_plateau")
    if not math.isfinite(significance) or significance < min_z:
        reasons.append("low_peak_significance")
    if valid_fraction < min_valid:
        reasons.append("low_bootstrap_valid_fraction")
    if broad_ci:
        reasons.append("broad_bootstrap_interval")
    if flat_curvature:
        reasons.append("curvature_not_negative")
    if exceeds_realization_limit:
        reasons.append("realization_limit_exceeded")
    return {
        "L": int(group["L"].iloc[0]), "N": int(group["N"].iloc[0]), "nmax": int(group["nmax"].iloc[0]),
        "filling": float(group["N"].iloc[0] / group["L"].iloc[0]), "W_over_t": float(group["W_over_t"].iloc[0]),
        "realizations": int(len(ids)), "U_points": int(U.size),
        "U_scan_min_over_t": float(np.min(U)), "U_scan_max_over_t": float(np.max(U)),
        "U_M_star_over_t": U_M,
        "U_M_same_samples_over_t": float(mix_median if math.isfinite(mix_median) else mix_estimate["grid_peak"]),
        "U_M_same_samples_ci95_low": mix_low, "U_M_same_samples_ci95_high": mix_high,
        "U_S_candidate_over_t": float(estimate["peak"]),
        "U_S_grid_max_over_t": float(estimate["grid_peak"]),
        "U_S_star_over_t": U_S,
        "U_S_ci95_low": low, "U_S_ci95_high": high,
        "bootstrap_valid_fraction": valid_fraction,
        "bootstrap_ci_width_over_t": ci_width,
        "grid_error_over_t": grid_step,
        "peak_resolved": bool(resolved), "peak_at_boundary": bool(estimate["boundary"]),
        "plateau_U_low_over_t": plateau_low, "plateau_U_high_over_t": plateau_high,
        "plateau_width_over_t": plateau_high - plateau_low, "plateau_points": plateau_points,
        "plateau_includes_U0": bool(plateau_zero),
        "peak_height_delta_entropy_norm": height, "peak_height_paired_sem": height_sem,
        "peak_significance_z": significance,
        "curvature": float(estimate["curvature"]), "curvature_ci95_low": curvature_low,
        "curvature_ci95_high": curvature_high, "curvature_bootstrap_valid_fraction": curvature_valid,
        "R_required_for_3sigma": required,
        "max_required_realizations": max_required,
        "unresolved_reasons": ";".join(reasons),
        "delta_U_star_over_t": U_S - U_M if resolved else float("nan"),
        "relative_peak_error": abs(U_S - U_M) / U_M if resolved and U_M > 0 else float("nan"),
        "theory_ED_status": agreement,
    }


def scan_extension_proposals(peaks: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "L", "N", "nmax", "W_over_t", "recommended_action", "reason",
        "proposed_U_min_over_t", "proposed_U_max_over_t", "proposed_U_values_over_t_json",
        "suggested_realizations",
    ]
    rows = []
    for _, row in peaks[peaks["peak_resolved"] == False].iterrows():  # noqa: E712
        lower = float(row["U_scan_min_over_t"])
        upper = float(row["U_scan_max_over_t"])
        candidate = float(row["U_S_grid_max_over_t"])
        span = max(upper - lower, 0.1)
        spacing = span / max(int(row["U_points"]) - 1, 1)
        action = "refine_plateau"
        proposed_low = max(0.0, float(row["plateau_U_low_over_t"]))
        proposed_high = float(row["plateau_U_high_over_t"])
        if bool(row["peak_at_boundary"]):
            if abs(candidate - upper) <= max(1.0e-10, spacing / 4.0):
                action = "extend_upper_boundary"
                proposed_low = max(0.0, upper - 2.0 * spacing)
                proposed_high = upper + max(0.5 * span, 4.0 * spacing)
            elif lower > 0:
                action = "extend_lower_boundary"
                proposed_low = max(0.0, lower - max(0.5 * span, 4.0 * spacing))
                proposed_high = min(upper, lower + 2.0 * spacing)
            else:
                action = "refine_near_U0"
                proposed_low = 0.0
                proposed_high = min(upper, max(float(row["plateau_U_high_over_t"]), 4.0 * spacing))
        elif bool(row["plateau_includes_U0"]):
            action = "refine_near_U0"
            proposed_low = 0.0
            proposed_high = min(upper, max(float(row["plateau_U_high_over_t"]), 4.0 * spacing))
        proposed_high = max(proposed_high, proposed_low + max(spacing, 0.05))
        values = np.linspace(proposed_low, proposed_high, 9)
        rows.append(
            {
                "L": int(row["L"]), "N": int(row["N"]), "nmax": int(row["nmax"]),
                "W_over_t": float(row["W_over_t"]), "recommended_action": action,
                "reason": str(row["unresolved_reasons"]),
                "proposed_U_min_over_t": proposed_low, "proposed_U_max_over_t": proposed_high,
                "proposed_U_values_over_t_json": json.dumps([round(float(value), 8) for value in values]),
                "suggested_realizations": min(
                    int(row["max_required_realizations"]),
                    max(int(row["realizations"]), int(math.ceil(float(row["R_required_for_3sigma"]))))
                    if math.isfinite(float(row["R_required_for_3sigma"])) else int(row["realizations"]),
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_validation_report(
    run_dir: Path, peaks: pd.DataFrame, fits: pd.DataFrame,
    coefficients: pd.DataFrame, asymptotic: pd.DataFrame,
) -> None:
    resolved = peaks[peaks["peak_resolved"] == True]  # noqa: E712
    consistent = resolved[resolved["theory_ED_status"] == "consistent"]
    inconsistent = resolved[resolved["theory_ED_status"] == "inconsistent"]
    unresolved = peaks[peaks["peak_resolved"] == False]  # noqa: E712
    lines = [
        "# Bosonic entropy-peak validation report",
        "",
        "This report separates resolved agreement, resolved disagreement, and numerically unresolved peaks.",
        "",
        "## Summary",
        "",
        "- Points analyzed: {}".format(len(peaks)),
        "- Resolved peaks: {}".format(len(resolved)),
        "- Consistent with independent channel theory: {}".format(len(consistent)),
        "- Inconsistent with independent channel theory: {}".format(len(inconsistent)),
        "- Numerically unresolved: {}".format(len(unresolved)),
        "",
        "A point is unresolved when the maximum lies on the scan boundary, its plateau includes U=0,",
        "the paired peak significance is too small, the bootstrap interval is too broad, the local",
        "curvature is not robustly negative, or the estimated realization requirement exceeds the cap.",
        "",
        "## Resolved points",
        "",
        "| L | N | nmax | W/t | U_M*/t | U_S*/t | 95% CI | status |",
        "|---:|---:|---:|---:|---:|---:|:---|:---|",
    ]
    for _, row in resolved.iterrows():
        lines.append(
            "| {} | {} | {} | {:g} | {:.5g} | {:.5g} | [{:.5g}, {:.5g}] | {} |".format(
                int(row["L"]), int(row["N"]), int(row["nmax"]), float(row["W_over_t"]),
                float(row["U_M_star_over_t"]), float(row["U_S_star_over_t"]),
                float(row["U_S_ci95_low"]), float(row["U_S_ci95_high"]), row["theory_ED_status"],
            )
        )
    if resolved.empty:
        lines.append("| - | - | - | - | - | - | - | no resolved points |")

    reason_counts: Dict[str, int] = {}
    for text in unresolved.get("unresolved_reasons", pd.Series(dtype=str)).fillna(""):
        for reason in str(text).split(";"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    small = asymptotic[asymptotic["small_W_regime"].astype(str).str.lower().isin(("true", "1"))]
    large = asymptotic[asymptotic["large_W_regime"].astype(str).str.lower().isin(("true", "1"))]
    peak_with_coefficients = peaks.merge(
        coefficients[["L", "N", "nmax", "Gamma_max_over_t", "n_channels"]], on=["L", "N", "nmax"], how="left"
    )
    large_ed = peak_with_coefficients[
        peak_with_coefficients["Gamma_max_over_t"] / peak_with_coefficients["W_over_t"] < 0.3
    ]
    finite_groups = []
    unit = resolved[resolved["L"] == resolved["N"]]
    for (W, nmax), group in unit.groupby(["W_over_t", "nmax"]):
        group = group.sort_values("L")
        if group["L"].nunique() >= 3:
            slope = float(np.polyfit(group["L"], np.abs(group["delta_U_star_over_t"]), 1)[0])
            finite_groups.append(slope)

    lines.extend(["", "## Answers to the validation questions", ""])
    lines.append("1. **Resolved maxima.** The complete list is the table above. Unresolved reasons: {}.".format(
        ", ".join("{}={}".format(key, value) for key, value in sorted(reason_counts.items())) or "none"
    ))
    lines.append(
        "2. **Direct theory/ED agreement.** Of {} resolved points, {} are consistent and {} are inconsistent after adding the separate grid error.".format(
            len(resolved), len(consistent), len(inconsistent)
        )
    )
    if small.empty:
        lines.append("3. **Small W.** No points satisfy the configured small-W regime test.")
    else:
        lines.append(
            "3. **Small W.** For the independent channel theory, the median relative error of `A W + B W^3` is {:.3g} (maximum {:.3g}). The ED coefficient fits and bootstrap intervals are in `ed_small_W_fits.csv`.".format(
                float(small["small_relative_error"].median()), float(small["small_relative_error"].max())
            )
        )
    if large.empty:
        lines.append("4. **Large W theory.** No points satisfy the configured large-W regime test.")
    else:
        lines.append(
            "4. **Large W theory.** The median relative error of `C sqrt(W)` is {:.3g} (maximum {:.3g}); `effective_exponent.pdf` shows the approach to beta=1/2.".format(
                float(large["large_relative_error"].median()), float(large["large_relative_error"].max())
            )
        )
    if large_ed.empty:
        lines.append("5. **Large W entropy.** No ED points reach the large-W criterion.")
    else:
        large_resolved = int(bool_mask(large_ed, "peak_resolved").sum()) if "peak_resolved" in large_ed else 0
        lines.append(
            "5. **Large W entropy.** {}/{} ED points remain resolved; the others are reported as plateaus and do not count against the theory.".format(
                large_resolved, len(large_ed)
            )
        )
    if resolved.empty:
        lines.append("6. **nmax, filling and size.** There are no resolved ED peaks from which to infer these trends.")
    else:
        lines.append(
            "6. **nmax, filling and size.** Median resolved `U_S*/t` by nmax: {}. Use the dedicated comparison figures for the matched-sector trends.".format(
                ", ".join(
                    "{}:{:.3g}".format(int(key), float(value))
                    for key, value in resolved.groupby("nmax")["U_S_star_over_t"].median().items()
                )
            )
        )
    nmax2_channels = sorted(set(int(value) for value in coefficients.loc[coefficients["nmax"] == 2, "n_channels"]))
    lines.append(
        "7. **nmax=2 control.** Its positive-channel counts are {}; it is plotted separately and is not assumed equivalent to nmax>2.".format(nmax2_channels)
    )
    if not finite_groups:
        lines.append("8. **Finite-size mismatch.** Fewer than three matched sizes have resolved peaks, so the L trend is unresolved.")
    else:
        decreasing = sum(slope < -1.0e-3 for slope in finite_groups)
        increasing = sum(slope > 1.0e-3 for slope in finite_groups)
        flat = len(finite_groups) - decreasing - increasing
        lines.append(
            "8. **Finite-size mismatch.** Across {} matched comparisons, |U_S*-U_M*| decreases in {}, is approximately flat in {}, and increases in {}.".format(
                len(finite_groups), decreasing, flat, increasing
            )
        )

    lines.extend(["", "## Small-W coefficient fits", ""])
    if fits.empty:
        lines.append("No sector has enough resolved small-W points for an ED fit.")
    else:
        lines.extend([
            "| L | N | nmax | points | A theory | A ED [95% CI] | B theory | B ED [95% CI] |",
            "|---:|---:|---:|---:|---:|:---|---:|:---|",
        ])
        for _, row in fits.iterrows():
            lines.append(
                "| {} | {} | {} | {} | {:.4g} | {:.4g} [{:.4g}, {:.4g}] | {:.4g} | {:.4g} [{:.4g}, {:.4g}] |".format(
                    int(row["L"]), int(row["N"]), int(row["nmax"]), int(row["points"]),
                    float(row["A_theory"]), float(row["A_ED_fit"]), float(row["A_ED_ci95_low"]), float(row["A_ED_ci95_high"]),
                    float(row["B_theory"]), float(row["B_ED_fit"]), float(row["B_ED_ci95_low"]), float(row["B_ED_ci95_high"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation checklist",
            "",
            "1. Resolved maxima and their theory comparison are listed above.",
            "2. Grid uncertainty is stored separately in `peak_summary.csv` and is added to the confidence interval only for the consistency test.",
            "3. The analytic small-W coefficients A and B are in `theory_coefficients.csv`; paired-bootstrap ED fits are in `ed_small_W_fits.csv`.",
            "4. The exact theory and its square-root asymptote are compared in `asymptotic_tests.csv`.",
            "5. Unresolved large-W plateaus are not counted as evidence against the theory.",
            "6. Dependence on nmax, N, and L is shown by the comparison figures.",
            "7. nmax=2 is marked separately because it contains only one positive compensating channel.",
            "8. Finite-size trends can be assessed only where multiple L values have resolved peaks.",
            "",
        ]
    )
    (run_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap", type=int)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    settings = config["analysis"]
    replicates = int(args.bootstrap if args.bootstrap is not None else settings.get("bootstrap", 2000))
    frame = pd.read_csv(run_dir / "ed_observables_by_realization.csv")
    mixing = pd.read_csv(run_dir / "sample_mixing_curves.csv")
    averaged, mixing_averaged = averaged_curves(frame, mixing)
    averaged.to_csv(run_dir / "ed_averaged_curves.csv", index=False)
    mixing_averaged.to_csv(run_dir / "sample_mixing_averaged_curves.csv", index=False)
    lookup = theory_lookup(run_dir)
    rng = np.random.default_rng(int(settings.get("bootstrap_seed", 20260908)))
    peak_rows = []
    bootstrap_cache = {}
    sectors = list(frame.groupby(SECTOR, sort=True))
    for sector_number, (sector_key, sector_frame) in enumerate(sectors, start=1):
        print("Analyzing sector {}/{}: L={}, N={}, nmax={}".format(sector_number, len(sectors), *sector_key), flush=True)
        sector_mixing = mixing[
            (mixing["L"] == sector_key[0]) & (mixing["N"] == sector_key[1]) & (mixing["nmax"] == sector_key[2])
        ]
        boot = bootstrap_sector(
            sector_frame, sector_mixing, replicates, rng, int(settings.get("local_fit_points", 5))
        )
        bootstrap_cache[sector_key] = boot
        for W, group in sector_frame.groupby("W_over_t", sort=True):
            mixing_group = sector_mixing[np.isclose(sector_mixing["W_over_t"], W)]
            theory = lookup[(int(sector_key[0]), int(sector_key[1]), int(sector_key[2]), round(float(W), 10))]
            peak_rows.append(analyze_group(group, mixing_group, boot[float(W)], theory, settings))
    peaks = pd.DataFrame(peak_rows).sort_values(GROUP)
    peaks.to_csv(run_dir / "peak_summary.csv", index=False)

    coefficients = pd.read_csv(run_dir / "theory_coefficients.csv")
    theory_peaks = pd.read_csv(run_dir / "theory_peaks.csv")
    asymptotic_rows = []
    for _, row in theory_peaks.iterrows():
        exact = float(row["U_M_star_over_t"])
        small = float(row["U_small_W_over_t"])
        large = float(row["U_large_W_over_t"])
        asymptotic_rows.append(
            {
                "L": int(row["L"]), "N": int(row["N"]), "nmax": int(row["nmax"]), "W_over_t": float(row["W_over_t"]),
                "U_M_star_over_t": exact, "U_small_W_over_t": small, "U_large_W_over_t": large,
                "small_relative_error": abs(small - exact) / exact if exact > 0 else float("nan"),
                "large_relative_error": abs(large - exact) / exact if exact > 0 else float("nan"),
                "small_W_regime": row["small_W_regime"], "large_W_regime": row["large_W_regime"],
                "beta_eff": float(row.get("beta_eff", float("nan"))),
            }
        )
    asymptotic = pd.DataFrame(asymptotic_rows)
    asymptotic.to_csv(run_dir / "asymptotic_tests.csv", index=False)

    fit_rows = []
    for sector_key, group in peaks.groupby(SECTOR, sort=True):
        coeff = coefficients[
            (coefficients["L"] == sector_key[0]) & (coefficients["N"] == sector_key[1]) & (coefficients["nmax"] == sector_key[2])
        ].iloc[0]
        selected = group[(group["peak_resolved"] == True)]  # noqa: E712
        selected = selected[selected["W_over_t"] / float(coeff["Gamma_min_over_t"]) < 0.7]
        if len(selected) >= 2:
            W = selected["W_over_t"].to_numpy(dtype=float)
            y = selected["U_S_star_over_t"].to_numpy(dtype=float)
            design = np.column_stack((W, W**3))
            A_fit, B_fit = np.linalg.lstsq(design, y, rcond=None)[0]
            A_bootstrap = []
            B_bootstrap = []
            for replicate in range(replicates):
                sample = np.asarray(
                    [bootstrap_cache[sector_key][float(value)]["entropy_peak"][replicate] for value in W],
                    dtype=float,
                )
                if np.all(np.isfinite(sample)):
                    sample_A, sample_B = np.linalg.lstsq(design, sample, rcond=None)[0]
                    A_bootstrap.append(float(sample_A))
                    B_bootstrap.append(float(sample_B))
            if A_bootstrap:
                A_low, A_median, A_high = np.quantile(A_bootstrap, [0.025, 0.5, 0.975])
                B_low, B_median, B_high = np.quantile(B_bootstrap, [0.025, 0.5, 0.975])
            else:
                A_low = A_median = A_high = float("nan")
                B_low = B_median = B_high = float("nan")
            fit_rows.append(
                {
                    "L": sector_key[0], "N": sector_key[1], "nmax": sector_key[2], "points": len(selected),
                    "A_theory": float(coeff["A"]), "B_theory": float(coeff["B"]),
                    "A_ED_fit": float(A_fit), "B_ED_fit": float(B_fit),
                    "A_ED_bootstrap_median": float(A_median),
                    "A_ED_ci95_low": float(A_low), "A_ED_ci95_high": float(A_high),
                    "B_ED_bootstrap_median": float(B_median),
                    "B_ED_ci95_low": float(B_low), "B_ED_ci95_high": float(B_high),
                    "bootstrap_valid_fraction": len(A_bootstrap) / replicates,
                }
            )
    fits = pd.DataFrame(
        fit_rows,
        columns=[
            "L", "N", "nmax", "points", "A_theory", "B_theory", "A_ED_fit", "B_ED_fit",
            "A_ED_bootstrap_median", "A_ED_ci95_low", "A_ED_ci95_high",
            "B_ED_bootstrap_median", "B_ED_ci95_low", "B_ED_ci95_high", "bootstrap_valid_fraction",
        ],
    )
    fits.to_csv(run_dir / "ed_small_W_fits.csv", index=False)
    proposals = scan_extension_proposals(peaks)
    proposals.to_csv(run_dir / "scan_extension_proposals.csv", index=False)
    write_validation_report(run_dir, peaks, fits, coefficients, asymptotic)
    print("Saved averaged curves: {}".format(run_dir / "ed_averaged_curves.csv"))
    print("Saved peak summary: {}".format(run_dir / "peak_summary.csv"))
    print("Saved scan proposals: {}".format(run_dir / "scan_extension_proposals.csv"))
    print("Saved report: {}".format(run_dir / "VALIDATION_REPORT.md"))


if __name__ == "__main__":
    main()
