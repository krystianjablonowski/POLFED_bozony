#!/usr/bin/env python3
"""Numerical verification of the two-component entropy-maximum mechanism.

The script consumes the realization-resolved output of ``merge_results.py``.
It performs no diagonalization.  For every sector and disorder value it checks

    S_F = H(P_Q) + S_intra,Q,

locates the maxima and curvatures of S_F, H(P_Q), S_intra,Q and the mixing
observable M, identifies which entropy component follows M, and tests the
parameter-free curvature-weighted prediction for the entropy maximum.

Bootstrap resampling is paired: a single realization resample is reused for
all W values and all observables within a particle-number sector.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import boson_peak_core as core


VERSION = "1.0.0"
SECTOR = ("L", "N", "nmax")
OBSERVABLES = ("S", "H_Q", "S_in", "M")


def finite_quantiles(values: np.ndarray) -> Tuple[float, float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (float("nan"),) * 4
    low, median, high = np.quantile(finite, [0.025, 0.5, 0.975])
    return float(low), float(median), float(high), float(finite.size / values.size)


def sem(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    return float(np.std(finite, ddof=1) / math.sqrt(finite.size))


def local_quadratic_peak(U: np.ndarray, y: np.ndarray, window: int) -> Dict[str, Any]:
    """Fit one local quadratic around the discrete maximum."""
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(U) & np.isfinite(y)
    U, y = U[valid], y[valid]
    order = np.argsort(U)
    U, y = U[order], y[order]
    if window not in (3, 5, 7) or U.size < window:
        return {"accepted": False, "reason": "insufficient_points"}
    peak_index = int(np.argmax(y))
    if peak_index == 0 or peak_index == U.size - 1:
        return {"accepted": False, "reason": "boundary", "grid_peak": float(U[peak_index])}
    half = window // 2
    start = min(max(peak_index - half, 0), U.size - window)
    stop = start + window
    x = U[start:stop]
    z = y[start:stop]
    center = float(np.mean(x))
    scale = float(np.ptp(x))
    if scale <= 0:
        return {"accepted": False, "reason": "singular_grid"}
    q = (x - center) / scale
    design = np.column_stack((q * q, q, np.ones_like(q)))
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > 1.0e10:
        return {"accepted": False, "reason": "ill_conditioned", "condition": condition}
    a, b, c = np.linalg.lstsq(design, z, rcond=None)[0]
    curvature = float(2.0 * a / scale**2)
    if not np.isfinite(curvature) or curvature >= 0:
        return {"accepted": False, "reason": "nonnegative_curvature", "curvature": curvature}
    vertex = float(center - scale * b / (2.0 * a))
    if vertex < float(x[0]) or vertex > float(x[-1]):
        return {"accepted": False, "reason": "vertex_outside_window", "peak": vertex}
    value = float(a * ((vertex - center) / scale) ** 2 + b * ((vertex - center) / scale) + c)
    return {
        "accepted": True,
        "reason": "ok",
        "peak": vertex,
        "peak_value": value,
        "curvature": curvature,
        "grid_peak": float(U[peak_index]),
        "fit_U_min": float(x[0]),
        "fit_U_max": float(x[-1]),
        "condition": condition,
    }


def stable_peak(U: np.ndarray, y: np.ndarray, max_spread_steps: float = 1.5) -> Dict[str, Any]:
    """Require stable, concave local fits in 3-, 5- and 7-point windows."""
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    fits = {window: local_quadratic_peak(U, y, window) for window in (3, 5, 7)}
    failed = [fits[w].get("reason", "rejected") for w in (3, 5, 7) if not fits[w].get("accepted")]
    result: Dict[str, Any] = {
        "accepted": False,
        "quality_flag": failed[0] if failed else "unstable_windows",
        "grid_peak": float(U[int(np.nanargmax(y))]) if U.size else float("nan"),
    }
    for window in (3, 5, 7):
        result["peak_w{}".format(window)] = float(fits[window].get("peak", float("nan")))
        result["curvature_w{}".format(window)] = float(fits[window].get("curvature", float("nan")))
    if failed:
        return result
    peaks = np.asarray([fits[w]["peak"] for w in (3, 5, 7)], dtype=float)
    spacing = float(np.median(np.diff(np.sort(np.unique(U))))) if np.unique(U).size > 1 else float("nan")
    spread = float(np.ptp(peaks))
    result["fit_window_spread"] = spread
    result["grid_spacing"] = spacing
    if not np.isfinite(spacing) or spread > max_spread_steps * spacing + 1.0e-12:
        result["quality_flag"] = "unstable_windows"
        return result
    central = fits[5]
    result.update(
        {
            "accepted": True,
            "quality_flag": "ok",
            "peak": float(central["peak"]),
            "peak_value": float(central["peak_value"]),
            "curvature": float(central["curvature"]),
            "fit_U_min": float(central["fit_U_min"]),
            "fit_U_max": float(central["fit_U_max"]),
        }
    )
    return result


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.std(x) <= 1.0e-14 or np.std(y) <= 1.0e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.ptp(x) <= 0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum(residual**2) / total if total > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def common_realizations(frame: pd.DataFrame, mixing: pd.DataFrame) -> np.ndarray:
    sets: List[set[int]] = []
    for _, group in frame.groupby("W_over_t", sort=True):
        sets.append(set(group["realization"].astype(int).unique()))
    for _, group in mixing.groupby("W_over_t", sort=True):
        sets.append(set(group["realization"].astype(int).unique()))
    common = set.intersection(*sets) if sets else set()
    return np.asarray(sorted(common), dtype=int)


def matrices_for_W(
    frame: pd.DataFrame,
    mixing: pd.DataFrame,
    realizations: np.ndarray,
    log_dimension: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    pivots: Dict[str, pd.DataFrame] = {}
    for name, column, source, factor in (
        ("S", "entropy_norm", frame, 1.0),
        ("H_Q", "H_Q", frame, 1.0 / log_dimension),
        ("S_in", "S_intra_Q", frame, 1.0 / log_dimension),
        ("M", "sample_mixing", mixing, 1.0),
    ):
        pivot = source.pivot_table(index="realization", columns="U_over_t", values=column, aggfunc="first")
        pivot = pivot.reindex(realizations)
        pivots[name] = pivot * factor
    common_U = set(float(x) for x in pivots["S"].columns)
    for pivot in pivots.values():
        common_U &= set(float(x) for x in pivot.columns)
    U = np.asarray(sorted(common_U), dtype=float)
    if U.size < 7:
        raise ValueError("Fewer than seven common U points remain")
    matrices = {name: pivot.reindex(columns=U).to_numpy(dtype=float) for name, pivot in pivots.items()}
    for name, matrix in matrices.items():
        if matrix.shape != (realizations.size, U.size) or not np.all(np.isfinite(matrix)):
            raise ValueError("Incomplete realization-U matrix for {}".format(name))
    return U, matrices


def classify_component(metric_rows: pd.DataFrame) -> Dict[str, Any]:
    """Use three transparent votes: peak, curve shape and derivative shape."""
    summary: Dict[str, Any] = {}
    votes = {"H_Q": 0, "S_in": 0}
    available = 0
    for metric, prefer_small in (("peak_distance", True), ("curve_corr", False), ("derivative_corr", False)):
        h = float(np.nanmedian(metric_rows[metric + "_H_Q"]))
        s = float(np.nanmedian(metric_rows[metric + "_S_in"]))
        summary[metric + "_H_Q_median"] = h
        summary[metric + "_S_in_median"] = s
        if not (math.isfinite(h) and math.isfinite(s)) or abs(h - s) <= 1.0e-12:
            continue
        available += 1
        winner = "H_Q" if ((h < s) if prefer_small else (h > s)) else "S_in"
        votes[winner] += 1
    if available == 0 or votes["H_Q"] == votes["S_in"]:
        resonance = "ambiguous"
    else:
        resonance = max(votes, key=votes.get)
    summary.update(
        {
            "resonance_component": resonance,
            "background_component": "S_in" if resonance == "H_Q" else ("H_Q" if resonance == "S_in" else "ambiguous"),
            "votes_H_Q": votes["H_Q"],
            "votes_S_in": votes["S_in"],
            "available_metrics": available,
            "classification_quality": "ok" if available >= 2 and resonance != "ambiguous" else "ambiguous",
        }
    )
    return summary


def bootstrap_slope(W: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Rows are bootstrap replicates; columns are W points."""
    output = np.full(values.shape[0], np.nan)
    for i, row in enumerate(values):
        valid = np.isfinite(W) & np.isfinite(row)
        if np.count_nonzero(valid) >= 3:
            output[i] = linear_fit(W[valid], row[valid])[0]
    return output


def save_figure(fig: plt.Figure, folder: Path, stem: str, dpi: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / (stem + ".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(folder / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze(
    run_dir: Path,
    output_dir: Path,
    bootstrap: int,
    seed: int,
    min_W: float,
    max_W: float,
    fit_W_minima: Sequence[float],
    max_spread_steps: float,
    identity_tol: float,
    only_L: int | None,
    dpi: int,
) -> None:
    observables_path = run_dir / "ed_observables_by_realization.csv"
    mixing_path = run_dir / "sample_mixing_curves.csv"
    if not observables_path.exists() or not mixing_path.exists():
        raise SystemExit(
            "Missing merged inputs. Run merge_results.py first; expected {} and {}".format(
                observables_path, mixing_path
            )
        )
    frame = pd.read_csv(observables_path)
    mixing = pd.read_csv(mixing_path)
    required = set(SECTOR) | {
        "W_over_t", "realization", "U_over_t", "entropy", "entropy_norm", "H_Q", "S_intra_Q"
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit("Merged ED file lacks required columns: " + ", ".join(missing))
    frame = frame[(frame["W_over_t"] >= min_W) & (frame["W_over_t"] <= max_W)].copy()
    mixing = mixing[(mixing["W_over_t"] >= min_W) & (mixing["W_over_t"] <= max_W)].copy()
    if only_L is not None:
        frame = frame[frame["L"] == only_L].copy()
        mixing = mixing[mixing["L"] == only_L].copy()
    if frame.empty:
        raise SystemExit("No ED rows remain after filtering")

    identity = np.abs(frame["entropy"] - frame["H_Q"] - frame["S_intra_Q"])
    recorded = np.abs(frame.get("entropy_identity_error", pd.Series(0.0, index=frame.index)))
    identity_max = float(max(identity.max(), recorded.max()))
    if not math.isfinite(identity_max) or identity_max > identity_tol:
        raise RuntimeError("Entropy identity failed: max error {:.3e} > {:.3e}".format(identity_max, identity_tol))

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    rng = np.random.default_rng(seed)
    curve_rows: List[Dict[str, Any]] = []
    peak_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    classification_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    slope_rows: List[Dict[str, Any]] = []
    eta_rows: List[Dict[str, Any]] = []
    bootstrap_cache: Dict[Tuple[int, int, int, float, str], Dict[str, np.ndarray]] = {}
    curve_cache: Dict[Tuple[int, int, int, float], Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}

    sectors = list(frame.groupby(list(SECTOR), sort=True))
    for sector_index, (sector_key_raw, sector_frame) in enumerate(sectors):
        L, N, nmax = (int(x) for x in sector_key_raw)
        sector_key = (L, N, nmax)
        sector_mixing = mixing[
            (mixing["L"] == L) & (mixing["N"] == N) & (mixing["nmax"] == nmax)
        ]
        realizations = common_realizations(sector_frame, sector_mixing)
        if realizations.size < 3:
            raise RuntimeError("Sector {} has fewer than three common realizations".format(sector_key))
        dimension = core.bounded_composition_dim(L, N, nmax)
        log_dimension = math.log(dimension)
        W_values = sorted(set(sector_frame["W_over_t"]) & set(sector_mixing["W_over_t"]))
        # One paired bootstrap index table is reused for all W and observables.
        draws = rng.integers(0, realizations.size, size=(bootstrap, realizations.size))
        for W in W_values:
            ed_w = sector_frame[np.isclose(sector_frame["W_over_t"], W)]
            mix_w = sector_mixing[np.isclose(sector_mixing["W_over_t"], W)]
            U, matrices = matrices_for_W(ed_w, mix_w, realizations, log_dimension)
            curve_cache[(L, N, nmax, float(W))] = (U, matrices)
            mean_curves = {name: np.mean(matrix, axis=0) for name, matrix in matrices.items()}
            for ui, u in enumerate(U):
                row: Dict[str, Any] = {
                    "L": L, "N": N, "nmax": nmax, "W_over_t": float(W), "U_over_t": float(u),
                    "realizations": int(realizations.size),
                }
                for name, matrix in matrices.items():
                    row[name + "_mean"] = float(np.mean(matrix[:, ui]))
                    row[name + "_sem"] = sem(matrix[:, ui])
                curve_rows.append(row)

            mean_peaks: Dict[str, Dict[str, Any]] = {}
            for name in OBSERVABLES:
                estimate = stable_peak(U, mean_curves[name], max_spread_steps)
                mean_peaks[name] = estimate
                bp = np.full(bootstrap, np.nan)
                bc = np.full(bootstrap, np.nan)
                for replicate in range(bootstrap):
                    boot_curve = np.mean(matrices[name][draws[replicate]], axis=0)
                    result = stable_peak(U, boot_curve, max_spread_steps)
                    if result.get("accepted"):
                        bp[replicate] = float(result["peak"])
                        bc[replicate] = float(result["curvature"])
                bootstrap_cache[(L, N, nmax, float(W), name)] = {"peak": bp, "curvature": bc}
                low, median, high, valid_fraction = finite_quantiles(bp)
                clow, cmedian, chigh, curvature_fraction = finite_quantiles(bc)
                quality = str(estimate.get("quality_flag", "rejected"))
                if estimate.get("accepted") and valid_fraction < 0.5:
                    quality = "unstable_bootstrap"
                peak_rows.append(
                    {
                        "L": L, "N": N, "nmax": nmax, "W_over_t": float(W), "observable": name,
                        "U_peak": float(estimate.get("peak", float("nan"))),
                        "peak_value": float(estimate.get("peak_value", float("nan"))),
                        "curvature": float(estimate.get("curvature", float("nan"))),
                        "quality_flag": quality,
                        "U_peak_ci95_low": low, "U_peak_bootstrap_median": median, "U_peak_ci95_high": high,
                        "bootstrap_valid_fraction": valid_fraction,
                        "curvature_ci95_low": clow, "curvature_bootstrap_median": cmedian,
                        "curvature_ci95_high": chigh, "curvature_bootstrap_valid_fraction": curvature_fraction,
                        "fit_window_spread": float(estimate.get("fit_window_spread", float("nan"))),
                        "U_peak_window3": float(estimate.get("peak_w3", float("nan"))),
                        "U_peak_window5": float(estimate.get("peak_w5", float("nan"))),
                        "U_peak_window7": float(estimate.get("peak_w7", float("nan"))),
                        "realizations": int(realizations.size),
                    }
                )

            M = mean_curves["M"]
            row = {"L": L, "N": N, "nmax": nmax, "W_over_t": float(W)}
            for component in ("H_Q", "S_in"):
                component_curve = mean_curves[component]
                m_peak = mean_peaks["M"]
                c_peak = mean_peaks[component]
                row["peak_distance_" + component] = (
                    abs(float(m_peak["peak"]) - float(c_peak["peak"]))
                    if m_peak.get("accepted") and c_peak.get("accepted") else float("nan")
                )
                row["curve_corr_" + component] = correlation(M, component_curve)
                row["derivative_corr_" + component] = correlation(
                    np.gradient(M, U), np.gradient(component_curve, U)
                )
            metric_rows.append(row)

        sector_metrics = pd.DataFrame([r for r in metric_rows if (r["L"], r["N"], r["nmax"]) == sector_key])
        classification = {"L": L, "N": N, "nmax": nmax, **classify_component(sector_metrics)}
        classification_rows.append(classification)
        resonance = classification["resonance_component"]
        background = classification["background_component"]

        peak_lookup = {
            (float(r["W_over_t"]), str(r["observable"])): r
            for r in peak_rows if (r["L"], r["N"], r["nmax"]) == sector_key
        }
        for W in W_values:
            row: Dict[str, Any] = {
                "L": L, "N": N, "nmax": nmax, "W_over_t": float(W),
                "resonance_component": resonance, "background_component": background,
            }
            for name, label in (("S", "S"), ("M", "M"), (resonance, "R"), (background, "B")):
                source = peak_lookup.get((float(W), str(name))) if name != "ambiguous" else None
                row["U_" + label] = float(source["U_peak"]) if source and source["quality_flag"] == "ok" else float("nan")
                row["kappa_" + label] = -float(source["curvature"]) if source and source["quality_flag"] == "ok" else float("nan")
            if all(math.isfinite(row[x]) for x in ("U_R", "U_B", "kappa_R", "kappa_B")) and row["kappa_R"] > 0 and row["kappa_B"] > 0:
                row["eta_curv"] = row["kappa_R"] / (row["kappa_R"] + row["kappa_B"])
                row["U_pred"] = row["eta_curv"] * row["U_R"] + (1.0 - row["eta_curv"]) * row["U_B"]
            else:
                row["eta_curv"] = row["U_pred"] = float("nan")
            row["eta_empirical"] = (
                (row["U_S"] - row["U_B"]) / (row["U_R"] - row["U_B"])
                if all(math.isfinite(row[x]) for x in ("U_S", "U_R", "U_B")) and abs(row["U_R"] - row["U_B"]) > 1.0e-12
                else float("nan")
            )
            row["prediction_error"] = row["U_pred"] - row["U_S"] if math.isfinite(row["U_pred"]) and math.isfinite(row["U_S"]) else float("nan")
            row["naive_M_error"] = row["U_M"] - row["U_S"] if math.isfinite(row["U_M"]) and math.isfinite(row["U_S"]) else float("nan")
            row["U0_mirror"] = 2.0 * row["U_S"] - row["U_M"] if math.isfinite(row["U_M"]) and math.isfinite(row["U_S"]) else float("nan")

            # Bootstrap the derived quantities without changing the selected component.
            derived: Dict[str, np.ndarray] = {}
            for name, label in (("S", "S"), ("M", "M"), (resonance, "R"), (background, "B")):
                if name == "ambiguous":
                    derived[label] = np.full(bootstrap, np.nan)
                    derived["k" + label] = np.full(bootstrap, np.nan)
                else:
                    cached = bootstrap_cache[(L, N, nmax, float(W), str(name))]
                    derived[label] = cached["peak"]
                    derived["k" + label] = -cached["curvature"]
            denom = derived["kR"] + derived["kB"]
            valid_pred = (
                np.isfinite(derived["R"]) & np.isfinite(derived["B"]) & np.isfinite(denom)
                & (derived["kR"] > 0) & (derived["kB"] > 0) & (denom > 0)
            )
            pred = np.full(bootstrap, np.nan)
            eta_curv = np.full(bootstrap, np.nan)
            eta_curv[valid_pred] = derived["kR"][valid_pred] / denom[valid_pred]
            pred[valid_pred] = eta_curv[valid_pred] * derived["R"][valid_pred] + (1.0 - eta_curv[valid_pred]) * derived["B"][valid_pred]
            mirror = 2.0 * derived["S"] - derived["M"]
            empirical = np.full(bootstrap, np.nan)
            empirical_denom = derived["R"] - derived["B"]
            valid_empirical = np.isfinite(derived["S"] + derived["R"] + derived["B"]) & (np.abs(empirical_denom) > 1.0e-12)
            empirical[valid_empirical] = (derived["S"][valid_empirical] - derived["B"][valid_empirical]) / empirical_denom[valid_empirical]
            for name, values in (("U_pred", pred), ("eta_curv", eta_curv), ("eta_empirical", empirical), ("U0_mirror", mirror)):
                low, median, high, fraction = finite_quantiles(values)
                row[name + "_ci95_low"] = low
                row[name + "_bootstrap_median"] = median
                row[name + "_ci95_high"] = high
                row[name + "_bootstrap_valid_fraction"] = fraction
                bootstrap_cache[(L, N, nmax, float(W), name)] = {"peak": values, "curvature": np.full(bootstrap, np.nan)}
            prediction_rows.append(row)

        sector_predictions = pd.DataFrame(
            [r for r in prediction_rows if (r["L"], r["N"], r["nmax"]) == sector_key]
        ).sort_values("W_over_t")
        for W_min in fit_W_minima:
            selected = sector_predictions[sector_predictions["W_over_t"] >= W_min]
            W = selected["W_over_t"].to_numpy(dtype=float)
            for label, column in (
                ("S", "U_S"), ("M", "U_M"), ("R", "U_R"), ("B", "U_B"),
                ("prediction", "U_pred"), ("mirror", "U0_mirror"),
            ):
                y = selected[column].to_numpy(dtype=float)
                slope, intercept, r2 = linear_fit(W, y)
                cache_name = {
                    "S": "S",
                    "M": "M",
                    "R": resonance,
                    "B": background,
                    "prediction": "U_pred",
                    "mirror": "U0_mirror",
                }[label]
                boot_values = np.column_stack(
                    [bootstrap_cache[(L, N, nmax, float(w), cache_name)]["peak"] for w in W]
                ) if W.size else np.empty((bootstrap, 0))
                boot_slopes = bootstrap_slope(W, boot_values) if W.size else np.full(bootstrap, np.nan)
                low, median, high, fraction = finite_quantiles(boot_slopes)
                slope_rows.append(
                    {
                        "L": L, "N": N, "nmax": nmax, "W_fit_min": W_min, "W_fit_max": max_W,
                        "observable": label, "points": int(np.count_nonzero(np.isfinite(y))),
                        "slope": slope, "intercept": intercept, "r2": r2,
                        "slope_ci95_low": low, "slope_bootstrap_median": median,
                        "slope_ci95_high": high, "bootstrap_valid_fraction": fraction,
                    }
                )
            valid = np.isfinite(selected["U_S"] + selected["U_R"] + selected["U_B"])
            delta = (selected.loc[valid, "U_R"] - selected.loc[valid, "U_B"]).to_numpy(dtype=float)
            target = (selected.loc[valid, "U_S"] - selected.loc[valid, "U_B"]).to_numpy(dtype=float)
            eta_fit = float(np.dot(delta, target) / np.dot(delta, delta)) if delta.size >= 3 and np.dot(delta, delta) > 0 else float("nan")
            eta_curv_mean = float(np.nanmean(selected["eta_curv"])) if np.any(np.isfinite(selected["eta_curv"])) else float("nan")
            eta_rows.append(
                {
                    "L": L, "N": N, "nmax": nmax, "W_fit_min": W_min, "W_fit_max": max_W,
                    "eta_fit": eta_fit, "eta_curv_mean": eta_curv_mean,
                    "difference": eta_curv_mean - eta_fit if math.isfinite(eta_fit) and math.isfinite(eta_curv_mean) else float("nan"),
                    "points": int(np.count_nonzero(valid)),
                }
            )

        # Figures: absolute peak comparison, prediction residual, mirror test, and shape comparison.
        p = sector_predictions
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        for column, label, marker in (("U_S", r"$U_S^*$", "s"), ("U_M", r"$U_M^*$", "^"),
                                      ("U_R", r"$U_R^*$", "o"), ("U_B", r"$U_B^*$", "D"),
                                      ("U_pred", r"$U_{pred}^*$", "x")):
            valid = np.isfinite(p[column])
            ax.plot(p.loc[valid, "W_over_t"], p.loc[valid, column], marker=marker, label=label)
        ax.set(xlabel=r"$W/t$", ylabel=r"$U^*/t$", title=rf"$L={L},\ N={N},\ n_{{max}}={nmax}$")
        ax.legend(frameon=False, ncol=2)
        save_figure(fig, figures_dir, f"peak_comparison_L{L}_N{N}_nmax{nmax}", dpi)

        fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        valid = np.isfinite(p["U_S"])
        axes[0].plot(p.loc[valid, "W_over_t"], p.loc[valid, "U_S"], "ks-", label=r"$U_S^*$ (ED)")
        valid_pred = np.isfinite(p["U_pred"])
        axes[0].plot(p.loc[valid_pred, "W_over_t"], p.loc[valid_pred, "U_pred"], "C3o--", label=r"curvature prediction")
        valid_m = np.isfinite(p["U_M"])
        axes[0].plot(p.loc[valid_m, "W_over_t"], p.loc[valid_m, "U_M"], "C0^:", label=r"naive $U_M^*$")
        axes[0].set_ylabel(r"$U^*/t$")
        axes[0].legend(frameon=False)
        axes[1].axhline(0.0, color="0.5", lw=1)
        axes[1].plot(p["W_over_t"], p["prediction_error"], "C3o-", label="prediction")
        axes[1].plot(p["W_over_t"], p["naive_M_error"], "C0^-", label="naive M")
        axes[1].set(xlabel=r"$W/t$", ylabel=r"error in $U^*/t$")
        axes[0].set_title(rf"$L={L},\ N={N},\ n_{{max}}={nmax}$")
        save_figure(fig, figures_dir, f"curvature_prediction_L{L}_N{N}_nmax{nmax}", dpi)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        valid = np.isfinite(p["U0_mirror"])
        ax.plot(p.loc[valid, "W_over_t"], p.loc[valid, "U0_mirror"], "o-")
        if np.count_nonzero(valid):
            ax.axhline(float(np.nanmean(p["U0_mirror"])), color="C3", ls="--", label="mean")
        ax.set(xlabel=r"$W/t$", ylabel=r"$U_0^{mirror}/t=2U_S^*/t-U_M^*/t$",
               title=rf"$L={L},\ N={N},\ n_{{max}}={nmax}$")
        ax.legend(frameon=False)
        save_figure(fig, figures_dir, f"mirror_test_L{L}_N{N}_nmax{nmax}", dpi)

        chosen_W = []
        for target in (0.8, 1.2, 1.7, 2.1, 2.5):
            if W_values:
                candidate = min(W_values, key=lambda x: abs(float(x) - target))
                if candidate not in chosen_W:
                    chosen_W.append(candidate)
        fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.4), sharex=False, sharey=True)
        for ax, W in zip(axes.flat, chosen_W):
            U, matrices = curve_cache[(L, N, nmax, float(W))]
            for name, color in zip(OBSERVABLES, ("k", "C1", "C2", "C0")):
                values = np.mean(matrices[name], axis=0)
                span = float(np.ptp(values))
                normalized = (values - np.min(values)) / span if span > 1.0e-14 else np.zeros_like(values)
                ax.plot(U, normalized, color=color, label=name)
            ax.set_title(rf"$W/t={W:g}$")
            ax.set_xlabel(r"$U/t$")
        for ax in axes[:, 0]:
            ax.set_ylabel("rescaled curve")
        for ax in axes.flat[len(chosen_W):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        fig.suptitle(rf"$L={L},\ N={N},\ n_{{max}}={nmax}$", y=1.02)
        fig.tight_layout()
        save_figure(fig, figures_dir, f"component_shapes_L{L}_N{N}_nmax{nmax}", dpi)

    curves_df = pd.DataFrame(curve_rows)
    peaks_df = pd.DataFrame(peak_rows)
    metrics_df = pd.DataFrame(metric_rows)
    classifications_df = pd.DataFrame(classification_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    slopes_df = pd.DataFrame(slope_rows)
    eta_df = pd.DataFrame(eta_rows)
    for name, table in (
        ("decomposition_curves.csv", curves_df),
        ("decomposition_peaks.csv", peaks_df),
        ("component_similarity_metrics.csv", metrics_df),
        ("component_classification.csv", classifications_df),
        ("curvature_prediction.csv", predictions_df),
        ("slope_robustness.csv", slopes_df),
        ("eta_comparison.csv", eta_df),
    ):
        table.to_csv(output_dir / name, index=False)

    summary_lines = [
        "Two-component entropy-mechanism verification v{}".format(VERSION),
        "Run directory: {}".format(run_dir),
        "Maximum |S-H_Q-S_in|: {:.3e} (tolerance {:.3e})".format(identity_max, identity_tol),
        "Paired bootstrap replicates: {}".format(bootstrap),
        "W/t range: {:.6g}..{:.6g}".format(min_W, max_W),
        "",
    ]
    for classification in classification_rows:
        key = (classification["L"], classification["N"], classification["nmax"])
        group = predictions_df[
            (predictions_df["L"] == key[0]) & (predictions_df["N"] == key[1]) & (predictions_df["nmax"] == key[2])
        ]
        pred_error = group["prediction_error"].to_numpy(dtype=float)
        naive_error = group["naive_M_error"].to_numpy(dtype=float)
        pred_rmse = float(np.sqrt(np.nanmean(pred_error**2))) if np.any(np.isfinite(pred_error)) else float("nan")
        naive_rmse = float(np.sqrt(np.nanmean(naive_error**2))) if np.any(np.isfinite(naive_error)) else float("nan")
        summary_lines.append(
            "L={} N={} nmax={}: S_R={}, votes H_Q:S_in={}:{}, valid predictions={}/{}, "
            "RMSE(pred)={:.6g}, RMSE(naive M)={:.6g}".format(
                *key, classification["resonance_component"], classification["votes_H_Q"],
                classification["votes_S_in"], int(np.count_nonzero(np.isfinite(group["U_pred"]))), len(group),
                pred_rmse, naive_rmse,
            )
        )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    metadata = {
        "analysis_version": VERSION,
        "run_dir": str(run_dir),
        "bootstrap": bootstrap,
        "bootstrap_seed": seed,
        "min_W": min_W,
        "max_W": max_W,
        "fit_W_minima": list(fit_W_minima),
        "max_spread_steps": max_spread_steps,
        "entropy_identity_tolerance": identity_tol,
        "entropy_identity_max_error": identity_max,
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print("Results: {}".format(output_dir))


def parse_float_list(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run containing merged realization-level CSV files")
    parser.add_argument("--output-dir", type=Path, help="default: RUN/entropy_mechanism_verification")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260913)
    parser.add_argument("--min-W", type=float, default=0.8)
    parser.add_argument("--max-W", type=float, default=2.5)
    parser.add_argument("--fit-W-minima", type=parse_float_list, default=[0.8, 1.0, 1.2])
    parser.add_argument("--max-peak-spread-steps", type=float, default=1.5)
    parser.add_argument("--identity-tol", type=float, default=1.0e-10)
    parser.add_argument("--L", type=int, help="optional size filter")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if args.max_W < args.min_W:
        parser.error("--max-W must be at least --min-W")
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "entropy_mechanism_verification"
    analyze(
        run_dir, output_dir, args.bootstrap, args.bootstrap_seed, args.min_W, args.max_W,
        args.fit_W_minima, args.max_peak_spread_steps, args.identity_tol, args.L, args.dpi,
    )


if __name__ == "__main__":
    main()
