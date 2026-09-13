#!/usr/bin/env python3
"""Fit and validate the reduced two-component model for entropy maxima.

The script consumes results produced by verify_entropy_mechanism.py.  All model
parameters are fitted only to the component peaks and curvature weights; the ED
entropy peak U_S is reserved for validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERSION = "1.0.0"
W_MINIMA = (0.8, 1.0, 1.2, 1.4)
KEYS = ("L", "N", "nmax")


def finite_rows(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        mask &= np.isfinite(frame[column].to_numpy(dtype=float))
    return frame.loc[mask].copy()


def line_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2 or np.ptp(x[valid]) <= 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    return float(intercept), float(slope)


def eta_fit(w: np.ndarray, eta: np.ndarray, model: str) -> Tuple[float, float]:
    valid = np.isfinite(w) & np.isfinite(eta) & (w > 0)
    if np.count_nonzero(valid) < (2 if model == "M1" else 1):
        return float("nan"), float("nan")
    if model == "M0":
        return float(np.mean(eta[valid])), 0.0
    design = np.column_stack((np.ones(np.count_nonzero(valid)), 1.0 / w[valid]))
    eta_inf, d = np.linalg.lstsq(design, eta[valid], rcond=None)[0]
    return float(eta_inf), float(d)


def fit_components(frame: pd.DataFrame, model: str) -> Dict[str, float]:
    data = finite_rows(frame, ("W_over_t", "U_R", "U_B", "eta_curv"))
    w = data["W_over_t"].to_numpy(dtype=float)
    a_r, b_r = line_fit(w, data["U_R"].to_numpy(dtype=float))
    a_b, b_b = line_fit(w, data["U_B"].to_numpy(dtype=float))
    eta_inf, d = eta_fit(w, data["eta_curv"].to_numpy(dtype=float), model)
    A = eta_inf * a_r + (1.0 - eta_inf) * a_b + d * (b_r - b_b)
    B = eta_inf * b_r + (1.0 - eta_inf) * b_b
    C = d * (a_r - a_b)
    return {
        "a_R": a_r, "b_R": b_r, "a_B": a_b, "b_B": b_b,
        "eta_inf": eta_inf, "d": d, "A": A, "B": B, "C": C,
        "n_fit": float(len(data)),
    }


def predict_formula(w: np.ndarray, parameters: Mapping[str, float]) -> np.ndarray:
    return parameters["A"] + parameters["B"] * w + parameters["C"] / w


def predict_composed(w: np.ndarray, parameters: Mapping[str, float]) -> np.ndarray:
    u_r = parameters["a_R"] + parameters["b_R"] * w
    u_b = parameters["a_B"] + parameters["b_B"] * w
    eta = parameters["eta_inf"] + parameters["d"] / w
    return eta * u_r + (1.0 - eta) * u_b


def metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(valid):
        return {"rmse": float("nan"), "bias": float("nan"), "max_abs_error": float("nan"), "n_eval": 0}
    residual = predicted[valid] - observed[valid]
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
        "max_abs_error": float(np.max(np.abs(residual))),
        "n_eval": int(len(residual)),
    }


def loocv(frame: pd.DataFrame, model: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    component_data = finite_rows(frame, ("W_over_t", "U_R", "U_B", "eta_curv"))
    targets = finite_rows(frame, ("W_over_t", "U_S"))
    for _, held in targets.iterrows():
        held_w = float(held["W_over_t"])
        train = component_data[~np.isclose(component_data["W_over_t"], held_w)]
        parameters = fit_components(train, model)
        w = held_w
        predicted = float(predict_formula(np.array([w]), parameters)[0])
        rows.append({"W_over_t": w, "U_S": float(held["U_S"]), "U_pred": predicted,
                     "residual": predicted - float(held["U_S"])})
    result = pd.DataFrame(rows)
    score = metrics(
        result["U_S"].to_numpy(dtype=float) if len(result) else np.array([]),
        result["U_pred"].to_numpy(dtype=float) if len(result) else np.array([]),
    )
    return result, score


def quantile(values: Iterable[float], probability: float) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability)) if array.size else float("nan")


def bootstrap_parameters(frame: pd.DataFrame, model: str, w_min: float) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    selected = frame[frame["W_over_t"] >= w_min]
    for replicate, group in selected.groupby("replicate"):
        parameters = fit_components(group, model)
        if all(math.isfinite(parameters[name]) for name in ("A", "B", "C", "eta_inf", "d")):
            rows.append({"replicate": int(replicate), **parameters})
    return pd.DataFrame(rows)


def style() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8.0, "axes.labelsize": 8.5, "axes.titlesize": 9.0,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, directory: Path, stem: str, dpi: int) -> None:
    fig.savefig(directory / (stem + ".png"), dpi=dpi)
    fig.savefig(directory / (stem + ".pdf"))
    plt.close(fig)


def read_verification(directory: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    peaks_path = directory / "curvature_prediction.csv"
    bootstrap_path = directory / "bootstrap_peaks.csv.gz"
    if not peaks_path.is_file():
        raise FileNotFoundError("missing {}".format(peaks_path))
    if not bootstrap_path.is_file():
        raise FileNotFoundError(
            "missing {}; rerun verify_entropy_mechanism.py with the current version".format(bootstrap_path)
        )
    peaks = pd.read_csv(peaks_path)
    bootstrap = pd.read_csv(bootstrap_path)
    peaks["source_dir"] = str(directory)
    bootstrap["source_dir"] = str(directory)
    return peaks, bootstrap


def sector_label(key: Tuple[int, int, int]) -> str:
    return "L{}_N{}_nmax{}".format(*key)


def analyze(inputs: Sequence[Path], output: Path, dpi: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    style()

    peak_tables, bootstrap_tables = zip(*(read_verification(path.resolve()) for path in inputs))
    peaks = pd.concat(peak_tables, ignore_index=True)
    boot = pd.concat(bootstrap_tables, ignore_index=True)
    peaks.to_csv(output / "entropy_component_peaks.csv", index=False)

    parameter_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    loo_rows: List[Dict[str, object]] = []
    slope_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    selected_models: Dict[Tuple[int, int, int], Tuple[str, Dict[str, float]]] = {}

    for raw_key, sector in peaks.groupby(list(KEYS)):
        key = tuple(int(value) for value in raw_key)
        sector_boot = boot[(boot["L"] == key[0]) & (boot["N"] == key[1]) & (boot["nmax"] == key[2])]
        model_results: Dict[Tuple[float, str], Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]] = {}
        for w_min in W_MINIMA:
            selected = sector[sector["W_over_t"] >= w_min].sort_values("W_over_t")
            for model in ("M0", "M1"):
                parameters = fit_components(selected, model)
                loo, score = loocv(selected, model)
                boot_parameters = bootstrap_parameters(sector_boot, model, w_min)
                d_low = quantile(boot_parameters["d"] if "d" in boot_parameters else [], 0.025)
                d_high = quantile(boot_parameters["d"] if "d" in boot_parameters else [], 0.975)
                eta_grid = parameters["eta_inf"] + parameters["d"] / selected["W_over_t"].to_numpy(dtype=float)
                eta_physical = bool(eta_grid.size and np.all((eta_grid >= 0.0) & (eta_grid <= 1.0)))
                row: Dict[str, object] = {**dict(zip(KEYS, key)), "W_fit_min": w_min, "model": model,
                    **parameters, **{"loocv_" + name: value for name, value in score.items()},
                    "d_ci95_low": d_low, "d_ci95_high": d_high,
                    "bootstrap_valid": len(boot_parameters), "eta_physical": eta_physical}
                for name in ("A", "B", "C", "eta_inf", "d"):
                    row[name + "_ci95_low"] = quantile(boot_parameters[name] if name in boot_parameters else [], 0.025)
                    row[name + "_ci95_high"] = quantile(boot_parameters[name] if name in boot_parameters else [], 0.975)
                parameter_rows.append(row)
                loo.insert(0, "model", model)
                loo.insert(0, "W_fit_min", w_min)
                for column, value in reversed(list(zip(KEYS, key))):
                    loo.insert(0, column, value)
                loo_rows.extend(loo.to_dict("records"))
                model_results[(w_min, model)] = (parameters, score, boot_parameters)

        base_m0 = model_results[(1.0, "M0")]
        base_m1 = model_results[(1.0, "M1")]
        improvement = (
            (base_m0[1]["rmse"] - base_m1[1]["rmse"]) / base_m0[1]["rmse"]
            if math.isfinite(base_m0[1]["rmse"]) and base_m0[1]["rmse"] > 0 else float("nan")
        )
        d_low = quantile(base_m1[2]["d"] if "d" in base_m1[2] else [], 0.025)
        d_high = quantile(base_m1[2]["d"] if "d" in base_m1[2] else [], 0.975)
        d_stable = math.isfinite(d_low) and math.isfinite(d_high) and (d_low > 0 or d_high < 0)
        w_grid = sector.loc[sector["W_over_t"] >= 1.0, "W_over_t"].to_numpy(dtype=float)
        eta_grid = base_m1[0]["eta_inf"] + base_m1[0]["d"] / w_grid
        eta_physical = bool(eta_grid.size and np.all((eta_grid >= 0) & (eta_grid <= 1)))
        chosen = "M1" if math.isfinite(improvement) and improvement >= 0.10 and d_stable and eta_physical else "M0"
        parameters, score, boot_parameters = model_results[(1.0, chosen)]
        selected_models[key] = (chosen, parameters)

        sector = sector.sort_values("W_over_t").copy()
        w = sector["W_over_t"].to_numpy(dtype=float)
        predicted = predict_formula(w, parameters)
        composed = predict_composed(w, parameters)
        if np.nanmax(np.abs(predicted - composed)) > 1.0e-10:
            raise RuntimeError("analytic reduction identity failed for {}".format(key))
        boot_prediction = np.vstack([predict_formula(w, row) for _, row in boot_parameters.iterrows()]) if len(boot_parameters) else np.empty((0, len(w)))
        for position, (_, source) in enumerate(sector.iterrows()):
            low = quantile(boot_prediction[:, position] if len(boot_prediction) else [], 0.025)
            high = quantile(boot_prediction[:, position] if len(boot_prediction) else [], 0.975)
            observed = float(source["U_S"])
            prediction_rows.append({**dict(zip(KEYS, key)), "W_over_t": float(source["W_over_t"]),
                "selected_model": chosen, "U_S": observed, "U_M": float(source["U_M"]),
                "U_R": float(source["U_R"]), "U_B": float(source["U_B"]),
                "eta_curv": float(source["eta_curv"]), "U_reduced": float(predicted[position]),
                "U_reduced_ci95_low": low, "U_reduced_ci95_high": high,
                "residual": float(predicted[position] - observed),
                "covered_by_model_ci": bool(math.isfinite(observed) and math.isfinite(low) and low <= observed <= high)})

        direct_score = metrics(sector["U_S"].to_numpy(dtype=float), predicted)
        evaluable = [row for row in prediction_rows if tuple(int(row[name]) for name in KEYS) == key
                     and math.isfinite(float(row["U_S"])) and math.isfinite(float(row["U_reduced_ci95_low"]))]
        model_coverage = float(np.mean([bool(row["covered_by_model_ci"]) for row in evaluable])) if evaluable else float("nan")
        entropy_halfwidth = 0.5 * (
            sector["U_S_ci95_high"].to_numpy(dtype=float) - sector["U_S_ci95_low"].to_numpy(dtype=float)
        )
        metric_rows.append({**dict(zip(KEYS, key)), "W_fit_min": 1.0, "selected_model": chosen,
            "M1_loocv_improvement": improvement, "M1_d_stable": d_stable, "M1_eta_physical": eta_physical,
            **{"direct_" + name: value for name, value in direct_score.items()},
            **{"loocv_" + name: value for name, value in score.items()},
            "bootstrap_ci_coverage": model_coverage,
            "median_ED_ci_halfwidth": float(np.nanmedian(entropy_halfwidth)),
            "loocv_within_ED_scale": bool(math.isfinite(score["rmse"]) and score["rmse"] <= float(np.nanmedian(entropy_halfwidth)))})

        valid_slope = finite_rows(sector[sector["W_over_t"] >= 1.0], ("W_over_t", "U_S", "U_M", "U_R", "U_B"))
        ws = valid_slope["W_over_t"].to_numpy(dtype=float)
        _, c_s = line_fit(ws, valid_slope["U_S"].to_numpy(dtype=float))
        _, c_m = line_fit(ws, valid_slope["U_M"].to_numpy(dtype=float))
        _, beta_r = line_fit(valid_slope["U_M"].to_numpy(dtype=float), valid_slope["U_R"].to_numpy(dtype=float))
        _, beta_b = line_fit(valid_slope["U_M"].to_numpy(dtype=float), valid_slope["U_B"].to_numpy(dtype=float))
        beta_eff = parameters["eta_inf"] * beta_r + (1.0 - parameters["eta_inf"]) * beta_b
        slope_rows.append({**dict(zip(KEYS, key)), "selected_model": chosen, "c_S": c_s, "c_M": c_m,
            "c_S_over_c_M": c_s / c_m if c_m else float("nan"), "beta_R": beta_r, "beta_B": beta_b,
            "beta_eff": beta_eff, "difference": beta_eff - c_s / c_m if c_m else float("nan"),
            "B_model": parameters["B"], "B_minus_c_S": parameters["B"] - c_s})

        fig, ax = plt.subplots(figsize=(3.45, 2.65))
        ax.errorbar(w, sector["U_S"], yerr=np.vstack((sector["U_S"] - sector["U_S_ci95_low"], sector["U_S_ci95_high"] - sector["U_S"])),
                    fmt="ko", ms=3.2, capsize=1.5, lw=0.7, label=r"$U_S^*$ (ED)")
        ax.plot(w, predicted, color="#D55E00", lw=1.2, label="reduced " + chosen)
        if len(boot_prediction):
            ax.fill_between(w, np.nanquantile(boot_prediction, 0.025, axis=0), np.nanquantile(boot_prediction, 0.975, axis=0), color="#D55E00", alpha=0.18, lw=0)
        ax.plot(w, sector["U_R"], "s--", color="#0072B2", ms=2.6, lw=0.8, label=r"$U_R^*$")
        ax.plot(w, sector["U_B"], "D:", color="#009E73", ms=2.4, lw=0.8, label=r"$U_B^*$")
        ax.set(xlabel=r"$W/t$", ylabel=r"$U^*/t$")
        ax.text(0.97, 0.04, r"$L={},\ N={},\ n_{{\max}}={}$".format(*key), transform=ax.transAxes, ha="right")
        ax.legend(frameon=False, ncol=2)
        save(fig, figures, "reduced_model_" + sector_label(key), dpi)

        fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.7), sharex=True)
        eta_model = parameters["eta_inf"] + parameters["d"] / w
        axes[0].plot(w, sector["eta_curv"], "ko", ms=3, label=r"$\eta_{\rm curv}$")
        axes[0].plot(w, eta_model, color="#D55E00", label=chosen)
        axes[0].set_ylabel(r"$\eta$")
        axes[0].legend(frameon=False)
        axes[1].axhline(0, color="0.5", lw=0.7)
        axes[1].plot(w, predicted - sector["U_S"].to_numpy(dtype=float), "o-", color="#D55E00", ms=3)
        axes[1].set(xlabel=r"$W/t$", ylabel="residual")
        save(fig, figures, "eta_residuals_" + sector_label(key), dpi)

    parameters_df = pd.DataFrame(parameter_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    loo_df = pd.DataFrame(loo_rows)
    slopes_df = pd.DataFrame(slope_rows)
    metrics_df = pd.DataFrame(metric_rows)

    transfer_rows: List[Dict[str, object]] = []
    l7_models = {key[1:]: value for key, value in selected_models.items() if key[0] == 7}
    for raw_key, sector in peaks[peaks["L"] == 8].groupby(list(KEYS)):
        key = tuple(int(value) for value in raw_key)
        if key[1:] not in l7_models:
            continue
        model, parameters = l7_models[key[1:]]
        for _, row in sector.iterrows():
            w = float(row["W_over_t"])
            strict = float(predict_formula(np.array([w]), parameters)[0])
            eta = parameters["eta_inf"] + parameters["d"] / w
            weighted = eta * float(row["U_R"]) + (1.0 - eta) * float(row["U_B"])
            transfer_rows.append({**dict(zip(KEYS, key)), "W_over_t": w, "source_L": 7,
                "model": model, "U_S": float(row["U_S"]), "U_transfer_strict": strict,
                "U_transfer_weight_only": weighted, "strict_error": strict - float(row["U_S"]),
                "weight_only_error": weighted - float(row["U_S"])})
    transfer_df = pd.DataFrame(transfer_rows)
    transfer_metric_rows: List[Dict[str, object]] = []
    if len(transfer_df):
        for raw_key, group in transfer_df.groupby(list(KEYS)):
            key = tuple(int(value) for value in raw_key)
            for label, column in (("strict_formula", "U_transfer_strict"), ("weight_only", "U_transfer_weight_only")):
                transfer_metric_rows.append({**dict(zip(KEYS, key)), "transfer": label,
                    **metrics(group["U_S"].to_numpy(dtype=float), group[column].to_numpy(dtype=float))})
    transfer_metrics_df = pd.DataFrame(transfer_metric_rows)

    parameters_df.to_csv(output / "reduced_model_parameters.csv", index=False)
    predictions_df.to_csv(output / "reduced_model_predictions.csv", index=False)
    loo_df.to_csv(output / "loocv_predictions.csv", index=False)
    slopes_df.to_csv(output / "slope_comparison.csv", index=False)
    metrics_df.to_csv(output / "reduced_model_metrics.csv", index=False)
    transfer_df.to_csv(output / "leave_one_system_out_L7_to_L8.csv", index=False)
    transfer_metrics_df.to_csv(output / "leave_one_system_out_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    x = np.arange(len(slopes_df))
    ax.axhline(0.5, color="0.5", ls="--", lw=0.8, label=r"$1/2$")
    ax.plot(x, slopes_df["c_S_over_c_M"], "ko", label=r"$c_S/c_M$")
    ax.plot(x, slopes_df["beta_eff"], "s", color="#D55E00", label=r"$\beta_{\rm eff}$")
    ax.set_xticks(x, [sector_label((int(r.L), int(r.N), int(r.nmax))) for r in slopes_df.itertuples()], rotation=55, ha="right")
    ax.set_ylabel("slope ratio")
    ax.legend(frameon=False)
    save(fig, figures, "slope_ratio_summary", dpi)

    coverage_values = metrics_df["bootstrap_ci_coverage"].to_numpy(dtype=float) if len(metrics_df) else np.array([])
    coverage = float(np.nanmean(coverage_values)) if np.any(np.isfinite(coverage_values)) else float("nan")
    lines = [
        "# Reduced entropy-peak model report", "", "Version: {}".format(VERSION),
        "", "The fit used only component peaks and curvature weights. ED entropy peaks were validation targets.",
        "", "Overall bootstrap-interval coverage: {:.1%}.".format(coverage), "", "## Selected models", "",
        "| L | N | nmax | model | LOOCV improvement M1 vs M0 |", "|---:|---:|---:|:---:|---:|",
    ]
    for key, (model, _) in sorted(selected_models.items()):
        sector_params = parameters_df[(parameters_df["L"] == key[0]) & (parameters_df["N"] == key[1]) & (parameters_df["nmax"] == key[2]) & (parameters_df["W_fit_min"] == 1.0)]
        r0 = float(sector_params.loc[sector_params["model"] == "M0", "loocv_rmse"].iloc[0])
        r1 = float(sector_params.loc[sector_params["model"] == "M1", "loocv_rmse"].iloc[0])
        improvement = (r0 - r1) / r0 if r0 > 0 else float("nan")
        lines.append("| {} | {} | {} | {} | {:.1%} |".format(*key, model, improvement))
    lines += ["", "M1 was selected only when LOOCV improved by at least 10%, d was bootstrap-stable, and eta stayed in [0,1].", ""]
    lines += ["## Diagnostic conclusions", ""]
    for row in metrics_df.itertuples():
        slope = slopes_df[(slopes_df["L"] == row.L) & (slopes_df["N"] == row.N) & (slopes_df["nmax"] == row.nmax)].iloc[0]
        slope_ok = abs(float(slope["B_minus_c_S"])) <= max(float(row.median_ED_ci_halfwidth), 0.02 * max(abs(float(slope["c_S"])), 1.0e-12))
        lines.append(
            "- L={} N={} nmax={}: LOOCV RMSE={:.4g}, median ED half-width={:.4g}, "
            "coverage={:.1%}, B-c_S={:.4g}; {}.".format(
                int(row.L), int(row.N), int(row.nmax), float(row.loocv_rmse),
                float(row.median_ED_ci_halfwidth), float(row.bootstrap_ci_coverage),
                float(slope["B_minus_c_S"]),
                "passes the local diagnostics" if bool(row.loocv_within_ED_scale) and slope_ok else "does not pass all diagnostics",
            )
        )
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    tex = "\\section*{{Reduced entropy-peak model}}\nThe component-only fit was validated against ED entropy maxima.\\\\\nBootstrap interval coverage: {:.1f}\\%.\n".format(100 * coverage)
    (output / "report.tex").write_text(tex, encoding="utf-8")
    metadata = {"version": VERSION, "inputs": [str(path.resolve()) for path in inputs], "W_minima": W_MINIMA}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Reduced entropy model v{}".format(VERSION))
    print("Sectors: {}; selected M1: {}".format(len(selected_models), sum(value[0] == "M1" for value in selected_models.values())))
    print("Results: {}".format(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    analyze(args.verification_dirs, args.output_dir.resolve(), args.dpi)


if __name__ == "__main__":
    main()
