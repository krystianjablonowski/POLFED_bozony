#!/usr/bin/env python3
"""Cluster runner and analysis for the covariance-compensation model."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from covariance_core import (
    VERSION,
    build_structure,
    covariance_observables,
    disorder_eta,
    selected_u0_states,
)


PREDICTORS = ("U_mean", "U_pool", "U_edge_mean", "U_edge_pool")


def load_config(path: str) -> Dict[str, object]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = str(config_path)
    validate_config(cfg)
    return cfg


def config_hash(cfg: Dict[str, object]) -> str:
    clean = {key: value for key, value in cfg.items() if not key.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def project_root(cfg: Dict[str, object]) -> Path:
    return Path(str(cfg["_config_path"])).parent


def result_root(cfg: Dict[str, object]) -> Path:
    path = Path(str(cfg["paths"]["results_dir"]))
    return path if path.is_absolute() else project_root(cfg) / path


def values_from_spec(spec: object) -> List[float]:
    if isinstance(spec, list):
        return [float(value) for value in spec]
    if not isinstance(spec, dict):
        raise ValueError("W_values must be a list or {start, stop, step}")
    start = float(spec["start"])
    stop = float(spec["stop"])
    step = float(spec["step"])
    if step <= 0 or stop < start:
        raise ValueError("Invalid W range")
    count = int(round((stop - start) / step))
    values = [start + index * step for index in range(count + 1)]
    if values[-1] < stop - 1.0e-10:
        values.append(stop)
    return [float(round(value, 12)) for value in values]


def sectors(cfg: Dict[str, object]) -> List[Dict[str, int]]:
    return [
        {"L": int(item["L"]), "N": int(item["N"]), "nmax": int(item["nmax"])}
        for item in cfg["model"]["sectors"]
    ]


def validate_config(cfg: Dict[str, object]) -> None:
    if str(cfg["model"].get("boundary", "")) != "open":
        raise ValueError("Only open boundary conditions match this study")
    if str(cfg["model"].get("interaction", "")) != "U_n_nminus1":
        raise ValueError("Interaction must be U*sum_i n_i(n_i-1), without 1/2")
    if str(cfg["disorder"].get("distribution", "")) != "uniform_minus_W_plus_W":
        raise ValueError("Disorder must be uniform on [-W,W]")
    if not (0.0 <= float(cfg["selection"]["center"]) <= 1.0):
        raise ValueError("selection.center must be in [0,1]")
    if int(cfg["selection"]["count"]) < 1:
        raise ValueError("selection.count must be positive")
    if int(cfg["sampling"]["n_realizations"]) < 1:
        raise ValueError("n_realizations must be positive")
    if int(cfg["sampling"]["block_size"]) < 1:
        raise ValueError("block_size must be positive")
    values_from_spec(cfg["disorder"]["W_values"])
    for sector in sectors(cfg):
        if sector["L"] < 2 or sector["N"] < 1 or sector["nmax"] < 1:
            raise ValueError("Invalid sector")
        if sector["N"] > sector["L"] * sector["nmax"]:
            raise ValueError("Sector has no Fock states")


def make_tasks(cfg: Dict[str, object]) -> List[Dict[str, object]]:
    n_samples = int(cfg["sampling"]["n_realizations"])
    block = int(cfg["sampling"]["block_size"])
    tasks: List[Dict[str, object]] = []
    for sector_id, sector in enumerate(sectors(cfg)):
        for W_index, W in enumerate(values_from_spec(cfg["disorder"]["W_values"])):
            for sample_start in range(0, n_samples, block):
                tasks.append({
                    "task_id": len(tasks),
                    "sector_id": sector_id,
                    "L": sector["L"],
                    "N": sector["N"],
                    "nmax": sector["nmax"],
                    "W_index": W_index,
                    "W_over_t": W,
                    "sample_start": sample_start,
                    "sample_stop": min(sample_start + block, n_samples),
                })
    return tasks


def atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(str(temporary), **arrays)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def initialize(cfg: Dict[str, object]) -> None:
    root = result_root(cfg)
    for directory in (root, root / "raw", root / "logs", root / "analysis"):
        directory.mkdir(parents=True, exist_ok=True)
    task_rows = make_tasks(cfg)
    with (root / "tasks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0].keys()))
        writer.writeheader()
        writer.writerows(task_rows)
    clean = {key: value for key, value in cfg.items() if not key.startswith("_")}
    atomic_text(root / "resolved_config.json", json.dumps(clean, indent=2, sort_keys=True) + "\n")
    atomic_text(root / "config.sha256", config_hash(cfg) + "\n")
    print("Initialized {} tasks in {}".format(len(task_rows), root))


def task_file(cfg: Dict[str, object], task_id: int) -> Path:
    return result_root(cfg) / "raw" / "task_{:06d}.npz".format(task_id)


def task_complete(path: Path, expected_hash: str, sample_stop: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(str(path), allow_pickle=False) as data:
            return (
                str(data["config_hash"].item()) == expected_hash
                and int(data["next_sample"].item()) >= int(sample_stop)
            )
    except Exception:
        return False


def run_worker(cfg: Dict[str, object], task_id: int) -> None:
    tasks = make_tasks(cfg)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError("task-id outside 0..{}".format(len(tasks) - 1))
    task = tasks[task_id]
    output = task_file(cfg, task_id)
    expected_hash = config_hash(cfg)
    stop = int(task["sample_stop"])
    if task_complete(output, expected_hash, stop):
        print("Task {} already complete".format(task_id))
        return

    model = cfg["model"]
    selection = cfg["selection"]
    numerics = cfg["numerics"]
    t = float(model["t"])
    L, N, nmax = int(task["L"]), int(task["N"]), int(task["nmax"])
    W_over_t = float(task["W_over_t"])
    structure = build_structure(L, N, nmax, t=t)

    records: Dict[str, List[object]] = {
        "sample_id": [], "state_id": [], "energy": [], "energy_density": [],
        "mean_Edis": [], "mean_Q": [], "cov_Edis_Q": [], "var_Q": [],
        "U_comp_state": [], "valid_state": [], "edge_numerator": [],
        "edge_denominator": [], "U_edge_state": [], "valid_edge_state": [],
        "residual_max": [], "orthogonality_error": [], "probability_norm_error": [],
        "construction_code": [],
    }
    start = int(task["sample_start"])
    if output.exists():
        try:
            with np.load(str(output), allow_pickle=False) as old:
                if str(old["config_hash"].item()) == expected_hash:
                    start = max(start, int(old["next_sample"].item()))
                    for name in records:
                        records[name] = old[name].tolist()
        except Exception:
            start = int(task["sample_start"])

    master_seed = int(cfg["disorder"]["master_seed"])
    construction_codes = {"anderson_orbitals": 0, "truncated_dense": 1, "truncated_sparse": 2}
    checkpoint_every = max(1, int(cfg["sampling"].get("checkpoint_every", 10)))
    for sample_id in range(start, stop):
        eta = disorder_eta(master_seed, L, sample_id)
        epsilon = W_over_t * t * eta
        states = selected_u0_states(
            structure=structure, N=N, nmax=nmax, t=t, epsilon=epsilon,
            count=int(selection["count"]), center=float(selection["center"]),
            dense_max=int(numerics["dense_max"]),
            tolerance=float(numerics["eigensolver_tolerance"]),
        )
        if states.residual_max > float(numerics["residual_tolerance"]):
            raise RuntimeError("H0 residual {} exceeds tolerance".format(states.residual_max))
        if states.orthogonality_error > float(numerics["orthogonality_tolerance"]):
            raise RuntimeError("orthogonality error {} exceeds tolerance".format(states.orthogonality_error))
        obs = covariance_observables(
            structure, epsilon, states, float(numerics["variance_tolerance"])
        )
        count = len(states.energies)
        for state_id in range(count):
            records["sample_id"].append(sample_id)
            records["state_id"].append(state_id)
            records["energy"].append(states.energies[state_id] / t)
            records["energy_density"].append(states.energy_density[state_id])
            for name in ("mean_Edis", "cov_Edis_Q", "U_comp_state", "edge_numerator", "U_edge_state"):
                records[name].append(float(obs[name][state_id]) / t)
            for name in ("mean_Q", "var_Q", "edge_denominator"):
                records[name].append(float(obs[name][state_id]))
            records["valid_state"].append(bool(obs["valid_state"][state_id]))
            records["valid_edge_state"].append(bool(obs["valid_edge_state"][state_id]))
            records["residual_max"].append(states.residual_max)
            records["orthogonality_error"].append(states.orthogonality_error)
            records["probability_norm_error"].append(float(obs["probability_norm_error"][0]))
            records["construction_code"].append(construction_codes[states.construction])
        checkpoint_due = ((sample_id + 1 - int(task["sample_start"])) % checkpoint_every == 0) or (sample_id + 1 == stop)
        if checkpoint_due:
            arrays = {name: np.asarray(values) for name, values in records.items()}
            atomic_savez(
                output, config_hash=np.asarray(expected_hash), task_id=np.asarray(task_id),
                sector_id=np.asarray(int(task["sector_id"])), W_index=np.asarray(int(task["W_index"])),
                W_over_t=np.asarray(W_over_t), L=np.asarray(L), N=np.asarray(N), nmax=np.asarray(nmax),
                next_sample=np.asarray(sample_id + 1), construction_legend=np.asarray("0=anderson,1=truncated_dense,2=truncated_sparse"),
                **arrays
            )
            print("task {}: sample {}/{}".format(task_id, sample_id + 1, stop), flush=True)


def status(cfg: Dict[str, object]) -> Tuple[int, int]:
    tasks = make_tasks(cfg)
    expected_hash = config_hash(cfg)
    complete = sum(
        task_complete(task_file(cfg, int(task["task_id"])), expected_hash, int(task["sample_stop"]))
        for task in tasks
    )
    print("Complete {}/{}; missing {}".format(complete, len(tasks), len(tasks) - complete))
    return complete, len(tasks)


def load_records(cfg: Dict[str, object]) -> Dict[str, np.ndarray]:
    tasks = make_tasks(cfg)
    expected_hash = config_hash(cfg)
    missing = [
        int(task["task_id"]) for task in tasks
        if not task_complete(task_file(cfg, int(task["task_id"])), expected_hash, int(task["sample_stop"]))
    ]
    if missing:
        raise RuntimeError("Missing/incomplete tasks: {}{}".format(missing[:20], "..." if len(missing) > 20 else ""))
    names = [
        "sample_id", "state_id", "energy", "energy_density", "mean_Edis", "mean_Q",
        "cov_Edis_Q", "var_Q", "U_comp_state", "valid_state", "edge_numerator",
        "edge_denominator", "U_edge_state", "valid_edge_state", "residual_max",
        "orthogonality_error", "probability_norm_error", "construction_code",
    ]
    accumulated = {name: [] for name in names}
    metadata = {name: [] for name in ("L", "N", "nmax", "W_over_t")}
    for task in tasks:
        with np.load(str(task_file(cfg, int(task["task_id"]))), allow_pickle=False) as data:
            length = len(data["sample_id"])
            for name in names:
                accumulated[name].append(data[name])
            for name in metadata:
                metadata[name].append(np.full(length, data[name].item()))
    return {
        **{name: np.concatenate(chunks) for name, chunks in accumulated.items()},
        **{name: np.concatenate(chunks) for name, chunks in metadata.items()},
    }


def write_raw_records(path: Path, records: Dict[str, np.ndarray]) -> None:
    fields = [
        "L", "N", "nmax", "W_over_t", "sample_id", "state_id", "energy_density",
        "energy_over_t", "mean_Edis_over_t", "mean_Q", "cov_Edis_Q_over_t", "var_Q",
        "U_comp_state_over_t", "edge_numerator_over_t", "edge_denominator",
        "U_edge_state_over_t", "valid_state", "valid_edge_state", "residual_max",
        "orthogonality_error", "probability_norm_error", "construction_code",
    ]
    sources = [
        "L", "N", "nmax", "W_over_t", "sample_id", "state_id", "energy_density",
        "energy", "mean_Edis", "mean_Q", "cov_Edis_Q", "var_Q", "U_comp_state",
        "edge_numerator", "edge_denominator", "U_edge_state", "valid_state",
        "valid_edge_state", "residual_max", "orthogonality_error",
        "probability_norm_error", "construction_code",
    ]
    with gzip.open(str(path), "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index in range(len(records["L"])):
            writer.writerow([records[name][index] for name in sources])


def sample_summaries(records: Dict[str, np.ndarray], mask: np.ndarray, sample_ids: np.ndarray) -> Dict[str, np.ndarray]:
    output = {name: np.zeros(len(sample_ids), dtype=float) for name in (
        "sum_cov", "sum_var", "sum_u", "count_u", "sum_edge_num", "sum_edge_den",
        "sum_edge_u", "count_edge", "count_total", "count_negative_cov", "count_positive_u",
        "count_negative_u",
    )}
    for row, sample_id in enumerate(sample_ids):
        selected = mask & (records["sample_id"] == sample_id)
        valid = selected & records["valid_state"].astype(bool) & np.isfinite(records["U_comp_state"])
        edge_valid = selected & records["valid_edge_state"].astype(bool) & np.isfinite(records["U_edge_state"])
        output["sum_cov"][row] = np.sum(records["cov_Edis_Q"][valid])
        output["sum_var"][row] = np.sum(records["var_Q"][valid])
        output["sum_u"][row] = np.sum(records["U_comp_state"][valid])
        output["count_u"][row] = np.sum(valid)
        output["sum_edge_num"][row] = np.sum(records["edge_numerator"][edge_valid])
        output["sum_edge_den"][row] = np.sum(records["edge_denominator"][edge_valid])
        output["sum_edge_u"][row] = np.sum(records["U_edge_state"][edge_valid])
        output["count_edge"][row] = np.sum(edge_valid)
        output["count_total"][row] = np.sum(selected)
        output["count_negative_cov"][row] = np.sum(records["cov_Edis_Q"][selected] < 0)
        output["count_positive_u"][row] = np.sum(records["U_comp_state"][valid] > 0)
        output["count_negative_u"][row] = np.sum(records["U_comp_state"][valid] < 0)
    return output


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > np.finfo(float).eps else float("nan")


def estimates(summary: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, float]:
    return {
        "U_mean": safe_ratio(np.sum(summary["sum_u"][indices]), np.sum(summary["count_u"][indices])),
        "U_pool": -safe_ratio(np.sum(summary["sum_cov"][indices]), np.sum(summary["sum_var"][indices])),
        "U_edge_mean": safe_ratio(np.sum(summary["sum_edge_u"][indices]), np.sum(summary["count_edge"][indices])),
        "U_edge_pool": -safe_ratio(np.sum(summary["sum_edge_num"][indices]), np.sum(summary["sum_edge_den"][indices])),
    }


def fit_line(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.ptp(x) <= np.finfo(float).eps:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan"), "rmse": float("nan"), "n": len(x)}
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    residual = y - prediction
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / denominator if denominator > 0 else float("nan")
    return {"intercept": float(intercept), "slope": float(slope), "r2": r2, "rmse": float(np.sqrt(np.mean(residual ** 2))), "n": len(x)}


def percentile_interval(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def load_ed_points(path: str, sector: Dict[str, int]) -> List[Dict[str, float]]:
    if not path or not Path(path).exists():
        return []
    rows: List[Dict[str, float]] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["L"])) != sector["L"] or int(float(row["N"])) != sector["N"] or int(float(row["nmax"])) != sector["nmax"]:
                continue
            if "peak_resolved" in row and not parse_bool(row["peak_resolved"]):
                continue
            value_key = "U_S_star_over_t" if "U_S_star_over_t" in row else "U_peak"
            if not row.get(value_key, ""):
                continue
            rows.append({
                "W": float(row["W_over_t"]), "U": float(row[value_key]),
                "low": float(row.get("U_S_ci95_low", row[value_key]) or row[value_key]),
                "high": float(row.get("U_S_ci95_high", row[value_key]) or row[value_key]),
            })
    return sorted(rows, key=lambda item: item["W"])


def load_model_points(path: str, sector: Dict[str, int]) -> List[Dict[str, float]]:
    if not path or not Path(path).exists():
        return []
    rows: List[Dict[str, float]] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["L"])) != sector["L"] or int(float(row["N"])) != sector["N"] or int(float(row["nmax"])) != sector["nmax"]:
                continue
            if row.get("observable", "") not in ("M_existing", "M_graph_sum", "M_sum"):
                continue
            if row.get("quality_flag", "ok") != "ok":
                continue
            rows.append({
                "W": float(row["W_over_t"]), "U": float(row["U_peak"]),
                "low": float(row.get("U_peak_ci_low", row.get("U_ci_low", row["U_peak"])) or row["U_peak"]),
                "high": float(row.get("U_peak_ci_high", row.get("U_ci_high", row["U_peak"])) or row["U_peak"]),
            })
    return sorted(rows, key=lambda item: item["W"])


def resolve_optional_path(cfg: Dict[str, object], cli_value: str, key: str) -> str:
    value = cli_value or str(cfg.get("comparison", {}).get(key, ""))
    if not value:
        return ""
    path = Path(value)
    return str(path if path.is_absolute() else project_root(cfg) / path)


def analyze(cfg: Dict[str, object], ed_cli: str, model_cli: str, no_plots: bool) -> None:
    records = load_records(cfg)
    output = result_root(cfg) / "analysis"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_raw_records(output / "state_records.csv.gz", records)

    analysis_cfg = cfg["analysis"]
    B = int(analysis_cfg["bootstrap_replicates"])
    rng = np.random.default_rng(int(analysis_cfg["bootstrap_seed"]))
    all_point_rows: List[Dict[str, object]] = []
    all_fit_rows: List[Dict[str, object]] = []
    all_comparison_rows: List[Dict[str, object]] = []
    bootstrap_store: Dict[Tuple[int, str], np.ndarray] = {}
    sector_plot_data: List[Dict[str, object]] = []
    ed_path = resolve_optional_path(cfg, ed_cli, "ed_peaks_csv")
    model_path = resolve_optional_path(cfg, model_cli, "model_peaks_csv")
    if ed_path and not Path(ed_path).exists():
        print("WARNING: ED comparison CSV does not exist: {}".format(ed_path), file=sys.stderr)
    if model_path and not Path(model_path).exists():
        print("WARNING: mixing-model comparison CSV does not exist: {}".format(model_path), file=sys.stderr)

    for sector_id, sector in enumerate(sectors(cfg)):
        sector_mask = (
            (records["L"] == sector["L"]) & (records["N"] == sector["N"])
            & (records["nmax"] == sector["nmax"])
        )
        W_values = np.unique(records["W_over_t"][sector_mask])
        sample_ids = np.arange(int(cfg["sampling"]["n_realizations"]), dtype=int)
        paired_indices = rng.integers(0, len(sample_ids), size=(B, len(sample_ids)))
        central = {name: np.full(len(W_values), np.nan) for name in PREDICTORS}
        boot = {name: np.full((B, len(W_values)), np.nan) for name in PREDICTORS}
        diagnostics: List[Dict[str, float]] = []
        summaries: List[Dict[str, np.ndarray]] = []
        for W_index, W in enumerate(W_values):
            mask = sector_mask & np.isclose(records["W_over_t"], W, rtol=0.0, atol=1.0e-10)
            summary = sample_summaries(records, mask, sample_ids)
            summaries.append(summary)
            full = estimates(summary, np.arange(len(sample_ids)))
            for name in PREDICTORS:
                central[name][W_index] = full[name]
            for replicate in range(B):
                estimate = estimates(summary, paired_indices[replicate])
                for name in PREDICTORS:
                    boot[name][replicate, W_index] = estimate[name]
            total = float(np.sum(summary["count_total"]))
            valid_total = float(np.sum(summary["count_u"]))
            diagnostics.append({
                "mean_cov": safe_ratio(np.sum(summary["sum_cov"]), valid_total),
                "mean_var": safe_ratio(np.sum(summary["sum_var"]), valid_total),
                "fraction_cov_negative": safe_ratio(np.sum(summary["count_negative_cov"]), total),
                "fraction_U_positive": safe_ratio(np.sum(summary["count_positive_u"]), valid_total),
                "fraction_U_negative": safe_ratio(np.sum(summary["count_negative_u"]), valid_total),
                "valid_states": valid_total, "total_states": total,
            })
            for name in PREDICTORS:
                low, high = percentile_interval(boot[name][:, W_index])
                all_point_rows.append({
                    **sector, "W_over_t": W, "predictor": name,
                    "estimate_over_t": central[name][W_index], "ci95_low": low, "ci95_high": high,
                    **diagnostics[-1],
                })
        fit_mask = (W_values >= float(analysis_cfg["fit_Wmin"])) & (W_values <= float(analysis_cfg["fit_Wmax"]))
        for name in PREDICTORS:
            fit = fit_line(W_values[fit_mask], central[name][fit_mask])
            slopes = np.asarray([fit_line(W_values[fit_mask], boot[name][replicate, fit_mask])["slope"] for replicate in range(B)])
            intercepts = np.asarray([fit_line(W_values[fit_mask], boot[name][replicate, fit_mask])["intercept"] for replicate in range(B)])
            slope_low, slope_high = percentile_interval(slopes)
            intercept_low, intercept_high = percentile_interval(intercepts)
            all_fit_rows.append({
                **sector, "series": name, **fit, "slope_ci95_low": slope_low,
                "slope_ci95_high": slope_high, "intercept_ci95_low": intercept_low,
                "intercept_ci95_high": intercept_high,
            })
            bootstrap_store[(sector_id, name)] = boot[name]

        minimum_w, maximum_w = float(np.min(W_values)), float(np.max(W_values))
        ed = [point for point in load_ed_points(ed_path, sector) if minimum_w - 1.0e-10 <= point["W"] <= maximum_w + 1.0e-10]
        model = [point for point in load_model_points(model_path, sector) if minimum_w - 1.0e-10 <= point["W"] <= maximum_w + 1.0e-10]
        for label, external in (("S_ED", ed), ("M_existing", model)):
            if not external:
                continue
            x = np.asarray([point["W"] for point in external])
            y = np.asarray([point["U"] for point in external])
            external_mask = (x >= float(analysis_cfg["fit_Wmin"])) & (x <= float(analysis_cfg["fit_Wmax"]))
            fit = fit_line(x[external_mask], y[external_mask])
            low = np.asarray([point["low"] for point in external]); high = np.asarray([point["high"] for point in external])
            sigma = np.maximum((high - low) / (2.0 * 1.96), 1.0e-12)
            draws = rng.normal(y[None, :], sigma[None, :], size=(B, len(y)))
            fit_draws = [fit_line(x[external_mask], draws[replicate, external_mask]) for replicate in range(B)]
            slope_low, slope_high = percentile_interval(np.asarray([item["slope"] for item in fit_draws]))
            intercept_low, intercept_high = percentile_interval(np.asarray([item["intercept"] for item in fit_draws]))
            all_fit_rows.append({**sector, "series": label, **fit, "slope_ci95_low": slope_low, "slope_ci95_high": slope_high, "intercept_ci95_low": intercept_low, "intercept_ci95_high": intercept_high})

        if ed:
            ed_map = {round(point["W"], 10): point for point in ed}
            matched_indices = [index for index, value in enumerate(W_values) if round(float(value), 10) in ed_map]
            ed_points = [ed_map[round(float(W_values[index]), 10)] for index in matched_indices]
            ed_y = np.asarray([point["U"] for point in ed_points])
            ed_sigma = np.maximum(np.asarray([point["high"] - point["low"] for point in ed_points]) / (2.0 * 1.96), 1.0e-12)
            ed_draws = rng.normal(ed_y[None, :], ed_sigma[None, :], size=(B, len(ed_y)))
            for name in PREDICTORS:
                prediction = central[name][matched_indices]
                difference = prediction - ed_y
                rmse_draw = np.sqrt(np.nanmean((boot[name][:, matched_indices] - ed_draws) ** 2, axis=1))
                rmse_low, rmse_high = percentile_interval(rmse_draw)
                all_comparison_rows.append({
                    **sector, "series": name, "scope": "all_entropy_points", "n_matched": len(matched_indices),
                    "rmse_to_entropy": float(np.sqrt(np.nanmean(difference ** 2))),
                    "rmse_ci95_low": rmse_low, "rmse_ci95_high": rmse_high,
                    "mae_to_entropy": float(np.nanmean(np.abs(difference))),
                    "bias_to_entropy": float(np.nanmean(difference)),
                })
            if model:
                model_map = {round(point["W"], 10): point for point in model}
                w_index_map = {round(float(value), 10): index for index, value in enumerate(W_values)}
                common = sorted(set(ed_map).intersection(model_map).intersection(w_index_map))
                if common:
                    common_indices = [w_index_map[key] for key in common]
                    ed_common = np.asarray([ed_map[key]["U"] for key in common])
                    model_common = np.asarray([model_map[key]["U"] for key in common])
                    difference = model_common - ed_common
                    ed_sig = np.maximum(np.asarray([ed_map[key]["high"] - ed_map[key]["low"] for key in common]) / (2.0 * 1.96), 1.0e-12)
                    model_sig = np.maximum(np.asarray([model_map[key]["high"] - model_map[key]["low"] for key in common]) / (2.0 * 1.96), 1.0e-12)
                    common_ed_draws = rng.normal(ed_common, ed_sig, size=(B, len(common)))
                    for name in PREDICTORS:
                        prediction = central[name][common_indices]
                        predictor_difference = prediction - ed_common
                        predictor_rmse_draw = np.sqrt(np.nanmean((boot[name][:, common_indices] - common_ed_draws) ** 2, axis=1))
                        predictor_rmse_low, predictor_rmse_high = percentile_interval(predictor_rmse_draw)
                        all_comparison_rows.append({
                            **sector, "series": name, "scope": "common_with_M", "n_matched": len(common),
                            "rmse_to_entropy": float(np.sqrt(np.nanmean(predictor_difference ** 2))),
                            "rmse_ci95_low": predictor_rmse_low, "rmse_ci95_high": predictor_rmse_high,
                            "mae_to_entropy": float(np.nanmean(np.abs(predictor_difference))),
                            "bias_to_entropy": float(np.nanmean(predictor_difference)),
                        })
                    metric_draw = np.sqrt(np.mean((rng.normal(model_common, model_sig, size=(B, len(common))) - common_ed_draws) ** 2, axis=1))
                    rmse_low, rmse_high = percentile_interval(metric_draw)
                    all_comparison_rows.append({
                        **sector, "series": "M_existing", "scope": "common_with_M", "n_matched": len(common),
                        "rmse_to_entropy": float(np.sqrt(np.mean(difference ** 2))),
                        "rmse_ci95_low": rmse_low, "rmse_ci95_high": rmse_high,
                        "mae_to_entropy": float(np.mean(np.abs(difference))),
                        "bias_to_entropy": float(np.mean(difference)),
                    })
        sector_plot_data.append({"sector": sector, "W": W_values, "central": central, "boot": boot, "diagnostics": diagnostics, "ed": ed, "model": model, "mask": sector_mask})

    point_fields = list(all_point_rows[0].keys())
    with (output / "predictor_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields)
        writer.writeheader(); writer.writerows(all_point_rows)
    fit_fields = list(all_fit_rows[0].keys())
    with (output / "linear_fits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fit_fields)
        writer.writeheader(); writer.writerows(all_fit_rows)
    if all_comparison_rows:
        comparison_fields = list(all_comparison_rows[0].keys())
        with (output / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=comparison_fields)
            writer.writeheader(); writer.writerows(all_comparison_rows)
    elif (output / "comparison_metrics.csv").exists():
        (output / "comparison_metrics.csv").unlink()

    summary_lines = [
        "Covariance-compensation model v{}".format(VERSION),
        "Hamiltonian: hopping + disorder epsilon in [-W,W] + U*sum n(n-1) (no 1/2)",
        "Complete tasks: {}/{}".format(*status(cfg)),
        "State rows: {}".format(len(records["L"])),
        "Paired bootstrap replicates: {}".format(B),
        "ED comparison: {}".format((ed_path + (" (missing)" if not Path(ed_path).exists() else "")) if ed_path else "not configured"),
        "Mixing-model comparison: {}".format((model_path + (" (missing)" if not Path(model_path).exists() else "")) if model_path else "not configured"),
    ]
    atomic_text(output / "summary.txt", "\n".join(summary_lines) + "\n")
    if not no_plots:
        make_plots(cfg, records, sector_plot_data, all_fit_rows, figures)
    print("Analysis written to {}".format(output))


def save_figure(figure: object, figures: Path, stem: str) -> None:
    figure.savefig(str(figures / (stem + ".png")), dpi=220, bbox_inches="tight")
    figure.savefig(str(figures / (stem + ".pdf")), bbox_inches="tight")


def make_plots(cfg: Dict[str, object], records: Dict[str, np.ndarray], data: List[Dict[str, object]], fit_rows: List[Dict[str, object]], figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"U_mean": "tab:blue", "U_pool": "tab:red", "U_edge_mean": "tab:purple", "U_edge_pool": "tab:green"}
    labels = {"U_mean": r"$U_{mean}^*$", "U_pool": r"$U_{pool}^*$", "U_edge_mean": r"$U_{edge,mean}^*$", "U_edge_pool": r"$U_{edge,pool}^*$"}
    for item in data:
        sector, W, central, boot = item["sector"], item["W"], item["central"], item["boot"]
        tag = "L{}_N{}_nmax{}".format(sector["L"], sector["N"], sector["nmax"])
        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        for name in PREDICTORS:
            low = np.nanpercentile(boot[name], 2.5, axis=0)
            high = np.nanpercentile(boot[name], 97.5, axis=0)
            ax.errorbar(W, central[name], yerr=[np.maximum(central[name] - low, 0.0), np.maximum(high - central[name], 0.0)], marker="o", capsize=3, label=labels[name], color=colors[name])
        if item["ed"]:
            ext = item["ed"]
            x = np.asarray([point["W"] for point in ext]); y = np.asarray([point["U"] for point in ext])
            lo = np.asarray([point["low"] for point in ext]); hi = np.asarray([point["high"] for point in ext])
            ax.errorbar(x, y, yerr=[np.maximum(y-lo, 0.0), np.maximum(hi-y, 0.0)], marker="s", linestyle="none", color="black", capsize=3, label=r"$U_S^*$ (ED)")
        if item["model"]:
            ext = item["model"]
            x = np.asarray([point["W"] for point in ext]); y = np.asarray([point["U"] for point in ext])
            lo = np.asarray([point["low"] for point in ext]); hi = np.asarray([point["high"] for point in ext])
            ax.errorbar(x, y, yerr=[np.maximum(y-lo, 0.0), np.maximum(hi-y, 0.0)], marker="^", linestyle="none", color="0.4", capsize=3, label=r"$U_M^*$")
        ax.set(xlabel=r"$W/t$", ylabel=r"$U^*/t$", title=r"$L={},\ N={},\ n_{{max}}={}$".format(sector["L"], sector["N"], sector["nmax"]))
        ax.grid(alpha=0.2); ax.legend(fontsize=9); fig.tight_layout(); save_figure(fig, figures, "predictors_" + tag); plt.close(fig)

        diagnostics = item["diagnostics"]
        fig, ax1 = plt.subplots(figsize=(7.5, 5.2)); ax2 = ax1.twinx()
        neg_cov = [-row["mean_cov"] for row in diagnostics]; var_q = [row["mean_var"] for row in diagnostics]
        first = ax1.plot(W, neg_cov, "o-", color="tab:blue", label=r"$-\langle\mathrm{Cov}(E_{dis},Q)\rangle/t$")
        second = ax2.plot(W, var_q, "s-", color="tab:orange", label=r"$\langle\mathrm{Var}(Q)\rangle$")
        ax1.set_xlabel(r"$W/t$"); ax1.set_ylabel(r"$-\mathrm{Cov}/t$", color="tab:blue"); ax2.set_ylabel(r"$\mathrm{Var}(Q)$", color="tab:orange")
        ax1.legend(first + second, [line.get_label() for line in first + second], fontsize=9); ax1.grid(alpha=0.2); fig.tight_layout(); save_figure(fig, figures, "covariance_variance_" + tag); plt.close(fig)

        requested = [float(value) for value in cfg["analysis"].get("histogram_W", [])]
        selected_w = sorted(set(float(W[np.argmin(np.abs(W - value))]) for value in requested))
        if selected_w:
            fig, axes = plt.subplots(1, len(selected_w), figsize=(5.0 * len(selected_w), 4.2), squeeze=False)
            for axis, chosen in zip(axes[0], selected_w):
                mask = item["mask"] & np.isclose(records["W_over_t"], chosen) & records["valid_state"].astype(bool) & np.isfinite(records["U_comp_state"])
                axis.hist(records["U_comp_state"][mask], bins=35, color="tab:blue", alpha=0.8)
                axis.axvline(0.0, color="black", linewidth=1); axis.set_title(r"$W/t={}$".format(chosen)); axis.set_xlabel(r"$U_{comp,m}/t$"); axis.set_ylabel("count")
            fig.tight_layout(); save_figure(fig, figures, "histograms_" + tag); plt.close(fig)

        if item["ed"]:
            ed_map = {round(point["W"], 10): point["U"] for point in item["ed"]}
            pairs = [(central["U_pool"][index], ed_map[round(value, 10)]) for index, value in enumerate(W) if round(float(value), 10) in ed_map]
            if pairs:
                x = np.asarray([pair[1] for pair in pairs]); y = np.asarray([pair[0] for pair in pairs])
                finite = np.isfinite(x) & np.isfinite(y); x, y = x[finite], y[finite]
            if pairs and len(x):
                fig, ax = plt.subplots(figsize=(5.5, 5.2)); ax.scatter(x, y, color="tab:red")
                limits = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]; ax.plot(limits, limits, "k--", label="equality")
                ax.set(xlabel=r"$U_S^*/t$ (ED)", ylabel=r"$U_{pool}^*/t$", title=tag.replace("_", ", ")); ax.legend(); ax.grid(alpha=0.2); fig.tight_layout(); save_figure(fig, figures, "pool_vs_entropy_" + tag); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    series_order = ["S_ED", "M_existing", "U_mean", "U_pool", "U_edge_pool"]
    offsets = np.linspace(-0.25, 0.25, len(series_order))
    for series, offset in zip(series_order, offsets):
        selected = [row for row in fit_rows if row["series"] == series]
        if not selected: continue
        x = np.asarray([row["N"] for row in selected], dtype=float) + offset
        y = np.asarray([row["slope"] for row in selected], dtype=float)
        low = np.asarray([row["slope_ci95_low"] for row in selected], dtype=float); high = np.asarray([row["slope_ci95_high"] for row in selected], dtype=float)
        finite_ci = np.isfinite(low) & np.isfinite(high)
        yerr = np.zeros((2, len(y))); yerr[0, finite_ci] = np.maximum(y[finite_ci] - low[finite_ci], 0.0); yerr[1, finite_ci] = np.maximum(high[finite_ci] - y[finite_ci], 0.0)
        ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="none", capsize=3, label=series)
    ax.set_xlabel(r"sector particle number $N$"); ax.set_ylabel(r"slope $c_X$"); ax.grid(alpha=0.2); ax.legend(); fig.tight_layout(); save_figure(fig, figures, "slope_comparison"); plt.close(fig)


def validate_and_report(cfg: Dict[str, object]) -> None:
    W = values_from_spec(cfg["disorder"]["W_values"])
    for sector in sectors(cfg):
        structure = build_structure(sector["L"], sector["N"], sector["nmax"], float(cfg["model"]["t"]))
        method = "exact Anderson-orbital construction" if sector["nmax"] >= sector["N"] else "truncated H0 diagonalization"
        print("L={} N={} nmax={}: dim={}, method={}".format(sector["L"], sector["N"], sector["nmax"], structure.dim, method))
    print("W/t={}..{} ({} points)".format(W[0], W[-1], len(W)))
    print("realizations={}, selected states={}, tasks={}".format(cfg["sampling"]["n_realizations"], cfg["selection"]["count"], len(make_tasks(cfg))))
    print("Hamiltonian: -t hopping + epsilon*n + U*sum n(n-1); epsilon in [-W,W]; open boundary")
    print("No fitted scale or offset is applied to the predictors.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for name in ("validate", "init", "task-count", "root", "status"):
        sub = subparsers.add_parser(name); sub.add_argument("--config", required=True)
    worker = subparsers.add_parser("worker"); worker.add_argument("--config", required=True); worker.add_argument("--task-id", type=int, required=True)
    analysis = subparsers.add_parser("analyze"); analysis.add_argument("--config", required=True); analysis.add_argument("--ed-csv", default=""); analysis.add_argument("--model-csv", default=""); analysis.add_argument("--no-plots", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args(); cfg = load_config(args.config)
    if args.command == "validate": validate_and_report(cfg)
    elif args.command == "init": initialize(cfg)
    elif args.command == "task-count": print(len(make_tasks(cfg)))
    elif args.command == "root": print(result_root(cfg))
    elif args.command == "status": status(cfg)
    elif args.command == "worker": run_worker(cfg, args.task_id)
    elif args.command == "analyze": analyze(cfg, args.ed_csv, args.model_csv, args.no_plots)


if __name__ == "__main__":
    main()
