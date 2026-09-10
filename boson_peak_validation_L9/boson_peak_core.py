#!/usr/bin/env python3
"""Shared numerical core for the bosonic entropy-peak validation workflow.

The module deliberately contains no cluster-specific code.  It implements the
fixed-N bosonic basis, the open-chain Bose-Hubbard Hamiltonian, the independent
compensating-channel theory, sparse/dense central-spectrum solvers, and the
observables needed by the analysis stage.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


PROGRAM_VERSION = "1.1.0"
SCHEMA_VERSION = 1
FLOAT_ATOL = 1.0e-12


@dataclass(frozen=True)
class BosonStructure:
    occupations: np.ndarray
    interaction_q: np.ndarray
    hopping: sp.csr_matrix

    @property
    def dim(self) -> int:
        return int(self.occupations.shape[0])

    @property
    def L(self) -> int:
        return int(self.occupations.shape[1])


@dataclass(frozen=True)
class Channel:
    L: int
    N: int
    nmax: int
    a: int
    b: int
    k: int
    alpha: float
    V_over_t: float
    Gamma_over_t: float
    C_ab: int
    q_ab: float


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def code_hash(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        fieldnames = fields
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_config(path: Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("This configuration is YAML; install PyYAML or use JSON syntax") from exc
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(config)
    return config


def apply_config_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply repeatable dotted KEY=JSON command-line overrides."""
    result = copy.deepcopy(config)
    for expression in overrides:
        if "=" not in expression:
            raise ValueError("Override must have the form section.key=JSON: " + expression)
        dotted_key, raw_value = expression.split("=", 1)
        keys = [key for key in dotted_key.split(".") if key]
        if not keys:
            raise ValueError("Override key cannot be empty")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        target: Dict[str, Any] = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError("Unknown configuration path: " + dotted_key)
            target = target[key]
        target[keys[-1]] = value
    validate_config(result)
    return result


def _number_list(value: Any, name: str, allow_zero: bool = True) -> List[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(name + " must be a non-empty list")
    result = [float(item) for item in value]
    if not all(np.isfinite(result)):
        raise ValueError(name + " contains a non-finite value")
    if (allow_zero and any(item < 0 for item in result)) or (not allow_zero and any(item <= 0 for item in result)):
        raise ValueError(name + " contains a value outside the allowed positive range")
    return result


def ed_sectors(config: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    output: List[Tuple[int, int, int]] = []
    for item in config["ed"]["sectors"]:
        L, N = int(item["L"]), int(item.get("N", item["L"]))
        nmax_values = item["nmax"] if isinstance(item["nmax"], list) else [item["nmax"]]
        for nmax in nmax_values:
            output.append((L, N, int(nmax)))
    return sorted(set(output))


def theory_sectors(config: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    section = config.get("theory", {})
    output: List[Tuple[int, int, int]] = []
    if bool(section.get("standard_sector_set", True)):
        for L in (6, 7, 8, 9):
            for nmax in range(2, min(L, 9) + 1):
                output.append((L, L, nmax))
        for N in (5, 6, 7, 8, 9):
            for nmax in range(2, min(N, 9) + 1):
                output.append((9, N, nmax))
    for item in section.get("sectors", []):
        L, N = int(item["L"]), int(item.get("N", item["L"]))
        values = item["nmax"] if isinstance(item["nmax"], list) else [item["nmax"]]
        output.extend((L, N, int(nmax)) for nmax in values)
    return sorted(set(output))


def theory_W_values(config: Dict[str, Any]) -> List[float]:
    section = config.get("theory", {})
    if section.get("W_values"):
        return sorted(set(_number_list(section["W_values"], "theory.W_values")))
    log_min = float(section.get("W_log_min", 0.1))
    log_max = float(section.get("W_log_max", 32.0))
    log_points = int(section.get("W_log_points", 40))
    linear = [float(x) for x in section.get("W_linear_values", [0.1, 0.25, 0.5, 0.75, 1, 2, 4, 8, 12, 16, 24, 32])]
    return sorted(set(linear + list(np.geomspace(log_min, log_max, log_points))))


def validate_config(config: Dict[str, Any]) -> None:
    for section in ("model", "ed", "eigensolver", "analysis", "cluster"):
        if not isinstance(config.get(section), dict):
            raise ValueError("Missing configuration section: " + section)
    model = config["model"]
    if str(model.get("boundary", "open")).lower() != "open":
        raise ValueError("This validation implements only an open chain (OBC)")
    if float(model.get("t", 1.0)) <= 0:
        raise ValueError("model.t must be positive")
    if int(model.get("dense_max", 5000)) < 2:
        raise ValueError("model.dense_max must be at least 2")
    sectors = ed_sectors(config)
    if not sectors:
        raise ValueError("ed.sectors cannot be empty")
    for L, N, nmax in sectors:
        if L < 2 or N < 0 or nmax < 1 or N > L * nmax:
            raise ValueError("Invalid ED sector L={}, N={}, nmax={}".format(L, N, nmax))
    _number_list(config["ed"]["W_values"], "ed.W_values")
    if int(config["ed"].get("realizations", 1)) < 1:
        raise ValueError("ed.realizations must be positive")
    if int(config["ed"].get("realization_block_size", 1)) < 1:
        raise ValueError("ed.realization_block_size must be positive")
    grid = config["ed"].get("U_grid", {})
    if grid.get("mode", "adaptive_theory") not in ("adaptive_theory", "explicit"):
        raise ValueError("ed.U_grid.mode must be adaptive_theory or explicit")
    if grid.get("mode") == "explicit":
        _number_list(grid["values"], "ed.U_grid.values")
    solver = config["eigensolver"]
    if solver.get("backend", "auto") not in ("auto", "scipy", "dense"):
        raise ValueError("eigensolver.backend must be auto, scipy, or dense")
    if int(solver.get("nev", 36)) < 4:
        raise ValueError("eigensolver.nev must be at least 4")
    if int(solver.get("n_states", 20)) < 1:
        raise ValueError("eigensolver.n_states must be positive")
    fraction = solver.get("spectrum_fraction")
    if fraction is not None and not (0.0 < float(fraction) <= 1.0):
        raise ValueError("eigensolver.spectrum_fraction must be in (0, 1]")
    if float(solver.get("residual_tol", 1.0e-9)) <= 0:
        raise ValueError("eigensolver.residual_tol must be positive")
    if int(config["analysis"].get("bootstrap", 2000)) < 1:
        raise ValueError("analysis.bootstrap must be positive")


def run_manifest(config: Dict[str, Any], task_count: int, folder: Path) -> Dict[str, Any]:
    return {
        "program_version": PROGRAM_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "master_seed": int(config["ed"].get("master_seed", 20260907)),
        "config_hash": sha256_json(config),
        "code_hash": code_hash(folder),
        "task_count": int(task_count),
        "config": config,
    }


def bounded_composition_dim(L: int, N: int, nmax: int) -> int:
    total = 0
    for j in range(0, min(L, N // (nmax + 1)) + 1):
        remaining = N - j * (nmax + 1)
        total += (-1) ** j * math.comb(L, j) * math.comb(remaining + L - 1, L - 1)
    return int(total)


def generate_basis(L: int, N: int, nmax: int) -> np.ndarray:
    if L < 1 or N < 0 or nmax < 0 or N > L * nmax:
        raise ValueError("Invalid basis parameters")
    states: List[Tuple[int, ...]] = []
    current = [0] * L

    def recurse(site: int, remaining: int) -> None:
        if site == L - 1:
            if 0 <= remaining <= nmax:
                current[site] = remaining
                states.append(tuple(current))
            return
        lower = max(0, remaining - nmax * (L - site - 1))
        upper = min(nmax, remaining)
        for occupation in range(lower, upper + 1):
            current[site] = occupation
            recurse(site + 1, remaining - occupation)

    recurse(0, N)
    basis = np.asarray(states, dtype=np.int16)
    expected = bounded_composition_dim(L, N, nmax)
    if basis.shape != (expected, L):
        raise AssertionError("Basis generator returned {} states, expected {}".format(basis.shape[0], expected))
    return basis


def build_structure(L: int, N: int, nmax: int, t: float = 1.0) -> BosonStructure:
    occupations = generate_basis(L, N, nmax)
    dim = int(occupations.shape[0])
    index = {tuple(int(x) for x in occupations[row]): row for row in range(dim)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for source, state_array in enumerate(occupations):
        state = [int(x) for x in state_array]
        for site in range(L - 1):
            for origin, target_site in ((site, site + 1), (site + 1, site)):
                a, b = state[origin], state[target_site]
                if a == 0 or b >= nmax:
                    continue
                target = list(state)
                target[origin] -= 1
                target[target_site] += 1
                destination = index[tuple(target)]
                rows.append(destination)
                cols.append(source)
                data.append(-float(t) * math.sqrt(a * (b + 1)))
    hopping = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()
    hopping.sum_duplicates()
    hopping.eliminate_zeros()
    difference = hopping - hopping.T
    if difference.nnz and float(np.max(np.abs(difference.data))) > 1.0e-13:
        raise AssertionError("Hopping matrix is not symmetric")
    interaction_q = np.sum(occupations * (occupations - 1), axis=1).astype(np.float64)
    return BosonStructure(occupations, interaction_q, hopping)


def remaining_configuration_count(sites: int, particles: int, nmax: int) -> int:
    if particles < 0 or particles > sites * nmax:
        return 0
    if sites == 0:
        return int(particles == 0)
    return bounded_composition_dim(sites, particles, nmax)


def channel_table(L: int, N: int, nmax: int, t: float = 1.0) -> List[Channel]:
    raw: List[Tuple[int, int, int, float, float, int]] = []
    for a in range(1, nmax + 1):
        for b in range(0, nmax):
            k = b - a + 1
            if k <= 0:
                continue
            count = remaining_configuration_count(L - 2, N - a - b, nmax)
            if count <= 0:
                continue
            V = float(t) * math.sqrt(a * (b + 1))
            raw.append((a, b, k, V, 2.0 * abs(V), count))
    total = sum(row[-1] for row in raw)
    if total <= 0:
        raise ValueError("No positive compensating channels for L={}, N={}, nmax={}".format(L, N, nmax))
    return [
        Channel(L, N, nmax, a, b, k, 2.0 * k, V / t, Gamma / t, count, count / total)
        for a, b, k, V, Gamma, count in raw
    ]


def channel_rows(channels: Sequence[Channel]) -> List[Dict[str, Any]]:
    return [
        {
            "L": c.L,
            "N": c.N,
            "nmax": c.nmax,
            "a": c.a,
            "b": c.b,
            "k": c.k,
            "alpha": c.alpha,
            "V_over_t": c.V_over_t,
            "Gamma_over_t": c.Gamma_over_t,
            "C_ab": c.C_ab,
            "q_ab": c.q_ab,
        }
        for c in channels
    ]


def mixing_integral(u: Any, gamma: Any) -> np.ndarray:
    u_array = np.asarray(u, dtype=float)
    gamma_array = np.asarray(gamma, dtype=float)
    if np.any(gamma_array <= 0):
        raise ValueError("gamma must be positive")
    atan_sum = np.arctan((2.0 - u_array) / gamma_array) + np.arctan(u_array / gamma_array)
    first = gamma_array * (2.0 - u_array) * atan_sum / 2.0
    ratio = ((2.0 - u_array) ** 2 + gamma_array**2) / (u_array**2 + gamma_array**2)
    second = gamma_array**2 * np.log(ratio) / 4.0
    return first - second


def mixing_integral_derivative(u: Any, gamma: Any) -> np.ndarray:
    """Return dI/du for the exact triangular-disorder mixing integral."""
    u_array = np.asarray(u, dtype=float)
    gamma_array = np.asarray(gamma, dtype=float)
    if np.any(gamma_array <= 0):
        raise ValueError("gamma must be positive")
    left = u_array**2 + gamma_array**2
    right = (2.0 - u_array) ** 2 + gamma_array**2
    atan_sum = np.arctan((2.0 - u_array) / gamma_array) + np.arctan(u_array / gamma_array)
    atan_derivative = -gamma_array / right + gamma_array / left
    log_derivative = -2.0 * (2.0 - u_array) / right - 2.0 * u_array / left
    return (
        -0.5 * gamma_array * atan_sum
        + 0.5 * gamma_array * (2.0 - u_array) * atan_derivative
        - 0.25 * gamma_array**2 * log_derivative
    )


def theory_mixing(U: Any, W: float, channels: Sequence[Channel], t: float = 1.0) -> np.ndarray:
    values = np.asarray(U, dtype=float)
    answer = np.zeros_like(values)
    if W <= FLOAT_ATOL:
        for c in channels:
            gamma = c.Gamma_over_t * t
            answer += c.q_ab * gamma**2 / ((c.alpha * values) ** 2 + gamma**2)
        return answer
    for c in channels:
        answer += c.q_ab * mixing_integral(c.alpha * values / W, c.Gamma_over_t * t / W)
    return answer


def sample_mixing(U: Any, epsilon: np.ndarray, channels: Sequence[Channel], t: float = 1.0) -> np.ndarray:
    values = np.asarray(U, dtype=float)
    x_bonds = np.abs(np.diff(np.asarray(epsilon, dtype=float)))
    answer = np.zeros_like(values)
    if x_bonds.size == 0:
        return answer
    for c in channels:
        gamma = c.Gamma_over_t * t
        detuning = c.alpha * values[:, None] - x_bonds[None, :]
        answer += c.q_ab * np.mean(gamma**2 / (detuning**2 + gamma**2), axis=1)
    return answer


def theory_coefficients(channels: Sequence[Channel]) -> Dict[str, float]:
    q = np.asarray([c.q_ab for c in channels])
    alpha = np.asarray([c.alpha for c in channels])
    gamma = np.asarray([c.Gamma_over_t for c in channels])
    D2 = float(np.sum(q * alpha**2 / gamma**2))
    A = float((2.0 / 3.0) * np.sum(q * alpha / gamma**2) / D2)
    mu3 = 4.0 / 5.0 - 2.0 * alpha * A + 2.0 * (alpha * A) ** 2 - (alpha * A) ** 3
    B = float(-2.0 * np.sum(q * alpha * mu3 / gamma**4) / D2)
    C2 = float((2.0 / math.pi) * np.sum(q * gamma**2 / alpha) / np.sum(q * alpha * gamma))
    return {
        "A": A,
        "B": B,
        "C": math.sqrt(max(C2, 0.0)),
        "D2": D2,
        "Gamma_min_over_t": float(np.min(gamma)),
        "Gamma_max_over_t": float(np.max(gamma)),
        "n_channels": int(len(channels)),
    }


def global_theory_peak(W: float, channels: Sequence[Channel], t: float, settings: Dict[str, Any]) -> Dict[str, float]:
    if W <= FLOAT_ATOL:
        return {"U_M_star_over_t": 0.0, "mixing_max": 1.0, "search_U_max_over_t": 0.0, "local_maxima": 1}
    points = max(101, int(settings.get("peak_grid_points", 2001)))
    upper = max(float(settings.get("peak_U_min_max_over_t", 4.0)), float(settings.get("peak_U_max_per_W", 2.0)) * W) * t
    expansions = max(0, int(settings.get("peak_max_expansions", 3)))
    for expansion in range(expansions + 1):
        grid = np.linspace(0.0, upper, points)
        curve = theory_mixing(grid, W, channels, t)
        if int(np.argmax(curve)) < points - 1 or expansion == expansions:
            break
        upper *= 2.0
    candidates = [0]
    candidates.extend(i for i in range(1, points - 1) if curve[i] >= curve[i - 1] and curve[i] >= curve[i + 1])
    if int(np.argmax(curve)) == points - 1:
        candidates.append(points - 1)
    refined: List[Tuple[float, float]] = []
    for index in candidates:
        if index == 0 or index == points - 1:
            refined.append((float(grid[index]), float(curve[index])))
            continue
        result = opt.minimize_scalar(
            lambda x: -float(theory_mixing(np.asarray([x]), W, channels, t)[0]),
            bounds=(float(grid[index - 1]), float(grid[index + 1])),
            method="bounded",
            options={"xatol": 1.0e-11},
        )
        refined.append((float(result.x), float(-result.fun)))
    peak_U, peak_value = max(refined, key=lambda item: item[1])
    return {
        "U_M_star_over_t": peak_U / t,
        "mixing_max": peak_value,
        "search_U_max_over_t": upper / t,
        "local_maxima": int(len(refined)),
    }


def adaptive_U_grid(U_peak_over_t: float, t: float, settings: Dict[str, Any]) -> np.ndarray:
    mode = settings.get("mode", "adaptive_theory")
    if mode == "explicit":
        return np.asarray(sorted(set(float(x) for x in settings["values"])), dtype=float)
    peak = max(0.0, float(U_peak_over_t) * t)
    half_width = max(float(settings.get("absolute_half_width_over_t", 0.25)) * t, float(settings.get("relative_half_width", 0.5)) * peak)
    lower, upper = max(0.0, peak - half_width), peak + half_width
    count = max(3, int(settings.get("points", 9)))
    values = list(np.linspace(lower, upper, count))
    values.extend([0.0, 0.5 * peak, 1.5 * peak])
    decimals = int(settings.get("round_decimals", 8))
    return np.asarray(sorted(set(round(float(x), decimals) for x in values if x >= 0.0)), dtype=float)


def disorder_eta(master_seed: int, L: int, realization: int) -> np.ndarray:
    seed = np.random.SeedSequence([int(master_seed), int(L), int(realization)])
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=L)


def hamiltonian(structure: BosonStructure, U: float, epsilon: np.ndarray) -> sp.csr_matrix:
    diagonal = float(U) * structure.interaction_q + structure.occupations @ np.asarray(epsilon, dtype=float)
    return structure.hopping + sp.diags(diagonal, format="csr")


def relative_residuals(H: sp.csr_matrix, energies: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    action = H @ vectors
    numerator = np.linalg.norm(action - vectors * energies[None, :], axis=0)
    denominator = np.maximum.reduce((np.linalg.norm(action, axis=0), np.abs(energies), np.ones(energies.size)))
    return numerator / denominator


def orthogonality_error(vectors: np.ndarray) -> float:
    gram = vectors.conj().T @ vectors
    return float(np.max(np.abs(gram - np.eye(gram.shape[0])), initial=0.0))


def _sparse_extreme(H: sp.csr_matrix, which: str, settings: Dict[str, Any]) -> float:
    return float(
        spla.eigsh(
            H,
            k=1,
            which=which,
            return_eigenvectors=False,
            tol=max(float(settings.get("tol", 1.0e-10)), 1.0e-8),
            maxiter=int(settings.get("maxiter", 30000)),
        )[0]
    )


def selected_state_count(dim: int, settings: Dict[str, Any]) -> int:
    fraction = settings.get("spectrum_fraction")
    if fraction is not None:
        return min(dim, max(1, int(math.ceil(float(fraction) * dim))))
    return min(dim, int(settings.get("n_states", 20)))


def requested_eigenpair_count(dim: int, settings: Dict[str, Any]) -> int:
    return min(dim, max(int(settings.get("nev", 36)), selected_state_count(dim, settings)))


def solve_central_spectrum(H: sp.csr_matrix, settings: Dict[str, Any], dense_max: int) -> Dict[str, Any]:
    dim = int(H.shape[0])
    requested = requested_eigenpair_count(dim, settings)
    backend = str(settings.get("backend", "auto"))
    use_dense = backend == "dense" or (backend == "auto" and dim <= dense_max)
    if backend == "dense" and dim > dense_max:
        raise MemoryError("Dense backend refused: D={} exceeds dense_max={}".format(dim, dense_max))
    started = time.perf_counter()
    retry_log: List[Dict[str, Any]] = []
    if use_dense:
        all_values, all_vectors = np.linalg.eigh(H.toarray())
        emin, emax = float(all_values[0]), float(all_values[-1])
        sigma = 0.5 * (emin + emax)
        chosen = np.argsort(np.abs(all_values - sigma), kind="stable")[:requested]
        values, vectors = all_values[chosen], all_vectors[:, chosen]
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        attempts = 1
        backend_used = "dense"
    else:
        if dim <= requested + 1:
            raise ValueError("Sparse eigsh requires nev < D-1")
        emin = _sparse_extreme(H, "SA", settings)
        emax = _sparse_extreme(H, "LA", settings)
        sigma = 0.5 * (emin + emax)
        width = max(emax - emin, 1.0)
        retries = max(0, int(settings.get("retries", 3)))
        base_ncv = max(requested + 2, int(settings.get("ncv", 0)), 2 * requested + 1)
        last_error: Optional[BaseException] = None
        for attempt in range(retries + 1):
            direction = 0 if attempt == 0 else (1 if attempt % 2 else -1)
            sigma_try = sigma + direction * ((attempt + 1) // 2) * float(settings.get("sigma_shift_fraction", 1.0e-8)) * width
            ncv = min(dim, base_ncv + attempt * int(settings.get("ncv_retry_growth", requested)))
            try:
                values, vectors = spla.eigsh(
                    H,
                    k=requested,
                    sigma=sigma_try,
                    which="LM",
                    tol=float(settings.get("tol", 1.0e-10)),
                    maxiter=int(settings.get("maxiter", 30000)),
                    ncv=ncv,
                )
                order = np.argsort(values)
                values, vectors = np.asarray(values)[order], np.asarray(vectors)[:, order]
                residuals = relative_residuals(H, values, vectors)
                orth_error = orthogonality_error(vectors)
                if float(np.max(residuals, initial=0.0)) > float(settings.get("residual_tol", 1.0e-9)):
                    raise RuntimeError("relative residual exceeds tolerance")
                if orth_error > float(settings.get("orthogonality_tol", 1.0e-8)):
                    raise RuntimeError("orthogonality error exceeds tolerance")
                attempts = attempt + 1
                sigma = sigma_try
                break
            except (spla.ArpackNoConvergence, RuntimeError, ValueError) as exc:
                last_error = exc
                retry_log.append(
                    {
                        "attempt": attempt + 1,
                        "sigma": float(sigma_try),
                        "ncv": int(ncv),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        else:
            raise RuntimeError(
                "shift-invert failed after {} attempts: {}; retry_log={}".format(
                    retries + 1, last_error, canonical_json(retry_log)
                )
            )
        backend_used = "scipy_shift_invert"
    residuals = relative_residuals(H, values, vectors)
    orth_error = orthogonality_error(vectors)
    elapsed = time.perf_counter() - started
    return {
        "energies": np.asarray(values, dtype=float),
        "vectors": np.asarray(vectors),
        "E_min": emin,
        "E_max": emax,
        "sigma": sigma,
        "residuals": residuals,
        "residual_max": float(np.max(residuals, initial=0.0)),
        "orthogonality_error": orth_error,
        "solver_attempts": int(attempts),
        "solver_retry_log": retry_log,
        "solver_backend": backend_used,
        "elapsed_s": elapsed,
    }


def select_central_states(solution: Dict[str, Any], n_states: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(solution["energies"])
    width = max(float(solution["E_max"] - solution["E_min"]), np.finfo(float).eps)
    normalized = (values - float(solution["E_min"])) / width
    count = min(int(n_states), values.size)
    selected = np.argsort(np.abs(normalized - 0.5), kind="stable")[:count]
    return values[selected], np.asarray(solution["vectors"])[:, selected]


def adjacent_gap_ratio(energies: np.ndarray, edge_discard: int = 2) -> Tuple[float, int]:
    values = np.sort(np.asarray(energies, dtype=float))
    discard = max(0, int(edge_discard))
    if 2 * discard < values.size - 2:
        values = values[discard : values.size - discard]
    gaps = np.diff(values)
    if gaps.size < 2:
        return float("nan"), 0
    denominator = np.maximum(gaps[:-1], gaps[1:])
    valid = denominator > 1.0e-12
    ratios = np.minimum(gaps[:-1], gaps[1:])[valid] / denominator[valid]
    return (float(np.mean(ratios)), int(ratios.size)) if ratios.size else (float("nan"), 0)


def state_observables(vectors: np.ndarray, interaction_q: np.ndarray, dim: int) -> Dict[str, Any]:
    probabilities = np.abs(np.asarray(vectors)) ** 2
    probabilities /= np.sum(probabilities, axis=0, keepdims=True)
    logarithm = np.zeros_like(probabilities)
    positive = probabilities > 0
    logarithm[positive] = np.log(probabilities[positive])
    entropy = -np.sum(probabilities * logarithm, axis=0)
    ipr = np.sum(probabilities**2, axis=0)
    entropy2 = -np.log(ipr)
    q_values = np.asarray(interaction_q, dtype=int)
    q_unique = np.arange(int(np.max(q_values, initial=0)) + 1)
    P_Q = np.zeros((q_unique.size, probabilities.shape[1]), dtype=float)
    conditional_entropy = np.zeros_like(P_Q)
    for q in q_unique:
        sector = probabilities[q_values == q]
        sector_weight = np.sum(sector, axis=0)
        P_Q[q] = sector_weight
        for column in np.flatnonzero(sector_weight > 0):
            conditional = sector[:, column] / sector_weight[column]
            nz = conditional > 0
            conditional_entropy[q, column] = -np.sum(conditional[nz] * np.log(conditional[nz]))
    log_PQ = np.zeros_like(P_Q)
    positive_q = P_Q > 0
    log_PQ[positive_q] = np.log(P_Q[positive_q])
    H_Q = -np.sum(P_Q * log_PQ, axis=0)
    S_intra_Q = np.sum(P_Q * conditional_entropy, axis=0)
    q_mean = np.sum(q_unique[:, None] * P_Q, axis=0)
    q_var = np.sum((q_unique[:, None] - q_mean[None, :]) ** 2 * P_Q, axis=0)
    identity_error = float(np.max(np.abs(entropy - H_Q - S_intra_Q), initial=0.0))
    return {
        "entropy": entropy,
        "entropy_norm": entropy / math.log(dim),
        "IPR": ipr,
        "entropy2": entropy2,
        "entropy2_norm": entropy2 / math.log(dim),
        "Q_mean": q_mean,
        "Q_var": q_var,
        "P_Q": P_Q,
        "H_Q": H_Q,
        "S_intra_Q": S_intra_Q,
        "identity_error": identity_error,
    }


def current_peak_memory_mb() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / (1024.0 if platform.system() != "Darwin" else 1024.0**2)
    except (ImportError, AttributeError):
        return float("nan")


def validate_dense_sparse(structure: BosonStructure, U: float, epsilon: np.ndarray, settings: Dict[str, Any]) -> Dict[str, float]:
    H = hamiltonian(structure, U, epsilon)
    dense_settings = dict(settings)
    dense_settings["backend"] = "dense"
    sparse_settings = dict(settings)
    sparse_settings["backend"] = "scipy"
    dense = solve_central_spectrum(H, dense_settings, max(structure.dim, 2))
    sparse = solve_central_spectrum(H, sparse_settings, max(structure.dim, 2))
    energy_error = float(np.max(np.abs(np.asarray(dense["energies"]) - np.asarray(sparse["energies"]))))
    count = selected_state_count(H.shape[0], settings)
    _, dense_vectors = select_central_states(dense, count)
    _, sparse_vectors = select_central_states(sparse, count)
    dense_obs = state_observables(dense_vectors, structure.interaction_q, structure.dim)
    sparse_obs = state_observables(sparse_vectors, structure.interaction_q, structure.dim)
    entropy_error = abs(float(np.mean(dense_obs["entropy_norm"])) - float(np.mean(sparse_obs["entropy_norm"])))
    dense_r, _ = adjacent_gap_ratio(dense["energies"], int(settings.get("gap_edge_discard", 2)))
    sparse_r, _ = adjacent_gap_ratio(sparse["energies"], int(settings.get("gap_edge_discard", 2)))
    return {"energy_error": energy_error, "entropy_error": entropy_error, "gap_ratio_error": abs(dense_r - sparse_r)}
