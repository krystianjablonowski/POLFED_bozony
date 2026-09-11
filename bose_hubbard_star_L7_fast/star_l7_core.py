#!/usr/bin/env python3
"""Vectorized depth-one star model for the disordered Bose-Hubbard chain."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np


VERSION = "3.0.0"


@dataclass(frozen=True)
class Graph:
    basis: np.ndarray
    interaction: np.ndarray
    neighbors: np.ndarray
    couplings: np.ndarray
    channels: np.ndarray
    degree: np.ndarray
    from_site: np.ndarray
    to_site: np.ndarray

    @property
    def dim(self) -> int:
        return int(len(self.basis))


def generate_basis(L: int, N: int, nmax: int) -> np.ndarray:
    states: list[tuple[int, ...]] = []

    def rec(prefix: list[int], remaining: int) -> None:
        site = len(prefix)
        if site == L - 1:
            if 0 <= remaining <= nmax:
                states.append(tuple(prefix + [remaining]))
            return
        lo = max(0, remaining - (L - site - 1) * nmax)
        hi = min(nmax, remaining)
        for value in range(lo, hi + 1):
            rec(prefix + [value], remaining - value)

    if L < 2 or N < 0 or nmax < 1 or N > L * nmax:
        raise ValueError("Require L>=2, N>=0, nmax>=1 and N<=L*nmax")
    rec([], N)
    return np.asarray(states, dtype=np.int16)


def build_graph(L: int, N: int, nmax: int, t: float = 1.0) -> Graph:
    basis = generate_basis(L, N, nmax)
    index = {tuple(map(int, state)): i for i, state in enumerate(basis)}
    rows: list[list[tuple[int, float, int, int, int]]] = []
    for state in basis:
        edges = []
        for bond in range(L - 1):
            for source, target in ((bond, bond + 1), (bond + 1, bond)):
                ni, nj = int(state[source]), int(state[target])
                if ni == 0 or nj >= nmax:
                    continue
                moved = state.copy()
                moved[source] -= 1
                moved[target] += 1
                edges.append((index[tuple(map(int, moved))], -t * math.sqrt(ni * (nj + 1)), nj - ni + 1, source, target))
        rows.append(edges)
    degree = np.asarray([len(row) for row in rows], dtype=np.int16)
    zmax = int(np.max(degree))
    neighbors = np.zeros((len(basis), zmax), dtype=np.int32)
    couplings = np.zeros((len(basis), zmax), dtype=np.float64)
    channels = np.zeros((len(basis), zmax), dtype=np.int16)
    from_site = np.zeros((len(basis), zmax), dtype=np.int16)
    to_site = np.zeros((len(basis), zmax), dtype=np.int16)
    for alpha, row in enumerate(rows):
        for edge, (beta, value, channel, source, target) in enumerate(row):
            neighbors[alpha, edge] = beta
            couplings[alpha, edge] = value
            channels[alpha, edge] = channel
            from_site[alpha, edge] = source
            to_site[alpha, edge] = target
    occupations = basis.astype(np.int64)
    interaction = np.sum(occupations * (occupations - 1), axis=1).astype(np.int32)
    return Graph(basis, interaction, neighbors, couplings, channels, degree, from_site, to_site)


def central_configurations(
    energies: np.ndarray,
    selection: Union[dict, float],
    maximum: int,
    priority_rank: np.ndarray,
) -> np.ndarray:
    """Select diagonal configurations around normalized spectral center 1/2.

    A numeric ``selection`` keeps the legacy rank-fraction mode only for
    regression tests. Production configurations must name their mode.
    """
    dim = len(energies)
    if isinstance(selection, (int, float)):
        mode = "rank_fraction"
        settings = {"fraction": float(selection)}
    else:
        settings = selection
        mode = str(settings["mode"])
    width = float(np.max(energies) - np.min(energies))
    normalized = ((energies - float(np.min(energies))) / width
                  if width > np.finfo(float).eps else np.full(dim, 0.5))
    distance = np.abs(normalized - float(settings.get("center", 0.5)))
    order = np.argsort(distance, kind="stable")
    if mode == "closest_count":
        count = max(1, min(dim, int(settings["count"])))
        eligible = order[:count]
    elif mode == "normalized_half_width":
        half_width = float(settings["half_width"])
        if not 0.0 < half_width <= 0.5:
            raise ValueError("normalized_half_width must be in (0, 0.5]")
        eligible = np.flatnonzero(distance <= half_width)
        if len(eligible) == 0:
            eligible = order[:1]
    elif mode == "rank_fraction":
        fraction = float(settings["fraction"])
        count = max(1, min(dim, int(math.ceil(fraction * dim))))
        eligible = order[:count]
    else:
        raise ValueError(f"Unknown central selection mode: {mode}")
    if maximum <= 0 or len(eligible) <= maximum:
        return eligible
    return eligible[np.argsort(priority_rank[eligible], kind="stable")[:maximum]]


def compensating_channels(graph: Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (k, |V|, q) for the original positive-k mixing model."""
    records: dict[tuple[int, float], int] = {}
    # One representative bond is sufficient; q counts Fock configurations.
    for alpha in range(graph.dim):
        for edge in range(int(graph.degree[alpha])):
            if graph.from_site[alpha, edge] != 0 or graph.to_site[alpha, edge] != 1:
                continue
            k = int(graph.channels[alpha, edge])
            if k > 0:
                coupling = float(abs(graph.couplings[alpha, edge]))
                records[(k, coupling)] = records.get((k, coupling), 0) + 1
    if not records:
        raise ValueError("No positive compensating channels")
    ordered = sorted(records)
    counts = np.asarray([records[key] for key in ordered], dtype=float)
    return (np.asarray([key[0] for key in ordered], dtype=np.int16),
            np.asarray([key[1] for key in ordered], dtype=float), counts / np.sum(counts))


def compensating_observables(
    U: float,
    epsilon: np.ndarray,
    graph: Graph,
    channel_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[float, float]:
    """Existing project definition: k>0 channels and |epsilon_i-epsilon_j|."""
    k, coupling, weight = channel_data if channel_data is not None else compensating_channels(graph)
    bond_disorder = np.abs(np.diff(np.asarray(epsilon, dtype=float)))
    delta = 2.0 * k[:, None] * float(U) - bond_disorder[None, :]
    v2 = coupling[:, None] ** 2
    mixing = 4.0 * v2 / (delta * delta + 4.0 * v2)
    entropy = two_level_entropy(mixing)
    return (float(np.sum(weight[:, None] * mixing) / len(bond_disorder)),
            float(np.sum(weight[:, None] * entropy) / len(bond_disorder)))


def entropy_from_probabilities(probabilities: np.ndarray, axis: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    terms = np.zeros_like(p)
    positive = p > 0.0
    terms[positive] = -p[positive] * np.log(p[positive])
    return np.sum(terms, axis=axis)


def two_level_entropy(mixing: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.maximum(0.0, 1.0 - mixing))
    p = 0.5 * (1.0 + root)
    return entropy_from_probabilities(np.stack((p, 1.0 - p), axis=-1), axis=-1)


def batch_star_metrics(delta: np.ndarray, coupling: np.ndarray, tie_tolerance: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Diagonalize a batch of equal-degree stars in one NumPy call."""
    count, z = delta.shape
    matrices = np.zeros((count, z + 1, z + 1), dtype=np.float64)
    diagonal = np.arange(1, z + 1)
    matrices[:, diagonal, diagonal] = delta
    matrices[:, 0, 1:] = coupling
    matrices[:, 1:, 0] = coupling
    _, vectors = np.linalg.eigh(matrices)
    probabilities = vectors * vectors
    overlaps = probabilities[:, 0, :]
    selected = overlaps >= np.max(overlaps, axis=1, keepdims=True) - tie_tolerance
    entropy_by_state = entropy_from_probabilities(probabilities, axis=1)
    participation_by_state = 1.0 / np.sum(probabilities * probabilities, axis=1)
    denominator = np.sum(selected, axis=1)
    entropy = np.sum(entropy_by_state * selected, axis=1) / denominator
    participation = np.sum(participation_by_state * selected, axis=1) / denominator
    normalization_error = float(np.max(np.abs(np.sum(probabilities, axis=1) - 1.0)))
    return entropy, participation, normalization_error


def scalar_star_metrics(delta: np.ndarray, coupling: np.ndarray, tie_tolerance: float = 1e-12) -> Tuple[float, float, float]:
    """Slow reference implementation used only by the validation suite."""
    delta = np.asarray(delta, dtype=float)
    coupling = np.asarray(coupling, dtype=float)
    if delta.ndim != 1 or coupling.shape != delta.shape:
        raise ValueError("delta and coupling must be one-dimensional and have equal shape")
    matrix = np.zeros((len(delta) + 1, len(delta) + 1), dtype=float)
    matrix[1:, 1:] = np.diag(delta)
    matrix[0, 1:] = coupling
    matrix[1:, 0] = coupling
    _, vectors = np.linalg.eigh(matrix)
    probabilities = vectors * vectors
    overlaps = probabilities[0]
    selected = overlaps >= np.max(overlaps) - tie_tolerance
    entropy = entropy_from_probabilities(probabilities, axis=0)
    participation = 1.0 / np.sum(probabilities * probabilities, axis=0)
    norm_error = float(np.max(np.abs(np.sum(probabilities, axis=0) - 1.0)))
    return float(np.mean(entropy[selected])), float(np.mean(participation[selected])), norm_error


def observables_at_u(
    graph: Graph,
    U: float,
    disorder_energy: np.ndarray,
    central: np.ndarray,
    tie_tolerance: float = 1e-12,
) -> Dict[str, object]:
    energies = U * graph.interaction + disorder_energy
    m_per_star = np.zeros(len(central), dtype=float)
    s2_per_star = np.zeros(len(central), dtype=float)
    star_entropy = np.zeros(len(central), dtype=float)
    star_participation = np.zeros(len(central), dtype=float)
    resonant = np.zeros(len(central), dtype=float)
    max_norm_error = 0.0
    channel_m: dict[int, float] = {}
    channel_s2: dict[int, float] = {}
    channel_fraction: dict[int, float] = {}

    central_degrees = graph.degree[central]
    for z in np.unique(central_degrees):
        positions = np.flatnonzero(central_degrees == z)
        alpha = central[positions]
        neighbor = graph.neighbors[alpha, :z]
        coupling = graph.couplings[alpha, :z]
        delta = energies[neighbor] - energies[alpha, None]
        v2 = coupling * coupling
        mixing = 4.0 * v2 / (delta * delta + 4.0 * v2)
        s2 = two_level_entropy(mixing)
        entropy, participation, norm_error = batch_star_metrics(delta, coupling, tie_tolerance)
        m_per_star[positions] = np.sum(mixing, axis=1)
        s2_per_star[positions] = np.sum(s2, axis=1)
        star_entropy[positions] = entropy
        star_participation[positions] = participation
        resonant[positions] = np.sum(np.abs(delta) < 2.0 * np.abs(coupling), axis=1)
        channels = graph.channels[alpha, :z]
        for channel in np.unique(channels):
            mask = channels == channel
            per_star_count = np.sum(mask, axis=1)
            channel_m[int(channel)] = channel_m.get(int(channel), 0.0) + float(np.sum(mixing * mask))
            channel_s2[int(channel)] = channel_s2.get(int(channel), 0.0) + float(np.sum(s2 * mask))
            channel_fraction[int(channel)] = channel_fraction.get(int(channel), 0.0) + float(np.sum(per_star_count / z))
        max_norm_error = max(max_norm_error, norm_error)
    return {
        "M_sum": float(np.mean(m_per_star)),
        "M_per_edge": float(np.mean(m_per_star / central_degrees)),
        "S2_sum": float(np.mean(s2_per_star)),
        "S2_per_edge": float(np.mean(s2_per_star / central_degrees)),
        "Sstar": float(np.mean(star_entropy)),
        "Sstar_median": float(np.median(star_entropy)),
        "Sstar_std": float(np.std(star_entropy, ddof=1)) if len(star_entropy) > 1 else 0.0,
        "Pstar": float(np.mean(star_participation)),
        "z_mean": float(np.mean(central_degrees)),
        "resonant_edges": float(np.mean(resonant)),
        "normalization_error": max_norm_error,
        "Sstar_values": star_entropy,
        "M_k": {k: value / len(central) for k, value in channel_m.items()},
        "S2_k": {k: value / len(central) for k, value in channel_s2.items()},
        "P_k": {k: value / len(central) for k, value in channel_fraction.items()},
    }


def verify_detuning(graph: Graph, U: float, epsilon: np.ndarray, seed: int = 0) -> float:
    disorder_energy = graph.basis @ epsilon
    energies = U * graph.interaction + disorder_energy
    rng = np.random.default_rng(seed)
    alphas = rng.integers(0, graph.dim, size=min(100, graph.dim))
    errors = []
    for alpha in alphas:
        z = int(graph.degree[alpha])
        edge = int(rng.integers(0, z))
        beta = graph.neighbors[alpha, edge]
        direct = energies[beta] - energies[alpha]
        formula = (epsilon[graph.to_site[alpha, edge]] - epsilon[graph.from_site[alpha, edge]]
                   + 2.0 * graph.channels[alpha, edge] * U)
        errors.append(abs(direct - formula))
    return float(max(errors, default=0.0))


def deterministic_seed(master: int, L: int, sample_id: int) -> int:
    """Match the ED convention SeedSequence([master_seed, L, realization])."""
    words = np.random.SeedSequence([master, L, sample_id]).generate_state(2, dtype=np.uint32)
    return int(words[0]) | (int(words[1]) << 32)


def disorder_vector(master: int, L: int, sample_id: int) -> np.ndarray:
    """Return exactly the unit disorder vector used by the ED workflow."""
    sequence = np.random.SeedSequence([int(master), int(L), int(sample_id)])
    return np.random.default_rng(sequence).uniform(-1.0, 1.0, size=L)


def configuration_priority(basis: np.ndarray, seed: int) -> np.ndarray:
    """Order-independent pseudorandom score attached to each occupation tuple."""
    scores = np.full(len(basis), np.uint64(seed), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for site in range(basis.shape[1]):
            value = basis[:, site].astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15 + site)
            scores ^= value + (scores << np.uint64(6)) + (scores >> np.uint64(2))
            scores *= np.uint64(0xBF58476D1CE4E5B9)
    return scores
