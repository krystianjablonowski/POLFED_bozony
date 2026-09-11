#!/usr/bin/env python3
"""Core routines for the depth-one Bose-Hubbard star model.

No full many-body Hamiltonian is constructed or diagonalized here.  The only
eigensystems are local (1 + coordination)-dimensional arrowhead matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


VERSION = "1.0.0"


@dataclass(frozen=True)
class FockGraph:
    basis: np.ndarray
    interaction_count: np.ndarray
    edge_ptr: np.ndarray
    edge_dst: np.ndarray
    edge_v: np.ndarray
    edge_k: np.ndarray
    edge_from_site: np.ndarray
    edge_to_site: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[0])


def generate_basis(L: int, N: int, nmax: int, reverse: bool = False) -> np.ndarray:
    """Generate all bounded weak compositions of N into L entries."""
    if L < 1 or N < 0 or nmax < 0 or N > L * nmax:
        raise ValueError("Require L>=1, N>=0, nmax>=0 and N<=L*nmax")
    states: list[tuple[int, ...]] = []

    def rec(prefix: list[int], remaining: int) -> None:
        site = len(prefix)
        if site == L - 1:
            if remaining <= nmax:
                states.append(tuple(prefix + [remaining]))
            return
        lo = max(0, remaining - (L - site - 1) * nmax)
        hi = min(nmax, remaining)
        values: Iterable[int] = range(lo, hi + 1)
        if reverse:
            values = reversed(range(lo, hi + 1))
        for value in values:
            rec(prefix + [value], remaining - value)

    rec([], N)
    return np.asarray(states, dtype=np.int16)


def build_fock_graph(L: int, N: int, nmax: int, t: float = 1.0, reverse_basis: bool = False) -> FockGraph:
    basis = generate_basis(L, N, nmax, reverse=reverse_basis)
    index = {tuple(int(x) for x in state): i for i, state in enumerate(basis)}
    ptr = [0]
    dst: list[int] = []
    amps: list[float] = []
    channels: list[int] = []
    from_sites: list[int] = []
    to_sites: list[int] = []
    for state in basis:
        for bond in range(L - 1):
            for source, target in ((bond, bond + 1), (bond + 1, bond)):
                ni, nj = int(state[source]), int(state[target])
                if ni == 0 or nj >= nmax:
                    continue
                moved = state.copy()
                moved[source] -= 1
                moved[target] += 1
                dst.append(index[tuple(int(x) for x in moved)])
                amps.append(-float(t) * math.sqrt(ni * (nj + 1)))
                channels.append(nj - ni + 1)
                from_sites.append(source)
                to_sites.append(target)
        ptr.append(len(dst))
    interaction = np.sum(basis.astype(np.int64) * (basis.astype(np.int64) - 1), axis=1)
    return FockGraph(
        basis=basis,
        interaction_count=interaction.astype(np.int32),
        edge_ptr=np.asarray(ptr, dtype=np.int64),
        edge_dst=np.asarray(dst, dtype=np.int32),
        edge_v=np.asarray(amps, dtype=np.float64),
        edge_k=np.asarray(channels, dtype=np.int16),
        edge_from_site=np.asarray(from_sites, dtype=np.int16),
        edge_to_site=np.asarray(to_sites, dtype=np.int16),
    )


def diagonal_energies(graph: FockGraph, U: float, epsilon: np.ndarray) -> np.ndarray:
    return float(U) * graph.interaction_count + graph.basis @ np.asarray(epsilon, dtype=float)


def microcanonical_indices(
    energies: np.ndarray,
    fraction: float,
    max_configurations: int,
    priority: np.ndarray | None = None,
) -> np.ndarray:
    """Select a central rank window, optionally subsampled by fixed priorities."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("energy-window fraction must lie in (0,1]")
    dim = len(energies)
    count = max(1, min(dim, int(math.ceil(fraction * dim))))
    order = np.argsort(energies, kind="stable")
    start = (dim - count) // 2
    eligible = order[start : start + count]
    if max_configurations <= 0 or len(eligible) <= max_configurations:
        return np.asarray(eligible, dtype=np.int64)
    if priority is None:
        return np.asarray(eligible[:max_configurations], dtype=np.int64)
    chosen = np.argsort(priority[eligible], kind="stable")[:max_configurations]
    return np.asarray(eligible[chosen], dtype=np.int64)


def binary_entropy_from_m(m: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.maximum(0.0, 1.0 - np.asarray(m, dtype=float)))
    p = 0.5 * (1.0 + root)
    q = 1.0 - p
    out = np.zeros_like(p)
    mask = p > 0.0
    out[mask] -= p[mask] * np.log(p[mask])
    mask = q > 0.0
    out[mask] -= q[mask] * np.log(q[mask])
    return out


def shannon(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    terms = np.zeros_like(p)
    mask = p > 0.0
    terms[mask] = -p[mask] * np.log(p[mask])
    return np.sum(terms, axis=axis)


def star_state_metrics(delta: np.ndarray, couplings: np.ndarray, tie_tolerance: float = 1e-12) -> tuple[float, float, float]:
    """Return entropy, participation number, and normalization error.

    If central overlaps tie within tie_tolerance, entropy and participation are
    averaged over all tied eigenvectors.  This is deterministic at exact
    resonances and is recorded as ``average_ties`` in result metadata.
    """
    z = len(delta)
    matrix = np.zeros((z + 1, z + 1), dtype=float)
    matrix[1:, 1:] = np.diag(delta)
    matrix[0, 1:] = couplings
    matrix[1:, 0] = couplings
    _, vectors = np.linalg.eigh(matrix)
    overlaps = vectors[0, :] ** 2
    best = float(np.max(overlaps))
    selected = np.flatnonzero(best - overlaps <= tie_tolerance)
    probs = vectors[:, selected] ** 2
    entropy = float(np.mean(shannon(probs, axis=0)))
    participation = float(np.mean(1.0 / np.sum(probs * probs, axis=0)))
    norm_error = float(np.max(np.abs(np.sum(probs, axis=0) - 1.0)))
    return entropy, participation, norm_error


def compute_observables(
    graph: FockGraph,
    U: float,
    epsilon: np.ndarray,
    central_indices: np.ndarray,
    tie_tolerance: float = 1e-12,
) -> dict[str, object]:
    energies = diagonal_energies(graph, U, epsilon)
    m_sums: list[float] = []
    s2_sums: list[float] = []
    star_values: list[float] = []
    pstar_values: list[float] = []
    z_values: list[int] = []
    resonant_values: list[int] = []
    all_m: list[np.ndarray] = []
    all_s2: list[np.ndarray] = []
    channel_m: dict[int, float] = {}
    channel_s2: dict[int, float] = {}
    channel_count: dict[int, int] = {}
    max_norm_error = 0.0

    for alpha in np.asarray(central_indices, dtype=np.int64):
        lo, hi = int(graph.edge_ptr[alpha]), int(graph.edge_ptr[alpha + 1])
        dst = graph.edge_dst[lo:hi]
        coupling = graph.edge_v[lo:hi]
        channels = graph.edge_k[lo:hi]
        delta = energies[dst] - energies[alpha]
        v2 = coupling * coupling
        m = 4.0 * v2 / (delta * delta + 4.0 * v2)
        s2 = binary_entropy_from_m(m)
        entropy, participation, norm_error = star_state_metrics(delta, coupling, tie_tolerance)
        max_norm_error = max(max_norm_error, norm_error)
        m_sums.append(float(np.sum(m)))
        s2_sums.append(float(np.sum(s2)))
        star_values.append(entropy)
        pstar_values.append(participation)
        z_values.append(len(dst))
        resonant_values.append(int(np.sum(np.abs(delta) < 2.0 * np.abs(coupling))))
        all_m.append(m)
        all_s2.append(s2)
        for channel in np.unique(channels):
            mask = channels == channel
            key = int(channel)
            channel_m[key] = channel_m.get(key, 0.0) + float(np.sum(m[mask]))
            channel_s2[key] = channel_s2.get(key, 0.0) + float(np.sum(s2[mask]))
            channel_count[key] = channel_count.get(key, 0) + int(np.sum(mask))

    ncfg = len(central_indices)
    edge_count = int(sum(z_values))
    flat_m = np.concatenate(all_m) if all_m else np.empty(0)
    flat_s2 = np.concatenate(all_s2) if all_s2 else np.empty(0)
    star = np.asarray(star_values, dtype=float)
    return {
        "M_sum": float(np.mean(m_sums)) if ncfg else math.nan,
        "M_per_edge": float(np.mean(flat_m)) if edge_count else 0.0,
        "S2_sum": float(np.mean(s2_sums)) if ncfg else math.nan,
        "S2_per_edge": float(np.mean(flat_s2)) if edge_count else 0.0,
        "Sstar_mean": float(np.mean(star)) if ncfg else math.nan,
        "Sstar_median": float(np.median(star)) if ncfg else math.nan,
        "Sstar_std": float(np.std(star, ddof=1)) if ncfg > 1 else 0.0,
        "Pstar_mean": float(np.mean(pstar_values)) if ncfg else math.nan,
        "z_mean": float(np.mean(z_values)) if ncfg else math.nan,
        "resonant_edges_mean": float(np.mean(resonant_values)) if ncfg else math.nan,
        "number_of_configurations": ncfg,
        "number_of_edges": edge_count,
        "max_normalization_error": max_norm_error,
        "Sstar_local": star,
        "channel_M_sum": channel_m,
        "channel_S2_sum": channel_s2,
        "channel_count": channel_count,
    }


def verify_detuning(graph: FockGraph, U: float, epsilon: np.ndarray, sample_size: int = 100, seed: int = 0) -> float:
    energies = diagonal_energies(graph, U, epsilon)
    rng = np.random.default_rng(seed)
    count = min(sample_size, len(graph.edge_dst))
    edges = rng.choice(len(graph.edge_dst), size=count, replace=False)
    sources = np.searchsorted(graph.edge_ptr[1:], edges, side="right")
    direct = energies[graph.edge_dst[edges]] - energies[sources]
    formula = (
        epsilon[graph.edge_to_site[edges]]
        - epsilon[graph.edge_from_site[edges]]
        + 2.0 * graph.edge_k[edges] * U
    )
    return float(np.max(np.abs(direct - formula))) if count else 0.0


def deterministic_seed(master_seed: int, L: int, N: int, nmax: int, sample_id: int) -> int:
    words = np.random.SeedSequence([master_seed, L, N, nmax, sample_id]).generate_state(2, dtype=np.uint32)
    return int(words[0]) | (int(words[1]) << 32)

