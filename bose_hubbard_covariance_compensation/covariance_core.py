#!/usr/bin/env python3
"""Numerical core for the covariance-compensation predictor.

Hamiltonian convention:
    H = -t hopping + sum_i epsilon_i n_i + U sum_i n_i(n_i-1),
    epsilon_i ~ Uniform[-W, W], open boundary conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


VERSION = "1.0.0"


@dataclass
class Structure:
    basis: np.ndarray
    interaction_q: np.ndarray
    hopping: sp.csr_matrix
    edge_alpha: np.ndarray
    edge_beta: np.ndarray
    edge_t: np.ndarray
    level_bases: List[np.ndarray]
    creation_targets: List[np.ndarray]
    creation_factors: List[np.ndarray]

    @property
    def dim(self) -> int:
        return int(self.basis.shape[0])

    @property
    def L(self) -> int:
        return int(self.basis.shape[1])


@dataclass
class SelectedStates:
    energies: np.ndarray
    energy_density: np.ndarray
    vectors: np.ndarray
    residual_max: float
    orthogonality_error: float
    construction: str


def generate_basis(L: int, N: int, nmax: int) -> np.ndarray:
    if L < 2 or N < 0 or nmax < 1 or N > L * nmax:
        raise ValueError("Require L>=2, N>=0, nmax>=1 and N<=L*nmax")
    states: List[Tuple[int, ...]] = []

    def recurse(prefix: List[int], remaining: int) -> None:
        site = len(prefix)
        if site == L - 1:
            if 0 <= remaining <= nmax:
                states.append(tuple(prefix + [remaining]))
            return
        lower = max(0, remaining - (L - site - 1) * nmax)
        upper = min(nmax, remaining)
        for value in range(lower, upper + 1):
            recurse(prefix + [value], remaining - value)

    recurse([], N)
    return np.asarray(states, dtype=np.int16)


def build_structure(L: int, N: int, nmax: int, t: float = 1.0) -> Structure:
    basis = generate_basis(L, N, nmax)
    index = {tuple(int(x) for x in state): i for i, state in enumerate(basis)}
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    edge_alpha: List[int] = []
    edge_beta: List[int] = []
    edge_t: List[float] = []
    for alpha, state in enumerate(basis):
        for bond in range(L - 1):
            for source, target in ((bond, bond + 1), (bond + 1, bond)):
                source_n = int(state[source])
                target_n = int(state[target])
                if source_n == 0 or target_n >= nmax:
                    continue
                moved = state.copy()
                moved[source] -= 1
                moved[target] += 1
                beta = index[tuple(int(x) for x in moved)]
                matrix_element = -float(t) * math.sqrt(source_n * (target_n + 1))
                rows.append(beta)
                cols.append(alpha)
                values.append(matrix_element)
                if alpha < beta:
                    edge_alpha.append(alpha)
                    edge_beta.append(beta)
                    edge_t.append(matrix_element)
    hopping = sp.coo_matrix((values, (rows, cols)), shape=(len(basis), len(basis))).tocsr()
    difference = hopping - hopping.T
    if difference.nnz and float(np.max(np.abs(difference.data))) > 1.0e-12:
        raise AssertionError("Hopping matrix is not symmetric")
    occupations = basis.astype(np.int64)
    interaction_q = np.sum(occupations * (occupations - 1), axis=1).astype(float)

    level_bases = [generate_basis(L, particles, nmax) for particles in range(N + 1)]
    creation_targets: List[np.ndarray] = []
    creation_factors: List[np.ndarray] = []
    for particles in range(N):
        current = level_bases[particles]
        following = level_bases[particles + 1]
        following_index = {tuple(int(x) for x in state): i for i, state in enumerate(following)}
        targets = np.full((L, len(current)), -1, dtype=np.int32)
        factors = np.zeros((L, len(current)), dtype=float)
        for site in range(L):
            for source_index, state in enumerate(current):
                if int(state[site]) >= nmax:
                    continue
                target_state = state.copy()
                target_state[site] += 1
                targets[site, source_index] = following_index[tuple(int(x) for x in target_state)]
                factors[site, source_index] = math.sqrt(int(state[site]) + 1)
        creation_targets.append(targets)
        creation_factors.append(factors)
    return Structure(
        basis=basis,
        interaction_q=interaction_q,
        hopping=hopping,
        edge_alpha=np.asarray(edge_alpha, dtype=np.int32),
        edge_beta=np.asarray(edge_beta, dtype=np.int32),
        edge_t=np.asarray(edge_t, dtype=float),
        level_bases=level_bases,
        creation_targets=creation_targets,
        creation_factors=creation_factors,
    )


def disorder_eta(master_seed: int, L: int, sample_id: int) -> np.ndarray:
    """Match SeedSequence([master_seed, L, realization]) used by entropy ED."""
    sequence = np.random.SeedSequence([int(master_seed), int(L), int(sample_id)])
    return np.random.default_rng(sequence).uniform(-1.0, 1.0, size=L)


def single_particle_hamiltonian(L: int, t: float, epsilon: np.ndarray) -> np.ndarray:
    epsilon = np.asarray(epsilon, dtype=float)
    if epsilon.shape != (L,):
        raise ValueError("epsilon must have shape (L,)")
    matrix = np.diag(epsilon)
    off_diagonal = -float(t) * np.ones(L - 1, dtype=float)
    matrix += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    return matrix


def select_by_energy_density(
    energies: np.ndarray, count: int, center: float
) -> Tuple[np.ndarray, np.ndarray]:
    energies = np.asarray(energies, dtype=float)
    width = float(np.max(energies) - np.min(energies))
    density = (
        (energies - float(np.min(energies))) / width
        if width > np.finfo(float).eps
        else np.full_like(energies, 0.5)
    )
    order = np.argsort(np.abs(density - float(center)), kind="stable")
    selected = order[: max(1, min(int(count), len(order)))]
    return selected, density[selected]


def anderson_fock_vector(
    orbital_occupations: np.ndarray,
    one_particle_vectors: np.ndarray,
    structure: Structure,
) -> np.ndarray:
    """Expand one normalized Anderson-orbital Fock state in the site basis."""
    orbital_occupations = np.asarray(orbital_occupations, dtype=int)
    if int(np.sum(orbital_occupations)) != len(structure.creation_targets):
        raise ValueError("orbital occupations have the wrong particle number")
    vector = np.ones(1, dtype=float)
    particle_number = 0
    for orbital, multiplicity in enumerate(orbital_occupations):
        for repetition in range(1, int(multiplicity) + 1):
            target_size = len(structure.level_bases[particle_number + 1])
            following = np.zeros(target_size, dtype=float)
            targets = structure.creation_targets[particle_number]
            factors = structure.creation_factors[particle_number]
            normalization = math.sqrt(repetition)
            for site in range(structure.L):
                valid = targets[site] >= 0
                np.add.at(
                    following,
                    targets[site, valid],
                    vector[valid] * factors[site, valid]
                    * float(one_particle_vectors[site, orbital]) / normalization,
                )
            vector = following
            particle_number += 1
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps:
        raise RuntimeError("constructed Anderson state has zero norm")
    return vector / norm


def exact_noninteracting_states(
    structure: Structure,
    N: int,
    nmax: int,
    t: float,
    epsilon: np.ndarray,
    count: int,
    center: float,
) -> SelectedStates:
    if nmax < N:
        raise ValueError("orbital construction requires nmax>=N")
    one_body = single_particle_hamiltonian(structure.L, t, epsilon)
    orbital_energies, orbital_vectors = np.linalg.eigh(one_body)
    orbital_basis = generate_basis(structure.L, N, N)
    many_body_energies = orbital_basis @ orbital_energies
    selected, density = select_by_energy_density(many_body_energies, count, center)
    vectors = np.column_stack(
        [anderson_fock_vector(orbital_basis[index], orbital_vectors, structure) for index in selected]
    )
    disorder_energy = structure.basis @ np.asarray(epsilon, dtype=float)
    h0 = structure.hopping + sp.diags(disorder_energy, format="csr")
    residuals = h0 @ vectors - vectors * many_body_energies[selected][None, :]
    scale = np.maximum(1.0, np.abs(many_body_energies[selected]))
    relative = np.linalg.norm(residuals, axis=0) / scale
    gram = vectors.T @ vectors
    orthogonality = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
    return SelectedStates(
        energies=many_body_energies[selected],
        energy_density=density,
        vectors=vectors,
        residual_max=float(np.max(relative, initial=0.0)),
        orthogonality_error=orthogonality,
        construction="anderson_orbitals",
    )


def truncated_noninteracting_states(
    structure: Structure,
    epsilon: np.ndarray,
    count: int,
    center: float,
    dense_max: int,
    tolerance: float,
) -> SelectedStates:
    """Fallback for nmax<N: diagonalize H0 in the truncated site basis."""
    disorder_energy = structure.basis @ np.asarray(epsilon, dtype=float)
    h0 = structure.hopping + sp.diags(disorder_energy, format="csr")
    dim = structure.dim
    if dim <= int(dense_max) or count >= dim - 1:
        all_energies, all_vectors = np.linalg.eigh(h0.toarray())
        selected, density = select_by_energy_density(all_energies, count, center)
        energies = all_energies[selected]
        vectors = all_vectors[:, selected]
        construction = "truncated_dense"
    else:
        minimum = float(spla.eigsh(h0, k=1, which="SA", return_eigenvectors=False, tol=tolerance)[0])
        maximum = float(spla.eigsh(h0, k=1, which="LA", return_eigenvectors=False, tol=tolerance)[0])
        sigma = minimum + float(center) * (maximum - minimum)
        requested = min(dim - 2, max(int(count) + 8, 2 * int(count)))
        values, candidate_vectors = spla.eigsh(
            h0, k=requested, sigma=sigma + 1.0e-12 * max(1.0, abs(sigma)),
            which="LM", tol=tolerance,
        )
        order = np.argsort(values, kind="stable")
        values = values[order]
        candidate_vectors = candidate_vectors[:, order]
        density_all = (values - minimum) / max(maximum - minimum, np.finfo(float).eps)
        nearest = np.argsort(np.abs(density_all - float(center)), kind="stable")[:count]
        energies = values[nearest]
        vectors = candidate_vectors[:, nearest]
        density = density_all[nearest]
        construction = "truncated_sparse"
    residuals = h0 @ vectors - vectors * energies[None, :]
    scale = np.maximum(1.0, np.abs(energies))
    relative = np.linalg.norm(residuals, axis=0) / scale
    gram = vectors.T @ vectors
    orthogonality = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
    return SelectedStates(
        energies=energies,
        energy_density=density,
        vectors=vectors,
        residual_max=float(np.max(relative, initial=0.0)),
        orthogonality_error=orthogonality,
        construction=construction,
    )


def selected_u0_states(
    structure: Structure,
    N: int,
    nmax: int,
    t: float,
    epsilon: np.ndarray,
    count: int,
    center: float,
    dense_max: int = 600,
    tolerance: float = 1.0e-10,
) -> SelectedStates:
    if nmax >= N:
        return exact_noninteracting_states(structure, N, nmax, t, epsilon, count, center)
    return truncated_noninteracting_states(structure, epsilon, count, center, dense_max, tolerance)


def covariance_observables(
    structure: Structure,
    epsilon: np.ndarray,
    states: SelectedStates,
    variance_tolerance: float,
) -> Dict[str, np.ndarray]:
    probabilities = np.abs(states.vectors) ** 2
    probabilities /= np.sum(probabilities, axis=0, keepdims=True)
    disorder_energy = structure.basis @ np.asarray(epsilon, dtype=float)
    q = structure.interaction_q
    mean_e = probabilities.T @ disorder_energy
    mean_q = probabilities.T @ q
    cov = probabilities.T @ (disorder_energy * q) - mean_e * mean_q
    var_q = probabilities.T @ (q * q) - mean_q * mean_q
    var_q = np.maximum(var_q, 0.0)
    valid = var_q > float(variance_tolerance)
    u_comp = np.full(len(var_q), np.nan, dtype=float)
    u_comp[valid] = -cov[valid] / var_q[valid]

    alpha = structure.edge_alpha
    beta = structure.edge_beta
    hopping_squared = structure.edge_t ** 2
    delta_e = disorder_energy[beta] - disorder_energy[alpha]
    delta_q = q[beta] - q[alpha]
    edge_numerator = np.zeros(len(var_q), dtype=float)
    edge_denominator = np.zeros(len(var_q), dtype=float)
    for state_index in range(len(var_q)):
        weights = (probabilities[alpha, state_index] + probabilities[beta, state_index]) * hopping_squared
        edge_numerator[state_index] = float(np.sum(weights * delta_e * delta_q))
        edge_denominator[state_index] = float(np.sum(weights * delta_q * delta_q))
    edge_valid = edge_denominator > float(variance_tolerance)
    u_edge = np.full(len(var_q), np.nan, dtype=float)
    u_edge[edge_valid] = -edge_numerator[edge_valid] / edge_denominator[edge_valid]
    return {
        "probability_norm_error": np.asarray(
            [float(np.max(np.abs(np.sum(probabilities, axis=0) - 1.0)))], dtype=float
        ),
        "mean_Edis": mean_e,
        "mean_Q": mean_q,
        "cov_Edis_Q": cov,
        "var_Q": var_q,
        "U_comp_state": u_comp,
        "valid_state": valid,
        "edge_numerator": edge_numerator,
        "edge_denominator": edge_denominator,
        "U_edge_state": u_edge,
        "valid_edge_state": edge_valid,
    }


def variance_of_diagonal_energy(
    probability: np.ndarray, disorder_energy: np.ndarray, q: np.ndarray, U: float
) -> float:
    probability = np.asarray(probability, dtype=float)
    values = np.asarray(disorder_energy, dtype=float) + float(U) * np.asarray(q, dtype=float)
    mean = float(np.sum(probability * values))
    return float(np.sum(probability * (values - mean) ** 2))
