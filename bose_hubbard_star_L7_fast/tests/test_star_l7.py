import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from star_l7_core import (
    Graph,
    batch_star_metrics,
    build_graph,
    central_configurations,
    compensating_channels,
    compensating_observables,
    configuration_priority,
    deterministic_seed,
    disorder_vector,
    generate_basis,
    observables_at_u,
    scalar_star_metrics,
    two_level_entropy,
    verify_detuning,
)
from run_l7_star import locate_peak, regular_grid, smoothing_matrix


class StarL7Tests(unittest.TestCase):
    def test_basis_dimensions_for_requested_sectors(self):
        self.assertEqual([len(generate_basis(7, n, n)) for n in (4, 5, 6, 7)], [210, 462, 924, 1716])

    def test_one_edge_star_matches_two_level_entropy(self):
        for delta in (-3.1, -0.2, 0.0, 1.7):
            coupling = 1.23
            mixing = 4 * coupling**2 / (delta**2 + 4 * coupling**2)
            expected = float(two_level_entropy(np.array([mixing]))[0])
            actual, _, error = batch_star_metrics(np.array([[delta]]), np.array([[coupling]]), 1e-12)
            self.assertLess(abs(actual[0] - expected), 1e-13)
            self.assertLess(error, 1e-14)

    def test_resonant_one_edge_is_log_two(self):
        entropy, participation, _ = batch_star_metrics(np.array([[0.0]]), np.array([[2.0]]), 1e-12)
        self.assertLess(abs(entropy[0] - math.log(2.0)), 1e-13)
        self.assertLess(abs(participation[0] - 2.0), 1e-13)

    def test_zero_coupling_is_zero_entropy(self):
        entropy, participation, _ = batch_star_metrics(np.array([[1.0]]), np.array([[0.0]]), 1e-12)
        self.assertEqual(entropy[0], 0.0)
        self.assertEqual(participation[0], 1.0)

    def test_batched_solver_matches_scalar_reference(self):
        rng = np.random.default_rng(123)
        for z in range(1, 9):
            delta = rng.normal(size=(17, z))
            coupling = rng.uniform(0.2, 2.0, size=(17, z))
            entropy, participation, error = batch_star_metrics(delta, coupling, 1e-12)
            for row in range(len(delta)):
                expected_entropy, expected_participation, expected_error = scalar_star_metrics(delta[row], coupling[row])
                self.assertLess(abs(entropy[row] - expected_entropy), 2e-13)
                self.assertLess(abs(participation[row] - expected_participation), 2e-13)
                self.assertLess(expected_error, 1e-14)
            self.assertLess(error, 1e-14)

    def test_detuning_identity_and_observable_normalization(self):
        graph = build_graph(7, 7, 7)
        epsilon = np.random.default_rng(44).uniform(-2.5, 2.5, size=7)
        self.assertLess(verify_detuning(graph, 0.37, epsilon, seed=91), 1e-12)
        disorder_energy = graph.basis @ epsilon
        result = observables_at_u(graph, 0.37, disorder_energy, np.arange(64))
        self.assertLess(result["normalization_error"], 1e-12)
        self.assertLessEqual(result["M_per_edge"], 1.0)
        self.assertGreaterEqual(result["M_per_edge"], 0.0)
        self.assertGreaterEqual(result["Sstar"], 0.0)

    def test_disorder_is_identical_to_ed_seed_convention(self):
        master, sample = 20260907, 13
        expected = np.random.default_rng(
            np.random.SeedSequence([master, 7, sample])
        ).uniform(-1.0, 1.0, size=7)
        np.testing.assert_array_equal(disorder_vector(master, 7, sample), expected)
        words = np.random.SeedSequence([master, 7, sample]).generate_state(2, dtype=np.uint32)
        expected_seed = int(words[0]) | (int(words[1]) << 32)
        self.assertEqual(deterministic_seed(master, 7, sample), expected_seed)

    def test_closest_count_selects_normalized_spectral_midpoint(self):
        energies = np.array([-10.0, -9.0, -8.0, 0.0, 1.0, 2.0, 10.0])
        priority = np.arange(len(energies), dtype=np.uint64)
        selected = central_configurations(
            energies, {"mode": "closest_count", "count": 2, "center": 0.5}, 0, priority
        )
        self.assertEqual(set(selected), {3, 4})

    def test_preserved_channel_model_matches_direct_formula(self):
        graph = build_graph(4, 4, 4)
        k, coupling, weight = compensating_channels(graph)
        self.assertTrue(np.all(k > 0))
        self.assertAlmostEqual(float(np.sum(weight)), 1.0)
        epsilon = np.array([-0.73, 0.19, 0.61, -0.28])
        U = 0.31
        actual_m, actual_s = compensating_observables(U, epsilon, graph, (k, coupling, weight))
        delta = 2.0 * k[:, None] * U - np.abs(np.diff(epsilon))[None, :]
        mixing = 4.0 * coupling[:, None] ** 2 / (delta ** 2 + 4.0 * coupling[:, None] ** 2)
        expected_m = float(np.sum(weight[:, None] * mixing) / 3)
        expected_s = float(np.sum(weight[:, None] * two_level_entropy(mixing)) / 3)
        self.assertAlmostEqual(actual_m, expected_m, places=14)
        self.assertAlmostEqual(actual_s, expected_s, places=14)

    def test_star_matrix_is_the_full_hamiltonian_local_block(self):
        graph = build_graph(4, 4, 4)
        U = 0.27
        epsilon = np.array([-0.8, 0.11, 0.37, 0.72])
        diagonal = U * graph.interaction + graph.basis @ epsilon
        full = np.diag(diagonal.astype(float))
        for alpha in range(graph.dim):
            z = int(graph.degree[alpha])
            full[alpha, graph.neighbors[alpha, :z]] = graph.couplings[alpha, :z]
        alpha = 23
        z = int(graph.degree[alpha])
        indices = np.concatenate(([alpha], graph.neighbors[alpha, :z]))
        extracted = full[np.ix_(indices, indices)] - np.eye(z + 1) * diagonal[alpha]
        expected = np.zeros((z + 1, z + 1))
        expected[1:, 1:] = np.diag(diagonal[indices[1:]] - diagonal[alpha])
        expected[0, 1:] = graph.couplings[alpha, :z]
        expected[1:, 0] = graph.couplings[alpha, :z]
        np.testing.assert_allclose(extracted, expected, atol=1e-14)

    def test_small_full_ed_and_star_same_disorder_are_finite(self):
        graph = build_graph(4, 4, 4)
        eta = np.random.default_rng(np.random.SeedSequence([20260907, 4, 0])).uniform(-1, 1, 4)
        epsilon = 1.3 * eta
        U = 0.3
        diagonal = U * graph.interaction + graph.basis @ epsilon
        full = np.diag(diagonal.astype(float))
        for alpha in range(graph.dim):
            z = int(graph.degree[alpha])
            full[alpha, graph.neighbors[alpha, :z]] = graph.couplings[alpha, :z]
        eigenvalues, eigenvectors = np.linalg.eigh(full)
        central_ed = np.argsort(np.abs((eigenvalues-eigenvalues[0])/(eigenvalues[-1]-eigenvalues[0])-0.5))[:20]
        probabilities = eigenvectors[:, central_ed] ** 2
        ed_entropy = float(np.mean(-np.sum(np.where(probabilities > 0, probabilities*np.log(probabilities), 0), axis=0)))
        priority = configuration_priority(graph.basis, 91)
        central = central_configurations(diagonal, {"mode": "closest_count", "count": 20, "center": 0.5}, 0, priority)
        star_entropy = float(observables_at_u(graph, U, graph.basis @ epsilon, central)["Sstar"])
        self.assertTrue(np.isfinite(ed_entropy))
        self.assertTrue(np.isfinite(star_entropy))
        self.assertGreaterEqual(star_entropy, 0.0)
        self.assertLessEqual(star_entropy, math.log(graph.neighbors.shape[1] + 1) + 1e-12)

    def test_observables_are_invariant_to_basis_order(self):
        graph = build_graph(4, 4, 4)
        rng = np.random.default_rng(1234)
        permutation = rng.permutation(graph.dim)
        old_to_new = np.empty(graph.dim, dtype=np.int32)
        old_to_new[permutation] = np.arange(graph.dim)
        permuted = Graph(
            graph.basis[permutation], graph.interaction[permutation],
            old_to_new[graph.neighbors[permutation]], graph.couplings[permutation],
            graph.channels[permutation], graph.degree[permutation],
            graph.from_site[permutation], graph.to_site[permutation],
        )
        epsilon = np.array([-0.51, 0.17, 0.83, -0.26])
        U = 0.24
        energy = graph.basis @ epsilon
        penergy = permuted.basis @ epsilon
        selection = {"mode": "closest_count", "count": 20, "center": 0.5}
        central = central_configurations(U*graph.interaction+energy, selection, 0,
                                         configuration_priority(graph.basis, 7))
        pcentral = central_configurations(U*permuted.interaction+penergy, selection, 0,
                                          configuration_priority(permuted.basis, 7))
        first = observables_at_u(graph, U, energy, central)
        second = observables_at_u(permuted, U, penergy, pcentral)
        for name in ("M_sum", "M_per_edge", "S2_sum", "S2_per_edge", "Sstar", "Pstar"):
            self.assertAlmostEqual(float(first[name]), float(second[name]), places=13)

    def test_central_selection_is_invariant_to_basis_order(self):
        basis = generate_basis(7, 7, 7)
        epsilon = np.array([-0.91, 0.22, 0.47, -0.38, 0.73, -0.64, 0.11])
        interaction = np.sum(basis.astype(np.int64)*(basis.astype(np.int64)-1), axis=1)
        energies = 0.31*interaction + basis @ epsilon
        priorities = configuration_priority(basis, 998877)
        selected = central_configurations(energies, 0.5, 128, priorities)
        rng = np.random.default_rng(4)
        order = rng.permutation(len(basis))
        selected_permuted = central_configurations(energies[order], 0.5, 128, priorities[order])
        original_states = {tuple(row) for row in basis[selected]}
        permuted_states = {tuple(row) for row in basis[order][selected_permuted]}
        self.assertEqual(original_states, permuted_states)

    def test_peak_estimator_recovers_known_parabola(self):
        x = np.linspace(0.0, 0.6, 121)
        y = 2.0 - 4.0*(x-0.273)**2
        cfg = {"analysis": {"smoothing_window": 9, "smoothing_degree": 3,
                              "boundary_margin_points": 3, "peak_fit_points": 7}}
        result = locate_peak(x, y, cfg, smoothing_matrix(len(x), 9, 3))
        self.assertEqual(result["quality_flag"], "ok")
        self.assertLess(abs(result["U_peak"]-0.273), 1e-12)

    def test_explicit_parameter_grid(self):
        values = regular_grid({"values": [0.8, 1.2, 1.6, 2.0, 2.5]})
        np.testing.assert_array_equal(values, [0.8, 1.2, 1.6, 2.0, 2.5])


if __name__ == "__main__":
    unittest.main()
