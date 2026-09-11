import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from star_l7_core import (
    batch_star_metrics,
    build_graph,
    central_configurations,
    configuration_priority,
    generate_basis,
    observables_at_u,
    scalar_star_metrics,
    two_level_entropy,
    verify_detuning,
)
from run_l7_star import locate_peak, smoothing_matrix


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


if __name__ == "__main__":
    unittest.main()
