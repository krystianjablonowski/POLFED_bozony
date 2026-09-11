import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from star_model_core import (  # noqa: E402
    binary_entropy_from_m,
    build_fock_graph,
    compute_observables,
    diagonal_energies,
    generate_basis,
    star_state_metrics,
    verify_detuning,
)


class StarModelTests(unittest.TestCase):
    def test_one_edge_matches_two_level_entropy(self):
        for delta in (-3.0, -0.2, 0.0, 1.7):
            coupling = 0.8
            m = 4 * coupling**2 / (delta**2 + 4 * coupling**2)
            expected = float(binary_entropy_from_m(np.array([m]))[0])
            actual, _, error = star_state_metrics(np.array([delta]), np.array([coupling]))
            self.assertAlmostEqual(actual, expected, places=13)
            self.assertLessEqual(error, 1e-14)

    def test_zero_coupling(self):
        entropy, participation, _ = star_state_metrics(np.array([1.0]), np.array([0.0]))
        self.assertEqual(entropy, 0.0)
        self.assertEqual(participation, 1.0)

    def test_resonant_single_edge(self):
        entropy, participation, _ = star_state_metrics(np.array([0.0]), np.array([1.0]))
        self.assertAlmostEqual(entropy, np.log(2.0), places=13)
        self.assertAlmostEqual(participation, 2.0, places=13)

    def test_detuning_identity(self):
        graph = build_fock_graph(4, 4, 4)
        epsilon = np.array([-0.7, 0.1, 0.4, -0.2])
        self.assertLess(verify_detuning(graph, 0.37, epsilon, sample_size=1000), 1e-12)

    def test_basis_order_invariance(self):
        epsilon = np.array([-0.7, 0.1, 0.4, -0.2])
        results = []
        for reverse in (False, True):
            graph = build_fock_graph(4, 4, 4, reverse_basis=reverse)
            energies = diagonal_energies(graph, 0.37, epsilon)
            order = np.argsort(energies)
            central = order[(len(order)-20)//2:(len(order)-20)//2+20]
            results.append(compute_observables(graph, 0.37, epsilon, central))
        for key in ("M_sum", "M_per_edge", "S2_sum", "S2_per_edge", "Sstar_mean"):
            self.assertAlmostEqual(results[0][key], results[1][key], places=12)


if __name__ == "__main__":
    unittest.main()

