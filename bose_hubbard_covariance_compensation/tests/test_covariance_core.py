import unittest

import numpy as np

from covariance_core import (
    build_structure,
    covariance_observables,
    disorder_eta,
    generate_basis,
    selected_u0_states,
    variance_of_diagonal_energy,
)
from covariance_compensation import load_ed_points, load_model_points


class CovarianceCoreTests(unittest.TestCase):
    def test_basis_dimension_and_interaction_definition(self):
        structure = build_structure(4, 4, 4)
        self.assertEqual(structure.dim, 35)
        state_index = {tuple(row): index for index, row in enumerate(structure.basis)}
        self.assertEqual(structure.interaction_q[state_index[(2, 1, 1, 0)]], 2.0)
        self.assertEqual(structure.interaction_q[state_index[(4, 0, 0, 0)]], 12.0)

    def test_disorder_seed_is_deterministic_and_in_minus_W_plus_W(self):
        first = disorder_eta(20260907, 7, 12)
        second = disorder_eta(20260907, 7, 12)
        other = disorder_eta(20260907, 7, 13)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(np.all(first >= -1.0) and np.all(first <= 1.0))

    def test_anderson_states_are_exact_H0_eigenstates(self):
        structure = build_structure(4, 4, 4)
        epsilon = 1.3 * disorder_eta(20260907, 4, 3)
        selected = selected_u0_states(structure, 4, 4, 1.0, epsilon, 20, 0.5)
        self.assertEqual(selected.construction, "anderson_orbitals")
        self.assertLess(selected.residual_max, 1.0e-11)
        self.assertLess(selected.orthogonality_error, 1.0e-11)

    def test_orbital_spectrum_matches_dense_many_body_H0(self):
        structure = build_structure(3, 2, 2)
        epsilon = np.asarray([-0.7, 0.2, 0.9])
        selected = selected_u0_states(structure, 2, 2, 1.0, epsilon, structure.dim, 0.5)
        h0 = structure.hopping.toarray() + np.diag(structure.basis @ epsilon)
        exact = np.linalg.eigvalsh(h0)
        np.testing.assert_allclose(np.sort(selected.energies), exact, atol=1.0e-11, rtol=1.0e-11)

    def test_covariance_ratio_minimizes_diagonal_variance(self):
        structure = build_structure(3, 2, 2)
        epsilon = np.asarray([-0.8, 0.1, 0.6])
        selected = selected_u0_states(structure, 2, 2, 1.0, epsilon, 4, 0.5)
        obs = covariance_observables(structure, epsilon, selected, 1.0e-14)
        index = int(np.flatnonzero(obs["valid_state"])[0])
        probability = np.abs(selected.vectors[:, index]) ** 2
        u_star = float(obs["U_comp_state"][index])
        disorder_energy = structure.basis @ epsilon
        center = variance_of_diagonal_energy(probability, disorder_energy, structure.interaction_q, u_star)
        left = variance_of_diagonal_energy(probability, disorder_energy, structure.interaction_q, u_star - 1.0e-4)
        right = variance_of_diagonal_energy(probability, disorder_energy, structure.interaction_q, u_star + 1.0e-4)
        self.assertLessEqual(center, left + 1.0e-14)
        self.assertLessEqual(center, right + 1.0e-14)

    def test_edge_formula_uses_every_unique_hopping_edge(self):
        structure = build_structure(3, 2, 2)
        self.assertEqual(len(structure.edge_alpha), structure.hopping.nnz // 2)
        self.assertTrue(np.all(structure.edge_t < 0.0))

    def test_comparison_csv_readers_use_reported_confidence_intervals(self):
        sector = {"L": 3, "N": 2, "nmax": 2}
        ed = load_ed_points("tests/data/entropy_peaks_smoke.csv", sector)
        model = load_model_points("tests/data/mixing_peaks_smoke.csv", sector)
        self.assertEqual(len(ed), 2)
        self.assertEqual(len(model), 2)
        self.assertAlmostEqual(ed[0]["low"], 0.09)
        self.assertAlmostEqual(ed[0]["high"], 0.11)
        self.assertAlmostEqual(model[0]["low"], 0.08)
        self.assertAlmostEqual(model[0]["high"], 0.12)


if __name__ == "__main__":
    unittest.main()
