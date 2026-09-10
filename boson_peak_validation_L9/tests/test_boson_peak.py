import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

import analyze_peaks as analysis
import boson_peak_cli as cli
import boson_peak_core as core
import boson_peak_hpc as engine
import monitor_progress


def solver_settings(nev=12, n_states=8):
    return {
        "backend": "scipy", "nev": nev, "n_states": n_states,
        "tol": 1.0e-11, "residual_tol": 1.0e-8,
        "orthogonality_tol": 1.0e-8, "maxiter": 20000,
        "ncv": 30, "retries": 2, "ncv_retry_growth": 12,
        "sigma_shift_fraction": 1.0e-8, "gap_edge_discard": 1,
    }


def test_01_reference_basis_dimensions():
    expected = {2: 3139, 3: 13051, 4: 19855, 5: 22825, 6: 23905, 9: 24310}
    assert {nmax: core.bounded_composition_dim(9, 9, nmax) for nmax in expected} == expected


def test_02_hermiticity_and_particle_number():
    structure = core.build_structure(5, 4, 3)
    assert np.all(np.sum(structure.occupations, axis=1) == 4)
    assert np.all((structure.occupations >= 0) & (structure.occupations <= 3))
    difference = structure.hopping - structure.hopping.T
    assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1.0e-13
    H = core.hamiltonian(structure, 0.37, np.linspace(-0.2, 0.2, 5))
    assert np.max(np.abs((H - H.T).data), initial=0.0) < 1.0e-13


def test_03_bosonic_hopping_amplitude():
    structure = core.build_structure(3, 3, 3, t=1.7)
    source_state = (2, 1, 0)
    target_state = (1, 2, 0)
    index = {tuple(row): i for i, row in enumerate(structure.occupations)}
    value = structure.hopping[index[target_state], index[source_state]]
    assert np.isclose(value, -1.7 * math.sqrt(2.0 * 2.0))


def test_04_interaction_detuning_is_two_k_U():
    a, b, U = 2, 3, 0.41
    k = b - a + 1
    q_before = a * (a - 1) + b * (b - 1)
    q_after = (a - 1) * (a - 2) + (b + 1) * b
    assert np.isclose(U * (q_after - q_before), 2.0 * k * U)


def test_05_channel_weights_are_normalized_and_compensating():
    channels = core.channel_table(9, 9, 5)
    assert np.isclose(sum(channel.q_ab for channel in channels), 1.0)
    assert all(channel.k > 0 and channel.alpha == 2 * channel.k for channel in channels)


def test_06_closed_mixing_integral_matches_quadrature():
    for u, gamma in ((0.0, 0.2), (0.7, 1.3), (2.4, 0.55)):
        direct = quad(lambda y: 0.5 * (2.0 - y) * gamma**2 / ((y - u) ** 2 + gamma**2), 0.0, 2.0)[0]
        assert np.isclose(float(core.mixing_integral(u, gamma)), direct, rtol=2.0e-11, atol=2.0e-12)


def test_07_mixing_integral_derivative_matches_finite_difference():
    u, gamma, step = 0.83, 0.47, 1.0e-6
    finite = (core.mixing_integral(u + step, gamma) - core.mixing_integral(u - step, gamma)) / (2.0 * step)
    exact = core.mixing_integral_derivative(u, gamma)
    assert np.isclose(float(exact), float(finite), rtol=2.0e-7, atol=2.0e-9)


def test_08_theory_coefficients_against_independent_enumeration():
    L, N, nmax = 4, 4, 3
    raw = []
    for a in range(1, nmax + 1):
        for b in range(nmax):
            k = b - a + 1
            if k <= 0:
                continue
            count = sum(sum(rest) == N - a - b for rest in itertools.product(range(nmax + 1), repeat=L - 2))
            if count:
                raw.append((2.0 * k, 2.0 * math.sqrt(a * (b + 1.0)), count))
    total = sum(item[2] for item in raw)
    q = np.asarray([item[2] / total for item in raw])
    alpha = np.asarray([item[0] for item in raw])
    gamma = np.asarray([item[1] for item in raw])
    D2 = np.sum(q * alpha**2 / gamma**2)
    A = (2.0 / 3.0) * np.sum(q * alpha / gamma**2) / D2
    mu3 = 4.0 / 5.0 - 2.0 * alpha * A + 2.0 * (alpha * A) ** 2 - (alpha * A) ** 3
    B = -2.0 * np.sum(q * alpha * mu3 / gamma**4) / D2
    C = math.sqrt((2.0 / math.pi) * np.sum(q * gamma**2 / alpha) / np.sum(q * alpha * gamma))
    result = core.theory_coefficients(core.channel_table(L, N, nmax))
    assert np.allclose([result["A"], result["B"], result["C"]], [A, B, C], rtol=1.0e-13)


def test_09_explicit_disorder_average_matches_integral():
    channels = core.channel_table(9, 9, 3)
    rng = np.random.default_rng(184)
    U, W = np.asarray([0.15, 0.55, 1.1]), 1.7
    accumulated = np.zeros_like(U)
    samples = 6000
    for _ in range(samples):
        epsilon = rng.uniform(-W, W, size=9)
        accumulated += core.sample_mixing(U, epsilon, channels)
    direct = core.theory_mixing(U, W, channels)
    assert np.allclose(accumulated / samples, direct, atol=1.2e-2, rtol=2.5e-2)


def test_10_sparse_solver_matches_dense_for_L6():
    structure = core.build_structure(6, 6, 2)
    epsilon = np.random.default_rng(51).uniform(-0.7, 0.7, size=6)
    result = core.validate_dense_sparse(structure, 0.43, epsilon, solver_settings())
    assert result["energy_error"] < 1.0e-8
    assert result["entropy_error"] < 1.0e-7
    assert result["gap_ratio_error"] < 1.0e-7


def test_11_entropy_decomposition_identity():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(21, 4))
    vectors, _ = np.linalg.qr(matrix)
    q = np.asarray([index % 5 for index in range(21)])
    result = core.state_observables(vectors[:, :4], q, 21)
    assert result["identity_error"] < 2.0e-14


def synthetic_bootstrap_frames():
    rows, mixing = [], []
    U_values = (0.0, 0.5, 1.0, 1.5, 2.0)
    rng = np.random.default_rng(7)
    for realization in range(18):
        shared = rng.normal(scale=0.01)
        for W in (1.0, 2.0):
            for U in U_values:
                entropy = 0.4 - 0.08 * (U - 1.0) ** 2 + shared + 0.002 * W
                rows.append({"L": 4, "N": 4, "nmax": 3, "W_over_t": W, "realization": realization, "U_over_t": U, "entropy_norm": entropy})
                mixing.append({"L": 4, "N": 4, "nmax": 3, "W_over_t": W, "realization": realization, "U_over_t": U, "sample_mixing": entropy})
    import pandas as pd
    return pd.DataFrame(rows), pd.DataFrame(mixing)


def test_12_paired_bootstrap_preserves_a_common_peak():
    frame, mixing = synthetic_bootstrap_frames()
    result = analysis.bootstrap_sector(frame, mixing, 40, np.random.default_rng(10), 5)
    for W in result:
        finite = result[W]["entropy_peak"][np.isfinite(result[W]["entropy_peak"])]
        assert finite.size >= 35
        assert abs(float(np.median(finite)) - 1.0) < 0.08


def test_13_boundary_plateau_and_positive_curvature_detection():
    boundary = analysis.curve_peak(np.arange(4.0), np.asarray([4.0, 3.0, 2.0, 1.0]))
    convex = analysis.curve_peak(np.arange(5.0), (np.arange(5.0) - 2.0) ** 2)
    assert boundary["boundary"] and not boundary["accepted"]
    assert not convex["accepted"]
    U = np.asarray([0.0, 1.0, 2.0])
    matrix = np.asarray([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [1.0, 1.0, 0.5]])
    low, high, count, includes_zero = analysis.plateau_interval(U, matrix, 0)
    assert (low, high, count, includes_zero) == (0.0, 1.0, 2, True)


def test_14_resume_recognizes_complete_realization(tmp_path: Path):
    path = tmp_path / "realization_000000.npz"
    metadata = {"core_hash": "abc", "status": "complete"}
    core.atomic_save_npz(path, metadata_json=np.asarray(core.canonical_json(metadata)), U_over_t=np.asarray([0.0, 0.5, 1.0]))
    assert engine.realization_is_complete(path, "abc", [0.0, 1.0])
    assert not engine.realization_is_complete(path, "abc", [0.0, 1.5])
    assert not engine.realization_is_complete(path, "wrong", [0.0, 1.0])


def test_15_monitor_aggregates_array_blocks(tmp_path: Path):
    rows = [
        {"task_index": 0, "L": 9, "N": 9, "nmax": 3, "W_over_t": 1.0, "realization_start": 0, "realization_stop": 3,
         "U_values_over_t_json": json.dumps([0.0, 0.5]), "core_hash": "x"},
        {"task_index": 1, "L": 9, "N": 9, "nmax": 3, "W_over_t": 1.0, "realization_start": 3, "realization_stop": 6,
         "U_values_over_t_json": json.dumps([0.0, 0.5]), "core_hash": "x"},
    ]
    core.atomic_write_csv(tmp_path / "tasks.csv", rows)
    core.atomic_write_json(tmp_path / "progress" / "task_000000.json", {"state": "complete", "completed": 3, "failed": 0, "elapsed_s": 30.0})
    core.atomic_write_json(tmp_path / "progress" / "task_000001.json", {"state": "running", "completed": 1, "failed": 0, "elapsed_s": 12.0})
    report = monitor_progress.snapshot(tmp_path)
    assert report["completed"] == 4 and report["total"] == 6
    assert report["tasks"]["complete"] == 1 and report["tasks"]["running"] == 1
    assert report["groups"][(9, 9, 3, 1.0)]["complete"] == 4


def test_16_named_cli_flags_create_cartesian_L6_scan():
    config = core.load_config(Path(__file__).resolve().parents[1] / "config_pilot.yaml")
    args = argparse.Namespace(
        sector=[], L_list="6", N_list="3,4", nmax_list="2,3",
        W_list="0.5,1", U_list="0,0.1,0.2", adaptive_U=False,
        realizations=2, block_size=1, seed=81, nev=20, n_states=10,
        backend="dense", theory_scope="scan", array_concurrency=1,
        memory="4gb", walltime="01:00:00", cpus_per_task=1,
        U_points=None, U_relative_half_width=None, U_absolute_half_width=None,
    )
    result = cli.apply_scan_arguments(config, args)
    assert core.ed_sectors(result) == [(6, 3, 2), (6, 3, 3), (6, 4, 2), (6, 4, 3)]
    assert result["ed"]["U_grid"] == {
        "mode": "explicit", "values": [0.0, 0.1, 0.2], "points": 9,
        "absolute_half_width_over_t": 0.25, "relative_half_width": 0.5,
        "round_decimals": 8,
    }
    assert result["ed"]["realizations"] == 2
    assert result["eigensolver"]["backend"] == "dense"
    assert result["theory"]["standard_sector_set"] is False
