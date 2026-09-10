#!/usr/bin/env python3
"""Shared command-line scan overrides for task generation and dry-runs."""

from __future__ import annotations

import argparse
import copy
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List

import boson_peak_core as core


def _csv_values(text: str, cast, name: str) -> List[Any]:
    try:
        values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("{} must be a comma-separated list".format(name)) from exc
    if not values:
        raise ValueError("{} cannot be empty".format(name))
    return values


def _sector(text: str) -> Dict[str, int]:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("--sector must have the form L:N:nmax, for example 6:4:3")
    try:
        L, N, nmax = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--sector values must be integers: " + text) from exc
    return {"L": L, "N": N, "nmax": nmax}


def _range_values(text: str, name: str) -> List[float]:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("{} must have the form START:STOP:STEP".format(name))
    try:
        start, stop, step = (Decimal(part.strip()) for part in parts)
    except InvalidOperation as exc:
        raise ValueError("{} contains a non-numeric value".format(name)) from exc
    if start < 0 or stop < start or step <= 0:
        raise ValueError("{} requires 0 <= START <= STOP and STEP > 0".format(name))
    intervals = (stop - start) / step
    if intervals != intervals.to_integral_value():
        raise ValueError("{} STEP must land exactly on STOP".format(name))
    return [float(start + index * step) for index in range(int(intervals) + 1)]


def add_scan_arguments(parser: argparse.ArgumentParser, include_cluster: bool = True) -> None:
    sector = parser.add_argument_group("direct scan parameters")
    sector.add_argument("--sector", action="append", default=[], metavar="L:N:NMAX",
                        help="exact sector; repeat this flag for several sectors")
    sector.add_argument("--L", "--L-list", dest="L_list", metavar="LIST",
                        help="one L or a comma-separated list")
    sector.add_argument("--N", "--N-list", dest="N_list", metavar="LIST",
                        help="one N or a comma-separated list")
    sector.add_argument("--nmax", "--nmax-list", dest="nmax_list", metavar="LIST",
                        help="one nmax or a comma-separated list")
    sector.add_argument("--W-list", "--w-list", dest="W_list", metavar="LIST",
                        help="comma-separated W/t values")
    sector.add_argument("--W-range", "--w-range", dest="W_range", metavar="START:STOP:STEP",
                        help="inclusive uniform W/t range")
    sector.add_argument("--U-list", "--u-list", dest="U_list", metavar="LIST",
                        help="explicit comma-separated nonnegative U/t values")
    sector.add_argument("--U-range", "--u-range", dest="U_range", metavar="START:STOP:STEP",
                        help="inclusive uniform nonnegative U/t range")
    sector.add_argument("--adaptive-U", "--adaptive-u", action="store_true", dest="adaptive_U",
                        help="use the theory-centered adaptive U grid")
    sector.add_argument("--U-points", "--u-points", type=int, dest="U_points",
                        help="number of points in an adaptive U grid")
    sector.add_argument("--U-relative-half-width", "--u-relative-half-width", type=float,
                        dest="U_relative_half_width")
    sector.add_argument("--U-absolute-half-width", "--u-absolute-half-width", type=float,
                        dest="U_absolute_half_width")
    sector.add_argument("--realizations", type=int, help="number of disorder realizations")
    sector.add_argument("--block-size", type=int, help="realizations per array task")
    sector.add_argument("--seed", type=int, help="master disorder seed")
    sector.add_argument("--nev", type=int, help="eigenvalues requested from the solver")
    sector.add_argument("--n-states", type=int, help="central eigenstates used for observables")
    sector.add_argument(
        "--spectrum-fraction", type=float,
        help="central fraction of the full spectrum used for observables, for example 0.8",
    )
    sector.add_argument("--backend", choices=("auto", "scipy", "dense"), help="eigensolver backend")
    sector.add_argument("--theory-scope", choices=("standard", "scan"),
                        help="full standard theory table or only sectors selected for ED")
    if include_cluster:
        cluster = parser.add_argument_group("cluster resources")
        cluster.add_argument("--array-concurrency", type=int, help="maximum simultaneous array tasks")
        cluster.add_argument("--memory", help="memory request, for example 16gb")
        cluster.add_argument("--walltime", help="walltime request, for example 08:00:00")
        cluster.add_argument("--cpus-per-task", type=int)


def apply_scan_arguments(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Return a validated config with named CLI flags applied after --set."""
    result = copy.deepcopy(config)
    exact = list(getattr(args, "sector", []) or [])
    cartesian = (getattr(args, "L_list", None), getattr(args, "N_list", None), getattr(args, "nmax_list", None))
    if exact and any(value is not None for value in cartesian):
        raise ValueError("Use either repeated --sector flags or --L/--N/--nmax lists, not both")
    if exact:
        result["ed"]["sectors"] = [_sector(text) for text in exact]
    elif any(value is not None for value in cartesian):
        if not all(value is not None for value in cartesian):
            raise ValueError("--L, --N and --nmax must be supplied together")
        L_values = _csv_values(cartesian[0], int, "--L")
        N_values = _csv_values(cartesian[1], int, "--N")
        nmax_values = _csv_values(cartesian[2], int, "--nmax")
        result["ed"]["sectors"] = [
            {"L": L, "N": N, "nmax": list(nmax_values)} for L in L_values for N in N_values
        ]

    W_list = getattr(args, "W_list", None)
    W_range = getattr(args, "W_range", None)
    if W_list is not None and W_range is not None:
        raise ValueError("--W-list and --W-range are mutually exclusive")
    if W_list is not None:
        result["ed"]["W_values"] = sorted(set(_csv_values(W_list, float, "--W-list")))
    elif W_range is not None:
        result["ed"]["W_values"] = _range_values(W_range, "--W-range")
    U_list = getattr(args, "U_list", None)
    U_range = getattr(args, "U_range", None)
    adaptive_U = bool(getattr(args, "adaptive_U", False))
    if sum(value is not None for value in (U_list, U_range)) + int(adaptive_U) > 1:
        raise ValueError("--U-list, --U-range, and --adaptive-U are mutually exclusive")
    if U_list is not None:
        result["ed"]["U_grid"]["mode"] = "explicit"
        result["ed"]["U_grid"]["values"] = sorted(set(_csv_values(U_list, float, "--U-list")))
    elif U_range is not None:
        result["ed"]["U_grid"]["mode"] = "explicit"
        result["ed"]["U_grid"]["values"] = _range_values(U_range, "--U-range")
    elif adaptive_U:
        result["ed"]["U_grid"]["mode"] = "adaptive_theory"
        result["ed"]["U_grid"].pop("values", None)

    assignments = (
        ("realizations", "ed", "realizations"),
        ("block_size", "ed", "realization_block_size"),
        ("seed", "ed", "master_seed"),
        ("nev", "eigensolver", "nev"),
        ("n_states", "eigensolver", "n_states"),
        ("spectrum_fraction", "eigensolver", "spectrum_fraction"),
        ("backend", "eigensolver", "backend"),
        ("U_points", "ed.U_grid", "points"),
        ("U_relative_half_width", "ed.U_grid", "relative_half_width"),
        ("U_absolute_half_width", "ed.U_grid", "absolute_half_width_over_t"),
        ("array_concurrency", "cluster", "array_concurrency"),
        ("memory", "cluster", "memory"),
        ("walltime", "cluster", "walltime"),
        ("cpus_per_task", "cluster", "cpus_per_task"),
    )
    for attribute, section, key in assignments:
        value = getattr(args, attribute, None)
        if value is None:
            continue
        target = result
        for part in section.split("."):
            target = target[part]
        target[key] = value

    theory_scope = getattr(args, "theory_scope", None)
    if theory_scope == "scan":
        result["theory"]["standard_sector_set"] = False
        result["theory"]["sectors"] = copy.deepcopy(result["ed"]["sectors"])
    elif theory_scope == "standard":
        result["theory"]["standard_sector_set"] = True

    core.validate_config(result)
    return result
