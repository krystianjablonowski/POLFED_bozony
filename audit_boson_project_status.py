#!/usr/bin/env python3
"""Audit computed bosonic POLFED / entropy-study outputs on the cluster.

Run from ``~/POLFED_bosons``.  The script only reads files and prints a compact
status table; it does not submit or delete jobs.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


OBS_RE = re.compile(
    r"(?:^|/)data/L(?P<L>\d+)/N_(?P<N>\d+)_nmax_(?P<nmax>\d+)/"
    r"U_(?P<U>[^/]+)/W_(?P<W>[^/]+)/observables_.*\.csv$"
)

ENERGY_RE = re.compile(
    r"(?:^|/)data/U_(?P<U>[^/]+)/W_(?P<W>[^/]+)/L_(?P<L>\d+)/"
    r"N_(?P<N>\d+)_nmax_(?P<nmax>\d+)(?:/boundary_(?P<boundary>[^/]+))?/"
    r"energies_(?P<mode>full|polfed)_.*\.txt$"
)


@dataclass(frozen=True)
class Point:
    L: int
    N: int
    nmax: int
    U: str
    W: str


def numeric_key(text: str) -> tuple[int, float | str]:
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def parse_list(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def find_observable_points(root: Path) -> tuple[dict[tuple[int, int, int], set[tuple[str, str]]], int]:
    sectors: dict[tuple[int, int, int], set[tuple[str, str]]] = defaultdict(set)
    total_files = 0
    for path in root.rglob("observables_*.csv"):
        total_files += 1
        match = OBS_RE.search(path.as_posix())
        if not match:
            continue
        L = int(match.group("L"))
        N = int(match.group("N"))
        nmax = int(match.group("nmax"))
        sectors[(L, N, nmax)].add((match.group("U"), match.group("W")))
    return sectors, total_files


def summarize_observables(root: Path, expected_u: set[str], expected_w: set[str]) -> None:
    sectors, total_files = find_observable_points(root)
    print("\n=== ENTROPY / MECHANISM observables_*.csv ===")
    print(f"observable files found: {total_files}")
    if not sectors:
        print("No path-matched observables found.")
        return

    expected_count = len(expected_u) * len(expected_w) if expected_u and expected_w else None
    header = "L   N  nmax  points"
    if expected_count:
        header += f"/{expected_count}  missing"
    print(header)
    print("-" * len(header))
    for (L, N, nmax), points in sorted(sectors.items()):
        line = f"{L:<3} {N:<3} {nmax:<5} {len(points):<6}"
        if expected_count:
            missing = sorted(
                (u, w)
                for u in expected_u
                for w in expected_w
                if (u, w) not in points
            )
            line += f"/{expected_count:<5} {len(missing):<7}"
            if 0 < len(missing) <= 12:
                line += " " + ", ".join(f"U{u}:W{w}" for u, w in missing)
        print(line)


def summarize_energy_files(root: Path) -> None:
    sectors: dict[tuple[int, int, int, str], set[tuple[str, str]]] = defaultdict(set)
    total_files = 0
    for path in root.rglob("energies_*.txt"):
        total_files += 1
        match = ENERGY_RE.search(path.as_posix())
        if not match:
            continue
        key = (
            int(match.group("L")),
            int(match.group("N")),
            int(match.group("nmax")),
            match.group("mode"),
        )
        sectors[key].add((match.group("U"), match.group("W")))

    print("\n=== GAP RATIO energy files ===")
    print(f"energy files found: {total_files}")
    if not sectors:
        print("No path-matched energy files found.")
        return
    print("L   N  nmax  mode    U,W points")
    print("-------------------------------")
    for (L, N, nmax, mode), points in sorted(sectors.items()):
        print(f"{L:<3} {N:<3} {nmax:<5} {mode:<7} {len(points)}")


def read_summary_shape(path: Path) -> tuple[int, list[str]]:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        return count, header
    except UnicodeDecodeError:
        with path.open(newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        return count, header


def summarize_tables(root: Path) -> None:
    patterns = [
        ("gap summaries", "gap_ratio_summary.csv"),
        ("midpoints", "gap_ratio_midpoints.csv"),
        ("SFF Thouless", "thouless_summary.csv"),
        ("entropy aggregate", "aggregate_observables.csv"),
        ("entropy Uhalf", "Uhalf_by_W.csv"),
        ("entropy Ux", "N_crossings.csv"),
        ("Wstar bootstrap", "Wstar_bootstrap.csv"),
    ]
    print("\n=== Analysis tables ===")
    for label, name in patterns:
        paths = sorted(root.rglob(name))
        print(f"{label}: {len(paths)}")
        for path in paths[:12]:
            rows, _ = read_summary_shape(path)
            print(f"  {rows:6d} rows  {rel(path, root)}")
        if len(paths) > 12:
            print(f"  ... {len(paths) - 12} more")


def summarize_plots(root: Path) -> None:
    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[str]] = defaultdict(list)
    for ext in (".png", ".pdf"):
        for path in root.rglob(f"*{ext}"):
            key = path.parent.as_posix()
            counts[key] += 1
            if len(examples[key]) < 3:
                examples[key].append(path.name)
    print("\n=== Plot folders (.png/.pdf) ===")
    if not counts:
        print("No plots found.")
        return
    for folder, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:25]:
        print(f"{count:4d}  {rel(Path(folder), root)}")
        print("      " + ", ".join(examples[folder]))


def summarize_qstat(user: str | None) -> None:
    if not user:
        return
    print("\n=== qstat ===")
    try:
        completed = subprocess.run(
            ["qstat", "-u", user],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("qstat not available on this machine.")
        return
    output = completed.stdout.strip()
    if not output:
        print("No active jobs printed by qstat.")
        return
    lines = output.splitlines()
    print("\n".join(lines[:40]))
    if len(lines) > 40:
        print(f"... {len(lines) - 40} more qstat lines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--expected-U", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--expected-W", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--qstat-user", default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    expected_u = parse_list(args.expected_U)
    expected_w = parse_list(args.expected_W)

    print(f"Audit root: {root}")
    summarize_qstat(args.qstat_user)
    summarize_observables(root, expected_u, expected_w)
    summarize_energy_files(root)
    summarize_tables(root)
    summarize_plots(root)


if __name__ == "__main__":
    main()
