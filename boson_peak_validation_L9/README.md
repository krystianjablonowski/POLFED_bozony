# Boson entropy-peak validation

## Pełny model zredukowany M0/M1 dla L=7 i L=8

Po wykonaniu weryfikacji dwuskładnikowej dla obu rozmiarów uruchom ponownie
`verify_entropy_mechanism.py` w aktualnej wersji. Oprócz dotychczasowych tabel
zapisuje on teraz `bootstrap_peaks.csv.gz`, potrzebny do pełnego bootstrapu
modelu zredukowanego. Nie jest przy tym wykonywana ponowna diagonalizacja.

Następnie uruchom `analyze_reduced_entropy_model.py`, podając katalogi
`entropy_mechanism_verification` dla L=7 i L=8. Program dopasowuje proste tylko
do składowych R/B i wag krzywiznowych, wybiera M0 albo M1 przez LOOCV, sprawdza
zakresy W/t od 0.8, 1.0, 1.2 i 1.4 oraz wykonuje transfer L=7 do L=8. Maksima
entropii z ED są używane wyłącznie do końcowej walidacji.

Na Kruku ten ostatni etap można wysłać skryptem
`pbs_reduced_entropy_model.sh`. Wymagane zmienne PBS to `PROJECT_DIR`,
`VERIFICATION_L7` i `VERIFICATION_L8`; opcjonalne to `OUTPUT_DIR` oraz
`PYTHON_BIN`.

Przykład po zakończeniu obu ponownych weryfikacji:

```bash
cd /home/2/kj405942/POLFED_bosons/boson_peak_validation_L9

PYTHON_BIN=/home/2/kj405942/conda_envs/boson_peak_analysis/bin/python
RUN7=/home/2/kj405942/POLFED_bosons/boson_peak_runs/run_L7_Neqnmax4_7_central80_b50
RUN8=/home/2/kj405942/POLFED_bosons/boson_peak_runs/run_L8_Neqnmax5_8_central80_b50

qsub -o "$PWD/logs/reduced_entropy_L7_L8.out" \
  -v PROJECT_DIR="$PWD",VERIFICATION_L7="$RUN7/entropy_mechanism_verification",VERIFICATION_L8="$RUN8/entropy_mechanism_verification",PYTHON_BIN="$PYTHON_BIN",OUTPUT_DIR="$PWD/reduced_entropy_model_L7_L8" \
  pbs_reduced_entropy_model.sh
```

Najważniejsze wyniki to `reduced_model_metrics.csv`,
`reduced_model_parameters.csv`, `slope_comparison.csv`,
`leave_one_system_out_metrics.csv`, `report.md` oraz katalog `figures/`.

Standalone HPC workflow for testing whether the maximum of the mean Fock-space
Shannon entropy in the disordered Bose-Hubbard chain agrees with the independent
compensating-channel prediction.

The Hamiltonian convention is fixed to

```text
H = -t sum_<i,j> (b_i^dagger b_j + h.c.)
    + U sum_i n_i(n_i-1) + sum_i epsilon_i n_i,
epsilon_i ~ Uniform[-W,W], OBC.
```

There is no factor `1/2` in the interaction term. Only `U >= 0` is used.
The local cutoff means `0 <= n_i <= nmax` (`sps=nmax+1`). The theory uses
only compensating channels with `k=b-a+1>0`; it is computed independently of
all eigenvectors and entropy data.

## Files

- `boson_peak_core.py`: basis, sparse Hamiltonian, channel theory, solvers, observables.
- `boson_peak_cli.py`: convenient named command-line scan overrides.
- `boson_peak_hpc.py`: theory stage, ED dry-run, and one array task.
- `make_tasks.py`: resumable task table plus PBS and SLURM array scripts.
- `monitor_progress.py`: durable aggregate progress and ETA.
- `merge_results.py`: validation and merge of atomic realization files.
- `analyze_peaks.py`: paired bootstrap, plateaus, peak decisions, validation report.
- `plot_results.py`: all PRB-style figure families, including absolute and baseline-subtracted entropy; no diagonalization.
- `config_pilot.yaml`: required `L=N=9` pilot (`nmax=2,3,4,6`, `R=12`).
- `config_production.yaml`: example production and finite-size configuration.
- `tests/`: 18 automatic checks covering the physics core, restart logic, cluster scripts, and CLI ranges.

The `.yaml` files intentionally contain valid JSON. They work even if PyYAML is
not installed.

## Copy to Kruk

From PowerShell on the local computer:

```powershell
scp -r "C:\Users\avoga\OneDrive\Dokumenty\POLFED\boson_peak_validation_L9" `
  kj405942@kruk.fuw.edu.pl:~/POLFED_bosons/
```

On the cluster:

```bash
cd ~/POLFED_bosons/boson_peak_validation_L9
python -c 'import numpy, scipy, pandas, matplotlib; print(numpy.__version__, scipy.__version__)'
python -m pytest tests -q
```

If a package is missing, use the active Conda environment or install into it:

```bash
python -m pip install numpy scipy pandas matplotlib PyYAML pyarrow pytest
```

Do not install Julia or `Polfed.jl` for this workflow. The sparse solver is
SciPy `eigsh` with shift-invert.

## Quick L=6 check using terminal flags

Named flags override the configuration file. This small scan checks several
fillings and local cutoffs before any L=9 calculation:

```bash
cd ~/POLFED_bosons/boson_peak_validation_L9

python boson_peak_hpc.py ed \
  --config config_pilot.yaml \
  --dry-run \
  --L 6 \
  --N-list 3,6 \
  --nmax-list 2,3,4 \
  --W-list 0.5,1,2 \
  --adaptive-U \
  --U-points 11 \
  --realizations 2 \
  --backend scipy \
  --nev 36 \
  --n-states 20
```

After inspecting the dry-run, create a separate run with the same physical
flags. The resolved values are saved in `config_resolved.json`, so the job
array cannot silently revert to the original L=9 configuration:

```bash
python make_tasks.py \
  --config config_pilot.yaml \
  --output ~/POLFED_bosons/boson_peak_runs/run_L6_flag_check \
  --L 6 \
  --N-list 3,6 \
  --nmax-list 2,3,4 \
  --W-list 0.5,1,2 \
  --adaptive-U \
  --U-points 11 \
  --realizations 2 \
  --block-size 2 \
  --backend scipy \
  --nev 36 \
  --n-states 20 \
  --theory-scope scan \
  --array-concurrency 2 \
  --memory 8gb \
  --walltime 04:00:00

python boson_peak_hpc.py ed \
  --config ~/POLFED_bosons/boson_peak_runs/run_L6_flag_check/config_resolved.json \
  --task-table ~/POLFED_bosons/boson_peak_runs/run_L6_flag_check/tasks.csv \
  --dry-run

bash ~/POLFED_bosons/boson_peak_runs/run_L6_flag_check/submit_pbs.sh
```

`--L`, `--N-list`, and `--nmax-list` form their Cartesian product. For an
irregular selection, repeat exact sectors instead, for example
`--sector 6:3:2 --sector 6:6:4`. An explicit grid can be supplied as
`--U-list 0,0.05,0.1,0.2,0.4,0.8`; it is mutually exclusive with
`--adaptive-U`. Inclusive uniform grids can instead be written compactly as
`--U-range 0:2:0.1` and `--W-range 0:5:0.5`. Run
`python make_tasks.py --help` for all solver and cluster flags.

The plotted entropy response is paired within each disorder realization:

```text
delta_S_F(U,W)/ln(D) = S_F(U,W)/ln(D) - S_F(0,W)/ln(D).
```

It is the interaction-induced entropy change at fixed disorder, not the
difference between disordered and clean systems. Plotting defaults to the ED
W range; `plot_results.py --min-W 0 --max-W 5 RUN` can override it explicitly.

## Stage A: channel theory

This stage is cheap, needs no ED and scans the full theory sector set:

```bash
cd ~/POLFED_bosons/boson_peak_validation_L9
mkdir -p ~/POLFED_bosons/boson_peak_runs

python boson_peak_hpc.py theory \
  --config config_pilot.yaml \
  --output-dir ~/POLFED_bosons/boson_peak_runs/theory_stage_A
```

It writes `channels.csv`, `theory_coefficients.csv`, and `theory_peaks.csv` for
the complete standard set of sectors from the specification. `make_tasks.py`
also places these full theory tables in every run directory, so all comparison
figures remain self-contained.

## Stage B: mandatory pilot

First inspect dimensions, matrix sparsity, U grids, number of diagonalizations,
and estimated memory:

```bash
python boson_peak_hpc.py ed --config config_pilot.yaml --dry-run
```

Create the run directory outside the source folder, which also avoids permission
problems after copying code from another machine:

```bash
python make_tasks.py \
  --config config_pilot.yaml \
  --output ~/POLFED_bosons/boson_peak_runs/run_pilot

python boson_peak_hpc.py ed \
  --config ~/POLFED_bosons/boson_peak_runs/run_pilot/config_resolved.json \
  --task-table ~/POLFED_bosons/boson_peak_runs/run_pilot/tasks.csv \
  --dry-run
```

Only after checking this output, submit the generated array:

```bash
bash ~/POLFED_bosons/boson_peak_runs/run_pilot/submit_pbs.sh
```

For SLURM use:

```bash
sbatch ~/POLFED_bosons/boson_peak_runs/run_pilot/job_array.slurm
```

The concurrency cap is stored in `cluster.array_concurrency`. The pilot default
is two simultaneous array jobs. Every task handles one `(L,N,nmax,W)` and a
small block of realizations; every realization keeps all U values together.

Monitor the whole array from a login node:

```bash
python monitor_progress.py ~/POLFED_bosons/boson_peak_runs/run_pilot --watch 30
```

To rerun one PBS array task manually, use its zero-based index from `tasks.csv`:

```bash
python boson_peak_hpc.py ed \
  --config ~/POLFED_bosons/boson_peak_runs/run_pilot/config_resolved.json \
  --task-table ~/POLFED_bosons/boson_peak_runs/run_pilot/tasks.csv \
  --task-index 0 --resume
```

Completed realization files are skipped after their configuration hash and U
grid are checked. Partial files receive only missing U points.

### Fraction of the spectrum

Use `--spectrum-fraction 0.8 --backend dense` to diagonalize the complete
Hamiltonian and average observables over the central 80% of its eigenstates.
The selected-state count is computed separately from each sector's full
Hilbert-space dimension. This is preferable to a common `--n-states` value
when sectors with different `nmax` are compared. Large fractions should use
the dense backend because requesting most eigenpairs from sparse shift-invert
is inefficient.

Add `--compact-output` for large scans. Realization-level entropy, gap ratio,
channel observables, uncertainty inputs, and solver diagnostics are retained,
while the redundant per-eigenpair energy and residual arrays are omitted. Raw
eigenvectors are never stored.

The merge records final failures and recovered shift-invert retries in
`failed_tasks.csv`. Full timing, memory, residual, solver-attempt, and retry-log
information is retained in `timing_and_memory.csv`.

## Merge, bootstrap, plots

Run these from the source folder after the array is complete:

```bash
cd ~/POLFED_bosons/boson_peak_validation_L9
RUN=~/POLFED_bosons/boson_peak_runs/run_pilot

python merge_results.py "$RUN"
python analyze_peaks.py "$RUN" --bootstrap 2000 \
  --peak-method weighted-loess --loess-span 9 --loess-dense-points 1001
python plot_results.py "$RUN"
```

The default entropy-maximum estimator is an inverse-variance-weighted local
quadratic LOESS smoother. Pointwise weights are proportional to
`1 / SEM(U)^2`. Its confidence interval is not obtained by treating U points
as independent: the analysis resamples complete disorder realizations, uses
the same realization indices along the whole U curve, and repeats the LOESS
maximum search. This paired bootstrap preserves correlations between U
points. `peak_summary.csv` stores `U_S_loess_over_t`, the bootstrap median in
`U_S_star_over_t`, its 95% interval, and the older local-quadratic/Savitzky-
Golay estimates for diagnostics. Use `--peak-method consensus` to reproduce
the previous estimator.

Each analysis also writes compact working tables `entropy_peak_points.csv`
and `mixing_function_peak_points.csv`; they are selected, documented subsets
of the complete `peak_summary.csv` table.

## Two-component entropy verification

The ED worker already stores `H_Q`, `S_intra_Q`, `P_Q`, and the numerical
error in the exact identity `S_F = H_Q + S_intra_Q`.  Therefore an existing
merged run can be tested without another diagonalization.  The additional
analysis implements the 3/5/7-point peak fits, paired bootstrap, component
classification, the curvature-weighted no-fit prediction, the mirror test,
and slope-window robustness checks.

First locate an existing merged run:

```bash
find ~/POLFED_bosons -name ed_observables_by_realization.csv -print
```

For a run stored in `~/POLFED_bosons/boson_peak_runs/run_L7`, submit the
analysis on Kruk with:

```bash
cd ~/POLFED_bosons/boson_peak_validation_L9
mkdir -p logs
RUN=~/POLFED_bosons/boson_peak_runs/run_L7

qsub \
  -o "$PWD/logs/entropy_mechanism.out" \
  -v PROJECT_DIR="$PWD",RUN_DIR="$RUN",PYTHON_BIN=/home/2/kj405942/conda_envs/boson_peak_analysis/bin/python,BOOTSTRAP=2000,ONLY_L=7 \
  pbs_verify_entropy_mechanism.sh
```

The same analysis for the existing L=8 run uses `ONLY_L=8` and its own run
directory, for example
`~/POLFED_bosons/boson_peak_runs/run_L8_Neqnmax5_8_central80_b50`.

If the run contains raw realization files but not the merged CSV files, run
`python merge_results.py "$RUN"` first.  The verification output is written to
`$RUN/entropy_mechanism_verification/`; figures are saved there in PNG and PDF
formats.  The main numerical result is `curvature_prediction.csv`, while
`summary.txt`, `component_classification.csv`, `slope_robustness.csv`, and
`eta_comparison.csv` contain the acceptance tests needed to decide whether the
mechanism explains the entropy maximum.

To exclude an intentionally omitted part of the disorder grid already during
completeness validation, pass the same bounds to the merge and plotting steps,
for example:

```bash
python merge_results.py "$RUN" --min-W 0.5 --max-W 5
python analyze_peaks.py "$RUN" --bootstrap 2000
python plot_results.py "$RUN" --min-W 0.5 --max-W 5
```

For shallow low-U maxima, the analysis requires agreement between a local
quadratic estimate and a Savitzky-Golay-smoothed estimate. The defaults are a
seven-point smoothing window and a maximum separation of 1.5 U-grid steps.
Both estimates and their separation are written to `peak_summary.csv`. They
can be tightened without repeating ED:

```bash
python analyze_peaks.py "$RUN" --bootstrap 2000 \
  --peak-smoothing-points 5 \
  --max-estimator-disagreement-steps 1.0
```

A recommended dense small-W validation grid for `L=N=6` is
`W/t=0.1:0.1:2.5` and `U/t=0:0.025:1`. For example, a single-nmax batch can be
prepared with:

```bash
python make_tasks.py \
  --config config_pilot.yaml \
  --output ~/POLFED_bosons/boson_peak_runs/run_L6_dense_smallW_nmax5 \
  --L 6 --N 6 --nmax 5 \
  --W-range 0.1:2.5:0.1 \
  --U-range 0:1:0.025 \
  --realizations 200 --block-size 50 \
  --backend scipy --nev 36 --n-states 20 \
  --theory-scope scan --array-concurrency 10 \
  --memory 8gb --walltime 24:00:00
```

The last command writes PNG and PDF versions of:

```text
entropy_change_curves
entropy_absolute_curves
peak_positions_vs_W
peak_difference_vs_W
peak_estimator_diagnostics
coefficients_vs_nmax_and_N
effective_exponent
peak_visibility
finite_size_comparison
nmax_comparison_L9
entropy_decomposition_Q
```

On the entropy panels, colored circles are points on the fixed-`W` curves,
stars mark the channel-theory prediction `U_M*`, filled diamonds mark resolved
ED entropy maxima `U_S*`, and open diamonds mark unresolved ED candidates.
Open markers indicate unresolved ED maxima. They are not connected by a line.
On entropy curves, horizontal segments indicate the statistically
indistinguishable interval in U. On `peak_positions_vs_W`, the same plateau is
shown as a vertical U interval at fixed W.
`VALIDATION_REPORT.md` separates `consistent`, `inconsistent`, and
`numerically unresolved` outcomes.

The analysis also writes `scan_extension_proposals.csv`. It proposes a denser
or wider positive-U grid for boundary maxima and broad plateaus, but never
submits those additional calculations automatically.

## Stage C: production

Do not submit `config_production.yaml` before inspecting pilot timing,
curvature, confidence-interval width, plateau width, and
`R_required_for_3sigma` in `peak_summary.csv`. A point requiring more than the
configured `analysis.max_required_realizations` is deliberately left
unresolved. The example includes the main `L=N=9` sectors, the
`nmax=2` control, finite-size sectors at `nmax=4`, and two filling checks.

Prepare production exactly as a new run:

```bash
python boson_peak_hpc.py ed --config config_production.yaml --dry-run

python make_tasks.py \
  --config config_production.yaml \
  --output ~/POLFED_bosons/boson_peak_runs/run_production

python boson_peak_hpc.py ed \
  --config ~/POLFED_bosons/boson_peak_runs/run_production/config_resolved.json \
  --task-table ~/POLFED_bosons/boson_peak_runs/run_production/tasks.csv \
  --dry-run

bash ~/POLFED_bosons/boson_peak_runs/run_production/submit_pbs.sh
```

For a smaller production subset, override parameters without editing the file:

```bash
python make_tasks.py \
  --config config_production.yaml \
  --output ~/POLFED_bosons/boson_peak_runs/run_L9_nmax4 \
  --set 'ed.sectors=[{"L":9,"N":9,"nmax":[4]}]' \
  --set 'ed.W_values=[0.5,1,2,4]' \
  --set ed.realizations=40
```

All important numerical parameters can be overridden as dotted `KEY=JSON`
values. Examples include `eigensolver.nev=48`, `eigensolver.n_states=24`,
`ed.U_grid.points=13`, and `cluster.array_concurrency=1`.

## Output safety

Raw files are written atomically as

```text
RUN/raw/L9_N9_nmax4_W2/realization_000027.npz
```

Each file contains the shared `eta`, all requested U values, energies,
observables, residuals, timing, memory, and metadata. Full eigenvectors are not
stored. Never use `make_tasks.py --force` to remove data; it only regenerates
metadata and leaves `raw/` untouched, but changing hashes intentionally makes
incompatible old realizations fail validation.

## GitHub

Raw NPZ files can be very large. Commit source, reports, tables, and figures;
keep the run's `raw/` and `logs/` outside Git:

```bash
cd ~/POLFED_bosons
git add boson_peak_validation_L9
git add boson_peak_runs/run_pilot/figures \
        boson_peak_runs/run_pilot/VALIDATION_REPORT.md \
        boson_peak_runs/run_pilot/peak_summary.csv \
        boson_peak_runs/run_pilot/ed_averaged_curves.csv
git commit -m "Add bosonic small-U entropy peak validation"
git push
```
