# Boson entropy prediction tests

This folder contains a standalone analysis for the bosonic entropy tests from
`instrukcja_testy_bozony_entropy.tex`.

The script uses existing full-diagonalization observable files:

```text
observables_*.csv
```

It does not run new diagonalizations.  It expects columns such as:

```text
L, N, nmax, U, W, dim,
fock_entropy, fock_entropy_normalized,
pr, pr_normalized,
onsite_pair_density
```

Missing dimensions are reconstructed exactly by dynamic programming.

## Copy to Kruk

From Windows PowerShell:

```powershell
scp -r "C:\Users\avoga\OneDrive\Dokumenty\POLFED\boson_entropy_prediction_tests" `
  kj405942@kruk-host.fuw.edu.pl:~/POLFED_bosons/
```

## Example: L=6, several nmax folders

On Kruk:

```bash
cd ~/POLFED_bosons

python boson_entropy_prediction_tests/boson_entropy_prediction_tests.py \
  --input-dir Wstar_filling_L6_bosons_study/cluster_many_fillings_nmax2 \
  --input-dir Wstar_filling_L6_bosons_study/cluster_many_fillings_nmax3 \
  --input-dir Wstar_filling_L6_bosons_study/cluster_many_fillings_nmax4 \
  --input-dir Wstar_filling_L6_bosons_study/cluster_many_fillings_nmax5 \
  --output-dir boson_entropy_prediction_tests/results_L6 \
  --observable fock_entropy_normalized \
  --L-list 6 \
  --plot-W-list 1,3,5,7,10,15
```

For the Renyi-2 version based on participation ratio:

```bash
python boson_entropy_prediction_tests/boson_entropy_prediction_tests.py \
  --input-dir Wstar_filling_L6_bosons_study/cluster_many_fillings_nmax2 \
  --input-dir Wstar_filling_L6_bosons_study/cluster_many_fillings_nmax3 \
  --input-dir Wstar_filling_L6_bosons_study/cluster_many_fillings_nmax4 \
  --input-dir Wstar_filling_L6_bosons_study/cluster_many_fillings_nmax5 \
  --output-dir boson_entropy_prediction_tests/results_L6_lnPR \
  --observable ln_pr_normalized \
  --L-list 6
```

## Example: L=8

```bash
python boson_entropy_prediction_tests/boson_entropy_prediction_tests.py \
  --input-dir Wstar_filling_L8_bosons_study/cluster_L8_common_nmax2 \
  --input-dir Wstar_filling_L8_bosons_study/cluster_L8_common_nmax3 \
  --input-dir Wstar_filling_L8_bosons_study/cluster_L8_common_nmax4 \
  --input-dir Wstar_filling_L8_bosons_study/cluster_L8_common_nmax5 \
  --output-dir boson_entropy_prediction_tests/results_L8 \
  --observable fock_entropy_normalized \
  --L-list 8
```

Adjust the input folder names to match the names actually present on the
cluster.

## Outputs

CSV files:

```text
aggregate_observables.csv
monotonicity_summary.csv
Uhalf_by_W.csv
Uhalf_linear_fits.csv
lambda_shape_comparison.csv
hardcore_plateau.csv
N_crossing_slopes.csv
N_crossings.csv
N_crossing_linear_fits.csv
channel_weights.csv
qeff_summary.csv
qeff_slope_correlation.csv
kappa_filter_model.csv
```

Figures:

```text
entropy_vs_U/
collapse_U_over_W/
Uhalf_vs_W/
lambda_comparison/
hardcore_plateau/
N_crossing/
qeff/
pair_correlation/
kappa_filter/
```

## What this can and cannot test

Can be tested from existing `observables_*.csv`:

- monotonic behavior of entropy and PR versus `U`;
- `U_1/2(W)` and collapse versus `U/W`;
- comparison with the resonance function `Lambda_B^site`;
- hard-core large-`U` plateau check;
- crossings between different `N`;
- channel weights `a_q`, `q_eff^2`, and their relation to entropy slopes;
- averaged correlation between entropy and onsite pair density;
- combinatorial filtering model based on `kappa(U,W)`.

Requires extra eigenstate-level output if we want it exactly:

- state-by-state scatter of entropy versus pair number;
- distributions of pair number in individual eigenstates.

