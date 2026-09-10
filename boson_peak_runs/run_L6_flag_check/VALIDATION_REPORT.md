# Bosonic entropy-peak validation report

This report separates resolved agreement, resolved disagreement, and numerically unresolved peaks.

## Summary

- Points analyzed: 18
- Resolved peaks: 6
- Consistent with independent channel theory: 2
- Inconsistent with independent channel theory: 4
- Numerically unresolved: 12

A point is unresolved when the maximum lies on the scan boundary, its plateau includes U=0,
the paired peak significance is too small, the bootstrap interval is too broad, the local
curvature is not robustly negative, or the estimated realization requirement exceeds the cap.

## Resolved points

| L | N | nmax | W/t | U_M*/t | U_S*/t | 95% CI | status |
|---:|---:|---:|---:|---:|---:|:---|:---|
| 6 | 3 | 3 | 2 | 0.49385 | 0.34508 | [0.31545, 0.37619] | inconsistent |
| 6 | 3 | 4 | 2 | 0.49385 | 0.34436 | [0.31414, 0.37583] | inconsistent |
| 6 | 6 | 2 | 1 | 0.32641 | 0.43871 | [0.38296, 0.51264] | inconsistent |
| 6 | 6 | 4 | 0.5 | 0.096564 | 0.3026 | [0.27987, 0.33576] | inconsistent |
| 6 | 6 | 4 | 1 | 0.1896 | 0.18413 | [0.13932, 0.26235] | consistent |
| 6 | 6 | 4 | 2 | 0.35993 | 0.30556 | [0.24641, 0.35543] | consistent |

## Answers to the validation questions

1. **Resolved maxima.** The complete list is the table above. Unresolved reasons: broad_bootstrap_interval=3, local_fit_rejected=10, low_bootstrap_valid_fraction=10, low_peak_significance=10, plateau_includes_U0=9, realization_limit_exceeded=2, scan_boundary=10, wide_plateau=2.
2. **Direct theory/ED agreement.** Of 6 resolved points, 2 are consistent and 4 are inconsistent after adding the separate grid error.
3. **Small W.** For the independent channel theory, the median relative error of `A W + B W^3` is 0.000119 (maximum 0.0417). The ED coefficient fits and bootstrap intervals are in `ed_small_W_fits.csv`.
4. **Large W theory.** The median relative error of `C sqrt(W)` is 0.0329 (maximum 0.114); `effective_exponent.pdf` shows the approach to beta=1/2.
5. **Large W entropy.** No ED points reach the large-W criterion.
6. **nmax, filling and size.** Median resolved `U_S*/t` by nmax: 2:0.439, 3:0.345, 4:0.304. Use the dedicated comparison figures for the matched-sector trends.
7. **nmax=2 control.** Its positive-channel counts are [1]; it is plotted separately and is not assumed equivalent to nmax>2.
8. **Finite-size mismatch.** Fewer than three matched sizes have resolved peaks, so the L trend is unresolved.

## Small-W coefficient fits

| L | N | nmax | points | A theory | A ED [95% CI] | B theory | B ED [95% CI] |
|---:|---:|---:|---:|---:|:---|---:|:---|
| 6 | 6 | 4 | 2 | 0.1945 | 0.7455 [0.6785, 0.8331] | -0.005604 | -0.5614 [-0.6578, -0.4445] |

## Interpretation checklist

1. Resolved maxima and their theory comparison are listed above.
2. Grid uncertainty is stored separately in `peak_summary.csv` and is added to the confidence interval only for the consistency test.
3. The analytic small-W coefficients A and B are in `theory_coefficients.csv`; paired-bootstrap ED fits are in `ed_small_W_fits.csv`.
4. The exact theory and its square-root asymptote are compared in `asymptotic_tests.csv`.
5. Unresolved large-W plateaus are not counted as evidence against the theory.
6. Dependence on nmax, N, and L is shown by the comparison figures.
7. nmax=2 is marked separately because it contains only one positive compensating channel.
8. Finite-size trends can be assessed only where multiple L values have resolved peaks.
