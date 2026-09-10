# Bosonic entropy-peak validation report

This report separates resolved agreement, resolved disagreement, and numerically unresolved peaks.

## Summary

- Points analyzed: 40
- Resolved peaks: 24
- Consistent with independent channel theory: 7
- Inconsistent with independent channel theory: 17
- Numerically unresolved: 16

A point is unresolved when the maximum lies on the scan boundary, its plateau includes U=0,
the paired peak significance is too small, the bootstrap interval is too broad, the local
curvature is not robustly negative, or the estimated realization requirement exceeds the cap.

## Resolved points

| L | N | nmax | W/t | U_M*/t | U_S*/t | 95% CI | status |
|---:|---:|---:|---:|---:|---:|:---|:---|
| 6 | 6 | 3 | 0.5 | 0.11525 | 0.38612 | [0.37154, 0.39775] | inconsistent |
| 6 | 6 | 4 | 0.5 | 0.096564 | 0.33865 | [0.32778, 0.34913] | inconsistent |
| 6 | 6 | 4 | 1 | 0.1896 | 0.1778 | [0.14997, 0.21798] | consistent |
| 6 | 6 | 4 | 1.5 | 0.2774 | 0.24584 | [0.19841, 0.26575] | consistent |
| 6 | 6 | 4 | 2 | 0.35993 | 0.29256 | [0.25502, 0.322] | consistent |
| 6 | 6 | 4 | 2.5 | 0.43782 | 0.28969 | [0.22366, 0.35009] | inconsistent |
| 6 | 6 | 5 | 0.5 | 0.087694 | 0.30574 | [0.29973, 0.3112] | inconsistent |
| 6 | 6 | 5 | 1 | 0.17204 | 0.28902 | [0.27439, 0.29608] | inconsistent |
| 6 | 6 | 5 | 1.5 | 0.25148 | 0.3257 | [0.31608, 0.3353] | inconsistent |
| 6 | 6 | 5 | 2 | 0.32604 | 0.34777 | [0.33517, 0.36204] | consistent |
| 6 | 6 | 5 | 2.5 | 0.39633 | 0.3473 | [0.32717, 0.36374] | consistent |
| 6 | 6 | 5 | 3 | 0.46304 | 0.32205 | [0.30264, 0.34543] | inconsistent |
| 6 | 6 | 5 | 3.5 | 0.52671 | 0.28952 | [0.2507, 0.31804] | inconsistent |
| 6 | 6 | 5 | 4 | 0.58776 | 0.26207 | [0.20937, 0.29408] | inconsistent |
| 6 | 6 | 6 | 0.5 | 0.085118 | 0.31243 | [0.30839, 0.3166] | inconsistent |
| 6 | 6 | 6 | 1 | 0.167 | 0.29465 | [0.2889, 0.29871] | inconsistent |
| 6 | 6 | 6 | 1.5 | 0.24413 | 0.2976 | [0.29186, 0.30345] | consistent |
| 6 | 6 | 6 | 2 | 0.31654 | 0.28219 | [0.27287, 0.29025] | consistent |
| 6 | 6 | 6 | 2.5 | 0.38481 | 0.26148 | [0.24711, 0.27353] | inconsistent |
| 6 | 6 | 6 | 3 | 0.4496 | 0.22336 | [0.20451, 0.24339] | inconsistent |
| 6 | 6 | 6 | 3.5 | 0.51143 | 0.1973 | [0.1688, 0.21754] | inconsistent |
| 6 | 6 | 6 | 4 | 0.57072 | 0.17252 | [0.14682, 0.19434] | inconsistent |
| 6 | 6 | 6 | 4.5 | 0.62778 | 0.16012 | [0.13202, 0.18611] | inconsistent |
| 6 | 6 | 6 | 5 | 0.68286 | 0.14467 | [0.11384, 0.17415] | inconsistent |

## Answers to the validation questions

1. **Resolved maxima.** The complete list is the table above. Unresolved reasons: local_fit_rejected=2, low_bootstrap_valid_fraction=3, low_peak_significance=16, plateau_includes_U0=11, realization_limit_exceeded=14, scan_boundary=2, wide_plateau=4.
2. **Direct theory/ED agreement.** Of 24 resolved points, 7 are consistent and 17 are inconsistent after adding the separate grid error.
3. **Small W.** For the independent channel theory, the median relative error of `A W + B W^3` is 0.000247 (maximum 0.0452). The ED coefficient fits and bootstrap intervals are in `ed_small_W_fits.csv`.
4. **Large W theory.** The median relative error of `C sqrt(W)` is 0.103 (maximum 0.14); `effective_exponent.pdf` shows the approach to beta=1/2.
5. **Large W entropy.** No ED points reach the large-W criterion.
6. **nmax, filling and size.** Median resolved `U_S*/t` by nmax: 3:0.386, 4:0.29, 5:0.314, 6:0.242. Use the dedicated comparison figures for the matched-sector trends.
7. **nmax=2 control.** Its positive-channel counts are []; it is plotted separately and is not assumed equivalent to nmax>2.
8. **Finite-size mismatch.** Fewer than three matched sizes have resolved peaks, so the L trend is unresolved.

## Small-W coefficient fits

| L | N | nmax | points | A theory | A ED [95% CI] | B theory | B ED [95% CI] |
|---:|---:|---:|---:|---:|:---|---:|:---|
| 6 | 6 | 4 | 3 | 0.1945 | 0.415 [0.3822, 0.4614] | -0.005604 | -0.1203 [-0.1461, -0.104] |
| 6 | 6 | 5 | 3 | 0.1767 | 0.4751 [0.4605, 0.484] | -0.005347 | -0.1196 [-0.1243, -0.1131] |
| 6 | 6 | 6 | 3 | 0.1715 | 0.4971 [0.4887, 0.5029] | -0.005182 | -0.1375 [-0.1406, -0.1331] |

## Interpretation checklist

1. Resolved maxima and their theory comparison are listed above.
2. Grid uncertainty is stored separately in `peak_summary.csv` and is added to the confidence interval only for the consistency test.
3. The analytic small-W coefficients A and B are in `theory_coefficients.csv`; paired-bootstrap ED fits are in `ed_small_W_fits.csv`.
4. The exact theory and its square-root asymptote are compared in `asymptotic_tests.csv`.
5. Unresolved large-W plateaus are not counted as evidence against the theory.
6. Dependence on nmax, N, and L is shown by the comparison figures.
7. nmax=2 is marked separately because it contains only one positive compensating channel.
8. Finite-size trends can be assessed only where multiple L values have resolved peaks.
