# Bosonic entropy-peak validation report

This report separates resolved agreement, resolved disagreement, and numerically unresolved peaks.

## Summary

- Points analyzed: 10
- Resolved peaks: 0
- Consistent with independent channel theory: 0
- Inconsistent with independent channel theory: 0
- Numerically unresolved: 10

A point is unresolved when the maximum lies on the scan boundary, its plateau includes U=0,
the paired peak significance is too small, the bootstrap interval is too broad, the local
curvature is not robustly negative, or the estimated realization requirement exceeds the cap.

## Resolved points

| L | N | nmax | W/t | U_M*/t | U_S*/t | 95% CI | status |
|---:|---:|---:|---:|---:|---:|:---|:---|
| - | - | - | - | - | - | - | no resolved points |

## Answers to the validation questions

1. **Resolved maxima.** The complete list is the table above. Unresolved reasons: broad_bootstrap_interval=5, local_fit_rejected=6, low_bootstrap_valid_fraction=6, low_peak_significance=10, peak_estimators_disagree=6, plateau_includes_U0=10, scan_boundary=6.
2. **Direct theory/ED agreement.** Of 0 resolved points, 0 are consistent and 0 are inconsistent after adding the separate grid error.
3. **Small W.** For the independent channel theory, the median relative error of `A W + B W^3` is 0.000259 (maximum 0.0452). The ED coefficient fits and bootstrap intervals are in `ed_small_W_fits.csv`.
4. **Large W theory.** The median relative error of `C sqrt(W)` is 0.126 (maximum 0.14); `effective_exponent.pdf` shows the approach to beta=1/2.
5. **Large W entropy.** No ED points reach the large-W criterion.
6. **nmax, filling and size.** There are no resolved ED peaks from which to infer these trends.
7. **nmax=2 control.** Its positive-channel counts are []; it is plotted separately and is not assumed equivalent to nmax>2.
8. **Finite-size mismatch.** Fewer than three matched sizes have resolved peaks, so the L trend is unresolved.

## Small-W coefficient fits

No sector has enough resolved small-W points for an ED fit.

## Interpretation checklist

1. Resolved maxima and their theory comparison are listed above.
2. Grid uncertainty is stored separately in `peak_summary.csv` and is added to the confidence interval only for the consistency test.
3. The analytic small-W coefficients A and B are in `theory_coefficients.csv`; paired-bootstrap ED fits are in `ed_small_W_fits.csv`.
4. The exact theory and its square-root asymptote are compared in `asymptotic_tests.csv`.
5. Unresolved large-W plateaus are not counted as evidence against the theory.
6. Dependence on nmax, N, and L is shown by the comparison figures.
7. nmax=2 is marked separately because it contains only one positive compensating channel.
8. Finite-size trends can be assessed only where multiple L values have resolved peaks.
