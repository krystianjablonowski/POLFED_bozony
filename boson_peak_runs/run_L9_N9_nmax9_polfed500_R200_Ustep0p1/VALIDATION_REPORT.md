# Bosonic entropy-peak validation report

This report separates resolved agreement, resolved disagreement, and numerically unresolved peaks.

## Summary

- Points analyzed: 13
- Resolved peaks: 13
- Consistent with independent channel theory: 7
- Inconsistent with independent channel theory: 6
- Numerically unresolved: 0

A point is unresolved when the maximum lies on the scan boundary, its plateau includes U=0,
the paired peak significance is too small, the bootstrap interval is too broad, the local
curvature is not robustly negative, or the estimated realization requirement exceeds the cap.

## Resolved points

| L | N | nmax | W/t | U_M*/t | U_S*/t | 95% CI | status |
|---:|---:|---:|---:|---:|---:|:---|:---|
| 9 | 9 | 9 | 0.1 | 0.016175 | 0.20549 | [0.20502, 0.20594] | inconsistent |
| 9 | 9 | 9 | 0.3 | 0.048408 | 0.19099 | [0.19024, 0.19166] | inconsistent |
| 9 | 9 | 9 | 0.5 | 0.080303 | 0.16126 | [0.16042, 0.1621] | inconsistent |
| 9 | 9 | 9 | 0.7 | 0.11168 | 0.15557 | [0.15492, 0.1562] | consistent |
| 9 | 9 | 9 | 0.9 | 0.14241 | 0.1605 | [0.15955, 0.1614] | consistent |
| 9 | 9 | 9 | 1.1 | 0.17242 | 0.17413 | [0.17249, 0.1756] | consistent |
| 9 | 9 | 9 | 1.3 | 0.20168 | 0.19229 | [0.18984, 0.19442] | consistent |
| 9 | 9 | 9 | 1.5 | 0.23019 | 0.21023 | [0.20739, 0.21273] | consistent |
| 9 | 9 | 9 | 1.7 | 0.25797 | 0.22373 | [0.22032, 0.22665] | consistent |
| 9 | 9 | 9 | 1.9 | 0.28506 | 0.23194 | [0.22785, 0.23539] | consistent |
| 9 | 9 | 9 | 2.1 | 0.31149 | 0.2368 | [0.23197, 0.24102] | inconsistent |
| 9 | 9 | 9 | 2.3 | 0.33732 | 0.23932 | [0.2337, 0.2445] | inconsistent |
| 9 | 9 | 9 | 2.5 | 0.36258 | 0.23967 | [0.2334, 0.24566] | inconsistent |

## Answers to the validation questions

1. **Resolved maxima.** The complete list is the table above. Unresolved reasons: none.
2. **Direct theory/ED agreement.** Of 13 resolved points, 7 are consistent and 6 are inconsistent after adding the separate grid error.
3. **Small W.** For the independent channel theory, the median relative error of `A W + B W^3` is 0.000594 (maximum 0.0461). The ED coefficient fits and bootstrap intervals are in `ed_small_W_fits.csv`.
4. **Large W theory.** No points satisfy the configured large-W regime test.
5. **Large W entropy.** No ED points reach the large-W criterion.
6. **nmax, filling and size.** Median resolved `U_S*/t` by nmax: 9:0.205. Use the dedicated comparison figures for the matched-sector trends.
7. **nmax=2 control.** Its positive-channel counts are []; it is plotted separately and is not assumed equivalent to nmax>2.
8. **Finite-size mismatch.** Fewer than three matched sizes have resolved peaks, so the L trend is unresolved.

## Small-W coefficient fits

| L | N | nmax | points | A theory | A ED [95% CI] | B theory | B ED [95% CI] |
|---:|---:|---:|---:|---:|:---|---:|:---|
| 9 | 9 | 9 | 10 | 0.1618 | 0.2337 [0.2327, 0.2347] | -0.004964 | -0.03525 [-0.03587, -0.03471] |

## Interpretation checklist

1. Resolved maxima and their theory comparison are listed above.
2. Grid uncertainty is stored separately in `peak_summary.csv` and is added to the confidence interval only for the consistency test.
3. The analytic small-W coefficients A and B are in `theory_coefficients.csv`; paired-bootstrap ED fits are in `ed_small_W_fits.csv`.
4. The exact theory and its square-root asymptote are compared in `asymptotic_tests.csv`.
5. Unresolved large-W plateaus are not counted as evidence against the theory.
6. Dependence on nmax, N, and L is shown by the comparison figures.
7. nmax=2 is marked separately because it contains only one positive compensating channel.
8. Finite-size trends can be assessed only where multiple L values have resolved peaks.
