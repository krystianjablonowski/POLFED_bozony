# Reduced entropy-peak model report

Version: 1.0.0

The fit used only component peaks and curvature weights. ED entropy peaks were validation targets.

Overall bootstrap-interval coverage: 78.5%.

## Selected models

| L | N | nmax | model | LOOCV improvement M1 vs M0 |
|---:|---:|---:|:---:|---:|
| 7 | 4 | 4 | M0 | -74.1% |
| 7 | 5 | 5 | M0 | 13.9% |
| 7 | 6 | 6 | M0 | -17.2% |
| 7 | 7 | 7 | M0 | -68.6% |
| 8 | 5 | 5 | M0 | -150.5% |
| 8 | 6 | 6 | M0 | -35.7% |
| 8 | 7 | 7 | M0 | -57.1% |
| 8 | 8 | 8 | M0 | -44.2% |

M1 was selected only when LOOCV improved by at least 10%, d was bootstrap-stable, and eta stayed in [0,1].

## Diagnostic conclusions

- L=7 N=4 nmax=4: LOOCV RMSE=0.02751, median ED half-width=0.01301, coverage=100.0%, B-c_S=-0.01298; does not pass all diagnostics.
- L=7 N=5 nmax=5: LOOCV RMSE=0.004019, median ED half-width=0.007199, coverage=100.0%, B-c_S=0.005336; passes the local diagnostics.
- L=7 N=6 nmax=6: LOOCV RMSE=0.00455, median ED half-width=0.005288, coverage=77.8%, B-c_S=-0.0008065; passes the local diagnostics.
- L=7 N=7 nmax=7: LOOCV RMSE=0.005648, median ED half-width=0.005201, coverage=66.7%, B-c_S=-0.001291; does not pass all diagnostics.
- L=8 N=5 nmax=5: LOOCV RMSE=0.003202, median ED half-width=0.007959, coverage=100.0%, B-c_S=0.001747; passes the local diagnostics.
- L=8 N=6 nmax=6: LOOCV RMSE=0.007854, median ED half-width=0.005517, coverage=50.0%, B-c_S=-0.002767; does not pass all diagnostics.
- L=8 N=7 nmax=7: LOOCV RMSE=0.006425, median ED half-width=0.005127, coverage=66.7%, B-c_S=-0.0004305; does not pass all diagnostics.
- L=8 N=8 nmax=8: LOOCV RMSE=0.005375, median ED half-width=0.004882, coverage=66.7%, B-c_S=0.001525; does not pass all diagnostics.
