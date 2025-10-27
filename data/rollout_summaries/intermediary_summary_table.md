**Argo CD deployment and rollout timing over current iterations (mean ± σ) with 95% confidence intervals.**

| Stage | Mean (s) ± σ | 95% CI (s) | n |
|---|---:|---:|---:|
| Repo sync | 5.51 ± 0.97 | [5.36, 5.66] | 163 |
| Manifest apply | 0.14 ± 0.03 | [0.13, 0.14] | 163 |
| Resource creation | 16.15 ± 4.46 | [15.47, 16.84] | 163 |
| Deployment total | 3.01 ± 0.75 | [2.89, 3.12] | 163 |
| Pod rollout | 18.60 ± 4.39 | [17.93, 19.28] | 163 |
| Rollout total | 21.61 ± 4.52 | [20.91, 22.30] | 163 |