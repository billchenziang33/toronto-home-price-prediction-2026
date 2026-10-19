# Six-month locked benchmark

The December 2025–May 2026 period has been examined before and is a retrospective benchmark, not a fresh independent confirmation set.

| Track | Frozen MdAPE | Candidate MdAPE | Relative improvement | Frozen MAPE | Candidate MAPE | Home-paired 95% CI (pp) | Months improved | Promotion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| list_aware | 4.3596% | 4.2357% | 2.84% | 5.5651% | 5.3575% | [-0.1937, -0.0601] | 6/6 | PASS |
| off_market | 10.5956% | 9.1733% | 13.42% | 13.4001% | 11.9803% | [-1.6102, -1.2453] | 6/6 | PASS |

## Monthly MdAPE

| Track | Month | Frozen | Candidate | Relative improvement |
|---|---|---:|---:|---:|
| list_aware | 2025-12 | 4.0908% | 3.9843% | 2.60% |
| list_aware | 2026-01 | 4.1635% | 4.0406% | 2.95% |
| list_aware | 2026-02 | 4.4158% | 4.3516% | 1.46% |
| list_aware | 2026-03 | 4.2386% | 4.1041% | 3.17% |
| list_aware | 2026-04 | 4.3213% | 4.2049% | 2.69% |
| list_aware | 2026-05 | 4.7162% | 4.6037% | 2.39% |
| off_market | 2025-12 | 10.9007% | 9.1523% | 16.04% |
| off_market | 2026-01 | 10.4342% | 9.1890% | 11.93% |
| off_market | 2026-02 | 10.3746% | 8.9937% | 13.31% |
| off_market | 2026-03 | 10.3973% | 8.9394% | 14.02% |
| off_market | 2026-04 | 10.4806% | 9.2505% | 11.74% |
| off_market | 2026-05 | 10.9733% | 9.3778% | 14.54% |

## Post-evaluation error diagnostics

Actual sold-price deciles and house types are reported only to locate residual error patterns. They did not affect model inputs or selection.

### list_aware by actual sold-price decile

| Decile | Frozen MdAPE | Candidate MdAPE | Frozen MAPE | Candidate MAPE |
|---:|---:|---:|---:|---:|
| 1 | 4.167% | 3.709% | 5.204% | 4.580% |
| 2 | 3.669% | 3.630% | 4.361% | 4.216% |
| 3 | 3.670% | 3.681% | 4.464% | 4.323% |
| 4 | 3.937% | 3.759% | 4.681% | 4.560% |
| 5 | 4.023% | 3.809% | 4.922% | 4.818% |
| 6 | 4.596% | 4.476% | 5.575% | 5.507% |
| 7 | 4.643% | 4.712% | 5.857% | 5.829% |
| 8 | 4.677% | 4.663% | 5.982% | 6.031% |
| 9 | 5.525% | 5.505% | 6.814% | 6.731% |
| 10 | 5.790% | 5.576% | 7.791% | 6.983% |

### list_aware by property type

| Property type | Frozen MdAPE | Candidate MdAPE | Frozen MAPE | Candidate MAPE |
|---|---:|---:|---:|---:|
| Co-Op Apt | 4.745% | 4.535% | 5.751% | 5.030% |
| Condo Apt | 3.593% | 3.497% | 4.310% | 4.118% |
| Condo Townhouse | 4.504% | 4.092% | 5.504% | 5.238% |
| Detached | 5.460% | 5.350% | 6.877% | 6.622% |
| Freehold Townhouse | 5.506% | 5.497% | 6.855% | 6.774% |
| Semi-Detached | 6.019% | 5.690% | 7.004% | 6.876% |

### off_market by actual sold-price decile

| Decile | Frozen MdAPE | Candidate MdAPE | Frozen MAPE | Candidate MAPE |
|---:|---:|---:|---:|---:|
| 1 | 13.104% | 9.548% | 16.719% | 12.555% |
| 2 | 7.952% | 6.241% | 10.302% | 8.422% |
| 3 | 7.975% | 6.552% | 10.923% | 9.381% |
| 4 | 9.352% | 7.962% | 11.640% | 10.092% |
| 5 | 9.983% | 8.802% | 12.306% | 11.120% |
| 6 | 9.938% | 9.389% | 12.417% | 11.677% |
| 7 | 9.666% | 8.926% | 11.646% | 11.062% |
| 8 | 11.524% | 10.716% | 13.658% | 12.854% |
| 9 | 13.765% | 12.697% | 15.402% | 14.543% |
| 10 | 16.550% | 15.078% | 18.988% | 18.099% |

### off_market by property type

| Property type | Frozen MdAPE | Candidate MdAPE | Frozen MAPE | Candidate MAPE |
|---|---:|---:|---:|---:|
| Co-Op Apt | 14.330% | 10.654% | 21.456% | 19.826% |
| Condo Apt | 9.973% | 8.046% | 12.805% | 10.499% |
| Condo Townhouse | 8.448% | 7.285% | 10.940% | 9.824% |
| Detached | 12.124% | 11.473% | 14.807% | 14.251% |
| Freehold Townhouse | 10.542% | 9.671% | 12.539% | 12.482% |
| Semi-Detached | 11.372% | 10.495% | 13.366% | 12.787% |

The month-cluster interval resamples six months and is highly uncertain; it is shown as a shock-sensitivity diagnostic, not a reliable population interval.
