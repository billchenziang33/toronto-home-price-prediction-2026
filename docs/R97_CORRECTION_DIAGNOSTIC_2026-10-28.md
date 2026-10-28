# R97 List-aware correction diagnostic

This is a descriptive post-hoc analysis of the existing eight-month development predictions, not a new model fit or promotion test. The development months have been reused in model search, so the groups must not be treated as independent confirmation or used to choose a correction threshold on these same rows.

Source: `outputs/bill_target74/bill_target74_unbounded_20261024/batches/r97m_20261025T210147Z/paired_predictions.csv`, SHA-256 `cea1400e859f1da376b51d9e9ccc9a73c297d188b71993050d106a63d2971704`, matching `run_manifest.json`.

Reproduce with `.venv-opt/Scripts/python.exe experiments/bill_target74/r97_correction_diagnostic.py`. Machine-readable output is `docs/model_evidence/2026-10-28/list_aware_r97_correction_diagnostic.json`.

Only `list_aware` rows from the eight registered development months were analyzed: 2023-03, 2023-07, 2023-11, 2024-03, 2024-06, 2024-09, 2025-01, and 2025-03. The file contains 15,642 such rows. No new access to reserve or hidden labels occurred. The already saved `correction_log` is the log-scale correction actually applied to the R35 base prediction.

For each row, APE is `100 * abs(prediction - sold_price) / sold_price`. Rows are partitioned by absolute `correction_log`; MdAPE is the median APE within each partition and MAPE is its mean. Improved/worsened compares each row's candidate APE to its parent APE. The partitions are descriptive and selected using a candidate-generated value.

| Absolute log correction | Rows | Parent MdAPE | R97 MdAPE | Parent MAPE | R97 MAPE | Improved / worsened / unchanged |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4,476 | 4.9120% | 4.9120% | 6.1244% | 6.1244% | 0 / 0 / 4,476 |
| (0, 0.01] | 6,393 | 3.9709% | 3.9602% | 5.0772% | 5.0307% | 3,292 / 3,101 / 0 |
| (0.01, 0.03] | 4,767 | 5.1667% | 5.2080% | 6.4134% | 6.4994% | 2,224 / 2,543 / 0 |
| (0.03, 0.05] | 6 | 10.7169% | 11.1512% | 12.4233% | 12.9245% | 3 / 3 / 0 |

The larger group with corrections between 0.01 and 0.03 has worse group-level median and mean error in this dataset; the six-row group is too small to interpret. This does not establish that reducing corrections improves future performance. A possible research question is whether correction reliability can be estimated using strictly earlier OOF data. Before any such experiment, search prior cards for equivalent reliability or clipping mechanisms and preregister a genuinely distinct method; estimate any decision rule only on past data and use the established screen and promotion gates.

The source hash, permitted months, 15,642-row total, positive prices, finite values and the 0.05 correction cap were checked. This diagnostic requires no training and creates no new fit-ledger events.
