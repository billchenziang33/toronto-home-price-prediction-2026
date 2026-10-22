# LightGBM capacity search

**Status (October 22, 2026):** no change adopted. Both tracks keep preset P0. Reserved 2025-05 / 2025-08 folds are still unopened.

`protocol.json` fixed the four presets, months, selection rule, and gates before any fit (commit `5facdcf`). Parents: off-market = v2 features + building history + lot/garage; list-aware = v2 features + building history.

Run: `python experiments/model_capacity/screen.py screen <out.csv>` and `... confirm <out.csv> <off_market_config> <list_aware_config>`. Logs with per-fit MdAPE and fit time are in `outputs/model_capacity/`.

## Screening (2025-06, 2025-09, 2025-11; n=6,185 per track)

| Track | Config | MdAPE vs P0 | Relative gain | MAPE change | Months improved | Qualifies |
|---|---|---|---:|---:|---:|---|
| Off-market | P1 | 8.257 → 8.156 | +1.23% | -0.62% | 2/3 | yes |
| Off-market | P2 | 8.257 → 8.082 | +2.12% | -1.81% | 3/3 | yes |
| Off-market | P3 | 8.257 → 8.098 | +1.93% | -2.18% | 3/3 | yes |
| Off-market | **P4** | 8.257 → 8.067 | +2.31% | -1.14% | 3/3 | yes, selected (lowest MdAPE) |
| List-aware | P1 | 4.264 → 4.351 | -2.04% | +0.61% | 0/3 | no |
| List-aware | P2 | 4.264 → 4.286 | -0.52% | +0.72% | 0/3 | no |
| List-aware | P3 | 4.264 → 4.312 | -1.12% | +0.08% | 0/3 | no |
| List-aware | P4 | 4.264 → 4.410 | -3.43% | +3.67% | 0/3 | no |

No list-aware config qualified, so list-aware stopped at P0.

## Confirmation of off-market P4 (six 2023-2024 months; n=12,372)

| MdAPE | Relative gain | MAPE | Months improved | Worst month | Paired 95% CI (pp) | Outcome |
|---|---:|---|---:|---|---:|---|
| 7.922% → 8.008% | -1.08% | 10.829 → 10.853 | 2/6 | 2023-03 +3.1% | [-0.061, +0.227] | Fail |

Monthly MdAPE (P0 → P4): 2023-03 7.800 → 8.040; 2023-07 8.000 → 8.167; 2023-11 8.061 → 8.151; 2024-03 7.599 → 7.761; 2024-06 7.904 → 7.889; 2024-09 8.480 → 8.132.

P4 helped in the 2025 screening months and in the latest confirmation month, but hurt in 2023 when the training history was shorter. The screening gain did not generalise. Following the protocol, P1-P3 were not run on the confirmation months after P4 failed: picking a runner-up after seeing the confirmation result would turn the confirmation months into another selection set. P4 also costs about 10× the fit time of P0 (≈55 s vs ≈5 s per fold).
