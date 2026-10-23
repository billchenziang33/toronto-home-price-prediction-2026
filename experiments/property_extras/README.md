# Lot size and garage

**Status (October 22, 2026):** off-market passed screening direction and confirmation; list-aware failed screening and is not adopted. Reserved 2025-05 / 2025-08 folds are still unopened; no v3 package or submission CSV has been produced.

v2's feature function never reads `lot_front`, `lot_depth`, or `garage`. The candidate adds those three plus `lot_area = lot_front * lot_depth` (missing for condos) to the building-history parent (`experiments/building_history`). Nothing else changes.

Run: `python experiments/property_extras/screen.py <months> <out.csv>`.

**Screening** (2025-06, 2025-09, 2025-11; n=6,185 per track):

| Track | Parent MdAPE | Candidate MdAPE | Relative gain | MAPE | Paired 95% CI (pp) | Months improved |
|---|---:|---:|---:|---|---:|---:|
| Off-market | 8.391% | 8.257% | +1.59% | 11.443 → 11.108 | [-0.313, +0.073] | 3/3 |
| List-aware | 4.264% | 4.292% | -0.66% | 5.401 → 5.388 | [-0.035, +0.096] | 1/3 |

Before confirmation we declared that only off-market would be judged, because list-aware had already failed screening.

**Confirmation** (2023-03, 2023-07, 2023-11, 2024-03, 2024-06, 2024-09; n=12,372 per track). These months were also used to confirm building history, so this is their second use for selection.

| Track | Parent MdAPE | Candidate MdAPE | Relative gain | MAPE | Paired 95% CI (pp) | Months improved | Worst month |
|---|---:|---:|---:|---|---:|---:|---|
| Off-market | 8.143% | 7.922% | +2.71% | 10.990 → 10.829 | [-0.335, -0.086] | 5/6 | 2023-07 +0.3% |
| List-aware (not adopted) | 4.889% | 4.853% | +0.74% | 6.035 → 6.028 | [-0.085, +0.018] | 5/6 | 2023-07 +0.5% |

Off-market Detached moves 11.81% → 11.27% and Semi-Detached 10.49% → 10.18%; Freehold Townhouse worsens 9.98% → 10.70% on 343 homes.

## Current best stack vs v2 C0 (confirmation months)

| Track | v2 C0 | Current best | Relative gain |
|---|---:|---:|---:|
| Off-market | 9.316% | 7.922% (building + lot/garage) | +14.96% |
| List-aware | 5.088% | 4.889% (building) | +3.91% |
