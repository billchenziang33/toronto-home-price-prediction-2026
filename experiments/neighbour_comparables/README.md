# Nearest-neighbour comparables

**Status (October 22, 2026):** failed screening; not sent to confirmation.

Parent: the building-history model (`experiments/building_history`). Candidate adds, for each home, stats of the 10 and 30 nearest sales of the same property type sold in the 24 months before its sale month: mean, median, and inverse-distance-weighted log price-per-sqft residual (against the lagged type index), mean distance in km, and mean age.

Run: `python experiments/neighbour_comparables/screen.py <months> <out.csv>`.

Screening (2025-06, 2025-09, 2025-11; n=6,185 per track):

| Track | Parent MdAPE | Candidate MdAPE | Relative gain | MAPE | Paired 95% CI (pp) | Months improved |
|---|---:|---:|---:|---|---:|---:|
| Off-market | 8.391% | 8.423% | -0.39% | 11.443 → 11.329 | [-0.137, +0.196] | 0/3 |
| List-aware | 4.264% | 4.257% | +0.17% | 5.401 → 5.399 | [-0.062, +0.066] | 1/3 |

Coordinates and community are already model inputs, so the trees appear to recover local price levels without explicit comparables. The six confirmation months were not used for this method.
