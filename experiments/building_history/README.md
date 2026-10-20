# Same-coordinate (building) price history

**Status (October 20, 2026):** screened and confirmed against v2 C0 on development folds. Not yet a model candidate: the reserved 2025-05 / 2025-08 folds are still unopened, and no v3 package or submission CSV has been produced.

## Idea

59% of test homes (off-market 59.3%, list-aware 59.1%) share exact coordinates with at least one training sale. Duplicate-coordinate groups are mostly condo buildings (81% Condo Apt, 11% Condo Townhouse; up to 328 sales per coordinate). Earlier work only used community-level priors (method card M08), never building-level history.

For each home we add, using only sales sold strictly before it (training rows: before their own sale date; evaluation rows: before the fold cutoff):

- `b_n`, `b_res_mean`, `b_last_lp`, `b_last_res`, `b_age`: count of earlier sales at the same coordinates, their mean and latest log price-per-sqft residual against the previous complete month's property-type median, the latest log sold price, and months since that sale.
- `u_n`, `u_last_lp`, `u_age`: the same for an exact unit match (coordinates, type, bedrooms, washrooms, area text).

Everything else is identical to v2: same feature function, LightGBM presets, and targets (log price off-market; log sold/list list-aware). List-aware is compared without v2's 0.75 HGBR blend on both arms.

## Results

Run: `python experiments/building_history/screen.py <months> <out.csv>`. Paired 95% CI is a 1,000-replicate home bootstrap of candidate-minus-C0 MdAPE in percentage points.

**Screening** (2025-06, 2025-09, 2025-11; v2 development months, n=6,185 per track). C0 off-market MdAPE reproduces v2's recorded 9.5107%.

| Track | C0 MdAPE | Building MdAPE | Relative gain | MAPE | Paired 95% CI (pp) | Months improved |
|---|---:|---:|---:|---|---:|---:|
| Off-market | 9.511% | 8.391% | +11.78% | 12.476 → 11.443 | [-1.367, -0.857] | 3/3 |
| List-aware | 4.399% | 4.264% | +3.08% | 5.455 → 5.401 | [-0.221, -0.053] | 3/3 |

**Confirmation** (2023-03, 2023-07, 2023-11, 2024-03, 2024-06, 2024-09; same six months as `model_research_next`, n=12,372 per track). No code or setting changed after screening.

| Track | C0 MdAPE | Building MdAPE | Relative gain | MAPE | Paired 95% CI (pp) | Months improved | Worst month |
|---|---:|---:|---:|---|---:|---:|---|
| Off-market | 9.316% | 8.143% | +12.59% | 12.174 → 10.990 | [-1.342, -1.001] | 6/6 | all improve |
| List-aware | 5.088% | 4.889% | +3.91% | 6.294 → 6.035 | [-0.271, -0.116] | 5/6 | 2024-09 +2.1% |

Both tracks meet the `model_research_next` confirmation gates (≥1% pooled gain, ≥5/6 months, worst month ≤5% regression, MAPE regression ≤1%, CI upper bound below zero).

Gains concentrate where buildings repeat: off-market Condo Apt 7.90% → 6.46% and Condo Townhouse 7.83% → 7.00% in confirmation, while Detached moves only 12.10% → 11.81%. Freehold houses rarely share coordinates, so their remaining error needs a different idea (for example nearest-neighbour comparables).

Per-home absolute percentage errors are in `outputs/building_history/`.
