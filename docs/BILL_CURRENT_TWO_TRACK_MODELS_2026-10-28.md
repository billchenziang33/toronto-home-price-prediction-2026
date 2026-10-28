# Current Best Models: Off-market and List-aware

**Prepared:** 2026-10-28
**Repository branch:** `bill`
**Campaign:** `bill_target74_unbounded_20261024`
**Purpose:** concise, auditable model notes and presentation-ready facts for the team.

## 1. Executive summary

The current best **observed eight-month development scores** are:

| Track | Best observed version | Parent used in the comparison | MdAPE: parent → candidate | MAPE: parent → candidate | Status |
|---|---|---|---:|---:|---|
| Off-market | **R151**: R52 base plus a past-only spatial residual correction | R22 | 7.1559% → **6.9989%** | 9.6679% → **9.5180%** | Passes the registered eight-month development gate and is strictly below the 7% MdAPE goal. |
| List-aware | **R97**: R35 base plus a past-only CatBoost residual model | R35 | 4.5785% → **4.5441%** | 5.7869% → **5.7943%** | Lowest observed pooled MdAPE, but the paired interval crosses zero, MAPE rises slightly, and the 4% goal is not met. Do **not** describe this as a confirmed upgrade. |

R151 reduces pooled MdAPE by **0.1570 percentage points** (2.19% relative) and pooled MAPE by 0.1498 points. It improves MdAPE in 7 of 8 development months. Its paired home-bootstrap 95% interval for candidate-minus-parent MdAPE is **−0.2313 to −0.0576 percentage points**, entirely below zero.

The 7% threshold is cleared by only about **0.0011 percentage points**, so describe it as just under target rather than a wide-margin win. R97 remains about **0.5441 points above** its strict 4% target.

R97 reduces pooled MdAPE by only **0.0344 points** (0.75% relative), while pooled MAPE increases by 0.0074 points. It improves MdAPE in 4 of 8 months. Its paired interval is **−0.0713 to +0.0258 points**, which includes zero. A safer list-aware reference remains R35 until a candidate passes the pre-registered gates.

These are development results, not a clean final holdout estimate. The same development months were used in a long model-search campaign. Hidden labels were not opened. The campaign has recorded a reserve-data access incident, so the reserved months are not eligible to be presented as an untouched validation set.

## 2. Task, data, and scoring

The challenge asks for sale-price predictions from the supplied synthetic listing data. The two tracks differ in what information is available at prediction time:

- **Off-market:** predict sold price without using list price.
- **List-aware:** predict sold price with list price available.

The model work described here uses the competition dataset, including its property fields, listing remarks where present in the parent model, location fields, and historical sale outcomes available before each evaluation month. The winning R151/R97 pipelines do not use outside housing datasets or current real-market prices. A Bank of Canada mortgage-rate feature was tried later in R179 but was not promoted.

The campaign data fingerprint is SHA-256 `18cf87fa122b544018fccbc15040403b0fed37080c44a75a03d87a1e5d2d4b2f`. The safe eight-month development window is:

`2023-03, 2023-07, 2023-11, 2024-03, 2024-06, 2024-09, 2025-01, 2025-03`.

Each track has **15,642** scored home-month predictions across those eight months. The evaluation is expanding/strict-past walk-forward: for a target month, historical rows must be earlier than that month; target-month or later sale outcomes do not enter its residual calibration or residual learner.

For home (i), define absolute percentage error (APE) as:

```text
APE_i = 100 × |predicted_price_i − sold_price_i| / sold_price_i
MdAPE = median(APE_i)
MAPE  = mean(APE_i)
```

MdAPE is the primary contest target. MAPE is a guardrail that helps catch a model that improves the median while worsening the average error. The confidence intervals below resample paired homes; they do not model shared month-level market shocks or the extra selection uncertainty from trying many candidates.

## 3. Off-market best observed model — R151

### 3.1 Model stack

R151 is a post-model calibration layer. It does not train a new price learner. For each evaluation month, it starts from saved R52 out-of-fold (OOF) predictions. R52 is a past-only adaptive blend built from the R22/R32 prediction lineage; R32 appends a same-house-type nearest-comparable feature block to R22. R22 includes fold-local listing-remarks TF-IDF in its underlying LightGBM components. R151's comparison parent in the result record is the saved R22 prediction.

The R151 adjustment uses only earlier R52 OOF prediction errors and the query's house type and coordinates. It does not use list price. Its logic is:

1. For each past OOF sale (j) strictly before cutoff (t), calculate the log residual:

   ```text
   r_j = log(sold_price_j / R52_prediction_j)
   ```

2. Restrict candidates to the same property type. Find up to the 50 nearest earlier homes within 1.5 km. The local residual weights combine a Gaussian distance kernel (0.75 km bandwidth) and a 12-month recency half-life:

   ```text
   w_j = exp(−0.5 × (distance_j / 0.75 km)^2)
         × exp(−ln(2) × age_months_j / 12)
   ```

3. Let `m_local` be the weighted median residual among those local neighbors, and `m_type` the recency-weighted median residual for all earlier sales of that property type. Calculate effective neighbor count and shrink the local-versus-type difference:

   ```text
   n_eff = (sum(w_j))^2 / sum(w_j^2)
   delta = clip((m_local − m_type) × n_eff / (n_eff + 20), −0.035, +0.035)
   ```

4. Apply the log-scale correction to the base prediction and round to the nearest dollar:

   ```text
   R151_prediction = round(R52_prediction × exp(delta))
   ```

Require at least 100 earlier rows of that property type and at least 15 spatial neighbors. Missing/invalid coordinates or insufficient support leaves the R52 prediction unchanged. The ±0.035 log cap limits the multiplicative adjustment to roughly −3.44% to +3.56%.

### 3.2 Eight-month result

| Month | Homes | R22 parent MdAPE | R151 MdAPE | Relative MdAPE reduction |
|---|---:|---:|---:|---:|
| 2023-03 | 2,466 | 7.4372% | 7.2570% | 2.423% |
| 2023-07 | 2,010 | 7.2227% | 7.2299% | −0.099% |
| 2023-11 | 1,625 | 7.3116% | 7.0624% | 3.409% |
| 2024-03 | 2,250 | 7.0159% | 6.9356% | 1.144% |
| 2024-06 | 2,196 | 7.1253% | 7.0107% | 1.608% |
| 2024-09 | 1,825 | 7.4820% | 7.3814% | 1.346% |
| 2025-01 | 1,339 | 6.7905% | 6.5977% | 2.838% |
| 2025-03 | 1,931 | 6.8104% | 6.7211% | 1.312% |
| **Pooled** | **15,642** | **7.1559%** | **6.9989%** | **2.194%** |

MAPE falls from 9.6679% to 9.5180%. The paired home-bootstrap used 2,000 replicates; its 95% interval for `R151 − R22` MdAPE is −0.2313 to −0.0576 points. The recorded R151 eight-month run used 8 calibration fits and 0 new learner-training fits.

## 4. List-aware best observed score — R97

### 4.1 Model stack

R97 adds a residual meta-learner to the saved R35 prediction. R35 uses the R22 listing model as its parent; for Detached listings it trains the existing G01 LightGBM core with fold-local R22 remarks TF-IDF and blends 75% of that specialist with 25% of the frozen HGBR component. Other homes keep the R22 prediction.

For each target month, R97 trains only from strictly earlier OOF rows, and only after at least 4,000 historical OOF rows are available. The regression target is the clipped log price residual:

```text
training_residual = clip(log(sold_price / R35_prediction), −0.15, +0.15)
```

It fits CatBoost with MAE loss, 500 iterations, depth 4, learning rate 0.03, L2 leaf regularization 20, random strength 1, seed 20261025, and four threads. The residual model uses:

- Numeric: log parent prediction, log house area, bedrooms, washrooms, parking total, latitude, longitude, and listing-month index.
- Categorical: house type, municipality, community, house style, and garage.
- List-aware additions: log list price and log(list price / parent prediction).

The residual model itself excludes sold price, sale date, outside AVMs, and remarks. Sale labels are used only to form earlier residual targets and score the held-out month. Remarks may still affect the inherited R35/R22 base prediction.

For predicted residual `r_hat`, R97 shrinks and bounds its effect before multiplying the parent price:

```text
correction = 0.50 × clip(r_hat, −0.10, +0.10)
R97_prediction = round(R35_prediction × exp(correction))
```

This caps the correction at ±0.05 log points (about −4.88% to +5.13%). The eight-month run recorded 12 CatBoost fit starts across both tracks. R97's off-market result was worse than its parent, so it is not the off-market winner.

### 4.2 List-aware eight-month result

| Month | Homes | R35 parent MdAPE | R97 MdAPE | Relative MdAPE reduction |
|---|---:|---:|---:|---:|
| 2023-03 | 2,466 | 4.5967% | 4.5967% | 0.000% |
| 2023-07 | 2,010 | 5.3862% | 5.3862% | 0.000% |
| 2023-11 | 1,625 | 4.7163% | 4.5402% | 3.733% |
| 2024-03 | 2,250 | 4.4573% | 4.6439% | −4.187% |
| 2024-06 | 2,196 | 4.8673% | 5.1144% | −5.076% |
| 2024-09 | 1,825 | 4.6694% | 4.3276% | 7.321% |
| 2025-01 | 1,339 | 3.9297% | 3.7710% | 4.038% |
| 2025-03 | 1,931 | 3.8909% | 3.8787% | 0.314% |
| **Pooled** | **15,642** | **4.5785%** | **4.5441%** | **0.752%** |

Pooled MAPE changes from 5.7869% to 5.7943%. Four months improve, two are unchanged, and two regress. The paired home-bootstrap used 2,000 replicates; its 95% interval for `R97 − R35` MdAPE is −0.0713 to +0.0258 points. It crosses zero, so the result does not establish a reliable gain. The strict `<4%` MdAPE target remains unmet.

## 5. How many attempts have run; what happened most recently?

The campaign protocol currently contains **170 numbered R experiment cards**, through R179; 149 card identities have at least one `fit_start` event. The ledger contains 1,005 fit starts, 991 `fit_complete` events with status `complete`, and 3 `fit_complete` events with status `failed`. It separately records 2 interrupted fits, 1 abandoned fit, and 6 fit-attempt reconciliation events. These are fit events, not 1,005 distinct model designs: the campaign includes calibration, model fits, retries, and screening. Of the starts, 136 are explicitly marked calibration-only; older ledger rows do not consistently tag every fit type.

The newest completed numbered candidate is **R179**, which added the Bank of Canada five-year uninsured mortgage lending rate lagged by two months. It failed the two-month screening gate and did not replace either winner:

| Track | Screen parent MdAPE | R179 MdAPE | Gate outcome |
|---|---:|---:|---|
| Off-market | 7.3044% | 7.3091% | Worse overall; no promotion. |
| List-aware | 4.7274% | 4.7209% | 0.138% relative gain, below the 0.25% gate; only 1 of 2 screen months improved; no promotion. |

The separate R179 post-run audit is currently waiting for the configured 4.5 GiB pre-import memory threshold. It has **0 audit fit starts** so far. R180 has not been registered or started. This waiting audit does not change the R151/R97 score summary.

## 6. Caveats to say aloud

1. These are scores on the eight-month development window after many rounds of experimentation. They are not a fresh, untouched final test.
2. Home-level bootstrap intervals do not capture month-to-month market shocks or model-selection optimism.
3. A separate preflight materialized the full parquet and opened reserved-month labels, though the incident record says they were not printed, scored, used for fitting, or used for model choice. The campaign therefore marks reserve validation as ineligible. Do not describe the reserved months as an untouched test; do not use them to tune or select another model.
4. The challenge data are synthetic. Do not present these predictions as verified forecasts of actual Toronto home prices or compare them directly with current TRREB averages.
5. R97's 4.5441% is the best observed List-aware point estimate, not a statistically confirmed replacement. Use that distinction in the report and video.

## 7. Presentation-ready story

Suggested five-slide sequence:

1. **Problem and tracks:** predict sold price from synthetic listings; explain why list price is available only in List-aware.
2. **Evaluation design:** strict-past walk-forward across eight months, with 15,642 scored homes per track; define MdAPE and the MAPE guardrail.
3. **Off-market method:** show `past OOF residuals → same-type nearby neighbors → reliability shrinkage → corrected price`; report R151's 6.9989% MdAPE and confidence interval.
4. **List-aware method:** show `R35 price → CatBoost predicts clipped log residual → bounded correction`; report 4.5441% as the best observed score and state that the paired interval crosses zero.
5. **What we learned / next step:** R151 meets its development target; List-aware still needs a reliable improvement below 4%; R179's mortgage-rate screen did not qualify; reserve-month scores are not presented as clean holdout evidence.

Presentation-safe wording:

> “On our eight-month walk-forward development set, the best observed Off-market pipeline reached 6.9989% MdAPE, down from 7.1559% for its comparison parent. The List-aware residual model reached 4.5441% versus 4.5785%, but that smaller difference was not statistically conclusive and did not meet our 4% target.”

## 8. Source and run evidence

The following source and evidence files define the runs:

- `experiments/bill_target74/r151_offmarket_r93_radius15km_clip035_vs_r22.py` — Off-market R151 implementation and registered constants.
- `experiments/bill_target74/r93_spatial_oof_calibration.py` — shared spatial weighting and strict-past helpers.
- `experiments/bill_target74/r52_r48_r50_adaptive_meta_blend.py` — saved R52 base blend.
- `experiments/bill_target74/r97_past_oof_catboost_residual_meta.py` — R97 learner, feature list, and correction.
- `experiments/bill_target74/r35_fixed_origin.py` — R35 list-aware base route.
- `experiments/bill_target74/campaign_protocol.json` — pre-registered months, gates, data fingerprint, fit history, candidate status, and current limitations.
- `docs/model_evidence/2026-10-28/off_market_r151_development_results.json` — full R151 score and monthly breakdown.
- `docs/model_evidence/2026-10-28/list_aware_r97_development_results.json` — full R97 score and monthly breakdown.
- Local full fit ledger: `outputs/bill_target74/bill_target74_unbounded_20261024/fits.jsonl`.

The data fingerprint is SHA-256 `18cf87fa122b544018fccbc15040403b0fed37080c44a75a03d87a1e5d2d4b2f`; the dataset is obtained from [Baliyang/SSU_data_dash](https://huggingface.co/datasets/Baliyang/SSU_data_dash). The result JSONs preserve the complete metrics and fold details, but a clean clone still needs the dataset and hash-matched parent OOF artifacts to replay these runs; this handoff does not regenerate the predictions or contest CSVs. No hidden test labels are included in this note.
