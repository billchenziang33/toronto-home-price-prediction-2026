# Fresh-fold model research design

## Objective

Start from the frozen `model_candidate_v2` configurations and test whether a small set of previously untried, mechanism-distinct objectives can improve both off-market and list-aware error on genuinely unused chronological folds. The shipped v2 bundles record `train_rows=150000`, so they contain labels from after the proposed 2024–2025 folds. To avoid temporal leakage, the baseline is reconstructed by fitting the locked C0 learners inside each fold, with training rows strictly earlier than the held-out month. Repeated use of December 2025–May 2026 is excluded from new model selection. Hidden-set labels remain out of scope.

## Evidence and fold selection

The v1 fit ledger records validation/tuning months 2025-06, 2025-09, 2025-11, and 2025-12 through 2026-05; the v2 ledger records 2025-02, 2025-04, 2025-06, 2025-07, 2025-10, 2025-11, and 2025-12 through 2026-05. All legacy prediction CSVs under `outputs`, `release_candidate`, and `model_candidate_v2` confirm these same periods. The training listing table has 150,000 rows, unique `ml_num` values, no missing sale dates or prices, and positive sale prices.

Freeze these unused folds:

- Development: 2024-10, 2024-12, 2025-01, 2025-03.
- Final one-time review, after candidate freeze: 2025-05 and 2025-08.

The training listing counts are 2,451; 1,345; 1,339; 1,931; 2,266; and 1,666 respectively. Each model trains only on sales strictly before the held-out month. December 2025–May 2026 remains a previously inspected retrospective diagnostic and cannot be used as a new validation set.

## Failure diagnosis informing hypotheses

On the old retrospective months, list-aware M13's pooled MdAPE moved from 4.2357% to 4.2496%, while MAPE moved from 5.3575% to 5.4441%. Its MdAPE improved in three of six months, but its MAPE was higher in every month. March 2026 was the largest MdAPE setback; its baseline median signed error was +5.64%, while M13 changed it to -0.36%. M13 also increased MdAPE in the lowest list-price quartile by 2.06% and increased MAPE in the lowest and highest quartiles by 3.01% and 3.76%. These are descriptive signs of a tail/calibration and temporal-generalization problem; they do not prove a cause.

Off-market C0's retrospective monthly MdAPE was comparatively stable (8.94%–9.38%), while error differed by property type: Detached 11.47% and Semi-Detached 10.50%, versus Condo Apt 8.05%. Earlier feature and type-calibration trials did not produce stable gains. The next experiments therefore test loss functions and learner behavior without repeating the previous feature cards or calibrators.

## Frozen candidate hypotheses

All candidates use the frozen v2 feature inputs, four-fold walk-forward development, one fixed configuration, and the existing v2 list-aware blend weight and rounding order. C0 uses the locked v2 LightGBM config per track; list-aware C0 also fits its locked HGBR log-sale-price component on that same fold and blends it at 0.75. The exact same fold-local HGBR predictions are used by the list-aware challenger blends.

1. **LightGBM median quantile (`Q50`)**: objective `quantile`, `alpha=0.5`; train on the frozen log-price target off-market and log-sale/list target list-aware. This tests a conditional-median loss aligned with MdAPE.
2. **CatBoost absolute error (`CB_MAE`)**: `loss_function=MAE`, 600 iterations, depth 6, learning rate 0.05, L2 3; use the same frozen log targets. This tests a different categorical learner under a median loss, instead of repeating the prior CatBoost RMSE configuration.
3. **LightGBM Gamma on dollar price (`GAMMA_PRICE`)**: objective `gamma` on positive raw sold-price targets; all other LightGBM settings stay at frozen C0. For list-aware, blend its dollar predictions at the frozen 0.75 challenger weight with the same-fold frozen HGBR component. This tests a relative-scale likelihood rather than the failed MAPE objective.

No new features, calibration, weights, or hyperparameter search are included in this stage. The experiment registry will mark each candidate completed, failed, or timed out and count every attempted learner fit.

## Evaluation and acceptance

Use identical homes and months for candidate and C0. Development promotion requires, independently for both tracks: at least 1% pooled MdAPE reduction, improvement in at least 3 of 4 months, no monthly MdAPE regression over 5%, MAPE regression no greater than 1%, and a paired-home bootstrap 95% upper bound below zero. Select each track's passing candidate by lowest development MdAPE; break ties by lower MAPE, then by the fixed order Q50, CB_MAE, GAMMA_PRICE. The final two months remain unopened until this selection is frozen.

If both tracks have eligible development candidates, run one final comparison on May and August 2025. Final adoption requires at least 1% pooled MdAPE reduction in each track, improvement in both final months, MAPE regression no greater than 1%, and a paired-home bootstrap upper bound below zero. Report month-cluster sensitivity as descriptive only because there are two final months. Export v3 and official CSVs only if both tracks pass. Otherwise retain `model_candidate_v2`.

## Artifacts and limits

Create an isolated `experiments/model_research_fresh/` package and `outputs/model_research_fresh/` evidence tree. Preserve all v1/v2 artifacts. Record the data, code, configuration, and environment hashes; row IDs and labels for each fold; monthly and pooled metrics; bootstrap results; exact fit counts and timings; the freeze decision; and the final decision. Enforce at most 180 real-data learner/calibrator fits or 240 minutes total, whichever comes first, use at most four CPU threads, train one learner at a time, and keep the final acceptance window available.
