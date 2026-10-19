# Toronto Home Price Challenge — Team Progress and Experiment Record

**Updated:** October 19, 2026

**Repository:** [billchenziang33/toronto-home-price-prediction-2026](https://github.com/billchenziang33/toronto-home-price-prediction-2026)

## Current decision

`model_candidate_v2/` is still our retained candidate for both competition tracks. It is the strongest complete, reproducible package we have, but the newest experiments did **not** establish a reliable improvement over it on both tracks. We therefore did not create a v3 model or use a new experiment to replace the existing competition-format CSVs.

The early result that made v2 look strong came from a six-month retrospective benchmark that had already informed model selection. Later, unused chronological folds and a larger preregistered search did not confirm a challenger that passed every promotion rule. Those results are not contradictory: they are measured on different time windows, and the later tests are the more relevant evidence for whether an apparent gain generalizes.

The challenge asks us to predict sold price in two tracks:

- **Off-market:** predict sold price without using a list price.
- **List-aware:** predict sold price when list price is available as an input.

We use **MdAPE** (median absolute percentage error) as the main selection metric and **MAPE** as a complementary average-error metric. Lower is better. “Relative MdAPE gain” is the percentage reduction in MdAPE versus the parent model on the same evaluation rows; a positive number is an improvement. Confidence intervals below are paired-home intervals for candidate-minus-parent MdAPE in **percentage points (pp)**. An interval containing zero does not establish that the candidate is better.

## What we tried and what happened

### 1. Baselines, feature experiments, and the retained v2 package

The repository preserves the starter baseline and earlier experiments with structured property features, native categorical handling, comparable-property ideas, and listing remarks. Some older category-gain claims could not be reproduced from the saved predictions and their exact baseline specification was unavailable, so those claims were removed from the candidate report and slides.

The first model search compared LightGBM targets and configurations on three predeclared 2025 development months. The selected LightGBM candidates were then measured on December 2025–May 2026. That later window had already been inspected during the project, so it is a **retrospective benchmark**, not an untouched final test.

| Track | Frozen reference MdAPE / MAPE | v2 candidate MdAPE / MAPE | Retrospective MdAPE gain | Paired 95% CI (pp) |
|---|---:|---:|---:|---:|
| Off-market | 10.5956% / 13.4001% | 9.1733% / 11.9803% | 13.42% | [-1.6102, -1.2453] |
| List-aware | 4.3596% / 5.5651% | 4.2357% / 5.3575% | 2.84% | [-0.1937, -0.0601] |

The retained package uses separate LightGBM models. The off-market model predicts log sold price. The list-aware package predicts a list-price-relative target and retains the selected 0.75 challenger blend with the frozen HGBR reference. The exact weights, feature columns, model files, environment lock, and checksums are in `model_candidate_v2/`.

These scores explain why v2 became the retained candidate; they do **not** estimate hidden-test performance. The hidden competition labels have not been read.

### 2. Listing-remarks text experiment

We tested fixed, interpretable word and topic counts from the remarks field. On the retrospective six-month window, the original point estimates were small: off-market MdAPE changed from about 10.596% to 10.454%, and list-aware from about 4.360% to 4.340%. A later audit corrected the paired-bootstrap calculation and separated the March–May confirmation months. In that confirmation period, the corrected intervals included zero for both tracks: off-market change -0.101 pp (95% CI -0.335 to +0.090) and list-aware -0.037 pp (-0.111 to +0.049). Remarks therefore remain an archived experiment and are not part of the retained candidate.

### 3. Three-checkpoint search against v2

This stage screened 20 method cards, with a staged C0 → C1 → C2 → C3 promotion design. List-aware M13, a LightGBM direct-dollar MAPE objective, improved development MdAPE by 2.15% over four development months, improved three of four months, reduced MAPE by 0.30%, and passed its development bootstrap criterion. However, three feature additions failed the incremental C2 gate, and the fixed 50/50 M13/CatBoost blend also failed. C3 was not run.

The one-time retrospective December 2025–May 2026 check did not confirm M13 as an upgrade. Off-market stayed at C0. List-aware MdAPE worsened by 0.33%, MAPE worsened by 1.62%, and only three of six months improved; the paired interval crossed zero. The package correctly retained v2 and exported no v3.

The tested method families included recency weighting, a rolling training window, time/area/type/coordinate features, as-of price and momentum features, residual calibration, alternative losses and targets, spline Ridge, ExtraTrees, XGBoost, CatBoost, and blending. XGBoost was unavailable in the locked environment; ExtraTrees and some as-of trials exceeded their development timeout. Their statuses are preserved in the ledgers rather than counted as successful experiments.

### 4. Fresh chronological folds

We audited prior runs and evaluated three frozen challenger families on four previously unused chronological folds. Each fold trained only on sales before its held-out month. The folds were 2024-10, 2024-12, 2025-01, and 2025-03, with 7,066 paired homes per track in total. Promotion required at least 1% pooled MdAPE gain, improvement in at least three of four months, no monthly regression above 5%, no MAPE regression above 1%, and a paired-home bootstrap upper bound below zero.

| Track / method | Relative MdAPE gain | Relative MAPE change | Months improved | Paired 95% CI (pp) | Outcome |
|---|---:|---:|---:|---:|---|
| Off-market Q50 | +0.57% | +0.83% worse | 1/4 | [-0.211, +0.156] | Fail |
| Off-market CatBoost MAE | -15.12% | +13.13% worse | 0/4 | [+1.148, +1.633] | Fail; materially worse |
| Off-market Gamma-price | +1.21% | +0.97% worse | 2/4 | [-0.237, +0.055] | Fail; inconsistent months and interval crosses zero |
| List-aware Q50 | +1.54% | -0.84% (better) | 4/4 | [-0.122, +0.023] | Fail; interval upper bound is above zero |
| List-aware CatBoost MAE | +1.41% | -0.26% (better) | 3/4 | [-0.144, +0.035] | Fail; interval upper bound is above zero |
| List-aware Gamma-price | -0.43% | +2.25% worse | 0/4 | [-0.033, +0.117] | Fail |

List-aware Q50 was the closest result, but its interval did not meet the predeclared confidence gate. No challenger was promoted. The two folds reserved for a final check remained unopened.

### 5. Preregistered 20-card calibration and blend search

The next stage audited eight unused 2022–2024 folds, used two months for screening, and confirmed the top four cards per track on six new months (2023-03, 2023-07, 2023-11, 2024-03, 2024-06, and 2024-09). Confirmation contained 12,372 paired homes. The 20 fixed cards tested raw Q50 and CatBoost pipelines; global, property-type, type-by-predicted-price-band, and rolling calibration; OOF-selected blends of C0/Q50, Q50/CatBoost, and C0/recency; and recency weighting. All calibration factors and blend weights used past-only out-of-fold predictions.

| Track / confirmed card | Baseline → candidate MdAPE | Relative MdAPE gain | Relative MAPE change | Months improved | Paired 95% CI (pp) | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Off-market `OM08_C0_recency_oof_blend` | 9.3161% → 9.3341% | -0.19% | +0.12% worse | 0/6 | [-0.033, +0.053] | Fail |
| Off-market `OM07_C0_Q50_oof_blend` | 9.3161% → 9.3229% | -0.07% | -0.51% better | 2/6 | [-0.081, +0.073] | Fail |
| Off-market `OM01_Q50_raw` | 9.3161% → 9.3565% | -0.43% | +0.70% worse | 1/6 | [-0.112, +0.196] | Fail |
| Off-market `OM09_recency_weighted` | 9.3161% → 9.6298% | -3.37% | +2.46% worse | 0/6 | [+0.163, +0.463] | Fail |
| List-aware `LA07_C0_Q50_oof_blend` | 5.0906% → 5.0757% | +0.29% | -0.28% better | 4/6 | [-0.062, +0.034] | Fail |
| List-aware `LA01_Q50_raw` | 5.0906% → 5.0619% | +0.56% | -0.13% better | 4/6 | [-0.091, +0.031] | Fail |
| List-aware `LA09_CB_MAE_raw` | 5.0906% → 5.0786% | +0.24% | -0.06% better | 3/6 | [-0.091, +0.063] | Fail |
| List-aware `LA08_Q50_CB_oof_blend` | 5.0906% → 5.0529% | +0.74% | -0.19% better | 5/6 | [-0.097, +0.048] | Fail |

The confirmation gate required at least 1% pooled MdAPE gain, at least five of six months improved, worst monthly MdAPE regression no more than 5%, MAPE regression no more than 1%, and a paired-bootstrap upper endpoint below zero. No method passed every gate. `LA08` was the strongest recent list-aware point estimate, but its gain was below 1% and its confidence interval included zero. The best off-market point estimate still increased MdAPE.

This run completed in 100.512 minutes with 126/180 fit starts; all 126 completed successfully. Its final-month lock stayed closed, no final results file was created, and no submission CSV was exported from this stage.

## What is in the repository

| Path | Contents / how to interpret it |
|---|---|
| `model_candidate_v2/` | Retained candidate models, inference code, configs, environment lock, benchmark comparison, and SHA-256 manifest. |
| `experiments/model_optimization_v1/` | First model search and benchmark runner. |
| `experiments/model_optimization_v2/` | Three-checkpoint staged search implementation. |
| `experiments/model_research_fresh/` | Audit and four-fold fresh-fold challenger study. |
| `experiments/model_research_next/` | Eight-fold audit, residual diagnostics, 20-card registry, and six-fold confirmation code. |
| `outputs/` | Aggregate results, fold-level and paired predictions, fit ledgers, run logs, manifests, and model artifacts for the recorded studies. |
| `reports/`, `docs/`, `tests/`, `tools/` | Validation reports, preregistered plans, focused tests, and reproducibility utilities. |
| `submissions/` | Current competition-format off-market and list-aware CSVs; latest research did not overwrite them. |
| `release_candidate/` | Preserved frozen **v1** package from commit `7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8`; useful for audit history, but its report/slides/CSVs should not be paired with v2 as the current submission. |
| `output/` | Earlier draft PDF and slide versions. These are working drafts, not v2-aligned final materials. |

The raw `data/` directory and local Python environments are not part of the shared commit. The dataset can be downloaded through the repository's documented Hugging Face workflow. The outputs preserve the experimental evidence without duplicating the raw dataset.

## Reproduce and coordinate

Start from the repository root, install `requirements.txt`, download the challenge dataset as described in `README.md`, then use the experiment-specific `protocol.json` and runner. The corresponding `trial_ledger.jsonl`, manifests, and result JSON files are the source of truth for what ran, what was cached, and what was unavailable or timed out. Focused tests include `tests/test_model_optimization.py`, `tests/test_model_research_v2.py`, `tests/test_fresh_fold_research.py`, and `tests/test_model_research_next.py`.

Before presenting or submitting, the team should align the PDF, slides, narration, code description, and CSVs to the retained v2 candidate. The existing `release_candidate/` material is a verified v1 archive, not a v2-ready submission. The video still needs to be recorded and linked, and the official form still needs to be submitted by the deadline.

## Suggested next handoff

1. Review the retained v2 package and current two submission CSVs as a team.
2. Decide whether the current negative research result is sufficient for modeling freeze or whether to authorize a separately scoped experiment; do not describe a point estimate as a promoted improvement unless it passes the stated gates.
3. Update the final report and slide deck so every metric and model description matches v2, then record the short video and submit the official form.
