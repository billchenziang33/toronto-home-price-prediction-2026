# Remarks Text Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an isolated, leakage-safe test of whether concise remarks topic counts improve either Toronto Fourcast prediction track.

**Architecture:** Keep `features.py`, `models.py`, and submission predictions unchanged. Add a deterministic predeclared lexicon transform and an experiment runner that fits the existing model with those features, compares predictions with the six-month baseline, and saves paired results under ignored `outputs/remarks_text/`.

**Tech Stack:** Python 3.13, pandas, NumPy, scikit-learn 1.7.2, pytest.

## Global Constraints

- Use the existing expanding validation months, December 2025 through May 2026.
- Define text patterns before scoring validation outcomes; do not tune the lexicon with validation text or labels.
- Keep list-price access restricted to the list-aware track.
- Do not use any outside or post-May-2026 data.
- Do not change the production model or public submission files during this experiment.
- Keep all metrics explicitly labeled as validation results; hidden-test performance is unknown.

---

### Task 1: Add deterministic remarks topic features

**Files:**
- Create: `experiments/remarks_text.py`
- Test: `tests/test_remarks_text.py`

**Interfaces:**
- `remarks_features(remarks, index=None)` returns a numeric DataFrame indexed to the input rows with word count and fixed counts for renovation, negative condition, systems, amenities, transit/schools, views, investment/rental, and luxury language.
- `append_remarks_features(structured, remarks)` appends those columns without changing row order or existing dtypes.

- [ ] Add tests for each phrase group, missing remarks, row alignment, and preservation of native categorical dtypes.
- [ ] Run `OMP_NUM_THREADS=1 .\.venv\Scripts\python.exe -m pytest -q tests\test_remarks_text.py` and confirm it fails because the feature helper does not exist.
- [ ] Implement the fixed case-insensitive phrase patterns in the experiment module; use regex boundaries to avoid matching a phrase inside an unrelated word. Do not inspect validation prices when defining them.
- [ ] Re-run the focused tests and confirm all pass.

### Task 2: Add the isolated temporal comparison runner

**Files:**
- Modify: `experiments/remarks_text.py`
- Test: `tests/test_remarks_text.py`
- Read: `outputs/validation_predictions.csv`, `data.py`, `features.py`, `models.py`, `run.py`

**Interfaces:**
- `run_experiment()` fits the existing histogram-gradient-boosting model with structured plus remarks features for each fold and track, then writes `outputs/remarks_text/predictions.csv` and `outputs/remarks_text/report.json`.
- Each prediction row contains `ml_num`, `track`, `month`, `sold_price`, `baseline_price`, and `remarks_price`.

- [ ] Add a test for the paired score helper: on fixed prices, confirm MdAPE/MAPE differences are remarks minus baseline, so a negative delta means improvement.
- [ ] Run the focused test and confirm it fails before the score helper exists.
- [ ] Implement one model per track/fold using the exact `HistGradientBoostingRegressor(random_state=0, categorical_features="from_dtype")` configuration; train text features are derived only from remarks available in the training frame. Obtain baseline predictions by joining on `ml_num` and `track`.
- [ ] Calculate paired MdAPE/MAPE changes by track and month, 1,000-resample paired home bootstrap intervals for overall and March–May MdAPE differences, median signed error, and elapsed runtime. Save predictions and JSON under `outputs/remarks_text/`.
- [ ] Run the focused tests and confirm all pass.

### Task 3: Execute and record the experiment

**Files:**
- Modify: `summary.md`
- Modify: `report/project_plan.md`
- Read: `report/report.tex`, `report/video_script.md`, `output/slides/toronto_fourcast_challenge_slides_v4.pptx`

- [ ] Run `OMP_NUM_THREADS=1 .\.venv\Scripts\python.exe -m experiments.remarks_text` and capture output, metric JSON, and elapsed time.
- [ ] Retain remarks in the final model only if March–May paired MdAPE shows a stable reduction for the affected track, its bootstrap interval excludes zero, and there is no material deterioration in the other track. Otherwise keep the production model and document the negative result.
- [ ] Update report/video claims to match the result and Discord clarification: remarks reflect market dynamics and latent variables; the text may be predictive, but only this experiment can show incremental value; rates are not directly modeled.
- [ ] Run `OMP_NUM_THREADS=1 .\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-cache\basetemp` and inspect the final diff. Do not publish experimental claims or modify submission predictions until the temporal confirmation is complete.
