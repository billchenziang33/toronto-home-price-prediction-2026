# Fresh-fold model research implementation plan

> **For agentic workers:** Execute inline in the current authorized goal. Preserve all prior worktree changes and the v1/v2 experiment artifacts.

**Goal:** Evaluate three predeclared model objectives against frozen v2 on four unused walk-forward development months, then use two untouched months once for final acceptance if both tracks qualify.

**Architecture:** Isolate the new protocol, runner, per-fit worker, audit, and output files under `experiments/model_research_fresh/` and `outputs/model_research_fresh/`. Reuse the frozen v2 fold builder, feature code, learner adapters, evaluation function, and C0 inference rules without changing their behavior. Store each fit as a start/terminal ledger pair and each prediction artifact with a configuration/data/code/runtime fingerprint.

**Tech Stack:** Python 3.13.1, pandas, NumPy, LightGBM 4.7.0, CatBoost 1.2.10, scikit-learn 1.7.2, PyArrow 22.0.0, pytest.

## Global Constraints

- Training source: `data/.hf_cache/hub/datasets--Baliyang--SSU_data_dash/snapshots/96a258439e2a85f650699be8339e52d8ac5614b7/listings.parquet` with SHA-256 `18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`.
- Development folds: 2024-10, 2024-12, 2025-01, 2025-03; one-time final folds: 2025-05, 2025-08.
- Never inspect hidden test targets. Off-market homes never receive `list_price`; each fold uses only `date_sold` strictly before its month.
- Use frozen v2 features, locked per-track configs, list blend weight 0.75, and existing price rounding order.
- Fixed candidate set: LightGBM quantile q=.5, CatBoost MAE on frozen log targets, LightGBM Gamma on positive dollar targets. No parameter search or new feature cards.
- Stop after 180 counted learner/calibration fit attempts or 240 minutes. Cap CPU threads at four and run one learner process at a time; preserve final acceptance time.
- Do not overwrite `release_candidate/`, `model_candidate_v2/`, or the v1/v2 experiment trees.

---

### Task 1: Freeze the month audit and protocol

**Files:**
- Create: `experiments/model_research_fresh/protocol.json`
- Create: `experiments/model_research_fresh/audit_months.py`
- Create: `outputs/model_research_fresh/month_audit.json`
- Test: `tests/test_fresh_fold_research.py`

**Interfaces:** `audit_months.py` reports all prior ledger validation months, prediction artifact month sets, available training months and counts, the selected splits, source hash, and basic listing quality checks. It must only read the training listings parquet and historical model prediction outputs.

- [ ] Add tests that reject overlap between used and selected months, reject a final month at or before the latest development month, and validate the held-out month row counts.
- [ ] Run those tests and confirm they fail because the new module is absent.
- [ ] Implement the audit and frozen protocol; do not fit a model in this task.
- [ ] Re-run the audit tests and verify all six selected folds have nonempty, unique-ID evaluation rows.

### Task 2: Specify candidate pipelines and adoption gates

**Files:**
- Create: `experiments/model_research_fresh/candidates.py`
- Test: `tests/test_fresh_fold_research.py`

**Interfaces:** `candidate_spec(candidate_id, track, c0_config)` returns a fixed learner family, target mode, config, and post-processing definition. `evaluate_gate(report, criteria)` returns each named criterion and a single pass boolean.

- [ ] Test exact Q50, CB_MAE, and GAMMA_PRICE configurations; test the positive-price requirement for Gamma and the frozen 0.75 list-aware blend.
- [ ] Verify the tests fail before adding the module.
- [ ] Implement fixed specs and gates without fitting data.
- [ ] Verify tests pass and run a toy fit for each learner family, checking the fitted objective and finite positive output.

### Task 3: Implement isolated, resumable fit orchestration

**Files:**
- Create: `experiments/model_research_fresh/worker.py`
- Create: `experiments/model_research_fresh/runner.py`
- Test: `tests/test_fresh_fold_research.py`

**Interfaces:** `worker.execute(job_path)` consumes one immutable fit job and writes one prediction CSV plus metadata. `runner.run_stage(stage, state, protocol)` records every fit attempt, enforces the time and fit caps, accepts only hash-verified cache entries, and can resume after interruption.

- [ ] Test fold target hiding, unique monthly IDs, fit accounting, timeout status, cache fingerprint invalidation, and resume of terminal artifacts.
- [ ] Confirm new tests fail before implementing the worker/runner.
- [ ] Build C0 and candidate development predictions for all four months, fitting one model process at a time.
- [ ] Run candidate tests and the focused model-research test suite.

### Task 4: Select candidates using development folds only

**Files:**
- Create: `outputs/model_research_fresh/development_results.json`
- Create: `outputs/model_research_fresh/freeze_manifest.json`
- Modify: `outputs/model_research_fresh/run_state.json`

- [ ] Compare Q50, CB_MAE, and GAMMA_PRICE to C0 on exact matched IDs for each track and development month.
- [ ] Calculate pooled and monthly MdAPE/MAPE, paired-home bootstrap intervals, and candidate gate results.
- [ ] Select a per-track candidate only if it passes all development criteria. If both tracks qualify, lock the method/configuration and record hashes before reading final-month labels.
- [ ] If either track has no passing candidate, record that decision, leave May/August unopened, update `summary.md`, and stop without v3 export.

### Task 5: One-time final review and conditional export

**Files:**
- Create: `outputs/model_research_fresh/final_results.json`
- Create: `outputs/model_research_fresh/predictions/final/`
- Conditional create only if both final gates pass: `model_candidate_v3/`

- [ ] Fit final C0 and frozen per-track candidates for 2025-05 and 2025-08 exactly once.
- [ ] Evaluate the final gate from the frozen manifest and publish the result without changing candidate selection.
- [ ] Export v3 only if both tracks pass; use hidden test homes for feature-only prediction and never inspect hidden sale labels.
- [ ] Verify official CSV schema/counts and all candidate/code/data/environment hashes if exported.

### Task 6: Close the research record

**Files:**
- Modify: `summary.md`
- Verify: `outputs/model_research_fresh/trial_ledger.jsonl`, `run_state.json`, `final_results.json`

- [ ] Record unused-month evidence, M13/off-market diagnostics, candidate states, all fits, development and final gates, and the retained/exported model decision.
- [ ] Run the full repository test suite with `OMP_NUM_THREADS=1` and `git diff --check`.
- [ ] Verify every stated result from the saved JSON/CSV and leave older artifacts untouched.
