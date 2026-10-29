# Candidate-route safe OOF assets

These files package the saved, row-level out-of-fold (OOF) predictions and the
training metadata needed to audit the R151 off-market and R97 list-aware research
routes. They are development artifacts, not final hidden-test predictions.

## Contents

- `off_market_r151_safe_oof.csv`: 15,642 OOF rows for the eight registered
  development months, with the paired R52/R22 predictions and exact source
  metadata used by R151.
- `off_market_r47_weight_history.csv`: 19,604 off-market OOF rows for the two
  registered 2022 seed folds and eight development months. R48/R50 use these
  strictly past rows to choose R22/R32 blend weights.
- `off_market_r48_r50_weight_history.csv`: 15,642 rows with aligned R48 and R50
  predictions across the eight development months. R52 uses them to choose its
  later blend share.
- `off_market_r151_candidate_oof.csv`: saved R151 parent and candidate prices
  plus the applied correction for the same 15,642 OOF rows.
- `list_aware_r97_safe_oof.csv`: 15,642 list-aware OOF rows, including R97's
  parent and candidate predictions, residual correction, and source metadata.
- `manifest.json`: data, month, run, source, and artifact hashes plus the metric
  and access-scope records.

The packaged months are March 2023, July 2023, November 2023, March 2024,
June 2024, September 2024, January 2025, and March 2025. The reserved months
2025-05 and 2025-08 are excluded. The R47 weight history also includes only the
registered seed months 2022-06 and 2022-11. No hidden test labels are included
or read by the asset builder. The campaign's reserve-validation eligibility
remains false.

## Rebuild and verify

The source run artifacts referenced by `manifest.json` must be present in the
campaign output tree, and the registered training parquet must match its SHA-256.
On that checkout, run:

```powershell
python tools/build_submission_assets.py
python tools/build_submission_assets.py --extend-offmarket-meta
python tools/build_submission_assets.py --verify
```

On an initial build, the first command verifies R151/R52/R22 and R97 artifacts,
then writes to a new `model_assets/` directory; it refuses to overwrite one that
already exists. The second command verifies archived R47 source snapshots and
hashes, checks its labels and dates against exact safe-month training rows,
verifies R48/R50 results, then adds the two weight-history files and R151
candidate output without replacing existing assets. Both operations reject
reserved months and check listing-key uniqueness. On a checkout that already contains this package, run
the third command: it verifies hashes, row counts, schemas, keys, months, and
positive predictions without opening the source parquet. To rebuild beside an
existing package, pass `--output-dir model_assets_rebuilt` to each of the first
two commands; the destination must be inside the repository.

## Interpretation

R151's 6.9988809893% MdAPE / 9.5180175461% MAPE is an eight-month development
result using R52 as its parent; R52 itself is recorded as `development_failed`.
R47 is also recorded as `development_failed`; its saved OOF rows are used only as
historical weight-selection evidence, with its original status preserved in the
manifest.
R97's 4.5440739072% MdAPE / 5.7942660284% MAPE is also an eight-month
development result. Its paired interval crosses zero and it did not pass all
registered promotion gates. Neither score is a hidden-test result. R151 and
R97 are integrated into `submission_models.py` and `run.py`. The factory selects
R52 for off-market and R97 for list-aware, based on the lowest MdAPE in the
six-month retrospective standard validation; R151 remains a measured
comparator. That six-month window was used in prior development and is not an
independent holdout. The current checked submission CSVs and run manifest are
under `submissions/`.
