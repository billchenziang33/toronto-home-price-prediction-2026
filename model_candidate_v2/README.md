# Toronto home price model candidate v2

This package contains the two LightGBM models selected on the predeclared June, September, and November 2025 development folds, plus the frozen HGBR list-aware model needed for the locked 0.75 challenger blend.

The retrospective December 2025–May 2026 benchmark improved off-market MdAPE from 10.5956% to 9.1733% and list-aware from 4.3596% to 4.2357%. These months have been inspected before and are not a new independent test.

Run from the repository root with the pinned dependencies in `environment-lock.txt`:

```powershell
python model_candidate_v2/predict_candidate.py
```

The script reads the competition test files through the repository's `data.py`, writes both CSVs under `submissions/`, and applies the official ID/schema/finite-positive-price checks. `features.py` is copied from the frozen candidate and its hash is recorded in the package manifest.

The package uses the competition's synthetic training data only. Model pickle files must be loaded with the Python/scikit-learn/LightGBM versions recorded in the metadata. The six-month benchmark is retrospective; hidden-set performance is unknown.
