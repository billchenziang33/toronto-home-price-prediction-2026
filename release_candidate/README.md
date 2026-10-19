# Toronto Fourcast - SSU Data Dash 2026

This folder is a reviewable submission candidate built from the team's last public model. The production code and the two submission CSVs are byte-for-byte copies of public commit `7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8`. No remarks features are in that frozen model. The remarks experiment is retained separately under `experiments/` and does not replace either CSV.

## Candidate status

The report PDF is within the competition's four-page limit and the deck has six slides. The English narration is written for a recording under five minutes. **The video has not been recorded yet.** The PDF labels that link as pending; replace it with the team's public or unlisted YouTube URL after recording. The candidate package has not been pushed to GitHub or submitted through the form.

## Frozen validation result

The saved predictions cover six expanding walk-forward months, December 2025 through May 2026, with 10,069 homes per track. Each month is predicted using sales from earlier months only.

| Track | Starter MdAPE | Frozen MdAPE (home-bootstrap 95% CI) | Frozen MAPE (home-bootstrap 95% CI) |
|---|---:|---:|---:|
| Off-market | 20.8% | 10.60% (10.38-10.85) | 13.40% (13.17-13.65) |
| List-aware | 7.3% | 4.36% (4.26-4.46) | 5.57% (5.47-5.66) |

The intervals resample homes and do not account for market shocks shared within a month. These are validation results; hidden-test prices and scores are unknown. The synthetic community names and coordinates do not represent a real Toronto map.

## Model and code

`features.py` prepares separate inputs for the off-market and list-aware tracks. Both use numeric property and time fields plus native categories for property type, house style, municipality, and community. Only the list-aware track includes list price. `models.py` fits a `HistGradientBoostingRegressor` to log sale price. `run.py` runs the expanding validation, writes reports, predicts the hidden test inputs, and checks the two CSVs. The model code and submission files are from the public commit listed in `SOURCE_COMMIT.txt`.

## Files

- `report/SSU_DataDash_submission_candidate.pdf`: typed report, currently marked video pending.
- `report/video_script_en.md`: six-part narration aligned with the deck and validation evidence.
- `slides/toronto_fourcast_final.pptx`: six-slide recording deck.
- `submissions/`: the exact two published prediction files.
- `evidence/validation_predictions_frozen.csv`: the saved homes used to recalculate the headline metrics and chart.
- `evidence/validation_metrics.json`: recalculated headline metrics and home-bootstrap intervals.
- `evidence/category_ablation_audit.json`: explains why the older category-gain claim is omitted.
- `experiments/remarks_experiment_audit.json`: corrected paired MdAPE analysis for the archived remarks experiment.
- `DELIVERABLE_CHECKLIST.md`: artifacts, verification evidence, and remaining human steps.
- `summary.md`: decisions and implementation order for this candidate.

## Reproduction

Use Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
python run.py validate
python run.py report
python run.py predict
python run.py check
```

The commands above refit the published model when run. This candidate was assembled from saved predictions and published Git objects without rerunning training. `python run.py check` validates the two CSVs against the supplied test-input IDs; the report and chart are based on the saved walk-forward predictions in `evidence/`.

The linked public repository contains the frozen code and both CSVs. The local candidate's corrected report, deck, narration, and statistical audit are packaged here. A team member should review this package, record the video, replace the pending video line in the PDF, publish the approved candidate files, and submit the official form.
