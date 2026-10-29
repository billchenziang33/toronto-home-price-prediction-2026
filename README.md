# Toronto Fourcast — SSU Data Dash 2026

Reproducible solution for the synthetic Toronto home-price challenge. It predicts sale prices on two separate tracks: homes without a list price (`off-market`) and homes with a list price (`list-aware`).

## Current candidate routes

The submission factory selects **R52 for off-market** and **R97 for list-aware**. These routes had the lowest MdAPE among the candidates in the six-month standard walk-forward validation below. Full route comparisons, uncertainty intervals, and caveats are in [`reports/validation_report.md`](reports/validation_report.md); model and run fingerprints are in [`submissions/submission_run_manifest.json`](submissions/submission_run_manifest.json).

| Track | Selected route | Homes | MdAPE | MAPE | P90 APE |
|---|---|---:|---:|---:|---:|
| Off-market | R52 | 10,069 | 7.0682% | 9.5223% | 20.2300% |
| List-aware | R97 | 10,069 | 3.9396% | 5.0294% | 10.5750% |

Validation covers December 2025 through May 2026. Each month is predicted using sales strictly before that month's first day. This is a retrospective standard validation window used in prior model development, not a fresh holdout or the hidden test set. The scores are not hidden-test results and may be optimistic after repeated model selection.

## Data and rules

The supplied challenge dataset contains synthetic sales through May 2026 and test listings from June through September 2026. No real listing or sale records, outside datasets, reserve-month labels, or hidden test labels are used. The off-market route excludes list price. Both routes use the supplied `remarks` description through fold-local TF-IDF features; no vocabulary or transformation is fitted using future validation rows.

## Method overview

- **Off-market R52:** combines the R22 and R32 learner routes, including a specialist blend for detached homes. It chooses global and property-group blend weights from strictly past OOF evidence; it does not use list price.
- **List-aware R97:** blends the R22 and frozen HGBR predictions, applies a detached-home expert, then adds a shrunk CatBoost residual correction trained on strictly past OOF rows. List price is available only on this track.
- Both final CSVs use nearest-dollar integer predictions. `run.py check` verifies their required columns, IDs, row counts, and positive finite prices.

## Reproduce

Use the pinned packages in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python run.py validate
python run.py report
python run.py predict
python run.py check
```

The first run downloads the competition data from [Hugging Face](https://huggingface.co/datasets/Baliyang/SSU_data_dash). `predict` writes `submissions/submission_off_market.csv`, `submissions/submission_list_aware.csv`, and a run manifest with data, source, model, and output hashes.

## Main files

- `features.py`, `models.py`, `run.py`, `data.py`: feature pipeline, selected models, validation, prediction, and checks.
- `submission_models.py`, `model_assets/`: frozen candidate routes and strictly-past OOF evidence used for safe post-processing.
- `reports/validation_report.md`, `outputs/error_analysis.csv`: validation results and error breakdowns by month, property type, price decile, and community.
- `report/report.tex`, `report/project_plan.md`, `report/video_script.md`: report source and presentation working drafts.
- `output/slides/toronto_fourcast_challenge_slides_v4.pptx`: editable draft slides for the video; a separate deck is not listed as a required submission.

## Competition submission

The handout and rubric require one typed PDF report of at most four pages, submitted through the form. The PDF links to a YouTube video of at most five minutes (unlisted is allowed; private is not) and a public GitHub repository or one shared with the organizers. The repository must contain the source code and both prediction CSVs. Include every teammate's full name and each track's walk-forward MdAPE in the report. The deadline is November 8, 2026 at 11:59 pm EST.
