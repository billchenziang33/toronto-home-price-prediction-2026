# Toronto Fourcast — SSU Data Dash 2026

> **Current team status (October 19, 2026):** `model_candidate_v2/` remains the retained candidate. The latest fresh-fold and 20-method research did not meet the pre-registered promotion gates, so no v3 model or replacement submission CSVs were produced. Read [`TEAM_PROGRESS.md`](TEAM_PROGRESS.md) for the results and experiment map. `release_candidate/` is an archived v1 snapshot; its PDF, slides, and CSVs are not aligned to the retained v2 candidate.

Reproducible submission for the SSU Data Dash synthetic Toronto home-price challenge. The project predicts sale prices for two separate tracks: homes without a list price (`off-market`) and homes with a list price (`list-aware`).

## Data and rules

The challenge data contain synthetic sales from January 2021 through May 2026. The test sets cover June through September 2026. We do not use real listing or sale records, external data, or hidden test labels. In particular, the synthetic community names and coordinates should not be interpreted as a real Toronto map.

The `remarks` field is excluded: the organizers have not clarified whether its text contains additional price information. A question about whether listing simulation accounts for interest rates is also unanswered in the Discord case forum, so we make no claim about that data-generating mechanism.

## Method

We fit one `HistGradientBoostingRegressor` per track to log sale price. The list-aware model may use list price; the off-market model does not. Both models use temporal and property features, with native categorical splits for house style, municipality, and community. Validation is a six-month expanding walk-forward evaluation: each month is predicted using only earlier sales.

| Track | MdAPE | MAPE | Median signed error |
|---|---:|---:|---:|
| Off-market | 10.60% | 13.40% | -0.13% |
| List-aware | 4.36% | 5.57% | +0.65% |

These are validation results, not hidden-test scores. Home-level bootstrap intervals and experiment details are in `reports/validation_report_final.md` when generated locally.

## Reproduce

Use Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
python run.py validate
python run.py report
python run.py predict
python run.py check
```

The first run downloads the competition dataset from [Hugging Face](https://huggingface.co/datasets/Baliyang/SSU_data_dash). `predict` writes the two checked files in `submissions/`.

## Files

- `features.py`, `models.py`, `run.py`, `data.py`: data pipeline, features, models, and commands.
- `submissions/submission_off_market.csv` and `submissions/submission_list_aware.csv`: predictions for the two test tracks.
- `report/report.tex`, `report/project_plan.md`, `report/video_script.md`: report source, working plan, and narration draft.
- `output/slides/toronto_fourcast_challenge_slides_v4.pptx`: editable six-slide presentation.

The PDF and video links are intentionally not presented as final until the team records the video and replaces the draft link. The challenge requires a typed PDF report of at most four pages, a YouTube video no longer than five minutes, and this public repository. The deadline is Sunday, November 8, 2026 at 11:59 pm Eastern.
