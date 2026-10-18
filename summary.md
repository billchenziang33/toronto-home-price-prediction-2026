# Project summary

## Current status (2026-10-18)

The official starter repository is checked out in this folder at upstream commit `6c5831e`. The working model uses separate off-market and list-aware feature sets, log-price histogram gradient boosting, and native categorical splits for `house_style`, `municipality`, and `community`.

## Best verified validation result so far

`python run.py validate` uses six expanding walk-forward months, December 2025 through May 2026, with 10,069 validation sales per track. The current model scores:

| Track | MdAPE (95% home-bootstrap CI) | MAPE (95% home-bootstrap CI) | Median signed error |
|---|---:|---:|---:|
| Off-market | 10.60% (10.38–10.85) | 13.40% (13.17–13.65) | -0.13% |
| List-aware | 4.36% (4.26–4.46) | 5.57% (5.47–5.66) | +0.65% |

Compared with the feature-only model on identical homes, the three categories reduced MdAPE by 3.80 points off-market (paired 95% bootstrap interval: 3.52–4.12) and 0.19 points list-aware (0.14–0.24). All six months improved on both tracks. The garage/lot-area experiment had small changes with paired intervals crossing zero and was removed. A leakage-safe community-comparable feature also had paired MdAPE intervals crossing zero and inconsistent monthly gains, while nearly doubling validation time, so it was removed. The first one-hot encoding attempt was too slow; native categorical support produced the successful run.

The official starter baselines were 20.8% MdAPE off-market and 7.3% list-aware. The current results are validation only; hidden-set performance is unknown. The home-level bootstrap intervals do not model shared month-to-month market shocks.

## Verified artifacts and commands

- Main source changes: `features.py`, `models.py`, `run.py`, `tests/test_starter.py`.
- Final validation predictions/reports are stored under ignored `outputs/`, `reports/`, and `figures/` folders. The latest copies use the `final` suffix.
- Submission predictions: `submissions/submission_off_market.csv` (3,974 rows) and `submissions/submission_list_aware.csv` (4,064 rows). `python run.py check` reports both `OK`; both files have unique IDs, no missing values, and only positive prices.
- Verification: `OMP_NUM_THREADS=1 python -m pytest -q --basetemp=.pytest_cache/basetemp` passed 22 tests. The final six-month `validate`, `report`, `predict`, and `check` commands completed. `OMP_NUM_THREADS=1` was used for local reproducibility and runtime control; it does not change the validation splits or metrics.
- Data cache: `data/.hf_cache`; set `HF_HOME` to that folder in PowerShell when running locally in this environment.

## Report and video draft

- `report/project_plan.md` records the challenge scope, rubric priorities, verified work, research boundaries, dated execution schedule through November 20, final deliverable layout, and remaining blockers.
- `report/report.tex` contains the filled official report template, but LaTeX compilation is unavailable because this machine has no Tectonic or TeX Live installation.
- `report/create_pdf_report.py` generates the two-page PDF draft at `output/pdf/SSU_DataDash_submission_draft.pdf` with ReportLab from the bundled Codex Python runtime. Member names are now filled in (Yan Cui, Jiayi Guan, Shengde Wang, Ziang Chen); team name, YouTube URL, and repository URL remain placeholders. Do not submit until those fields are replaced. Rebuilt after adding names; `pdfinfo` confirms two Letter pages and no encryption, and the first page was visually checked.
- `report/video_script.md` contains the six-slide, under-five-minute English storyboard and draft narration. All numeric claims in it come from the challenge's own walk-forward validation or reproduced starter validation.

## Submission requirements and limits

The official report must be at most four pages, link to a YouTube video no longer than five minutes and a GitHub repository containing the code and both prediction CSVs, and list every team member's full name. The form requires one team member to submit. The deadline is November 8 at 11:59 pm EST; the in-person final is November 20.

No outside data is used in the model or report. Contest rules restrict outside data to periods through May 2026, ban real listings/sales, and require every number in the video to come from our validation. TRREB's September 2026 market report was reviewed only as background and falls after the allowed data cutoff, so do not use it in the submission or video. Challenge data are synthetic and its communities/coordinates do not match actual Toronto maps.

## Research and presentation guidance

- A Toronto real-estate study describes spatial neighborhood relationships and temporal price changes as useful signals, supporting our test of location categories and time features: https://arxiv.org/abs/2008.05880
- A public Zillow Prize solution extracts region aggregates and interaction features, compares LightGBM/CatBoost, and reports offline cross-validation to reduce leaderboard overfit. It supports the experiment design, but its real Zillow data and error metric are not comparable to this synthetic task: https://github.com/junjiedong/Zillow-Kaggle
- scikit-learn documents native categorical support in `HistGradientBoostingRegressor`; it avoids a large one-hot matrix and is the implementation used here: https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- TRREB's September 2026 report says the GTA average sale price was $1,006,409, down 5.1% year over year. This is post-cutoff contextual research only, not an admissible modeling or video input: https://public.trreb.ca/market-data/market-watch/
- The local `Presentations` skill is available for the optional five-minute slide deck/video. Short-talk guidance recommends one main point per slide and a single clear story: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005373

The logged-in Discord desktop window was inspected read-only on 2026-10-18. `#announcements` confirms the Friday, November 20, 2:30–5:00 pm finale; the top five teams present, and every team that submits a video is invited to attend. `#links-and-documents` contains the case handout/rubric and the spatial-modelling workshop slides. Its follow-up announces Workshop 2 for November 2, 1:00–2:00 pm at 700 University Ave, room 9199. `#rules` prohibits academic-integrity violations, spam, off-topic posts, and sharing other people's personal information without permission. The case forum has unanswered questions about whether simulated listings incorporate interest rates and whether `remarks` contains price signal beyond the structured fields. We therefore exclude `remarks` and describe the interest-rate mechanism as unknown. The case-question forum also says to ask one specific question per post and not post solutions, models, or findings; no messages were sent.

## Next steps

1. The team name is Toronto Fourcast and a public GitHub repository has been created at https://github.com/billchenziang33/toronto-home-price-prediction-2026. Populate and verify its contents. Member names are recorded in the report; their email addresses were not included in the public report draft.
2. Record and check the storyboard against the five-minute limit, then update the PDF links and re-render the final PDF.
3. Prepare the public or organizer-shared GitHub repository with the code and both checked prediction CSVs; verify access in a private browser session.
4. Before the November 8 deadline, have the team proofread the PDF/video and submit the Google Form. Do not claim completion until the form confirms it.
