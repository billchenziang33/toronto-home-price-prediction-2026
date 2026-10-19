# Submission candidate checklist

## Candidate artifacts

| Requirement | File | Status / evidence |
|---|---|---|
| Typed PDF, at most four pages | `report/SSU_DataDash_submission_candidate.pdf` | Built from `evidence/validation_metrics.json`; current PDF is two pages. Video line is explicitly pending recording. |
| Team names | PDF title block | Yan Cui, Jiayi Guan, Shengde Wang, Ziang Chen. |
| Video under five minutes | `report/video_script_en.md` | Six sections aligned with the six-slide deck; target narration is under five minutes. Recording and actual duration remain pending. |
| Six-slide presentation | `slides/toronto_fourcast_final.pptx` | Six-slide deck; native text/table and the validation chart are present. Build receipt is in `build/slide_finalizer/validation.json`. |
| Reproducible code description | `README.md` | Describes code files, two track boundaries, frozen commit, and reproduction commands. |
| Off-market and list-aware submissions | `submissions/submission_off_market.csv`; `submissions/submission_list_aware.csv` | Materialized from Git commit `7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8`; later verified with the starter `check` command. |
| Matching validation evidence | `evidence/validation_predictions_frozen.csv`; `evidence/validation_metrics.json` | Saved predictions reproduce 10.60% / 4.36% MdAPE and the report's confidence intervals. |
| Remarks experiment retained and corrected | `experiments/` | Paired baseline and targets match frozen predictions exactly. Corrected confirmation-period intervals include zero for both tracks. |
| Version and file audit | `SOURCE_COMMIT.txt`; `SHA256SUMS.json` | Exact public commit and package hashes. |

## Known corrections

- The prior remarks bootstrap used `median(remarks APE - baseline APE)` even though its point estimate was `median(remarks APE) - median(baseline APE)`. The corrected bootstrap recomputes both medians on each paired resample. The old report is retained with a warning; it no longer supplies the candidate's inference.
- The prior categorical-feature gain claim (3.80 and 0.19 percentage points with intervals) does not reproduce from the archived `feature_v1` and frozen prediction pair. The saved pair yields a different comparison, and the exact baseline specification attached to the old claim is missing. The claim is therefore omitted from the PDF, deck, and narration.
- The frozen report, chart, deck, and narration use the published no-remarks model and its saved validation predictions. The two submission CSVs are taken directly from the same published commit.

## Remaining team actions

1. Review the PDF, slides, script, and exact CSV files.
2. Record the English narration and keep the finished YouTube video below five minutes.
3. Replace the pending video line in the PDF, regenerate it, and verify the final video link.
4. Publish the approved candidate files to the linked GitHub repository.
5. Submit the official form and keep its confirmation.
