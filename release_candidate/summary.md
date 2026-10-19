# SSU Data Dash candidate summary

## Frozen version and current result

- Freeze the last public no-remarks model: GitHub commit `7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8`.
- Both submission CSVs were extracted from that exact Git commit. The model source files in this candidate are also Git objects from that commit.
- Six expanding walk-forward months (December 2025-May 2026), 10,069 homes per track. Recomputed results match the published report: off-market MdAPE 10.60% (95% home-bootstrap CI 10.38-10.85), MAPE 13.40% (13.17-13.65); list-aware MdAPE 4.36% (4.26-4.46), MAPE 5.57% (5.47-5.66).
- No feature or model changes were made for the candidate. No full validation or test prediction training was run to assemble this package; saved predictions were used.

## Statistical corrections

- `experiments/remarks_experiment_audit.json` verifies an exact row, target, and baseline-prediction match between the remarks experiment and the frozen validation predictions (20,138 rows; zero mismatches).
- The corrected estimand is `median(remarks APE) - median(frozen APE)`. Each bootstrap replicate resamples paired homes and recalculates both medians. For March-May, the off-market change is -0.101 percentage points (95% CI -0.335 to +0.090) and list-aware is -0.037 points (-0.111 to +0.049). Both intervals cross zero. Remarks remain an experiment and are excluded from the frozen release.
- The earlier categorical-feature gains and intervals cannot be reproduced from the archived comparison files. The candidate does not repeat those claims. `evidence/category_ablation_audit.json` records the mismatch and the recomputed saved-file comparison.
- Home bootstrap intervals assume independent homes and do not capture common month shocks.

## Implementation order used

1. Pin source code and both CSVs to the same published commit.
2. Recalculate headline metrics from the saved six-month predictions and verify them against the published report.
3. Correct the remarks bootstrap on the exact saved paired predictions; keep the result in a separate experiment archive.
4. Remove unverified categorical-gain statements from the report, slides, and narration; align all other figures with the frozen predictions.
5. Build the PDF, six-slide deck, English narration, code explanation, evidence, and hash manifest.
6. Check page/slide counts, PDF text and links, slide renderings, CSV structure, and source hashes.

The completed audit is recorded in `VERIFICATION_RECORD.md`. It confirms 22 passing tests, both submission CSV checks, the pinned PDF repository link, six slides, and matching SHA-256 hashes for all 53 other package files. Native PowerPoint rendering and the final recorded video remain pending.

## Remaining work outside this candidate stage

The team must record the video, insert and test its YouTube link, publish the approved candidate files, and submit the official form. The current PDF marks the video as pending; missing video is not treated as a reason to keep this preparation stage open.
