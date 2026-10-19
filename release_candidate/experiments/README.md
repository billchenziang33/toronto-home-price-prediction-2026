# Remarks experiment archive

`remarks_text_predictions.csv` holds the saved paired predictions. Its baseline prices and sold-price targets match `evidence/validation_predictions_frozen.csv` exactly across all 20,138 validation rows.

`remarks_text.py` contains the original experiment runner and the corrected bootstrap helper. Its feature definitions live in `remarks_feature_builder.py`, separate from the frozen production `features.py`. Running the experiment runner again would fit models; this candidate was audited from the saved predictions and did not rerun training.

`remarks_text_report_legacy.json` is preserved as generated history. Its previous `paired_mdape_delta_95ci` values used the median of per-home APE differences, which does not estimate the reported difference between two MdAPEs. Do not use those old intervals. The corrected same-home bootstrap is in `remarks_experiment_audit.json`; it recomputes each MdAPE on every resample. Confirmation-period intervals include zero for both tracks, so remarks remain an experiment and are not part of the frozen submission model.
