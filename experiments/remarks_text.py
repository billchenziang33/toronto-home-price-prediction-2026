"""Leakage-safe, interpretable experiment for remarks-derived text features.

Run from the repository root with:
    python -m experiments.remarks_text

This experiment changes no submission files and compares the pre-remarks model
with the remarks-enabled model under identical monthly folds.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import data
import run
from . import remarks_feature_builder as features

OUTPUT_DIR = Path(run.ROOT) / "outputs" / "remarks_text"


def _track_features(homes, track, include_remarks):
    make_features = features.list_aware_features if track == "list_aware" else features.off_market_features
    return make_features(homes, include_remarks=include_remarks)


def _predict(train, homes, track, include_remarks):
    X_train = _track_features(train, track, include_remarks)
    X_homes = _track_features(data.hide_answers(homes, track), track, include_remarks)
    categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns
    for column in categorical_columns:
        categories = X_train[column].dropna().unique().tolist()
        X_train[column] = pd.Categorical(X_train[column], categories=categories)
        X_homes[column] = pd.Categorical(X_homes[column], categories=categories)

    model = HistGradientBoostingRegressor(random_state=0, categorical_features="from_dtype")
    model.fit(X_train, np.log(train["sold_price"].to_numpy(dtype=float)))
    return np.exp(model.predict(X_homes))


def paired_mdape_interval(remarks_error, baseline_error, repeats=1000):
    """Bootstrap the reported statistic: MdAPE(remarks) - MdAPE(baseline).

    Resampling the per-home error differences and taking their median estimates a
    different quantity, so each resample must calculate both medians separately.
    Negative values favor remarks.
    """
    remarks_error = np.asarray(remarks_error, dtype=float)
    baseline_error = np.asarray(baseline_error, dtype=float)
    if len(remarks_error) != len(baseline_error):
        raise ValueError("paired error arrays must have equal length")
    if not len(remarks_error):
        return [None, None]
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(repeats):
        indices = rng.integers(0, len(remarks_error), size=len(remarks_error))
        samples.append(np.median(remarks_error[indices]) - np.median(baseline_error[indices]))
    return [round(float(value), 3) for value in np.percentile(samples, [2.5, 97.5])]


def score_pair(rows, repeats=1000):
    """Compare predictions on the exact same homes; negative deltas favor remarks."""
    rows = rows.sort_values("ml_num", kind="stable")
    baseline_error = 100 * (rows["baseline_price"] - rows["sold_price"]).abs() / rows["sold_price"]
    remarks_error = 100 * (rows["remarks_price"] - rows["sold_price"]).abs() / rows["sold_price"]
    signed_error = 100 * (rows["remarks_price"] - rows["sold_price"]) / rows["sold_price"]
    return {
        "n": int(len(rows)),
        "baseline_mdape": round(float(baseline_error.median()), 3),
        "remarks_mdape": round(float(remarks_error.median()), 3),
        "mdape_delta_pp_remarks_minus_baseline": round(float(remarks_error.median() - baseline_error.median()), 3),
        "baseline_mape": round(float(baseline_error.mean()), 3),
        "remarks_mape": round(float(remarks_error.mean()), 3),
        "mape_delta_pp_remarks_minus_baseline": round(float(remarks_error.mean() - baseline_error.mean()), 3),
        "remarks_median_signed_error": round(float(signed_error.median()), 3),
        "paired_mdape_delta_95ci": paired_mdape_interval(remarks_error.to_numpy(), baseline_error.to_numpy(), repeats),
    }


def _build_report(predictions, elapsed_seconds):
    report = {"elapsed_seconds": round(float(elapsed_seconds), 1), "tracks": {}}
    for track in run.TRACKS:
        rows = predictions[predictions["track"] == track]
        track_report = {"overall": score_pair(rows), "months": {}}
        for month, month_rows in rows.groupby("month", sort=True):
            track_report["months"][str(month)] = score_pair(month_rows)
        for name, months in {
            "selection_dec_to_feb": ["2025-12", "2026-01", "2026-02"],
            "confirmation_mar_to_may": ["2026-03", "2026-04", "2026-05"],
        }.items():
            track_report[name] = score_pair(rows[rows["month"].isin(months)])
        report["tracks"][track] = track_report
    return report


def run_experiment():
    """Run paired six-month folds for structured-only and remarks-enabled models."""
    started = time.perf_counter()
    listings = data.load("listings")
    rows = []

    for month in run.VALIDATION_MONTHS:
        train = listings[listings["date_sold"] < month.start_time]
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        validation_month = str(month)
        for track in run.TRACKS:
            baseline_prices = _predict(train, homes, track, include_remarks=False)
            remarks_prices = _predict(train, homes, track, include_remarks=True)
            rows.append(pd.DataFrame({
                "ml_num": homes["ml_num"].astype(str).to_numpy(),
                "track": track,
                "month": validation_month,
                "sold_price": homes["sold_price"].to_numpy(dtype=float),
                "baseline_price": np.round(baseline_prices),
                "remarks_price": np.round(remarks_prices),
            }))
        print(f"{month}: fitted lexical text features on {len(train):,} earlier sales", flush=True)

    predictions = pd.concat(rows, ignore_index=True)
    report = _build_report(predictions, time.perf_counter() - started)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT_DIR / 'predictions.csv'} and {OUTPUT_DIR / 'report.json'}")
    return report


if __name__ == "__main__":
    run_experiment()
