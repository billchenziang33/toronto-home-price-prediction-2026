"""Run the reproducible competition pipeline.

    python run.py validate   score the frozen candidate routes on the standard window
    python run.py report     write route, month, and segment error summaries
    python run.py predict    write one official submission CSV per track
    python run.py check      check both submission files against the test IDs

On Windows, type py instead of python if python is not found.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

import data
import models
from check_submission import check_all
from submission_models import model_manifest

ROOT = Path(__file__).parent
TRACKS = ["off_market", "list_aware"]
# This window was used in earlier experiments. It is retrospective standard
# validation, not an untouched holdout.
VALIDATION_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")
VALIDATION_SCOPE = "retrospective standard validation; not an independent holdout"
COMMUNITY_METRIC_MIN_ROWS = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text_file(path: Path) -> str:
    """Hash text artifacts with CRLF normalized so Git checkouts are portable."""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _cutoff_after_last_training_sale(listings: pd.DataFrame) -> pd.Timestamp:
    dates = pd.to_datetime(listings["date_sold"], errors="raise")
    if dates.empty or dates.isna().any():
        raise ValueError("training data must contain valid sale dates")
    last_month = dates.max().to_period("M")
    return (last_month + 1).start_time


def _predict_routes(model, homes: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, np.ndarray]:
    if hasattr(model, "predict_routes"):
        routes = model.predict_routes(homes, cutoff)
    else:
        route = getattr(model, "route", "model")
        routes = {route: model.predict(homes)}
    if not routes:
        raise RuntimeError("the model returned no prediction routes")
    checked = {}
    for name, values in routes.items():
        prediction = np.asarray(values, dtype=float)
        if (prediction.shape != (len(homes),) or not np.isfinite(prediction).all()
                or (prediction <= 0).any()):
            raise RuntimeError(f"{name} returned invalid prices for {len(homes)} homes")
        checked[str(name)] = prediction
    return checked


def _write_run_manifest(*, run_type: str, predictions_path: Path,
                        cutoffs: dict[str, str], row_count: int,
                        validation_months=None, run_id: str | None = None) -> Path:
    training_path = data.find_file("listings")
    manifest = {
        "format_version": 1,
        "run_type": run_type,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_scope": VALIDATION_SCOPE if run_type == "validation" else None,
        "validation_months": ([str(month) for month in validation_months]
                              if run_type == "validation" else None),
        "exclusive_training_cutoffs": cutoffs,
        "training_data": {
            "path": str(Path(training_path).resolve()),
            "sha256": sha256_file(Path(training_path)),
        },
        "model": model_manifest(),
        "prediction_artifacts": {
            predictions_path.name: {
                "path": str(predictions_path.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(row_count),
                "sha256": sha256_text_file(predictions_path),
            },
        },
        "reserved_month_labels_read": False,
        "hidden_test_labels_read": False,
    }
    destination = predictions_path.parent / f"{run_type}_run_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def validate(months=None, output_dir: Path | None = None):
    validation_months = pd.PeriodIndex(
        VALIDATION_MONTHS if months is None else months, freq="M")
    unsupported = set(validation_months) - set(VALIDATION_MONTHS)
    if unsupported:
        raise ValueError(f"validation months must be from the registered window: {sorted(map(str, unsupported))}")
    if len(validation_months) == 0 or len(set(validation_months)) != len(validation_months):
        raise ValueError("validation months must be a nonempty sequence without duplicates")
    listings = data.load("listings")
    run_id = f"validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    results = []
    cutoffs = {}
    for month in validation_months:
        cutoff = month.start_time
        train = listings.loc[listings["date_sold"] < cutoff].copy()
        homes_truth = listings.loc[listings["date_sold"].dt.to_period("M") == month].copy()
        if train.empty or homes_truth.empty:
            raise RuntimeError(f"validation fold {month} has empty training or evaluation rows")
        cutoffs[str(month)] = cutoff.isoformat()
        for track in TRACKS:
            model = models.make_model(track)
            if hasattr(model, "run_id"):
                model.run_id = run_id
            model = model.fit(train)
            homes = data.hide_answers(homes_truth, track).reset_index(drop=True)
            predicted_routes = _predict_routes(model, homes, cutoff)
            if homes["ml_num"].astype(str).duplicated().any():
                raise RuntimeError(f"duplicate listing IDs in {track}/{month} validation input")
            for route, predicted in predicted_routes.items():
                results.append(pd.DataFrame({
                    "ml_num": homes["ml_num"].astype(str).to_numpy(),
                    "track": track,
                    "model": route,
                    "month": str(month),
                    "house_type_name": homes["house_type_name"].to_numpy(),
                    "community": homes["community"].to_numpy() if "community" in homes else "__MISSING__",
                    "sold_price": homes_truth["sold_price"].to_numpy(float),
                    # Match the official nearest-dollar submission convention.
                    "predicted_price": np.round(predicted),
                }))
        print(f"{month}: trained strictly before {cutoff.date()}, predicted {len(homes_truth):,} homes per track")

    output_dir = ROOT / "outputs" if output_dir is None else Path(output_dir)
    path = output_dir / "validation_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(results, ignore_index=True)
    predictions.to_csv(path, index=False, lineterminator="\n")
    manifest = _write_run_manifest(run_type="validation", predictions_path=path,
                                   cutoffs=cutoffs, row_count=len(predictions),
                                   validation_months=validation_months, run_id=run_id)
    print(f"Wrote {path.relative_to(ROOT)} ({len(predictions):,} route-home rows)")
    print(f"Wrote {manifest.relative_to(ROOT)}; scope: {VALIDATION_SCOPE}")


def bootstrap_ci(values, statistic, repeats=1000):
    """Home-level bootstrap 95% interval; this does not resample whole months."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(0)
    results = [statistic(rng.choice(values, size=len(values), replace=True))
               for _ in range(repeats)]
    return [float(x) for x in np.percentile(results, [2.5, 97.5])]


def _paired_mdape_delta_ci(baseline_ape, candidate_ape, *, repeats=1000, seed=0):
    """Bootstrap the paired difference in route MdAPE, in percentage points."""
    baseline_ape = np.asarray(baseline_ape, dtype=float)
    candidate_ape = np.asarray(candidate_ape, dtype=float)
    if (baseline_ape.ndim != 1 or candidate_ape.shape != baseline_ape.shape
            or baseline_ape.size == 0 or not np.isfinite(baseline_ape).all()
            or not np.isfinite(candidate_ape).all()):
        raise ValueError("paired route APE arrays must be aligned, nonempty, and finite")
    rng = np.random.default_rng(seed)
    results = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(baseline_ape), size=len(baseline_ape))
        results[index] = np.median(candidate_ape[sample]) - np.median(baseline_ape[sample])
    return [float(x) for x in np.percentile(results, [2.5, 97.5])]


def metrics(predictions):
    """Full-precision scores for one track and route."""
    actual = predictions["sold_price"].to_numpy(dtype=float)
    predicted = predictions["predicted_price"].to_numpy(dtype=float)
    signed = 100.0 * (predicted - actual) / actual
    absolute = np.abs(signed)
    return {
        "n": int(len(predictions)),
        "mape_percent": float(np.mean(absolute)),
        "mape_95ci": bootstrap_ci(absolute, np.mean),
        "mdape_percent": float(np.median(absolute)),
        "mdape_95ci": bootstrap_ci(absolute, np.median),
        "p90_ape_percent": float(np.quantile(absolute, 0.90)),
        "median_signed_error_percent": float(np.median(signed)),
        "mean_signed_error_percent": float(np.mean(signed)),
    }


def _paired_comparison(predictions: pd.DataFrame, track: str,
                       baseline: str, candidate: str) -> dict:
    keys = ["month", "ml_num", "sold_price"]
    base = predictions.loc[predictions.track.eq(track) & predictions.model.eq(baseline), keys + ["predicted_price"]]
    cand = predictions.loc[predictions.track.eq(track) & predictions.model.eq(candidate), keys + ["predicted_price"]]
    paired = base.merge(cand, on=keys, how="outer", validate="one_to_one",
                        suffixes=("_baseline", "_candidate"), indicator=True)
    if paired.empty or not paired._merge.eq("both").all():
        raise RuntimeError(f"{track} {baseline}/{candidate} rows do not align one-to-one")
    actual = paired.sold_price.to_numpy(float)
    base_ape = 100.0 * np.abs(paired.predicted_price_baseline.to_numpy(float) - actual) / actual
    cand_ape = 100.0 * np.abs(paired.predicted_price_candidate.to_numpy(float) - actual) / actual
    return {
        "track": track, "baseline": baseline, "candidate": candidate,
        "n": int(len(paired)),
        "mdape_delta_percentage_points_candidate_minus_baseline": float(np.median(cand_ape) - np.median(base_ape)),
        "mdape_delta_home_bootstrap_95ci_percentage_points": _paired_mdape_delta_ci(base_ape, cand_ape),
        "mape_delta_percentage_points_candidate_minus_baseline": float(np.mean(cand_ape) - np.mean(base_ape)),
        "months": sorted(paired.month.astype(str).unique().tolist()),
        "uncertainty_scope": "home-level bootstrap; shared month shocks and model selection are not represented",
    }


def price_decile_labels(prices):
    """Assign common price deciles; duplicate cut points are collapsed safely."""
    labels = pd.qcut(prices, 10, labels=False, duplicates="drop")
    return labels + 1


def error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def append_group(track, model, dimension, group_name, frame):
        n = int(len(frame))
        base = {"track": track, "model": model, "dimension": dimension,
                "group": str(group_name), "n": n,
                "ranking_eligible": not (dimension == "community" and n < COMMUNITY_METRIC_MIN_ROWS)}
        if not base["ranking_eligible"]:
            base.update({"mdape_percent": np.nan, "mape_percent": np.nan,
                         "p90_ape_percent": np.nan, "median_signed_error_percent": np.nan})
        else:
            actual = frame.sold_price.to_numpy(float)
            signed = 100.0 * (frame.predicted_price.to_numpy(float) - actual) / actual
            ape = np.abs(signed)
            base.update({"mdape_percent": float(np.median(ape)),
                         "mape_percent": float(np.mean(ape)),
                         "p90_ape_percent": float(np.quantile(ape, 0.90)),
                         "median_signed_error_percent": float(np.median(signed))})
        rows.append(base)

    for (track, model), frame in predictions.groupby(["track", "model"], sort=True):
        append_group(track, model, "all", "all", frame)
        for column, dimension in (("month", "month"), ("house_type_name", "house_type"),
                                  ("community", "community")):
            values = frame[column].astype("string").fillna("__MISSING__")
            for label, index in values.groupby(values, sort=True, dropna=False).groups.items():
                append_group(track, model, dimension, label, frame.loc[index])
        unique_actual = frame.drop_duplicates("ml_num").set_index("ml_num")["sold_price"]
        deciles = price_decile_labels(unique_actual)
        decile_by_id = deciles.to_dict()
        price_group = frame.ml_num.astype(str).map(decile_by_id)
        for label, index in price_group.groupby(price_group, sort=True, dropna=False).groups.items():
            if pd.isna(label):
                continue
            append_group(track, model, "sold_price_decile", int(label), frame.loc[index])
    return pd.DataFrame(rows)


def report():
    path = ROOT / "outputs" / "validation_predictions.csv"
    if not path.exists():
        sys.exit(f"{path} does not exist yet: run `python run.py validate` first")
    predictions = pd.read_csv(path, dtype={"ml_num": str})
    if "model" not in predictions:
        raise RuntimeError("validation predictions do not identify model routes; rerun `python run.py validate`")
    scores = {
        track: {
            model: metrics(group)
            for model, group in frame.groupby("model", sort=True)
        }
        for track, frame in predictions.groupby("track", sort=True)
    }
    comparisons = [
        _paired_comparison(predictions, "off_market", "R22", candidate)
        for candidate in ("R52", "R151")
    ] + [
        _paired_comparison(predictions, "off_market", "R52", "R151"),
        _paired_comparison(predictions, "list_aware", "R35", "R97"),
    ]
    error_rows = error_analysis(predictions)

    (ROOT / "reports").mkdir(exist_ok=True)
    error_path = ROOT / "outputs" / "error_analysis.csv"
    error_rows.to_csv(error_path, index=False, lineterminator="\n")
    summary = {
        "validation_scope": VALIDATION_SCOPE,
        "validation_months": [str(month) for month in VALIDATION_MONTHS],
        "routes": scores,
        "paired_comparisons": comparisons,
        "community_metric_minimum_rows": COMMUNITY_METRIC_MIN_ROWS,
        "reserved_month_labels_read": False,
        "hidden_test_labels_read": False,
    }
    (ROOT / "reports" / "validation_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Validation report", "",
        f"**Scope:** {VALIDATION_SCOPE}.",
        f"Window: {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}. Each month is predicted using sales strictly before its first day.",
        "Scores use nearest-dollar predictions, as in the final CSV. All JSON metrics retain full precision.",
        "MdAPE is the primary competition metric; MAPE is the tiebreak metric. Brackets show a home-level bootstrap 95% interval.", "",
        "| Track | Route | Homes | MdAPE % | MAPE % | P90 APE % | Median signed % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for track, route_scores in scores.items():
        for model, score in route_scores.items():
            lines.append(
                f"| {track} | {model} | {score['n']:,} | {score['mdape_percent']:.8f} "
                f"({score['mdape_95ci'][0]:.8f}–{score['mdape_95ci'][1]:.8f}) | "
                f"{score['mape_percent']:.8f} ({score['mape_95ci'][0]:.8f}–{score['mape_95ci'][1]:.8f}) | "
                f"{score['p90_ape_percent']:.8f} | {score['median_signed_error_percent']:+.8f} |"
            )
    lines.extend(["", "## Paired route comparisons", "",
                  "A negative MdAPE delta favors the candidate. These home-level intervals do not capture common market movement within a month or repeated model selection.",
                  "", "| Track | Baseline | Candidate | Homes | MdAPE delta pp (95% interval) | MAPE delta pp |",
                  "|---|---|---|---:|---:|---:|"])
    for comparison in comparisons:
        lo, hi = comparison["mdape_delta_home_bootstrap_95ci_percentage_points"]
        lines.append(
            f"| {comparison['track']} | {comparison['baseline']} | {comparison['candidate']} | "
            f"{comparison['n']:,} | {comparison['mdape_delta_percentage_points_candidate_minus_baseline']:+.8f} "
            f"({lo:+.8f} to {hi:+.8f}) | "
            f"{comparison['mape_delta_percentage_points_candidate_minus_baseline']:+.8f} |"
        )
    lines.extend([
        "", "## Error breakdown", "",
        f"`outputs/error_analysis.csv` includes overall, month, property type, sold-price decile, and community results. Community metrics are suppressed below n={COMMUNITY_METRIC_MIN_ROWS}; those groups retain counts only.",
        "", "## Interpretation limits", "",
        "The six-month window has been used in prior work, so it is retrospective validation rather than an independent holdout. Home-level bootstrap intervals treat homes as independent and omit shared monthly shocks.",
        "R97's earlier eight-month development result and promotion-gate outcome are reported separately in the technical audit; this report is a new fixed-window pipeline run.",
        "", "![Error by price range](../figures/error_by_price_range.png)", "",
    ])
    (ROOT / "reports" / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_latex_table(scores)
    error_by_price_range(predictions)
    print("\n".join(lines))
    print("Wrote reports/validation_report.md, reports/validation_report.json, reports/results_table.tex, "
          "outputs/error_analysis.csv and figures/error_by_price_range.png")


def write_latex_table(scores):
    """Write a LaTeX table with one row per validation route."""
    rows = []
    for track, route_scores in scores.items():
        for model, score in route_scores.items():
            name = f"{track.replace('_', '-')} ({model})"
            rows.append(
                f"{name} & {score['n']:,} & {score['mdape_percent']:.4f} "
                f"({score['mdape_95ci'][0]:.4f}--{score['mdape_95ci'][1]:.4f}) "
                f"& {score['mape_percent']:.4f} "
                f"({score['mape_95ci'][0]:.4f}--{score['mape_95ci'][1]:.4f}) "
                f"& {score['median_signed_error_percent']:+.4f} \\\\"
            )
    table = "\n".join([
        "% Written by `python run.py report`. Do not edit by hand: rerun the command instead.",
        "\\begin{tabular}{lrccc}", "\\hline",
        r"Track (route) & Homes & MdAPE \% (95\% CI) & MAPE \% (95\% CI) & Median signed error \% " + r"\\", "\\hline",
        *rows, "\\hline", "\\end{tabular}",
    ])
    (ROOT / "reports" / "results_table.tex").write_text(table + "\n", encoding="utf-8")


def error_by_price_range(predictions):
    """Plot route-level median APE by sold-price decile."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, track in zip(axes, TRACKS):
        track_rows = predictions.loc[predictions.track.eq(track)].copy()
        for model, rows in track_rows.groupby("model", sort=True):
            unique_actual = rows.drop_duplicates("ml_num").set_index("ml_num")["sold_price"]
            deciles = price_decile_labels(unique_actual)
            rows["price_decile"] = rows.ml_num.astype(str).map(deciles.to_dict())
            error = 100.0 * np.abs(rows.predicted_price - rows.sold_price) / rows.sold_price
            by_decile = error.groupby(rows.price_decile, dropna=True)
            middle = by_decile.median()
            axis.plot(middle.index, middle.values, marker="o", linewidth=1.8, label=model)
            axis.fill_between(middle.index, by_decile.quantile(0.25), by_decile.quantile(0.75), alpha=0.10)
        axis.set_title(track.replace("_", " "))
        axis.set_xlabel("Sold-price decile (1 = lowest)")
        axis.set_xticks(range(1, 11))
        axis.grid(axis="y", color="#e5e5e5")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Absolute percentage error (median; band = middle 50%)")
    fig.suptitle(f"Retrospective validation error by price range, {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "error_by_price_range.png", dpi=150)
    plt.close(fig)


def predict():
    listings = data.load("listings")
    cutoff = _cutoff_after_last_training_sale(listings)
    run_id = f"submission_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    if (listings["date_sold"] >= cutoff).any():
        raise RuntimeError("training data include a sale at or after the submission cutoff")
    submissions = ROOT / "submissions"
    submissions.mkdir(exist_ok=True)
    artifact_rows = {}
    test_inputs = {}
    for track in TRACKS:
        test_path = Path(data.find_file(track))
        raw_homes = data.load(track)
        homes = data.hide_answers(raw_homes, track).reset_index(drop=True)
        model = models.make_model(track)
        if hasattr(model, "run_id"):
            model.run_id = run_id
        model = model.fit(listings)
        routes = _predict_routes(model, homes, cutoff)
        selected = getattr(model, "route", None)
        if selected not in routes:
            raise RuntimeError(f"selected route {selected!r} is missing for {track}")
        path = submissions / f"submission_{track}.csv"
        pd.DataFrame({
            "ml_num": homes["ml_num"].astype(str),
            "predicted_price": np.round(routes[selected]),
        }).to_csv(path, index=False, lineterminator="\n")
        artifact_rows[path.name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                    "rows": len(homes), "sha256": sha256_text_file(path),
                                    "model_route": selected}
        test_inputs[track] = {
            "path": str(test_path.resolve()), "sha256": sha256_file(test_path),
            "rows": len(raw_homes), "columns": list(raw_homes.columns),
        }
        print(f"Wrote {path.relative_to(ROOT)} ({len(homes):,} homes), route {selected}")
    manifest_path = submissions / "submission_run_manifest.json"
    manifest = {
        "format_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "exclusive_training_cutoff": cutoff.isoformat(),
        "training_data": {
            "path": str(Path(data.find_file("listings")).resolve()),
            "sha256": sha256_file(Path(data.find_file("listings"))),
            "rows": len(listings),
        },
        "test_inputs": test_inputs,
        "model": model_manifest(),
        "submissions": artifact_rows,
        "reserved_month_labels_read": False,
        "hidden_test_labels_read": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")


def check():
    return check_all(ROOT / "submissions")


COMMANDS = {"validate": validate, "report": report, "predict": predict, "check": check}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    result = COMMANDS[sys.argv[1]]()
    sys.exit(1 if result is False else 0)
