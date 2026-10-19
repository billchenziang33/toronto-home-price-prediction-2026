"""Assemble a reviewable release candidate from the last public model.

This reads saved predictions and Git blobs only. It does not fit a model or
access hidden test labels.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release_candidate"
COMMIT = "7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8"
REPEATS = 1000


def git_blob(relative_path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{COMMIT}:{relative_path}"]
    )


def copy_published_files() -> None:
    files = [
        "data.py", "features.py", "models.py", "run.py", "check_submission.py",
        "requirements.txt", "LICENSE", "pytest.ini", "tests/test_starter.py",
        "submissions/submission_off_market.csv",
        "submissions/submission_list_aware.csv",
    ]
    for relative in files:
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git_blob(relative))
    (OUT / "SOURCE_COMMIT.txt").write_text(
        f"Public frozen model and submission CSVs: {COMMIT}\n"
        "https://github.com/billchenziang33/toronto-home-price-prediction-2026/commit/"
        f"{COMMIT}\n",
        encoding="utf-8",
    )


def home_bootstrap(values: np.ndarray, statistic, repeats: int = REPEATS) -> list[float]:
    rng = np.random.default_rng(0)
    samples = [statistic(rng.choice(values, size=len(values))) for _ in range(repeats)]
    return [round(float(x), 2) for x in np.percentile(samples, [2.5, 97.5])]


def metric_rows(frame: pd.DataFrame) -> dict:
    result = {}
    for track, rows in frame.groupby("track", sort=True):
        signed = 100 * (rows.predicted_price - rows.sold_price) / rows.sold_price
        absolute = signed.abs().to_numpy()
        result[track] = {
            "n": int(len(rows)),
            "mape": round(float(absolute.mean()), 2),
            "mape_95ci": home_bootstrap(absolute, np.mean),
            "mdape": round(float(np.median(absolute)), 2),
            "mdape_95ci": home_bootstrap(absolute, np.median),
            "median_signed_error": round(float(np.median(signed)), 2),
        }
    return result


def paired_median_delta(baseline: np.ndarray, candidate: np.ndarray,
                        repeats: int = REPEATS) -> dict:
    if len(baseline) != len(candidate):
        raise ValueError("Paired arrays must have equal length")
    point = float(np.median(candidate) - np.median(baseline))
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(repeats):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        deltas.append(float(np.median(candidate[indices]) - np.median(baseline[indices])))
    interval = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_pp_candidate_minus_baseline": round(point, 3),
        "paired_home_bootstrap_95ci": [round(float(x), 3) for x in interval],
    }


def audit_category_claim() -> dict:
    baseline = pd.read_csv(ROOT / "outputs/validation_predictions_feature_v1.csv")
    frozen = pd.read_csv(ROOT / "outputs/validation_predictions_final.csv")
    keys = ["ml_num", "track", "month"]
    merged = baseline.merge(frozen, on=keys, suffixes=("_feature_v1", "_frozen"),
                            validate="one_to_one")
    if len(merged) != len(frozen):
        raise ValueError("Feature-v1 and frozen predictions do not pair one-to-one")
    if not np.allclose(merged.sold_price_feature_v1, merged.sold_price_frozen):
        raise ValueError("Targets differ across the paired validation files")
    result = {}
    for track, rows in merged.groupby("track", sort=True):
        old_ape = (100 * (rows.predicted_price_feature_v1 - rows.sold_price_feature_v1)
                   .abs() / rows.sold_price_feature_v1).to_numpy()
        new_ape = (100 * (rows.predicted_price_frozen - rows.sold_price_frozen)
                   .abs() / rows.sold_price_frozen).to_numpy()
        result[track] = {
            "n": int(len(rows)),
            "feature_v1_mdape": round(float(np.median(old_ape)), 3),
            "frozen_mdape": round(float(np.median(new_ape)), 3),
            **paired_median_delta(old_ape, new_ape, repeats=2000),
        }
    return {
        "archived_pair": [
            "outputs/validation_predictions_feature_v1.csv",
            "outputs/validation_predictions_final.csv",
        ],
        "reported_claim": {
            "off_market_gain_pp": 3.80,
            "off_market_interval": [3.52, 4.12],
            "list_aware_gain_pp": 0.19,
            "list_aware_interval": [0.14, 0.24],
        },
        "recomputed_from_archived_pair": result,
        "disposition": (
            "Do not use the prior feature-ablation claim in submission materials. "
            "The published point estimates and intervals are not reproduced by the "
            "archived feature_v1/frozen prediction pair, and the exact comparison "
            "baseline specification is not preserved with that claim."
        ),
        "method": (
            "For each bootstrap sample, calculate median(candidate APE) minus "
            "median(baseline APE) on the same resampled home indices."
        ),
        "bootstrap_repeats": 2000,
        "seed": 0,
    }


def corrected_remarks_audit() -> dict:
    frozen = pd.read_csv(ROOT / "outputs/validation_predictions_final.csv")
    experiment = pd.read_csv(ROOT / "outputs/remarks_text/predictions.csv")
    keys = ["ml_num", "track", "month"]
    base = frozen.set_index(keys).sort_index()
    rows = experiment.set_index(keys).sort_index()
    if not base.index.equals(rows.index):
        raise ValueError("Remarks experiment rows do not match frozen validation rows")
    if not np.array_equal(base.sold_price.to_numpy(), rows.sold_price.to_numpy()):
        raise ValueError("Remarks and frozen validation targets differ")
    if not np.array_equal(base.predicted_price.to_numpy(), rows.baseline_price.to_numpy()):
        raise ValueError("Remarks experiment baseline does not match the frozen model")

    tracks = {}
    for track in ["off_market", "list_aware"]:
        selected = rows[rows.index.get_level_values("track") == track]
        segments = {
            "overall": selected,
            "selection_dec_to_feb": selected[
                selected.index.get_level_values("month").isin(
                    ["2025-12", "2026-01", "2026-02"]
                )
            ],
            "confirmation_mar_to_may": selected[
                selected.index.get_level_values("month").isin(
                    ["2026-03", "2026-04", "2026-05"]
                )
            ],
        }
        track_result = {}
        for name, segment in segments.items():
            base_ape = (100 * (segment.baseline_price - segment.sold_price).abs()
                        / segment.sold_price).to_numpy()
            remarks_ape = (100 * (segment.remarks_price - segment.sold_price).abs()
                           / segment.sold_price).to_numpy()
            track_result[name] = {
                "n": int(len(segment)),
                "baseline_mdape": round(float(np.median(base_ape)), 3),
                "remarks_mdape": round(float(np.median(remarks_ape)), 3),
                **paired_median_delta(base_ape, remarks_ape),
            }
        tracks[track] = track_result
    return {
        "frozen_commit": COMMIT,
        "baseline_prediction_match": "exact; 0 price mismatches across 20,138 rows",
        "target_match": "exact; 0 target mismatches across 20,138 rows",
        "estimand": "median(remarks APE) - median(frozen APE), in percentage points",
        "bootstrap": (
            "Paired resampling of home indices; each resample calculates the two "
            "medians separately. Homes are treated as independent, so shared "
            "month-level shocks are not covered."
        ),
        "bootstrap_repeats": REPEATS,
        "seed": 0,
        "tracks": tracks,
        "decision": (
            "The confirmation-period 95% intervals include zero for both tracks. "
            "Keep the public no-remarks model frozen; retain remarks as an experiment."
        ),
        "legacy_issue": (
            "The earlier helper bootstrapped median(per-home APE difference), while "
            "its point estimate was the difference between two MdAPEs. These are "
            "different estimands; earlier intervals are superseded."
        ),
    }


def write_chart(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["ape"] = 100 * (frame.predicted_price - frame.sold_price).abs() / frame.sold_price
    frame["decile"] = pd.qcut(frame.sold_price, 10, labels=False, duplicates="drop") + 1
    fig, ax = plt.subplots(figsize=(8.2, 4.25))
    colors = {"off_market": "#147D80", "list_aware": "#D35E3D"}
    labels = {"off_market": "Off-market", "list_aware": "List-aware"}
    for track in ["off_market", "list_aware"]:
        rows = frame[frame.track == track]
        median = rows.groupby("decile", observed=True).ape.median()
        q25 = rows.groupby("decile", observed=True).ape.quantile(0.25)
        q75 = rows.groupby("decile", observed=True).ape.quantile(0.75)
        ax.plot(median.index, median.values, marker="o", linewidth=2.4,
                color=colors[track], label=labels[track])
        ax.fill_between(median.index, q25, q75, color=colors[track], alpha=0.14)
    ax.set_xlabel("Shared sold-price decile (1 = lowest)")
    ax.set_ylabel("Absolute percentage error")
    ax.set_title("Six-month walk-forward validation")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D7DFE4", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "evidence/error_by_price_range.png", dpi=200)
    plt.close(fig)


def main() -> None:
    if not (ROOT / ".git").exists():
        raise SystemExit("Run this script from the competition worktree")
    (OUT / "evidence").mkdir(parents=True, exist_ok=True)
    (OUT / "experiments").mkdir(parents=True, exist_ok=True)
    copy_published_files()

    frozen = pd.read_csv(ROOT / "outputs/validation_predictions_final.csv")
    if set(frozen.track.unique()) != {"off_market", "list_aware"}:
        raise ValueError("Unexpected validation tracks")
    if len(frozen) != 20138 or frozen.groupby("track").size().to_dict() != {
        "list_aware": 10069, "off_market": 10069
    }:
        raise ValueError("Validation predictions do not match the published report scope")
    if sorted(frozen.month.unique()) != [
        "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"
    ]:
        raise ValueError("Validation months do not match the published report")

    metrics = metric_rows(frozen)
    expected = {
        "off_market": {"mdape": 10.60, "mape": 13.40},
        "list_aware": {"mdape": 4.36, "mape": 5.57},
    }
    for track, values in expected.items():
        for name, value in values.items():
            if metrics[track][name] != value:
                raise ValueError(f"{track} {name} does not match the published report")
    (OUT / "evidence/validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(ROOT / "outputs/validation_predictions_final.csv",
                 OUT / "evidence/validation_predictions_frozen.csv")
    shutil.copy2(ROOT / "reports/validation_report_final.md",
                 OUT / "evidence/validation_report_published.md")
    shutil.copy2(ROOT / "outputs/validation_predictions_starter_baseline.csv",
                 OUT / "evidence/validation_predictions_starter_baseline.csv")
    shutil.copy2(ROOT / "outputs/validation_predictions_feature_v1.csv",
                 OUT / "evidence/validation_predictions_feature_v1.csv")
    remarks_predictions = ROOT / "outputs/remarks_text/predictions.csv"
    shutil.copy2(remarks_predictions,
                 OUT / "experiments/remarks_text_predictions.csv")
    shutil.copy2(ROOT / "outputs/remarks_text/report.json",
                 OUT / "experiments/remarks_text_report_legacy.json")
    shutil.copy2(ROOT / "experiments/remarks_feature_builder.py",
                 OUT / "experiments/remarks_feature_builder.py")
    shutil.copy2(ROOT / "experiments/remarks_text.py",
                 OUT / "experiments/remarks_text.py")
    (OUT / "evidence/category_ablation_audit.json").write_text(
        json.dumps(audit_category_claim(), indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "experiments/remarks_experiment_audit.json").write_text(
        json.dumps(corrected_remarks_audit(), indent=2) + "\n", encoding="utf-8"
    )
    write_chart(frozen)
    print(f"Prepared {OUT}")
    print(json.dumps(metrics, indent=2))
    print("Frozen predictions: 20,138 rows; remarks baseline exact match verified.")


if __name__ == "__main__":
    main()
