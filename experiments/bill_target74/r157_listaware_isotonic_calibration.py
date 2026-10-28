"""R157: strict-past isotonic calibration of the list-aware R35 sale/list ratio."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from experiments.bill_target74 import runner as campaign


CARD = "R157_listaware_r35_isotonic_ratio_calibration"
TRACK = "list_aware"
R35_CARD = "R35_listaware_detached_r22_text_expert"
R44_CARD = "R44_past_oof_r35_blend"
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
RESERVED_MONTHS = set(campaign.PROTOCOL["reserved_months"])
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261157
MIN_HISTORY_ROWS = 1000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        campaign.PACKAGE / "r157_independent_audit.py",
        campaign.PACKAGE / "r156_listaware_price_band_experts.py",
        campaign.PACKAGE / "r44_past_oof_r35_shrinkage.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "runner.py",
    ]


def _code_hashes() -> dict[str, str]:
    return {_relative(path): _sha256(path) for path in _source_files()}


def _verified_parent() -> dict:
    from experiments.bill_target74 import r156_listaware_price_band_experts as r156

    return r156._verified_r35_parent()


def _parent_artifact_hashes(parent: dict) -> dict[str, str]:
    from experiments.bill_target74 import r156_listaware_price_band_experts as r156

    return r156._parent_artifact_hashes(parent)


def _verified_r44_source() -> tuple[dict, dict, Path, Path, Path]:
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    result_path = campaign.ROOT / progress["r44_result"]
    run_dir = result_path.parent
    manifest_path = run_dir / "run_manifest.json"
    oof_path = run_dir / "past_oof_predictions.csv"
    if not all(path.is_file() for path in (result_path, manifest_path, oof_path)):
        raise FileNotFoundError("R157 requires the saved R44 strict-past OOF artifact")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    r44_card = campaign.PROTOCOL["cards"].get(R44_CARD, {})
    if result.get("run_id") != r44_card.get("run_id") or manifest.get("run_id") != result.get("run_id"):
        raise RuntimeError("R44 OOF run identity does not match its registered card")
    if result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R44 OOF data hash differs from the active campaign")
    if result.get("reserved_months_labels_opened") is not False or result.get("hidden_labels_opened") is not False:
        raise RuntimeError("R44 OOF source reports restricted-label access")
    if manifest.get("seed_oof_sha256") != _sha256(oof_path):
        raise RuntimeError("R44 past-OOF file hash differs from its run manifest")
    if manifest.get("result_sha256") != _sha256(result_path):
        raise RuntimeError("R44 result hash differs from its run manifest")
    safe_months = set(result.get("safe_oof_months_read", []))
    if not safe_months or safe_months.intersection(RESERVED_MONTHS):
        raise RuntimeError("R44 safe OOF month inventory is empty or contains reserve months")
    return result, manifest, result_path, manifest_path, oof_path


def _load_safe_month_metadata(months: list[str]) -> pd.DataFrame:
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings parquet hash differs from the campaign data hash")
    if set(months).intersection(RESERVED_MONTHS):
        raise RuntimeError("R157 attempted to read a reserved month")
    pieces = []
    for month in sorted(set(months)):
        start = pd.Timestamp(month + "-01")
        stop = start + pd.offsets.MonthBegin(1)
        piece = pd.read_parquet(
            campaign.DATA,
            engine="pyarrow",
            columns=["ml_num", "date_sold", "sold_price", "list_price"],
            filters=[[("date_sold", ">=", start), ("date_sold", "<", stop)]],
        )
        if piece.empty:
            raise RuntimeError(f"date-filtered safe-month metadata is empty for {month}")
        piece["date_sold"] = pd.to_datetime(piece.date_sold, errors="raise")
        observed = set(piece.date_sold.dt.to_period("M").astype(str).unique())
        if observed != {month}:
            raise RuntimeError(f"date filter for {month} returned unexpected months: {sorted(observed)}")
        piece["month"] = month
        piece["ml_num"] = piece.ml_num.astype(str)
        piece["sold_price"] = pd.to_numeric(piece.sold_price, errors="raise")
        piece["list_price"] = pd.to_numeric(piece.list_price, errors="raise")
        pieces.append(piece[["month", "ml_num", "date_sold", "sold_price", "list_price"]])
    metadata = pd.concat(pieces, ignore_index=True)
    if metadata.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("date-filtered metadata has duplicate month/listing keys")
    if not np.isfinite(metadata[["sold_price", "list_price"]].to_numpy(float)).all():
        raise RuntimeError("safe-month sold/list prices contain non-finite values")
    if (metadata[["sold_price", "list_price"]] <= 0.0).any().any():
        raise RuntimeError("safe-month sold/list prices must be positive")
    return metadata


def _load_inputs(stage: str, parent: dict | None = None,
                 r44_source: tuple | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if stage not in {"screen", "development"}:
        raise ValueError("stage must be screen or development")
    parent = parent or _verified_parent()
    r44_source = r44_source or _verified_r44_source()
    r44_result, r44_manifest, r44_result_path, r44_manifest_path, oof_path = r44_source
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    max_month = max(months)
    safe_oof_months = sorted(
        month for month in r44_result["safe_oof_months_read"]
        if month <= max_month
    )
    if set(months) - set(safe_oof_months):
        raise RuntimeError(f"R44 OOF source does not cover all {stage} target months")
    if set(safe_oof_months).intersection(RESERVED_MONTHS):
        raise RuntimeError("R157 safe OOF month selection contains a reserved month")

    oof = pd.read_csv(oof_path, dtype={"ml_num": str})
    oof["month"] = oof.month.astype(str)
    oof = oof.loc[oof.month.isin(safe_oof_months)].copy()
    if oof.empty or oof.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R44 safe OOF input is empty or has duplicate row keys")
    if oof.track.dropna().ne(TRACK).any():
        raise RuntimeError("R44 OOF source contains a non-list-aware prediction route")
    if set(oof.month.unique()) != set(safe_oof_months):
        raise RuntimeError("R44 OOF rows do not cover their registered safe-month inventory")
    if (not np.isfinite(oof[["sold_price", "variant_prediction"]].to_numpy(float)).all()
            or (oof[["sold_price", "variant_prediction"]] <= 0.0).any().any()):
        raise RuntimeError("R44 OOF labels or R35 predictions are invalid")

    metadata_months = sorted(set(safe_oof_months).union(months))
    metadata = _load_safe_month_metadata(metadata_months)
    oof = oof.merge(
        metadata[["month", "ml_num", "sold_price", "list_price"]],
        on=["month", "ml_num"], how="left", validate="one_to_one",
        suffixes=("_oof", "_source"),
    )
    if oof.list_price.isna().any() or oof.sold_price_source.isna().any():
        raise RuntimeError("safe OOF rows lack exact filtered source metadata")
    if not np.array_equal(oof.sold_price_oof.to_numpy(float),
                          oof.sold_price_source.to_numpy(float)):
        raise RuntimeError("R44 OOF labels differ from exact filtered safe-month source labels")

    parent_rows = parent["screen_rows"] if stage == "screen" else parent["development_rows"]
    parent_rows = parent_rows.loc[parent_rows.track.eq(TRACK), [
        "month", "ml_num", "sold_price", "candidate_prediction",
    ]].copy()
    parent_rows["month"] = parent_rows.month.astype(str)
    parent_rows["ml_num"] = parent_rows.ml_num.astype(str)
    if set(parent_rows.month.unique()) != set(months):
        raise RuntimeError(f"R35 parent rows do not cover the exact {stage} months")
    if parent_rows.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R35 parent contains duplicate target listing keys")
    parent_rows = parent_rows.rename(columns={
        "sold_price": "sold_price_parent",
        "candidate_prediction": "parent_prediction",
    })
    source_metadata = metadata[["month", "ml_num", "sold_price", "list_price"]].rename(
        columns={"sold_price": "sold_price_source"}
    )
    parent_rows = parent_rows.merge(
        source_metadata,
        on=["month", "ml_num"], how="left", validate="one_to_one",
    )
    if parent_rows.list_price.isna().any() or parent_rows.sold_price_source.isna().any():
        raise RuntimeError("R35 target rows lack exact filtered source metadata")
    if not np.array_equal(parent_rows.sold_price_parent.to_numpy(float),
                          parent_rows.sold_price_source.to_numpy(float)):
        raise RuntimeError("R35 target labels differ from exact filtered safe-month labels")

    # R44 stores the R35 OOF route in variant_prediction; verify each outer fold.
    for month in months:
        history_variant = oof.loc[oof.month.eq(month), ["ml_num", "variant_prediction"]]
        parent_variant = parent_rows.loc[
            parent_rows.month.eq(month), ["ml_num", "parent_prediction"]
        ]
        left = history_variant.set_index("ml_num").sort_index()
        right = parent_variant.set_index("ml_num").sort_index()
        if (not left.index.equals(right.index)
                or not np.array_equal(left.variant_prediction.to_numpy(float),
                                      right.parent_prediction.to_numpy(float))):
            raise RuntimeError(f"R44 R35 OOF predictions do not align with R35 target rows for {month}")

    parent_rows = parent_rows.rename(columns={"sold_price_parent": "sold_price"})
    parent_rows["track"] = TRACK
    parent_rows = parent_rows.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
    return oof, parent_rows, {
        "r44_result_path": _relative(r44_result_path),
        "r44_result_sha256": _sha256(r44_result_path),
        "r44_manifest_path": _relative(r44_manifest_path),
        "r44_manifest_sha256": _sha256(r44_manifest_path),
        "r44_oof_path": _relative(oof_path),
        "r44_oof_sha256": _sha256(oof_path),
        "safe_oof_months_used": safe_oof_months,
        "metadata_months_read": metadata_months,
        "metadata_rows_read": int(len(metadata)),
        "parent_run_id": parent["screen"]["run_id"],
        "parent_development_run_id": parent["development"]["run_id"],
    }


def _preflight_support() -> dict:
    parent = _verified_parent()
    r44_source = _verified_r44_source()
    oof, queries, _ = _load_inputs("screen", parent, r44_source)
    support = {}
    for month in SCREEN_MONTHS:
        history = oof.loc[oof.month.lt(month)].copy()
        if len(history) < MIN_HISTORY_ROWS:
            raise RuntimeError(f"R157 lacks the registered minimum history for {month}")
        x = np.log(history.variant_prediction.to_numpy(float) / history.list_price.to_numpy(float))
        support[month] = {
            "history_rows": int(len(history)),
            "history_months": sorted(history.month.unique().tolist()),
            "unique_log_ratio_predictions": int(np.unique(x).size),
            "evaluation_rows": int(queries.month.eq(month).sum()),
        }
    return support


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    code_hashes = _code_hashes()
    limits = campaign.PROTOCOL.get("limits", {})
    limit_keys = [
        "campaign_fit_starts", "campaign_minutes", "exploration_fit_starts",
        "exploration_minutes", "final_reserve_fit_starts", "final_reserve_minutes",
        "batch_fit_starts", "batch_minutes", "single_fit_seconds",
    ]
    if any(limits.get(key) is not None for key in limit_keys):
        raise RuntimeError("R157 requires the active unbounded campaign limits")
    parent = _verified_parent()
    parent_hashes = _parent_artifact_hashes(parent)
    r44_result, _, r44_result_path, r44_manifest_path, oof_path = _verified_r44_source()
    from experiments.bill_target74 import r04_ablation as identity_helpers

    identity = identity_helpers._source_identity()
    if identity.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("campaign source data identity changed")
    preflight = _preflight_support()
    input_hashes = {
        "r44_result": _sha256(r44_result_path),
        "r44_manifest": _sha256(r44_manifest_path),
        "r44_past_oof_predictions": _sha256(oof_path),
    }
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} already has status {existing.get('status')}; inspect it before retry")
        if (existing.get("code_sha256") != code_hashes
                or existing.get("parent_artifact_hashes") != parent_hashes
                or existing.get("input_hashes") != input_hashes
                or existing.get("preflight_support") != preflight):
            raise RuntimeError("R157 registration identity changed after its initial registration")
        return existing

    card = {
        "status": "registered",
        "stage": "fixed_two_month_screen_then_eight_month_development",
        "track": TRACK,
        "tracks": [TRACK],
        "screen_months": list(SCREEN_MONTHS),
        "development_months": list(DEVELOPMENT_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS),
        "development_fit_starts_if_screen_passes": len(DEVELOPMENT_MONTHS),
        "comparison_parent": R35_CARD,
        "parent_run_id": parent["screen"]["run_id"],
        "parent_development_run_id": parent["development"]["run_id"],
        "parent_artifact_hashes": parent_hashes,
        "history_source_card": R44_CARD,
        "history_source_run_id": r44_result["run_id"],
        "history_source_policy": "Use only R35 OOF rows whose sale month is strictly earlier than each target month.",
        "history_prediction_column": "variant_prediction (verified equal to R35 candidate_prediction)",
        "history_target": "log(sold_price/list_price)",
        "calibration_input": "log(R35 OOF prediction/list_price)",
        "calibration": {
            "estimator": "sklearn.isotonic.IsotonicRegression",
            "increasing": True,
            "out_of_bounds": "clip",
            "y_min": None,
            "y_max": None,
            "sample_weight": None,
            "loss": "least_squares",
            "minimum_past_oof_rows": MIN_HISTORY_ROWS,
            "price_reconstruction": "round(list_price * exp(isotonic(log(R35_prediction/list_price))))",
        },
        "method_id": "R157_LISTAWARE_PAST_OOF_MONOTONE_RATIO_CALIBRATION",
        "method": (
            "Keep every R35 model, feature, route, HGBR blend and other prediction unchanged. For each "
            "target fold, fit one increasing isotonic map from log(R35 OOF prediction/list_price) to "
            "log(sold_price/list_price), using only verified R35 OOF rows from strictly earlier safe "
            "months. Apply the fold-local map to target listings, reconstruct sale price from the known "
            "listing-time list_price, and round to the nearest dollar."
        ),
        "why_distinct": (
            "R97 fits a multivariable CatBoost residual meta-learner; R113/R116 gate correction magnitude "
            "or direction; R153 estimates local spatial residual differences; R154 fits a residual learner. "
            "R157 instead applies a one-dimensional monotone calibration function to the R35 predicted "
            "sale/list ratio, with no new feature, learner, spatial neighborhood or screen-selected tuning."
        ),
        "preflight_support": preflight,
        "input_hashes": input_hashes,
        "screen_gates": {
            "relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": True,
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": True,
        },
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "target_mdape_strictly_below_percent": 4.0,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "source_identity": identity,
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "code_sha256": code_hashes,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
    }
    campaign.PROTOCOL["cards"][CARD] = card
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_registered",
        "card": CARD,
        "expected_fit_starts": card["expected_fit_starts"],
        "development_fit_starts_if_screen_passes": card["development_fit_starts_if_screen_passes"],
        "parent_artifact_hashes": parent_hashes,
        "input_hashes": input_hashes,
        "source_sha256": code_hashes[_relative(Path(__file__))],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r157_registered",
        next_action="run the preregistered two-fit R157 isotonic calibration screen; retain R35 unless all gates pass",
        r157_card=CARD,
    )
    return card


def _score(paired: pd.DataFrame, months: tuple[str, ...], stage: str) -> tuple[dict, dict]:
    from experiments.bill_target74 import r22_remarks_tfidf as r22

    gates = campaign.PROTOCOL["gates"]
    frame = paired[["month", "ml_num", "sold_price", "parent_prediction",
                    "candidate_prediction"]].copy()
    if stage == "screen":
        score = r22.score_development_confirmation(
            frame,
            months=months,
            target_mdape_strictly_below=4.0,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
            minimum_relative_gain_percent=gates["screen_relative_mdape_improvement_percent"],
            minimum_months_improved=len(months),
            maximum_worst_month_regression_percent=0.0,
            maximum_mape_regression_percent=gates["screen_mape_regression_max_percent"],
        )
        checks = {
            "relative_mdape_gain": score["relative_mdape_improvement_percent"]
                                   >= gates["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": score["months_improved"] == len(months),
            "mape_guard": score["mape_relative_change_percent"]
                          <= gates["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero":
                score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
        }
    else:
        score = r22.score_development_confirmation(
            frame,
            months=months,
            target_mdape_strictly_below=4.0,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
            minimum_relative_gain_percent=gates["development_relative_mdape_improvement_percent"],
            minimum_months_improved=gates["development_months_improved_min"],
            maximum_worst_month_regression_percent=gates["development_worst_month_regression_max_percent"],
            maximum_mape_regression_percent=gates["development_mape_regression_max_percent"],
        )
        checks = {
            "development_relative_gain": score["relative_mdape_improvement_percent"]
                                         >= gates["development_relative_mdape_improvement_percent"],
            "minimum_months_improved": score["months_improved"]
                                       >= gates["development_months_improved_min"],
            "worst_month_guard": score["worst_month_regression_percent"]
                                 <= gates["development_worst_month_regression_max_percent"],
            "mape_guard": score["mape_relative_change_percent"]
                          <= gates["development_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero":
                score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
            "strict_target": score["target_met"],
        }
    return score, checks


def run(stage: str) -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    expected_status = "registered" if stage == "screen" else "screen_passed"
    if not card or card.get("status") != expected_status:
        raise RuntimeError(f"R157 {stage} requires card status {expected_status!r}")
    if card.get("code_sha256") != _code_hashes():
        raise RuntimeError("R157 runner/auditor source hashes changed after registration")
    if card.get("data_sha256") != _sha256(campaign.DATA):
        raise RuntimeError("R157 data hash differs from registration")
    parent = _verified_parent()
    parent_hashes = _parent_artifact_hashes(parent)
    if card.get("parent_artifact_hashes") != parent_hashes:
        raise RuntimeError("R35 parent artifact hashes changed after R157 registration")
    r44_source = _verified_r44_source()
    input_hashes = {
        "r44_result": _sha256(r44_source[2]),
        "r44_manifest": _sha256(r44_source[3]),
        "r44_past_oof_predictions": _sha256(r44_source[4]),
    }
    if card.get("input_hashes") != input_hashes:
        raise RuntimeError("R44 OOF input hashes changed after R157 registration")
    oof, queries, input_identity = _load_inputs(stage, parent, r44_source)
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    expected_fit_starts = len(months)
    run_id = ("r157s_" if stage == "screen" else "r157d_") + datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    paired_parts = []
    profiles = {}
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id, "stage": stage,
        "track": TRACK, "months": list(months),
        "expected_fit_starts": expected_fit_starts,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    for month in months:
        history = oof.loc[oof.month.lt(month)].copy()
        current = queries.loc[queries.month.eq(month)].copy()
        if len(history) < MIN_HISTORY_ROWS or current.empty:
            raise RuntimeError(f"R157 history/evaluation support is insufficient for {month}")
        history_months = sorted(history.month.unique().tolist())
        if not history_months or any(past >= month for past in history_months):
            raise RuntimeError(f"R157 history violates the strict-past cutoff for {month}")
        x_history = np.log(
            history.variant_prediction.to_numpy(float) / history.list_price.to_numpy(float)
        )
        y_history = np.log(
            history.sold_price_source.to_numpy(float) / history.list_price.to_numpy(float)
        )
        x_query = np.log(
            current.parent_prediction.to_numpy(float) / current.list_price.to_numpy(float)
        )
        if not (np.isfinite(x_history).all() and np.isfinite(y_history).all()
                and np.isfinite(x_query).all()):
            raise RuntimeError(f"R157 log-ratio values are non-finite for {month}")
        membership = history[[
            "month", "ml_num", "variant_prediction", "sold_price_source", "list_price",
        ]].sort_values(["month", "ml_num"], kind="mergesort")
        membership_hash = hashlib.sha256(
            pd.util.hash_pandas_object(membership, index=False).to_numpy().tobytes()
        ).hexdigest()
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{month}:{run_id}"
        campaign._append({
            "event": "fit_start", "card": CARD, "run_id": run_id, "stage": stage,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "model_family": "sklearn.isotonic.IsotonicRegression",
            "expected_objective": "least_squares_monotone_calibration",
            "target": "log(sold_price/list_price)",
            "training_rows": int(len(history)),
            "training_months": history_months,
            "training_cutoff_exclusive": f"{month}-01T00:00:00",
            "training_membership_sha256": membership_hash,
            "evaluation_rows": int(len(current)),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        try:
            calibrator = IsotonicRegression(
                increasing=True, out_of_bounds="clip", y_min=None, y_max=None,
            )
            calibrator.fit(x_history, y_history)
            calibrated_log_ratio = calibrator.predict(x_query)
            candidate = np.rint(current.list_price.to_numpy(float)
                                * np.exp(calibrated_log_ratio))
            if not np.isfinite(candidate).all() or (candidate <= 0.0).any():
                raise RuntimeError(f"R157 produced non-finite or non-positive prices for {month}")
        except BaseException as exc:
            campaign._append({
                "event": "fit_fail", "card": CARD, "run_id": run_id,
                "stage": stage, "fit_id": fit_id, "track": TRACK, "month": month,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        profile = {
            "fit_id": fit_id,
            "training_rows": int(len(history)),
            "training_months": history_months,
            "training_membership_sha256": membership_hash,
            "unique_log_ratio_predictions": int(np.unique(x_history).size),
            "x_min": float(np.min(x_history)),
            "x_max": float(np.max(x_history)),
            "y_min": float(np.min(y_history)),
            "y_max": float(np.max(y_history)),
            "isotonic_threshold_count": int(len(calibrator.X_thresholds_)),
            "isotonic_x_thresholds": calibrator.X_thresholds_.tolist(),
            "isotonic_y_thresholds": calibrator.y_thresholds_.tolist(),
        }
        profiles[month] = profile
        campaign._append({
            "event": "fit_complete", "card": CARD, "run_id": run_id, "stage": stage,
            "fit_id": fit_id, "track": TRACK, "month": month, "status": "complete",
            "actual_objective": "least_squares_monotone_calibration",
            "model_family": "sklearn.isotonic.IsotonicRegression",
            "training_rows": int(len(history)), "training_months": history_months,
            "training_membership_sha256": membership_hash,
            "evaluation_rows": int(len(current)),
            "isotonic_threshold_count": profile["isotonic_threshold_count"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        paired_parts.append(pd.DataFrame({
            "month": month, "track": TRACK, "ml_num": current.ml_num.to_numpy(str),
            "sold_price": current.sold_price.to_numpy(float),
            "list_price": current.list_price.to_numpy(float),
            "parent_prediction": current.parent_prediction.to_numpy(float),
            "candidate_prediction": candidate.astype(float),
            "calibrated_log_sale_list_ratio": calibrated_log_ratio,
            "history_oof_rows": int(len(history)),
            "history_months": "|".join(history_months),
        }))

    paired = pd.concat(paired_parts, ignore_index=True)
    if len(paired) != len(queries) or paired.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R157 paired predictions are incomplete or duplicated")
    score, gate_checks = _score(paired, months, stage)
    passed = all(gate_checks.values())
    status = ("screen_passed" if passed else "screen_failed") if stage == "screen" else (
        "development_complete" if passed else "development_failed"
    )
    paired_path = output / "paired_predictions.csv"
    result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
    profile_path = output / "calibration_profiles.json"
    manifest_path = output / "run_manifest.json"
    paired.to_csv(paired_path, index=False)
    _write_json(profile_path, profiles)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "stage": stage, "track": TRACK, "months": list(months),
        "fit_starts": expected_fit_starts,
        "target_mdape_strictly_below_percent": 4.0,
        "tracks": {TRACK: score}, "gate_checks": gate_checks,
        "qualifies_by_track": {TRACK: passed},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
        "safe_oof_months_read": input_identity["safe_oof_months_used"],
        "metadata_months_read": input_identity["metadata_months_read"],
    }
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "stage": stage, "source_identity": card["source_identity"],
        "code_sha256": _code_hashes(), "data_sha256": card["data_sha256"],
        "parent_artifact_hashes": parent_hashes, "input_identity": input_identity,
        "input_hashes": input_hashes, "fit_starts": expected_fit_starts,
        "fit_details": profiles,
        "calibration_profiles_sha256": _sha256(profile_path),
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
    }
    _write_json(manifest_path, manifest)
    report = [
        "# R157 — list-aware isotonic ratio calibration", "",
        f"Run: {run_id}; stage: {stage}; status: {status}.",
        "The only change is a fold-local increasing isotonic map fitted on strictly earlier R35 OOF log sale/list ratios.",
        f"Pooled MdAPE: {score['parent']['mdape_percent']:.7f}% -> {score['candidate']['mdape_percent']:.7f}%.",
        f"Relative MdAPE improvement: {score['relative_mdape_improvement_percent']:.4f}%; MAPE change: {score['mape_relative_change_percent']:.4f}%.",
        f"Gate checks: {gate_checks}.",
        "Only exact safe-month date filters were read. Reserved and hidden labels were not opened.",
    ]
    (output / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    card = campaign.PROTOCOL["cards"][CARD]
    card["status"] = status
    card[f"{stage}_run_id"] = run_id
    card[f"{stage}_fit_starts"] = expected_fit_starts
    card[f"{stage}_result_path"] = _relative(result_path)
    card[f"{stage}_result_sha256"] = _sha256(result_path)
    card[f"{stage}_paired_predictions_path"] = _relative(paired_path)
    card[f"{stage}_paired_predictions_sha256"] = _sha256(paired_path)
    card[f"{stage}_manifest_path"] = _relative(manifest_path)
    card[f"{stage}_manifest_sha256"] = _sha256(manifest_path)
    card[f"{stage}_metrics"] = score
    card[f"{stage}_gate_checks"] = gate_checks
    card[f"{stage}_qualifies"] = passed
    card["reserved_months_labels_opened"] = False
    card["hidden_labels_opened"] = False
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id, "stage": stage,
        "fit_starts": expected_fit_starts, "qualifies_by_track": {TRACK: passed},
        "result_path": _relative(result_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        f"r157_{stage}_{'passed' if passed else 'failed'}",
        next_action=(
            "independently audit R157 before deciding whether to run development"
            if stage == "screen" and passed else
            "retain R35 and register a distinct method after R157 fails screening"
            if stage == "screen" else
            "independently audit R157 development before any parent update"
        ),
        r157_card=CARD, r157_run_id=run_id,
        r157_result=_relative(result_path), r157_fit_starts=expected_fit_starts,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--stage", choices=["screen", "development"])
    args = parser.parse_args()
    if args.register_only:
        print(json.dumps(register_card(), indent=2, ensure_ascii=False), flush=True)
        return
    if not args.stage:
        parser.error("--stage is required unless --register-only is used")
    print(json.dumps(run(args.stage), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
