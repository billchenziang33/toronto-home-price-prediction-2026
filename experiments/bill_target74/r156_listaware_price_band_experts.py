"""R156: train separate list-aware Detached experts by a training-only price median."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


CARD = "R156_listaware_detached_listprice_median_experts"
VARIANT = CARD
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
COMPONENT = "detached_listprice_band_expert"
BANDS = ("low", "high")
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
RESERVED_MONTHS = set(campaign.PROTOCOL["reserved_months"])
TARGET_MDAPE = 4.0
SPECIALIST_WEIGHT = 0.75
MIN_TRAINING_ROWS_PER_BAND = 1000
MIN_EVALUATION_ROWS_PER_BAND = 100
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261156
R35_CARD = "R35_listaware_detached_r22_text_expert"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        campaign.PACKAGE / "r156_independent_audit.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "data.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "model_candidate_v2/winner_config.json",
        campaign.ROOT / "model_candidate_v2/models/frozen_list_aware.json",
    ]


def _code_hashes() -> dict[str, str]:
    return {_relative(path): _sha256(path) for path in _source_files()}


def _verified_r35_parent() -> dict:
    parent_card = campaign.PROTOCOL["cards"].get(R35_CARD)
    if not parent_card or parent_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R156 requires the fixed-origin-validated R35 list-aware route")
    screen_path = campaign.ROOT / parent_card["result_path"]
    screen_rows_path = screen_path.parent / "paired_predictions.csv"
    screen_manifest_path = screen_path.parent / "run_manifest.json"
    development_path = campaign.ROOT / parent_card["development_result_path"]
    development_rows_path = development_path.parent / "paired_predictions.csv"
    development_manifest_path = development_path.parent / "run_manifest.json"
    paths = (screen_path, screen_rows_path, screen_manifest_path,
             development_path, development_rows_path, development_manifest_path)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("one or more registered R35 artifacts are missing")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    screen_manifest = json.loads(screen_manifest_path.read_text(encoding="utf-8"))
    development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    checks = (
        screen.get("run_id") == parent_card.get("run_id") == screen_manifest.get("run_id"),
        development.get("run_id") == parent_card.get("development_run_id") == development_manifest.get("run_id"),
        screen_manifest.get("result_sha256") == _sha256(screen_path),
        screen_manifest.get("paired_predictions_sha256") == _sha256(screen_rows_path),
        development_manifest.get("result_sha256") == _sha256(development_path),
        development_manifest.get("paired_predictions_sha256") == _sha256(development_rows_path),
        screen_manifest.get("source_identity", {}).get("data_sha256") == campaign.PROTOCOL["data_sha256"],
        development_manifest.get("source_identity", {}).get("data_sha256") == campaign.PROTOCOL["data_sha256"],
        screen.get("tracks", {}).get(TRACK, {}).get("qualifies_screen") is True,
        development.get("tracks", {}).get(TRACK, {}).get("qualifies_development") is True,
    )
    if not all(checks):
        raise RuntimeError("R35 parent results, metrics, or manifests failed verification")
    for result in (screen, development):
        if result.get("reserved_months_labels_opened") is not False or result.get("hidden_labels_opened") is not False:
            raise RuntimeError("R35 parent records restricted-label access")
    screen_rows = pd.read_csv(screen_rows_path, dtype={"ml_num": str})
    development_rows = pd.read_csv(development_rows_path, dtype={"ml_num": str})
    screen_rows = screen_rows.loc[screen_rows.track.eq(TRACK)].copy()
    development_rows = development_rows.loc[development_rows.track.eq(TRACK)].copy()
    for name, frame, expected_months in (
        ("screen", screen_rows, set(SCREEN_MONTHS)),
        ("development", development_rows, set(DEVELOPMENT_MONTHS)),
    ):
        frame["month"] = frame.month.astype(str)
        if set(frame.month) != expected_months or frame.duplicated(["month", "ml_num"]).any():
            raise RuntimeError(f"R35 {name} predictions have incomplete or duplicate month/listing keys")
        values = frame[["sold_price", "candidate_prediction"]].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise RuntimeError(f"R35 {name} predictions or labels are non-positive/non-finite")
    return {
        "card": parent_card,
        "screen": screen,
        "screen_path": screen_path,
        "screen_rows_path": screen_rows_path,
        "screen_manifest_path": screen_manifest_path,
        "screen_rows": screen_rows,
        "development": development,
        "development_path": development_path,
        "development_rows_path": development_rows_path,
        "development_manifest_path": development_manifest_path,
        "development_rows": development_rows,
    }


def _parent_artifact_hashes(parent: dict) -> dict[str, str]:
    return {
        "screen_result_sha256": _sha256(parent["screen_path"]),
        "screen_predictions_sha256": _sha256(parent["screen_rows_path"]),
        "screen_manifest_sha256": _sha256(parent["screen_manifest_path"]),
        "development_result_sha256": _sha256(parent["development_path"]),
        "development_predictions_sha256": _sha256(parent["development_rows_path"]),
        "development_manifest_sha256": _sha256(parent["development_manifest_path"]),
    }


def _split_from_training(train: pd.DataFrame, homes: pd.DataFrame, month: str) -> dict:
    from experiments.bill_greedy import runner as greedy

    cutoff = pd.Period(month, freq="M").start_time
    train_dates = pd.to_datetime(train.date_sold, errors="raise")
    if (train_dates >= cutoff).any():
        raise RuntimeError(f"R156 training rows reach the {month} cutoff")
    if train_dates.dt.to_period("M").astype(str).isin(RESERVED_MONTHS).any():
        raise RuntimeError(f"R156 training rows include a reserved month for {month}")
    train_detached = train.loc[greedy._specialist_mask(train, HOUSE_TYPE)].copy()
    train_price = pd.to_numeric(train_detached.list_price, errors="raise").to_numpy(float)
    eval_detached = homes.loc[greedy._specialist_mask(homes, HOUSE_TYPE)].copy()
    eval_price = pd.to_numeric(eval_detached.list_price, errors="raise").to_numpy(float)
    if (not np.isfinite(train_price).all() or (train_price <= 0).any()
            or not np.isfinite(eval_price).all() or (eval_price <= 0).any()):
        raise RuntimeError(f"R156 requires finite positive list prices for Detached rows in {month}")
    threshold = float(np.median(train_price))
    train_band = np.where(train_price <= threshold, "low", "high")
    eval_band = np.where(eval_price <= threshold, "low", "high")
    details = {
        "month": month,
        "cutoff_exclusive": cutoff.isoformat(),
        "threshold_method": "median of positive list_price among strict-past Detached training sales",
        "threshold": threshold,
        "training_rows_total": int(len(train_detached)),
        "evaluation_rows_total": int(len(eval_detached)),
        "bands": {},
    }
    for band in BANDS:
        training_rows = int(np.count_nonzero(train_band == band))
        evaluation_rows = int(np.count_nonzero(eval_band == band))
        if (training_rows < MIN_TRAINING_ROWS_PER_BAND
                or evaluation_rows < MIN_EVALUATION_ROWS_PER_BAND):
            raise RuntimeError(
                f"R156 {month}/{band} support too small: train={training_rows}, eval={evaluation_rows}"
            )
        details["bands"][band] = {
            "training_rows": training_rows,
            "evaluation_rows": evaluation_rows,
        }
    return {"details": details, "train_detached": train_detached,
            "eval_detached": eval_detached, "train_band": train_band,
            "eval_band": eval_band, "threshold": threshold}


def _preflight(months: tuple[str, ...]) -> dict[str, dict]:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy

    result = {}
    for month in months:
        _, train, _, _, heldout = greedy._load_fold(month)
        homes = hide_answers(heldout, TRACK).reset_index(drop=True)
        result[month] = _split_from_training(train, homes, month)["details"]
    return result


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    code_hashes = _code_hashes()
    parent = _verified_r35_parent()
    parent_hashes = _parent_artifact_hashes(parent)
    from experiments.bill_target74 import r04_ablation as identity_helpers
    source = identity_helpers._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active data identity differs from campaign protocol")
    preflight = _preflight(SCREEN_MONTHS)
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} is already {existing.get('status')}; inspect saved output first")
        if existing.get("code_sha256") != code_hashes:
            raise RuntimeError("R156 source or dependency hashes changed after registration")
        if existing.get("parent_artifact_hashes") != parent_hashes:
            raise RuntimeError("R35 parent artifacts changed after R156 registration")
        if existing.get("preflight_support") != preflight:
            raise RuntimeError("R156 screen support changed after registration")
        return existing

    model_config = parent["screen"]["fit_details"][0]["model_config"]
    card = {
        "status": "registered",
        "stage": "fixed_two_month_screen_then_eight_month_development",
        "track": TRACK,
        "tracks": [TRACK],
        "house_type": HOUSE_TYPE,
        "component": COMPONENT,
        "screen_months": list(SCREEN_MONTHS),
        "development_months": list(DEVELOPMENT_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS) * len(BANDS),
        "development_fit_starts_if_screen_passes": len(DEVELOPMENT_MONTHS) * len(BANDS),
        "comparison_parent": R35_CARD,
        "parent_run_id": parent["screen"]["run_id"],
        "parent_development_run_id": parent["development"]["run_id"],
        "parent_artifact_hashes": parent_hashes,
        "model_family": "LightGBM",
        "model_config": model_config,
        "target": "log(sold_price/list_price), unchanged from R35",
        "method_id": "R156_DETACHED_LISTPRICE_MEDIAN_SPECIALISTS",
        "method": (
            "Within the existing R35 Detached route, compute the median positive list_price from only the "
            "strict-past Detached training rows for each fold. Train separate low-band (<= median) and "
            "high-band (> median) LightGBM experts with the unchanged R35 settings, log sale/list target, "
            "structured/history features, and per-expert fold-local word TF-IDF. Assign each held-out "
            "Detached listing by its listing-time list_price, blend its routed expert 75/25 with the same "
            "frozen HGBR prediction, and leave every non-Detached R35 prediction unchanged."
        ),
        "why_distinct": (
            "R49 chose past-only blend weights between existing models by predicted-price tercile; it did "
            "not train separate price-band experts. R123 gated an R97 residual model by house type, and "
            "R134/R135 changed the Detached feature path. R156 instead fits two disjoint R35-model training "
            "subsets using a per-fold list-price median computed without screen labels. The earlier R35 "
            "price-quartile error audit motivates testing price heterogeneity but does not select a threshold "
            "from these target-fold labels."
        ),
        "band_definition": {
            "threshold": "median of positive list_price among strict-past Detached training rows per fold",
            "low": "list_price <= threshold",
            "high": "list_price > threshold",
            "minimum_training_rows_per_band": MIN_TRAINING_ROWS_PER_BAND,
            "minimum_evaluation_rows_per_band": MIN_EVALUATION_ROWS_PER_BAND,
        },
        "preflight_support": preflight,
        "blend": {"challenger_weight": SPECIALIST_WEIGHT, "frozen_hgbr_weight": 1.0 - SPECIALIST_WEIGHT},
        "screen_gates": {
            "relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": True,
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": True,
        },
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "target_mdape_strictly_below_percent": TARGET_MDAPE,
        "source_identity": source,
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "code_sha256": code_hashes,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
    }
    campaign.PROTOCOL["cards"][CARD] = card
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_registered", "card": CARD,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_fit_starts": card["expected_fit_starts"],
        "development_fit_starts_if_screen_passes": card["development_fit_starts_if_screen_passes"],
        "parent_artifact_hashes": parent_hashes,
        "source_sha256": code_hashes[_relative(Path(__file__))],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r156_registered",
        next_action="run the preregistered four-fit R156 list-price-band screen; retain R35 unless every gate passes",
        r156_card=CARD,
    )
    return card


def _score(paired: pd.DataFrame, months: tuple[str, ...], stage: str) -> tuple[dict, dict]:
    from experiments.bill_target74 import r22_remarks_tfidf as r22

    frame = paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
    common = {
        "months": months,
        "target_mdape_strictly_below": TARGET_MDAPE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if stage == "screen":
        score = r22.score_development_confirmation(
            frame,
            **common,
            minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            minimum_months_improved=len(months),
            maximum_worst_month_regression_percent=0.0,
            maximum_mape_regression_percent=campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        )
        return score, {
            "relative_mdape_gain": score["relative_mdape_improvement_percent"]
                                   >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": score["months_improved"] == len(months),
            "mape_guard": score["mape_relative_change_percent"]
                          <= campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
        }
    score = r22.score_development_confirmation(frame, **common)
    return score, {
        "development_gates": bool(score["qualifies_development"]),
        "strict_mdape_target": bool(score["target_met"]),
    }


def run(stage: str) -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as cache_helpers
    from experiments.bill_target74 import r04_ablation as identity_helpers
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    card = campaign.PROTOCOL["cards"].get(CARD)
    expected_status = "registered" if stage == "screen" else "screen_passed"
    if not card or card.get("status") != expected_status:
        raise RuntimeError(f"R156 {stage} requires card status {expected_status}")
    if card.get("code_sha256") != _code_hashes():
        raise RuntimeError("R156 source or dependency hashes changed after preregistration")
    parent = _verified_r35_parent()
    parent_hashes = _parent_artifact_hashes(parent)
    if card.get("parent_artifact_hashes") != parent_hashes:
        raise RuntimeError("R35 parent artifact hashes changed after R156 registration")
    source = identity_helpers._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active data identity differs from the preregistered campaign")

    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    if set(months).intersection(RESERVED_MONTHS):
        raise RuntimeError("R156 cannot read a reserved-month input artifact")
    parent_rows = parent["screen_rows"] if stage == "screen" else parent["development_rows"]
    run_id = ("r156s_" if stage == "screen" else "r156d_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    parent_sources = [
        (parent["screen_path"], "r35_screen_results.json"),
        (parent["screen_rows_path"], "r35_screen_paired_predictions.csv"),
        (parent["screen_manifest_path"], "r35_screen_run_manifest.json"),
        (parent["development_path"], "r35_development_results.json"),
        (parent["development_rows_path"], "r35_development_paired_predictions.csv"),
        (parent["development_manifest_path"], "r35_development_run_manifest.json"),
    ]
    for source_path, name in parent_sources:
        shutil.copy2(source_path, provenance / name)
    code_hashes = _code_hashes()
    source_snapshots = {}
    for index, path in enumerate(_source_files()):
        relative = _relative(path)
        snapshot_name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / snapshot_name)
        source_snapshots[relative] = snapshot_name

    safe_input_artifacts = {}
    for month in months:
        cache_path = greedy.OUTPUT / "frozen_hgbr" / TRACK / f"{month}.csv"
        if not cache_path.is_file():
            raise FileNotFoundError(f"R156 frozen HGBR safe-month cache is missing: {cache_path}")
        snapshot_name = f"frozen_hgbr_{month}.csv"
        shutil.copy2(cache_path, provenance / snapshot_name)
        safe_input_artifacts[month] = {
            "original_path": _relative(cache_path),
            "snapshot": snapshot_name,
            "sha256": _sha256(cache_path),
        }

    expected_fit_starts = len(months) * len(BANDS)
    campaign._append({
        "event": "card_start", "card": CARD, "stage": stage, "run_id": run_id,
        "track": TRACK, "parent_run_id": parent["screen"]["run_id"],
        "months": list(months), "bands": list(BANDS),
        "expected_fit_starts": expected_fit_starts,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    fit_details: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    fit_context: dict[str, object] = {}
    old_fit_record = greedy._fit_record
    old_variant_exists = VARIANT in greedy.VARIANTS
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = {}

    def record_fit(_original_id: str, fit_fn):
        month = str(fit_context["month"])
        band = str(fit_context["band"])
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{month}:{band}:{run_id}"
        membership_path = Path(fit_context["membership_path"])
        membership_hash = _sha256(membership_path)
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": COMPONENT, "price_band": band,
            "training_rows": fit_context["training_rows"],
            "evaluation_rows": fit_context["evaluation_rows"],
            "training_membership_sha256": membership_hash,
            "list_price_median_threshold": fit_context["threshold"],
            "target": "log_sold_over_list", "expected_objective": "regression",
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            if fitted.get("model_family") != "lightgbm" or fitted.get("target") != "log_sold_over_list":
                raise RuntimeError("R156 fit did not use the R35 LightGBM/log-ratio route")
            if fitted.get("actual_objective") != "regression":
                raise RuntimeError(f"R156 expected regression objective, found {fitted.get('actual_objective')!r}")
            if fitted.get("model_config") != card["model_config"]:
                raise RuntimeError("R156 model parameters differ from the locked R35 configuration")
            predictions = np.asarray(fitted["predictions"], dtype=float)
            if not np.isfinite(predictions).all() or (predictions <= 0).any():
                raise RuntimeError("R156 core produced invalid dollar predictions")
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": stage, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "component": COMPONENT, "price_band": band, "status": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": COMPONENT, "price_band": band,
            "seconds": time.perf_counter() - started,
            "training_rows": int(fitted["training_rows"]),
            "evaluation_rows": int(fitted["evaluation_rows"]),
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "actual_objective": fitted["actual_objective"], "target": fitted["target"],
            "feature_columns": fitted["feature_columns"],
            "text_feature_count": sum(name.startswith("remarks_tfidf_") for name in fitted["feature_columns"]),
            "prediction_sha256": fitted["prediction_sha256"],
            "training_membership_path": _relative(membership_path),
            "training_membership_sha256": membership_hash,
            "list_price_median_threshold": fit_context["threshold"],
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "stage": stage, "run_id": run_id,
            **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in months:
            _, train, _, truth, heldout = greedy._load_fold(month)
            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            split = _split_from_training(train, homes, month)
            support = split["details"]
            if stage == "screen" and card["preflight_support"].get(month) != support:
                raise RuntimeError(f"R156 {month} support or median changed since preregistration")
            cutoff = pd.Period(month, freq="M").start_time
            parent_month = parent_rows.loc[parent_rows.month.astype(str).eq(month)].copy()
            parent_month = parent_month.set_index(parent_month.ml_num.astype(str), drop=False)
            ids = homes.ml_num.astype(str).to_numpy()
            if len(parent_month) != len(homes) or set(parent_month.index) != set(ids):
                raise RuntimeError(f"R35 parent listing IDs do not align for {month}")
            parent_month = parent_month.loc[ids]
            parent_values = parent_month.candidate_prediction.to_numpy(float)
            if not np.array_equal(parent_month.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
                raise RuntimeError(f"R35 parent labels do not align to safe fold {month}")
            if not np.array_equal(parent_month.ml_num.astype(str).to_numpy(), ids):
                raise RuntimeError(f"R35 parent order is not aligned to safe fold {month}")

            eval_detached = split["eval_detached"]
            eval_band = split["eval_band"]
            train_detached = split["train_detached"]
            train_band = split["train_band"]
            threshold = split["threshold"]
            route_mask = greedy._specialist_mask(homes, HOUSE_TYPE)
            if int(route_mask.sum()) != len(eval_detached):
                raise RuntimeError(f"R156 Detached route changed for {month}")
            cache_path = greedy.OUTPUT / "frozen_hgbr" / TRACK / f"{month}.csv"
            if _sha256(cache_path) != safe_input_artifacts[month]["sha256"]:
                raise RuntimeError(f"R156 {month} frozen-HGBR cache changed after snapshot")
            frozen_all = cache_helpers._verified_hgbr_cache(greedy, month, TRACK, train, homes)
            if _sha256(cache_path) != safe_input_artifacts[month]["sha256"]:
                raise RuntimeError(f"R156 {month} frozen-HGBR cache changed while being read")

            band_components = []
            band_outputs = {}
            for band in BANDS:
                band_train_mask = train_band == band
                band_eval_mask = eval_band == band
                fit_rows = train_detached.iloc[np.flatnonzero(band_train_mask)].copy()
                eval_rows = eval_detached.iloc[np.flatnonzero(band_eval_mask)].copy()
                train_ids = sorted(fit_rows.ml_num.astype(str).tolist())
                eval_ids = eval_rows.ml_num.astype(str).tolist()
                membership_path = output / "fit_membership" / f"{month}_{band}.json"
                membership_path.parent.mkdir(exist_ok=True)
                _write_json(membership_path, {
                    "month": month, "band": band, "threshold": threshold,
                    "threshold_method": support["threshold_method"],
                    "cutoff_exclusive": cutoff.isoformat(),
                    "training_ids": train_ids, "evaluation_ids": eval_ids,
                    "training_rows": len(fit_rows), "evaluation_rows": len(eval_rows),
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
                transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
                fit_context.update({
                    "month": month, "band": band,
                    "training_rows": len(fit_rows), "evaluation_rows": len(eval_rows),
                    "threshold": threshold, "membership_path": membership_path,
                })
                try:
                    fitted = greedy._fit_history_candidate(
                        month, TRACK, train, eval_rows, fit_rows=fit_rows,
                        variant=VARIANT, fit_tag=f"R156-{month}-{band}-Detached",
                    )
                finally:
                    restore()
                frozen_band = frozen_all[route_mask][band_eval_mask]
                blended_band = blend_prices(
                    fitted["predictions"], frozen_band, challenger_weight=SPECIALIST_WEIGHT,
                )
                if (len(blended_band) != len(eval_rows) or not np.isfinite(blended_band).all()
                        or (blended_band <= 0).any()):
                    raise RuntimeError(f"R156 invalid blend for {month}/{band}")
                band_outputs[band] = {
                    "ids": eval_ids,
                    "core": np.asarray(fitted["predictions"], dtype=float),
                    "frozen": np.asarray(frozen_band, dtype=float),
                    "blended": np.asarray(blended_band, dtype=float),
                    "feature_columns": fitted["feature_columns"],
                    "text_feature_count": len(transformer.columns_ or []),
                    "training_rows": len(fit_rows),
                    "evaluation_rows": len(eval_rows),
                    "membership_path": _relative(membership_path),
                }

            # Build a Detached-only component table in held-out row order.
            detached_ids = eval_detached.ml_num.astype(str).to_numpy()
            detached_band_labels = np.asarray(eval_band, dtype=object)
            core_by_id, frozen_by_id, blend_by_id = {}, {}, {}
            for band in BANDS:
                block = band_outputs[band]
                for listing_id, core_value, frozen_value, blend_value in zip(
                    block["ids"], block["core"], block["frozen"], block["blended"], strict=True
                ):
                    if listing_id in core_by_id:
                        raise RuntimeError(f"R156 listing assigned to multiple price bands: {listing_id}")
                    core_by_id[listing_id] = float(core_value)
                    frozen_by_id[listing_id] = float(frozen_value)
                    blend_by_id[listing_id] = float(blend_value)
            if set(core_by_id) != set(detached_ids):
                raise RuntimeError(f"R156 band assignment does not cover every Detached row in {month}")
            specialist = np.asarray([blend_by_id[x] for x in detached_ids], dtype=float)
            prediction = greedy._overlay_specialist_predictions(
                parent_values, homes, specialist, HOUSE_TYPE,
            )
            if not np.array_equal(prediction[~route_mask], parent_values[~route_mask]):
                raise RuntimeError(f"R156 changed non-Detached predictions in {month}")
            component_frame = pd.DataFrame({
                "month": month,
                "ml_num": detached_ids,
                "list_price": pd.to_numeric(eval_detached.list_price, errors="raise").to_numpy(float),
                "price_band": detached_band_labels,
                "list_price_median_threshold": threshold,
                "core_price_prediction": [core_by_id[x] for x in detached_ids],
                "frozen_hgbr_prediction": [frozen_by_id[x] for x in detached_ids],
                "blended_prediction": specialist,
                "parent_prediction": parent_values[route_mask],
                "sold_price": truth.sold_price.to_numpy(float)[route_mask],
            })
            component_path = output / f"detached_components_{month}.csv"
            component_frame.to_csv(component_path, index=False)
            paired_parts.append(pd.DataFrame({
                "month": month, "track": TRACK, "ml_num": ids,
                "sold_price": truth.sold_price.to_numpy(float),
                "parent_prediction": parent_values,
                "candidate_prediction": prediction,
                "detached_route": route_mask.astype(bool),
                "price_band": np.where(route_mask, np.where(
                    pd.to_numeric(homes.list_price, errors="coerce").to_numpy(float) <= threshold,
                    "low", "high"), "non_detached"),
            }))
            route_truth = truth.sold_price.to_numpy(float)[route_mask]
            route_details[month] = {
                **support,
                "parent": campaign._metrics(route_truth, parent_values[route_mask]),
                "candidate": campaign._metrics(route_truth, prediction[route_mask]),
                "non_detached_predictions_unchanged": True,
                "component_path": _relative(component_path),
                "component_sha256": _sha256(component_path),
                "bands": {
                    band: {
                        "training_rows": band_outputs[band]["training_rows"],
                        "evaluation_rows": band_outputs[band]["evaluation_rows"],
                        "training_membership_path": band_outputs[band]["membership_path"],
                        "feature_columns": band_outputs[band]["feature_columns"],
                        "text_feature_count": band_outputs[band]["text_feature_count"],
                    }
                    for band in BANDS
                },
                "training_cutoff_exclusive": cutoff.isoformat(),
                "frozen_hgbr_cache_sha256": safe_input_artifacts[month]["sha256"],
            }
            campaign._append({
                "event": "screen_fold_complete" if stage == "screen" else "development_fold_complete",
                "card": CARD, "stage": stage, "run_id": run_id, "track": TRACK,
                "month": month, "routed_rows": int(route_mask.sum()),
                "routed_training_rows": int(len(train_detached)),
                "list_price_median_threshold": threshold,
                "training_cutoff_exclusive": cutoff.isoformat(),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            pd.concat(paired_parts, ignore_index=True).to_csv(
                output / "partial_paired_predictions.csv", index=False,
            )
    except Exception as exc:
        greedy._fit_record = old_fit_record
        failure = {
            "run_id": run_id, "card": CARD, "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "route_details": route_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_starts": len(fit_details), "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign.PROTOCOL["cards"][CARD].update({
            "status": f"{stage}_failed_runtime", "run_id": run_id,
            "failure_path": _relative(output / "failure_checkpoint.json"),
            "fit_starts": len(fit_details),
        })
        _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._write_progress(
            f"r156_{stage}_failed_runtime",
            next_action="audit the R156 failure checkpoint and ledger before any retry",
            r156_run_id=run_id,
        )
        raise
    finally:
        greedy._fit_record = old_fit_record
        if old_variant_exists:
            greedy.VARIANTS[VARIANT] = old_variant
        else:
            greedy.VARIANTS.pop(VARIANT, None)

    if len(fit_details) != expected_fit_starts:
        raise RuntimeError(f"R156 expected {expected_fit_starts} fits; recorded {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    if paired.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R156 paired predictions contain duplicate keys")
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score, gate_checks = _score(paired, tuple(months), stage)
    passed = all(gate_checks.values()) if stage == "screen" else bool(
        score["qualifies_development"] and score["target_met"]
    )
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "track": TRACK, "parent_card": R35_CARD,
        "parent_run_id": parent["screen"]["run_id"],
        "stage": stage, "months": list(months), "bands": list(BANDS),
        "fit_starts": len(fit_details), "bootstrap_seed": BOOTSTRAP_SEED,
        "model_family": "LightGBM", "model_config": card["model_config"],
        "target": "log_sold_over_list", "blend": card["blend"],
        "tracks": {TRACK: score}, "gate_checks": gate_checks,
        "qualifies_by_track": {TRACK: passed},
        "routed_diagnostics": route_details, "fit_details": fit_details,
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
    }
    result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "source_snapshots": source_snapshots,
        "parent_card": R35_CARD, "parent_artifact_hashes": parent_hashes,
        "safe_input_artifacts": safe_input_artifacts,
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = [
        "# R156 list-aware Detached list-price-band experts", "",
        f"Run: `{run_id}`; parent: validated R35 `{parent['screen']['run_id']}`.", "",
        "Each fold uses its strict-past Detached training median list price to split low/high experts. "
        "The experts retain R35 LightGBM settings, log sale/list target, fold-local word TF-IDF, "
        "and the 75/25 frozen-HGBR blend; non-Detached rows remain exactly R35.", "",
        f"Stage: {stage}; months: {', '.join(months)}; fit starts: {len(fit_details)}.",
        f"Parent MdAPE: {score['parent']['mdape_percent']:.6f}%.",
        f"Candidate MdAPE: {score['candidate']['mdape_percent']:.6f}%.",
        f"Relative MdAPE gain: {score['relative_mdape_improvement_percent']:+.4f}%.",
        f"MAPE relative change: {score['mape_relative_change_percent']:+.4f}%.",
        f"Gate checks: `{json.dumps(gate_checks, ensure_ascii=False)}`.",
        "No reserve-month or hidden labels were opened.",
        f"Artifacts: `{_relative(output)}`.", "",
    ]
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")
    campaign._append({
        "event": "card_complete", "card": CARD, "stage": stage, "run_id": run_id,
        "fit_starts": len(fit_details), "result_path": _relative(result_path),
        "qualifies_by_track": {TRACK: passed},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    updated = campaign.PROTOCOL["cards"][CARD]
    updated.update({
        "status": ("screen_passed" if passed else "screen_failed") if stage == "screen" else "development_complete",
        "run_id": run_id,
        f"{stage}_fit_starts": len(fit_details),
        f"{stage}_result_path": _relative(result_path),
        f"{stage}_result_sha256": _sha256(result_path),
        f"{stage}_paired_predictions_path": _relative(paired_path),
        f"{stage}_paired_predictions_sha256": _sha256(paired_path),
        f"{stage}_manifest_path": _relative(output / "run_manifest.json"),
        f"{stage}_manifest_sha256": _sha256(output / "run_manifest.json"),
        f"{stage}_metrics": score,
        f"{stage}_gate_checks": gate_checks,
        "qualifies_screen": passed if stage == "screen" else updated.get("qualifies_screen", False),
        "qualifies_development": passed if stage == "development" else updated.get("qualifies_development", False),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    if stage == "screen":
        next_action = "R156 passed every fixed screen gate; run eight-month development" if passed else \
            "R156 failed screening; retain R35 and register a distinct method"
        progress_status = "r156_screen_passed" if passed else "r156_screen_failed"
    else:
        next_action = "R156 met the strict development target; review eligible fixed-origin procedure" if passed else \
            "R156 did not meet the strict target and development gates; retain R35"
        progress_status = "r156_development_passed" if passed else "r156_development_failed"
    campaign._write_progress(
        progress_status, next_action=next_action,
        r156_run_id=run_id, r156_result=_relative(result_path),
        r156_fit_starts=len(fit_details),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="R156 price-band experts on the R35 list-aware route")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register-only", action="store_true")
    group.add_argument("--screen", action="store_true")
    group.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if args.register_only:
        print(json.dumps(register_card(), indent=2, ensure_ascii=False), flush=True)
    else:
        print(json.dumps(run("screen" if args.screen else "development"), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
