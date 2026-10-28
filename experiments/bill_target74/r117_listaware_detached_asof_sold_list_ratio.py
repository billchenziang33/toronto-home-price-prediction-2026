"""R117: screen strictly-as-of local historical sale-to-list features in R35 Detached."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R117_listaware_detached_asof_sold_list_ratio"
VARIANT = CARD
PARENT_CARD = "R35_listaware_detached_r22_text_expert"
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
SCREEN_MONTHS = ("2023-03", "2024-06")
CORE_WEIGHT = 0.75
MODEL_OVERRIDES = {}
FEATURE_COLUMNS = [
    "community_type_log_sold_list_mean_90d",
    "community_type_sold_list_count_90d",
    "community_type_log_sold_list_mean_365d",
    "community_type_sold_list_count_365d",
    "municipality_type_log_sold_list_mean_90d",
    "municipality_type_sold_list_count_90d",
    "municipality_type_log_sold_list_mean_365d",
    "municipality_type_sold_list_count_365d",
]
R99_RESULT = "outputs/bill_target74/bill_target74_unbounded_20261024/batches/r99m_20261025T212503Z/results.json"
R102_RESULT = "outputs/bill_target74/bill_target74_unbounded_20261024/batches/r102m_20261025T221146Z/results.json"


def _source_files(campaign) -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r34_house_type_text_experts.py",
        campaign.PACKAGE / "r91_listaware_detached_log_list_ppsf.py",
        campaign.PACKAGE / "r99_asof_community_type_price_area.py",
        campaign.PACKAGE / "r102_type_market_momentum.py",
        campaign.ROOT / R99_RESULT,
        campaign.ROOT / R102_RESULT,
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/model_optimization_v2/features.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
        campaign.ROOT / "model_candidate_v2/winner_config.json",
        campaign.ROOT / "model_candidate_v2/models/frozen_list_aware.json",
    ]


def _group_key(row, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("__MISSING__" if pd.isna(row[column]) else str(row[column]) for column in columns)


def _prefix_ratio_groups(history: pd.DataFrame, group_columns: tuple[str, ...]) -> dict:
    records = {}
    for key, rows in history.groupby(list(group_columns), dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        rows = rows.sort_values("date_sold", kind="mergesort")
        dates = pd.to_datetime(rows.date_sold).to_numpy(dtype="datetime64[ns]")
        values = rows.log_sold_list.to_numpy(dtype=float)
        ids = rows.ml_num.astype(str).to_numpy()
        sums = np.concatenate(([0.0], np.cumsum(values)))
        records[tuple("__MISSING__" if pd.isna(value) else str(value) for value in key)] = (
            dates, values, ids, sums
        )
    return records


def _ratio_stats(records: dict, key: tuple[str, ...], query_time: pd.Timestamp,
                 query_id: str, window_days: int) -> tuple[float, int]:
    item = records.get(key)
    if item is None:
        return 0.0, 0
    dates, values, ids, sums = item
    query_date = np.datetime64(query_time)
    end = int(np.searchsorted(dates, query_date, side="left"))
    start = int(np.searchsorted(
        dates, query_date - np.timedelta64(window_days, "D"), side="left"
    ))
    total = float(sums[end] - sums[start])
    count = end - start
    if count:
        matching = np.flatnonzero(ids[start:end] == query_id)
        if len(matching):
            offset = start + int(matching[0])
            total -= float(values[offset])
            count -= 1
    return (total / count if count else 0.0), count


def _asof_negotiation_features(queries: pd.DataFrame, history: pd.DataFrame,
                               outer_cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Return past-only log(sold/list) means and counts by local type and community."""
    required_q = {"ml_num", "date_listed", "community", "municipality", "house_type_name"}
    required_h = {"ml_num", "date_sold", "date_listed", "sold_price", "list_price",
                  "community", "municipality", "house_type_name"}
    if required_q - set(queries) or required_h - set(history):
        raise ValueError("R117 as-of ratio inputs are missing required columns")
    if queries.ml_num.astype(str).duplicated().any() or history.ml_num.astype(str).duplicated().any():
        raise RuntimeError("R117 listing IDs must be unique within query and history frames")

    cutoff = pd.Timestamp(outer_cutoff)
    valid = history.copy()
    valid["date_sold"] = pd.to_datetime(valid.date_sold, errors="raise")
    valid["date_listed"] = pd.to_datetime(valid.date_listed, errors="raise")
    valid["sold_price"] = pd.to_numeric(valid.sold_price, errors="coerce")
    valid["list_price"] = pd.to_numeric(valid.list_price, errors="coerce")
    if (valid.date_sold >= cutoff).any():
        raise RuntimeError("R117 history contains sales at or after the outer fold cutoff")
    sold_month = valid.date_sold.dt.to_period("M").astype(str)
    if sold_month.isin(["2025-05", "2025-08"]).any():
        raise RuntimeError("R117 history contains restricted reserve-month labels")
    usable = (valid.date_sold > valid.date_listed
              ) & np.isfinite(valid.sold_price) & (valid.sold_price > 0.0
              ) & np.isfinite(valid.list_price) & (valid.list_price > 0.0)
    valid = valid.loc[usable].copy()
    valid["log_sold_list"] = np.log(valid.sold_price / valid.list_price)
    if not np.isfinite(valid.log_sold_list.to_numpy(float)).all():
        raise RuntimeError("R117 history generated non-finite log sale-to-list values")

    community_cols = ("community", "municipality", "house_type_name")
    municipality_cols = ("municipality", "house_type_name")
    community_records = _prefix_ratio_groups(valid, community_cols)
    municipality_records = _prefix_ratio_groups(valid, municipality_cols)

    q = queries.copy()
    q["date_listed"] = pd.to_datetime(q.date_listed, errors="raise")
    if q.date_listed.isna().any():
        raise RuntimeError("R117 query has a missing listing date")
    output = []
    support = {"query_rows": int(len(q)), "eligible_history_rows": int(len(valid)),
               "community": {}, "municipality": {}}
    for row in q.itertuples(index=False):
        values = row._asdict()
        query_time = min(pd.Timestamp(values["date_listed"]), cutoff)
        identity = str(values["ml_num"])
        community_key = _group_key(values, community_cols)
        municipality_key = _group_key(values, municipality_cols)
        features = {}
        for scope, records, key in (
            ("community", community_records, community_key),
            ("municipality", municipality_records, municipality_key),
        ):
            for window in (90, 365):
                mean, count = _ratio_stats(records, key, query_time, identity, window)
                features[f"{scope}_type_log_sold_list_mean_{window}d"] = mean
                features[f"{scope}_type_sold_list_count_{window}d"] = count
        output.append(features)
    result = pd.DataFrame(output, columns=FEATURE_COLUMNS)
    if result.shape != (len(q), len(FEATURE_COLUMNS)) or not np.isfinite(result.to_numpy(float)).all():
        raise RuntimeError("R117 as-of feature table has invalid dimensions or values")
    for scope in ("community", "municipality"):
        for window in (90, 365):
            count_column = f"{scope}_type_sold_list_count_{window}d"
            counts = result[count_column].to_numpy(float)
            support[scope][f"{window}d"] = {
                "supported_rows": int(np.count_nonzero(counts)),
                "zero_support_rows": int(np.count_nonzero(counts == 0.0)),
                "max_history_count": int(counts.max(initial=0.0)),
            }
    result.insert(0, "ml_num", q.ml_num.astype(str).to_numpy())
    return result, support


def _install_negotiation_text_features(r22, fit_rows: pd.DataFrame, eval_rows: pd.DataFrame,
                                       feature_rows: pd.DataFrame):
    """Append registered as-of negotiation features and fold-local TF-IDF."""
    from model_candidate_v2 import features as frozen_features

    original = frozen_features.list_aware_features
    fit_identity, eval_identity = id(fit_rows), id(eval_rows)
    support: dict[str, dict[str, int]] = {}
    lookup = feature_rows.set_index(feature_rows.ml_num.astype(str)).drop(columns="ml_num")
    if lookup.index.has_duplicates or set(lookup.columns) != set(FEATURE_COLUMNS):
        raise RuntimeError("R117 feature lookup has duplicate IDs or an unexpected schema")

    def with_negotiation(homes: pd.DataFrame) -> pd.DataFrame:
        result = original(homes)
        collisions = set(FEATURE_COLUMNS).intersection(result.columns)
        if collisions:
            raise RuntimeError(f"R117 features already exist in the frozen builder: {sorted(collisions)}")
        ids = homes.ml_num.astype(str).to_numpy()
        if len(set(ids)) != len(ids) or set(ids) - set(lookup.index):
            raise RuntimeError("R117 feature lookup does not cover the requested rows")
        values = lookup.loc[ids, FEATURE_COLUMNS].copy()
        values.index = result.index
        if not np.isfinite(values.to_numpy(float)).all():
            raise RuntimeError("R117 feature lookup contains non-finite values")
        result = pd.concat([result, values], axis=1)
        key = "training" if id(homes) == fit_identity else "evaluation" if id(homes) == eval_identity else None
        if key is None:
            raise RuntimeError("R117 feature builder received an unregistered frame")
        support[key] = {"rows": int(len(result)), "feature_columns": list(FEATURE_COLUMNS)}
        return result

    frozen_features.list_aware_features = with_negotiation
    try:
        transformer, restore_text = r22._install_text_features(fit_rows, eval_rows, TRACK)
    except Exception:
        frozen_features.list_aware_features = original
        raise

    def restore() -> None:
        restore_text()
        frozen_features.list_aware_features = original

    return transformer, restore, support

def _parent_artifacts(campaign) -> tuple[dict, Path, Path, dict, pd.DataFrame]:
    parent_card = campaign.PROTOCOL["cards"].get(PARENT_CARD)
    if not parent_card or parent_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R117 requires the validated R35 list-aware parent")
    result_path = campaign.ROOT / parent_card["result_path"]
    paired_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, paired_path, manifest_path)):
        raise FileNotFoundError("R35 screen result, predictions, or manifest are missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("card") != PARENT_CARD
            or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or result.get("run_id") != manifest.get("run_id")
            or manifest.get("run_id") != parent_card.get("run_id")):
        raise RuntimeError("R35 screen artifacts do not identify the registered parent")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("source_identity", {}).get("data_sha256")
            != campaign.PROTOCOL["data_sha256"]):
        raise RuntimeError("R35 screen artifacts fail hash or data-snapshot verification")
    if result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened"):
        raise RuntimeError("R35 parent records restricted-label access")
    if result.get("tracks", {}).get(TRACK, {}).get("qualifies_screen") is not True:
        raise RuntimeError("R35 did not pass its registered list-aware screen")
    rows = pd.read_csv(paired_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
    if missing := required - set(rows.columns):
        raise ValueError(f"R35 paired predictions lack columns: {sorted(missing)}")
    rows = rows.loc[rows.track.eq(TRACK) & rows.month.isin(SCREEN_MONTHS)].copy()
    if rows.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R35 screen rows contain duplicate IDs")
    if set(rows.month.unique()) != set(SCREEN_MONTHS):
        raise RuntimeError("R35 screen predictions do not cover both registered months")
    return result, result_path, paired_path, manifest, rows


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered":
            return existing
        raise RuntimeError(f"R117 is already {existing.get('status')}; inspect before rerunning")
    parent = campaign.PROTOCOL["cards"].get(PARENT_CARD)
    if not parent or parent.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R117 parent R35 has not passed fixed-origin confirmation")
    source = budget._source_identity()
    prior_evidence = {}
    for label, relative_path, expected_card in (
        ("R99", R99_RESULT, "R99_asof_community_type_price_area"),
        ("R102", R102_RESULT, "R102_type_market_momentum"),
    ):
        path = campaign.ROOT / relative_path
        result = json.loads(path.read_text(encoding="utf-8"))
        if (result.get("card") != expected_card
                or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
                or result.get("data_sha256") != source["data_sha256"]
                or result.get("reserved_months_labels_opened") is not False
                or result.get("hidden_labels_opened") is not False):
            raise RuntimeError(f"R117 cannot verify safe {label} prior-work evidence")
        prior_evidence[label] = {
            "path": relative_path,
            "sha256": _sha256(path),
            "card": expected_card,
        }
    registration = {
        "status": "registered",
        "track": TRACK,
        "parent": "hash-verified R35 list-aware predictions on identical rows",
        "parent_run_id": parent["run_id"],
        "house_type": HOUSE_TYPE,
        "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS),
        "model_family": "lightgbm",
        "model_config_overrides": MODEL_OVERRIDES,
        "target": "log(sold_price/list_price), unchanged from R35",
        "new_features": FEATURE_COLUMNS,
        "prior_work_evidence": prior_evidence,
        "method": (
            "Keep the validated R35 list-aware parent unchanged outside Detached homes. Fit only a Detached "
            "LightGBM text expert on strictly earlier Detached rows. For each query listing, add the means and "
            "support counts of historical log(sold_price/list_price) across trailing 90-day and 365-day windows, "
            "grouped by community × municipality × house type and municipality × house type. A historical sale "
            "is eligible only when its sale date is strictly earlier than the query listing date and earlier than "
            "the outer fold cutoff. Blend the expert 75/25 with the same frozen HGBR prediction on Detached rows "
            "and overlay it on R35. Preserve all non-Detached R35 predictions, target, other learner settings, "
            "and rounding."
        ),
        "difference_from_prior_work": (
            "R99 and R102 added strictly-as-of historical log(price-per-square-foot) summaries; both failed their "
            "registered screens. R117 instead summarizes the historical log(sold/list) negotiation outcome, which "
            "is directly aligned with the list-aware target, while keeping the R35 Detached-only expert and frozen "
            "HGBR blend. R103 used log(sold/list) as the learner target but did not add past-only local "
            "sale-to-list history features. This is a new feature signal, not a retune of R91's listing-price-per-area."
        ),
        "incremental_screen_gate": campaign.PROTOCOL["gates"],
        "policy": (
            "Use only the two exact safe development-month folds; verify every history sale predates the query "
            "listing and outer cutoff; fit TF-IDF on fold-local training remarks; verify the new features reach "
            "both learner folds and preserve HGBR and R35 row alignment. Retain R35 unless every screen gate "
            "passes. Never read 2025-05/08 or hidden labels."
        ),
        "source_sha256": _sha256(Path(__file__)),
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "track": TRACK,
        "parent": PARENT_CARD, "parent_run_id": parent["run_id"],
        "screen_months": list(SCREEN_MONTHS), "expected_fit_starts": len(SCREEN_MONTHS),
        "model_config_overrides": MODEL_OVERRIDES,
        "new_features": FEATURE_COLUMNS,
        "prior_work_evidence": prior_evidence,
        "data_sha256": source["data_sha256"],
        "source_sha256": registration["source_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r117_registered",
        next_action="run the preregistered two-fold R117 as-of sale-to-list feature screen; keep R35 unless every gate passes",
        r117_card=CARD,
    )
    return registration


def run_feature_preflight() -> tuple[dict, dict[str, pd.DataFrame]]:
    """Build safe, label-free query feature matrices before starting learner fits."""
    from data import hide_answers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R117 feature preflight requires a registered card")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"] or _sha256(Path(__file__)) != card["source_sha256"]:
        raise RuntimeError("R117 source or data identity changed after preregistration")

    # Reuse a verified zero-fit preflight if this screen is resumed.
    relative_result = card.get("feature_preflight_path")
    if relative_result:
        result_path = campaign.ROOT / relative_result
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (_sha256(result_path) != card.get("feature_preflight_sha256")
                or result.get("card") != CARD
                or result.get("data_sha256") != source["data_sha256"]
                or result.get("source_sha256") != card["source_sha256"]
                or result.get("fit_starts") != 0
                or result.get("reserved_months_labels_opened") is not False
                or result.get("hidden_labels_opened") is not False):
            raise RuntimeError("R117 saved feature preflight failed identity or safety verification")
        matrices = {}
        for month, metadata in result["feature_files"].items():
            path = campaign.ROOT / metadata["path"]
            if _sha256(path) != metadata["sha256"]:
                raise RuntimeError(f"R117 saved feature matrix hash mismatch for {month}")
            matrices[month] = pd.read_csv(path, dtype={"ml_num": str})
        return result, matrices

    preflight_id = "r117q_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "preflights" / preflight_id
    output.mkdir(parents=True, exist_ok=False)
    feature_files = {}
    support_by_month = {}
    matrices = {}
    try:
        for month in SCREEN_MONTHS:
            _, train, _, _, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            sold = pd.to_datetime(train.date_sold, errors="raise")
            if (sold >= cutoff).any():
                raise RuntimeError(f"R117 training rows are not strictly before {month}")
            training_months = sold.dt.to_period("M").astype(str)
            if training_months.isin(["2025-05", "2025-08"]).any():
                raise RuntimeError("R117 training fold contains a restricted reserve month")
            homes = hide_answers(heldout, TRACK)
            if {"sold_price", "date_sold"}.intersection(homes.columns):
                raise RuntimeError("R117 preflight query retained post-sale labels")
            fit_rows = train.loc[greedy._specialist_mask(train, HOUSE_TYPE)].copy()
            eval_rows = homes.loc[greedy._specialist_mask(homes, HOUSE_TYPE)].copy()
            if fit_rows.ml_num.astype(str).isin(eval_rows.ml_num.astype(str)).any():
                raise RuntimeError(f"R117 preflight IDs overlap between training and evaluation in {month}")
            queries = pd.concat([fit_rows, eval_rows], ignore_index=True)
            feature_rows, support = _asof_negotiation_features(queries, train, cutoff)
            expected_ids = set(queries.ml_num.astype(str))
            if set(feature_rows.ml_num.astype(str)) != expected_ids:
                raise RuntimeError(f"R117 preflight feature coverage is incomplete in {month}")
            path = output / f"features_{month}.csv"
            feature_rows.to_csv(path, index=False, float_format="%.12g")
            relative_path = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
            feature_files[month] = {"path": relative_path, "sha256": _sha256(path),
                                    "rows": int(len(feature_rows))}
            support_by_month[month] = {
                **support,
                "detached_training_rows": int(len(fit_rows)),
                "detached_evaluation_rows": int(len(eval_rows)),
                "outer_cutoff_exclusive": cutoff.isoformat(),
            }
            matrices[month] = feature_rows
    except Exception as exc:
        failure_path = output / "failure_checkpoint.json"
        _write_json(failure_path, {
            "preflight_id": preflight_id, "card": CARD,
            "error": f"{type(exc).__name__}: {exc}", "fit_starts": 0,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._append({
            "event": "feature_preflight_failed", "card": CARD, "run_id": preflight_id,
            "fit_starts": 0, "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        raise

    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "card": CARD,
        "preflight_id": preflight_id, "screen_months": list(SCREEN_MONTHS),
        "feature_columns": FEATURE_COLUMNS, "feature_files": feature_files,
        "support_by_month": support_by_month, "fit_starts": 0,
        "source_sha256": card["source_sha256"], "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    result_path = output / "preflight.json"
    _write_json(result_path, result)
    relative_result = str(result_path.relative_to(campaign.ROOT)).replace("\\", "/")
    card["feature_preflight_path"] = relative_result
    card["feature_preflight_sha256"] = _sha256(result_path)
    card["feature_preflight_id"] = preflight_id
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "feature_preflight_complete", "card": CARD, "run_id": preflight_id,
        "fit_starts": 0, "feature_preflight_path": relative_result,
        "feature_preflight_sha256": card["feature_preflight_sha256"],
        "feature_files": feature_files, "support_by_month": support_by_month,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r117_preflight_complete",
        next_action="review the zero-fit R117 as-of feature preflight, then run the frozen two-fold screen",
        r117_preflight_id=preflight_id, r117_preflight_path=relative_result,
    )
    return result, matrices


def run_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as scoring
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R117 must be preregistered before fitting")
    source = budget._source_identity()
    if (source["data_sha256"] != card["data_sha256"]
            or _sha256(Path(__file__)) != card["source_sha256"]):
        raise RuntimeError("R117 source or data snapshot changed after preregistration")
    parent_result, parent_path, parent_rows_path, parent_manifest, parent_rows = _parent_artifacts(campaign)
    if parent_result["run_id"] != card["parent_run_id"]:
        raise RuntimeError("R117 parent run differs from preregistration")
    preflight, preflight_matrices = run_feature_preflight()

    run_id = "r117p_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(parent_path, provenance / "r35_screen_results.json")
    shutil.copy2(parent_rows_path, provenance / "r35_screen_paired_predictions.csv")
    shutil.copy2(parent_path.parent / "run_manifest.json", provenance / "r35_screen_run_manifest.json")
    preflight_path = campaign.ROOT / card["feature_preflight_path"]
    shutil.copy2(preflight_path, provenance / "r117_feature_preflight.json")
    for month, metadata in preflight["feature_files"].items():
        shutil.copy2(campaign.ROOT / metadata["path"], provenance / f"r117_features_{month}.csv")

    plan = [(month, "detached_expert") for month in SCREEN_MONTHS]
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id, "track": TRACK,
        "parent_candidate": parent_result["run_id"], "screen_months": list(SCREEN_MONTHS),
        "model_family": "lightgbm", "model_config_overrides": MODEL_OVERRIDES,
        "expected_fit_starts": len(plan), "data_sha256": source["data_sha256"],
        "parent_result_sha256": _sha256(parent_path),
        "parent_predictions_sha256": _sha256(parent_rows_path),
        "feature_preflight_path": card["feature_preflight_path"],
        "feature_preflight_sha256": card["feature_preflight_sha256"],
        "feature_files": preflight["feature_files"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = {"model_config_overrides": MODEL_OVERRIDES}
    fit_details: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    active: dict[str, object] = {}
    fit_start_count = 0

    def record_fit(_parent_fit_id: str, fit_fn):
        nonlocal fit_start_count
        month = str(active["month"])
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:screen:{TRACK}:{month}:detached_expert:{run_id}"
        fit_start_count += 1
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": "screen", "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": "detached_expert", "training_rows": active["training_rows"],
            "evaluation_rows": active["evaluation_rows"], "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            if fitted.get("target") != "log_sold_over_list":
                raise RuntimeError(f"R117 expected log_sold_over_list; got {fitted.get('target')!r}")
            missing_features = set(FEATURE_COLUMNS) - set(fitted.get("feature_columns", []))
            if missing_features:
                raise RuntimeError(f"R117 features did not reach the learner: {sorted(missing_features)}")
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": "screen", "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month, "status": "failed",
                "seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": "detached_expert", "seconds": time.perf_counter() - started,
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "actual_objective": fitted["actual_objective"], "target": fitted["target"],
            "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
            "feature_columns": fitted["feature_columns"],
            "prediction_sha256": fitted["prediction_sha256"],
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "stage": "screen", "run_id": run_id,
            **detail, "status": "complete", "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R117 training rows are not strictly before {month}")
            training_months = pd.to_datetime(train.date_sold, errors="raise").dt.to_period("M").astype(str)
            if training_months.isin(["2025-05", "2025-08"]).any():
                raise RuntimeError("R117 training fold contains a restricted reserve month")
            homes = hide_answers(heldout, TRACK)
            detached_mask = greedy._specialist_mask(homes, HOUSE_TYPE)
            fit_mask = greedy._specialist_mask(train, HOUSE_TYPE)
            fit_rows = train.loc[fit_mask].copy()
            eval_rows = homes.loc[detached_mask].copy()
            if len(fit_rows) < 500 or len(eval_rows) < 100 or not (~detached_mask).any():
                raise RuntimeError(f"R117 Detached route support is too small in {month}")

            saved = parent_rows.loc[parent_rows.month.eq(month)].copy()
            saved = saved.set_index(saved.ml_num.astype(str)).reindex(homes.ml_num.astype(str))
            if (saved.index.has_duplicates
                    or not np.array_equal(saved.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
                    or not np.isfinite(saved.candidate_prediction.to_numpy(float)).all()):
                raise RuntimeError(f"R35 parent predictions do not align for {month}")
            parent_prediction = saved.candidate_prediction.to_numpy(float)

            feature_rows = preflight_matrices[month]
            asof_support = preflight["support_by_month"][month]
            transformer, restore, frame_support = _install_negotiation_text_features(
                r22, fit_rows, eval_rows, feature_rows,
            )
            feature_support = {"asof_history": asof_support, "model_frames": frame_support}
            active.update(month=month, training_rows=len(fit_rows), evaluation_rows=len(eval_rows))
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows, fit_rows=fit_rows, variant=VARIANT,
                    base_variant=None, fit_tag=f"R117-{month}-Detached-asof-sold-list",
                )
            finally:
                restore()
            if set(frame_support) != {"training", "evaluation"}:
                raise RuntimeError(f"R117 feature coverage was not recorded for {month}")
            if set(FEATURE_COLUMNS) - set(fitted.get("feature_columns", [])):
                raise RuntimeError("R117 as-of sale-to-list features did not reach the learner")
            frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, TRACK, train, homes)
            specialist_prediction = blend_prices(
                fitted["predictions"], frozen_hgbr[detached_mask], challenger_weight=CORE_WEIGHT,
            )
            prediction = parent_prediction.copy()
            prediction[detached_mask] = specialist_prediction
            if not np.array_equal(prediction[~detached_mask], parent_prediction[~detached_mask]):
                raise RuntimeError(f"R117 changed non-Detached R35 predictions in {month}")
            parent_frame = pd.DataFrame({
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "predicted_price": parent_prediction,
            })
            candidate_frame = pd.DataFrame({
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "predicted_price": prediction,
            })
            paired_parts.append(helpers._candidate_frame(month, TRACK, truth, parent_frame, candidate_frame))
            actual = truth.sold_price.to_numpy(float)
            route_details[month] = {
                "training_rows": int(len(fit_rows)), "evaluation_rows": int(len(homes)),
                "detached_training_rows": int(len(fit_rows)),
                "detached_evaluation_rows": int(detached_mask.sum()),
                "non_detached_predictions_unchanged": True,
                "feature_support": feature_support,
                "parent_all": campaign._metrics(actual, parent_prediction),
                "candidate_all": campaign._metrics(actual, prediction),
                "parent_detached": campaign._metrics(actual[detached_mask], parent_prediction[detached_mask]),
                "candidate_detached": campaign._metrics(actual[detached_mask], prediction[detached_mask]),
                "feature_columns": fitted["feature_columns"],
                "tfidf_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                "actual_objective": fitted["actual_objective"], "target": fitted["target"],
            }
            campaign._append({
                "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                "track": TRACK, "month": month, "training_cutoff_exclusive": cutoff.isoformat(),
                "training_rows": int(len(fit_rows)), "evaluation_rows": int(detached_mask.sum()),
                "new_features": FEATURE_COLUMNS,
                "non_detached_predictions_unchanged": True,
                "feature_support": feature_support,
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
    except Exception as exc:
        failure_path = output / "failure_checkpoint.json"
        _write_json(failure_path, {
            "run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": fit_start_count,
            "fit_details": fit_details, "completed_months": list(route_details),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "fit_starts": len(fit_details), "error": f"{type(exc).__name__}: {exc}"})
        campaign._write_progress("r117_screen_failed",
                                 next_action="inspect R117 failure checkpoint and ledger before any retry",
                                 r117_failure=str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if fit_start_count != len(plan) or len(fit_details) != len(plan):
        raise RuntimeError(
            f"R117 expected {len(plan)} completed fits; starts={fit_start_count}, completed={len(fit_details)}"
        )
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = scoring._track_result(paired)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "parent_candidate": parent_result["run_id"], "track": TRACK,
        "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
        "tracks": {TRACK: score}, "fit_details": fit_details,
        "route_diagnostics": route_details, "model_config_overrides": MODEL_OVERRIDES,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False, "data_sha256": source["data_sha256"],
    }
    result_path = output / "screen_results.json"
    _write_json(result_path, result)
    source_files = _source_files(campaign)
    code_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_files
    }
    for index, path in enumerate(source_files):
        shutil.copy2(path, provenance / f"src_{index:02d}_{path.name}.source")
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "data_sha256": source["data_sha256"], "fit_starts": len(fit_details),
        "fit_details": fit_details, "result_sha256": _sha256(result_path),
        "paired_predictions_sha256": _sha256(paired_path),
        "parent_run_id": parent_result["run_id"],
        "parent_result_sha256": _sha256(parent_path),
        "parent_predictions_sha256": _sha256(parent_rows_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = [
        "# R117: Detached-only as-of historical sale-to-list text expert screen", "",
        f"Run: `{run_id}`; parent: validated R35 `{parent_result['run_id']}`.",
        "The only new inputs are strict-past means and support counts for historical log(sold_price/list_price) "
        "over 90- and 365-day windows, grouped by community/type and municipality/type. The model keeps the "
        "unchanged log(sold/list) target, fold-local word TF-IDF, LightGBM settings, strict outer cutoff, and "
        "75/25 LightGBM/frozen-HGBR blend. The feature-augmented expert replaces predictions only for Detached "
        "rows; every non-Detached R35 prediction is copied exactly.", "",
        f"- R35 screen MdAPE: {score['parent']['mdape_percent']:.5f}%",
        f"- R117 screen MdAPE: {score['candidate']['mdape_percent']:.5f}%",
        f"- Relative MdAPE improvement: {score['relative_mdape_improvement_percent']:+.3f}%",
        f"- Relative MAPE change: {score['mape_relative_change_percent']:+.3f}%",
        f"- Screen passed: {score['qualifies_screen']}", "",
        "## Monthly results", "", "| Month | R35 MdAPE | R117 MdAPE | Relative improvement |",
        "|---|---:|---:|---:|",
    ]
    for month, values in score["monthly"].items():
        report.append(f"| {month} | {values['parent']['mdape_percent']:.5f}% | "
                      f"{values['candidate']['mdape_percent']:.5f}% | "
                      f"{values['relative_mdape_improvement_percent']:+.3f}% |")
    report.extend(["", "Restricted reserve and hidden labels were not read. A screen failure ends this hypothesis; "
                   "do not tune the feature or model on these months.", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    card.update({
        "status": "screen_passed" if score["qualifies_screen"] else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        "screen_metrics": score, "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details), "result_path": card["result_path"],
        "qualifies_screen": score["qualifies_screen"], "screen_metrics": score,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    next_action = (
        "R117 passed its fixed screen; run an eight-month confirmation against R35 while preserving non-Detached predictions"
        if score["qualifies_screen"] else
        "R117 failed its fixed screen; keep R35 and register a distinct list-aware hypothesis"
    )
    campaign._write_progress(
        "r117_screen_passed" if score["qualifies_screen"] else "r117_screen_failed",
        next_action=next_action,
        r117_run_id=run_id, r117_result=card["result_path"],
        r117_fit_starts=len(fit_details),
    )
    return result


def run() -> dict:
    register_card()
    return run_screen()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true",
                        help="record the immutable experiment card without starting learner fits")
    parser.add_argument("--preflight-only", action="store_true",
                        help="register and generate safe feature matrices without starting learner fits")
    args = parser.parse_args()
    if args.register_only and args.preflight_only:
        parser.error("choose at most one of --register-only and --preflight-only")
    if args.register_only:
        result = register_card()
    elif args.preflight_only:
        register_card()
        result, _ = run_feature_preflight()
    else:
        result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
