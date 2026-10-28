"""Run R46's preregistered eight-month, past-only development confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as catboost_helpers
from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import r22_remarks_tfidf as scoring
from experiments.bill_target74 import r40_past_oof_r26_blend as weight_selection
from experiments.bill_target74 import r43_listaware_condo_apt_text as parent_verifier
from experiments.bill_target74 import r44_past_oof_r35_shrinkage as dates_helper
from experiments.bill_target74 import r46_catboost_r35_oof_blend as r46
from experiments.bill_target74 import r45_listaware_detached_catboost as r45
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy


CARD = r46.CARD
TRACK = r46.TRACK
NEW_MONTHS = ("2024-09", "2025-01", "2025-03")
DEVELOPMENT_CARD_KEY = "development_confirmation"
LIMIT_KEYS = (
    "campaign_fit_starts", "campaign_minutes", "exploration_fit_starts",
    "exploration_minutes", "final_reserve_fit_starts", "final_reserve_minutes",
    "batch_fit_starts", "batch_minutes", "single_fit_seconds",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _verify_r46_inputs() -> tuple[dict, dict, pd.DataFrame, Path, Path, Path, Path]:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "screen_passed" or not card.get("qualifies_screen"):
        raise RuntimeError("R46 development confirmation requires its passed screen")
    if any(campaign.PROTOCOL["limits"].get(key) is not None for key in LIMIT_KEYS):
        raise RuntimeError("active campaign limits changed from the registered unbounded policy")
    dev = card.get(DEVELOPMENT_CARD_KEY, {})
    expected_months = tuple(dev.get("development_months", ()))
    if len(expected_months) != 8 or set(expected_months).intersection({"2025-05", "2025-08"}):
        raise RuntimeError("R46 development months are missing or include an ineligible reserve month")
    if set(NEW_MONTHS).difference(expected_months):
        raise RuntimeError("R46 new OOF months are not part of its preregistered development window")
    if int(dev.get("expected_new_fit_starts", -1)) != len(NEW_MONTHS):
        raise RuntimeError("R46 development fit count differs from the preregistered plan")

    parent = parent_verifier._verified_r35_parent(campaign)
    result_path = campaign.ROOT / card["result_path"]
    rows_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    oof_path = result_path.parent / "past_oof_predictions.csv"
    if not all(path.is_file() for path in (result_path, rows_path, manifest_path, oof_path)):
        raise FileNotFoundError("R46 screen result, rows, OOF rows, or manifest is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("run_id") != card.get("run_id")
            or result.get("fit_starts") != 5
            or result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result.get("reserved_months_labels_opened")
            or result.get("hidden_labels_opened")):
        raise RuntimeError("R46 screen result identity or label-safety fields failed validation")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(rows_path)
            or manifest.get("past_oof_sha256") != _sha256(oof_path)
            or manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or manifest.get("fit_starts") != 5
            or manifest.get("reserved_months_labels_opened")
            or manifest.get("hidden_labels_opened")):
        raise RuntimeError("R46 screen manifest or saved artifact hashes failed validation")
    parent_hashes = manifest.get("parent_artifact_sha256", {})
    expected_parent_hashes = {
        "r35_screen_result_sha256": _sha256(parent["screen_path"]),
        "r35_screen_rows_sha256": _sha256(parent["screen_rows_path"]),
        "r35_development_result_sha256": _sha256(parent["development_path"]),
        "r35_development_rows_sha256": _sha256(parent["development_rows_path"]),
        "r35_fixed_result_sha256": _sha256(parent["fixed_path"]),
        "r35_fixed_rows_sha256": _sha256(parent["fixed_path"].parent / "paired_predictions.csv"),
    }
    if any(parent_hashes.get(key) != value for key, value in expected_parent_hashes.items()):
        raise RuntimeError("R46 screen no longer matches the hash-verified R35 parent")
    snapshots = manifest.get("source_snapshots", {})
    for source_key, snapshot_name in snapshots.items():
        snapshot_path = result_path.parent / "provenance" / snapshot_name
        if (not snapshot_path.is_file()
                or manifest.get("code_sha256", {}).get(source_key) != _sha256(snapshot_path)):
            raise RuntimeError(f"R46 source snapshot failed verification: {source_key}")

    oof = pd.read_csv(oof_path, dtype={"ml_num": str})
    if (oof.duplicated(["month", "ml_num"]).any()
            or set(oof.month.astype(str)) != set(r46.PAST_SOURCE_MONTHS)
            or set(oof.month.astype(str)).intersection({"2025-05", "2025-08"})
            or not pd.to_datetime(oof.date_sold, errors="raise").dt.to_period("M").astype(str).eq(
                oof.month.astype(str)
            ).all()):
        raise RuntimeError("R46 saved OOF rows are duplicated or outside registered safe months")
    values = oof[["sold_price", "r35_prediction", "catboost_prediction"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise RuntimeError("R46 saved OOF prices are not finite and positive")

    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("current source identity differs from R46's data snapshot")
    if metadata.version("catboost") != card.get("catboost_version"):
        raise RuntimeError("CatBoost version differs from R46 preregistration")
    return parent, card, oof, result_path, rows_path, manifest_path, oof_path


def _preflight() -> dict:
    from data import hide_answers

    parent, card, oof, result_path, rows_path, manifest_path, oof_path = _verify_r46_inputs()
    dev = card[DEVELOPMENT_CARD_KEY]
    dev_months = tuple(dev["development_months"])
    r35_dev = pd.read_csv(parent["development_rows_path"], dtype={"ml_num": str})
    r35_dev = r35_dev.loc[r35_dev.track.eq(TRACK)].copy()
    r35_dev["month"] = r35_dev.month.astype(str)
    r35_dev = r35_dev.loc[r35_dev.month.isin(dev_months)].copy()
    if (set(r35_dev.month.astype(str)) != set(dev_months)
            or r35_dev.duplicated(["month", "ml_num"]).any()):
        raise RuntimeError("R35 development rows do not exactly cover the registered R46 months")

    metadata_rows = dates_helper._safe_date_metadata(NEW_MONTHS, campaign.DATA)
    preflight = {}
    for month in NEW_MONTHS:
        _, train, _, truth, heldout = greedy._load_fold(month)
        cutoff = pd.Period(month, freq="M").start_time
        if pd.to_datetime(train.date_sold, errors="raise").ge(cutoff).any():
            raise RuntimeError(f"R46 {month} training rows reach the evaluation cutoff")
        homes = hide_answers(heldout, TRACK).reset_index(drop=True)
        ids = homes.ml_num.astype(str).to_numpy()
        if (not np.array_equal(ids, truth.ml_num.astype(str).to_numpy())
                or not pd.to_datetime(heldout.date_sold, errors="raise").dt.to_period("M").astype(str).eq(month).all()):
            raise RuntimeError(f"R46 {month} fold rows do not align to the registered sale month")
        route_train = greedy._specialist_mask(train, r46.HOUSE_TYPE)
        route_eval = greedy._specialist_mask(homes, r46.HOUSE_TYPE)

        parent_rows = r35_dev.loc[r35_dev.month.eq(month)].copy()
        if (len(parent_rows) != len(ids)
                or parent_rows.ml_num.astype(str).duplicated().any()
                or set(parent_rows.ml_num.astype(str)) != set(ids)):
            raise RuntimeError(f"R35 parent rows do not align to the {month} development fold")
        parent_rows = parent_rows.set_index(parent_rows.ml_num.astype(str)).loc[ids]
        if not np.array_equal(parent_rows.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
            raise RuntimeError(f"R35 labels do not align to the {month} development fold")

        cache_path = greedy.OUTPUT / "frozen_hgbr" / TRACK / f"{month}.csv"
        if not cache_path.is_file():
            raise FileNotFoundError(f"R46 frozen HGBR cache is missing for {month}: {cache_path}")
        cache_prediction = catboost_helpers._verified_hgbr_cache(
            greedy, month, TRACK, train, homes,
        )
        if cache_prediction.shape != (len(homes),):
            raise RuntimeError(f"R46 HGBR predictions are not row-aligned for {month}")
        support = {
            "training_rows": int(route_train.sum()),
            "evaluation_rows": int(route_eval.sum()),
            "all_evaluation_rows": int(len(homes)),
        }
        preflight[month] = {
            "support": support,
            "training_rows_all": int(len(train)),
            "r35_development_rows": int(len(parent_rows)),
            "hgb_cache_path": str(cache_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "hgb_cache_sha256": r46._sha256(cache_path),
            "max_training_date": str(pd.to_datetime(train.date_sold, errors="raise").max()),
            "min_evaluation_date": str(pd.to_datetime(heldout.date_sold, errors="raise").min()),
            "date_metadata_rows": int(metadata_rows.month.astype(str).eq(month).sum()),
            "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        }

    return {
        "campaign_id": campaign.PROTOCOL["campaign_id"],
        "card": CARD,
        "screen_run_id": card["run_id"],
        "screen_result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "screen_result_sha256": r46._sha256(result_path),
        "screen_rows_sha256": r46._sha256(rows_path),
        "screen_manifest_sha256": r46._sha256(manifest_path),
        "screen_oof_sha256": r46._sha256(oof_path),
        "parent_development_rows_sha256": r46._sha256(parent["development_rows_path"]),
        "development_months": list(dev_months),
        "new_fit_months": list(NEW_MONTHS),
        "expected_fit_starts": len(NEW_MONTHS),
        "support_and_cache": preflight,
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "catboost_version": metadata.version("catboost"),
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }


def preflight() -> dict:
    result = _preflight()
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def register_stage() -> dict:
    details = _preflight()
    card = campaign.PROTOCOL["cards"][CARD]
    dev = card[DEVELOPMENT_CARD_KEY]
    if dev.get("status") not in {None, "pending", "preflight_complete"}:
        raise RuntimeError(f"R46 development stage is already {dev.get('status')!r}")
    run_id = "bill_target74_r46_dev_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dev.update({
        "status": "registered",
        "run_id": run_id,
        "preflight": details,
        "support": {month: row["support"] for month, row in details["support_and_cache"].items()},
        "hgbr_cache_paths": {
            month: row["hgb_cache_path"] for month, row in details["support_and_cache"].items()
        },
        "hgbr_cache_sha256": {
            month: row["hgb_cache_sha256"] for month, row in details["support_and_cache"].items()
        },
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                      encoding="utf-8")
    campaign._append({
        "event": "card_stage_registered", "card": CARD,
        "stage": "development_confirmation", "run_id": run_id,
        "fit_starts_expected": len(NEW_MONTHS), "fit_starts_completed": 0,
        "months": list(NEW_MONTHS), "support_and_cache": details["support_and_cache"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r46_development_registered",
        next_action="Run R46's registered three-fold development confirmation",
        r46_development_run_id=run_id,
    )
    print(json.dumps({"run_id": run_id, "registered": True,
                      "expected_fit_starts": len(NEW_MONTHS)}, indent=2))
    return dev


def _new_month_references(parent: dict, months: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    development = pd.read_csv(parent["development_rows_path"], dtype={"ml_num": str})
    development = development.loc[
        development.track.eq(TRACK) & development.month.astype(str).isin(months)
    ].copy()
    development["month"] = development.month.astype(str)
    development = development[["month", "ml_num", "sold_price", "candidate_prediction"]]
    development = development.rename(columns={"candidate_prediction": "r35_prediction"})
    metadata_rows = dates_helper._safe_date_metadata(months, campaign.DATA)
    development = dates_helper._attach_oof_dates(development, metadata_rows)
    return {month: development.loc[development.month.eq(month)].copy() for month in months}


def _write_provenance(output: Path, parent: dict, screen_paths: tuple[Path, ...]) -> dict:
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(r46.__file__), Path(weight_selection.__file__),
               Path(scoring.__file__), Path(campaign.__file__), Path(greedy.__file__),
               Path(dates_helper.__file__), Path(parent_verifier.__file__)]
    copied_sources = {}
    source_hashes = {}
    for index, source in enumerate(sources):
        key = str(source.relative_to(campaign.ROOT)).replace("\\", "/")
        snapshot_name = f"src_{index:02d}{source.suffix}.source"
        shutil.copy2(source, provenance / snapshot_name)
        copied_sources[key] = snapshot_name
        source_hashes[key] = _sha256(source)
    inputs = {
        "r35_development_result": parent["development_path"],
        "r35_development_rows": parent["development_rows_path"],
        "r46_screen_result": screen_paths[0],
        "r46_screen_rows": screen_paths[1],
        "r46_screen_manifest": screen_paths[2],
        "r46_past_oof": screen_paths[3],
    }
    input_hashes = {name: _sha256(path) for name, path in inputs.items()}
    return {"source_paths": copied_sources, "source_sha256": source_hashes,
            "parent_input_sha256": input_hashes}


def run() -> dict:
    from data import hide_answers

    preflight_details = _preflight()
    parent, card, base_oof, result_path, rows_path, manifest_path, oof_path = _verify_r46_inputs()
    dev = card[DEVELOPMENT_CARD_KEY]
    if dev.get("status") != "registered" or not dev.get("run_id"):
        raise RuntimeError("R46 development confirmation must be preregistered before fitting")
    if dev.get("preflight") != preflight_details:
        raise RuntimeError("R46 development support or artifact hashes changed after preregistration")
    run_id = dev["run_id"]
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    source = budget._source_identity()
    provenance_details = _write_provenance(
        output, parent, (result_path, rows_path, manifest_path, oof_path),
    )
    fit_details: list[dict] = []
    fit_start_ids: list[str] = []
    runtime_card = dict(card)
    runtime_card["past_oof_support"] = dev["support"]
    runtime_card["hgbr_cache_paths"] = dev["hgbr_cache_paths"]
    runtime_card["hgbr_cache_sha256"] = dev["hgbr_cache_sha256"]
    references = _new_month_references(parent, NEW_MONTHS)
    r44_root = result_path.parent
    campaign._append({
        "event": "card_start", "card": CARD, "stage": "development_confirmation",
        "run_id": run_id, "track": TRACK,
        "parent_run_id": card["parent_run_id"],
        "development_months": dev["development_months"],
        "new_fit_months": list(NEW_MONTHS), "expected_fit_starts": len(NEW_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r46_development_running",
        next_action="Complete registered R46 development CatBoost folds",
        r46_development_run_id=run_id,
    )
    try:
        new_oof = []
        for month in NEW_MONTHS:
            new_oof.append(r46._fit_catboost_oof(
                month, references[month], runtime_card, run_id, output, source,
                fit_details, fit_start_ids, r44_root,
                fit_stage="development_confirmation",
            ))
        all_oof = pd.concat([base_oof, *new_oof], ignore_index=True)
        all_oof["month"] = all_oof.month.astype(str)
        all_oof["ml_num"] = all_oof.ml_num.astype(str)
        if (all_oof.duplicated(["month", "ml_num"]).any()
                or set(all_oof.month.astype(str)) != (set(r46.PAST_SOURCE_MONTHS) | set(NEW_MONTHS))):
            raise RuntimeError("R46 combined OOF rows do not have the exact registered safe-month coverage")
        all_oof.to_csv(output / "development_oof_predictions.csv", index=False)

        r35_dev = pd.read_csv(parent["development_rows_path"], dtype={"ml_num": str})
        r35_dev = r35_dev.loc[
            r35_dev.track.eq(TRACK)
            & r35_dev.month.astype(str).isin(dev["development_months"])
        ].copy()
        r35_dev["month"] = r35_dev.month.astype(str)
        r35_dev["ml_num"] = r35_dev.ml_num.astype(str)
        paired_rows = []
        selections = {}
        oof_for_selection = all_oof.rename(columns={
            "r35_prediction": "parent_prediction",
            "catboost_prediction": "variant_prediction",
        })
        for month in dev["development_months"]:
            month = str(month)
            expected = r35_dev.loc[r35_dev.month.eq(month)].copy()
            observed = all_oof.loc[all_oof.month.eq(month)].copy()
            if (expected.empty or observed.empty
                    or expected.ml_num.duplicated().any()
                    or observed.ml_num.duplicated().any()
                    or set(expected.ml_num) != set(observed.ml_num)):
                raise RuntimeError(f"R46 OOF IDs do not exactly match R35 development rows for {month}")
            observed = observed.set_index("ml_num").loc[expected.ml_num.to_list()].reset_index()
            if (not np.array_equal(observed.sold_price.to_numpy(float), expected.sold_price.to_numpy(float))
                    or not np.array_equal(observed.r35_prediction.to_numpy(float),
                                          expected.candidate_prediction.to_numpy(float))):
                raise RuntimeError(f"R46 OOF labels or R35 parent predictions changed for {month}")
            cutoff = pd.Period(month, freq="M").start_time
            past_rows = oof_for_selection.loc[
                pd.to_datetime(oof_for_selection.date_sold, errors="raise").lt(cutoff)
            ]
            picked = weight_selection.select_weight(
                oof_for_selection, cutoff,
                minimum_rows=weight_selection.MINIMUM_OOF_ROWS,
                max_mape_regression_percent=1.0,
            )
            if (picked["reason"] != "selected_from_strictly_past_oof"
                    or len(past_rows) != picked["oof_rows"]
                    or (len(past_rows) and not pd.to_datetime(past_rows.date_sold).lt(cutoff).all())):
                raise RuntimeError(f"R46 could not select a strictly past-only weight for {month}")
            if month in card["selected_catboost_share_by_screen_month"]:
                expected_weight = float(card["selected_catboost_share_by_screen_month"][month])
                if not np.isclose(float(picked["weight"]), expected_weight, atol=0.0, rtol=0.0):
                    raise RuntimeError(f"R46 recreated a different screen-month weight for {month}")
            selections[month] = picked
            prediction = weight_selection.blend(
                observed.r35_prediction.to_numpy(float),
                observed.catboost_prediction.to_numpy(float),
                float(picked["weight"]),
            )
            paired_rows.append(pd.DataFrame({
                "month": month,
                "ml_num": expected.ml_num.to_numpy(str),
                "date_sold": observed.date_sold.to_numpy(),
                "sold_price": expected.sold_price.to_numpy(float),
                "parent_prediction": observed.r35_prediction.to_numpy(float),
                "catboost_prediction": observed.catboost_prediction.to_numpy(float),
                "selected_catboost_share": float(picked["weight"]),
                "candidate_prediction": prediction,
            }))
        paired = pd.concat(paired_rows, ignore_index=True)
        if set(paired.month.astype(str)) != set(dev["development_months"]):
            raise RuntimeError("R46 final paired rows do not cover exactly the registered development window")
        paired.to_csv(output / "paired_predictions.csv", index=False)
        metrics = scoring.score_development_confirmation(
            paired.rename(columns={"parent_prediction": "parent_prediction",
                                   "candidate_prediction": "candidate_prediction"}),
            months=tuple(dev["development_months"]),
            target_mdape_strictly_below=float(dev["target_mdape_strictly_below"]),
            bootstrap_replicates=int(dev["bootstrap_replicates"]),
            bootstrap_seed=int(dev["bootstrap_seed"]),
            minimum_relative_gain_percent=float(dev["relative_mdape_improvement_percent_min"]),
            minimum_months_improved=int(dev["months_improved_min"]),
            maximum_worst_month_regression_percent=float(dev["worst_month_regression_max_percent"]),
            maximum_mape_regression_percent=float(dev["mape_regression_max_percent"]),
        )
        qualifies = bool(metrics["qualifies_development"] and metrics["target_met"])
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"],
            "card": CARD, "stage": "development_confirmation", "run_id": run_id,
            "track": TRACK, "parent_run_id": card["parent_run_id"],
            "screen_run_id": card["run_id"],
            "development_months": list(dev["development_months"]),
            "new_fit_months": list(NEW_MONTHS), "fit_starts": len(fit_details),
            "fit_details": fit_details, "selected_catboost_share_by_month": selections,
            "metrics": metrics, "qualifies_development": qualifies,
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        result_path_out = output / "development_results.json"
        _write_json(result_path_out, result)
        paired_path = output / "paired_predictions.csv"
        oof_path_out = output / "development_oof_predictions.csv"
        source_paths = provenance_details["source_paths"]
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"],
            "run_id": run_id, "source_identity": source,
            "source_snapshots": source_paths,
            "code_sha256": provenance_details["source_sha256"],
            "parent_input_sha256": provenance_details["parent_input_sha256"],
            "preflight": preflight_details,
            "new_oof_sha256": r46._sha256(oof_path_out),
            "paired_predictions_sha256": r46._sha256(paired_path),
            "result_sha256": r46._sha256(result_path_out),
            "catboost_version": metadata.version("catboost"),
            "catboost_model_config": dict(r46.r03.CATBOOST_CONFIG),
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        _write_json(output / "run_manifest.json", manifest)
        report = [
            "# R46 八个月开发确认：R35 与 CatBoost Detached 专家的过去 OOF 融合", "",
            f"Run：`{run_id}`；父模型 R35：`{card['parent_run_id']}`。",
            "仅对 list-aware Detached 路由融合 CatBoost 与 R35 候选。每个月的融合权重只用该月开始日前的 OOF 标签选择；其他房型沿用 R35。",
            f"新增 learner fits：{len(fit_details)}；开发门槛通过：{qualifies}。",
            "",
            f"Pooled MdAPE：{metrics['parent']['mdape_percent']:.5f}% → "
            f"{metrics['candidate']['mdape_percent']:.5f}%；相对改善 "
            f"{metrics['relative_mdape_improvement_percent']:+.3f}%；MAPE 相对变化 "
            f"{metrics['mape_relative_change_percent']:+.3f}%；"
            f"{metrics['months_improved']}/8 月改善；bootstrap 上界 "
            f"{metrics['bootstrap_mdape_difference_percent']['upper95']:+.5f} 个百分点；"
            f"目标 <4%：{metrics['target_met']}。",
            "",
            "## 逐月结果与严格过去 OOF 权重", "",
            "| 月份 | past-OOF 月份/行数 | CatBoost share | R35 MdAPE | 候选 MdAPE | 相对改善 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for month in dev["development_months"]:
            month = str(month)
            row = metrics["monthly"][month]
            picked = selections[month]
            report.append(
                f"| {month} | {','.join(picked['oof_months'])} / {picked['oof_rows']} | "
                f"{picked['weight']:.2f} | {row['parent']['mdape_percent']:.5f}% | "
                f"{row['candidate']['mdape_percent']:.5f}% | "
                f"{row['relative_mdape_improvement_percent']:+.3f}% |"
            )
        report.extend([
            "", "2025-05/08 保留月份不具备独立验证资格，本轮未读取；隐藏标签未读取。",
            "只有所有预登记开发门槛通过，才运行固定起点确认；未通过则保留 R35。", "",
        ])
        (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

        current = campaign.PROTOCOL["cards"][CARD]
        stage = current[DEVELOPMENT_CARD_KEY]
        stage.update({
            "status": "development_passed" if qualifies else "development_failed",
            "fit_starts": len(fit_details),
            "result_path": str(result_path_out.relative_to(campaign.ROOT)).replace("\\", "/"),
            "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
            "metrics": metrics,
            "qualifies_development": qualifies,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        current["status"] = "development_passed" if qualifies else "development_failed"
        current["development_result_path"] = stage["result_path"]
        current["development_qualifies"] = qualifies
        campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                          encoding="utf-8")
        campaign._append({
            "event": "card_complete", "card": CARD,
            "stage": "development_confirmation", "run_id": run_id,
            "fit_starts": len(fit_details), "result_path": stage["result_path"],
            "qualifies_by_track": {TRACK: qualifies},
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        if qualifies:
            next_action = "R46 passed eight-month development confirmation; register and run fixed-origin confirmation"
            progress_status = "r46_development_passed"
        else:
            next_action = "R46 failed development confirmation; preserve R35 and register a distinct hypothesis"
            progress_status = "r46_development_failed"
        campaign._write_progress(
            progress_status, next_action=next_action,
            r46_development_result=stage["result_path"],
            r46_development_fit_starts=len(fit_details),
        )
        print(
            f"R46 development: parent_mdape={metrics['parent']['mdape_percent']:.5f}% "
            f"candidate_mdape={metrics['candidate']['mdape_percent']:.5f}% "
            f"relative_gain={metrics['relative_mdape_improvement_percent']:+.3f}% "
            f"months_improved={metrics['months_improved']}/8 "
            f"bootstrap_upper95={metrics['bootstrap_mdape_difference_percent']['upper95']:+.5f} "
            f"target_met={metrics['target_met']} pass={qualifies} run={run_id}",
            flush=True,
        )
        return result
    except Exception as exc:
        failure = {
            "run_id": run_id, "card": CARD, "stage": "development_confirmation",
            "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_start_ids), "fit_completions": len(fit_details),
            "fit_ids": fit_start_ids, "fit_details": fit_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        _write_json(output / "failure_checkpoint.json", failure)
        stage = campaign.PROTOCOL["cards"][CARD][DEVELOPMENT_CARD_KEY]
        stage.update({"status": "failed", "fit_starts": len(fit_start_ids),
                      "failure_path": str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
                      "error": failure["error"]})
        campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                          encoding="utf-8")
        campaign._append({
            "event": "card_failed", "card": CARD,
            "stage": "development_confirmation", "run_id": run_id,
            "fit_starts": failure["fit_starts"],
            "fit_completions": failure["fit_completions"],
            "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r46_development_failed",
            next_action="Inspect R46 development failure checkpoint and fit ledger before retrying",
            r46_development_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--register", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.register:
        register_stage()
    else:
        run()
