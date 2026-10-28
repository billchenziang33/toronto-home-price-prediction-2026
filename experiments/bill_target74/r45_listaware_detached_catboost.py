"""R45: replace only R35's Detached list-aware text expert with CatBoost."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd


CARD = "R45_listaware_detached_catboost_r22_text_expert"
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
COMPONENT = "detached_catboost_core_expert"
VARIANT = CARD
SCREEN_MONTHS = ("2023-03", "2024-06")
SPECIALIST_WEIGHT = 0.75
MIN_TRAINING_ROWS = 500
MIN_EVALUATION_ROWS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _save_component_prediction(output: Path, month: str, ids, predictions) -> dict:
    frame = pd.DataFrame({
        "ml_num": pd.Series(ids, dtype=str).to_numpy(),
        "predicted_price": np.asarray(predictions, dtype=float),
    })
    if (frame.ml_num.duplicated().any() or not np.isfinite(frame.predicted_price).all()
            or (frame.predicted_price <= 0.0).any()):
        raise ValueError(f"invalid R45 component predictions for {month}")
    path = output / f"component_{month}.csv"
    frame.to_csv(path, index=False)
    return {"path": path.name, "sha256": _sha256(path), "rows": len(frame)}


def _source_files(campaign) -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r24_catboost_word_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r35_fixed_origin.py",
        campaign.PACKAGE / "r43_listaware_condo_apt_text.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
    ]


def run() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as r03
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as scoring
    from experiments.bill_target74 import r24_catboost_word_tfidf as r24
    from experiments.bill_target74 import r43_listaware_condo_apt_text as r35_verifier
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74.r19_expert_weight import _write_json as write_json
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R45 must be preregistered before fitting")
    limit_keys = (
        "campaign_fit_starts", "campaign_minutes", "exploration_fit_starts",
        "exploration_minutes", "final_reserve_fit_starts", "final_reserve_minutes",
        "batch_fit_starts", "batch_minutes", "single_fit_seconds",
    )
    if any(campaign.PROTOCOL["limits"].get(key) is not None for key in limit_keys):
        raise RuntimeError("R45 requires the active campaign's unbounded fit and time limits")
    parent = r35_verifier._verified_r35_parent(campaign)
    if (card.get("parent_run_id") != parent["screen"]["run_id"]
            or card.get("variant_run_id") != parent["screen"]["run_id"]):
        raise RuntimeError("R45's frozen parent is not the verified R35 screen run")
    for manifest in (parent["screen_manifest"], parent["development_manifest"],
                     parent["fixed_manifest"]):
        if manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
            raise RuntimeError("R35 parent artifacts use a different data snapshot")
    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active data hash differs from the preregistered campaign")
    if metadata.version("catboost") != card.get("catboost_version"):
        raise RuntimeError("CatBoost runtime version differs from the preregistered version")

    run_id = "bill_target74_r45_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    parent_artifacts = [
        (parent["screen_path"], "r35_screen_results.json"),
        (parent["screen_rows_path"], "r35_screen_paired_predictions.csv"),
        (parent["screen_manifest_path"], "r35_screen_run_manifest.json"),
        (parent["development_path"], "r35_development_results.json"),
        (parent["development_rows_path"], "r35_development_paired_predictions.csv"),
        (parent["development_manifest_path"], "r35_development_run_manifest.json"),
        (parent["fixed_path"], "r35_fixed_origin_results.json"),
        (parent["fixed_manifest_path"], "r35_fixed_origin_run_manifest.json"),
    ]
    fixed_rows_path = parent["fixed_path"].parent / "paired_predictions.csv"
    if not fixed_rows_path.is_file():
        raise FileNotFoundError("verified R35 fixed-origin paired predictions are missing")
    parent_artifacts.append((fixed_rows_path, "r35_fixed_origin_paired_predictions.csv"))
    for source_path, name in parent_artifacts:
        shutil.copy2(source_path, provenance / name)

    code_hashes, source_snapshots = {}, {}
    for index, path in enumerate(_source_files(campaign)):
        relative = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        code_hashes[relative] = _sha256(path)
        name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / name)
        source_snapshots[relative] = name

    campaign._append({
        "event": "card_start", "card": CARD, "stage": "screen",
        "run_id": run_id, "track": TRACK,
        "parent_run_id": parent["screen"]["run_id"],
        "house_type": HOUSE_TYPE, "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS),
        "model_family": "catboost", "model_config": r03.CATBOOST_CONFIG,
        "specialist_weight": SPECIALIST_WEIGHT,
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })

    fit_details: list[dict] = []
    fit_start_ids: list[str] = []
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    fit_context: dict[str, object] = {}
    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    had_variant, old_variant = VARIANT in greedy.VARIANTS, greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = {
        "model_family": "catboost", "model_config": dict(r03.CATBOOST_CONFIG),
    }

    def record_fit(parent_fit_id: str, fit_fn):
        context = fit_context.copy()
        if not context:
            raise RuntimeError("R45 fit context was not initialized")
        month = str(context["month"])
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:{month}:{COMPONENT}:{run_id}"
        campaign._append({
            "event": "fit_start", "card": CARD, "run_id": run_id,
            "fit_id": fit_id, "parent_fit_id": parent_fit_id,
            "track": TRACK, "month": month, "component": COMPONENT,
            "house_type": HOUSE_TYPE, "training_rows": context["training_rows"],
            "evaluation_rows": context["evaluation_rows"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        fit_start_ids.append(fit_id)
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            saved = _save_component_prediction(
                output, month, context["evaluation_ids"], fitted["predictions"],
            )
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "component": COMPONENT, "status": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        detail = {
            "fit_id": fit_id, "parent_fit_id": parent_fit_id,
            "month": month, "track": TRACK, "component": COMPONENT,
            "seconds": time.perf_counter() - started,
            "training_rows": fitted["training_rows"],
            "evaluation_rows": fitted["evaluation_rows"],
            "model_family": fitted["model_family"],
            "model_config": fitted["model_config"], "target": fitted["target"],
            "text_feature_count": sum(name.startswith("remarks_tfidf_")
                                       for name in fitted["feature_columns"]),
            "feature_columns": fitted["feature_columns"],
            "prediction_sha256": fitted["prediction_sha256"],
            "saved_predictions": saved,
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "run_id": run_id,
            **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: r03._verified_hgbr_cache(
        greedy, month, track, train, homes,
    )
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if pd.to_datetime(train.date_sold, errors="raise").ge(cutoff).any():
                raise RuntimeError(f"R45 training rows reach the {month} cutoff")
            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            training_mask = greedy._specialist_mask(train, HOUSE_TYPE)
            route_mask = greedy._specialist_mask(homes, HOUSE_TYPE)
            fit_rows = train.loc[training_mask].copy()
            eval_rows = homes.loc[route_mask].copy()
            support = {
                "training_rows": int(len(fit_rows)),
                "evaluation_rows": int(len(eval_rows)),
            }
            expected = card["route_support"].get(month)
            if expected != support:
                raise RuntimeError(f"R45 route support changed for {month}: {support}")
            if len(fit_rows) < MIN_TRAINING_ROWS or len(eval_rows) < MIN_EVALUATION_ROWS:
                raise RuntimeError(f"R45 {month} Detached route is below minimum support")

            reference = parent["screen_rows"].loc[
                parent["screen_rows"].month.astype(str).eq(month)
            ].copy()
            if (reference.ml_num.astype(str).tolist() != homes.ml_num.astype(str).tolist()
                    or not np.array_equal(reference.sold_price.to_numpy(float),
                                          truth.sold_price.to_numpy(float))):
                raise RuntimeError(f"R35 screen rows do not align with the R45 {month} fold")

            transformer, restore = r24.r24_installer(fit_rows, eval_rows, TRACK)
            fit_context.update(
                month=month, training_rows=len(fit_rows), evaluation_rows=len(eval_rows),
                evaluation_ids=eval_rows.ml_num.astype(str).tolist(),
            )
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows,
                    fit_rows=fit_rows, variant=VARIANT,
                    fit_tag=f"R45-{month}-Detached-CatBoost",
                )
                text_feature_count = len(transformer.columns_ or [])
            finally:
                restore()

            frozen_hgbr = r03._verified_hgbr_cache(greedy, month, TRACK, train, homes)
            specialist = blend_prices(
                fitted["predictions"], frozen_hgbr[route_mask],
                challenger_weight=SPECIALIST_WEIGHT,
            )
            r35_prediction = reference.candidate_prediction.to_numpy(float)
            prediction = greedy._overlay_specialist_predictions(
                r35_prediction, homes, specialist, HOUSE_TYPE,
            )
            if not np.array_equal(prediction[~route_mask], r35_prediction[~route_mask]):
                raise RuntimeError(f"R45 changed non-Detached predictions in {month}")
            paired = pd.DataFrame({
                "month": month, "track": TRACK,
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "parent_prediction": r35_prediction,
                "candidate_prediction": prediction,
                "initial_parent_prediction": reference.parent_prediction.to_numpy(float),
            })
            paired_parts.append(paired)
            route_truth = truth.sold_price.to_numpy(float)[route_mask]
            route_details[month] = {
                **support,
                "training_cutoff_exclusive": cutoff.isoformat(),
                "text_feature_count": text_feature_count,
                "parent": campaign._metrics(route_truth, r35_prediction[route_mask]),
                "candidate": campaign._metrics(route_truth, prediction[route_mask]),
                "non_routed_predictions_unchanged": True,
                "frozen_hgbr_rows_verified": int(len(frozen_hgbr)),
            }
            campaign._append({
                "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                "track": TRACK, "month": month, "route_support": support,
                "training_cutoff_exclusive": cutoff.isoformat(),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            pd.concat(paired_parts, ignore_index=True).to_csv(
                output / "partial_paired_predictions.csv", index=False,
            )
    except Exception as exc:
        failure = {
            "run_id": run_id, "card": CARD, "stage": "screen",
            "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_start_ids), "fit_completions": len(fit_details),
            "fit_ids": fit_start_ids,
            "fit_details": fit_details, "route_details": route_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": failure["fit_starts"],
            "fit_completions": failure["fit_completions"],
            "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r45_screen_failed",
            next_action="Inspect the R45 failure checkpoint and fit ledger before any retry",
            r45_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        raise
    finally:
        greedy._fit_record, greedy._frozen_hgbr_predictions = old_fit_record, old_hgbr
        if had_variant:
            greedy.VARIANTS[VARIANT] = old_variant
        else:
            greedy.VARIANTS.pop(VARIANT, None)

    if len(fit_details) != len(SCREEN_MONTHS):
        raise RuntimeError(f"R45 expected {len(SCREEN_MONTHS)} fit starts, recorded {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = scoring._track_result(paired)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "track": TRACK,
        "parent_run_id": parent["screen"]["run_id"],
        "initial_parent_run_id": card["initial_parent_run_id"],
        "catboost_version": metadata.version("catboost"),
        "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
        "fit_details": fit_details, "route_details": route_details,
        "tracks": {TRACK: score}, "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
        "reserve_incident_id": campaign.PROTOCOL.get("reserve_label_access_incident", {}).get("incident_id"),
    }
    result_path = output / "screen_results.json"
    write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "source_snapshots": source_snapshots,
        "catboost_version": metadata.version("catboost"),
        "parent_run_id": parent["screen"]["run_id"],
        "parent_screen_result_sha256": _sha256(parent["screen_path"]),
        "parent_screen_rows_sha256": _sha256(parent["screen_rows_path"]),
        "parent_development_result_sha256": _sha256(parent["development_path"]),
        "parent_development_rows_sha256": _sha256(parent["development_rows_path"]),
        "parent_fixed_result_sha256": _sha256(parent["fixed_path"]),
        "parent_fixed_rows_sha256": _sha256(fixed_rows_path),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    write_json(output / "run_manifest.json", manifest)
    (output / "research_report.md").write_text(
        "# R45：CatBoost Detached 专家替换 R35 LightGBM\n\n"
        f"Run：`{run_id}`。筛选父模型是已通过固定起点确认的 R35：`{parent['screen']['run_id']}`。\n\n"
        "唯一改变是把 R35 的 Detached 列表感知文本专家从 LightGBM 换为 CatBoost；训练折词级 TF-IDF、历史与结构化特征、训练标签、Detached 路由、75% 专家与 25% 冻结 HGBR 融合保持一致。所有非 Detached 预测与 R35 完全相同。\n\n"
        f"R35 MdAPE：{score['parent']['mdape_percent']:.5f}%；R45 MdAPE：{score['candidate']['mdape_percent']:.5f}%；"
        f"相对改善：{score['relative_mdape_improvement_percent']:+.3f}%；"
        f"MAPE 相对变化：{score['mape_relative_change_percent']:+.3f}%；"
        f"筛选通过：{score['qualifies_screen']}。\n\n"
        f"逐月：`{json.dumps(score['monthly'], ensure_ascii=False, sort_keys=True)}`。\n\n"
        "CatBoost 版本、数据、父版行标签、组件预测、结果和配对预测均记录在 manifest 与账本中。只读取已登记安全筛选月；2025-05/08 与隐藏标签未读取。\n",
        encoding="utf-8",
    )
    qualifies = bool(score["qualifies_screen"])
    campaign.PROTOCOL["cards"][CARD].update({
        "status": "screen_passed" if qualifies else "screen_failed",
        "run_id": run_id,
        "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_screen": qualifies,
        "screen_metrics": {
            "parent_mdape_percent": score["parent"]["mdape_percent"],
            "candidate_mdape_percent": score["candidate"]["mdape_percent"],
            "relative_mdape_improvement_percent": score["relative_mdape_improvement_percent"],
            "mape_relative_change_percent": score["mape_relative_change_percent"],
            "monthly_relative_mdape_improvement_percent": {
                month: values["relative_mdape_improvement_percent"]
                for month, values in score["monthly"].items()
            },
        },
    })
    write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": {TRACK: qualifies},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r45_screen_complete",
        next_action=("R45 passed; execute its pre-registered development and fixed-origin confirmation"
                     if qualifies else
                     "R45 failed screen; preserve R35 and register a distinct list-aware hypothesis"),
        latest_fit_count=len(fit_details),
        r45_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
    )
    return result


if __name__ == "__main__":
    result = run()
    score = result["tracks"][TRACK]
    print(
        f"R45: parent_mdape={score['parent']['mdape_percent']:.5f}% "
        f"candidate_mdape={score['candidate']['mdape_percent']:.5f}% "
        f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
        f"pass={score['qualifies_screen']} run={result['run_id']}",
        flush=True,
    )
