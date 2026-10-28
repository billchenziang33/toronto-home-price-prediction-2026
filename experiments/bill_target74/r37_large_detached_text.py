"""Screen a large-Detached off-market R22-text expert against R22."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CARD = "R37_offmarket_large_detached_r22_text_expert"
TRACK = "off_market"
HOUSE_TYPE = "Detached"
MIN_SQUARE_FEET_EXCLUSIVE = 1600.0
SPECIALIST_WEIGHT = 0.75
COMPONENT = "large_detached_text_expert"
SCREEN_MONTHS = ("2023-03", "2024-06")
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


def _large_detached_mask(frame: pd.DataFrame, greedy) -> np.ndarray:
    from model_candidate_v2.features import square_feet

    if "house_area" not in frame:
        raise ValueError("large-Detached route requires house_area")
    area = pd.to_numeric(square_feet(frame["house_area"]), errors="coerce")
    return (greedy._specialist_mask(frame, HOUSE_TYPE)
            & area.gt(MIN_SQUARE_FEET_EXCLUSIVE).fillna(False).to_numpy(dtype=bool))


def _source_files(campaign) -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r34_house_type_text_experts.py",
        campaign.PACKAGE / "r35_fixed_origin.py",
        campaign.PACKAGE / "r36_listaware_semidetached_text.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
    ]


def _verified_r22_parent(campaign, r22):
    r37_card = campaign.PROTOCOL["cards"].get(CARD, {})
    expected_run = r37_card.get("parent_run_id")
    if not expected_run:
        raise RuntimeError("R37 is not preregistered against an R22 off-market screen run")
    result, result_path, manifest = r22._verified_screen_result(campaign, expected_run)
    if result.get("tracks", {}).get(TRACK, {}).get("qualifies_screen") is not True:
        raise RuntimeError("R22 off-market screen did not qualify as R37's current reference")
    paired_path = result_path.parent / "paired_predictions.csv"
    rows = pd.read_csv(paired_path, dtype={"ml_num": str})
    rows = rows.loc[rows.track.eq(TRACK)].copy()
    if set(rows.month.astype(str)) != set(SCREEN_MONTHS):
        raise RuntimeError("R22 off-market parent rows do not cover the R37 screen months")
    return result, result_path, paired_path, manifest, rows


def run() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as scoring
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74.r19_expert_weight import _write_json as write_json
    from experiments.bill_greedy import runner as greedy

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R37 must be preregistered before fitting")
    parent_result, parent_path, parent_rows_path, parent_manifest, parent_rows = _verified_r22_parent(campaign, r22)
    if parent_result["run_id"] != card["parent_run_id"]:
        raise RuntimeError("R37 parent result changed after preregistration")
    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active data hash differs from the preregistered campaign")

    # Check route support on both registered folds before starting any learner fit.
    support = {}
    for month in SCREEN_MONTHS:
        _, train, _, _, heldout = greedy._load_fold(month)
        train_mask = _large_detached_mask(train, greedy)
        homes = hide_answers(heldout, TRACK)
        eval_mask = _large_detached_mask(homes, greedy)
        support[month] = {"training_rows": int(train_mask.sum()),
                          "evaluation_rows": int(eval_mask.sum())}
        if support[month]["training_rows"] < MIN_TRAINING_ROWS or support[month]["evaluation_rows"] < MIN_EVALUATION_ROWS:
            raise RuntimeError(f"R37 route lacks support in {month}: {support[month]}")
        del train, heldout, homes

    run_id = "bill_target74_r37_large_detached_r22_text_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for source_path, name in [
        (parent_path, "r22_screen_results.json"),
        (parent_rows_path, "r22_screen_paired_predictions.csv"),
        (parent_path.parent / "run_manifest.json", "r22_screen_run_manifest.json"),
    ]:
        shutil.copy2(source_path, provenance / name)

    code_hashes = {}
    snapshots = {}
    for index, path in enumerate(_source_files(campaign)):
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        code_hashes[key] = _sha256(path)
        name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / name)
        snapshots[key] = name

    campaign._append({
        "event": "card_start", "card": CARD, "stage": "screen",
        "run_id": run_id, "track": TRACK,
        "parent_run_id": parent_result["run_id"],
        "house_type": HOUSE_TYPE,
        "square_feet_exclusive_minimum": MIN_SQUARE_FEET_EXCLUSIVE,
        "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS),
        "specialist_weight": SPECIALIST_WEIGHT,
        "route_support": support,
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })

    fit_details: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    fit_context: dict[str, object] = {}
    old_fit_record = greedy._fit_record

    def record_fit(_original_id: str, fit_fn):
        month = str(fit_context["month"])
        fit_id = (f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:{month}:"
                  f"{COMPONENT}:{run_id}")
        campaign._append({
            "event": "fit_start", "card": CARD, "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": COMPONENT,
            "training_rows": fit_context["training_rows"],
            "evaluation_rows": fit_context["evaluation_rows"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "component": COMPONENT, "status": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        saved = r22._save_component_predictions(
            output, TRACK, month, COMPONENT,
            fit_context["evaluation_ids"], fitted["predictions"],
        )
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": COMPONENT, "seconds": time.perf_counter() - started,
            "training_rows": fitted["training_rows"],
            "evaluation_rows": fitted["evaluation_rows"],
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "target": fitted["target"], "feature_columns": fitted["feature_columns"],
            "text_feature_count": sum(name.startswith("remarks_tfidf_")
                                       for name in fitted["feature_columns"]),
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
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R37 training data reach the {month} cutoff")
            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            train_mask = _large_detached_mask(train, greedy)
            route_mask = _large_detached_mask(homes, greedy)
            fit_rows = train.loc[train_mask].copy()
            eval_rows = homes.loc[route_mask].copy()
            if len(fit_rows) != support[month]["training_rows"] or int(route_mask.sum()) != support[month]["evaluation_rows"]:
                raise RuntimeError(f"R37 route support changed after preregistration in {month}")

            reference = scoring.r22_reference_frame(parent_rows, truth, month, TRACK)
            parent_prediction = reference.predicted_price.to_numpy(float)
            transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
            fit_context.update(month=month, training_rows=len(fit_rows),
                               evaluation_rows=len(eval_rows),
                               evaluation_ids=eval_rows.ml_num.astype(str).tolist())
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows, fit_rows=fit_rows,
                    variant=r22.VARIANT, base_variant="G06_lot_garage",
                    fit_tag=f"R37-{month}-Detached-over-1600sqft",
                )
            finally:
                restore()

            candidate = parent_prediction.copy()
            candidate[route_mask] = np.round(
                (1.0 - SPECIALIST_WEIGHT) * parent_prediction[route_mask]
                + SPECIALIST_WEIGHT * fitted["predictions"],
            )
            if not np.array_equal(candidate[~route_mask], parent_prediction[~route_mask]):
                raise RuntimeError(f"R37 changed predictions outside the large-Detached route in {month}")
            paired = pd.DataFrame({
                "month": month, "track": TRACK,
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "parent_prediction": parent_prediction,
                "candidate_prediction": candidate,
            })
            paired_parts.append(paired)
            route_truth = truth.sold_price.to_numpy(float)[route_mask]
            route_details[month] = {
                "training_rows": len(fit_rows),
                "evaluation_rows": int(route_mask.sum()),
                "parent": campaign._metrics(route_truth, parent_prediction[route_mask]),
                "candidate": campaign._metrics(route_truth, candidate[route_mask]),
                "non_routed_predictions_unchanged": True,
                "text_feature_count": len(transformer.columns_ or []),
                "training_cutoff_exclusive": cutoff.isoformat(),
            }
            campaign._append({
                "event": "screen_fold_complete", "card": CARD,
                "run_id": run_id, "track": TRACK, "month": month,
                "routed_rows": int(route_mask.sum()),
                "routed_training_rows": len(fit_rows),
                "training_cutoff_exclusive": cutoff.isoformat(),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            pd.concat(paired_parts, ignore_index=True).to_csv(
                output / "partial_paired_predictions.csv", index=False)
    except Exception as exc:
        greedy._fit_record = old_fit_record
        failure = {
            "run_id": run_id, "card": CARD, "stage": "screen",
            "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(fit_details),
            "fit_details": fit_details, "route_details": route_details,
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False, "hidden_labels_opened": False,
        }
        write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": len(fit_details), "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign.PROTOCOL["cards"][CARD].update({
            "status": "failed", "run_id": run_id,
            "failure_path": str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
            "fit_starts": len(fit_details),
        })
        _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._write_progress(
            "r37_screen_failed", next_action="inspect R37 failure checkpoint before retrying",
            latest_fit_count=len(fit_details),
        )
        raise
    finally:
        greedy._fit_record = old_fit_record

    if len(fit_details) != len(SCREEN_MONTHS):
        raise RuntimeError(f"R37 expected {len(SCREEN_MONTHS)} fits, recorded {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    rows = scoring._track_result(paired)
    passed = bool(rows["qualifies_screen"])
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "track": TRACK,
        "parent_run_id": parent_result["run_id"],
        "parent_candidate": "R22_train_fold_tfidf_remarks",
        "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
        "specialist": {"house_type": HOUSE_TYPE,
                       "square_feet_exclusive_minimum": MIN_SQUARE_FEET_EXCLUSIVE,
                       "weight": SPECIALIST_WEIGHT,
                       "training_rows_are_routed_rows_only": True,
                       "tfidf_is_fit_on_routed_rows_only": True},
        "route_support": support,
        "tracks": {TRACK: rows}, "routed_diagnostics": route_details,
        "fit_details": fit_details, "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    result_path = output / "screen_results.json"
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "source_snapshots": snapshots,
        "parent_run_id": parent_result["run_id"],
        "parent_result_sha256": _sha256(parent_path),
        "parent_paired_predictions_sha256": _sha256(parent_rows_path),
        "parent_manifest_sha256": _sha256(parent_path.parent / "run_manifest.json"),
        "route_support": support, "fit_starts": len(fit_details),
        "fit_details": fit_details,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report_lines = [
        "# R37：off-market 大户型 Detached R22-text 专家筛选", "",
        f"Run：`{run_id}`；对照是哈希核验的 R22 run `{parent_result['run_id']}`。", "",
        f"路由：`Detached AND square_feet > {MIN_SQUARE_FEET_EXCLUSIVE:.0f}`；",
        f"固定融合：{SPECIALIST_WEIGHT:.0%} 子群专家 + {1-SPECIALIST_WEIGHT:.0%} R22。", "",
        f"R22 pooled MdAPE：{rows['parent']['mdape_percent']:.5f}%；",
        f"R37 pooled MdAPE：{rows['candidate']['mdape_percent']:.5f}%；",
        f"相对改善：{rows['relative_mdape_improvement_percent']:+.3f}%；",
        f"MAPE 相对变化：{rows['mape_relative_change_percent']:+.3f}%。", "",
        f"两个筛选月均改善：{all(item['relative_mdape_improvement_percent'] > 0 for item in rows['monthly'].values())}；",
        f"筛选门槛通过：{passed}。", "",
        "该假设与 R20 的大/小面积现成专家权重选择不同：R37 只在大户型 Detached 子群重训文本模型，权重固定。",
        "只读取 2023-03 与 2024-06 开发折；未读取 2025-05/08 或隐藏标签。"
        "保留集因先前完整 parquet 载入事件而不具备独立验证资格。", "",
        f"逐房预测、账本快照和哈希位于 `{output.relative_to(campaign.ROOT)}`。", "",
    ]
    (output / "research_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": {TRACK: passed},
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })
    campaign.PROTOCOL["cards"][CARD].update({
        "status": "screen_passed" if passed else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_screen": passed,
        "screen_metrics": {
            "parent_mdape_percent": rows["parent"]["mdape_percent"],
            "candidate_mdape_percent": rows["candidate"]["mdape_percent"],
            "relative_mdape_improvement_percent": rows["relative_mdape_improvement_percent"],
            "mape_relative_change_percent": rows["mape_relative_change_percent"],
            "monthly_relative_mdape_improvement_percent": {
                month: value["relative_mdape_improvement_percent"]
                for month, value in rows["monthly"].items()
            },
        },
    })
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._write_progress(
        "r37_screen_complete",
        next_action=("R37 passed screen; register eight-month off-market development confirmation against R22"
                     if passed else "R37 failed screen; preserve R22 and register a distinct off-market hypothesis"),
        r37_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=len(fit_details),
    )
    return result


if __name__ == "__main__":
    outcome = run()
    score = outcome["tracks"][TRACK]
    print(
        f"R37 screen: parent_mdape={score['parent']['mdape_percent']:.5f}% "
        f"candidate_mdape={score['candidate']['mdape_percent']:.5f}% "
        f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
        f"pass={score['qualifies_screen']} run={outcome['run_id']}",
        flush=True,
    )
