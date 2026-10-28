"""R34/R35: screen one R22-text house-type expert per track on development folds."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARDS = {
    "R34": {
        "card": "R34_offmarket_semi_detached_r22_text_expert",
        "track": "off_market",
        "house_type": "Semi-Detached",
        "base_variant": "G06_lot_garage",
        "specialist_weight": 0.75,
        "component": "semi_detached_expert",
    },
    "R35": {
        "card": "R35_listaware_detached_r22_text_expert",
        "track": "list_aware",
        "house_type": "Detached",
        "base_variant": None,
        "specialist_weight": 0.75,
        "component": "detached_core_expert",
    },
    "R38": {
        "card": "R38_offmarket_detached_r22_text_expert",
        "track": "off_market",
        "house_type": "Detached",
        "base_variant": "G06_lot_garage",
        "specialist_weight": 0.75,
        "component": "detached_expert",
        "route_support": {
            "2023-03": {"training_rows": 19767, "evaluation_rows": 625},
            "2024-06": {"training_rows": 29041, "evaluation_rows": 755},
        },
    },
    "R39": {
        "card": "R39_offmarket_detached_word_char_text_expert",
        "track": "off_market",
        "house_type": "Detached",
        "base_variant": "G06_lot_garage",
        "specialist_weight": 0.75,
        "component": "detached_word_char_expert",
        "text_family": "word_char",
        "variant": "R23_train_fold_word_char_tfidf_remarks",
        "route_support": {
            "2023-03": {"training_rows": 19767, "evaluation_rows": 625},
            "2024-06": {"training_rows": 29041, "evaluation_rows": 755},
        },
    },
    "R42": {
        "card": "R42_offmarket_condo_apt_r22_text_expert",
        "track": "off_market",
        "house_type": "Condo Apt",
        "base_variant": "G06_lot_garage",
        "specialist_weight": 0.75,
        "component": "condo_apt_expert",
        "route_support": {
            "2023-03": {"training_rows": 38202, "evaluation_rows": 1396},
            "2024-06": {"training_rows": 54995, "evaluation_rows": 974},
        },
    },
}
SCREEN_MONTHS = ("2023-03", "2024-06")
MIN_TRAINING_ROWS = 500
MIN_EVALUATION_ROWS = 100


def _text_vocabulary(transformer) -> dict[str, list[str]]:
    if hasattr(transformer, "vectorizer"):
        return {"word": transformer.vectorizer.get_feature_names_out().tolist()}
    if hasattr(transformer, "word_vectorizer") and hasattr(transformer, "char_vectorizer"):
        return {
            "word": transformer.word_vectorizer.get_feature_names_out().tolist(),
            "character": transformer.char_vectorizer.get_feature_names_out().tolist(),
        }
    raise TypeError(f"unsupported text transformer: {type(transformer).__name__}")


def run_card(short_name: str) -> dict:
    if short_name not in CARDS:
        raise ValueError(f"unknown card: {short_name!r}")

    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as report_helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as shared
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    spec = CARDS[short_name]
    card = spec["card"]
    protocol_card = campaign.PROTOCOL["cards"].get(card)
    if not protocol_card or protocol_card.get("status") != "registered":
        raise RuntimeError(f"{card} must be registered before running")
    source = budget._source_identity()
    parent_result, parent_path, parent_manifest = r22._verified_screen_result(campaign)
    expected_r22_run = protocol_card.get("parent_run_id")
    if expected_r22_run and parent_result.get("run_id") != expected_r22_run:
        raise RuntimeError("R22 parent run differs from the preregistered run")
    parent_rows_path = parent_path.parent / "paired_predictions.csv"
    parent_rows = pd.read_csv(parent_rows_path, dtype={"ml_num": str})
    run_id = f"bill_target74_{short_name.lower()}_house_type_text_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(parent_path, provenance / "r22_screen_results.json")
    shutil.copy2(parent_rows_path, provenance / "r22_screen_paired_predictions.csv")
    shutil.copy2(parent_path.parent / "run_manifest.json", provenance / "r22_screen_run_manifest.json")

    source_files = [
        campaign.PROTOCOL_PATH,
        campaign.PACKAGE / "r34_house_type_text_experts.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
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
    code_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_files
    }
    for path in source_files:
        relative_name = str(path.relative_to(campaign.ROOT)).replace("\\", "__").replace("/", "__")
        target = provenance / (relative_name + ".source")
        shutil.copy2(path, target)

    campaign._append({
        "event": "card_start", "card": card, "run_id": run_id,
        "track": spec["track"], "house_type": spec["house_type"],
        "parent_candidate": parent_result["run_id"],
        "screen_months": list(SCREEN_MONTHS), "expected_fit_starts": len(SCREEN_MONTHS),
        "specialist_weight": spec["specialist_weight"],
        "text_family": spec.get("text_family", "word"),
        "route_support": spec.get("route_support", {}),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
        "reserve_incident_id": campaign.PROTOCOL.get("reserve_label_access_incident", {}).get("incident_id"),
    })

    fit_details: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    old_fit_record = greedy._fit_record

    def record_fit(parent_fit_id: str, fit_fn):
        context = active_context.copy()
        fit_id = (f"bill_target74:{card}:{spec['track']}:{context['month']}"
                  f":{spec['component']}:{run_id}")
        campaign._append({
            "event": "fit_start", "card": card, "run_id": run_id, "fit_id": fit_id,
            "parent_fit_id": parent_fit_id, "track": spec["track"],
            "month": context["month"], "component": spec["component"],
            "house_type": spec["house_type"], "training_rows": context["training_rows"],
            "evaluation_rows": context["evaluation_rows"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            elapsed = time.perf_counter() - started
            campaign._append({
                "event": "fit_complete", "card": card, "run_id": run_id,
                "fit_id": fit_id, "track": spec["track"], "month": context["month"],
                "component": spec["component"], "status": "failed", "seconds": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        elapsed = time.perf_counter() - started
        detail = {
            "fit_id": fit_id, "parent_fit_id": parent_fit_id,
            "track": spec["track"], "month": context["month"],
            "component": spec["component"], "seconds": elapsed,
            "training_rows": fitted["training_rows"],
            "evaluation_rows": fitted["evaluation_rows"],
            "feature_columns": fitted["feature_columns"],
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "target": fitted["target"], "prediction_sha256": fitted["prediction_sha256"],
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": card, "run_id": run_id,
            **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    active_context: dict[str, object] = {}
    greedy._fit_record = record_fit
    try:
        for month in SCREEN_MONTHS:
            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"training data are not strictly before the {month} cutoff")
            homes = hide_answers(heldout, spec["track"])
            route_mask = greedy._specialist_mask(homes, spec["house_type"])
            training_mask = greedy._specialist_mask(train, spec["house_type"])
            fit_rows = train.loc[training_mask].copy()
            eval_rows = homes.loc[route_mask].copy()
            expected_support = spec.get("route_support", {}).get(month)
            actual_support = {
                "training_rows": int(len(fit_rows)),
                "evaluation_rows": int(len(eval_rows)),
            }
            if expected_support is not None and actual_support != expected_support:
                raise RuntimeError(
                    f"{short_name} route support changed in {month}: "
                    f"expected {expected_support}, found {actual_support}"
                )
            if len(fit_rows) < MIN_TRAINING_ROWS or len(eval_rows) < MIN_EVALUATION_ROWS:
                raise RuntimeError(
                    f"{month} route too small: train={len(fit_rows)}, eval={len(eval_rows)}; "
                    f"minimums are {MIN_TRAINING_ROWS}/{MIN_EVALUATION_ROWS}"
                )

            reference = shared.r22_reference_frame(parent_rows, truth, month, spec["track"])
            if spec.get("text_family") == "word_char":
                transformer, restore = shared._install_text_features(fit_rows, eval_rows, spec["track"])
            else:
                transformer, restore = r22._install_text_features(fit_rows, eval_rows, spec["track"])
            active_context = {"month": month, "training_rows": len(fit_rows),
                              "evaluation_rows": len(eval_rows)}
            try:
                fitted = greedy._fit_history_candidate(
                    month, spec["track"], train, eval_rows,
                    fit_rows=fit_rows, variant=spec.get("variant", r22.VARIANT),
                    base_variant=spec["base_variant"],
                    fit_tag=f"{short_name}-{month}-{spec['house_type']}",
                )
            finally:
                restore()

            if spec["track"] == "off_market":
                prediction = greedy._blend_specialist_predictions(
                    reference.predicted_price.to_numpy(float), fitted["predictions"],
                    homes, spec["house_type"], spec["specialist_weight"],
                )
            else:
                from experiments.bill_target74.r03_catboost import _verified_hgbr_cache

                all_hgbr = _verified_hgbr_cache(greedy, month, spec["track"], train, homes)
                specialist_prediction = blend_prices(
                    fitted["predictions"], all_hgbr[route_mask], challenger_weight=0.75,
                )
                prediction = greedy._overlay_specialist_predictions(
                    reference.predicted_price.to_numpy(float), homes,
                    specialist_prediction, spec["house_type"],
                )

            candidate = pd.DataFrame({
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "predicted_price": np.asarray(prediction, dtype=float),
            })
            paired_parts.append(report_helpers._candidate_frame(
                month, spec["track"], truth, reference, candidate,
            ))
            route_truth = truth.loc[route_mask, "sold_price"].to_numpy(float)
            route_parent = reference.predicted_price.to_numpy(float)[route_mask]
            route_candidate = np.asarray(prediction, dtype=float)[route_mask]
            route_details[month] = {
                "routed_rows": int(route_mask.sum()),
                "training_rows": int(len(fit_rows)),
                "parent": campaign._metrics(route_truth, route_parent),
                "candidate": campaign._metrics(route_truth, route_candidate),
                "non_routed_predictions_unchanged": bool(np.array_equal(
                    reference.predicted_price.to_numpy(float)[~route_mask],
                    np.asarray(prediction, dtype=float)[~route_mask],
                )),
                "tfidf_vocabulary": _text_vocabulary(transformer),
                "feature_columns": fitted["feature_columns"],
            }
            campaign._append({
                "event": "screen_fold_complete", "card": card, "run_id": run_id,
                "track": spec["track"], "month": month,
                "training_cutoff_exclusive": cutoff.isoformat(),
                "routed_rows": int(route_mask.sum()),
                "routed_training_rows": int(len(fit_rows)),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
    except Exception as exc:
        campaign._append({
            "event": "card_failed", "card": card, "run_id": run_id,
            "fit_starts": len(fit_details), "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            f"{short_name.lower()}_screen_failed",
            next_action=f"inspect {card} failure event and artifacts before retry",
            latest_fit_count=len(fit_details),
        )
        raise
    finally:
        greedy._fit_record = old_fit_record

    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    rows = shared._track_result(paired)
    passed = bool(rows["qualifies_screen"])
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": card, "track": spec["track"], "parent_candidate": parent_result["run_id"],
        "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
        "specialist": {"house_type": spec["house_type"],
                        "weight": spec["specialist_weight"],
                        "text_family": spec.get("text_family", "word"),
                        "minimum_training_rows": MIN_TRAINING_ROWS,
                        "minimum_evaluation_rows_per_month": MIN_EVALUATION_ROWS,
                        "route_support": spec.get("route_support", {})},
        "tracks": {spec["track"]: rows}, "routed_diagnostics": route_details,
        "fit_details": fit_details,
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False,
        "reserve_incident_id": campaign.PROTOCOL.get("reserve_label_access_incident", {}).get("incident_id"),
        "hidden_labels_opened": False, "data_sha256": source["data_sha256"],
    }
    result_path = output / "screen_results.json"
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "parent_run_id": parent_result["run_id"],
        "parent_result_sha256": _sha256(parent_path),
        "parent_paired_predictions_sha256": _sha256(parent_rows_path),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = output / "research_report.md"
    report.write_text(
        f"# {card}：R22 逐折文本特征的房型专家\n\n"
        f"Run：`{run_id}`；父模型：已核验 R22 run `{parent_result['run_id']}`。\n\n"
        f"只筛选 `{spec['track']}` 的 `{spec['house_type']}`。专家只使用该组的严格过去训练行，"
        f"其词表与 IDF 也只在该组的训练 remarks 上拟合；固定专家权重为 {spec['specialist_weight']:.2f}。\n\n"
        f"- R22 pooled MdAPE：{rows['parent']['mdape_percent']:.5f}%\n"
        f"- 候选 pooled MdAPE：{rows['candidate']['mdape_percent']:.5f}%\n"
        f"- 相对改善：{rows['relative_mdape_improvement_percent']:+.3f}%\n"
        f"- MAPE相对变化：{rows['mape_relative_change_percent']:+.3f}%\n"
        f"- 两筛选月均改善：{all(value['relative_mdape_improvement_percent'] > 0 for value in rows['monthly'].values())}\n"
        f"- 筛选门槛通过：{passed}\n\n"
        "该 run 只使用 2023-03、2024-06 开发折。2025-05/08 保留集已不具备独立验证资格；"
        "本 run 未读取保留月份或隐藏标签。门槛未通过则保留 R22，不调筛选月权重。\n",
        encoding="utf-8",
    )
    campaign._append({
        "event": "card_complete", "card": card, "run_id": run_id,
        "fit_starts": len(fit_details), "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": {spec["track"]: passed},
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })
    campaign.PROTOCOL["cards"][card].update({
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
        f"{short_name.lower()}_screen_complete",
        next_action=(f"register a new distinct experiment; {card} passed only its two-month screen and requires development confirmation"
                     if passed else f"{card} failed its fixed screen; preserve R22 and register a distinct hypothesis"),
        latest_fit_count=len(fit_details),
        **{f"{short_name.lower()}_result": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/")},
    )
    return result


def run_development_confirmation(short_name: str) -> dict:
    if short_name != "R35":
        raise ValueError("only R35 passed its frozen screen and is eligible for development confirmation")

    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as report_helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as shared
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    spec = CARDS[short_name]
    card = spec["card"]
    protocol_card = campaign.PROTOCOL["cards"].get(card, {})
    if protocol_card.get("status") != "screen_passed" or not protocol_card.get("qualifies_screen"):
        raise RuntimeError("R35 development confirmation requires a hash-verified passing screen")
    screen_run_id = protocol_card.get("run_id")
    screen_output = campaign.CAMPAIGN_ROOT / "batches" / str(screen_run_id)
    screen_result_path = screen_output / "screen_results.json"
    screen_pairs_path = screen_output / "paired_predictions.csv"
    screen_manifest_path = screen_output / "run_manifest.json"
    for path in (screen_result_path, screen_pairs_path, screen_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    screen_result = json.loads(screen_result_path.read_text(encoding="utf-8"))
    screen_manifest = json.loads(screen_manifest_path.read_text(encoding="utf-8"))
    if (screen_result.get("run_id") != screen_run_id
            or screen_result.get("card") != card
            or screen_manifest.get("result_sha256") != _sha256(screen_result_path)
            or screen_manifest.get("paired_predictions_sha256") != _sha256(screen_pairs_path)
            or screen_manifest.get("parent_run_id") != protocol_card.get("parent_run_id")):
        raise RuntimeError("R35 screen artifacts fail identity, parent, or hash verification")
    script_key = "experiments/bill_target74/r34_house_type_text_experts.py"
    script_snapshot = screen_output / "provenance" / (script_key.replace("/", "__") + ".source")
    if (not script_snapshot.is_file()
            or _sha256(script_snapshot) != screen_manifest.get("code_sha256", {}).get(script_key)):
        raise RuntimeError("R35 screen source snapshot does not match its manifest")

    r22_dev, r22_dev_path, r22_dev_manifest = r22._verified_development_result(campaign)
    if "list_aware" not in r22_dev.get("eligible_tracks", []):
        raise RuntimeError("R22 list-aware development result is not an eligible parent")
    source = budget._source_identity()
    r22_rows_path = r22_dev_path.parent / "paired_predictions.csv"
    r22_rows = pd.read_csv(r22_rows_path, dtype={"ml_num": str})
    screen_pairs = pd.read_csv(screen_pairs_path, dtype={"ml_num": str})
    dev_months = tuple(r22.DEVELOPMENT_MONTHS)
    screen_months = tuple(r22.SCREEN_MONTHS)
    if set(screen_months) - set(dev_months):
        raise RuntimeError("R35 screen months are outside R22's fixed development window")

    def r22_parent(month: str) -> pd.DataFrame:
        rows = r22_rows.loc[r22_rows.month.eq(month) & r22_rows.track.eq(spec["track"])].copy()
        if rows.ml_num.astype(str).duplicated().any() or rows.empty:
            raise RuntimeError(f"R22 development rows are duplicated or missing for {month}")
        return rows

    # Confirm screen candidate and R22's eight-month parent predictions are the same on reused folds.
    for month in screen_months:
        expected = r22_parent(month).assign(ml_num=lambda frame: frame.ml_num.astype(str)).set_index("ml_num")
        candidate_rows = screen_pairs.loc[
            screen_pairs.month.eq(month) & screen_pairs.track.eq(spec["track"])
        ].copy()
        if candidate_rows.ml_num.astype(str).duplicated().any():
            raise RuntimeError(f"R35 screen predictions contain duplicate IDs for {month}")
        observed = candidate_rows.assign(ml_num=lambda frame: frame.ml_num.astype(str)).set_index("ml_num")
        observed = observed.reindex(expected.index)
        if (observed.candidate_prediction.isna().any()
                or not np.array_equal(observed.sold_price.to_numpy(float), expected.sold_price.to_numpy(float))
                or not np.array_equal(observed.parent_prediction.to_numpy(float), expected.candidate_prediction.to_numpy(float))):
            raise RuntimeError(f"R35 screen rows or R22 predictions do not match the development parent in {month}")

    run_id = "bill_target74_r35_house_type_text_development_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(screen_result_path, provenance / "r35_screen_results.json")
    shutil.copy2(screen_pairs_path, provenance / "r35_screen_paired_predictions.csv")
    shutil.copy2(screen_manifest_path, provenance / "r35_screen_run_manifest.json")
    shutil.copy2(r22_dev_path, provenance / "r22_development_results.json")
    shutil.copy2(r22_rows_path, provenance / "r22_development_paired_predictions.csv")

    source_files = [
        campaign.PROTOCOL_PATH,
        campaign.PACKAGE / "r34_house_type_text_experts.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
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
    code_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_files
    }
    source_snapshots = {}
    for index, path in enumerate(source_files):
        source_key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        snapshot_name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / snapshot_name)
        source_snapshots[source_key] = snapshot_name

    expected_fits = len(dev_months) - len(screen_months)
    campaign._append({
        "event": "card_start", "card": card, "stage": "development_confirmation",
        "run_id": run_id, "screen_run_id": screen_run_id,
        "parent_development_run_id": r22_dev["run_id"], "track": spec["track"],
        "development_months": list(dev_months), "reused_screen_months": list(screen_months),
        "expected_fit_starts": expected_fits,
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })

    fit_details: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    old_fit_record = greedy._fit_record
    active_context: dict[str, object] = {}

    def record_fit(parent_fit_id: str, fit_fn):
        context = active_context.copy()
        fit_id = (f"bill_target74:{card}:development_confirmation:{spec['track']}"
                  f":{context['month']}:{spec['component']}:{run_id}")
        campaign._append({
            "event": "fit_start", "card": card, "stage": "development_confirmation",
            "run_id": run_id, "fit_id": fit_id, "parent_fit_id": parent_fit_id,
            "track": spec["track"], "month": context["month"],
            "component": spec["component"], "house_type": spec["house_type"],
            "training_rows": context["training_rows"], "evaluation_rows": context["evaluation_rows"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            elapsed = time.perf_counter() - started
            campaign._append({
                "event": "fit_complete", "card": card, "stage": "development_confirmation",
                "run_id": run_id, "fit_id": fit_id, "track": spec["track"],
                "month": context["month"], "component": spec["component"],
                "status": "failed", "seconds": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        elapsed = time.perf_counter() - started
        detail = {
            "fit_id": fit_id, "parent_fit_id": parent_fit_id,
            "track": spec["track"], "month": context["month"],
            "component": spec["component"], "seconds": elapsed,
            "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
            "feature_columns": fitted["feature_columns"],
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "target": fitted["target"], "prediction_sha256": fitted["prediction_sha256"],
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": card, "stage": "development_confirmation",
            "run_id": run_id, **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in dev_months:
            parent_rows = r22_parent(month)
            expected_ids = parent_rows.ml_num.astype(str).to_numpy()
            parent_candidate = parent_rows.candidate_prediction.to_numpy(float)
            if month in screen_months:
                observed = screen_pairs.loc[
                    screen_pairs.month.eq(month) & screen_pairs.track.eq(spec["track"])
                ].copy().assign(ml_num=lambda frame: frame.ml_num.astype(str)).set_index("ml_num").reindex(expected_ids)
                truth = pd.DataFrame({"ml_num": expected_ids,
                                      "sold_price": parent_rows.sold_price.to_numpy(float)})
                parent_frame = pd.DataFrame({"ml_num": expected_ids,
                                             "sold_price": parent_rows.sold_price.to_numpy(float),
                                             "predicted_price": parent_candidate})
                candidate_frame = pd.DataFrame({
                    "ml_num": expected_ids,
                    "sold_price": observed.sold_price.to_numpy(float),
                    "predicted_price": observed.candidate_prediction.to_numpy(float),
                })
                paired_parts.append(report_helpers._candidate_frame(
                    month, spec["track"], truth, parent_frame, candidate_frame,
                ))
                continue

            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R35 development training data include sales at or after cutoff: {month}")
            homes = hide_answers(heldout, spec["track"])
            if (not np.array_equal(homes.ml_num.astype(str).to_numpy(), expected_ids)
                    or not np.array_equal(truth.sold_price.to_numpy(float), parent_rows.sold_price.to_numpy(float))):
                raise RuntimeError(f"R35 development fold does not align with R22 parent for {month}")
            route_mask = greedy._specialist_mask(homes, spec["house_type"])
            training_mask = greedy._specialist_mask(train, spec["house_type"])
            fit_rows = train.loc[training_mask].copy()
            eval_rows = homes.loc[route_mask].copy()
            if len(fit_rows) < MIN_TRAINING_ROWS or len(eval_rows) < MIN_EVALUATION_ROWS:
                raise RuntimeError(f"R35 development route is too small for {month}")

            transformer, restore = r22._install_text_features(fit_rows, eval_rows, spec["track"])
            active_context = {"month": month, "training_rows": len(fit_rows),
                              "evaluation_rows": len(eval_rows)}
            try:
                fitted = greedy._fit_history_candidate(
                    month, spec["track"], train, eval_rows,
                    fit_rows=fit_rows, variant=r22.VARIANT,
                    base_variant=spec["base_variant"],
                    fit_tag=f"R35-dev-{month}-Detached",
                )
            finally:
                restore()
            frozen_hgbr = r03_hgbr_predictions(greedy, month, train, homes)
            specialist_prediction = blend_prices(
                fitted["predictions"], frozen_hgbr[route_mask], challenger_weight=0.75,
            )
            prediction = greedy._overlay_specialist_predictions(
                parent_candidate, homes, specialist_prediction, spec["house_type"],
            )
            truth_frame = truth[["ml_num", "sold_price"]].copy()
            parent_frame = pd.DataFrame({"ml_num": expected_ids,
                                         "sold_price": truth.sold_price.to_numpy(float),
                                         "predicted_price": parent_candidate})
            candidate_frame = pd.DataFrame({"ml_num": expected_ids,
                                            "sold_price": truth.sold_price.to_numpy(float),
                                            "predicted_price": prediction})
            paired_parts.append(report_helpers._candidate_frame(
                month, spec["track"], truth_frame, parent_frame, candidate_frame,
            ))
            route_details[month] = {
                "routed_rows": int(route_mask.sum()),
                "training_rows": int(len(fit_rows)),
                "parent": campaign._metrics(truth.sold_price.to_numpy(float)[route_mask],
                                             parent_candidate[route_mask]),
                "candidate": campaign._metrics(truth.sold_price.to_numpy(float)[route_mask],
                                                prediction[route_mask]),
                "non_routed_predictions_unchanged": bool(np.array_equal(
                    parent_candidate[~route_mask], prediction[~route_mask],
                )),
                "text_feature_count": len(transformer.columns_ or []),
            }
    except Exception as exc:
        campaign._append({
            "event": "card_failed", "card": card, "stage": "development_confirmation",
            "run_id": run_id, "fit_starts": len(fit_details),
            "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r35_development_confirmation_failed",
            next_action="inspect R35 development failure event and saved artifacts; do not restart completed fits",
            latest_fit_count=len(fit_details),
        )
        raise
    finally:
        greedy._fit_record = old_fit_record

    if len(fit_details) != expected_fits:
        raise RuntimeError(f"R35 expected {expected_fits} new fits, recorded {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = r22.score_development_confirmation(
        paired, months=dev_months,
        target_mdape_strictly_below=campaign.PROTOCOL["initial_development_metrics"]["list_aware"]["target_mdape_strictly_below"],
        bootstrap_replicates=2000, bootstrap_seed=20261024,
        minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["development_relative_mdape_improvement_percent"],
        minimum_months_improved=campaign.PROTOCOL["gates"]["development_months_improved_min"],
        maximum_worst_month_regression_percent=campaign.PROTOCOL["gates"]["development_worst_month_regression_max_percent"],
        maximum_mape_regression_percent=campaign.PROTOCOL["gates"]["development_mape_regression_max_percent"],
    )
    passed = bool(score["qualifies_development"])
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": card, "stage": "development_confirmation", "track": spec["track"],
        "screen_run_id": screen_run_id, "parent_development_run_id": r22_dev["run_id"],
        "fit_starts": len(fit_details), "development_months": list(dev_months),
        "reused_screen_months": list(screen_months),
        "tracks": {spec["track"]: score}, "routed_diagnostics": route_details,
        "fit_details": fit_details, "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    result_path = output / "development_results.json"
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "source_snapshots": source_snapshots,
        "screen_run_id": screen_run_id,
        "screen_result_sha256": _sha256(screen_result_path),
        "screen_paired_predictions_sha256": _sha256(screen_pairs_path),
        "parent_development_result_sha256": _sha256(r22_dev_path),
        "parent_development_predictions_sha256": _sha256(r22_rows_path),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = output / "research_report.md"
    report.write_text(
        f"# R35：八个月开发确认\n\n"
        f"Run：`{run_id}`；筛选 run：`{screen_run_id}`；R22 开发父 run：`{r22_dev['run_id']}`。\n\n"
        f"R22 pooled MdAPE：{score['parent']['mdape_percent']:.5f}%；"
        f"R35 pooled MdAPE：{score['candidate']['mdape_percent']:.5f}%；"
        f"相对改善：{score['relative_mdape_improvement_percent']:+.3f}%；"
        f"MAPE相对变化：{score['mape_relative_change_percent']:+.3f}%。\n\n"
        f"- 改善月份：{score['months_improved']}/8\n"
        f"- 最差月退化：{score['worst_month_regression_percent']:.3f}%\n"
        f"- 配对 MdAPE 差值 95% 区间：[{score['bootstrap_mdape_difference_percent']['lower95']:.5f}, "
        f"{score['bootstrap_mdape_difference_percent']['upper95']:.5f}] 个百分点\n"
        f"- list-aware `<4%`：{score['target_met']}\n"
        f"- 开发晋级门槛：{passed}\n\n"
        f"本次新增 {len(fit_details)} fits；两个筛选月使用哈希核验的已存预测。2025-05/08 保留集没有读取，"
        "但已不具备独立验证资格；隐藏标签没有读取。\n",
        encoding="utf-8",
    )
    campaign._append({
        "event": "card_complete", "card": card, "stage": "development_confirmation",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": {spec["track"]: passed},
        "targets_met_by_track": {spec["track"]: bool(score["target_met"])},
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })
    campaign.PROTOCOL["cards"][card].update({
        "status": "development_confirmation_passed" if passed else "development_confirmation_failed",
        "development_run_id": run_id,
        "development_result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_development": passed,
        "development_metrics": {
            "parent_mdape_percent": score["parent"]["mdape_percent"],
            "candidate_mdape_percent": score["candidate"]["mdape_percent"],
            "relative_mdape_improvement_percent": score["relative_mdape_improvement_percent"],
            "mape_relative_change_percent": score["mape_relative_change_percent"],
            "months_improved": score["months_improved"],
            "target_met": score["target_met"],
        },
    })
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._write_progress(
        "r35_development_confirmation_complete",
        next_action=("run fixed-origin confirmation for R35; reserve validation remains unavailable"
                     if passed else "R35 failed development confirmation; keep R22 and register a distinct list-aware hypothesis"),
        r35_development_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=len(fit_details),
    )
    return result


def r03_hgbr_predictions(greedy, month: str, train: pd.DataFrame, homes: pd.DataFrame) -> np.ndarray:
    from experiments.bill_target74.r03_catboost import _verified_hgbr_cache

    return _verified_hgbr_cache(greedy, month, "list_aware", train, homes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True, choices=sorted(CARDS))
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    result = run_development_confirmation(args.card) if args.development else run_card(args.card)
    track = result["track"]
    score = result["tracks"][track]
    metrics = score["candidate"]
    print(
        f"{args.card}: candidate_mdape={metrics['mdape_percent']:.5f}% "
        f"candidate_mape={metrics['mape_percent']:.5f}% "
        f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
        f"pass={score.get('qualifies_development', score.get('qualifies_screen'))} "
        f"run={result['run_id']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
