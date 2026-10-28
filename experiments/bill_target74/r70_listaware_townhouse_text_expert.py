"""Screen a combined townhouse R22-text expert against validated R35."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CARD = "R70_listaware_townhouse_r22_text_expert"
TRACK = "list_aware"
HOUSE_TYPES = ("Freehold Townhouse", "Condo Townhouse")
COMPONENT = "townhouse_core_expert"
SCREEN_MONTHS = ("2023-03", "2024-06")
SPECIALIST_WEIGHT = 0.75
MIN_TRAINING_ROWS = 500
MIN_EVALUATION_ROWS = 100
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261026


def _route_mask(frame: pd.DataFrame) -> np.ndarray:
    if "house_type_name" not in frame:
        raise ValueError("townhouse specialist routing requires house_type_name")
    return (frame["house_type_name"].astype("string")
            .isin(HOUSE_TYPES).fillna(False).to_numpy(dtype=bool))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == _sha256(Path(__file__)):
            return existing
        raise RuntimeError(f"R70 already exists with status {existing.get('status')}; inspect before rerunning")
    parent = campaign.PROTOCOL["cards"].get("R35_listaware_detached_r22_text_expert")
    if not parent or parent.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R70 requires the validated R35 list-aware parent")
    source = budget._source_identity()
    route_support = {
        "2023-03": {"training_rows": 7629, "evaluation_rows": 236},
        "2024-06": {"training_rows": 10716, "evaluation_rows": 231},
    }
    registration = {
        "status": "registered",
        "track": TRACK,
        "tracks": [TRACK],
        "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS),
        "parent": "R35_listaware_detached_r22_text_expert",
        "parent_run_id": parent["run_id"],
        "house_types": list(HOUSE_TYPES),
        "route_support": route_support,
        "specialist_weight": SPECIALIST_WEIGHT,
        "minimum_training_rows": MIN_TRAINING_ROWS,
        "minimum_evaluation_rows_per_month": MIN_EVALUATION_ROWS,
        "model_family": "lightgbm",
        "model_config_overrides": {},
        "method": (
            "Keep the validated R35 list-aware predictions on all non-townhouse rows. Fit one existing G01 "
            "LightGBM core on the combined Freehold Townhouse and Condo Townhouse training rows, with R22 "
            "fold-local word TF-IDF and the unchanged structured/history features, target, and cutoff. "
            "On those two townhouse categories only, blend 75% specialist core with the verified frozen "
            "HGBR component; preserve the 75/25 inference blend and all other R35 rows."
        ),
        "difference_from_prior_work": (
            "R35 tested Detached, R36 Semi-Detached, and R43 Condo Apt list-aware experts; no registered "
            "candidate trains a shared expert across both townhouse categories. Safe eight-month R35 "
            "diagnostics show unchanged Freehold Townhouse MdAPE 5.8805% (n=460) and Condo Townhouse "
            "MdAPE 4.5230% (n=1,052); both subgroup medians remain above the <4% target, while Freehold "
            "Townhouse is also worse than the 4.5785% pooled R35 score. The combined route has 236/231 "
            "screen rows and 7,629/10,716 "
            "training rows."
        ),
        "incremental_screen_gate": {
            "relative_mdape_gain_min_percent": 0.25,
            "both_screen_months_improve": True,
            "mape_regression_max_percent": 1.0,
            "paired_home_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "paired_home_bootstrap_seed": BOOTSTRAP_SEED,
            "paired_home_bootstrap_upper95_below_zero": True,
        },
        "policy": (
            "Run only 2023-03 and 2024-06; fit the TF-IDF vocabulary on routed training remarks only; "
            "train strictly before each cutoff; verify exact routed support, R35 parent hashes, frozen HGBR "
            "provenance, and unchanged non-townhouse predictions. Pass only if pooled MdAPE improves by "
            "at least 0.25%, both months improve, relative MAPE regression is at most 1%, and paired-home "
            "bootstrap upper 95% is below zero. Otherwise keep R35 and do not run confirmation. Never read "
            "2025-05/08 or hidden labels."
        ),
        "source_sha256": _sha256(Path(__file__)),
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_registered", "card": CARD, "track": TRACK,
        "parent_run_id": parent["run_id"], "house_types": list(HOUSE_TYPES),
        "screen_months": list(SCREEN_MONTHS), "route_support": route_support,
        "expected_fit_starts": len(SCREEN_MONTHS),
        "source_sha256": registration["source_sha256"], "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r70_registered",
        next_action="run the preregistered two-month combined-townhouse expert screen against R35",
        r70_card=CARD,
    )
    return registration


def _verified_r35_parent(campaign):
    card = campaign.PROTOCOL["cards"].get("R35_listaware_detached_r22_text_expert")
    if not card or card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R70 requires R35 to pass development and fixed-origin confirmation")
    screen_path = campaign.ROOT / card["result_path"]
    screen_rows_path = screen_path.parent / "paired_predictions.csv"
    screen_manifest_path = screen_path.parent / "run_manifest.json"
    fixed_info = card.get("fixed_origin_confirmation", {})
    fixed_path = campaign.ROOT / fixed_info.get("result_path", "")
    fixed_manifest_path = fixed_path.parent / "run_manifest.json"
    development_path = campaign.ROOT / card["development_result_path"]
    development_rows_path = development_path.parent / "paired_predictions.csv"
    development_manifest_path = development_path.parent / "run_manifest.json"
    paths = [screen_path, screen_rows_path, screen_manifest_path, fixed_path,
             fixed_manifest_path, development_path, development_rows_path,
             development_manifest_path]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("one or more verified R35 parent artifacts are missing")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    screen_manifest = json.loads(screen_manifest_path.read_text(encoding="utf-8"))
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    fixed_manifest = json.loads(fixed_manifest_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    checks = [
        (screen_manifest.get("result_sha256") == _sha256(screen_path)
         and screen_manifest.get("paired_predictions_sha256") == _sha256(screen_rows_path)),
        (fixed_manifest.get("result_sha256") == _sha256(fixed_path)
         and fixed_manifest.get("paired_predictions_sha256") == _sha256(fixed_path.parent / "paired_predictions.csv")),
        (development_manifest.get("result_sha256") == _sha256(development_path)
         and development_manifest.get("paired_predictions_sha256") == _sha256(development_rows_path)),
    ]
    if not all(checks):
        raise RuntimeError("R35 parent artifacts fail result or paired-row hash verification")
    if (screen.get("run_id") != card.get("run_id")
            or screen.get("track") != TRACK
            or screen.get("tracks", {}).get(TRACK, {}).get("qualifies_screen") is not True
            or fixed.get("passes_fixed_origin") is not True
            or fixed.get("target_met") is not False
            or development.get("tracks", {}).get(TRACK, {}).get("target_met") is not False):
        raise RuntimeError("R35 is not the expected validated, still-above-target list-aware parent")
    if (screen.get("reserved_months_labels_opened")
            or fixed.get("reserved_months_labels_opened")
            or development.get("reserved_months_labels_opened")
            or screen.get("hidden_labels_opened")
            or fixed.get("hidden_labels_opened")
            or development.get("hidden_labels_opened")):
        raise RuntimeError("R35 parent artifacts report reserve or hidden-label access")
    rows = pd.read_csv(screen_rows_path, dtype={"ml_num": str})
    rows = rows.loc[rows.track.eq(TRACK)].copy()
    if set(rows.month.astype(str)) != set(SCREEN_MONTHS):
        raise RuntimeError("R35 screen rows do not cover both preregistered R70 months")
    return {
        "screen_path": screen_path, "screen_rows_path": screen_rows_path,
        "screen_manifest_path": screen_manifest_path, "screen": screen,
        "screen_manifest": screen_manifest, "screen_rows": rows,
        "fixed_path": fixed_path, "fixed_manifest_path": fixed_manifest_path,
        "fixed": fixed, "fixed_manifest": fixed_manifest,
        "development_path": development_path,
        "development_rows_path": development_rows_path,
        "development_manifest_path": development_manifest_path,
        "development": development, "development_manifest": development_manifest,
    }


def _source_files(campaign) -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r34_house_type_text_experts.py",
        campaign.PACKAGE / "r35_fixed_origin.py",
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


def run() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as r03
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as scoring
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74.r19_expert_weight import _write_json as write_json
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R70 must be preregistered before fitting")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R70 source changed after preregistration")
    parent = _verified_r35_parent(campaign)
    if card.get("parent_run_id") != parent["screen"]["run_id"]:
        raise RuntimeError("R70 parent run differs from the preregistered R35 screen run")
    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active data hash differs from the preregistered campaign")

    run_id = "bill_target74_r70_listaware_townhouse_text_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for source_path, name in [
        (parent["screen_path"], "r35_screen_results.json"),
        (parent["screen_rows_path"], "r35_screen_paired_predictions.csv"),
        (parent["screen_manifest_path"], "r35_screen_run_manifest.json"),
        (parent["development_path"], "r35_development_results.json"),
        (parent["development_rows_path"], "r35_development_paired_predictions.csv"),
        (parent["development_manifest_path"], "r35_development_run_manifest.json"),
        (parent["fixed_path"], "r35_fixed_origin_results.json"),
        (parent["fixed_manifest_path"], "r35_fixed_origin_run_manifest.json"),
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
        "parent_run_id": parent["screen"]["run_id"],
        "house_types": list(HOUSE_TYPES), "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS),
        "specialist_weight": SPECIALIST_WEIGHT,
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
            "component": COMPONENT, "house_types": list(HOUSE_TYPES),
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
                raise RuntimeError(f"R70 training rows reach the {month} cutoff")
            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            training_mask = _route_mask(train)
            route_mask = _route_mask(homes)
            fit_rows = train.loc[training_mask].copy()
            eval_rows = homes.loc[route_mask].copy()
            expected_support = campaign.PROTOCOL["cards"][CARD].get("route_support", {}).get(month)
            actual_support = {
                "training_rows": int(len(fit_rows)),
                "evaluation_rows": int(route_mask.sum()),
            }
            if expected_support is not None and actual_support != expected_support:
                raise RuntimeError(
                    f"R70 preregistered route support changed for {month}: "
                    f"expected {expected_support}, found {actual_support}"
                )
            if len(fit_rows) < MIN_TRAINING_ROWS or int(route_mask.sum()) < MIN_EVALUATION_ROWS:
                raise RuntimeError(f"R70 {month} route is too small")

            reference = scoring.r22_reference_frame(parent["screen_rows"], truth, month, TRACK)
            transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
            fit_context.update(month=month, training_rows=len(fit_rows),
                               evaluation_rows=len(eval_rows),
                               evaluation_ids=eval_rows.ml_num.astype(str).tolist())
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows, fit_rows=fit_rows,
                    variant=r22.VARIANT,
                    fit_tag=f"R70-{month}-Townhouse",
                )
            finally:
                restore()

            all_hgbr = r03._verified_hgbr_cache(greedy, month, TRACK, train, homes)
            specialist = blend_prices(fitted["predictions"], all_hgbr[route_mask],
                                      challenger_weight=SPECIALIST_WEIGHT)
            parent_values = reference.predicted_price.to_numpy(float)
            prediction = parent_values.copy()
            prediction[route_mask] = specialist
            if not np.array_equal(prediction[~route_mask], parent_values[~route_mask]):
                raise RuntimeError(f"R70 changed non-townhouse homes in {month}")
            candidate = pd.DataFrame({
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "predicted_price": prediction,
            })
            paired = pd.DataFrame({
                "month": month, "track": TRACK,
                "ml_num": candidate.ml_num,
                "sold_price": candidate.sold_price,
                "parent_prediction": parent_values,
                "candidate_prediction": candidate.predicted_price,
            })
            paired_parts.append(paired)
            route_truth = truth.sold_price.to_numpy(float)[route_mask]
            route_details[month] = {
                "training_rows": int(len(fit_rows)), "evaluation_rows": int(route_mask.sum()),
                "parent": campaign._metrics(route_truth, parent_values[route_mask]),
                "candidate": campaign._metrics(route_truth, prediction[route_mask]),
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
            "r70_screen_failed", next_action="inspect R70 failure checkpoint before retrying",
            latest_fit_count=len(fit_details),
        )
        raise
    finally:
        greedy._fit_record = old_fit_record

    if len(fit_details) != len(SCREEN_MONTHS):
        raise RuntimeError(f"R70 expected {len(SCREEN_MONTHS)} fits, recorded {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    rows = r22.score_development_confirmation(
        paired, months=SCREEN_MONTHS, bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED, minimum_relative_gain_percent=0.25,
        minimum_months_improved=2, maximum_worst_month_regression_percent=0.0,
        maximum_mape_regression_percent=1.0,
    )
    rows["qualifies_screen"] = rows.pop("qualifies_development")
    passed = bool(rows["qualifies_screen"])
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "track": TRACK,
        "parent_run_id": parent["screen"]["run_id"],
        "parent_candidate": "R35_listaware_detached_r22_text_expert",
        "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
        "specialist": {"house_types": list(HOUSE_TYPES), "weight": SPECIALIST_WEIGHT,
                       "training_rows_are_routed_house_type_only": True,
                       "tfidf_is_fit_on_routed_remarks_only": True},
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
        "parent_run_id": parent["screen"]["run_id"],
        "parent_screen_result_sha256": _sha256(parent["screen_path"]),
        "parent_screen_predictions_sha256": _sha256(parent["screen_rows_path"]),
        "parent_screen_manifest_sha256": _sha256(parent["screen_manifest_path"]),
        "parent_development_result_sha256": _sha256(parent["development_path"]),
        "parent_fixed_origin_result_sha256": _sha256(parent["fixed_path"]),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report_lines = [
        "# R70：list-aware 合并 Townhouse R22-text 专家筛选", "",
        f"Run：`{run_id}`；比较对象是已通过固定起点检查的 R35 run `{parent['screen']['run_id']}`。", "",
        "候选只在 Freehold Townhouse 与 Condo Townhouse 评估行替换 R35 预测。专家训练与 TF-IDF 词表只使用这两个房型的训练行，"
        "并固定采用 75% 专家 + 25% 已验证 HGBR。", "",
        f"R35 pooled MdAPE：{rows['parent']['mdape_percent']:.5f}%；",
        f"R70 pooled MdAPE：{rows['candidate']['mdape_percent']:.5f}%；",
        f"相对改善：{rows['relative_mdape_improvement_percent']:+.3f}%；",
        f"MAPE 相对变化：{rows['mape_relative_change_percent']:+.3f}%。", "",
        f"两个筛选月均改善：{all(item['relative_mdape_improvement_percent'] > 0 for item in rows['monthly'].values())}；",
        f"配对 bootstrap 95% MdAPE 差值上界：{rows['bootstrap_mdape_difference_percent']['upper95']:+.5f} 个百分点。",
        f"筛选门槛通过：{passed}。", "",
        "只读取 2023-03 与 2024-06 开发折；未读取 2025-05/08 或隐藏标签。"
        "保留集因先前完整 parquet 载入事件而不具备独立验证资格。", "",
        f"完整预测、账本快照和哈希位于 `{output.relative_to(campaign.ROOT)}`。", "",
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
        "r70_screen_passed" if passed else "r70_screen_failed",
        next_action=("R70 passed screen; register eight-month development confirmation against R35"
                     if passed else "R70 failed screen; preserve R35 and register a distinct list-aware hypothesis"),
        r70_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=len(fit_details),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register or run the R70 combined-townhouse expert screen")
    parser.add_argument("--register-only", action="store_true",
                        help="register R70 without fitting either screen fold")
    args = parser.parse_args()
    registration = register_card()
    if args.register_only:
        print(json.dumps(registration, indent=2, ensure_ascii=False))
    else:
        outcome = run()
        score = outcome["tracks"][TRACK]
        print(
            f"R70 screen: parent_mdape={score['parent']['mdape_percent']:.5f}% "
            f"candidate_mdape={score['candidate']['mdape_percent']:.5f}% "
            f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
            f"pass={score['qualifies_screen']} run={outcome['run_id']}",
            flush=True,
        )
