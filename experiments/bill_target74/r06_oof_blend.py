from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as helpers
from experiments.bill_target74 import r04_ablation as r04
from experiments.bill_target74 import runner as campaign
from experiments.model_optimization_v2.pipeline import blend_prices

ROOT = campaign.ROOT
OOF_MONTHS = ("2023-07", "2023-11", "2024-03")
PARENTS = dict(campaign.PROTOCOL["parent_by_track"])
CARD = "R06_unit_history_past_oof_complementarity"
CORRELATION_LIMIT = 0.95
COMPONENT_METRIC_REGRESSION_LIMIT = 1.02
BLEND_CANDIDATE_WEIGHT = 0.25
RECOVERED_RUN_ID = "bill_target74_r06_unit_oof_20261024T042011Z"
RECOVERED_FIT_SOURCE_SHA256 = "79e61eabc739aa55a4dc853f8a5d64d2d0ce9278f2761c5de03e385423b73d48"


def r06_fit_id(run_id: str, track: str, month: str, component: str) -> str:
    if track not in {"off_market", "list_aware"}:
        raise ValueError(f"unknown track {track!r}")
    if month not in OOF_MONTHS:
        raise ValueError(f"month {month!r} is not a registered past OOF month")
    if component not in {"base", "expert", "core"}:
        raise ValueError(f"unknown component {component!r}")
    return f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{component}:{run_id}"


def qualifies_complementarity(*, rows: int, error_correlation: float | None,
                              parent_mdape: float, candidate_mdape: float,
                              parent_mape: float, candidate_mape: float) -> bool:
    metrics = (parent_mdape, candidate_mdape, parent_mape, candidate_mape)
    return bool(
        rows >= 1_000
        and error_correlation is not None
        and np.isfinite(error_correlation)
        and error_correlation < CORRELATION_LIMIT
        and all(np.isfinite(value) and value > 0.0 for value in metrics)
        and candidate_mdape <= COMPONENT_METRIC_REGRESSION_LIMIT * parent_mdape
        and candidate_mape <= COMPONENT_METRIC_REGRESSION_LIMIT * parent_mape
    )


def fixed_parent_blend(parent_prediction, candidate_prediction) -> np.ndarray:
    parent = np.asarray(parent_prediction, dtype=float)
    candidate = np.asarray(candidate_prediction, dtype=float)
    if parent.shape != candidate.shape or parent.ndim != 1:
        raise ValueError("parent and candidate predictions must be aligned vectors")
    if (not np.isfinite(parent).all() or not np.isfinite(candidate).all()
            or (parent <= 0.0).any() or (candidate <= 0.0).any()):
        raise ValueError("parent and candidate predictions must be finite positive prices")
    return blend_prices(candidate, parent, challenger_weight=BLEND_CANDIDATE_WEIGHT)


def format_correlation(value: float | None) -> str:
    return f"{value:.5f}" if value is not None and np.isfinite(value) else "n/a"


def concat_available_frames(frames: list[pd.DataFrame | None]) -> pd.DataFrame:
    available = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(available, ignore_index=True) if available else pd.DataFrame()


def fit_records_for_run(events: list[dict], run_id: str) -> list[dict]:
    scoped = [event for event in events if event.get("run_id") == run_id and event.get("card") == CARD]
    starts = [event for event in scoped if event.get("event") == "fit_start"]
    terminals = [event for event in scoped if event.get("event") == "fit_complete"]
    start_ids = [event.get("fit_id") for event in starts]
    terminal_ids = [event.get("fit_id") for event in terminals]
    if (None in start_ids or len(start_ids) != len(set(start_ids))
            or len(terminal_ids) != len(set(terminal_ids)) or set(start_ids) != set(terminal_ids)):
        raise RuntimeError("each fit start must have exactly one terminal event")
    failed = [event for event in terminals if event.get("status") != "complete"]
    if failed:
        raise RuntimeError("recovery requires every fit to have a successful terminal event")
    return terminals


def _source_identity() -> dict:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    if branch != "bill" or commit != campaign.PROTOCOL["parent_commit"]:
        raise RuntimeError(f"expected bill at {campaign.PROTOCOL['parent_commit']}, found {branch} at {commit}")
    if helpers._sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings parquet hash differs from the campaign data hash")
    return {"branch": branch, "commit": commit, "data_sha256": campaign.PROTOCOL["data_sha256"]}


def _r04_screen_artifact() -> tuple[pd.DataFrame, dict]:
    if not campaign.STATE.is_file():
        raise FileNotFoundError("campaign progress is missing; cannot identify the completed R04 screen")
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    relative = str(progress.get("r04_result", "")).replace("\\", "/")
    results_path = (ROOT / relative).resolve()
    campaign_root = campaign.CAMPAIGN_ROOT.resolve()
    if campaign_root not in results_path.parents or not results_path.is_file():
        raise RuntimeError("campaign progress does not point to an R04 result inside this campaign")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if result.get("card") != "R04_unit_history_ablation" or result.get("run_id") != results_path.parent.name:
        raise RuntimeError("progress points to an unexpected R04 result")
    paired_path = results_path.parent / "paired_predictions.csv"
    manifest_path = results_path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paired_hash = helpers._sha256(paired_path)
    if manifest.get("paired_prediction_csv_sha256") != paired_hash:
        raise RuntimeError("R04 screen prediction hash differs from its manifest")
    rows = pd.read_csv(paired_path, dtype={"ml_num": str})
    if not {"track", "month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"}.issubset(rows.columns):
        raise RuntimeError("R04 paired predictions lack required columns")
    return rows, {"run_id": result["run_id"], "results_sha256": helpers._sha256(results_path),
                  "paired_prediction_csv": str(paired_path.relative_to(ROOT)).replace("\\", "/"),
                  "paired_prediction_csv_sha256": paired_hash}


def _result(rows: pd.DataFrame, candidate_column: str) -> dict:
    parent_metric = campaign._metrics(rows.sold_price.to_numpy(float), rows.parent_prediction.to_numpy(float))
    candidate_metric = campaign._metrics(rows.sold_price.to_numpy(float), rows[candidate_column].to_numpy(float))
    monthly = {}
    for month in OOF_MONTHS:
        part = rows.loc[rows.month.eq(month)]
        p = campaign._metrics(part.sold_price.to_numpy(float), part.parent_prediction.to_numpy(float))
        c = campaign._metrics(part.sold_price.to_numpy(float), part[candidate_column].to_numpy(float))
        monthly[month] = {"parent": p, "candidate": c,
                          "relative_mdape_improvement_percent": campaign._relative_gain(
                              p["mdape_percent"], c["mdape_percent"])}
    actual = rows.sold_price.to_numpy(float)
    parent_error = np.log(rows.parent_prediction.to_numpy(float) / actual)
    candidate_error = np.log(rows[candidate_column].to_numpy(float) / actual)
    correlation = (float(np.corrcoef(parent_error, candidate_error)[0, 1])
                   if len(rows) >= 2 and np.std(parent_error) > 0 and np.std(candidate_error) > 0 else None)
    return {"rows": int(len(rows)), "months": list(OOF_MONTHS), "parent": parent_metric,
            "candidate": candidate_metric, "signed_log_error_correlation": correlation,
            "monthly": monthly,
            "qualifies_complementarity": qualifies_complementarity(
                rows=len(rows), error_correlation=correlation,
                parent_mdape=parent_metric["mdape_percent"],
                candidate_mdape=candidate_metric["mdape_percent"],
                parent_mape=parent_metric["mape_percent"],
                candidate_mape=candidate_metric["mape_percent"])}


def _report(result: dict) -> str:
    lines = [
        "# BILL <7% / <4% 持续研究：R06 R04 消融候选过去 OOF 互补性检查", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        f"过去 OOF 月份：{', '.join(result['oof_months'])}；真实 learner fit starts：{result['fit_starts']}。",
        "所有 OOF 折严格早于外层 2024-06 筛选折；预留月份和隐藏标签未读取。候选仅移除 R04 中的 `u_*` unit-history 特征。",
        "互补条件预先固定为：合计至少 1,000 行、父/候选有符号 log 相对误差相关性 <0.95、候选 pooled OOF MdAPE 与 MAPE 均不高于父版的 102%。",
        "", "## 过去 OOF 结果", "",
        "| 赛道 | 行数 | 父 MdAPE | 候选 MdAPE | 父 MAPE | 候选 MAPE | log误差相关性 | 互补条件通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for track, score in result["oof_tracks"].items():
        corr = score["signed_log_error_correlation"]
        corr_text = format_correlation(corr)
        lines.append(
            f"| {track} | {score['rows']} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | {score['parent']['mape_percent']:.5f}% | "
            f"{score['candidate']['mape_percent']:.5f}% | {corr_text} | "
            f"{'是' if score['qualifies_complementarity'] else '否'} |"
        )
    lines.extend(["", "## 固定 75/25 双月筛选", "", "权重事先固定为 75% 父模型 + 25% R04 候选；没有按筛选月调权。", "",
                  "| 赛道 | 父 MdAPE | 固定融合 MdAPE | 相对改善 | MAPE变化 | 筛选通过 |",
                  "|---|---:|---:|---:|---:|---|"])
    for track, score in result["blend_screen"].items():
        if score is None:
            lines.append(f"| {track} | — | — | — | — | 未评估：OOF互补门槛未过 |")
        else:
            lines.append(
                f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
                f"{score['candidate']['mdape_percent']:.5f}% | "
                f"{score['relative_mdape_improvement_percent']:+.3f}% | "
                f"{score['mape_relative_change_percent']:+.3f}% | "
                f"{'是' if score['qualifies_screen'] else '否'} |"
            )
    lines.extend(["", "## 决策与边界", "",
                  result["decision"],
                  "该检查用于选择是否进入下一步，不构成独立测试；OOF 开发月份此前参与过研究。候选不会因通过互补性检查而自动晋级。",
                  "历史父预测文件存在已记录的 sidecar 代码哈希差异，仍仅作为冻结 walk-forward 比较器。",
                  "", "- OOF 逐房预测：`paired_oof_predictions.csv`",
                  "- 双月固定融合结果（若触发）：`paired_blend_screen_predictions.csv`",
                  "- 来源、参数和 fit ledger 摘要：`run_manifest.json` / campaign `fits.jsonl`", ""])
    return "\n".join(lines)


def run_r06_oof_complementarity() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2 import worker

    source = _source_identity()
    started_at = datetime.now(timezone.utc)
    run_id = "bill_target74_r06_unit_oof_" + started_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "oof_months": list(OOF_MONTHS), "fit_start_count": 0,
                      "component": "R04_unit_history_ablation", "outer_screen_reused_only_after_oof_gate": True})

    original_normalize = worker._normalize_categories
    original_fit_record = greedy._fit_record
    original_hgbr = greedy._frozen_hgbr_predictions
    active_context: dict[str, str] = {}
    fit_details: list[dict] = []
    removed_features: set[str] = set()
    component_manifest: list[dict] = []
    rows_by_track: dict[str, list[pd.DataFrame]] = {track: [] for track in PARENTS}

    def normalize_without_unit(train_x, eval_x, family):
        train_norm, eval_norm, categories = original_normalize(train_x, eval_x, family)
        train_out, eval_out, categories_out, removed = r04.remove_unit_history_features(
            train_norm, eval_norm, categories)
        active_context["removed_features"] = removed
        return train_out, eval_out, categories_out

    def isolated_fit_record(parent_fit_id: str, fit_fn):
        limits = campaign.PROTOCOL["limits"]
        current_fits = r04._fit_count()
        elapsed_campaign = r04._campaign_elapsed_seconds()
        elapsed_batch = (datetime.now(timezone.utc) - started_at).total_seconds()
        if campaign.limit_reached(current_fits, limits["exploration_fit_starts"]):
            raise RuntimeError("R06 exploration fit-start reserve reached")
        if campaign.limit_exceeded(elapsed_campaign,
                                   None if limits["exploration_minutes"] is None
                                   else limits["exploration_minutes"] * 60):
            raise RuntimeError("R06 exploration wall-clock reserve reached")
        if campaign.limit_exceeded(elapsed_batch,
                                   None if limits["batch_minutes"] is None
                                   else limits["batch_minutes"] * 60):
            raise RuntimeError("R06 batch wall-clock cap reached before fit start")
        context = dict(active_context)
        fit_id = r06_fit_id(run_id, context["track"], context["month"], context["component"])
        campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "parent_fit_id": parent_fit_id, **context,
                          "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                          "data_sha256": source["data_sha256"]})
        timer = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "status": "failed", "seconds": time.perf_counter() - timer,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        seconds = time.perf_counter() - timer
        status = "complete" if not campaign.limit_exceeded(seconds, limits["single_fit_seconds"]) else "over_limit"
        detail = {"fit_id": fit_id, "parent_fit_id": parent_fit_id, **context, "seconds": seconds,
                  "status": status, "feature_columns": fitted["feature_columns"],
                  "prediction_sha256": fitted["prediction_sha256"], "training_rows": fitted["training_rows"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, **detail})
        if status != "complete":
            raise TimeoutError(f"fit finished after {limits['single_fit_seconds']}s limit")
        return fitted

    worker._normalize_categories = normalize_without_unit
    greedy._fit_record = isolated_fit_record
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: helpers._verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in OOF_MONTHS:
            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = (pd.Period(month, freq="M") + 1).start_time
            if pd.to_datetime(listings.date_sold, errors="raise").max() >= cutoff:
                raise RuntimeError(f"fold input includes sales at or after next-month cutoff: {month}")
            for track, parent_id in PARENTS.items():
                homes = hide_answers(heldout, track)
                parent, parent_path = r04._check_parent_artifact(greedy, month, track, homes, truth, parent_id)
                if track == "off_market":
                    base_context = {"track": track, "month": month, "component": "base"}
                    active_context.clear(); active_context.update(base_context)
                    base_fit = greedy._fit_history_candidate(
                        month, track, train, homes, base_variant="G06_lot_garage",
                        fit_tag=f"R06unit-base-{month}")
                    removed_features.update(active_context["removed_features"])
                    component_manifest.append({**base_context, "parent_id": parent_id,
                                               "feature_parent": "G06_lot_garage",
                                               "feature_columns": base_fit["feature_columns"],
                                               "model_family": base_fit["model_family"],
                                               "model_config": base_fit["model_config"],
                                               "training_rows": base_fit["training_rows"]})
                    base_prediction = np.round(base_fit["predictions"])
                    train_mask = greedy._specialist_mask(train, "Detached")
                    eval_mask = greedy._specialist_mask(homes, "Detached")
                    specialist_train = train.loc[train_mask].copy()
                    specialist_homes = homes.loc[eval_mask].copy()
                    if specialist_train.empty or not eval_mask.any():
                        raise RuntimeError(f"Detached specialist has no data in {month}")
                    expert_context = {"track": track, "month": month, "component": "expert"}
                    active_context.clear(); active_context.update(expert_context)
                    expert_fit = greedy._fit_history_candidate(
                        month, track, train, specialist_homes, base_variant="G06_lot_garage",
                        fit_tag=f"R06unit-expert-{month}", fit_rows=specialist_train)
                    removed_features.update(active_context["removed_features"])
                    component_manifest.append({**expert_context, "parent_id": parent_id,
                                               "feature_parent": "G06_lot_garage",
                                               "feature_columns": expert_fit["feature_columns"],
                                               "model_family": expert_fit["model_family"],
                                               "model_config": expert_fit["model_config"],
                                               "training_rows": expert_fit["training_rows"]})
                    candidate_prediction = greedy._blend_specialist_predictions(
                        base_prediction, np.round(expert_fit["predictions"]), homes, "Detached", 0.75)
                else:
                    core_context = {"track": track, "month": month, "component": "core"}
                    active_context.clear(); active_context.update(core_context)
                    core_fit = greedy._fit_history_candidate(
                        month, track, train, homes, base_variant=None,
                        fit_tag=f"R06unit-core-{month}")
                    removed_features.update(active_context["removed_features"])
                    component_manifest.append({**core_context, "parent_id": parent_id,
                                               "feature_parent": "G01", "feature_columns": core_fit["feature_columns"],
                                               "model_family": core_fit["model_family"],
                                               "model_config": core_fit["model_config"],
                                               "training_rows": core_fit["training_rows"]})
                    frozen = helpers._verified_hgbr_cache(greedy, month, track, train, homes)
                    candidate_prediction = np.round(0.25 * np.round(frozen) + 0.75 * core_fit["predictions"])

                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": candidate_prediction})
                paired = helpers._candidate_frame(month, track, truth, parent, candidate)
                rows_by_track[track].append(paired)
                campaign._append({"event": "oof_fold_complete", "card": CARD, "run_id": run_id,
                                  "track": track, "month": month, "parent_id": parent_id,
                                  "parent_csv_sha256": helpers._sha256(parent_path),
                                  "removed_features": sorted(active_context["removed_features"]),
                                  "date_cutoff_exclusive": cutoff.isoformat()})
    finally:
        worker._normalize_categories = original_normalize
        greedy._fit_record = original_fit_record
        greedy._frozen_hgbr_predictions = original_hgbr

    paired_oof = pd.concat([frame for frames in rows_by_track.values() for frame in frames], ignore_index=True)
    oof_path = output_dir / "paired_oof_predictions.csv"
    paired_oof.to_csv(oof_path, index=False)
    oof_tracks = {track: _result(pd.concat(frames, ignore_index=True), "candidate_prediction")
                  for track, frames in rows_by_track.items()}

    r04_screen, r04_identity = _r04_screen_artifact()
    blend_rows_by_track: dict[str, pd.DataFrame | None] = {track: None for track in PARENTS}
    blend_screen = {track: None for track in PARENTS}
    eligible_tracks = [track for track, score in oof_tracks.items() if score["qualifies_complementarity"]]
    for track in eligible_tracks:
        screen_track = r04_screen.loc[r04_screen.track.eq(track)].copy()
        if set(screen_track.month.astype(str)) != set(campaign.PROTOCOL["screen_months"]):
            raise RuntimeError(f"R04 screen months are incomplete for {track}")
        screen_track["candidate_prediction"] = fixed_parent_blend(
            screen_track.parent_prediction.to_numpy(float), screen_track.candidate_prediction.to_numpy(float))
        blend_rows_by_track[track] = screen_track
        blend_screen[track] = helpers._track_result(screen_track)
    paired_blend = concat_available_frames(list(blend_rows_by_track.values()))
    blend_path = output_dir / "paired_blend_screen_predictions.csv"
    if not paired_blend.empty:
        paired_blend.to_csv(blend_path, index=False)
    else:
        blend_path = None

    qualified_blends = [track for track, score in blend_screen.items()
                        if score is not None and score["qualifies_screen"]]
    if qualified_blends:
        decision = "Past OOF complementarity and the fixed-weight two-month screen both passed for: " + ", ".join(qualified_blends) + ". Advance those tracks to the registered eight-month development confirmation; do not promote from this screen alone."
        next_action = "run eight-month development confirmation for fixed 75/25 blend: " + ", ".join(qualified_blends)
    elif eligible_tracks:
        decision = "At least one R04 component passed the past OOF complementarity gate, but the fixed 75/25 two-month blend screen failed. Do not promote; proceed to the registered coordinate/comparable-home feature-group ablation."
        next_action = "fixed R04 75/25 blend failed screen; test pre-registered coordinate/comparable-home feature ablation"
    else:
        decision = "Both tracks met the standalone OOF quality allowance, but neither met the <0.95 error-correlation threshold. No blend screen was run. Proceed to the registered coordinate/comparable-home feature-group ablation."
        next_action = "R04 component not complementary on past OOF; test pre-registered coordinate/comparable-home feature ablation"

    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
              "oof_months": list(OOF_MONTHS), "parents": PARENTS,
              "removed_features": sorted(removed_features), "oof_tracks": oof_tracks,
              "blend_candidate_weight": BLEND_CANDIDATE_WEIGHT, "blend_screen": blend_screen,
              "fit_starts": len(fit_details), "reserved_months_labels_opened": False,
              "hidden_labels_opened": False, "data_sha256": source["data_sha256"],
              "decision": decision, "next_action": next_action, "r04_screen_identity": r04_identity}
    results_path = output_dir / "r06_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (output_dir / "component_manifest.json").write_text(
        json.dumps(component_manifest, indent=2, default=str), encoding="utf-8")
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r06_oof_blend.py",
                    campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "r03_catboost.py",
                    ROOT / "experiments/bill_greedy/runner.py", ROOT / "experiments/bill_greedy/history.py",
                    ROOT / "experiments/model_optimization_v2/worker.py",
                    ROOT / "experiments/model_optimization_v2/learners.py",
                    ROOT / "experiments/model_optimization_v2/pipeline.py",
                    ROOT / "model_candidate_v2/features.py", ROOT / "model_candidate_v2/winner_config.json"]
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source,
        "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): helpers._sha256(path)
                        for path in source_files},
        "runtime": {"python": platform.python_version(), "lightgbm": metadata.version("lightgbm"),
                    "catboost": metadata.version("catboost"), "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit_learn": metadata.version("scikit-learn")},
        "parent_artifacts": {
            f"{track}/{month}": {"path": str(campaign._row_prediction_path(month, track).relative_to(ROOT)),
                                 "csv_sha256": helpers._sha256(campaign._row_prediction_path(month, track)),
                                 "sidecar_sha256": helpers._sha256(campaign._row_prediction_path(month, track).with_suffix(".json"))}
            for track in PARENTS for month in OOF_MONTHS},
        "frozen_hgbr_artifacts": {
            month: {"path": str((greedy.OUTPUT / "frozen_hgbr/list_aware" / f"{month}.csv").relative_to(ROOT)),
                    "csv_sha256": helpers._sha256(greedy.OUTPUT / "frozen_hgbr/list_aware" / f"{month}.csv")}
            for month in OOF_MONTHS},
        "r04_screen_identity": r04_identity,
        "paired_oof_prediction_csv_sha256": helpers._sha256(oof_path),
        "paired_blend_screen_prediction_csv_sha256": helpers._sha256(blend_path) if blend_path else None,
        "results_sha256": helpers._sha256(results_path), "fit_starts": len(fit_details),
        "fit_details": fit_details}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "research_report.md").write_text(_report(result), encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "qualifies_by_track": {
                          track: score["qualifies_complementarity"] for track, score in oof_tracks.items()},
                      "blend_screen_by_track": {track: score["qualifies_screen"] if score else None
                                                 for track, score in blend_screen.items()},
                      "result_path": str(results_path.relative_to(ROOT)).replace("\\", "/"),
                      "paired_oof_prediction_csv_sha256": manifest["paired_oof_prediction_csv_sha256"]})
    campaign._write_progress("r06_oof_complete", next_action=next_action,
                             r06_result=str(results_path.relative_to(ROOT)).replace("\\", "/"))
    return result


def recover_r06_after_scoring_failure(run_id: str = RECOVERED_RUN_ID) -> dict:
    """Finish scoring an already-fit OOF run without starting or repeating any learner fits."""
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy

    output_dir = campaign.CAMPAIGN_ROOT / "batches" / run_id
    oof_path = output_dir / "paired_oof_predictions.csv"
    results_path = output_dir / "r06_results.json"
    if results_path.exists():
        raise FileExistsError(f"R06 results already exist for {run_id}")
    if not oof_path.is_file():
        raise FileNotFoundError(f"saved OOF predictions are missing for {run_id}")

    raw_events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    fit_details = fit_records_for_run(raw_events, run_id)
    expected_components = {
        (track, month, component)
        for month in OOF_MONTHS for track in PARENTS
        for component in (("base", "expert") if track == "off_market" else ("core",))
    }
    found_components = {(row.get("track"), row.get("month"), row.get("component")) for row in fit_details}
    if found_components != expected_components or len(fit_details) != 9:
        raise RuntimeError("saved R06 fit ledger does not contain the complete nine-fit OOF design")

    source = _source_identity()
    paired_oof = pd.read_csv(oof_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"}
    if not required.issubset(paired_oof.columns):
        raise RuntimeError("saved OOF predictions lack required columns")
    if set(paired_oof.month.astype(str)) != set(OOF_MONTHS) or set(paired_oof.track.astype(str)) != set(PARENTS):
        raise RuntimeError("saved OOF predictions do not cover exactly the registered months and tracks")

    # Re-validate row IDs, labels, and frozen parent vectors against the source fold files.
    for month in OOF_MONTHS:
        _, _, _, truth, heldout = greedy._load_fold(month)
        for track, parent_id in PARENTS.items():
            homes = hide_answers(heldout, track)
            parent, _ = r04._check_parent_artifact(greedy, month, track, homes, truth, parent_id)
            saved = paired_oof.loc[paired_oof.month.eq(month) & paired_oof.track.eq(track)].copy()
            ids = homes.ml_num.astype(str).to_numpy()
            if (saved.ml_num.astype(str).duplicated().any()
                    or not np.array_equal(saved.ml_num.astype(str).to_numpy(), ids)
                    or not np.array_equal(saved.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
                    or not np.array_equal(saved.parent_prediction.to_numpy(float), parent.predicted_price.to_numpy(float))
                    or (saved.candidate_prediction.to_numpy(float) <= 0.0).any()
                    or not np.isfinite(saved.candidate_prediction.to_numpy(float)).all()):
                raise RuntimeError(f"saved OOF predictions failed source alignment for {month}/{track}")

    oof_tracks = {track: _result(paired_oof.loc[paired_oof.track.eq(track)].copy(), "candidate_prediction")
                  for track in PARENTS}
    r04_screen, r04_identity = _r04_screen_artifact()
    blend_rows_by_track: dict[str, pd.DataFrame | None] = {track: None for track in PARENTS}
    blend_screen = {track: None for track in PARENTS}
    eligible = [track for track, score in oof_tracks.items() if score["qualifies_complementarity"]]
    for track in eligible:
        screen_track = r04_screen.loc[r04_screen.track.eq(track)].copy()
        if set(screen_track.month.astype(str)) != set(campaign.PROTOCOL["screen_months"]):
            raise RuntimeError(f"R04 screen months are incomplete for {track}")
        screen_track["candidate_prediction"] = fixed_parent_blend(
            screen_track.parent_prediction.to_numpy(float), screen_track.candidate_prediction.to_numpy(float))
        blend_rows_by_track[track] = screen_track
        blend_screen[track] = helpers._track_result(screen_track)
    paired_blend = concat_available_frames(list(blend_rows_by_track.values()))
    blend_path = output_dir / "paired_blend_screen_predictions.csv" if not paired_blend.empty else None
    if blend_path is not None:
        paired_blend.to_csv(blend_path, index=False)

    qualified = [track for track, score in blend_screen.items()
                 if score is not None and score["qualifies_screen"]]
    if qualified:
        decision = "Past OOF complementarity and the fixed-weight two-month screen both passed for: " + ", ".join(qualified) + ". Advance those tracks to the registered eight-month development confirmation; do not promote from this screen alone."
        next_action = "run eight-month development confirmation for fixed 75/25 blend: " + ", ".join(qualified)
    elif eligible:
        decision = "At least one R04 component passed the past OOF complementarity gate, but the fixed 75/25 two-month blend screen failed. Do not promote; proceed to the registered coordinate/comparable-home feature-group ablation."
        next_action = "fixed R04 75/25 blend failed screen; test pre-registered coordinate/comparable-home feature ablation"
    else:
        decision = "Both tracks met the standalone OOF quality allowance, but neither met the <0.95 error-correlation threshold. No blend screen was run. Proceed to the registered coordinate/comparable-home feature-group ablation."
        next_action = "R04 component not complementary on past OOF; test pre-registered coordinate/comparable-home feature ablation"

    failure_recorded = any(event.get("event") == "card_finalize_failed" and event.get("run_id") == run_id
                           for event in raw_events)
    if not failure_recorded:
        campaign._append({"event": "card_finalize_failed", "card": CARD, "run_id": run_id,
                          "stage": "post_fit_scoring", "error": "ValueError: No objects to concatenate when neither track passed OOF complementarity",
                          "recovered_without_refit": True,
                          "candidate_generation_code_sha256": RECOVERED_FIT_SOURCE_SHA256})

    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
              "oof_months": list(OOF_MONTHS), "parents": PARENTS, "oof_tracks": oof_tracks,
              "blend_candidate_weight": BLEND_CANDIDATE_WEIGHT, "blend_screen": blend_screen,
              "fit_starts": len(fit_details), "reserved_months_labels_opened": False,
              "hidden_labels_opened": False, "data_sha256": source["data_sha256"],
              "decision": decision, "next_action": next_action, "r04_screen_identity": r04_identity,
              "scoring_recovered_without_refit": True}
    results_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    component_manifest = [{key: event.get(key) for key in
                           ("fit_id", "parent_fit_id", "track", "month", "component", "feature_columns",
                            "training_rows", "prediction_sha256", "seconds", "status")}
                          for event in fit_details]
    (output_dir / "component_manifest.json").write_text(json.dumps(component_manifest, indent=2), encoding="utf-8")
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r06_oof_blend.py",
                    campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "r03_catboost.py",
                    ROOT / "experiments/bill_greedy/runner.py", ROOT / "experiments/bill_greedy/history.py",
                    ROOT / "experiments/model_optimization_v2/worker.py",
                    ROOT / "experiments/model_optimization_v2/learners.py",
                    ROOT / "experiments/model_optimization_v2/pipeline.py",
                    ROOT / "model_candidate_v2/features.py", ROOT / "model_candidate_v2/winner_config.json"]
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "source_identity": source,
                "code_sha256_at_recovery": {str(path.relative_to(ROOT)).replace("\\", "/"):
                                             helpers._sha256(path) for path in source_files},
                "candidate_generation_code_sha256_at_fit_time": RECOVERED_FIT_SOURCE_SHA256,
                "scoring_recovered_without_refit": True,
                "runtime": {"python": platform.python_version(), "lightgbm": metadata.version("lightgbm"),
                            "catboost": metadata.version("catboost"), "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit_learn": metadata.version("scikit-learn")},
                "parent_artifacts": {
                    f"{track}/{month}": {"path": str(campaign._row_prediction_path(month, track).relative_to(ROOT)),
                                         "csv_sha256": helpers._sha256(campaign._row_prediction_path(month, track)),
                                         "sidecar_sha256": helpers._sha256(campaign._row_prediction_path(month, track).with_suffix(".json"))}
                    for track in PARENTS for month in OOF_MONTHS},
                "frozen_hgbr_artifacts": {
                    month: {"path": str((greedy.OUTPUT / "frozen_hgbr/list_aware" / f"{month}.csv").relative_to(ROOT)),
                            "csv_sha256": helpers._sha256(greedy.OUTPUT / "frozen_hgbr/list_aware" / f"{month}.csv")}
                    for month in OOF_MONTHS},
                "r04_screen_identity": r04_identity,
                "paired_oof_prediction_csv_sha256": helpers._sha256(oof_path),
                "paired_blend_screen_prediction_csv_sha256": helpers._sha256(blend_path) if blend_path else None,
                "results_sha256": helpers._sha256(results_path), "fit_starts": len(fit_details),
                "fit_details": fit_details,
                "audit_note": "The initial post-fit report step failed before results/manifest were written. This manifest was reconstructed from the append-only fit ledger and saved predictions; component configuration is available from the hashed fit source rather than stored as a serialized model."}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "research_report.md").write_text(_report(result), encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "qualifies_by_track": {
                          track: score["qualifies_complementarity"] for track, score in oof_tracks.items()},
                      "blend_screen_by_track": {track: score["qualifies_screen"] if score else None
                                                 for track, score in blend_screen.items()},
                      "result_path": str(results_path.relative_to(ROOT)).replace("\\", "/"),
                      "paired_oof_prediction_csv_sha256": manifest["paired_oof_prediction_csv_sha256"],
                      "recovered_without_refit": True})
    campaign._write_progress("r06_oof_complete", next_action=next_action,
                             r06_result=str(results_path.relative_to(ROOT)).replace("\\", "/"))
    return result


def main() -> None:
    result = (recover_r06_after_scoring_failure() if RECOVERED_RUN_ID
              and (campaign.CAMPAIGN_ROOT / "batches" / RECOVERED_RUN_ID / "paired_oof_predictions.csv").is_file()
              and not (campaign.CAMPAIGN_ROOT / "batches" / RECOVERED_RUN_ID / "r06_results.json").is_file()
              else run_r06_oof_complementarity())
    for track, score in result["oof_tracks"].items():
        print(f"{track}: OOF rows={score['rows']} parent MdAPE={score['parent']['mdape_percent']:.5f}% "
              f"candidate={score['candidate']['mdape_percent']:.5f}% "
              f"log_error_corr={score['signed_log_error_correlation']} "
              f"complementary={score['qualifies_complementarity']}", flush=True)
        blend = result["blend_screen"][track]
        if blend is not None:
            print(f"  fixed_blend MdAPE={blend['candidate']['mdape_percent']:.5f}% "
                  f"relative_gain={blend['relative_mdape_improvement_percent']:+.3f}% "
                  f"screen_pass={blend['qualifies_screen']}", flush=True)
    print(f"decision={result['decision']}", flush=True)
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
