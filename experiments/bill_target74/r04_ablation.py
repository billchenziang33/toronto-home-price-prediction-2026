from __future__ import annotations

import hashlib
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
from experiments.bill_target74 import runner as campaign

ROOT = campaign.ROOT
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
PARENTS = dict(campaign.PROTOCOL["parent_by_track"])
CARD = "R04_unit_history_ablation"


def remove_unit_history_features(train_x: pd.DataFrame, eval_x: pd.DataFrame,
                                 categories: list[str] | tuple[str, ...]):
    if list(train_x.columns) != list(eval_x.columns):
        raise ValueError("training and evaluation feature columns must align before ablation")
    removed = [str(column) for column in train_x.columns if str(column).startswith("u_")]
    keep = [column for column in train_x.columns if column not in removed]
    if not removed:
        raise ValueError("unit-history ablation found no u_-prefixed columns")
    train_out = train_x.loc[:, keep].copy()
    eval_out = eval_x.loc[:, keep].copy()
    categories_out = [column for column in categories if column not in removed]
    if list(train_out.columns) != list(eval_out.columns):
        raise RuntimeError("unit-history ablation produced misaligned features")
    return train_out, eval_out, categories_out, removed


def r04_fit_id(run_id: str, track: str, month: str, component: str) -> str:
    if track not in {"off_market", "list_aware"}:
        raise ValueError(f"unknown track {track!r}")
    if month not in SCREEN_MONTHS:
        raise ValueError(f"month {month!r} is not a registered screen fold")
    if component not in {"base", "expert", "core"}:
        raise ValueError(f"unknown component {component!r}")
    return f"{campaign.PROTOCOL['campaign_id']}:R04_unit_history_ablation:{track}:{month}:{component}:{run_id}"


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


def _check_parent_artifact(greedy, month: str, track: str, homes: pd.DataFrame,
                           truth: pd.DataFrame, parent_id: str) -> tuple[pd.DataFrame, Path]:
    path = campaign._row_prediction_path(month, track)
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if meta.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError(f"parent data hash mismatch for {month}/{track}")
    parent = greedy._load_parent_prediction(month, track, homes, truth, parent_id)
    return parent, path


def _fit_count() -> int:
    if not campaign.LEDGER.exists():
        return 0
    return sum(
        json.loads(line).get("event") == "fit_start"
        for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _campaign_elapsed_seconds() -> float:
    if not campaign.LEDGER.exists():
        return 0.0
    rows = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    timestamps = [row.get("timestamp_utc") for row in rows if row.get("timestamp_utc")]
    if not timestamps:
        return 0.0
    started = min(datetime.fromisoformat(value) for value in timestamps)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _report(result: dict) -> str:
    lines = [
        "# BILL <7% / <4% 持续研究：R04 unit-history 特征消融", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "唯一变化：移除所有 `u_*` unit-history 特征；保留 `b_*` building-history 与其余特征。",
        "Off-market 以同样的移除规则重训 G06 base 和 Detached expert，再按原 G19 75/25 路由；list-aware 重训 G01 LightGBM 核心并保留原 75/25 HGBR 混合。",
        "", "## 双月合并结果", "",
        "| 赛道 | 父 MdAPE | 候选 MdAPE | 相对改善 | MAPE变化 | 筛选通过 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for track, score in result["tracks"].items():
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | "
            f"{'是' if score['qualifies_screen'] else '否'} |"
        )
    lines.extend(["", "## 各筛选月份", "", "| 赛道 | 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 |", "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, monthly in score["monthly"].items():
            lines.append(
                f"| {track} | {month} | {monthly['parent']['mdape_percent']:.5f}% | "
                f"{monthly['candidate']['mdape_percent']:.5f}% | "
                f"{monthly['relative_mdape_improvement_percent']:+.3f}% |"
            )
    lines.extend([
        "", "## 运行边界", "",
        f"真实 learner fit starts：{result['fit_starts']}；移除列：`{', '.join(result['removed_features'])}`。",
        "只有通过双月 MdAPE、逐月方向和 MAPE 三项筛选的赛道才进入八个月开发确认；本筛选不改变父模型。",
        "预留折与隐藏标签保持封存。",
        "",
        "- 逐房配对结果：`paired_predictions.csv`",
        "- 每个模型组件的特征列和参数：`component_manifest.json`",
        "- 来源哈希与运行环境：`run_manifest.json`",
        "- Campaign 事件账本：`../../fits.jsonl`",
        "",
    ])
    return "\n".join(lines)


def run_r04_unit_history_screen() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2 import worker

    source = _source_identity()
    run_started_at = datetime.now(timezone.utc)
    run_id = "bill_target74_r04_unit_history_" + run_started_at.strftime("%Y%m%dT%H%M%SZ")
    batch_root = campaign.CAMPAIGN_ROOT / "batches" / run_id
    batch_root.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "fit_start_count": 0,
                      "removed_prefix": "u_", "retained_prefix": "b_"})

    original_normalize = worker._normalize_categories
    original_fit_record = greedy._fit_record
    original_hgbr = greedy._frozen_hgbr_predictions
    active_context: dict[str, str] = {}
    fit_details: list[dict] = []

    def normalize_without_unit(train_x, eval_x, family):
        train_norm, eval_norm, categories = original_normalize(train_x, eval_x, family)
        train_out, eval_out, categories_out, removed = remove_unit_history_features(
            train_norm, eval_norm, categories)
        active_context["removed_features"] = removed
        return train_out, eval_out, categories_out

    def isolated_fit_record(parent_fit_id: str, fit_fn):
        limits = campaign.PROTOCOL["limits"]
        if campaign.limit_reached(_fit_count(), limits["exploration_fit_starts"]):
            raise RuntimeError("R04 exploration fit-start reserve reached")
        if campaign.limit_exceeded(_campaign_elapsed_seconds(),
                                   None if limits["exploration_minutes"] is None
                                   else limits["exploration_minutes"] * 60):
            raise RuntimeError("R04 exploration wall-clock reserve reached")
        if campaign.limit_exceeded((datetime.now(timezone.utc) - run_started_at).total_seconds(),
                                   None if limits["batch_minutes"] is None
                                   else limits["batch_minutes"] * 60):
            raise RuntimeError("R04 batch wall-clock cap reached before fit start")
        context = dict(active_context)
        fit_id = r04_fit_id(run_id, context["track"], context["month"], context["component"])
        campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "parent_fit_id": parent_fit_id,
                          **context, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                          "data_sha256": source["data_sha256"]})
        timer = time.perf_counter()
        try:
            result = fit_fn()
        except Exception as exc:
            elapsed = time.perf_counter() - timer
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "status": "failed", "seconds": elapsed,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        elapsed = time.perf_counter() - timer
        status = ("complete" if not campaign.limit_exceeded(
            elapsed, campaign.PROTOCOL["limits"]["single_fit_seconds"]) else "over_limit")
        record = {"fit_id": fit_id, "parent_fit_id": parent_fit_id, **context,
                  "seconds": elapsed, "status": status, "feature_columns": result["feature_columns"],
                  "prediction_sha256": result["prediction_sha256"],
                  "training_rows": result["training_rows"]}
        fit_details.append(record)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, **record})
        if status != "complete":
            raise TimeoutError(f"fit finished after the registered {campaign.PROTOCOL['limits']['single_fit_seconds']}s ceiling")
        return result

    worker._normalize_categories = normalize_without_unit
    greedy._fit_record = isolated_fit_record
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: helpers._verified_hgbr_cache(
        greedy, month, track, train, homes)
    rows_by_track: dict[str, list[pd.DataFrame]] = {track: [] for track in PARENTS}
    removed_features: set[str] = set()
    component_manifest = []
    try:
        for month in SCREEN_MONTHS:
            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = (pd.Period(month, freq="M") + 1).start_time
            if pd.to_datetime(listings.date_sold, errors="raise").max() >= cutoff:
                raise RuntimeError(f"fold input includes sales at or after the next-month cutoff: {month}")
            for track, parent_id in PARENTS.items():
                homes = hide_answers(heldout, track)
                parent, parent_path = _check_parent_artifact(greedy, month, track, homes, truth, parent_id)
                if track == "off_market":
                    base_context = {"track": track, "month": month, "component": "base"}
                    active_context.clear(); active_context.update(base_context)
                    base_fit = greedy._fit_history_candidate(
                        month, track, train, homes, base_variant="G06_lot_garage",
                        fit_tag=f"R04unit-base-{month}")
                    removed_features.update(active_context["removed_features"])
                    component_manifest.append({**base_context, "parent_id": parent_id,
                                               "feature_parent": "G06_lot_garage",
                                               "feature_columns": base_fit["feature_columns"],
                                               "model_family": base_fit["model_family"],
                                               "model_config": base_fit["model_config"],
                                               "training_rows": base_fit["training_rows"]})
                    base_prediction = np.round(base_fit["predictions"])

                    mask = greedy._specialist_mask(train, "Detached")
                    specialist_train = train.loc[mask].copy()
                    specialist_mask = greedy._specialist_mask(homes, "Detached")
                    specialist_homes = homes.loc[specialist_mask].copy()
                    if specialist_train.empty or not specialist_mask.any():
                        raise RuntimeError(f"Detached specialist has no data in {month}")
                    expert_context = {"track": track, "month": month, "component": "expert"}
                    active_context.clear(); active_context.update(expert_context)
                    expert_fit = greedy._fit_history_candidate(
                        month, track, train, specialist_homes, base_variant="G06_lot_garage",
                        fit_tag=f"R04unit-expert-{month}", fit_rows=specialist_train)
                    removed_features.update(active_context["removed_features"])
                    component_manifest.append({**expert_context, "parent_id": parent_id,
                                               "feature_parent": "G06_lot_garage",
                                               "feature_columns": expert_fit["feature_columns"],
                                               "model_family": expert_fit["model_family"],
                                               "model_config": expert_fit["model_config"],
                                               "training_rows": expert_fit["training_rows"]})
                    candidate_prediction = greedy._blend_specialist_predictions(
                        base_prediction, np.round(expert_fit["predictions"]), homes,
                        "Detached", 0.75)
                else:
                    core_context = {"track": track, "month": month, "component": "core"}
                    active_context.clear(); active_context.update(core_context)
                    core_fit = greedy._fit_history_candidate(
                        month, track, train, homes, base_variant=None,
                        fit_tag=f"R04unit-core-{month}")
                    removed_features.update(active_context["removed_features"])
                    component_manifest.append({**core_context, "parent_id": parent_id,
                                               "feature_parent": "G01",
                                               "feature_columns": core_fit["feature_columns"],
                                               "model_family": core_fit["model_family"],
                                               "model_config": core_fit["model_config"],
                                               "training_rows": core_fit["training_rows"]})
                    frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, track, train, homes)
                    candidate_prediction = np.round(
                        0.25 * np.round(frozen_hgbr) + 0.75 * core_fit["predictions"])

                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": candidate_prediction})
                paired = helpers._candidate_frame(month, track, truth, parent, candidate)
                rows_by_track[track].append(paired)
                campaign._append({"event": "screen_fold_complete", "card": CARD,
                                  "run_id": run_id, "track": track, "month": month,
                                  "parent_id": parent_id,
                                  "parent_csv_sha256": helpers._sha256(parent_path),
                                  "removed_features": sorted(active_context["removed_features"])})
    finally:
        worker._normalize_categories = original_normalize
        greedy._fit_record = original_fit_record
        greedy._frozen_hgbr_predictions = original_hgbr

    paired_all = pd.concat([frame for frames in rows_by_track.values() for frame in frames], ignore_index=True)
    paired_path = batch_root / "paired_predictions.csv"
    paired_all.to_csv(paired_path, index=False)
    track_results = {track: helpers._track_result(pd.concat(frames, ignore_index=True))
                     for track, frames in rows_by_track.items()}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "screen_months": list(SCREEN_MONTHS), "parents": PARENTS,
              "removed_features": sorted(removed_features), "tracks": track_results,
              "fit_starts": len(fit_details), "reserved_months_labels_opened": False,
              "hidden_labels_opened": False, "data_sha256": source["data_sha256"]}
    results_path = batch_root / "r04_screen_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (batch_root / "component_manifest.json").write_text(
        json.dumps(component_manifest, indent=2, default=str), encoding="utf-8")
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "r03_catboost.py", ROOT / "experiments/bill_greedy/runner.py",
                    ROOT / "experiments/bill_greedy/history.py", ROOT / "experiments/model_optimization_v2/worker.py",
                    ROOT / "experiments/model_optimization_v2/learners.py",
                    ROOT / "model_candidate_v2/features.py", ROOT / "model_candidate_v2/winner_config.json"]
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "source_identity": source,
        "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): helpers._sha256(path) for path in source_files},
        "runtime": {"python": platform.python_version(), "lightgbm": metadata.version("lightgbm"),
                    "catboost": metadata.version("catboost"), "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit_learn": metadata.version("scikit-learn")},
        "parent_artifacts": {
            f"{track}/{month}": {
                "path": str(campaign._row_prediction_path(month, track).relative_to(ROOT)),
                "csv_sha256": helpers._sha256(campaign._row_prediction_path(month, track)),
                "sidecar_sha256": helpers._sha256(campaign._row_prediction_path(month, track).with_suffix(".json")),
            }
            for track in PARENTS for month in SCREEN_MONTHS
        },
        "frozen_hgbr_artifacts": {
            month: {"path": str((greedy.OUTPUT / "frozen_hgbr/list_aware" / f"{month}.csv").relative_to(ROOT)),
                    "csv_sha256": helpers._sha256(greedy.OUTPUT / "frozen_hgbr/list_aware" / f"{month}.csv")}
            for month in SCREEN_MONTHS
        },
        "paired_prediction_csv_sha256": helpers._sha256(paired_path),
        "results_sha256": helpers._sha256(results_path),
        "fit_starts": len(fit_details), "fit_details": fit_details,
    }
    (batch_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (batch_root / "research_report.md").write_text(_report(result), encoding="utf-8")
    passed = {track: score["qualifies_screen"] for track, score in track_results.items()}
    next_action = (
        "run eight-month R04 development confirmation for: " + ", ".join(track for track, ok in passed.items() if ok)
        if any(passed.values()) else
        "R04 unit-history ablation did not qualify; test the pre-registered coordinate-feature ablation next"
    )
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "qualifies_by_track": passed,
                      "result_path": str(results_path.relative_to(ROOT)),
                      "paired_prediction_csv_sha256": manifest["paired_prediction_csv_sha256"]})
    campaign._write_progress("r04_screen_complete", next_action=next_action,
                             r04_result=str(results_path.relative_to(ROOT)))
    return result


def main() -> None:
    try:
        result = run_r04_unit_history_screen()
    except Exception as exc:
        campaign._append({"event": "card_failed", "card": CARD,
                          "error": f"{type(exc).__name__}: {exc}"})
        campaign._write_progress("r04_failed", next_action="inspect the R04 failure record before any retry",
                                 latest_error=f"{type(exc).__name__}: {exc}")
        raise
    for track, score in result["tracks"].items():
        print(f"{track}: parent MdAPE={score['parent']['mdape_percent']:.5f}% "
              f"candidate={score['candidate']['mdape_percent']:.5f}% "
              f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
              f"MAPE_change={score['mape_relative_change_percent']:+.3f}% "
              f"screen_pass={score['qualifies_screen']}", flush=True)
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
