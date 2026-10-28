from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "experiments/bill_target74"
PROTOCOL_PATH = PACKAGE / "campaign_protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
DATA = ROOT / PROTOCOL["data_path"]
CAMPAIGN_ROOT = ROOT / "outputs/bill_target74" / PROTOCOL["campaign_id"]
LEDGER = CAMPAIGN_ROOT / "fits.jsonl"
STATE = CAMPAIGN_ROOT / "progress.json"
NEXT_STAGE_ROOT = ROOT / "outputs/bill_next_stage/bill_next_stage_20261023T234500Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def limit_reached(value: float, limit: float | int | None) -> bool:
    """Return whether a finite inclusive cap has been reached; None means unbounded."""
    return limit is not None and value >= limit


def limit_exceeded(value: float, limit: float | int | None) -> bool:
    """Return whether a finite exclusive cap has been exceeded; None means unbounded."""
    return limit is not None and value > limit


def _append(event: dict) -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
    with LEDGER.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("actual and predicted prices must be aligned vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual <= 0).any():
        raise ValueError("metrics require finite positive actual and predicted prices")
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape)), "p90_ape_percent": float(np.quantile(ape, 0.9)),
            "ape_over_20_percent": float(100.0 * np.mean(ape > 20.0))}


def _row_prediction_path(month: str, track: str) -> Path:
    if track == "off_market":
        return ROOT / "outputs/bill_greedy/neighbors/G19_detached_shrink_075" / track / f"{month}.csv"
    return ROOT / "outputs/bill_greedy/predictions/G01" / track / f"{month}.csv"


def _read_screen_fold(listings: pd.DataFrame, month: str, track: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from experiments.model_optimization_v2.folds import build_month_fold

    train, homes, truth = build_month_fold(listings, month, track)
    path = _row_prediction_path(month, track)
    if not path.is_file():
        raise FileNotFoundError(f"frozen parent predictions are missing: {path}")
    pred = pd.read_csv(path, dtype={"ml_num": str})
    price_column = "predicted_price"
    required = {"ml_num", "sold_price", price_column}
    if missing := required - set(pred.columns):
        raise ValueError(f"parent file {path} missing columns {sorted(missing)}")
    expected_ids = homes.ml_num.astype(str).to_numpy()
    if (pred.ml_num.astype(str).duplicated().any()
            or not np.array_equal(pred.ml_num.astype(str).to_numpy(), expected_ids)
            or not np.array_equal(pred.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
        raise ValueError(f"parent prediction IDs or labels do not align for {month}/{track}")
    if not np.isfinite(pred.predicted_price.to_numpy(float)).all() or (pred.predicted_price.to_numpy(float) <= 0).any():
        raise ValueError(f"parent predictions are not finite positive prices for {month}/{track}")
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"parent sidecar is missing: {sidecar}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    known_data_sha = meta.get("data_sha256")
    if known_data_sha != PROTOCOL["data_sha256"]:
        raise ValueError(f"parent sidecar data hash differs for {month}/{track}")
    pred["ml_num"] = pred.ml_num.astype(str)
    pred["month"] = month
    pred["date_sold"] = pd.to_datetime(
        listings.set_index(listings.ml_num.astype(str)).loc[expected_ids, "date_sold"].to_numpy(),
        errors="raise")
    pred["house_type_name"] = homes.house_type_name.to_numpy()
    pred["municipality"] = homes.municipality.to_numpy()
    pred["house_area"] = homes.house_area.to_numpy()
    pred["bedrooms"] = homes.bedrooms.to_numpy()
    pred["washrooms"] = homes.washrooms.to_numpy()
    pred.rename(columns={price_column: "parent_prediction"}, inplace=True)
    pred = pred[["month", "ml_num", "date_sold", "house_type_name", "municipality",
                 "house_area", "bedrooms", "washrooms", "sold_price", "parent_prediction"]]
    truth_frame = pred[["ml_num", "sold_price", "parent_prediction"]].copy()
    truth_frame["month"] = month
    truth_frame["track"] = track
    truth_frame["prediction_csv"] = str(path.relative_to(ROOT))
    truth_frame["prediction_sha256"] = _sha256(path)
    truth_frame["sidecar_sha256"] = _sha256(sidecar)
    return pred, truth_frame


def _seed_oof_rows(listings: pd.DataFrame, track: str) -> pd.DataFrame:
    audit_path = NEXT_STAGE_ROOT / "n03_seed_oof_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = []
    metadata = listings.copy()
    metadata["ml_num"] = metadata.ml_num.astype(str)
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata = metadata.set_index("ml_num", drop=False)
    for month in ("2022-06", "2022-11"):
        path = NEXT_STAGE_ROOT / "predictions/N03_seed_oof" / track / f"{month}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"past OOF seed is missing: {path}")
        expected_hash = audit["tracks"][track][month]["feature_file_sha256"]
        if _sha256(path) != expected_hash:
            raise ValueError(f"past OOF seed hash differs from its closed-run audit: {path}")
        frame = pd.read_csv(path, dtype={"ml_num": str})
        if frame.ml_num.astype(str).duplicated().any():
            raise ValueError(f"duplicate IDs in past OOF seed: {path}")
        source = metadata.loc[frame.ml_num.astype(str).to_list()].reset_index(drop=True)
        if (not np.array_equal(source.ml_num.astype(str).to_numpy(), frame.ml_num.astype(str).to_numpy())
                or not np.array_equal(source.sold_price.to_numpy(float), frame.sold_price.to_numpy(float))
                or not np.array_equal(source.date_sold.to_numpy(dtype="datetime64[ns]"),
                                      pd.to_datetime(frame.date_sold).to_numpy(dtype="datetime64[ns]"))):
            raise ValueError(f"seed labels or sale dates do not match source data: {path}")
        combined = source[["ml_num", "date_sold", "house_type_name", "municipality", "house_area",
                           "bedrooms", "washrooms", "sold_price"]].copy()
        combined["parent_prediction"] = frame.parent_prediction.to_numpy(float)
        combined["month"] = month
        rows.append(combined)
    return pd.concat(rows, ignore_index=True)


def _history_oof_rows(listings: pd.DataFrame, track: str, cutoff: str) -> tuple[pd.DataFrame, list[dict]]:
    cutoff_at = pd.Timestamp(cutoff)
    candidate_months = ["2022-06", "2022-11", *PROTOCOL["development_months"]]
    candidate_months = [month for month in candidate_months if pd.Timestamp(month + "-01") < cutoff_at]
    frames = [_seed_oof_rows(listings, track)]
    sources = []
    for month in candidate_months:
        if month in {"2022-06", "2022-11"}:
            p = NEXT_STAGE_ROOT / "predictions/N03_seed_oof" / track / f"{month}.csv"
            sources.append({"month": month, "path": str(p.relative_to(ROOT)), "sha256": _sha256(p),
                            "kind": "past_only_seed_oof"})
            continue
        frame, _ = _read_screen_fold(listings, month, track)
        frames.append(frame)
        path = _row_prediction_path(month, track)
        sidecar = path.with_suffix(".json")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        sources.append({"month": month, "path": str(path.relative_to(ROOT)), "sha256": _sha256(path),
                        "sidecar_sha256": _sha256(sidecar), "recorded_code_sha256": meta.get("code_sha256"),
                        "kind": "frozen_walk_forward_parent_oof"})
    history = pd.concat(frames, ignore_index=True)
    history["date_sold"] = pd.to_datetime(history.date_sold, errors="raise")
    history = history.loc[history.date_sold < cutoff_at].copy()
    if history.duplicated("ml_num").any():
        # An ID must contribute one out-of-fold historical sale only.
        duplicate_ids = history.loc[history.duplicated("ml_num", keep=False), "ml_num"].astype(str).unique()
        raise ValueError(f"historical OOF contains duplicate home IDs: {duplicate_ids[:5].tolist()}")
    return history, sources


def _relative_gain(parent: float, candidate: float) -> float:
    return 100.0 * (parent - candidate) / parent


def run_r02_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74.calibration import apply_oof_correction, build_oof_neighbor_features

    if not DATA.is_file():
        raise FileNotFoundError(f"challenge listings parquet is missing: {DATA}")
    current_data_sha = _sha256(DATA)
    if current_data_sha != PROTOCOL["data_sha256"]:
        raise ValueError("challenge data hash differs from the frozen campaign protocol")

    batch_stem = "bill_target74_first_batch_20261024T035422Z"
    batch_root = CAMPAIGN_ROOT / "batches" / batch_stem
    retry = 0
    while batch_root.exists():
        retry += 1
        batch_root = CAMPAIGN_ROOT / "batches" / f"{batch_stem}_retry{retry}"
    batch_root.mkdir(parents=True)
    listings = pd.read_parquet(DATA)
    _append({"event": "card_start", "card": "R02_structure_matched_past_oof", "run_id": batch_root.name,
             "fit_start_count": 0, "screen_months": PROTOCOL["screen_months"]})

    all_rows: dict[str, list[pd.DataFrame]] = {track: [] for track in PROTOCOL["parent_by_track"]}
    all_sources: dict[str, dict[str, list[dict]]] = {track: {} for track in PROTOCOL["parent_by_track"]}
    monthly_results: dict[str, dict] = {track: {} for track in PROTOCOL["parent_by_track"]}
    coverage: dict[str, dict] = {track: {} for track in PROTOCOL["parent_by_track"]}
    for month in PROTOCOL["screen_months"]:
        cutoff = pd.Timestamp(month + "-01")
        for track in PROTOCOL["parent_by_track"]:
            _, truth = _read_screen_fold(listings, month, track)
            hidden_homes = listings.loc[pd.to_datetime(listings.date_sold).dt.to_period("M").astype(str).eq(month)].copy()
            hidden_homes["ml_num"] = hidden_homes.ml_num.astype(str)
            homes = hide_answers(hidden_homes, track)
            if not np.array_equal(homes.ml_num.astype(str).to_numpy(), truth.ml_num.astype(str).to_numpy()):
                raise ValueError(f"hidden query IDs differ from frozen parent rows: {month}/{track}")
            homes = homes.copy()
            homes["parent_prediction"] = truth.parent_prediction.to_numpy(float)
            history, sources = _history_oof_rows(listings, track, str(cutoff.date()))
            if not history.empty and (history.date_sold >= cutoff).any():
                raise ValueError(f"future or same-month OOF sale entered the query history: {month}/{track}")
            features = build_oof_neighbor_features(history, homes, cutoff)
            prediction = apply_oof_correction(truth.parent_prediction.to_numpy(float), features)
            prediction = np.round(prediction)
            parent_metric = _metrics(truth.sold_price.to_numpy(float), truth.parent_prediction.to_numpy(float))
            candidate_metric = _metrics(truth.sold_price.to_numpy(float), prediction)
            relative = _relative_gain(parent_metric["mdape_percent"], candidate_metric["mdape_percent"])
            mape_change = 100.0 * (candidate_metric["mape_percent"] - parent_metric["mape_percent"]) / parent_metric["mape_percent"]
            frame = truth[["month", "track", "ml_num", "sold_price", "parent_prediction"]].copy()
            frame["candidate_prediction"] = prediction
            frame["covered"] = features.covered.to_numpy(bool)
            frame["neighbor_count"] = features.neighbor_count.to_numpy(int)
            frame["effective_n"] = features.effective_n.to_numpy(float)
            frame["weighted_log_residual"] = features.weighted_log_residual.to_numpy(float)
            frame["correction_log"] = features.correction_log.to_numpy(float)
            frame["parent_ape_percent"] = (100.0 * np.abs(frame.sold_price - frame.parent_prediction) / frame.sold_price)
            frame["candidate_ape_percent"] = (100.0 * np.abs(frame.sold_price - frame.candidate_prediction) / frame.sold_price)
            all_rows[track].append(frame)
            all_sources[track][month] = sources
            coverage[track][month] = {"rows": int(len(frame)), "covered_rows": int(features.covered.sum()),
                                      "coverage_percent": float(100.0 * features.covered.mean()),
                                      "median_neighbors": float(features.neighbor_count.median()),
                                      "p90_neighbors": float(features.neighbor_count.quantile(0.9))}
            monthly_results[track][month] = {"parent": parent_metric, "candidate": candidate_metric,
                                              "relative_mdape_improvement_percent": relative,
                                              "mape_relative_change_percent": mape_change,
                                              "history_rows": int(len(history)),
                                              "max_oof_sale_date": str(history.date_sold.max().date()) if not history.empty else None}

    track_results = {}
    rows_for_export = []
    for track, frames in all_rows.items():
        combined = pd.concat(frames, ignore_index=True)
        parent_metric = _metrics(combined.sold_price.to_numpy(float), combined.parent_prediction.to_numpy(float))
        candidate_metric = _metrics(combined.sold_price.to_numpy(float), combined.candidate_prediction.to_numpy(float))
        relative = _relative_gain(parent_metric["mdape_percent"], candidate_metric["mdape_percent"])
        mape_change = 100.0 * (candidate_metric["mape_percent"] - parent_metric["mape_percent"]) / parent_metric["mape_percent"]
        both_months = all(monthly_results[track][m]["relative_mdape_improvement_percent"] > 0.0
                          for m in PROTOCOL["screen_months"])
        qualifies = (relative >= PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"]
                     and both_months
                     and mape_change <= PROTOCOL["gates"]["screen_mape_regression_max_percent"])
        track_results[track] = {"parent": parent_metric, "candidate": candidate_metric,
                                "relative_mdape_improvement_percent": relative,
                                "mape_relative_change_percent": mape_change,
                                "both_screen_months_improved": both_months,
                                "qualifies_screen": qualifies,
                                "monthly": monthly_results[track], "coverage": coverage[track],
                                "oof_sources": all_sources[track]}
        rows_for_export.append(combined)
    result = {"campaign_id": PROTOCOL["campaign_id"], "run_id": batch_root.name,
              "card": "R02_structure_matched_past_oof", "screen_months": PROTOCOL["screen_months"],
              "fit_starts": 0, "tracks": track_results,
              "qualification": "both tracks" if all(x["qualifies_screen"] for x in track_results.values()) else
                               "track-level results; no automatic parent promotion",
              "provenance_limit": "Frozen development parent predictions have a recorded code-hash mismatch with the current worktree; they are used as paired historical walk-forward comparators. N03 seed OOF rows were hash-verified against the closed run manifest.",
              "data_sha256": current_data_sha}
    predictions = pd.concat(rows_for_export, ignore_index=True)
    prediction_path = batch_root / "paired_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    results_path = batch_root / "r02_screen_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    code_files = [PROTOCOL_PATH, PACKAGE / "calibration.py", PACKAGE / "runner.py",
                  ROOT / "experiments/bill_next_stage/runner.py", ROOT / "experiments/bill_next_stage/models.py",
                  ROOT / "experiments/bill_greedy/history.py", ROOT / "experiments/model_optimization_v2/folds.py",
                  ROOT / "model_candidate_v2/features.py"]
    manifest = {"campaign_id": PROTOCOL["campaign_id"], "run_id": batch_root.name,
                "data_sha256": current_data_sha,
                "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): _sha256(p) for p in code_files},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "lightgbm": metadata.version("lightgbm"),
                            "catboost": metadata.version("catboost"), "scikit_learn": metadata.version("scikit-learn")},
                "parent_artifacts": sorted({entry["path"] for track in track_results.values()
                                             for month_sources in track["oof_sources"].values() for entry in month_sources}),
                "parent_artifact_provenance": "historical paired walk-forward predictions; sidecar code hash differs from current worktree",
                "paired_prediction_csv_sha256": _sha256(prediction_path),
                "result_sha256": _sha256(results_path), "fit_starts": 0}
    (batch_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = _render_report(result, manifest)
    (batch_root / "research_report.md").write_text(report, encoding="utf-8")
    _append({"event": "card_complete", "card": "R02_structure_matched_past_oof", "run_id": batch_root.name,
             "fit_start_count": 0, "qualifies_by_track": {track: data["qualifies_screen"] for track, data in track_results.items()},
             "result_path": str(results_path.relative_to(ROOT)), "prediction_sha256": _sha256(prediction_path)})
    _write_progress("batch_screen_complete", next_action=_next_action(track_results), r02_result=str(results_path.relative_to(ROOT)))
    return result


def _next_action(track_results: dict) -> str:
    if any(row["qualifies_screen"] for row in track_results.values()):
        return "run eight-month development confirmation for each track whose R02 screen passes"
    return "finish two local XGBoost dependency diagnostics, then run R03 current-feature CatBoost screen"


def _write_progress(status: str, **updates) -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    old = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    ledger = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()] if LEDGER.exists() else []
    starts = sum(row.get("event") == "fit_start" for row in ledger)
    state = {**old, "campaign_id": PROTOCOL["campaign_id"], "status": status,
             "current_parents": PROTOCOL["parent_by_track"], "targets_mdape_strictly_below": {k: v["target_mdape_strictly_below"] for k, v in PROTOCOL["initial_development_metrics"].items()},
             "initial_development_metrics": PROTOCOL["initial_development_metrics"],
             "fit_starts": starts,
             "reserved_months_labels_opened": bool(PROTOCOL.get(
                 "reserved_months_labels_opened", old.get("reserved_months_labels_opened", False))),
             "reserve_validation_eligible": bool(PROTOCOL.get(
                 "reserve_validation_eligible", old.get("reserve_validation_eligible", True))),
             "hidden_labels_opened": bool(PROTOCOL.get(
                 "hidden_labels_opened", old.get("hidden_labels_opened", False))),
             "next_action": updates.pop("next_action", old.get("next_action")),
             **updates}
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def _render_report(result: dict, manifest: dict) -> str:
    lines = ["# BILL <7% / <4% 持续研究：R02 首轮筛选", "",
             f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
             "父模型：off-market G19，list-aware G01。屏幕月份：2023-03 与 2024-06。",
             "本卡使用固定过去 OOF 残差修正，没有训练新学习器，因此本轮真实 learner fit starts 为 0。",
             "", "## 赛道结果", "",
             "| 赛道 | 父模型 MdAPE | 候选 MdAPE | 相对改善 | 父模型 MAPE | 候选 MAPE | MAPE 变化 | 双月均改善 | 筛选通过 |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for track, data in result["tracks"].items():
        lines.append(f"| {track} | {data['parent']['mdape_percent']:.5f}% | {data['candidate']['mdape_percent']:.5f}% | {data['relative_mdape_improvement_percent']:+.3f}% | {data['parent']['mape_percent']:.5f}% | {data['candidate']['mape_percent']:.5f}% | {data['mape_relative_change_percent']:+.3f}% | {'是' if data['both_screen_months_improved'] else '否'} | {'是' if data['qualifies_screen'] else '否'} |")
    lines.extend(["", "## 月份和覆盖", "", "| 赛道 | 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 | MAPE 变化 | OOF 历史行 | 修正覆盖 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for track, data in result["tracks"].items():
        for month, score in data["monthly"].items():
            cov = data["coverage"][month]
            lines.append(f"| {track} | {month} | {score['parent']['mdape_percent']:.5f}% | {score['candidate']['mdape_percent']:.5f}% | {score['relative_mdape_improvement_percent']:+.3f}% | {score['mape_relative_change_percent']:+.3f}% | {score['history_rows']} | {cov['coverage_percent']:.1f}% ({cov['covered_rows']}/{cov['rows']}) |")
    lines.extend(["", "## 解释与限制", "",
                  "每个历史残差来自相应月份的 walk-forward OOF 预测；查询月只能读取严格早于月初的 OOF 行。2022-06/11 种子文件哈希已与关闭的 N03 run manifest 核对。",
                  "2023/2024 父预测 CSV 按房源 ID 与标签配对，并检查数据哈希；其 sidecar 代码哈希与当前工作区不一致，因此这里只作为已保存的历史父模型 walk-forward 对照，不能称为当前代码的重新复现。",
                  "screen 通过条件是合并 MdAPE 相对改善至少 0.25%、两个筛选月均改善、MAPE 相对退化不超过 1%。只通过的赛道可进入八个月确认；该筛选自身不改变父模型。",
                  "隐藏标签及 2025-05/08 预留月未读取。", "", "## 文件", "",
                  "- 配对逐房结果：`paired_predictions.csv`",
                  "- 完整分数、月度分数、覆盖与 OOF 来源：`r02_screen_results.json`",
                  "- 代码、数据、环境及输入哈希：`run_manifest.json`",
                  "- Campaign 事件账本：`../../fits.jsonl`",
                  ""])
    return "\n".join(lines)


def main() -> None:
    try:
        result = run_r02_screen()
    except Exception as exc:
        _append({"event": "card_failed", "card": "R02_structure_matched_past_oof",
                 "fit_start_count": 0, "error": f"{type(exc).__name__}: {exc}"})
        _write_progress("batch_failed", next_action="inspect the failure record, fix the runner, and retry under a new batch id",
                         latest_error=f"{type(exc).__name__}: {exc}")
        raise
    for track, row in result["tracks"].items():
        print(f"{track}: parent MdAPE={row['parent']['mdape_percent']:.5f}% candidate={row['candidate']['mdape_percent']:.5f}% relative_gain={row['relative_mdape_improvement_percent']:+.3f}% screen_pass={row['qualifies_screen']}", flush=True)
    print(f"report={CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
