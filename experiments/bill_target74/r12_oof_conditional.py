from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


def apply_oof_group_correction(
    history_oof: pd.DataFrame,
    queries: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    *,
    min_group_rows: int = 100,
    shrinkage_pseudocount: float = 200.0,
    max_abs_log_correction: float = 0.10,
) -> pd.DataFrame:
    """Apply past-OOF median log-residual corrections by type and price quartile.

    Quartile boundaries are estimated from eligible past predictions only. Query
    outcomes are forbidden, and history rows on or after the outer cutoff are
    ignored before any residual or threshold is calculated.
    """
    history_required = {"date_sold", "house_type_name", "parent_prediction", "sold_price"}
    query_required = {"house_type_name", "parent_prediction"}
    missing_history = sorted(history_required - set(history_oof.columns))
    missing_queries = sorted(query_required - set(queries.columns))
    if missing_history:
        raise ValueError(f"past OOF history is missing columns: {missing_history}")
    if missing_queries:
        raise ValueError(f"queries are missing required columns: {missing_queries}")
    leaked = {"sold_price", "date_sold"}.intersection(queries.columns)
    if leaked:
        raise ValueError(f"query outcome columns are forbidden: {sorted(leaked)}")
    if min_group_rows < 1 or shrinkage_pseudocount < 0.0 or max_abs_log_correction <= 0.0:
        raise ValueError("group size must be positive; shrinkage and correction cap must be nonnegative")

    cutoff_at = pd.Timestamp(cutoff)
    if cutoff_at.day != 1 or cutoff_at != cutoff_at.normalize():
        raise ValueError("cutoff must be the first day of a prediction month")

    history = history_oof.copy()
    history["date_sold"] = pd.to_datetime(history.date_sold, errors="coerce")
    if history.date_sold.isna().any():
        raise ValueError("past OOF history contains invalid sale dates")
    history = history.loc[history.date_sold < cutoff_at].copy()

    history_prediction = pd.to_numeric(history.parent_prediction, errors="coerce").to_numpy(float)
    history_actual = pd.to_numeric(history.sold_price, errors="coerce").to_numpy(float)
    query_prediction = pd.to_numeric(queries.parent_prediction, errors="coerce").to_numpy(float)
    if (not np.isfinite(history_prediction).all() or not np.isfinite(history_actual).all()
            or (history_prediction <= 0.0).any() or (history_actual <= 0.0).any()):
        raise ValueError("past OOF predictions and prices must be finite and positive")
    if not np.isfinite(query_prediction).all() or (query_prediction <= 0.0).any():
        raise ValueError("query parent predictions must be finite and positive")

    if history.empty:
        history_bins = np.empty(0, dtype=np.int8)
        query_bins = np.zeros(len(queries), dtype=np.int8)
        thresholds = np.empty(0, dtype=float)
    else:
        thresholds = np.quantile(history_prediction, [0.25, 0.50, 0.75])
        history_bins = np.searchsorted(thresholds, history_prediction, side="right").astype(np.int8)
        query_bins = np.searchsorted(thresholds, query_prediction, side="right").astype(np.int8)

    history_types = history.house_type_name.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    query_types = queries.house_type_name.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    log_residual = np.log(history_actual / history_prediction)
    corrections: dict[tuple[str, int], tuple[float, int]] = {}
    for group_type in pd.unique(history_types):
        for price_bin in range(4):
            mask = (history_types == group_type) & (history_bins == price_bin)
            count = int(mask.sum())
            if count < min_group_rows:
                continue
            median_residual = float(np.median(log_residual[mask]))
            weight = count / (count + shrinkage_pseudocount)
            correction = float(np.clip(median_residual, -max_abs_log_correction,
                                       max_abs_log_correction) * weight)
            corrections[(str(group_type), price_bin)] = (correction, count)

    output_correction = np.zeros(len(queries), dtype=float)
    output_history_rows = np.zeros(len(queries), dtype=np.int64)
    output_group = []
    for index, (group_type, price_bin) in enumerate(zip(query_types, query_bins, strict=True)):
        key = (str(group_type), int(price_bin))
        output_group.append(f"{key[0]}|Q{key[1] + 1}")
        if key in corrections:
            output_correction[index], output_history_rows[index] = corrections[key]

    result = pd.DataFrame(index=queries.index)
    result["calibration_group"] = output_group
    result["history_rows"] = output_history_rows
    result["correction_log"] = output_correction
    result["corrected_prediction"] = query_prediction * np.exp(output_correction)
    result.attrs["cutoff"] = cutoff_at.isoformat()
    result.attrs["eligible_history_rows"] = int(len(history))
    result.attrs["prediction_quartile_thresholds"] = thresholds.tolist()
    return result


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "experiments/bill_target74"
PROTOCOL = json.loads((PACKAGE / "campaign_protocol.json").read_text(encoding="utf-8"))
CAMPAIGN_ROOT = ROOT / "outputs/bill_target74" / PROTOCOL["campaign_id"]
SEED_ROOT = ROOT / "outputs/bill_next_stage/bill_next_stage_20261023T234500Z"
R06_ROOT = CAMPAIGN_ROOT / "batches/bill_target74_r06_unit_oof_20261024T042011Z"
SEED_MONTHS = ("2022-06", "2022-11")
OOf_MONTHS = (*SEED_MONTHS, "2023-07", "2023-11", "2024-03")
VALIDATION_MONTHS = ("2023-07", "2023-11", "2024-03")
SCREEN_MONTHS = tuple(PROTOCOL["screen_months"])
CARD = "R12_past_oof_conditional_residual_calibration"
MIN_GROUP_ROWS = 100
SHRINKAGE_PSEUDOCOUNT = 200.0
MAX_ABS_LOG_CORRECTION = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append(event: dict) -> None:
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
    with (CAMPAIGN_ROOT / "fits.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("actual and predicted prices must be aligned vectors")
    if (not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or (actual <= 0.0).any() or (predicted <= 0.0).any()):
        raise ValueError("metrics require finite positive prices")
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {
        "n": int(len(actual)),
        "mdape_percent": float(np.median(ape)),
        "mape_percent": float(np.mean(ape)),
        "p90_ape_percent": float(np.quantile(ape, 0.90)),
        "ape_over_20_percent": float(100.0 * np.mean(ape > 20.0)),
    }


def _relative_gain(parent: float, candidate: float) -> float:
    return 100.0 * (parent - candidate) / parent


def _join_oof_source_rows(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.ml_num.astype(str).duplicated().any():
        raise ValueError("per-home source metadata must have unique IDs")
    if predictions.duplicated(["track", "ml_num"]).any():
        raise ValueError("OOF predictions must have unique IDs within each track")
    joined = predictions.merge(metadata, on="ml_num", how="left", suffixes=("_saved", "_source"),
                               validate="many_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("one or more saved OOF rows do not match permitted source data")
    if (joined.month_saved.astype(str) != joined.month_source.astype(str)).any():
        raise ValueError("saved OOF month does not match source sale date")
    saved_dates = pd.to_datetime(joined.date_sold_saved, errors="coerce")
    source_dates = pd.to_datetime(joined.date_sold_source, errors="raise")
    if saved_dates.notna().any() and not np.array_equal(
            saved_dates.loc[saved_dates.notna()].to_numpy(),
            source_dates.loc[saved_dates.notna()].to_numpy()):
        raise ValueError("saved OOF sale dates do not match source dates")
    if not np.array_equal(joined.sold_price_saved.to_numpy(float), joined.sold_price_source.to_numpy(float)):
        raise ValueError("saved OOF sale labels do not match the permitted source rows")
    joined["date_sold"] = source_dates
    joined["sold_price"] = joined.sold_price_source
    joined["month"] = joined.month_source.astype(str)
    return joined[["month", "track", "ml_num", "date_sold", "house_type_name",
                   "sold_price", "parent_prediction"]].copy()


def _read_only_allowed_rows() -> tuple[pd.DataFrame, dict]:
    """Read labels only from registered OOF and development-screen months."""
    data_path = ROOT / PROTOCOL["data_path"]
    if not data_path.is_file() or _sha256(data_path) != PROTOCOL["data_sha256"]:
        raise ValueError("campaign data file is missing or its frozen hash differs")

    all_read_months = (*OOf_MONTHS, *SCREEN_MONTHS)
    date_filters = []
    for month in all_read_months:
        start = pd.Timestamp(month + "-01")
        stop = start + pd.DateOffset(months=1)
        date_filters.append([("date_sold", ">=", start), ("date_sold", "<", stop)])
    metadata = pd.read_parquet(
        data_path,
        engine="pyarrow",
        columns=["ml_num", "date_sold", "sold_price", "house_type_name"],
        filters=date_filters,
    )
    metadata["ml_num"] = metadata.ml_num.astype(str)
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["month"] = metadata.date_sold.dt.strftime("%Y-%m")
    if metadata.ml_num.duplicated().any():
        raise ValueError("registered OOF/screen source rows contain duplicate IDs")
    if not set(metadata.month.astype(str)).issubset(set(all_read_months)):
        raise ValueError("Parquet month filter returned rows outside registered months")

    prediction_frames = []
    sources = {"data_sha256": PROTOCOL["data_sha256"], "allowed_label_months_read": list(all_read_months),
               "prediction_files": []}
    audit = json.loads((SEED_ROOT / "n03_seed_oof_audit.json").read_text(encoding="utf-8"))
    for track in PROTOCOL["parent_by_track"]:
        for month in SEED_MONTHS:
            path = SEED_ROOT / "predictions/N03_seed_oof" / track / f"{month}.csv"
            expected_hash = audit["tracks"][track][month]["feature_file_sha256"]
            if _sha256(path) != expected_hash:
                raise ValueError(f"N03 seed OOF hash differs from its closed-run audit: {path}")
            frame = pd.read_csv(path, dtype={"ml_num": str})
            frame = frame[["month", "ml_num", "date_sold", "sold_price", "parent_prediction"]].copy()
            frame["track"] = track
            if set(frame.month.astype(str)) != {month}:
                raise ValueError(f"seed OOF file contains an unexpected month: {path}")
            prediction_frames.append(frame)
            sources["prediction_files"].append({"path": str(path.relative_to(ROOT)), "sha256": _sha256(path),
                                                 "kind": "N03_verified_seed_oof", "month": month,
                                                 "track": track})

    r06_predictions = R06_ROOT / "paired_oof_predictions.csv"
    r06_manifest = json.loads((R06_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    r06_hash = _sha256(r06_predictions)
    if r06_hash != r06_manifest["paired_oof_prediction_csv_sha256"]:
        raise ValueError("R06 parent OOF prediction file hash differs from its run manifest")
    r06 = pd.read_csv(r06_predictions, dtype={"ml_num": str})
    r06 = r06[["month", "track", "ml_num", "sold_price", "parent_prediction"]].copy()
    r06["month"] = r06.month.astype(str)
    if set(r06.month) != set(VALIDATION_MONTHS):
        raise ValueError("R06 OOF file does not cover exactly the registered validation months")
    prediction_frames.append(r06)
    sources["prediction_files"].append({"path": str(r06_predictions.relative_to(ROOT)), "sha256": r06_hash,
                                         "kind": "R06_verified_parent_oof"})

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["ml_num"] = predictions.ml_num.astype(str)
    predictions["month"] = predictions.month.astype(str)
    predictions["date_sold"] = pd.to_datetime(predictions.date_sold, errors="coerce")
    if predictions.duplicated(["track", "ml_num"]).any():
        raise ValueError("verified OOF prediction sources contain duplicate track/ID rows")
    rows = _join_oof_source_rows(predictions, metadata)
    rows["parent_prediction"] = pd.to_numeric(rows.parent_prediction, errors="raise")
    if set(rows.month) != set(OOf_MONTHS):
        raise ValueError("the OOF row union does not cover exactly the registered five months")
    if (rows.parent_prediction <= 0.0).any() or not np.isfinite(rows.parent_prediction.to_numpy(float)).all():
        raise ValueError("OOF parent predictions must be finite and positive")
    return rows, sources


def _read_screen_parent_rows(metadata: pd.DataFrame, month: str, track: str) -> tuple[pd.DataFrame, dict]:
    if track == "off_market":
        path = ROOT / "outputs/bill_greedy/neighbors/G19_detached_shrink_075" / track / f"{month}.csv"
    else:
        path = ROOT / "outputs/bill_greedy/predictions/G01" / track / f"{month}.csv"
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"frozen parent predictions or sidecar are missing: {path}")
    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
    if sidecar_data.get("data_sha256") != PROTOCOL["data_sha256"]:
        raise ValueError(f"frozen parent prediction data hash differs for {month}/{track}")
    pred = pd.read_csv(path, dtype={"ml_num": str})
    if pred.ml_num.astype(str).duplicated().any():
        raise ValueError(f"frozen parent predictions contain duplicate IDs for {month}/{track}")
    part = metadata.loc[metadata.month.eq(month)].copy()
    part["ml_num"] = part.ml_num.astype(str)
    if set(pred.ml_num.astype(str)) != set(part.ml_num.astype(str)):
        raise ValueError(f"frozen parent prediction IDs do not exactly match source rows for {month}/{track}")
    joined = part.merge(pred[["ml_num", "sold_price", "predicted_price"]], on="ml_num", how="left",
                        validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all() or joined.predicted_price.isna().any():
        raise ValueError(f"frozen parent predictions do not cover screen rows for {month}/{track}")
    if not np.array_equal(joined.sold_price_x.to_numpy(float), joined.sold_price_y.to_numpy(float)):
        raise ValueError(f"frozen parent labels do not match permitted screen rows for {month}/{track}")
    joined["parent_prediction"] = joined.predicted_price.astype(float)
    joined["track"] = track
    joined["month"] = month
    result = joined[["month", "track", "ml_num", "date_sold", "house_type_name",
                     "sold_price_x", "parent_prediction"]].rename(columns={"sold_price_x": "sold_price"})
    return result, {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path),
                    "sidecar_sha256": _sha256(sidecar), "recorded_code_sha256": sidecar_data.get("code_sha256")}


def _walk_forward_scores(rows: pd.DataFrame, validation_months: tuple[str, ...]) -> tuple[dict, list[pd.DataFrame]]:
    track_results = {}
    exports = []
    for track in PROTOCOL["parent_by_track"]:
        month_results = {}
        track_frames = []
        for month in validation_months:
            history = rows.loc[rows.track.eq(track) & rows.month.lt(month)].copy()
            evaluation = rows.loc[rows.track.eq(track) & rows.month.eq(month)].copy().reset_index(drop=True)
            query = evaluation[["house_type_name", "parent_prediction"]].copy()
            calibrated = apply_oof_group_correction(
                history, query, month + "-01", min_group_rows=MIN_GROUP_ROWS,
                shrinkage_pseudocount=SHRINKAGE_PSEUDOCOUNT,
                max_abs_log_correction=MAX_ABS_LOG_CORRECTION,
            )
            evaluation["candidate_prediction"] = np.rint(calibrated.corrected_prediction.to_numpy(float))
            evaluation["calibration_group"] = calibrated.calibration_group.to_numpy()
            evaluation["history_rows"] = calibrated.history_rows.to_numpy(int)
            evaluation["correction_log"] = calibrated.correction_log.to_numpy(float)
            parent_score = _metrics(evaluation.sold_price.to_numpy(float),
                                    evaluation.parent_prediction.to_numpy(float))
            candidate_score = _metrics(evaluation.sold_price.to_numpy(float),
                                       evaluation.candidate_prediction.to_numpy(float))
            gain = _relative_gain(parent_score["mdape_percent"], candidate_score["mdape_percent"])
            mape_change = 100.0 * (candidate_score["mape_percent"] - parent_score["mape_percent"]) / parent_score["mape_percent"]
            month_results[month] = {
                "rows": int(len(evaluation)), "eligible_history_rows": int(len(history)),
                "corrected_rows": int(evaluation.history_rows.gt(0).sum()),
                "coverage_percent": float(100.0 * evaluation.history_rows.gt(0).mean()),
                "parent": parent_score, "candidate": candidate_score,
                "relative_mdape_improvement_percent": gain,
                "mape_relative_change_percent": mape_change,
            }
            evaluation["stage"] = "past_oof_walk_forward"
            track_frames.append(evaluation)
        combined = pd.concat(track_frames, ignore_index=True)
        parent_score = _metrics(combined.sold_price.to_numpy(float), combined.parent_prediction.to_numpy(float))
        candidate_score = _metrics(combined.sold_price.to_numpy(float), combined.candidate_prediction.to_numpy(float))
        pooled_gain = _relative_gain(parent_score["mdape_percent"], candidate_score["mdape_percent"])
        mape_change = 100.0 * (candidate_score["mape_percent"] - parent_score["mape_percent"]) / parent_score["mape_percent"]
        improved_months = sum(result["relative_mdape_improvement_percent"] > 0.0
                              for result in month_results.values())
        qualifies = (pooled_gain >= 0.25 and improved_months >= 2 and mape_change <= 1.0)
        track_results[track] = {
            "validation_months": list(validation_months), "parent": parent_score, "candidate": candidate_score,
            "relative_mdape_improvement_percent": pooled_gain,
            "mape_relative_change_percent": mape_change,
            "months_improved": int(improved_months), "pre_screen_qualifies": bool(qualifies),
            "monthly": month_results,
        }
        exports.extend(track_frames)
    return track_results, exports


def _screen_scores(rows: pd.DataFrame, oof_rows: pd.DataFrame,
                   screen_months: tuple[str, ...], sources: dict) -> tuple[dict, list[pd.DataFrame]]:
    track_results = {}
    exports = []
    for track in PROTOCOL["parent_by_track"]:
        if not sources.get("pre_screen_qualifies", {}).get(track, False):
            track_results[track] = {"evaluated": False, "reason": "past OOF conditional screen did not qualify"}
            continue
        monthly = {}
        track_frames = []
        for month in screen_months:
            evaluation = rows.loc[rows.track.eq(track) & rows.month.eq(month)].copy().reset_index(drop=True)
            history = oof_rows.loc[oof_rows.track.eq(track) & oof_rows.month.lt(month)].copy()
            query = evaluation[["house_type_name", "parent_prediction"]].copy()
            calibrated = apply_oof_group_correction(
                history, query, month + "-01", min_group_rows=MIN_GROUP_ROWS,
                shrinkage_pseudocount=SHRINKAGE_PSEUDOCOUNT,
                max_abs_log_correction=MAX_ABS_LOG_CORRECTION,
            )
            evaluation["candidate_prediction"] = np.rint(calibrated.corrected_prediction.to_numpy(float))
            evaluation["calibration_group"] = calibrated.calibration_group.to_numpy()
            evaluation["history_rows"] = calibrated.history_rows.to_numpy(int)
            evaluation["correction_log"] = calibrated.correction_log.to_numpy(float)
            parent_score = _metrics(evaluation.sold_price.to_numpy(float), evaluation.parent_prediction.to_numpy(float))
            candidate_score = _metrics(evaluation.sold_price.to_numpy(float), evaluation.candidate_prediction.to_numpy(float))
            gain = _relative_gain(parent_score["mdape_percent"], candidate_score["mdape_percent"])
            mape_change = 100.0 * (candidate_score["mape_percent"] - parent_score["mape_percent"]) / parent_score["mape_percent"]
            monthly[month] = {"parent": parent_score, "candidate": candidate_score,
                              "relative_mdape_improvement_percent": gain,
                              "mape_relative_change_percent": mape_change,
                              "corrected_rows": int(evaluation.history_rows.gt(0).sum()),
                              "coverage_percent": float(100.0 * evaluation.history_rows.gt(0).mean())}
            evaluation["stage"] = "fixed_screen"
            track_frames.append(evaluation)
        combined = pd.concat(track_frames, ignore_index=True)
        parent_score = _metrics(combined.sold_price.to_numpy(float), combined.parent_prediction.to_numpy(float))
        candidate_score = _metrics(combined.sold_price.to_numpy(float), combined.candidate_prediction.to_numpy(float))
        pooled_gain = _relative_gain(parent_score["mdape_percent"], candidate_score["mdape_percent"])
        mape_change = 100.0 * (candidate_score["mape_percent"] - parent_score["mape_percent"]) / parent_score["mape_percent"]
        both_months_improve = all(monthly[m]["relative_mdape_improvement_percent"] > 0.0 for m in screen_months)
        qualifies = pooled_gain >= 0.25 and both_months_improve and mape_change <= 1.0
        track_results[track] = {
            "evaluated": True, "parent": parent_score, "candidate": candidate_score,
            "relative_mdape_improvement_percent": pooled_gain,
            "mape_relative_change_percent": mape_change,
            "both_screen_months_improved": both_months_improve,
            "qualifies_screen": bool(qualifies), "monthly": monthly,
        }
        exports.extend(track_frames)
    return track_results, exports


def _render_report(result: dict) -> str:
    lines = [
        "# BILL <7% / <4% 持续研究：R12 past-OOF conditional residual calibration", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "方法：按房型 × 过去 OOF 父预测四分位组，估计严格过去 OOF 的中位 log(sold/prediction)；每组至少 100 行，按 n/(n+200) 收缩，修正截断到 ±0.10。四分位阈值只由各折历史预测估计。", "",
        "## 前推 OOF 诊断", "",
        "| 轨道 | 参考 / 候选 MdAPE | 相对改善 | MAPE 变化 | 月份改善数 | 进入正式筛选 |", "|---|---:|---:|---:|---:|---|",
    ]
    for track, score in result["oof_diagnostic"].items():
        lines.append(f"| {track} | {score['parent']['mdape_percent']:.5f}% / {score['candidate']['mdape_percent']:.5f}% | {score['relative_mdape_improvement_percent']:+.3f}% | {score['mape_relative_change_percent']:+.3f}% | {score['months_improved']}/3 | {'是' if score['pre_screen_qualifies'] else '否'} |")
    lines.extend(["", "## 逐月前推分数", "", "| 轨道 | 月份 | 父 MdAPE | 校准 MdAPE | 相对改善 | 覆盖率 |", "|---|---|---:|---:|---:|---:|"])
    for track, score in result["oof_diagnostic"].items():
        for month, monthly in score["monthly"].items():
            lines.append(f"| {track} | {month} | {monthly['parent']['mdape_percent']:.5f}% | {monthly['candidate']['mdape_percent']:.5f}% | {monthly['relative_mdape_improvement_percent']:+.3f}% | {monthly['coverage_percent']:.1f}% |")
    lines.extend(["", "## 固定双月筛选", ""])
    for track, score in result["screen"].items():
        if not score.get("evaluated"):
            lines.append(f"- {track}：未运行；{score['reason']}。")
        else:
            lines.append(f"- {track}：父→候选 pooled MdAPE {score['parent']['mdape_percent']:.5f}%→{score['candidate']['mdape_percent']:.5f}%；相对改善 {score['relative_mdape_improvement_percent']:+.3f}%；MAPE 变化 {score['mape_relative_change_percent']:+.3f}%；双月均改善：{score['both_screen_months_improved']}；筛选通过：{score['qualifies_screen']}。")
    lines.extend(["", "## 来源和边界", "",
                  f"- 本轮 learner fit starts：0；campaign 累计：{result['campaign_fit_starts']}。",
                  f"- 仅读取 OOF 月份 {', '.join(OOf_MONTHS)} 与固定开发筛选月 {', '.join(SCREEN_MONTHS)} 的 sold_price；没有读取 reserved 2025-05/08 或 hidden labels。",
                  "- N03 两个 seed OOF 文件哈希与关闭 run audit 核对；R06 OOF CSV 哈希与 run manifest 核对。历史父预测 sidecar 的代码哈希差异沿用既有 provenance 限制，仍是冻结比较器，不称为当前代码的重新训练复现。",
                  "- 该诊断/筛选使用已重复用于开发研究的月份，不能当成独立最终验证。未通过正式双月筛选的轨道不晋级；通过筛选也必须继续走八个月开发与固定起点确认。",
                  "- 逐房配对结果：`paired_predictions.csv`；来源与哈希：`run_manifest.json`；汇总：`r12_results.json`。", ""])
    return "\n".join(lines)


def run_r12_diagnostic() -> dict:
    state_path = CAMPAIGN_ROOT / "progress.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("campaign_id") != PROTOCOL["campaign_id"]:
        raise ValueError("campaign progress belongs to a different campaign")
    if state.get("reserved_months_labels_opened") or state.get("hidden_labels_opened"):
        raise ValueError("campaign integrity flags prohibit continuing")
    if campaign.limit_reached(int(state.get("fit_starts", 0)),
                              PROTOCOL["limits"]["exploration_fit_starts"]):
        raise RuntimeError("exploration fit-start ceiling reached")

    run_id = "bill_target74_r12_oof_conditional_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_root = CAMPAIGN_ROOT / "batches" / run_id
    batch_root.mkdir(parents=True, exist_ok=False)
    _append({"event": "card_start", "card": CARD, "run_id": run_id, "fit_start_count": 0,
             "oof_months": list(OOf_MONTHS), "validation_months": list(VALIDATION_MONTHS),
             "screen_months": list(SCREEN_MONTHS), "method": "type_x_parent_prediction_quartile_median_log_residual"})

    data_path = ROOT / PROTOCOL["data_path"]
    if not data_path.is_file() or _sha256(data_path) != PROTOCOL["data_sha256"]:
        raise ValueError("campaign data file is missing or its frozen hash differs")
    allowed_months = (*OOf_MONTHS, *SCREEN_MONTHS)
    date_filters = []
    for month in allowed_months:
        start = pd.Timestamp(month + "-01")
        date_filters.append([("date_sold", ">=", start),
                             ("date_sold", "<", start + pd.DateOffset(months=1))])
    screen_metadata = pd.read_parquet(data_path, engine="pyarrow",
                                      columns=["ml_num", "date_sold", "sold_price", "house_type_name"],
                                      filters=date_filters)
    screen_metadata["ml_num"] = screen_metadata.ml_num.astype(str)
    screen_metadata["date_sold"] = pd.to_datetime(screen_metadata.date_sold, errors="raise")
    screen_metadata["month"] = screen_metadata.date_sold.dt.strftime("%Y-%m")
    if not set(screen_metadata.month).issubset(set(allowed_months)):
        raise ValueError("source filters returned unregistered months")
    oof_rows, oof_sources = _read_only_allowed_rows()
    oof_results, oof_exports = _walk_forward_scores(oof_rows, VALIDATION_MONTHS)
    qualify_map = {track: score["pre_screen_qualifies"] for track, score in oof_results.items()}
    screen_rows = []
    screen_sources = []
    for month in SCREEN_MONTHS:
        for track in PROTOCOL["parent_by_track"]:
            frame, source = _read_screen_parent_rows(screen_metadata, month, track)
            screen_rows.append(frame)
            screen_sources.append({"month": month, "track": track, **source})
    screen_frame = pd.concat(screen_rows, ignore_index=True)
    screen_results, screen_exports = _screen_scores(screen_frame, oof_rows, SCREEN_MONTHS,
                                                    {"pre_screen_qualifies": qualify_map})
    result = {
        "campaign_id": PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "parents": PROTOCOL["parent_by_track"], "oof_months": list(OOf_MONTHS),
        "validation_months": list(VALIDATION_MONTHS), "screen_months": list(SCREEN_MONTHS),
        "method": {"group": ["house_type_name", "parent_prediction_quartile"],
                   "min_group_rows": MIN_GROUP_ROWS, "shrinkage_pseudocount": SHRINKAGE_PSEUDOCOUNT,
                   "max_abs_log_correction": MAX_ABS_LOG_CORRECTION,
                   "correction": "clip(median(log(sold_price/parent_prediction)), +/-0.10) * n/(n+200)"},
        "oof_diagnostic": oof_results, "screen": screen_results,
        "fit_starts": 0, "campaign_fit_starts": int(state["fit_starts"]),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "data_sha256": PROTOCOL["data_sha256"],
    }
    result_path = batch_root / "r12_results.json"
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    predictions = pd.concat([*oof_exports, *screen_exports], ignore_index=True) if screen_exports else pd.concat(oof_exports, ignore_index=True)
    predictions.to_csv(batch_root / "paired_predictions.csv", index=False)
    sources = {
        "campaign_id": PROTOCOL["campaign_id"], "run_id": run_id, "data_sha256": PROTOCOL["data_sha256"],
        "oof_sources": oof_sources, "screen_parent_sources": screen_sources,
        "code_sha256": _sha256(Path(__file__)),
        "test_sha256": _sha256(ROOT / "tests/test_bill_target74_r12.py"),
        "only_read_label_months": list(allowed_months),
        "r06_provenance_caveat": "Historical parent predictions are frozen OOF comparators; known source-code sidecar mismatch remains disclosed.",
    }
    (batch_root / "run_manifest.json").write_text(json.dumps(sources, indent=2, default=str) + "\n", encoding="utf-8")
    report = _render_report(result)
    (batch_root / "research_report.md").write_text(report, encoding="utf-8")

    _append({"event": "card_complete", "card": CARD, "run_id": run_id, "fit_starts": 0,
             "result_path": str(result_path.relative_to(ROOT)),
             "oof_qualifies_by_track": qualify_map,
             "screen_qualifies_by_track": {track: bool(score.get("qualifies_screen", False))
                                           for track, score in screen_results.items()},
             "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    state["status"] = "r12_conditional_oof_diagnostic_complete"
    state["r12_result"] = str(result_path.relative_to(ROOT))
    qualifying_screens = [track for track, score in screen_results.items() if score.get("qualifies_screen")]
    if qualifying_screens:
        state["next_action"] = "R12 conditional calibration passed screen for " + ", ".join(qualifying_screens) + "; run registered eight-month confirmation"
    else:
        state["next_action"] = "R12 conditional calibration did not qualify; register next distinct evidence-based method"
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(report)
    print(f"report={batch_root / 'research_report.md'}")
    return result


if __name__ == "__main__":
    run_r12_diagnostic()
