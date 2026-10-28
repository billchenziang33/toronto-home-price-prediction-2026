from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TRACK = "off_market"
PARENT = "G19_detached_shrink_075"
BASE = "G06_lot_garage"
EXPERT = "G18_detached_specialist"
CARD = "R19_past_oof_selected_G19_expert_weight"
SEED_MONTHS = ("2022-06", "2022-11")
OOF_MONTHS = ("2022-06", "2022-11", "2023-03", "2023-07", "2023-11", "2024-03")
SCREEN_MONTHS = ("2023-03", "2024-06")
PROCESS_MONTHS = ("2022-06", "2022-11", "2023-03", "2023-07", "2023-11", "2024-03", "2024-06")
PARENT_EXPERT_WEIGHT = 0.75
MINIMUM_OOF_ROWS = 1000
MINIMUM_DETACHED_OOF_ROWS = 500
MAX_MAPE_REGRESSION_PERCENT = 1.0


def blend_detached_expert(base_prediction, expert_prediction, detached_mask, expert_weight: float) -> np.ndarray:
    """Use G06 for all homes and substitute a weighted G06/G18 mix for Detached."""
    base = np.asarray(base_prediction, dtype=float)
    mask = np.asarray(detached_mask, dtype=bool)
    expert = np.asarray(expert_prediction, dtype=float)
    if base.ndim != 1 or mask.ndim != 1 or len(mask) != len(base):
        raise ValueError("base predictions and Detached mask must be aligned one-dimensional vectors")
    if expert.ndim != 1 or len(expert) != int(mask.sum()):
        raise ValueError("expert predictions must contain exactly the Detached rows")
    if (not np.isfinite(base).all() or (base <= 0.0).any()
            or not np.isfinite(expert).all() or (expert <= 0.0).any()):
        raise ValueError("base and expert predictions must be finite positive prices")
    if not np.isfinite(expert_weight) or not 0.0 <= float(expert_weight) <= 1.0:
        raise ValueError("expert_weight must be finite and between 0 and 1")
    result = np.round(base)
    result[mask] = np.round((1.0 - float(expert_weight)) * base[mask] + float(expert_weight) * expert)
    if not np.isfinite(result).all() or (result <= 0.0).any():
        raise ValueError("blended predictions must be finite positive prices")
    return result


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {"mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def _mape_change(parent_mape: float, candidate_mape: float) -> float:
    if parent_mape == 0.0:
        return 0.0 if candidate_mape == 0.0 else float("inf")
    return 100.0 * (candidate_mape - parent_mape) / parent_mape


def select_expert_weight(oof: pd.DataFrame, cutoff, *, minimum_rows: int = MINIMUM_OOF_ROWS,
                         minimum_detached_rows: int = MINIMUM_DETACHED_OOF_ROWS,
                         max_mape_regression_percent: float = MAX_MAPE_REGRESSION_PERCENT) -> dict:
    required = {"month", "ml_num", "date_sold", "sold_price", "parent_prediction",
                "g06_prediction", "g18_prediction", "detached"}
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"OOF frame is missing columns: {', '.join(missing)}")
    if minimum_rows < 1 or minimum_detached_rows < 0:
        raise ValueError("OOF minimum row counts are invalid")
    if not np.isfinite(max_mape_regression_percent) or max_mape_regression_percent < 0.0:
        raise ValueError("max_mape_regression_percent must be finite and nonnegative")

    cutoff_ts = pd.Timestamp(cutoff)
    dates = pd.to_datetime(oof.date_sold, errors="raise")
    past = oof.loc[dates < cutoff_ts].copy()
    past["date_sold"] = dates.loc[past.index]
    if not past.empty:
        expected_month = past.date_sold.dt.to_period("M").astype(str)
        if not expected_month.eq(past.month.astype(str)).all():
            raise ValueError("OOF month values do not match their sale dates")
        if past.duplicated(["month", "ml_num"]).any():
            raise ValueError("OOF rows must be unique by month and ml_num")
        mask = past.detached.to_numpy(dtype=bool)
        actual = past.sold_price.to_numpy(dtype=float)
        parent = past.parent_prediction.to_numpy(dtype=float)
        base = past.g06_prediction.to_numpy(dtype=float)
        expert = past.loc[past.detached.astype(bool), "g18_prediction"].to_numpy(dtype=float)
        prices = np.column_stack((actual, parent, base))
        expert_all = past.loc[past.detached.astype(bool), "g18_prediction"].to_numpy(dtype=float)
        if (not np.isfinite(prices).all() or (prices <= 0.0).any()
                or not np.isfinite(expert_all).all() or (expert_all <= 0.0).any()):
            raise ValueError("OOF prices must be finite positive values")
        if int(mask.sum()) != len(expert):
            raise ValueError("Detached masks and expert predictions do not align")
    else:
        mask = np.array([], dtype=bool)
        actual = parent = base = expert = np.array([], dtype=float)

    parent_metrics = _metrics(actual, parent) if len(past) else {
        "mdape_percent": float("nan"), "mape_percent": float("nan")
    }
    result = {
        "weight": PARENT_EXPERT_WEIGHT,
        "oof_rows": int(len(past)),
        "detached_oof_rows": int(mask.sum()),
        "oof_months": sorted(past.month.astype(str).unique().tolist()) if len(past) else [],
        "parent": parent_metrics,
        "selected": parent_metrics,
        "selected_mape_relative_change_percent": 0.0,
        "eligible_weights": [PARENT_EXPERT_WEIGHT],
        "reason": "insufficient_past_oof_rows",
    }
    if len(past) < minimum_rows or int(mask.sum()) < minimum_detached_rows:
        return result

    eligible = []
    for weight in np.linspace(0.0, 1.0, 21):
        prediction = blend_detached_expert(base, expert, mask, float(weight))
        metrics = _metrics(actual, prediction)
        change = _mape_change(parent_metrics["mape_percent"], metrics["mape_percent"])
        if change <= max_mape_regression_percent:
            eligible.append({"weight": float(round(weight, 10)), "metrics": metrics,
                             "mape_relative_change_percent": float(change)})
    if not eligible:
        return {**result, "reason": "no_weight_passed_mape_guard"}

    selected = min(eligible, key=lambda row: (
        row["metrics"]["mdape_percent"], row["metrics"]["mape_percent"],
        abs(row["weight"] - PARENT_EXPERT_WEIGHT), row["weight"],
    ))
    return {
        **result,
        "weight": selected["weight"],
        "selected": selected["metrics"],
        "selected_mape_relative_change_percent": selected["mape_relative_change_percent"],
        "eligible_weights": [row["weight"] for row in eligible],
        "reason": "selected_from_strictly_past_oof",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _verified_seed_components(month: str, homes: pd.DataFrame, truth: pd.DataFrame,
                              campaign) -> tuple[pd.DataFrame, dict[str, str]]:
    run_root = campaign.NEXT_STAGE_ROOT
    manifest_path = run_root / "run_manifest.json"
    audit_path = run_root / "n03_seed_oof_audit.json"
    state_path = run_root / "run_state.json"
    ledger_path = run_root / "fit_ledger.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (manifest.get("commit") != campaign.PROTOCOL["parent_commit"]
            or manifest.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or manifest.get("reserved_months_labels_opened") is not False
            or state.get("active_fit") is not None
            or state.get("active_stage") != "screen_search_complete_no_qualifier"):
        raise RuntimeError("N03 off-market seed is not a closed expected-data run")

    audit_row = audit["tracks"][TRACK][month]
    oof_path = run_root / "predictions/N03_seed_oof" / TRACK / f"{month}.csv"
    if _sha256(oof_path) != audit_row["feature_file_sha256"]:
        raise RuntimeError(f"N03 off-market OOF hash mismatch: {oof_path}")
    oof = pd.read_csv(oof_path, dtype={"ml_num": str})
    expected_ids = homes.ml_num.astype(str).to_numpy()
    if (not np.array_equal(oof.ml_num.astype(str).to_numpy(), expected_ids)
            or not np.array_equal(oof.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
            or not pd.to_datetime(oof.date_sold, errors="raise").dt.to_period("M").astype(str).eq(month).all()):
        raise RuntimeError(f"N03 off-market OOF labels/IDs do not align for {month}")
    vector_hash = hashlib.sha256(oof.parent_prediction.to_numpy(dtype="float64").tobytes()).hexdigest()
    if vector_hash != audit_row["prediction_vector_sha256"]:
        raise RuntimeError(f"N03 G19 seed prediction vector hash mismatch for {month}")

    ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    components: dict[str, pd.DataFrame] = {}
    inputs = {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
              for path in (manifest_path, audit_path, state_path, ledger_path, oof_path)}
    detached = homes.house_type_name.astype("string").eq("Detached").fillna(False).to_numpy()
    for role in ("g06", "g18"):
        fit_id = f"N03-seed:{role}:{TRACK}:{month}"
        fit = next((row for row in audit["component_fits"] if row.get("fit_id") == fit_id), None)
        csv_path = run_root / "predictions/N03_components" / TRACK / f"{month}_{role}.csv"
        meta_path = csv_path.with_suffix(".json")
        start = next((row for row in ledger_rows if row.get("event") == "fit_start"
                      and row.get("fit_id") == fit_id), None)
        complete = next((row for row in ledger_rows if row.get("event") == "fit_complete"
                         and row.get("fit_id") == fit_id and row.get("status") == "complete"), None)
        if (fit is None or start is None or complete is None
                or start.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
                or _sha256(csv_path) != fit.get("prediction_sha256")
                or _sha256(meta_path) != fit.get("metadata_sha256")
                or _sha256(csv_path) != complete.get("prediction_file_sha256")):
            raise RuntimeError(f"N03 component audit/ledger mismatch: {fit_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fit_id") != fit_id or meta.get("track") != TRACK or meta.get("month") != month:
            raise RuntimeError(f"N03 component metadata identity mismatch: {fit_id}")
        components[role] = pd.read_csv(csv_path, dtype={"ml_num": str})
        inputs[str(csv_path.relative_to(campaign.ROOT)).replace("\\", "/")] = _sha256(csv_path)
        inputs[str(meta_path.relative_to(campaign.ROOT)).replace("\\", "/")] = _sha256(meta_path)

    base = components["g06"]
    expert = components["g18"]
    if (not np.array_equal(base.ml_num.astype(str).to_numpy(), expected_ids)
            or not np.array_equal(expert.ml_num.astype(str).to_numpy(), expected_ids[detached])):
        raise RuntimeError(f"N03 G06/G18 IDs do not align for {month}")
    base_prediction = base.raw_prediction.to_numpy(float)
    expert_prediction = expert.raw_prediction.to_numpy(float)
    reconstructed = blend_detached_expert(base_prediction, expert_prediction, detached, PARENT_EXPERT_WEIGHT)
    if not np.array_equal(reconstructed, oof.parent_prediction.to_numpy(float)):
        raise RuntimeError(f"N03 components do not reconstruct G19 in {month}")
    result = pd.DataFrame({"month": month, "ml_num": expected_ids,
                           "date_sold": pd.to_datetime(oof.date_sold, errors="raise").to_numpy(),
                           "sold_price": truth.sold_price.to_numpy(float),
                           "parent_prediction": reconstructed,
                           "g06_prediction": base_prediction,
                           "g18_prediction": np.nan,
                           "detached": detached})
    result.loc[detached, "g18_prediction"] = expert_prediction
    return result, inputs


def validate_g18_specialist_artifact(metadata: dict, expected_ids, expert_ids,
                                     expert_prediction, base_prediction, detached_mask, *,
                                     expected_detached_rows: int,
                                     expected_month: str,
                                     expected_variant: str = EXPERT,
                                     expected_data_sha256: str | None = None) -> dict:
    """Verify G18 routing from artifact identity and its actual row predictions.

    Several frozen historical G18 sidecars predate the specialist metadata fields.
    For those files, full-row predictions are accepted only when all non-Detached
    prices exactly match G06; the routed Detached rows are then directly auditable.
    """
    expected = np.asarray(expected_ids, dtype=str)
    observed = np.asarray(expert_ids, dtype=str)
    expert = np.asarray(expert_prediction, dtype=float)
    base = np.asarray(base_prediction, dtype=float)
    mask = np.asarray(detached_mask, dtype=bool)
    if (expected.ndim != 1 or observed.ndim != 1 or expert.ndim != 1 or base.ndim != 1
            or mask.ndim != 1 or len(expected) != len(base) or len(expected) != len(mask)
            or len(observed) != len(expert)):
        raise ValueError("G18/G06 IDs, prices, and route mask must be aligned one-dimensional vectors")
    if not np.array_equal(observed, expected):
        raise ValueError("G18 prediction IDs do not align with the full evaluation fold")
    if (not np.isfinite(expert).all() or (expert <= 0.0).any()
            or not np.isfinite(base).all() or (base <= 0.0).any()):
        raise ValueError("G18 and G06 prices must be finite positive values")
    if int(mask.sum()) != int(expected_detached_rows):
        raise ValueError("expected Detached row count does not match the route mask")
    identity = {"month": metadata.get("month"), "track": metadata.get("track"),
                "variant": metadata.get("variant"), "parent_id": metadata.get("parent_id")}
    if (identity["month"] != expected_month or identity["track"] != TRACK
            or identity["variant"] != expected_variant or identity["parent_id"] != BASE):
        raise ValueError("G18 metadata does not identify the expected month/track/variant/parent")
    data_hash = metadata.get("data_sha256")
    if expected_data_sha256 is not None and data_hash != expected_data_sha256:
        raise ValueError("G18 metadata data hash differs from the expected dataset")
    metadata_type = metadata.get("specialist_house_type")
    if metadata_type not in (None, "Detached"):
        raise ValueError("G18 metadata identifies a non-Detached specialist")
    metadata_rows = metadata.get("evaluation_rows")
    if metadata_rows not in (None, int(mask.sum())):
        raise ValueError("G18 metadata evaluation row count differs from Detached route")
    if not np.array_equal(expert[~mask], base[~mask]):
        raise ValueError("G18 changed non-Detached rows outside its specialist route")
    return {
        "legacy_metadata_fields_missing": metadata_type is None or metadata_rows is None,
        "verified_expert_rows": int(mask.sum()),
        "full_prediction_rows": int(len(expected)),
        "data_sha256": data_hash,
    }


def _load_fold_components(month: str, greedy, budget, campaign) -> tuple[pd.DataFrame, dict[str, str]]:
    from experiments.model_optimization_v2.folds import build_month_fold

    listings, _, _, _, _ = greedy._load_fold(month)
    train, homes, truth = build_month_fold(listings, month, TRACK)
    cutoff = pd.Period(month, freq="M").start_time
    if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
        raise RuntimeError(f"G06/G18 fold includes future rows in {month}")
    if month in SEED_MONTHS:
        return _verified_seed_components(month, homes, truth, campaign)

    parent, parent_path = budget._check_parent_artifact(greedy, month, TRACK, homes, truth, PARENT)
    base = greedy._load_parent_prediction(month, TRACK, homes, truth, BASE)
    expert = greedy._load_parent_prediction(month, TRACK, homes, truth, EXPERT)
    detached = homes.house_type_name.astype("string").eq("Detached").fillna(False).to_numpy()
    if not detached.any():
        raise RuntimeError(f"no Detached evaluation rows in {month}")
    g18_meta_path = (greedy.OUTPUT / "neighbors" / EXPERT / TRACK / f"{month}.json")
    g18_metadata = json.loads(g18_meta_path.read_text(encoding="utf-8"))
    expert_validation = validate_g18_specialist_artifact(
        g18_metadata, homes.ml_num.astype(str).to_numpy(), expert.ml_num.astype(str).to_numpy(),
        expert.predicted_price.to_numpy(float), base.predicted_price.to_numpy(float), detached,
        expected_detached_rows=int(detached.sum()), expected_month=month,
        expected_data_sha256=campaign.PROTOCOL["data_sha256"],
    )
    reconstructed = blend_detached_expert(
        base.predicted_price.to_numpy(float), expert.loc[detached, "predicted_price"].to_numpy(float),
        detached, PARENT_EXPERT_WEIGHT,
    )
    if not np.array_equal(reconstructed, parent.predicted_price.to_numpy(float)):
        raise RuntimeError(f"saved G06/G18 components do not reconstruct G19 for {month}")
    expert_values = np.full(len(homes), np.nan, dtype=float)
    expert_values[detached] = expert.loc[detached, "predicted_price"].to_numpy(float)
    frame = pd.DataFrame({"month": month, "ml_num": homes.ml_num.astype(str).to_numpy(),
                          "date_sold": pd.to_datetime(listings.set_index(listings.ml_num.astype(str))
                              .loc[homes.ml_num.astype(str), "date_sold"].to_numpy(), errors="raise"),
                          "sold_price": truth.sold_price.to_numpy(float),
                          "parent_prediction": parent.predicted_price.to_numpy(float),
                          "g06_prediction": base.predicted_price.to_numpy(float),
                          "g18_prediction": expert_values, "detached": detached})
    inputs = {}
    for path in (parent_path, parent_path.with_suffix(".json")):
        inputs[str(path.relative_to(campaign.ROOT)).replace("\\", "/")] = _sha256(path)
    for component in (BASE, EXPERT):
        path = greedy.OUTPUT / "neighbors" / component / TRACK / f"{month}.csv"
        meta_path = path.with_suffix(".json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_variant = BASE if component == BASE else EXPERT
        if (metadata.get("month") != month or metadata.get("track") != TRACK
                or metadata.get("variant") != expected_variant
                or metadata.get("data_sha256") != campaign.PROTOCOL["data_sha256"]):
            raise RuntimeError(f"saved {component} metadata identity/data mismatch for {month}")
        inputs[str(path.relative_to(campaign.ROOT)).replace("\\", "/")] = _sha256(path)
        inputs[str(meta_path.relative_to(campaign.ROOT)).replace("\\", "/")] = _sha256(meta_path)
    return frame, inputs


def _report(result: dict) -> str:
    score = result["tracks"][TRACK]
    lines = [
        "# R19：根据严格过去 OOF 选择 G19 的 Detached 专家权重", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "只调整 G19 中 Detached 人群的 G18/G06 组合系数；非 Detached 预测仍完全使用 G06。候选系数按每个外层 cutoff 单独从严格更早的 OOF 月份选择。", "",
        "## 选择的专家权重", "", "| 外层月份 | 专家权重 | OOF月份 | 总行数 | Detached行数 | 父→选中 MdAPE | MAPE相对变化 |",
        "|---|---:|---|---:|---:|---:|---:|"]
    for row in result["weight_selections"]:
        lines.append(
            f"| {row['screen_month']} | {row['weight']:.2f} | {', '.join(row['oof_months'])} | "
            f"{row['oof_rows']} | {row['detached_oof_rows']} | "
            f"{row['parent']['mdape_percent']:.5f}% → {row['selected']['mdape_percent']:.5f}% | "
            f"{row['selected_mape_relative_change_percent']:+.3f}% |"
        )
    lines.extend(["", "## 双月筛选", "",
                  "| 赛道 | 父→候选 MdAPE | 相对改善 | MAPE相对变化 | 父→候选 P90 APE | 通过筛选 |",
                  "|---|---:|---:|---:|---:|---|"])
    lines.append(
        f"| {TRACK} | {score['parent']['mdape_percent']:.5f}% → {score['candidate']['mdape_percent']:.5f}% | "
        f"{score['relative_mdape_improvement_percent']:+.3f}% | "
        f"{score['mape_relative_change_percent']:+.3f}% | "
        f"{score['parent']['p90_ape_percent']:.3f}% → {score['candidate']['p90_ape_percent']:.3f}% | "
        f"{'是' if score['qualifies_screen'] else '否'} |"
    )
    lines.extend(["", "## 逐月结果", "", "| 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 | 父 MAPE | 候选 MAPE |",
                  "|---|---:|---:|---:|---:|---:|"])
    for month, monthly in score["monthly"].items():
        lines.append(
            f"| {month} | {monthly['parent']['mdape_percent']:.5f}% | "
            f"{monthly['candidate']['mdape_percent']:.5f}% | "
            f"{monthly['relative_mdape_improvement_percent']:+.3f}% | "
            f"{monthly['parent']['mape_percent']:.5f}% | {monthly['candidate']['mape_percent']:.5f}% |"
        )
    lines.extend(["", "## 运行边界", "",
                  "实际 learner fits：0。N03 seed 组件、各月 G06/G18 与 G19 父输出均逐文件核验，并在全部七折以 0.75 专家权重精确重建 G19。",
                  "筛选门槛：pooled MdAPE 相对改善至少 0.25%、两个月都改善、MAPE 恶化不超过 1%。失败则不确认、不改父模型。保留月和隐藏标签未读取。", "",
                  "- 逐房筛选预测：`paired_predictions.csv`",
                  "- OOF 权重选择与指标：`r19_screen_results.json`",
                  "- 来源文件哈希：`run_manifest.json`",
                  "- Campaign 事件账本：`../../fits.jsonl`", ""])
    return "\n".join(lines)


def run_r19_screen() -> dict:
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy

    ROOT = campaign.ROOT
    source = budget._source_identity()
    if campaign.PROTOCOL["cards"].get(CARD, {}).get("status") != "registered":
        raise RuntimeError("R19 must be registered in campaign_protocol.json before scoring")
    run_started = datetime.now(timezone.utc)
    run_id = "bill_target74_r19_expert_weight_" + run_started.strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "oof_months": list(OOF_MONTHS),
                      "fit_start_count": 0, "parent": PARENT, "base": BASE, "expert": EXPERT,
                      "parent_expert_weight": PARENT_EXPERT_WEIGHT,
                      "minimum_oof_rows": MINIMUM_OOF_ROWS,
                      "minimum_detached_oof_rows": MINIMUM_DETACHED_OOF_ROWS,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    input_hashes: dict[str, str] = {}
    fit_months: dict[str, dict] = {}
    weight_selections = []
    screen_rows = []
    try:
        for month in PROCESS_MONTHS:
            fold, hashes = _load_fold_components(month, greedy, budget, campaign)
            input_hashes.update(hashes)
            fit_months[month] = fold

        all_oof = pd.concat([fit_months[month] for month in OOF_MONTHS], ignore_index=True)
        if all_oof.duplicated(["month", "ml_num"]).any():
            raise RuntimeError("G06/G18 OOF rows contain duplicate fold/home pairs")
        for month in SCREEN_MONTHS:
            cutoff = pd.Period(month, freq="M").start_time
            selection = select_expert_weight(
                all_oof, cutoff, minimum_rows=MINIMUM_OOF_ROWS,
                minimum_detached_rows=MINIMUM_DETACHED_OOF_ROWS,
                max_mape_regression_percent=MAX_MAPE_REGRESSION_PERCENT,
            )
            selection["screen_month"] = month
            weight_selections.append(selection)
            frame = fit_months[month]
            candidate_prediction = blend_detached_expert(
                frame.g06_prediction.to_numpy(float),
                frame.loc[frame.detached.astype(bool), "g18_prediction"].to_numpy(float),
                frame.detached.to_numpy(bool), selection["weight"],
            )
            parent = pd.DataFrame({"ml_num": frame.ml_num.astype(str).to_numpy(),
                                   "sold_price": frame.sold_price.to_numpy(float),
                                   "predicted_price": frame.parent_prediction.to_numpy(float)})
            candidate = pd.DataFrame({"ml_num": frame.ml_num.astype(str).to_numpy(),
                                      "sold_price": frame.sold_price.to_numpy(float),
                                      "predicted_price": candidate_prediction})
            truth = pd.DataFrame({"ml_num": frame.ml_num.astype(str).to_numpy(),
                                  "sold_price": frame.sold_price.to_numpy(float)})
            screen_rows.append(helpers._candidate_frame(month, TRACK, truth, parent, candidate))
            campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                              "track": TRACK, "month": month, "parent_id": PARENT,
                              "expert_weight": selection["weight"], "oof_rows": selection["oof_rows"],
                              "detached_oof_rows": selection["detached_oof_rows"],
                              "oof_months": selection["oof_months"],
                              "date_cutoff_exclusive": cutoff.isoformat()})

        paired = pd.concat(screen_rows, ignore_index=True)
        paired_path = output / "paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        tracks = {TRACK: helpers._track_result(paired)}
        result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                  "card": CARD, "parents": {TRACK: PARENT}, "screen_months": list(SCREEN_MONTHS),
                  "weight_selections": weight_selections, "tracks": tracks,
                  "fit_starts": 0, "verified_component_reconstruction_months": list(PROCESS_MONTHS),
                  "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                  "data_sha256": source["data_sha256"]}
        result_path = output / "r19_screen_results.json"
        _write_json(output / "r19_screen_results.json", result)
        source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r19_expert_weight.py",
                        campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                        ROOT / "experiments/bill_greedy/runner.py",
                        ROOT / "experiments/bill_greedy/aligned_inference.py",
                        ROOT / "experiments/model_optimization_v2/folds.py"]
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": source,
            "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                            for path in source_files},
            "input_sha256": input_hashes, "weight_selections": weight_selections,
            "fit_starts": 0, "paired_predictions_sha256": _sha256(paired_path),
            "result_sha256": _sha256(result_path), "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        }
        _write_json(output / "run_manifest.json", manifest)
        (output / "research_report.md").write_text(_report(result), encoding="utf-8")
        passed = tracks[TRACK]["qualifies_screen"]
        next_action = ("R19 off-market passed screen; run strict sequential eight-month confirmation" if passed
                      else "R19 did not qualify; keep G19 and register a distinct next candidate")
        campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                          "fit_starts": 0, "result_path": str(result_path.relative_to(ROOT)),
                          "qualifies_by_track": {TRACK: passed}, "reserved_months_labels_opened": False,
                          "hidden_labels_opened": False})
        campaign._write_progress("r19_expert_weight_screen_complete", next_action=next_action,
                                 r19_result=str(result_path.relative_to(ROOT)).replace("\\", "/"),
                                 latest_fit_count=0)
        return result
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
                   "fit_starts": 0, "completed_component_months": list(fit_months),
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": failure["error"], "fit_starts": 0})
        campaign._write_progress("r19_expert_weight_failed",
                                 next_action="inspect R19 failure checkpoint before choosing a next candidate",
                                 r19_failure=str((output / "failure_checkpoint.json").relative_to(ROOT)).replace("\\", "/"))
        raise
