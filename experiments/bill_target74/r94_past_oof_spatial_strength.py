"""R94: choose each house-type's R93 spatial correction strength from earlier OOF rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r93_spatial_oof_calibration as r93
from experiments.bill_target74 import runner as campaign


CARD = "R94_past_oof_house_type_spatial_strength"
TRACKS = r93.TRACKS
MONTHS = r93.MONTHS
MIN_TYPE_HISTORY_ROWS = 500
MIN_TYPE_HISTORY_MONTHS = 2
GAMMA_GRID = tuple(np.round(np.arange(0.0, 1.0001, 0.1), 2).tolist())
MAX_MAPE_REGRESSION_PERCENT = 1.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261028
R93_RESULT_REL = Path(
    "outputs/bill_target74/bill_target74_unbounded_20261024/batches/"
    "r93s_20261025T201838Z/development_results.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _r93_paths() -> tuple[Path, Path, Path]:
    result_path = campaign.ROOT / R93_RESULT_REL
    prediction_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    return result_path, prediction_path, manifest_path


def _verify_r93() -> dict:
    result_path, prediction_path, manifest_path = _r93_paths()
    if not all(path.is_file() for path in (result_path, prediction_path, manifest_path)):
        raise FileNotFoundError("R94 requires complete saved R93 result, paired rows and manifest")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("card") != r93.CARD or result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R93 result identity or data hash mismatch")
    if result.get("reserved_months_labels_opened") is not False or result.get("hidden_labels_opened") is not False:
        raise RuntimeError("R93 result reports restricted label access")
    if (manifest.get("paired_predictions_sha256") != _sha256(prediction_path)
            or manifest.get("result_sha256") != _sha256(result_path)):
        raise RuntimeError("R93 result/prediction hash does not match its manifest")
    return {
        "run_id": result["run_id"],
        "result_path": _relative(result_path),
        "result_sha256": _sha256(result_path),
        "predictions_path": _relative(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as identity

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == _sha256(Path(__file__)):
            return existing
        raise RuntimeError(f"R94 already has status {existing.get('status')!r}; inspect before changing it")
    source = identity._source_identity()
    r93_identity = _verify_r93()
    registration = {
        "status": "registered",
        "tracks": list(TRACKS),
        "development_months": list(MONTHS),
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "base_candidates": {
            "off_market": "R52 candidate plus saved strict-past R93 spatial correction values",
            "list_aware": "R35 candidate plus saved strict-past R93 spatial correction values",
        },
        "r93_source": r93_identity,
        "method": (
            "For each track, outer development month and house type, select a multiplier gamma "
            "for the already-OOF R93 local spatial correction using only earlier R93 walk-forward "
            "rows. Candidate gamma values are 0.0..1.0 in 0.1 steps. Minimize past pooled MdAPE "
            "within that house type, subject to no more than 1% past MAPE regression versus the "
            "unchanged base prediction. Require at least 500 earlier rows across at least two "
            "earlier OOF months; otherwise use gamma=0. Ties favor lower MAPE, then smaller gamma. "
            "Apply the selected gamma only to the current month, round to the nearest dollar, and "
            "never use current-month sale labels to select gamma."
        ),
        "difference_from_prior_work": (
            "R93 used one fixed local-residual shrinkage rule for all house types; it improved "
            "off-market Detached/Semi-Detached slices but worsened Condo Apt. R94 chooses only "
            "the correction strength per house type from strictly earlier OOF rows. It neither "
            "changes base predictions or spatial neighbor rules nor selects on the target fold."
        ),
        "parameters": {
            "minimum_type_history_rows": MIN_TYPE_HISTORY_ROWS,
            "minimum_type_history_months": MIN_TYPE_HISTORY_MONTHS,
            "gamma_grid": list(GAMMA_GRID),
            "maximum_past_mape_regression_percent": MAX_MAPE_REGRESSION_PERCENT,
            "tie_break": "lower past MAPE, then smaller gamma",
        },
        "development_gate": dict(campaign.PROTOCOL["gates"]),
        "data_sha256": source["data_sha256"],
        "parent_commit": source["commit"],
        "source_sha256": _sha256(Path(__file__)),
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "tracks": list(TRACKS),
        "development_months": list(MONTHS),
        "expected_calibration_fit_starts": registration["expected_calibration_fit_starts"],
        "r93_result_sha256": r93_identity["result_sha256"],
        "r93_predictions_sha256": r93_identity["predictions_sha256"],
        "data_sha256": source["data_sha256"], "source_sha256": registration["source_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r94_registered",
        next_action="run R94 strict-past per-house-type selection of R93 spatial correction strength",
        r94_card=CARD,
    )
    return registration


def amend_registration(reason: str) -> dict:
    """Amend only the executable source hash after a verified zero-fit setup failure."""
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if not existing or existing.get("status") != "registered":
        raise RuntimeError("R94 can only amend an existing registered card")
    events = [
        json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed_runs = {
        row.get("run_id") for row in events
        if row.get("event") == "card_failed" and row.get("card") == CARD
    }
    for failed_run in failed_runs:
        starts = [
            row for row in events
            if row.get("event") == "fit_start" and row.get("card") == CARD
            and row.get("run_id") == failed_run
        ]
        if starts:
            raise RuntimeError("R94 failed run already has fit starts; source amendment is not setup-only")
    old_hash = existing.get("source_sha256")
    new_hash = _sha256(Path(__file__))
    if old_hash == new_hash:
        return existing
    existing["source_sha256"] = new_hash
    existing.setdefault("registration_amendments", []).append({
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "old_source_sha256": old_hash,
        "new_source_sha256": new_hash,
        "reason": reason,
        "hypothesis_parameters_and_gates_unchanged": True,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registration_amended", "card": CARD,
        "old_source_sha256": old_hash, "new_source_sha256": new_hash,
        "reason": reason, "fit_starts_in_failed_attempts": 0,
        "hypothesis_parameters_and_gates_unchanged": True,
    })
    campaign._write_progress(
        "r94_registration_amended",
        next_action="retry R94 after the zero-fit prediction-column setup correction",
        latest_error=None,
    )
    return existing


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = 100.0 * np.abs(prediction - actual) / actual
    return {
        "n": int(len(ape)),
        "mdape_percent": float(np.median(ape)),
        "mape_percent": float(np.mean(ape)),
        "p90_ape_percent": float(np.quantile(ape, 0.90)),
        "ape_over_20_percent": float(np.mean(ape > 20.0) * 100.0),
    }


def _select_gamma(history: pd.DataFrame) -> tuple[float, dict]:
    count = len(history)
    month_count = int(history.month.nunique())
    if count < MIN_TYPE_HISTORY_ROWS or month_count < MIN_TYPE_HISTORY_MONTHS:
        return 0.0, {
            "selected_gamma": 0.0, "history_rows": count,
            "history_months": month_count, "fallback": True,
            "reason": "insufficient_past_type_rows_or_months",
        }
    actual = history.sold_price.to_numpy(float)
    base = history.base_prediction.to_numpy(float)
    correction = history.correction_log.to_numpy(float)
    base_metrics = _metrics(actual, base)
    eligible = []
    for gamma in GAMMA_GRID:
        prediction = np.rint(base * np.exp(gamma * correction))
        metrics = _metrics(actual, prediction)
        mape_change = 100.0 * (metrics["mape_percent"] - base_metrics["mape_percent"]) / base_metrics["mape_percent"]
        if mape_change <= MAX_MAPE_REGRESSION_PERCENT:
            eligible.append((metrics["mdape_percent"], metrics["mape_percent"], gamma))
    if not eligible:
        return 0.0, {
            "selected_gamma": 0.0, "history_rows": count,
            "history_months": month_count, "fallback": True,
            "reason": "no_gamma_passed_mape_guard",
        }
    mdape, mape, gamma = min(eligible)
    return float(gamma), {
        "selected_gamma": float(gamma), "history_rows": count,
        "history_months": month_count, "fallback": False,
        "past_base_mdape_percent": base_metrics["mdape_percent"],
        "past_selected_mdape_percent": mdape,
        "past_base_mape_percent": base_metrics["mape_percent"],
        "past_selected_mape_percent": mape,
        "past_mape_relative_change_percent": 100.0 * (mape - base_metrics["mape_percent"]) / base_metrics["mape_percent"],
    }


def _score_track(frame: pd.DataFrame) -> dict:
    actual = frame.sold_price.to_numpy(float)
    base = frame.base_prediction.to_numpy(float)
    candidate = frame.candidate_prediction.to_numpy(float)
    r93_prediction = frame.r93_candidate_prediction.to_numpy(float)
    base_metrics = _metrics(actual, base)
    candidate_metrics = _metrics(actual, candidate)
    r93_metrics = _metrics(actual, r93_prediction)
    monthly = {}
    for month, group in frame.groupby("month", sort=True):
        before = _metrics(group.sold_price.to_numpy(float), group.base_prediction.to_numpy(float))
        after = _metrics(group.sold_price.to_numpy(float), group.candidate_prediction.to_numpy(float))
        monthly[str(month)] = {
            "n": len(group), "base": before, "candidate": after,
            "relative_mdape_improvement_percent": 100.0 * (
                before["mdape_percent"] - after["mdape_percent"]
            ) / before["mdape_percent"],
        }
    gain = 100.0 * (base_metrics["mdape_percent"] - candidate_metrics["mdape_percent"]) / base_metrics["mdape_percent"]
    mape_change = 100.0 * (candidate_metrics["mape_percent"] - base_metrics["mape_percent"]) / base_metrics["mape_percent"]
    improvements = [x["relative_mdape_improvement_percent"] for x in monthly.values()]
    months_improved = sum(value > 0 for value in improvements)
    worst_regression = max([0.0] + [-value for value in improvements])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_ape = 100.0 * np.abs(base - actual) / actual
    candidate_ape = 100.0 * np.abs(candidate - actual) / actual
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(actual), size=len(actual))
        differences[replicate] = np.median(candidate_ape[indices]) - np.median(base_ape[indices])
    bootstrap = {
        "lower95": float(np.quantile(differences, 0.025)),
        "upper95": float(np.quantile(differences, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }
    gates = campaign.PROTOCOL["gates"]
    target = 7.0 if frame.track.iloc[0] == "off_market" else 4.0
    gate_checks = {
        "relative_mdape_improvement": gain >= gates["development_relative_mdape_improvement_percent"],
        "months_improved": months_improved >= gates["development_months_improved_min"],
        "worst_month_regression": worst_regression <= gates["development_worst_month_regression_max_percent"],
        "mape_regression": mape_change <= gates["development_mape_regression_max_percent"],
        "bootstrap_upper_below_zero": bootstrap["upper95"] < 0.0,
        "strict_target": candidate_metrics["mdape_percent"] < target,
    }
    return {
        "base": base_metrics,
        "r93_candidate": r93_metrics,
        "candidate": candidate_metrics,
        "relative_mdape_improvement_percent": gain,
        "mape_relative_change_percent": mape_change,
        "months_improved": months_improved,
        "worst_month_regression_percent": worst_regression,
        "paired_bootstrap_mdape_difference_pp": bootstrap,
        "monthly": monthly,
        "gate_checks": gate_checks,
        "qualifies_development": all(gate_checks.values()),
        "target_mdape_strictly_below": target,
        "relative_mdape_improvement_vs_r93_percent": 100.0 * (
            r93_metrics["mdape_percent"] - candidate_metrics["mdape_percent"]
        ) / r93_metrics["mdape_percent"],
    }


def run() -> dict:
    from experiments.bill_target74 import r04_ablation as identity

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R94 must be registered before execution")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R94 source hash changed after preregistration")
    source = identity._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R94 source data differs from registration")
    r93_identity = _verify_r93()
    r93_rows = pd.read_csv(campaign.ROOT / r93_identity["predictions_path"], dtype={"ml_num": str})
    if set(r93_rows.month.unique()) != set(MONTHS) or len(r93_rows) != 31284:
        raise RuntimeError("R93 saved predictions do not cover exact development months and tracks")
    if set(r93_rows.track.unique()) != set(TRACKS) or r93_rows.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R93 saved predictions have unexpected tracks or duplicate IDs")
    if set(r93_rows.month.unique()).intersection(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("R94 input contains a reserved month")
    if not np.isfinite(r93_rows.correction_log.to_numpy(float)).all():
        raise RuntimeError("R93 correction log contains non-finite values")
    r93_rows["r93_candidate_prediction"] = r93_rows.candidate_prediction.astype(float)
    run_id = "r94g_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(campaign.PROTOCOL_PATH, provenance / "registered_campaign_protocol.json")
    result_path, prediction_path, manifest_path = _r93_paths()
    shutil.copy2(result_path, provenance / "r93_development_results.json")
    shutil.copy2(manifest_path, provenance / "r93_run_manifest.json")
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "tracks": list(TRACKS), "development_months": list(MONTHS),
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "r93_predictions_sha256": r93_identity["predictions_sha256"],
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    fit_details = []
    output_frames = []
    fit_start_count = 0
    try:
        for track in TRACKS:
            track_rows = r93_rows.loc[r93_rows.track.eq(track)].copy()
            track_rows["month"] = track_rows.month.astype(str)
            track_rows["house_type_name"] = (
                track_rows.house_type_name.astype("string").fillna("__MISSING__").astype(str)
            )
            monthly_rows = []
            for month_index, month in enumerate(MONTHS):
                current = track_rows.loc[track_rows.month.eq(month)].copy().reset_index(drop=True)
                if current.empty:
                    raise RuntimeError(f"R93 source has no rows for {track}/{month}")
                history = track_rows.loc[track_rows.month.isin(MONTHS[:month_index])].copy()
                gamma_by_type = {}
                type_details = {}
                fit_id = None
                if month_index > 0:
                    fit_start_count += 1
                    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{run_id}"
                    campaign._append({
                        "event": "fit_start", "card": CARD, "run_id": run_id,
                        "fit_id": fit_id, "stage": "calibration_strength_selection",
                        "track": track, "month": month, "training_oof_rows": len(history),
                        "training_oof_months": sorted(history.month.unique().tolist()),
                        "current_fold_labels_used_for_selection": False,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "data_sha256": source["data_sha256"],
                    })
                try:
                    for property_type in sorted(current.house_type_name.unique().tolist()):
                        past_type = history.loc[history.house_type_name.eq(property_type)].copy()
                        gamma, detail = _select_gamma(past_type)
                        gamma_by_type[property_type] = gamma
                        type_details[property_type] = detail
                    current["selected_gamma"] = current.house_type_name.map(gamma_by_type).fillna(0.0).astype(float)
                    current["applied_correction_log"] = current.correction_log * current.selected_gamma
                    current["candidate_prediction"] = np.rint(
                        current.base_prediction.to_numpy(float)
                        * np.exp(current.applied_correction_log.to_numpy(float))
                    )
                    if (not np.isfinite(current.candidate_prediction.to_numpy(float)).all()
                            or (current.candidate_prediction <= 0).any()):
                        raise RuntimeError("R94 generated invalid predictions")
                except Exception as exc:
                    if fit_id is not None:
                        campaign._append({
                            "event": "fit_complete", "card": CARD, "run_id": run_id,
                            "fit_id": fit_id, "stage": "calibration_strength_selection",
                            "track": track, "month": month, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                    raise
                if fit_id is not None:
                    detail = {
                        "fit_id": fit_id, "track": track, "month": month,
                        "status": "complete", "training_oof_rows": len(history),
                        "training_oof_months": sorted(history.month.unique().tolist()),
                        "selected_gamma_by_house_type": gamma_by_type,
                        "selection_details_by_house_type": type_details,
                        "current_fold_labels_used_for_selection": False,
                    }
                    fit_details.append(detail)
                    campaign._append({
                        "event": "fit_complete", "card": CARD, "run_id": run_id,
                        "stage": "calibration_strength_selection", **detail,
                        "reserved_months_labels_opened": False,
                        "hidden_labels_opened": False,
                    })
                current["base_ape_percent"] = (
                    100.0 * np.abs(current.base_prediction - current.sold_price) / current.sold_price
                )
                current["candidate_ape_percent"] = (
                    100.0 * np.abs(current.candidate_prediction - current.sold_price) / current.sold_price
                )
                monthly_rows.append(current[[
                    "month", "track", "ml_num", "sold_price", "base_prediction",
                    "r93_candidate_prediction", "candidate_prediction", "base_ape_percent",
                    "candidate_ape_percent", "house_type_name", "correction_log",
                    "selected_gamma", "applied_correction_log", "neighbor_count",
                    "effective_neighbor_count",
                ]])
            output_frames.extend(monthly_rows)
        paired = pd.concat(output_frames, ignore_index=True)
        if len(paired) != 31284 or paired.duplicated(["month", "track", "ml_num"]).any():
            raise RuntimeError("R94 output row count or ID uniqueness is invalid")
        tracks = {
            track: _score_track(paired.loc[paired.track.eq(track)].copy())
            for track in TRACKS
        }
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": "strict_past_walk_forward_development",
            "development_months": list(MONTHS), "fit_starts": fit_start_count,
            "tracks": tracks, "fit_details": fit_details,
            "r93_source": r93_identity,
            "gamma_grid": list(GAMMA_GRID),
            "minimum_type_history_rows": MIN_TYPE_HISTORY_ROWS,
            "minimum_type_history_months": MIN_TYPE_HISTORY_MONTHS,
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
            "data_sha256": source["data_sha256"],
        }
        paired_path = output / "paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        result_path = output / "development_results.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        code_paths = [Path(__file__), campaign.PROTOCOL_PATH, Path(r93.__file__), campaign.PACKAGE / "runner.py"]
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": source,
            "source_sha256": {_relative(path): _sha256(path) for path in code_paths if path.is_file()},
            "r93_source": r93_identity,
            "fit_starts": fit_start_count, "fit_details": fit_details,
            "paired_predictions_sha256": _sha256(paired_path),
            "result_sha256": _sha256(result_path),
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        lines = [
            "# R94：严格过去 OOF 按房型选择空间修正强度", "",
            f"运行：`{run_id}`；输入为哈希核验的 R93 OOF 预测。",
            "每个外层月份只从更早月份选择房型 gamma；至少 500 条历史房源、至少两个历史 OOF 月，"
            "并要求过去 MAPE 回归不超过 1%。不足时 gamma=0。", "",
            "| Track | Base MdAPE | R93 MdAPE | R94 MdAPE | 相对 Base 改善 | MAPE变化 | 月份改善 | bootstrap upper95 (pp) | 达到目标 | 晋级 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for track, score in tracks.items():
            lines.append(
                f"| {track} | {score['base']['mdape_percent']:.5f}% | "
                f"{score['r93_candidate']['mdape_percent']:.5f}% | "
                f"{score['candidate']['mdape_percent']:.5f}% | "
                f"{score['relative_mdape_improvement_percent']:+.3f}% | "
                f"{score['mape_relative_change_percent']:+.3f}% | "
                f"{score['months_improved']}/8 | "
                f"{score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} | "
                f"{'是' if score['candidate']['mdape_percent'] < score['target_mdape_strictly_below'] else '否'} | "
                f"{'是' if score['qualifies_development'] else '否'} |"
            )
        lines.extend(["", "## 逐月 MdAPE", "", "| Track | Month | Base | R94 | 相对改善 |", "|---|---|---:|---:|---:|"])
        for track, score in tracks.items():
            for month, monthly in score["monthly"].items():
                lines.append(
                    f"| {track} | {month} | {monthly['base']['mdape_percent']:.5f}% | "
                    f"{monthly['candidate']['mdape_percent']:.5f}% | "
                    f"{monthly['relative_mdape_improvement_percent']:+.3f}% |"
                )
        lines.extend([
            "", f"Calibration-strength fit starts：{fit_start_count}。",
            "R94 使用先前已冻结的 R93 walk-forward 修正值，不重新读取挂牌数据，也不读取保留或隐藏标签。",
            "保留集验证资格仍不可用。", "",
        ])
        report_path = output / "research_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        passed = {track: score["qualifies_development"] for track, score in tracks.items()}
        campaign._append({
            "event": "card_complete", "card": CARD, "run_id": run_id,
            "fit_starts": fit_start_count, "qualifies_by_track": passed,
            "result_path": _relative(result_path),
            "paired_predictions_sha256": manifest["paired_predictions_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r94_development_complete",
            next_action=(
                "R94 passed development for: " + ", ".join(track for track, ok in passed.items() if ok)
                if any(passed.values()) else
                "R94 spatial-strength selection failed development; register a distinct evidence-based hypothesis"
            ),
            r94_result=_relative(result_path), r94_run_id=run_id,
            r94_fit_starts=fit_start_count,
        )
        return result
    except Exception as exc:
        checkpoint = {
            "run_id": run_id, "card": CARD, "fit_starts": fit_start_count,
            "completed_calibrations": fit_details,
            "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        (output / "failure_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": fit_start_count, "error": checkpoint["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r94_failed", next_action="audit the R94 failure checkpoint and ledger before any retry",
            r94_failure=_relative(output / "failure_checkpoint.json"), latest_error=checkpoint["error"],
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="preregister the R94 hypothesis")
    parser.add_argument("--amend-registration", action="store_true",
                        help="record a source-hash-only correction after a verified zero-fit failure")
    args = parser.parse_args()
    if args.register:
        card = register_card()
        print(json.dumps({"card": CARD, "status": card["status"],
                          "expected_calibration_fit_starts": card["expected_calibration_fit_starts"]},
                         ensure_ascii=False), flush=True)
        return
    if args.amend_registration:
        card = amend_registration(
            "fixed the R93 prediction-column alias before any calibration fit; method, data, folds, grid and gates unchanged"
        )
        print(json.dumps({"card": CARD, "status": card["status"],
                          "source_sha256": card["source_sha256"]}, ensure_ascii=False), flush=True)
        return
    result = run()
    for track, score in result["tracks"].items():
        print(
            f"{track}: base={score['base']['mdape_percent']:.5f}% "
            f"R93={score['r93_candidate']['mdape_percent']:.5f}% "
            f"R94={score['candidate']['mdape_percent']:.5f}% "
            f"gain={score['relative_mdape_improvement_percent']:+.3f}% "
            f"MAPE_change={score['mape_relative_change_percent']:+.3f}% "
            f"months={score['months_improved']}/8 "
            f"bootstrap_upper95={score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} "
            f"qualifies={score['qualifies_development']}", flush=True,
        )
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
