"""R49: past-only R22/R32 weights by within-month parent-price tercile."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import r22_remarks_tfidf as scoring
from experiments.bill_target74 import r40_past_oof_r26_blend as weight_selection
from experiments.bill_target74 import runner as campaign


CARD = "R49_parent_price_tercile_past_oof_blend"
SOURCE_CARD = "R47_past_oof_r22_r32_adaptive_blend"
TRACK = "off_market"
SEED_MONTHS = ("2022-06", "2022-11")
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
GROUPS = ("low", "middle", "high")
MINIMUM_GROUP_ROWS = 500
LIMIT_KEYS = (
    "campaign_fit_starts", "campaign_minutes", "exploration_fit_starts",
    "exploration_minutes", "final_reserve_fit_starts", "final_reserve_minutes",
    "batch_fit_starts", "batch_minutes", "single_fit_seconds",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _price_groups(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("parent predictions for price terciles must be finite positive values")
    q33, q67 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    groups = np.where(values < q33, "low", np.where(values < q67, "middle", "high"))
    return groups, {"q33": float(q33), "q67": float(q67)}


def register_card() -> dict:
    protocol = campaign.PROTOCOL
    if CARD in protocol["cards"]:
        raise RuntimeError(f"R49 already exists with status {protocol['cards'][CARD].get('status')!r}")
    if any(protocol["limits"].get(key) is not None for key in LIMIT_KEYS):
        raise RuntimeError("R49 requires the active campaign's unbounded limits")
    card = {
        "status": "registered", "track": TRACK,
        "parent": "R22 off-market", "variant": "R32 comparable-feature candidate",
        "source_card": SOURCE_CARD,
        "difference_from_prior_work": (
            "R47 selected one global R32 share per cutoff; R48 selected house-type shares and "
            "improved MdAPE to 7.06179% but missed the strict <7% target. R49 uses within-month "
            "R22 predicted-price rank terciles, a different predeclared grouping."
        ),
        "method": (
            "Reuse R47's hash-verified R22/R32 OOF predictions. In each OOF month, group rows "
            "into low/middle/high using that month's R22 parent-prediction 33rd and 67th "
            "percentiles (linear quantiles); assign current-month rows using current query "
            "predictions only. At each outer cutoff, choose global and tercile-specific R32 "
            "shares on the fixed 0.05 grid using only OOF rows dated strictly earlier than the "
            "cutoff. Require 1000 global or 500 group rows, enforce a 1% MAPE guard, and tie-break "
            "by MdAPE, MAPE, then smaller share. Unsupported groups or groups with no eligible "
            "weight use that cutoff's global share. Blend in price space and round once."
        ),
        "expected_fit_starts": 0,
        "development_confirmation": {
            "development_months": list(DEVELOPMENT_MONTHS),
            "target_mdape_strictly_below": 7.0,
            "relative_mdape_improvement_percent_min": 0.5,
            "months_improved_min": 6,
            "worst_month_regression_max_percent": 5.0,
            "mape_regression_max_percent": 1.0,
            "paired_bootstrap_upper95_below_zero": True,
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 20261028,
        },
        "weight_selection": {
            "grid": [float(x) for x in weight_selection.WEIGHTS],
            "minimum_global_past_oof_rows": int(weight_selection.MINIMUM_OOF_ROWS),
            "minimum_group_past_oof_rows": MINIMUM_GROUP_ROWS,
            "past_mape_regression_max_percent": 1.0,
            "group_field": "parent_prediction within-month linear tercile; low < q33, middle [q33,q67), high >= q67",
            "cutoff": "date_sold strictly earlier than outer month start",
            "fallback": "cutoff-specific global R32 share",
            "tie_break": ["lower MdAPE", "lower MAPE", "smaller R32 share"],
        },
        "fixed_origin_policy": (
            "R33 already scored the registered fixed-origin rows; any R49 re-score is descriptive only, "
            "not independent fixed-origin confirmation."
        ),
        "reserve_validation_eligible": False,
        "policy": "Do not read or use 2025-05/08 or hidden labels.",
    }
    protocol["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    campaign._append({
        "event": "card_registered", "card": CARD, "track": TRACK,
        "expected_fit_starts": 0, "development_months": list(DEVELOPMENT_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r49_registered",
        next_action="Verify the hash-locked R47 OOF inputs for price-tercile selection")
    print(json.dumps({"card": CARD, "status": "registered", "expected_fit_starts": 0}, indent=2))
    return card


def _load_inputs() -> dict:
    if CARD not in campaign.PROTOCOL["cards"]:
        raise RuntimeError("R49 must be preregistered before reading candidate OOF predictions")
    if any(campaign.PROTOCOL["limits"].get(key) is not None for key in LIMIT_KEYS):
        raise RuntimeError("active campaign is no longer unbounded")
    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R49 dataset identity differs from the active campaign")
    source_card = campaign.PROTOCOL["cards"].get(SOURCE_CARD, {})
    result_path = campaign.ROOT / source_card.get("development_result_path", "")
    if not result_path.is_file():
        raise FileNotFoundError("R47 development result is missing")
    folder = result_path.parent
    manifest_path = folder / "run_manifest.json"
    oof_path = folder / "past_oof_predictions.csv"
    paired_path = folder / "paired_predictions.csv"
    if not all(path.is_file() for path in (manifest_path, oof_path, paired_path)):
        raise FileNotFoundError("R47 result, OOF predictions, or paired rows are incomplete")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("card") != SOURCE_CARD
            or result.get("run_id") != source_card.get("run_id")
            or result.get("data_sha256") != source["data_sha256"]
            or result.get("reserved_months_labels_opened")
            or result.get("hidden_labels_opened")
            or manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("seed_oof_sha256") != _sha256(oof_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("source_identity", {}).get("data_sha256") != source["data_sha256"]):
        raise RuntimeError("R47 result or hash-verified OOF inputs failed validation")
    for key, snapshot in manifest.get("source_snapshots", {}).items():
        source_path = folder / "provenance" / snapshot
        if not source_path.is_file() or manifest.get("code_sha256", {}).get(key) != _sha256(source_path):
            raise RuntimeError(f"R47 source snapshot failed verification: {key}")
    oof = pd.read_csv(oof_path, dtype={"ml_num": str})
    allowed_months = set((*SEED_MONTHS, *DEVELOPMENT_MONTHS))
    if (oof.duplicated(["month", "ml_num"]).any()
            or set(oof.month.astype(str)) != allowed_months
            or set(oof.month.astype(str)).intersection({"2025-05", "2025-08"})):
        raise RuntimeError("R47 OOF rows do not exactly cover registered safe folds")
    for month in allowed_months:
        mask = oof.month.astype(str).eq(month)
        predictions = oof.loc[mask, "parent_prediction"].to_numpy(float)
        groups, thresholds = _price_groups(predictions)
        oof.loc[mask, "parent_price_tercile"] = groups
        oof.loc[mask, "tercile_q33"] = thresholds["q33"]
        oof.loc[mask, "tercile_q67"] = thresholds["q67"]
    details = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "card": CARD,
        "source_card": SOURCE_CARD, "source_run_id": result["run_id"],
        "source_result_sha256": _sha256(result_path),
        "source_oof_sha256": _sha256(oof_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "data_sha256": source["data_sha256"],
        "safe_months": sorted(allowed_months),
        "rows_by_month": {str(month): int((oof.month.astype(str) == month).sum())
                           for month in sorted(allowed_months)},
        "tercile_thresholds_by_month": {
            str(month): {"q33": float(oof.loc[oof.month.astype(str).eq(month), "tercile_q33"].iloc[0]),
                         "q67": float(oof.loc[oof.month.astype(str).eq(month), "tercile_q67"].iloc[0])}
            for month in sorted(allowed_months)
        },
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    return {"preflight": details, "oof": oof}


def preflight() -> dict:
    details = _load_inputs()["preflight"]
    print(json.dumps(details, indent=2, sort_keys=True))
    return details


def register_stage() -> dict:
    details = _load_inputs()["preflight"]
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered" or card.get("stage_status"):
        raise RuntimeError("R49 card must be newly registered before stage registration")
    run_id = "bill_target74_r49_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    card.update({"stage_status": "registered", "run_id": run_id,
                 "preflight": details,
                 "registered_at_utc": datetime.now(timezone.utc).isoformat()})
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                      encoding="utf-8")
    campaign._append({
        "event": "card_stage_registered", "card": CARD,
        "stage": "past_oof_price_tercile_development_blend", "run_id": run_id,
        "fit_starts_expected": 0, "fit_starts_completed": 0,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r49_registered",
        next_action="Evaluate the registered price-tercile past-OOF R22/R32 blend",
        r49_run_id=run_id)
    print(json.dumps({"run_id": run_id, "registered": True, "expected_fit_starts": 0}, indent=2))
    return card


def run() -> dict:
    inputs = _load_inputs()
    card = campaign.PROTOCOL["cards"].get(CARD)
    if (not card or card.get("status") != "registered"
            or card.get("stage_status") != "registered"
            or card.get("preflight") != inputs["preflight"]):
        raise RuntimeError("R49 source artifacts must be registered and reverified before scoring")
    run_id = card["run_id"]
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    source = budget._source_identity()
    (output / "provenance").mkdir()
    snapshots = {}
    code_hashes = {}
    for index, path in enumerate((Path(__file__), Path(weight_selection.__file__),
                                  Path(scoring.__file__), Path(campaign.__file__))):
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, output / "provenance" / name)
        snapshots[key] = name
        code_hashes[key] = _sha256(path)
    campaign._append({
        "event": "card_start", "card": CARD,
        "stage": "past_oof_price_tercile_development_blend", "run_id": run_id,
        "track": TRACK, "expected_fit_starts": 0,
        "development_months": list(DEVELOPMENT_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r49_running",
        next_action="Evaluate price-tercile R32 weights using strict-past OOF rows",
        r49_run_id=run_id)
    try:
        oof = inputs["oof"].copy()
        selections = {}
        paired_parts = []
        for month in DEVELOPMENT_MONTHS:
            cutoff = pd.Period(month, freq="M").start_time
            dates = pd.to_datetime(oof.date_sold, errors="raise")
            past = oof.loc[dates.lt(cutoff)].copy()
            global_pick = weight_selection.select_weight(
                oof, cutoff,
                minimum_rows=int(card["weight_selection"]["minimum_global_past_oof_rows"]),
                max_mape_regression_percent=float(card["weight_selection"]["past_mape_regression_max_percent"]),
            )
            if len(past) != global_pick["oof_rows"] or (
                len(past) and not pd.to_datetime(past.date_sold, errors="raise").lt(cutoff).all()
            ):
                raise RuntimeError(f"R49 global weight for {month} used non-past OOF rows")
            global_weight = float(global_pick["weight"])
            current = oof.loc[oof.month.astype(str).eq(month)].copy()
            current_groups, thresholds = _price_groups(current.parent_prediction.to_numpy(float))
            current["parent_price_tercile"] = current_groups
            current["tercile_q33"] = thresholds["q33"]
            current["tercile_q67"] = thresholds["q67"]
            group_selections = {}
            candidate = np.empty(len(current), dtype=float)
            for group in GROUPS:
                group_past = past.loc[past.parent_price_tercile.astype(str).eq(group)].copy()
                if len(group_past) < int(card["weight_selection"]["minimum_group_past_oof_rows"]):
                    picked = {"weight": global_weight,
                              "reason": "fallback_to_global_insufficient_group_rows",
                              "oof_rows": int(len(group_past)),
                              "oof_months": sorted(group_past.month.astype(str).unique().tolist())}
                else:
                    picked = weight_selection.select_weight(
                        group_past, cutoff,
                        minimum_rows=int(card["weight_selection"]["minimum_group_past_oof_rows"]),
                        max_mape_regression_percent=float(card["weight_selection"]["past_mape_regression_max_percent"]),
                    )
                    if picked["reason"] != "selected_from_strictly_past_oof":
                        picked = {**picked, "weight": global_weight,
                                  "reason": f"fallback_to_global_{picked['reason']}"}
                group_selections[group] = picked
                mask = current.parent_price_tercile.eq(group).to_numpy()
                candidate[mask] = weight_selection.blend(
                    current.loc[mask, "parent_prediction"].to_numpy(float),
                    current.loc[mask, "variant_prediction"].to_numpy(float),
                    float(picked["weight"]),
                )
            selections[month] = {"global": global_pick, "tercile_thresholds": thresholds,
                                 "groups": group_selections}
            paired_parts.append(pd.DataFrame({
                "month": month, "track": TRACK, "ml_num": current.ml_num.astype(str).to_numpy(),
                "date_sold": pd.to_datetime(current.date_sold.to_numpy(), errors="raise"),
                "parent_price_tercile": current.parent_price_tercile.to_numpy(str),
                "tercile_q33": float(thresholds["q33"]), "tercile_q67": float(thresholds["q67"]),
                "sold_price": current.sold_price.to_numpy(float),
                "parent_prediction": current.parent_prediction.to_numpy(float),
                "variant_prediction": current.variant_prediction.to_numpy(float),
                "global_r32_share": global_weight,
                "selected_r32_share": np.array([
                    group_selections[g]["weight"] for g in current.parent_price_tercile.astype(str)
                ], dtype=float),
                "candidate_prediction": candidate,
            }))
        paired = pd.concat(paired_parts, ignore_index=True)
        if paired.duplicated(["month", "ml_num"]).any() or set(paired.month.astype(str)) != set(DEVELOPMENT_MONTHS):
            raise RuntimeError("R49 paired rows do not cover exactly the registered development window")
        paired.to_csv(output / "paired_predictions.csv", index=False)
        metrics = scoring.score_development_confirmation(
            paired,
            months=DEVELOPMENT_MONTHS,
            target_mdape_strictly_below=float(card["development_confirmation"]["target_mdape_strictly_below"]),
            bootstrap_replicates=int(card["development_confirmation"]["bootstrap_replicates"]),
            bootstrap_seed=int(card["development_confirmation"]["bootstrap_seed"]),
            minimum_relative_gain_percent=float(card["development_confirmation"]["relative_mdape_improvement_percent_min"]),
            minimum_months_improved=int(card["development_confirmation"]["months_improved_min"]),
            maximum_worst_month_regression_percent=float(card["development_confirmation"]["worst_month_regression_max_percent"]),
            maximum_mape_regression_percent=float(card["development_confirmation"]["mape_regression_max_percent"]),
        )
        qualifies = bool(metrics["qualifies_development"] and metrics["target_met"])
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "card": CARD,
            "stage": "development_confirmation", "run_id": run_id,
            "track": TRACK, "source_card": SOURCE_CARD,
            "fit_starts": 0, "selected_r32_share_by_month_and_tercile": selections,
            "metrics": metrics, "qualifies_development": qualifies,
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        result_path = output / "development_results.json"
        _write_json(result_path, result)
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": source, "source_snapshots": snapshots,
            "code_sha256": code_hashes, "preflight": card["preflight"],
            "source_oof_sha256": card["preflight"]["source_oof_sha256"],
            "paired_predictions_sha256": _sha256(output / "paired_predictions.csv"),
            "result_sha256": _sha256(result_path), "fit_starts": 0,
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        _write_json(output / "run_manifest.json", manifest)
        report = [
            "# R49：R22/R32 按预测价位三分位的严格过去 OOF 融合", "",
            f"Run：`{run_id}`；新增 learner fits：0。",
            "组别由每个 OOF 月内部 R22 父模型预测价的 33rd/67th 百分位确定；当前月只用当前预测分布生成组别，不读当前标签。过去组内少于 500 行或没有合格权重时回退到同一截止日的全局权重。",
            f"Pooled MdAPE：{metrics['parent']['mdape_percent']:.5f}% → "
            f"{metrics['candidate']['mdape_percent']:.5f}%；相对改善 "
            f"{metrics['relative_mdape_improvement_percent']:+.3f}%；MAPE 相对变化 "
            f"{metrics['mape_relative_change_percent']:+.3f}%；"
            f"{metrics['months_improved']}/8 月改善；bootstrap 上界 "
            f"{metrics['bootstrap_mdape_difference_percent']['upper95']:+.5f} 个百分点；"
            f"目标 <7%：{metrics['target_met']}；通过：{qualifies}。",
            "", "## 逐月权重与结果", "",
            "| 月份 | 全局 share | 低/中/高 share | R22 MdAPE | 候选 MdAPE | 相对改善 |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for month in DEVELOPMENT_MONTHS:
            row = metrics["monthly"][month]
            sel = selections[month]
            shares = "/".join(f"{sel['groups'][g]['weight']:.2f}" for g in GROUPS)
            report.append(
                f"| {month} | {sel['global']['weight']:.2f} | {shares} | "
                f"{row['parent']['mdape_percent']:.5f}% | {row['candidate']['mdape_percent']:.5f}% | "
                f"{row['relative_mdape_improvement_percent']:+.3f}% |"
            )
        report.extend([
            "", "2025-05/08 未读取且保留验证不具备独立资格；隐藏标签未读取。R33 已使用固定起点行，因此本轮不构成独立固定起点确认。", "",
        ])
        (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")
        current = campaign.PROTOCOL["cards"][CARD]
        current.update({
            "status": "development_passed" if qualifies else "development_failed",
            "stage_status": "complete", "fit_starts": 0,
            "development_result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "development_metrics": metrics,
            "qualifies_development": qualifies,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                          encoding="utf-8")
        campaign._append({
            "event": "card_complete", "card": CARD,
            "stage": "past_oof_price_tercile_development_blend", "run_id": run_id,
            "fit_starts": 0, "result_path": current["development_result_path"],
            "qualifies_by_track": {TRACK: qualifies},
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        state = "r49_development_passed" if qualifies else "r49_development_failed"
        next_action = (
            "R49 passed development gate; preserve as development candidate and continue list-aware improvement"
            if qualifies else
            "R49 failed; preserve R22 and register a distinct off-market or list-aware hypothesis"
        )
        campaign._write_progress(state, next_action=next_action,
            r49_result=current["development_result_path"], r49_fit_starts=0)
        print(
            f"R49: parent_mdape={metrics['parent']['mdape_percent']:.5f}% "
            f"candidate_mdape={metrics['candidate']['mdape_percent']:.5f}% "
            f"relative_gain={metrics['relative_mdape_improvement_percent']:+.3f}% "
            f"months_improved={metrics['months_improved']}/8 "
            f"bootstrap_upper95={metrics['bootstrap_mdape_difference_percent']['upper95']:+.5f} "
            f"target_met={metrics['target_met']} pass={qualifies} run={run_id}",
            flush=True,
        )
        return result
    except Exception as exc:
        current = campaign.PROTOCOL["cards"][CARD]
        current.update({"stage_status": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                          encoding="utf-8")
        campaign._append({
            "event": "card_failed", "card": CARD,
            "stage": "past_oof_price_tercile_development_blend", "run_id": run_id,
            "fit_starts": 0, "error": current["failure"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress("r49_failed",
            next_action="Inspect R49 failure before any retry", r49_failure=current["failure"])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register-card", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--register-stage", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.register_card:
        register_card()
    elif args.preflight:
        preflight()
    elif args.register_stage:
        register_stage()
    else:
        run()
