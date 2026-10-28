"""R50: hierarchical R22/R32 past-OOF blend by house type and parent-price tercile."""
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
from experiments.bill_target74 import r48_groupwise_past_oof_r22_r32_blend as house_type_source
from experiments.bill_target74 import r49_parent_price_tercile_past_oof_blend as price_source
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy


CARD = "R50_type_price_interaction_past_oof_blend"
SOURCE_CARD = "R47_past_oof_r22_r32_adaptive_blend"
TRACK = "off_market"
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


def register_card() -> dict:
    protocol = campaign.PROTOCOL
    if CARD in protocol["cards"]:
        raise RuntimeError(f"R50 already exists with status {protocol['cards'][CARD].get('status')!r}")
    if any(protocol["limits"].get(key) is not None for key in LIMIT_KEYS):
        raise RuntimeError("R50 requires the active campaign's unbounded limits")
    card = {
        "status": "registered",
        "track": TRACK,
        "parent": "R22 off-market",
        "variant": "R32 comparable-feature candidate",
        "source_card": SOURCE_CARD,
        "difference_from_prior_work": (
            "R48 selected R32 shares by house type and reached 7.06179% development MdAPE; "
            "R49 selected by within-month parent-price tercile and reached 7.09571%. R50 "
            "tests whether those signals interact, with a predeclared backoff hierarchy."
        ),
        "method": (
            "Reuse R47's hash-verified R22/R32 OOF rows and attach house_type_name from the "
            "corresponding safe folds. Within each safe month, assign low/middle/high groups "
            "from that month's R22 parent-prediction 33rd/67th linear quantiles. At each outer "
            "cutoff, select global, house-type, and house-type-by-price-tercile R32 shares from "
            "the fixed 0.05 grid using only rows dated strictly before the cutoff. Require 1000 "
            "global rows and 500 rows for either group level; apply the 1% past-OOF MAPE guard; "
            "tie-break by lower MdAPE, lower MAPE, then smaller R32 share. Use an eligible "
            "interaction weight when supported; otherwise fall back to its house-type weight, "
            "then the cutoff-specific global weight. Current query terciles use current parent "
            "predictions only. Blend in price space and round once. Refit no models."
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
            "bootstrap_seed": 20261029,
        },
        "weight_selection": {
            "grid": [float(x) for x in weight_selection.WEIGHTS],
            "minimum_global_past_oof_rows": int(weight_selection.MINIMUM_OOF_ROWS),
            "minimum_house_type_past_oof_rows": MINIMUM_GROUP_ROWS,
            "minimum_interaction_past_oof_rows": MINIMUM_GROUP_ROWS,
            "past_mape_regression_max_percent": 1.0,
            "interaction_fields": ["house_type_name", "within_month_parent_price_tercile"],
            "price_tercile_definition": "low < q33; middle [q33,q67); high >= q67; linear quantiles",
            "cutoff": "date_sold strictly earlier than outer month start",
            "fallback": "interaction -> house type -> cutoff-specific global weight",
            "tie_break": ["lower MdAPE", "lower MAPE", "smaller R32 share"],
        },
        "fixed_origin_policy": (
            "R33 already scored the registered fixed-origin rows; any R50 re-score is descriptive "
            "only and is not independent fixed-origin confirmation."
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
    campaign._write_progress("r50_registered",
        next_action="Verify the R47 OOF rows and safe house-type metadata for interaction weights")
    print(json.dumps({"card": CARD, "status": "registered", "expected_fit_starts": 0}, indent=2))
    return card


def _load_inputs() -> dict:
    if CARD not in campaign.PROTOCOL["cards"]:
        raise RuntimeError("R50 must be preregistered before scoring candidate predictions")
    if any(campaign.PROTOCOL["limits"].get(key) is not None for key in LIMIT_KEYS):
        raise RuntimeError("active campaign is no longer unbounded")
    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R50 data identity differs from the active campaign")

    # This loader verifies the R47 result, OOF and manifest hashes, then joins types only
    # from the ten already-registered safe folds. It never opens either reserve month.
    loaded = house_type_source._load_inputs()
    oof = loaded["oof"].copy()
    allowed_months = set((*house_type_source.SEED_MONTHS, *DEVELOPMENT_MONTHS))
    if (oof.duplicated(["month", "ml_num"]).any()
            or set(oof.month.astype(str)) != allowed_months
            or set(oof.month.astype(str)).intersection({"2025-05", "2025-08"})):
        raise RuntimeError("R50 OOF rows do not exactly cover the registered safe folds")

    tercile_thresholds = {}
    interaction_support = {}
    for month in sorted(allowed_months):
        mask = oof.month.astype(str).eq(month)
        price_groups, thresholds = price_source._price_groups(
            oof.loc[mask, "parent_prediction"].to_numpy(float))
        oof.loc[mask, "parent_price_tercile"] = price_groups
        oof.loc[mask, "tercile_q33"] = thresholds["q33"]
        oof.loc[mask, "tercile_q67"] = thresholds["q67"]
        tercile_thresholds[str(month)] = thresholds
        pairs = oof.loc[mask].assign(
            parent_price_tercile=price_groups).groupby(
                ["house_type_name", "parent_price_tercile"], dropna=False).size()
        interaction_support[str(month)] = {
            f"{house_type}|{tercile}": int(count)
            for (house_type, tercile), count in pairs.items()
        }

    details = {
        **loaded["preflight"],
        "card": CARD,
        "source_card": SOURCE_CARD,
        "safe_months": sorted(allowed_months),
        "tercile_thresholds_by_month": tercile_thresholds,
        "interaction_support_by_month": interaction_support,
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
        raise RuntimeError("R50 card must be newly registered before stage registration")
    run_id = "bill_target74_r50_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    card.update({"stage_status": "registered", "run_id": run_id,
                 "preflight": details,
                 "registered_at_utc": datetime.now(timezone.utc).isoformat()})
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                      encoding="utf-8")
    campaign._append({
        "event": "card_stage_registered", "card": CARD,
        "stage": "past_oof_type_price_interaction_development_blend", "run_id": run_id,
        "fit_starts_expected": 0, "fit_starts_completed": 0,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r50_registered",
        next_action="Evaluate strict-past R22/R32 interaction weights with hierarchical backoff",
        r50_run_id=run_id)
    print(json.dumps({"run_id": run_id, "registered": True, "expected_fit_starts": 0}, indent=2))
    return card


def _pick_group_weight(frame: pd.DataFrame, cutoff: pd.Timestamp, minimum_rows: int,
                       mape_guard: float, fallback_weight: float,
                       fallback_reason: str) -> dict:
    if len(frame) < minimum_rows:
        return {
            "weight": float(fallback_weight), "reason": fallback_reason,
            "oof_rows": int(len(frame)),
            "oof_months": sorted(frame.month.astype(str).unique().tolist()),
        }
    picked = weight_selection.select_weight(
        frame, cutoff, minimum_rows=minimum_rows,
        max_mape_regression_percent=mape_guard,
    )
    if picked["reason"] != "selected_from_strictly_past_oof":
        return {**picked, "weight": float(fallback_weight),
                "reason": f"{fallback_reason}_{picked['reason']}"}
    return picked


def run() -> dict:
    inputs = _load_inputs()
    card = campaign.PROTOCOL["cards"].get(CARD)
    if (not card or card.get("status") != "registered"
            or card.get("stage_status") != "registered"
            or card.get("preflight") != inputs["preflight"]):
        raise RuntimeError("R50 source artifacts must be registered and reverified before scoring")

    run_id = card["run_id"]
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    source = budget._source_identity()
    (output / "provenance").mkdir()
    snapshots = {}
    code_hashes = {}
    source_paths = (
        Path(__file__), Path(weight_selection.__file__), Path(scoring.__file__),
        Path(campaign.__file__), Path(greedy.__file__),
        Path(house_type_source.__file__), Path(price_source.__file__),
    )
    for index, path in enumerate(source_paths):
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, output / "provenance" / name)
        snapshots[key] = name
        code_hashes[key] = _sha256(path)

    campaign._append({
        "event": "card_start", "card": CARD,
        "stage": "past_oof_type_price_interaction_development_blend", "run_id": run_id,
        "track": TRACK, "expected_fit_starts": 0,
        "development_months": list(DEVELOPMENT_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r50_running",
        next_action="Select strict-past R32 weights with house-type/price-tercile backoff",
        r50_run_id=run_id)

    try:
        oof = inputs["oof"].copy()
        minimum_global = int(card["weight_selection"]["minimum_global_past_oof_rows"])
        minimum_group = int(card["weight_selection"]["minimum_house_type_past_oof_rows"])
        mape_guard = float(card["weight_selection"]["past_mape_regression_max_percent"])
        selections = {}
        paired_parts = []

        for month in DEVELOPMENT_MONTHS:
            cutoff = pd.Period(month, freq="M").start_time
            dates = pd.to_datetime(oof.date_sold, errors="raise")
            past = oof.loc[dates.lt(cutoff)].copy()
            global_pick = weight_selection.select_weight(
                oof, cutoff, minimum_rows=minimum_global,
                max_mape_regression_percent=mape_guard,
            )
            if len(past) != global_pick["oof_rows"] or (
                    len(past) and not pd.to_datetime(past.date_sold, errors="raise").lt(cutoff).all()):
                raise RuntimeError(f"R50 global selector used non-past rows for {month}")
            global_weight = float(global_pick["weight"])

            current = oof.loc[oof.month.astype(str).eq(month)].copy()
            current_groups, thresholds = price_source._price_groups(
                current.parent_prediction.to_numpy(float))
            current["parent_price_tercile"] = current_groups
            current["tercile_q33"] = thresholds["q33"]
            current["tercile_q67"] = thresholds["q67"]
            current["house_type_name"] = current.house_type_name.fillna("__MISSING__").astype(str)

            house_type_picks = {}
            for house_type in sorted(current.house_type_name.unique().tolist()):
                type_past = past.loc[past.house_type_name.astype(str).eq(house_type)].copy()
                house_type_picks[house_type] = _pick_group_weight(
                    type_past, cutoff, minimum_group, mape_guard, global_weight,
                    "fallback_to_global_insufficient_or_ineligible_house_type",
                )

            interaction_picks = {}
            candidate = np.empty(len(current), dtype=float)
            house_types = current.house_type_name.to_numpy(str)
            terciles = current.parent_price_tercile.to_numpy(str)
            for house_type, tercile in sorted(set(zip(house_types, terciles))):
                key = f"{house_type}|{tercile}"
                interaction_past = past.loc[
                    past.house_type_name.astype(str).eq(house_type)
                    & past.parent_price_tercile.astype(str).eq(tercile)
                ].copy()
                type_pick = house_type_picks[house_type]
                interaction_picks[key] = _pick_group_weight(
                    interaction_past, cutoff, minimum_group, mape_guard,
                    float(type_pick["weight"]),
                    "fallback_to_house_type_insufficient_or_ineligible_interaction",
                )
                mask = (house_types == house_type) & (terciles == tercile)
                candidate[mask] = weight_selection.blend(
                    current.loc[mask, "parent_prediction"].to_numpy(float),
                    current.loc[mask, "variant_prediction"].to_numpy(float),
                    float(interaction_picks[key]["weight"]),
                )

            selections[month] = {
                "global": global_pick,
                "house_type": house_type_picks,
                "interaction": interaction_picks,
                "tercile_thresholds": thresholds,
            }
            row_weights = np.array([
                interaction_picks[f"{house_type}|{tercile}"]["weight"]
                for house_type, tercile in zip(house_types, terciles)
            ], dtype=float)
            type_weights = np.array([
                house_type_picks[house_type]["weight"] for house_type in house_types
            ], dtype=float)
            paired_parts.append(pd.DataFrame({
                "month": month, "track": TRACK,
                "ml_num": current.ml_num.astype(str).to_numpy(),
                "date_sold": pd.to_datetime(current.date_sold.to_numpy(), errors="raise"),
                "house_type_name": house_types,
                "parent_price_tercile": terciles,
                "tercile_q33": float(thresholds["q33"]),
                "tercile_q67": float(thresholds["q67"]),
                "sold_price": current.sold_price.to_numpy(float),
                "parent_prediction": current.parent_prediction.to_numpy(float),
                "variant_prediction": current.variant_prediction.to_numpy(float),
                "global_r32_share": global_weight,
                "house_type_r32_share": type_weights,
                "selected_r32_share": row_weights,
                "candidate_prediction": candidate,
            }))

        paired = pd.concat(paired_parts, ignore_index=True)
        if (paired.duplicated(["month", "ml_num"]).any()
                or set(paired.month.astype(str)) != set(DEVELOPMENT_MONTHS)):
            raise RuntimeError("R50 paired rows do not cover exactly the registered development folds")
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
            "fit_starts": 0, "selected_r32_share_by_month": selections,
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
            "# R50：R22/R32 按房型×预测价位分组的严格过去 OOF 融合", "",
            f"Run：`{run_id}`；新增 learner fits：0。",
            "权重按 global → house type → house type×within-month parent-price tercile 三级回退。当前月分组只依赖 R22 预测价，不读取当前月标签；所有权重只使用严格早于预测月的 OOF 行。",
            f"Pooled MdAPE：{metrics['parent']['mdape_percent']:.5f}% → "
            f"{metrics['candidate']['mdape_percent']:.5f}%；相对改善 "
            f"{metrics['relative_mdape_improvement_percent']:+.3f}%；MAPE 相对变化 "
            f"{metrics['mape_relative_change_percent']:+.3f}%；"
            f"{metrics['months_improved']}/8 月改善；最差月退化 "
            f"{metrics['worst_month_regression_percent']:.3f}%；bootstrap 上界 "
            f"{metrics['bootstrap_mdape_difference_percent']['upper95']:+.5f} pp；"
            f"目标 <7%：{metrics['target_met']}；通过：{qualifies}。", "",
            "## 逐月结果与选中权重", "",
            "| 月份 | 全局权重 | 最终交互组权重范围 | R22 MdAPE | 候选 MdAPE | 相对改善 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for month in DEVELOPMENT_MONTHS:
            monthly = metrics["monthly"][month]
            interaction_weights = [row["weight"] for row in selections[month]["interaction"].values()]
            report.append(
                f"| {month} | {selections[month]['global']['weight']:.2f} | "
                f"{min(interaction_weights):.2f}–{max(interaction_weights):.2f} | "
                f"{monthly['parent']['mdape_percent']:.5f}% | "
                f"{monthly['candidate']['mdape_percent']:.5f}% | "
                f"{monthly['relative_mdape_improvement_percent']:+.3f}% |"
            )
        report.extend([
            "", "2025-05/08 未读取且保留验证仍不具备独立资格；隐藏标签未读取。固定起点行曾被 R33 使用，本轮不构成独立固定起点确认。", "",
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
            "stage": "past_oof_type_price_interaction_development_blend", "run_id": run_id,
            "fit_starts": 0, "result_path": current["development_result_path"],
            "qualifies_by_track": {TRACK: qualifies},
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        state = "r50_development_passed" if qualifies else "r50_development_failed"
        next_action = (
            "R50 passed development gate; preserve result and continue list-aware improvement"
            if qualifies else
            "R50 failed; keep R22/R35 and register a distinct off-market or list-aware hypothesis"
        )
        campaign._write_progress(state, next_action=next_action,
            r50_result=current["development_result_path"], r50_fit_starts=0)
        print(
            f"R50: parent_mdape={metrics['parent']['mdape_percent']:.5f}% "
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
            "stage": "past_oof_type_price_interaction_development_blend", "run_id": run_id,
            "fit_starts": 0, "error": current["failure"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress("r50_failed",
            next_action="Inspect R50 failure before any retry", r50_failure=current["failure"])
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
