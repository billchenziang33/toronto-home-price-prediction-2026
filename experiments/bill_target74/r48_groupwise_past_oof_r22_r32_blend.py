"""R48: house-type-specific past-only weights for the R22/R32 blend."""
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
from experiments.bill_greedy import runner as greedy


CARD = "R48_groupwise_past_oof_r22_r32_blend"
SOURCE_CARD = "R47_past_oof_r22_r32_adaptive_blend"
TRACK = "off_market"
SEED_MONTHS = ("2022-06", "2022-11")
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
MINIMUM_GROUP_OOF_ROWS = 500
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
        raise RuntimeError(f"R48 already exists with status {protocol['cards'][CARD].get('status')!r}")
    if any(protocol["limits"].get(key) is not None for key in LIMIT_KEYS):
        raise RuntimeError("R48 requires the active campaign's unbounded limits")
    card = {
        "status": "registered",
        "track": TRACK,
        "parent": "R22 off-market",
        "variant": "R32 comparable-feature candidate",
        "source_card": SOURCE_CARD,
        "difference_from_prior_work": (
            "R41 selected house-type weights for the distinct R26 text-width candidate and failed. "
            "R47 found a promising but insufficient global R32 blend. R48 tests whether R32 "
            "complementarity differs by house_type_name, with predeclared minimum group support."
        ),
        "method": (
            "Reuse R47's hash-verified strict OOF R22/R32 predictions; attach house_type_name "
            "from the corresponding safe fold. At each outer cutoff, choose a global R32 share "
            "and separate per-house-type shares on the fixed 0.05 grid using only rows with "
            "date_sold strictly before the cutoff. Require 1000 global or 500 group OOF rows; "
            "guard MAPE regression at 1%; ties favor lower MdAPE, then lower MAPE, then smaller "
            "R32 share. Unsupported groups or groups with no eligible share use that cutoff's "
            "global share. Blend in price space and round once. No models are refit."
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
            "bootstrap_seed": 20261027,
        },
        "weight_selection": {
            "grid": [float(x) for x in weight_selection.WEIGHTS],
            "minimum_global_past_oof_rows": int(weight_selection.MINIMUM_OOF_ROWS),
            "minimum_group_past_oof_rows": MINIMUM_GROUP_OOF_ROWS,
            "past_mape_regression_max_percent": 1.0,
            "group_field": "house_type_name; missing values map to __MISSING__",
            "cutoff": "date_sold strictly earlier than the outer month start",
            "fallback": "cutoff-specific global R32 share",
            "tie_break": ["lower MdAPE", "lower MAPE", "smaller R32 share"],
        },
        "fixed_origin_policy": (
            "R33 already scored the registered fixed-origin rows. Any R48 rescore is descriptive only, "
            "not independent fixed-origin confirmation."
        ),
        "reserve_validation_eligible": False,
        "policy": "Do not read or use 2025-05/08 or hidden labels.",
    }
    protocol["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    campaign._append({
        "event": "card_registered", "card": CARD, "track": TRACK,
        "expected_fit_starts": 0,
        "development_months": list(DEVELOPMENT_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r48_registered",
        next_action="Verify R47 OOF artifacts and safe house-type metadata before groupwise selection")
    print(json.dumps({"card": CARD, "status": "registered", "expected_fit_starts": 0}, indent=2))
    return card


def _load_inputs() -> dict:
    if CARD not in campaign.PROTOCOL["cards"]:
        raise RuntimeError("R48 must be preregistered before reading R47 OOF predictions")
    if any(campaign.PROTOCOL["limits"].get(key) is not None for key in LIMIT_KEYS):
        raise RuntimeError("active campaign is no longer unbounded")
    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R48 data identity differs from the campaign")
    source_card = campaign.PROTOCOL["cards"].get(SOURCE_CARD, {})
    result_path = campaign.ROOT / source_card.get("development_result_path", "")
    if not result_path.is_file():
        raise FileNotFoundError("R47 development result is missing")
    folder = result_path.parent
    manifest_path = folder / "run_manifest.json"
    oof_path = folder / "past_oof_predictions.csv"
    rows_path = folder / "paired_predictions.csv"
    if not all(path.is_file() for path in (manifest_path, oof_path, rows_path)):
        raise FileNotFoundError("R47 development artifacts are incomplete")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("card") != SOURCE_CARD
            or result.get("run_id") != source_card.get("run_id")
            or result.get("data_sha256") != source["data_sha256"]
            or result.get("reserved_months_labels_opened")
            or result.get("hidden_labels_opened")
            or manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("seed_oof_sha256") != _sha256(oof_path)
            or manifest.get("paired_predictions_sha256") != _sha256(rows_path)
            or manifest.get("source_identity", {}).get("data_sha256") != source["data_sha256"]):
        raise RuntimeError("R47 result, OOF hashes, or label-safety metadata failed verification")
    for source_key, snapshot in manifest.get("source_snapshots", {}).items():
        snapshot_path = folder / "provenance" / snapshot
        if (not snapshot_path.is_file()
                or manifest.get("code_sha256", {}).get(source_key) != _sha256(snapshot_path)):
            raise RuntimeError(f"R47 source snapshot failed verification: {source_key}")
    oof = pd.read_csv(oof_path, dtype={"ml_num": str})
    expected_months = set((*SEED_MONTHS, *DEVELOPMENT_MONTHS))
    if (oof.duplicated(["month", "ml_num"]).any()
            or set(oof.month.astype(str)) != expected_months
            or set(oof.month.astype(str)).intersection({"2025-05", "2025-08"})):
        raise RuntimeError("R47 OOF rows do not exactly cover registered safe months")

    from data import hide_answers
    typed_parts = []
    support = {}
    for month in (*SEED_MONTHS, *DEVELOPMENT_MONTHS):
        _, _, _, truth, heldout = greedy._load_fold(month)
        homes = hide_answers(heldout, TRACK).reset_index(drop=True)
        ids = homes.ml_num.astype(str).to_numpy()
        rows = oof.loc[oof.month.astype(str).eq(month)].copy()
        if rows.ml_num.astype(str).duplicated().any() or set(rows.ml_num.astype(str)) != set(ids):
            raise RuntimeError(f"R47 OOF IDs do not align with safe {month} fold")
        rows["ml_num"] = rows.ml_num.astype(str)
        rows = rows.set_index("ml_num").loc[ids].reset_index()
        if (not np.array_equal(rows.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
                or not np.array_equal(rows.date_sold.astype(str).to_numpy(), heldout.date_sold.astype(str).to_numpy())):
            raise RuntimeError(f"R47 labels or sale dates do not align with safe {month} fold")
        rows["house_type_name"] = homes.house_type_name.fillna("__MISSING__").astype(str).to_numpy()
        typed_parts.append(rows)
        support[month] = {
            "rows": int(len(rows)),
            "house_type_counts": {str(k): int(v) for k, v in rows.house_type_name.value_counts(dropna=False).items()},
        }
    typed = pd.concat(typed_parts, ignore_index=True)
    details = {
        "campaign_id": campaign.PROTOCOL["campaign_id"],
        "card": CARD, "source_card": SOURCE_CARD,
        "source_run_id": result["run_id"],
        "source_result_sha256": _sha256(result_path),
        "source_oof_sha256": _sha256(oof_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "data_sha256": source["data_sha256"],
        "safe_months": list((*SEED_MONTHS, *DEVELOPMENT_MONTHS)),
        "support": support,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    return {"preflight": details, "oof": typed}


def preflight() -> dict:
    details = _load_inputs()["preflight"]
    print(json.dumps(details, indent=2, sort_keys=True))
    return details


def register_stage() -> dict:
    details = _load_inputs()["preflight"]
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered" or card.get("stage_status"):
        raise RuntimeError("R48 card must be newly registered before stage registration")
    run_id = "bill_target74_r48_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    card.update({"stage_status": "registered", "run_id": run_id,
                 "preflight": details,
                 "registered_at_utc": datetime.now(timezone.utc).isoformat()})
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                      encoding="utf-8")
    campaign._append({
        "event": "card_stage_registered", "card": CARD,
        "stage": "past_oof_groupwise_development_blend", "run_id": run_id,
        "fit_starts_expected": 0, "fit_starts_completed": 0,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r48_registered",
        next_action="Run the registered groupwise past-OOF R22/R32 development blend",
        r48_run_id=run_id)
    print(json.dumps({"run_id": run_id, "registered": True, "expected_fit_starts": 0}, indent=2))
    return card


def run() -> dict:
    inputs = _load_inputs()
    card = campaign.PROTOCOL["cards"].get(CARD)
    if (not card or card.get("status") != "registered"
            or card.get("stage_status") != "registered"
            or card.get("preflight") != inputs["preflight"]):
        raise RuntimeError("R48 must be preregistered and its source hashes reverified before scoring")
    run_id = card["run_id"]
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    source = budget._source_identity()
    (output / "provenance").mkdir()
    snapshots = {}
    code_hashes = {}
    for i, path in enumerate((Path(__file__), Path(weight_selection.__file__),
                              Path(scoring.__file__), Path(campaign.__file__), Path(greedy.__file__))):
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        name = f"src_{i:02d}{path.suffix}.source"
        shutil.copy2(path, output / "provenance" / name)
        snapshots[key] = name
        code_hashes[key] = _sha256(path)

    campaign._append({
        "event": "card_start", "card": CARD,
        "stage": "past_oof_groupwise_development_blend", "run_id": run_id,
        "track": TRACK, "expected_fit_starts": 0,
        "development_months": list(DEVELOPMENT_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r48_running",
        next_action="Evaluate groupwise R32 shares using strict-past OOF rows",
        r48_run_id=run_id)
    try:
        oof = inputs["oof"].copy()
        selections = {}
        paired_rows = []
        for month in DEVELOPMENT_MONTHS:
            cutoff = pd.Period(month, freq="M").start_time
            past_mask = pd.to_datetime(oof.date_sold, errors="raise").lt(cutoff)
            past = oof.loc[past_mask].copy()
            global_pick = weight_selection.select_weight(
                oof, cutoff,
                minimum_rows=int(card["weight_selection"]["minimum_global_past_oof_rows"]),
                max_mape_regression_percent=float(card["weight_selection"]["past_mape_regression_max_percent"]),
            )
            if len(past) != global_pick["oof_rows"]:
                raise RuntimeError(f"R48 global selector did not use all and only past rows for {month}")
            global_weight = float(global_pick["weight"])
            current = oof.loc[oof.month.astype(str).eq(month)].copy()
            group_weights = {}
            candidate = np.empty(len(current), dtype=float)
            groups = current.house_type_name.fillna("__MISSING__").astype(str).to_numpy()
            for group in sorted(set(groups)):
                group_past = past.loc[past.house_type_name.astype(str).eq(group)].copy()
                if len(group_past) < int(card["weight_selection"]["minimum_group_past_oof_rows"]):
                    picked = {
                        "weight": global_weight,
                        "reason": "fallback_to_global_insufficient_group_rows",
                        "oof_rows": int(len(group_past)),
                        "oof_months": sorted(group_past.month.astype(str).unique().tolist()),
                    }
                else:
                    picked = weight_selection.select_weight(
                        group_past, cutoff,
                        minimum_rows=int(card["weight_selection"]["minimum_group_past_oof_rows"]),
                        max_mape_regression_percent=float(card["weight_selection"]["past_mape_regression_max_percent"]),
                    )
                    if picked["reason"] != "selected_from_strictly_past_oof":
                        picked = {**picked, "weight": global_weight,
                                  "reason": f"fallback_to_global_{picked['reason']}"}
                group_weights[group] = picked
                mask = groups == group
                candidate[mask] = weight_selection.blend(
                    current.loc[mask, "parent_prediction"].to_numpy(float),
                    current.loc[mask, "variant_prediction"].to_numpy(float),
                    float(picked["weight"]),
                )
            selections[month] = {"global": global_pick, "groups": group_weights}
            paired_rows.append(pd.DataFrame({
                "month": month, "track": TRACK,
                "ml_num": current.ml_num.astype(str).to_numpy(),
                "date_sold": pd.to_datetime(current.date_sold.to_numpy(), errors="raise"),
                "house_type_name": groups,
                "sold_price": current.sold_price.to_numpy(float),
                "parent_prediction": current.parent_prediction.to_numpy(float),
                "variant_prediction": current.variant_prediction.to_numpy(float),
                "global_r32_share": global_weight,
                "selected_r32_share": np.array([group_weights[g]["weight"] for g in groups], dtype=float),
                "candidate_prediction": candidate,
            }))
        paired = pd.concat(paired_rows, ignore_index=True)
        if (paired.duplicated(["month", "ml_num"]).any()
                or set(paired.month.astype(str)) != set(DEVELOPMENT_MONTHS)):
            raise RuntimeError("R48 paired rows do not exactly cover the registered eight months")
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
            "fit_starts": 0, "selected_r32_share_by_month_and_group": selections,
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
            "result_sha256": _sha256(result_path),
            "fit_starts": 0,
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        _write_json(output / "run_manifest.json", manifest)
        report = [
            "# R48：R22/R32 按房型的严格过去 OOF 融合", "",
            f"Run：`{run_id}`；新增 learner fits：0。",
            "每个外层月份按 house_type_name 分别选择 R32 share；组内过去 OOF 少于 500 行或无合格权重时，回退该截止日的全局 past-OOF 权重。筛选没有读取当前月或未来月的标签。",
            f"Pooled MdAPE：{metrics['parent']['mdape_percent']:.5f}% → "
            f"{metrics['candidate']['mdape_percent']:.5f}%；相对改善 "
            f"{metrics['relative_mdape_improvement_percent']:+.3f}%；MAPE 相对变化 "
            f"{metrics['mape_relative_change_percent']:+.3f}%；"
            f"{metrics['months_improved']}/8 月改善；bootstrap 上界 "
            f"{metrics['bootstrap_mdape_difference_percent']['upper95']:+.5f} 个百分点；"
            f"目标 <7%：{metrics['target_met']}；通过：{qualifies}。",
            "", "## 逐月结果", "",
            "| 月份 | 全局 share | 实际组别 share | R22 MdAPE | 候选 MdAPE | 相对改善 |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for month in DEVELOPMENT_MONTHS:
            row = metrics["monthly"][month]
            month_selection = selections[month]
            shares = ", ".join(f"{g}:{v['weight']:.2f}" for g, v in month_selection["groups"].items())
            report.append(
                f"| {month} | {month_selection['global']['weight']:.2f} | {shares} | "
                f"{row['parent']['mdape_percent']:.5f}% | {row['candidate']['mdape_percent']:.5f}% | "
                f"{row['relative_mdape_improvement_percent']:+.3f}% |"
            )
        report.extend([
            "", "2025-05/08 未读取且保留验证不具备独立资格；隐藏标签未读取。R33 已使用固定起点行，故此轮仅为开发集结果，不报告为独立固定起点确认。", "",
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
            "stage": "past_oof_groupwise_development_blend", "run_id": run_id,
            "fit_starts": 0, "result_path": current["development_result_path"],
            "qualifies_by_track": {TRACK: qualifies},
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        state = "r48_development_passed" if qualifies else "r48_development_failed"
        next_action = (
            "R48 passed off-market development gates; preserve candidate, then continue list-aware optimization"
            if qualifies else
            "R48 failed; preserve R22 and continue with a distinct off-market or list-aware hypothesis"
        )
        campaign._write_progress(state, next_action=next_action,
            r48_result=current["development_result_path"], r48_fit_starts=0)
        print(
            f"R48: parent_mdape={metrics['parent']['mdape_percent']:.5f}% "
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
            "stage": "past_oof_groupwise_development_blend", "run_id": run_id,
            "fit_starts": 0, "error": current["failure"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress("r48_failed",
            next_action="Inspect R48 failure before any retry", r48_failure=current["failure"])
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
