"""R41: choose R22/R26 blend weights by house type from past-only OOF rows."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as helpers
from experiments.bill_target74 import r04_ablation as data_helpers
from experiments.bill_target74 import r19_expert_weight as ledger
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r23_word_char_tfidf as shared
from experiments.bill_target74 import r26_word_tfidf_512 as r26
from experiments.bill_target74 import r40_past_oof_r26_blend as r40
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy


CARD = "R41_house_type_past_oof_r26_blend"
TRACK = "off_market"
SCREEN_MONTHS = r40.SCREEN_MONTHS
MINIMUM_GROUP_OOF_ROWS = 500


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files() -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        campaign.PACKAGE / "r41_house_type_past_oof_blend.py",
        campaign.PACKAGE / "r40_past_oof_r26_blend.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r26_word_tfidf_512.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "r19_expert_weight.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
    ]


def _normalize_group(values: pd.Series) -> pd.Series:
    result = values.fillna("__MISSING__").astype(str).str.strip()
    return result.mask(result.eq(""), "__MISSING__")


def _attach_house_type(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    _, _, _, truth, heldout = greedy._load_fold(month)
    ids = frame.ml_num.astype(str).tolist()
    if ids != truth.ml_num.astype(str).tolist():
        raise RuntimeError(f"OOF rows do not align with safe fold labels for {month}")
    metadata = heldout.assign(ml_num=heldout.ml_num.astype(str)).set_index("ml_num")
    if not metadata.index.is_unique or not set(ids).issubset(metadata.index):
        raise RuntimeError(f"house-type rows do not align with OOF IDs for {month}")
    result = frame.copy()
    result["house_type_name"] = _normalize_group(
        metadata.loc[ids, "house_type_name"].reset_index(drop=True)
    ).to_numpy()
    return result


def _select_weights(oof: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    global_selection = r40.select_weight(oof, cutoff)
    if global_selection["reason"] != "selected_from_strictly_past_oof":
        raise RuntimeError(f"insufficient global past OOF support at {cutoff.date()}")
    past = oof.loc[pd.to_datetime(oof.date_sold, errors="raise") < cutoff]
    by_group = {}
    weights = {}
    for group in sorted(past.house_type_name.astype(str).unique()):
        group_rows = oof.loc[oof.house_type_name.astype(str).eq(group)]
        count = int((pd.to_datetime(group_rows.date_sold, errors="raise") < cutoff).sum())
        selection = None
        if count >= MINIMUM_GROUP_OOF_ROWS:
            selection = r40.select_weight(group_rows, cutoff)
        if selection and selection["reason"] == "selected_from_strictly_past_oof":
            weights[group] = float(selection["weight"])
            by_group[group] = {
                "oof_rows": count,
                "weight": float(selection["weight"]),
                "fallback_to_global": False,
                "selection": selection,
            }
        else:
            weights[group] = float(global_selection["weight"])
            by_group[group] = {
                "oof_rows": count,
                "weight": float(global_selection["weight"]),
                "fallback_to_global": True,
                "fallback_reason": (
                    "below_minimum_group_rows" if count < MINIMUM_GROUP_OOF_ROWS
                    else selection["reason"] if selection else "below_minimum_group_rows"
                ),
                "selection": selection,
            }
    return {
        "global_selection": global_selection,
        "group_weights": weights,
        "groups": by_group,
    }


def _blend_by_group(parent: np.ndarray, variant: np.ndarray, groups: pd.Series,
                    selections: dict) -> tuple[np.ndarray, np.ndarray]:
    labels = _normalize_group(groups)
    global_weight = float(selections["global_selection"]["weight"])
    weights = labels.map(selections["group_weights"]).fillna(global_weight).to_numpy(float)
    prediction = np.empty(len(weights), dtype=float)
    for weight in np.unique(weights):
        mask = weights == weight
        prediction[mask] = r40.blend(parent[mask], variant[mask], float(weight))
    return prediction, weights


def preregister() -> dict:
    if CARD in campaign.PROTOCOL["cards"]:
        card = campaign.PROTOCOL["cards"][CARD]
        if card.get("status") == "registered":
            progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
            r40_result_path = campaign.ROOT / progress["r40_result"]
            r40_screen = json.loads(r40_result_path.read_text(encoding="utf-8"))
            changed = False
            if card.get("variant_run_id") != r40_screen.get("variant_screen_run_id"):
                card["variant_run_id"] = r40_screen["variant_screen_run_id"]
                changed = True
            expected_oof = str((r40_result_path.parent / "past_oof_predictions.csv").relative_to(
                campaign.ROOT
            )).replace("\\", "/")
            if card.get("oof_source") != expected_oof:
                card["oof_source"] = expected_oof
                changed = True
            if changed:
                ledger._write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
                campaign._append({
                    "event": "card_registration_amended",
                    "card": CARD,
                    "variant_run_id": card["variant_run_id"],
                    "oof_source": card["oof_source"],
                    "reason": "record exact verified R26 run and R40 OOF file paths before scoring",
                })
            return card
        raise RuntimeError(f"{CARD} already exists with status {card.get('status')!r}")
    r40_card = campaign.PROTOCOL["cards"].get(r40.CARD, {})
    if r40_card.get("status") != "development_failed":
        raise RuntimeError("R41 registration requires a completed failed R40 development stage")

    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    r40_relative = progress.get("r40_result")
    r40_development_relative = progress.get("r40_development_result")
    if not r40_relative or not r40_development_relative:
        raise RuntimeError("R41 requires verified R40 screen and development artifacts")
    r40_result_path = campaign.ROOT / r40_relative
    r40_development_path = campaign.ROOT / r40_development_relative
    screen = json.loads(r40_result_path.read_text(encoding="utf-8"))
    development = json.loads(r40_development_path.read_text(encoding="utf-8"))
    if (screen["tracks"][TRACK]["qualifies_screen"] is not True
            or development["tracks"][TRACK]["qualifies_development"] is True):
        raise RuntimeError("R40 outcomes do not match the current next-step state")

    card = {
        "status": "registered",
        "difference_from_prior_work": (
            "R40 selected one global R26 share per cutoff and improved its two-month screen, but failed the "
            "eight-month paired-bootstrap gate. Safe seed OOF diagnostics show different R26 complementarity by "
            "house type: Condo Apt (n=2,004) favored an R26-only blend on the seed folds, while Detached "
            "(n=1,187) favored a mixed blend; all other seed-fold types have fewer than 500 rows and use the "
            "global fallback. R41 selects separate house-type weights using only strictly earlier OOF rows."
        ),
        "track": TRACK,
        "parent_run_id": screen["parent_run_id"],
        "variant_run_id": screen["variant_screen_run_id"],
        "screen_months": list(SCREEN_MONTHS),
        "oof_source": str((r40_result_path.parent / "past_oof_predictions.csv").relative_to(
            campaign.ROOT
        )).replace("\\", "/"),
        "oof_seed_months": list(r40.SEED_MONTHS),
        "method": (
            "For each cutoff, choose a global R26 share on the fixed 0.05 grid from all strictly earlier OOF rows. "
            "For each exact house_type_name group with at least 500 strictly earlier OOF rows, independently choose "
            "a group R26 share on the same grid. Minimize MdAPE subject to group or global MAPE regression <=1%; "
            "ties prefer lower MAPE then smaller R26 share. Sparse, unseen, or ineligible groups use that cutoff's "
            "global share. Blend frozen R22 and R26 price predictions and round the resulting price once."
        ),
        "minimum_global_oof_rows": r40.MINIMUM_OOF_ROWS,
        "minimum_group_oof_rows": MINIMUM_GROUP_OOF_ROWS,
        "weight_grid": list(r40.WEIGHTS),
        "mape_regression_guard_percent": 1.0,
        "fallback": "use the current cutoff's global R22/R26 share",
        "expected_fit_starts": 0,
        "screen_gate": {
            "relative_mdape_improvement_percent_min": campaign.PROTOCOL["gates"][
                "screen_relative_mdape_improvement_percent"
            ],
            "both_months_must_improve": True,
            "mape_regression_percent_max": campaign.PROTOCOL["gates"][
                "screen_mape_regression_max_percent"
            ],
        },
        "policy": (
            "Before reading the two screen outcomes, verify all input hashes, then select weights using only OOF rows "
            "with date_sold strictly before each cutoff. Do not change group definitions or fallbacks after scoring. "
            "Do not access reserved or hidden labels."
        ),
    }
    campaign.PROTOCOL["cards"][CARD] = card
    ledger._write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_registered",
        "card": CARD,
        "track": TRACK,
        "parent_run_id": card["parent_run_id"],
        "variant_run_id": card["variant_run_id"],
        "screen_months": list(SCREEN_MONTHS),
        "seed_oof_months": list(r40.SEED_MONTHS),
        "minimum_group_oof_rows": MINIMUM_GROUP_OOF_ROWS,
        "expected_fit_starts": 0,
        "strictly_past_only_selection": True,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r41_registered",
        next_action="run preregistered R41 house-type past-OOF blend screen",
        r41_registered=True,
    )
    return card


def run_screen() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R41 must be preregistered before scoring")
    source_identity = data_helpers._source_identity()
    if source_identity["data_sha256"] != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R41 source data hash differs from the active campaign")

    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    r22_screen, r22_path, r22_manifest = r22._verified_screen_result(campaign)
    r26_screen, r26_path, r26_manifest = r26._verified_r26_screen_result(campaign)
    r22_rows = r40._read_checked_result(r22_path, r22_screen, r22_manifest, r22.CARD, source_identity["data_sha256"])
    r26_rows = r40._read_checked_result(r26_path, r26_screen, r26_manifest, r26.CARD, source_identity["data_sha256"])

    r40_card = campaign.PROTOCOL["cards"][r40.CARD]
    r40_result_path = campaign.ROOT / r40_card["result_path"]
    r40_result = json.loads(r40_result_path.read_text(encoding="utf-8"))
    r40_manifest = json.loads((r40_result_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    r40_oof_path = r40_result_path.parent / "past_oof_predictions.csv"
    if (r40_result.get("tracks", {}).get(TRACK, {}).get("qualifies_screen") is not True
            or r40_result.get("data_sha256") != source_identity["data_sha256"]
            or r40_result.get("reserved_months_labels_opened")
            or r40_result.get("hidden_labels_opened")
            or r40_manifest.get("result_sha256") != _sha256(r40_result_path)
            or r40_manifest.get("paired_predictions_sha256")
            != _sha256(r40_result_path.parent / "paired_predictions.csv")
            or r40_manifest.get("past_oof_predictions_sha256") != _sha256(r40_oof_path)):
        raise RuntimeError("R40 screen/OOF inputs failed identity or hash checks")

    raw_oof = pd.read_csv(r40_oof_path, dtype={"ml_num": str})
    if raw_oof.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R40 OOF source contains duplicate month/listing IDs")
    oof_parts = []
    for month in sorted(raw_oof.month.astype(str).unique()):
        part = raw_oof.loc[raw_oof.month.astype(str).eq(month)].copy()
        oof_parts.append(_attach_house_type(part, month))
    oof = pd.concat(oof_parts, ignore_index=True)
    expected_oof_months = set(r40.SEED_MONTHS) | {
        month for month in r40.DEVELOPMENT_MONTHS
        if pd.Period(month, freq="M").start_time < pd.Period(SCREEN_MONTHS[-1], freq="M").start_time
    }
    if (set(oof.month.astype(str).unique()) != set(r40_result["oof_months_for_screen"])
            or set(oof.month.astype(str).unique()) != expected_oof_months):
        raise RuntimeError("R41 OOF months differ from the verified R40 screen OOF manifest")

    run_id = "bill_target74_r41_house_type_blend_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({
        "event": "card_start",
        "card": CARD,
        "track": TRACK,
        "run_id": run_id,
        "expected_fit_starts": 0,
        "screen_months": list(SCREEN_MONTHS),
        "strictly_past_only_selection": True,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign._write_progress("r41_running", next_action="score registered R41 safe screen folds")

    source_paths = _source_files()
    code_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_paths
    }
    selections = []
    paired_parts = []
    try:
        # Select all weights before loading any screen-fold truth for scoring.
        for month in SCREEN_MONTHS:
            cutoff = pd.Period(month, freq="M").start_time
            selection = _select_weights(oof, cutoff)
            selections.append({
                "screen_month": month,
                "cutoff_exclusive": cutoff.isoformat(),
                "global": selection["global_selection"],
                "group_weights": selection["groups"],
            })

        for month, selection in zip(SCREEN_MONTHS, selections):
            _, _, _, truth, heldout = greedy._load_fold(month)
            parent = shared.r22_reference_frame(r22_rows, truth, month, TRACK)
            alternate = r26_rows.loc[
                r26_rows.month.astype(str).eq(month) & r26_rows.track.eq(TRACK)
            ].copy()
            ids = parent.ml_num.astype(str).tolist()
            if (alternate.ml_num.astype(str).tolist() != ids
                    or not np.array_equal(alternate.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
                    or not np.array_equal(alternate.parent_prediction.to_numpy(float),
                                          parent.predicted_price.to_numpy(float))):
                raise RuntimeError(f"R26 screen rows do not align to R22 for {month}")
            metadata = heldout.assign(ml_num=heldout.ml_num.astype(str)).set_index("ml_num")
            if not metadata.index.is_unique or not set(ids).issubset(metadata.index):
                raise RuntimeError(f"screen house-type rows do not align for {month}")
            groups = _normalize_group(metadata.loc[ids, "house_type_name"].reset_index(drop=True))
            prediction, row_weights = _blend_by_group(
                parent.predicted_price.to_numpy(float),
                alternate.candidate_prediction.to_numpy(float),
                groups,
                {"global_selection": selection["global"],
                 "group_weights": {name: entry["weight"] for name, entry in selection["group_weights"].items()}},
            )
            candidate = pd.DataFrame({
                "ml_num": parent.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "predicted_price": prediction,
            })
            paired = helpers._candidate_frame(month, TRACK, truth, parent, candidate)
            paired["house_type_name"] = groups.to_numpy()
            paired["r26_blend_weight"] = row_weights
            paired_parts.append(paired)

        paired = pd.concat(paired_parts, ignore_index=True)
        score = shared._track_result(paired)
        paired_path = output / "paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        oof_output = output / "past_oof_predictions.csv"
        oof.to_csv(oof_output, index=False)
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"],
            "run_id": run_id,
            "card": CARD,
            "track": TRACK,
            "parent_run_id": r22_screen["run_id"],
            "variant_run_id": r26_screen["run_id"],
            "r40_screen_run_id": r40_result["run_id"],
            "screen_months": list(SCREEN_MONTHS),
            "fit_starts": 0,
            "new_fit_starts": 0,
            "tracks": {TRACK: score},
            "weight_selections": selections,
            "minimum_global_oof_rows": r40.MINIMUM_OOF_ROWS,
            "minimum_group_oof_rows": MINIMUM_GROUP_OOF_ROWS,
            "weight_grid": list(r40.WEIGHTS),
            "mape_regression_guard_percent": 1.0,
            "strictly_past_only_selection": True,
            "data_sha256": source_identity["data_sha256"],
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        result_path = output / "screen_results.json"
        ledger._write_json(result_path, result)
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"],
            "run_id": run_id,
            "source_identity": source_identity,
            "code_sha256": code_hashes,
            "input_result_sha256": {
                "r22_screen": _sha256(r22_path),
                "r26_screen": _sha256(r26_path),
                "r40_screen": _sha256(r40_result_path),
                "r40_past_oof": _sha256(r40_oof_path),
            },
            "past_oof_predictions_sha256": _sha256(oof_output),
            "paired_predictions_sha256": _sha256(paired_path),
            "result_sha256": _sha256(result_path),
            "fit_starts": 0,
            "new_fit_starts": 0,
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        ledger._write_json(output / "run_manifest.json", manifest)

        lines = [
            "# R41：按房屋类型使用严格过去 OOF 选择 R22/R26 权重",
            "",
            f"运行：`{run_id}`；本轮重用保存的 R22/R26 预测，新增 learner fits 为 0。",
            "",
            "每个筛选截止日先用全部过去 OOF 行选择全局 R26 权重；某一 `house_type_name` 至少有 500 条严格过去 OOF 行时，再单独选择该组权重。组内 MAPE 恶化超过 1%、样本不足或类型未见时回退到该截止日的全局权重。权重网格为 0.00–1.00，步长 0.05。",
            "",
            "| 筛选月 | 全局 R26 权重 | House type | 过去 OOF 行 | 使用权重 | 回退全局 | 过去 OOF MdAPE 父→混合 | MAPE 变化 |",
            "|---|---:|---|---:|---:|---|---:|---:|",
        ]
        for selection in selections:
            month = selection["screen_month"]
            global_selection = selection["global"]
            lines.append(
                f"| {month} | {global_selection['weight']:.2f} | **全局** | {global_selection['oof_rows']} | "
                f"{global_selection['weight']:.2f} | 否 | {global_selection['parent']['mdape_percent']:.5f}%→"
                f"{global_selection['selected']['mdape_percent']:.5f}% | "
                f"{global_selection['selected_mape_relative_change_percent']:+.3f}% |"
            )
            for group, choice in selection["group_weights"].items():
                details = choice.get("selection") or {}
                parent_metric = details.get("parent") or {}
                selected_metric = details.get("selected") or {}
                mdape_text = (
                    f"{parent_metric['mdape_percent']:.5f}%→{selected_metric['mdape_percent']:.5f}%"
                    if parent_metric and selected_metric else "fallback"
                )
                mape_text = (
                    f"{details['selected_mape_relative_change_percent']:+.3f}%"
                    if "selected_mape_relative_change_percent" in details else "fallback"
                )
                lines.append(
                    f"| {month} | {global_selection['weight']:.2f} | {group} | {choice['oof_rows']} | "
                    f"{choice['weight']:.2f} | {'是' if choice['fallback_to_global'] else '否'} | "
                    f"{mdape_text} | {mape_text} |"
                )
        lines.extend([
            "",
            "## 双月筛选结果",
            "",
            f"R22 pooled MdAPE：{score['parent']['mdape_percent']:.5f}%；R41：{score['candidate']['mdape_percent']:.5f}%；",
            f"相对 MdAPE 改善：{score['relative_mdape_improvement_percent']:+.3f}%；MAPE 相对变化：{score['mape_relative_change_percent']:+.3f}%；",
            f"筛选门槛通过：{score['qualifies_screen']}。",
            "",
            "本轮只读取注册的 2023-03 和 2024-06 筛选折及截至各自截止日的 OOF 行；没有读取 2025-05/08 或隐藏标签。预留集因先前完整 parquet 载入事件不具备独立确认资格。",
            f"配对预测、past-OOF 文件和 SHA-256 清单位于 `{output.relative_to(campaign.ROOT)}`。",
            "",
        ])
        (output / "research_report.md").write_text("\n".join(lines), encoding="utf-8")

        card.update({
            "status": "screen_passed" if score["qualifies_screen"] else "screen_failed",
            "run_id": run_id,
            "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "fit_starts": 0,
            "qualifies_screen": score["qualifies_screen"],
            "screen_metrics": {
                "parent_mdape_percent": score["parent"]["mdape_percent"],
                "candidate_mdape_percent": score["candidate"]["mdape_percent"],
                "relative_mdape_improvement_percent": score["relative_mdape_improvement_percent"],
                "mape_relative_change_percent": score["mape_relative_change_percent"],
                "monthly_relative_mdape_improvement_percent": {
                    month: row["relative_mdape_improvement_percent"]
                    for month, row in score["monthly"].items()
                },
                "weight_by_screen_month_and_group": {
                    row["screen_month"]: {
                        "global": row["global"]["weight"],
                        "groups": {name: value["weight"] for name, value in row["group_weights"].items()},
                    }
                    for row in selections
                },
            },
        })
        ledger._write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._append({
            "event": "card_complete",
            "card": CARD,
            "run_id": run_id,
            "fit_starts": 0,
            "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "qualifies_by_track": {TRACK: score["qualifies_screen"]},
            "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r41_screen_complete",
            next_action=(
                "run R41 eight-month development confirmation with past-only group weights"
                if score["qualifies_screen"]
                else "preserve R22 and register a distinct off-market hypothesis"
            ),
            r41_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        return result
    except Exception as exc:
        failure = {
            "run_id": run_id,
            "card": CARD,
            "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": 0,
            "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        }
        ledger._write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed",
            "card": CARD,
            "run_id": run_id,
            "fit_starts": 0,
            "error": failure["error"],
            "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r41_failed",
            next_action="audit R41 failure checkpoint and inputs before any retry",
            r41_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true", help="register the screen card without scoring")
    args = parser.parse_args()
    if args.register:
        print(json.dumps(preregister(), indent=2))
    else:
        result = run_screen()
        score = result["tracks"][TRACK]
        print(
            f"R41: parent_mdape={score['parent']['mdape_percent']:.5f}% "
            f"candidate_mdape={score['candidate']['mdape_percent']:.5f}% "
            f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
            f"mape_change={score['mape_relative_change_percent']:+.3f}% "
            f"pass={score['qualifies_screen']} run={result['run_id']}",
            flush=True,
        )
