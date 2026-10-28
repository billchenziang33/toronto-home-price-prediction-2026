"""Audit and score saved R33 fixed-origin predictions without refitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R33_R22_R32_fixed_half_blend"
DEFAULT_RUN_ID = "bill_target74_r33_fixedorigin_20261025T032803Z"
TRACK = "off_market"
SEALED_MONTHS = {"2025-05", "2025-08"}
PAIRED_COLUMNS = ["month", "track", "ml_num", "sold_price", "parent_prediction",
                  "candidate_prediction", "origin", "horizon"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_id_hash(frame: pd.DataFrame) -> str:
    values = sorted(frame.ml_num.astype(str).tolist())
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _audit_saved_run(run_id: str) -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r32_comparable_r22 as r32
    from experiments.bill_target74 import r33_fixed_origin as r33
    from experiments.bill_target74 import runner as campaign

    protocol = _read_json(campaign.PROTOCOL_PATH)
    _check(protocol.get("campaign_id") == "bill_target74_unbounded_20261024",
            "active campaign ID changed")
    _check(protocol.get("budget_policy") == "unbounded_until_objective_and_validation_complete",
            "active campaign budget policy is no longer unbounded")
    fit_time_limits = (
        "campaign_fit_starts", "campaign_minutes", "exploration_fit_starts",
        "exploration_minutes", "final_reserve_fit_starts", "final_reserve_minutes",
        "batch_fit_starts", "batch_minutes", "single_fit_seconds",
    )
    _check(all(protocol["limits"].get(key) is None for key in fit_time_limits),
            "active campaign contains a finite fit/time cap")

    card = protocol["cards"][CARD]
    fixed = card["fixed_origin_confirmation"]
    _check(card.get("status") == "fixed_origin_confirmation_running",
            "R33 card is not awaiting fixed-origin scoring")
    _check(fixed.get("status") == "running" and fixed.get("run_id") == run_id,
            "R33 fixed-origin protocol does not point to this run")
    _check(fixed.get("reserved_months_labels_opened") is False
            and fixed.get("hidden_labels_opened") is False,
            "R33 protocol indicates a sealed label set was opened")

    progress = _read_json(campaign.STATE)
    _check(progress.get("status") == "r33_fixed_origin_confirmation_running"
            and progress.get("r33_fixed_origin_run_id") == run_id,
            "campaign progress does not identify this running R33 confirmation")

    run_output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    checkpoint_path = run_output / "scoring_error_checkpoint.json"
    checkpoint = _read_json(checkpoint_path)
    _check(checkpoint.get("run_id") == run_id and checkpoint.get("card") == CARD,
            "scoring checkpoint identity mismatch")
    _check(checkpoint.get("fit_starts") == 4
            and checkpoint.get("reserved_months_labels_opened") is False
            and checkpoint.get("hidden_labels_opened") is False,
            "scoring checkpoint has unexpected fit count or label-access state")

    source_snapshot = campaign.ROOT / checkpoint["fit_source_snapshot"]
    _check(_sha256(source_snapshot) == checkpoint["fit_source_sha256"],
            "original fit source snapshot hash mismatch")
    preregistered = run_output / "provenance" / "campaign_protocol_preregistered.json"
    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    run_events = [row for row in ledger if row.get("run_id") == run_id]
    starts = [row for row in run_events if row.get("event") == "fit_start"]
    completes = [row for row in run_events if row.get("event") == "fit_complete"]
    card_starts = [row for row in run_events if row.get("event") == "card_start"]
    scoring_errors = [row for row in run_events if row.get("event") == "card_scoring_error"]
    origin_completes = [row for row in run_events if row.get("event") == "fixed_origin_complete"]
    _check(len(card_starts) == 1 and len(scoring_errors) == 1,
            "expected one card_start and one scoring-error ledger event")
    _check(len(starts) == 4 and len(completes) == 4 and len(origin_completes) == 2,
            "ledger does not contain four completed fits and two completed origins")
    _check(not any(row.get("event") in {"card_complete", "card_failed"} for row in run_events),
            "R33 run is already terminal in the ledger")
    _check(all(row.get("status") == "complete" for row in completes),
            "one or more registered model fits did not complete")
    _check([row.get("fit_id") for row in starts] == checkpoint["fit_ids_started"]
            == checkpoint["fit_ids_completed"]
            == [row.get("fit_id") for row in completes],
            "checkpoint and fit ledger IDs or ordering differ")

    card_start = card_starts[0]
    _check(card_start.get("expected_fit_starts") == 4
            and card_start.get("data_sha256") == protocol["data_sha256"],
            "card_start plan or dataset hash mismatch")
    _check(_sha256(preregistered) == card_start.get("protocol_snapshot_sha256"),
            "preregistered protocol snapshot does not match card_start")
    snapshot = _read_json(preregistered)
    snapshot_fixed = snapshot["cards"][CARD]["fixed_origin_confirmation"]
    _check(snapshot_fixed.get("status") == "registered"
            and snapshot_fixed.get("expected_new_fit_starts") == 4,
            "saved preregistration does not contain the four-fit registered method")
    snapshot_gates = snapshot_fixed["gates"]
    registered_horizon_gate = snapshot_gates.get(
        "fixed_origin_horizon_regression_max_percent",
        snapshot_gates.get("fixed_origin_horizon_mdape_regression_max_percent"),
    )
    _check(registered_horizon_gate == 5.0
            and fixed["gates"].get("fixed_origin_horizon_regression_max_percent") == 5.0,
            "corrected horizon gate differs from its preregistered 5% numeric threshold")
    _check(fixed["gates"].get("fixed_origin_pooled_mdape_regression_max_percent") == 0.0
            and fixed["gates"].get("fixed_origin_mape_regression_max_percent") == 1.0,
            "fixed-origin pooled MdAPE or MAPE gate changed")

    # Reuse the original R33 input audit while giving it an in-memory view of the
    # preregistered state. The on-disk protocol and append-only ledger are untouched.
    audit_protocol = json.loads(json.dumps(protocol))
    audit_protocol["cards"][CARD]["status"] = "fixed_origin_confirmation_registered"
    audit_protocol["cards"][CARD]["fixed_origin_confirmation"]["status"] = "registered"
    audit_campaign = SimpleNamespace(
        PROTOCOL=audit_protocol, PACKAGE=campaign.PACKAGE,
        CAMPAIGN_ROOT=campaign.CAMPAIGN_ROOT, DATA=campaign.DATA, ROOT=campaign.ROOT,
    )
    inputs = r33._load_inputs(audit_campaign, budget, r22, r32)
    _check(inputs["source"]["data_sha256"] == checkpoint.get("data_sha256", protocol["data_sha256"]),
            "active dataset hash differs from the saved scoring run")
    _check(card_start.get("reference_result_sha256") == _sha256(inputs["r22_result_path"])
            and card_start.get("reference_predictions_sha256") == _sha256(inputs["r22_pairs_path"]),
            "R22 reference hashes differ from the registered card_start")
    _check(card_start.get("development_result_sha256") == _sha256(inputs["r33_dev_result_path"]),
            "R33 development result hash differs from the registered card_start")

    return {
        "campaign": campaign, "budget": budget, "r22": r22, "r32": r32,
        "r33": r33, "greedy": greedy, "hide_answers": hide_answers,
        "protocol": protocol,
        "fixed": fixed, "inputs": inputs, "run_output": run_output,
        "source_snapshot": source_snapshot, "preregistered": preregistered,
        "checkpoint": checkpoint, "ledger": ledger, "run_events": run_events,
        "starts": starts, "completes": completes, "origin_completes": origin_completes,
    }


def _reconstruct_and_verify(run_id: str) -> tuple[dict, pd.DataFrame, list[dict]]:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy

    audit = _audit_saved_run(run_id)
    campaign, r22, r32 = audit["campaign"], audit["r22"], audit["r32"]
    protocol, inputs = audit["protocol"], audit["inputs"]
    run_output, fixed = audit["run_output"], audit["fixed"]
    component_events = {(row["origin"], row["component"]): row for row in audit["completes"]}
    _check(len(component_events) == 4, "completed fit events do not identify four unique components")

    partial_path = run_output / "partial_off_market.csv"
    partial = pd.read_csv(partial_path, dtype={"ml_num": str})
    _check(set(PAIRED_COLUMNS) == set(partial.columns), "partial paired predictions have unexpected columns")
    _check(not partial.duplicated(["origin", "month", "ml_num"]).any(),
            "partial paired predictions contain duplicate home rows")
    _check(not (set(partial.month.astype(str)) & SEALED_MONTHS),
            "partial predictions contain a reserved month")
    _check(set(partial.origin.astype(str)) == set(inputs["origins"])
            and set(partial.horizon.astype(int)) == set(inputs["horizons"]),
            "partial predictions do not cover the registered origins and horizons")

    paired_parts: list[pd.DataFrame] = []
    component_manifest: list[dict] = []
    for origin_text in inputs["origins"]:
        origin = pd.Timestamp(origin_text)
        end = origin + pd.DateOffset(months=4)
        train = r22.load_fixed_origin_training(campaign.DATA, origin_text)
        train["ml_num"] = train.ml_num.astype(str)
        train["date_sold"] = pd.to_datetime(train.date_sold, errors="raise")
        _check(not (set(train.date_sold.dt.to_period("M").astype(str)) & SEALED_MONTHS),
                f"sealed month labels entered reloaded training data at {origin_text}")
        _check((train.date_sold < origin).all(), f"training data is not strictly pre-origin at {origin_text}")

        heldout = pd.read_parquet(
            campaign.DATA,
            filters=[("date_sold", ">=", origin), ("date_sold", "<", end)],
        )
        heldout["date_sold"] = pd.to_datetime(heldout.date_sold, errors="raise")
        heldout["ml_num"] = heldout.ml_num.astype(str)
        _check(not heldout.empty and not heldout.ml_num.duplicated().any(),
                f"invalid held-out homes at {origin_text}")
        homes = hide_answers(heldout, "off_market").reset_index(drop=True)
        homes["ml_num"] = homes.ml_num.astype(str)
        ids = homes.ml_num.to_numpy()
        months = heldout.date_sold.dt.to_period("M").astype(str).to_numpy()
        expected_months = {(origin.to_period("M") + offset).strftime("%Y-%m") for offset in range(4)}
        _check(set(months) == expected_months and not (set(months) & SEALED_MONTHS),
                f"held-out rows contain an unexpected or sealed month at {origin_text}")
        horizons = greedy._month_horizons(heldout.date_sold, origin)

        reference = inputs["r22_rows"].loc[
            inputs["r22_rows"].origin.astype(str).eq(origin_text)
        ].copy()
        _check(not reference.ml_num.astype(str).duplicated().any(),
                f"R22 reference contains duplicate IDs at {origin_text}")
        aligned = reference.set_index(reference.ml_num.astype(str), drop=False).reindex(ids)
        _check(not aligned.ml_num.isna().any(), f"R22 reference misses source IDs at {origin_text}")
        sold = heldout.sold_price.to_numpy(dtype="float64")
        _check(np.array_equal(sold, aligned.sold_price.to_numpy(dtype="float64")),
                f"R22 saved labels differ from source labels at {origin_text}")
        _check(np.array_equal(months, aligned.month.astype(str).to_numpy()),
                f"R22 saved months differ from source months at {origin_text}")
        _check(np.array_equal(horizons, aligned.horizon.to_numpy(dtype=int)),
                f"R22 saved horizons differ from source horizons at {origin_text}")
        r22_prediction = aligned.candidate_prediction.to_numpy(dtype="float64")

        detached_train_mask = greedy._specialist_mask(train, "Detached")
        detached_eval_mask = greedy._specialist_mask(homes, "Detached")
        component_predictions: dict[str, pd.Series] = {}
        for component, fit_rows, eval_rows in (
            ("R32_base", train, homes),
            ("R32_expert", train.loc[detached_train_mask].copy(),
             homes.loc[detached_eval_mask].copy()),
        ):
            event = component_events.get((origin_text, component))
            _check(event is not None, f"missing completed fit event for {origin_text}/{component}")
            _check(event.get("training_rows") == len(fit_rows)
                    and event.get("training_row_ids_sha256") == _row_id_hash(fit_rows),
                    f"training rows differ from the completed fit ledger for {origin_text}/{component}")
            _check(event.get("evaluation_rows") == len(eval_rows)
                    and event.get("evaluation_row_ids_sha256") == _row_id_hash(eval_rows),
                    f"evaluation rows differ from the completed fit ledger for {origin_text}/{component}")
            artifact = (campaign.ROOT / event["saved_predictions"]).resolve()
            _check(artifact.is_relative_to(campaign.ROOT.resolve()),
                    "saved component path escapes the project root")
            _check(_sha256(artifact) == event.get("saved_predictions_sha256"),
                    f"saved component CSV hash mismatch for {origin_text}/{component}")
            frame = pd.read_csv(artifact, dtype={"ml_num": str})
            _check(list(frame.columns) == ["ml_num", "prediction"]
                    and np.array_equal(frame.ml_num.astype(str).to_numpy(),
                                       eval_rows.ml_num.astype(str).to_numpy()),
                    f"saved component rows differ from fit order for {origin_text}/{component}")
            prediction = frame.prediction.to_numpy(dtype="float64")
            _check(np.isfinite(prediction).all() and (prediction > 0).all(),
                    f"invalid saved predictions for {origin_text}/{component}")
            component_predictions[component] = pd.Series(prediction, index=eval_rows.ml_num.astype(str))
            component_manifest.append({
                "origin": origin_text, "component": component,
                "fit_id": event["fit_id"], "status": event["status"],
                "training_rows": int(len(fit_rows)),
                "training_max_date_sold": str(pd.to_datetime(fit_rows.date_sold).max().date()),
                "training_row_ids_sha256": _row_id_hash(fit_rows),
                "evaluation_rows": int(len(eval_rows)),
                "evaluation_row_ids_sha256": _row_id_hash(eval_rows),
                "feature_columns": event.get("feature_columns", []),
                "tfidf_feature_count": event.get("tfidf_feature_count"),
                "neighbor_comparable_feature_count": event.get("neighbor_comparable_feature_count"),
                "fit_prediction_sha256_from_ledger": event.get("prediction_sha256"),
                "saved_predictions": event["saved_predictions"],
                "saved_predictions_sha256": event["saved_predictions_sha256"],
                "fit_prediction_vector_hash_recomputed_from_csv": False,
                "tfidf_vocabulary_recovered": False,
            })

        base_prediction = component_predictions["R32_base"].reindex(ids).to_numpy(dtype="float64")
        expert_prediction = component_predictions["R32_expert"].reindex(
            homes.loc[detached_eval_mask, "ml_num"].astype(str).to_numpy()
        ).to_numpy(dtype="float64")
        _check(np.isfinite(base_prediction).all() and np.isfinite(expert_prediction).all(),
                f"component predictions do not cover evaluation homes at {origin_text}")
        r32_prediction = greedy._blend_specialist_predictions(
            np.round(base_prediction), np.round(expert_prediction), homes, "Detached", 0.75,
        )
        r33_prediction = np.round(0.5 * r22_prediction + 0.5 * r32_prediction)
        paired_parts.append(pd.DataFrame({
            "month": months, "track": TRACK, "ml_num": ids, "sold_price": sold,
            "parent_prediction": r22_prediction, "candidate_prediction": r33_prediction,
            "origin": origin_text, "horizon": horizons,
        }))

    paired = pd.concat(paired_parts, ignore_index=True)
    _check(len(paired) == len(partial) == 13184,
            f"unexpected row count: reconstructed={len(paired)}, saved={len(partial)}")
    for column in PAIRED_COLUMNS:
        left, right = paired[column], partial[column]
        if column in {"sold_price", "parent_prediction", "candidate_prediction"}:
            equal = np.array_equal(left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64"))
        elif column == "horizon":
            equal = np.array_equal(left.to_numpy(dtype=int), right.to_numpy(dtype=int))
        else:
            equal = np.array_equal(left.astype(str).to_numpy(), right.astype(str).to_numpy())
        _check(equal, f"saved partial predictions differ from audited reconstruction in {column}")
    return audit, paired, component_manifest


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def finalize(run_id: str) -> dict:
    audit, paired, component_manifest = _reconstruct_and_verify(run_id)
    campaign, r22 = audit["campaign"], audit["r22"]
    protocol, fixed, inputs = audit["protocol"], audit["fixed"], audit["inputs"]
    run_output = audit["run_output"]
    result_path = run_output / "r33_fixed_origin_results.json"
    manifest_path = run_output / "run_manifest.json"
    _check(not result_path.exists() and not manifest_path.exists(),
            "final scoring artifacts already exist; inspect them instead of overwriting")

    gate = fixed["gates"]
    score = r22.score_fixed_origin_confirmation(
        paired, origins=inputs["origins"], horizons=inputs["horizons"],
        maximum_pooled_mdape_regression_percent=gate["fixed_origin_pooled_mdape_regression_max_percent"],
        maximum_horizon_mdape_regression_percent=gate["fixed_origin_horizon_regression_max_percent"],
        maximum_mape_regression_percent=gate["fixed_origin_mape_regression_max_percent"],
    )
    fit_details = audit["completes"]
    result = {
        "campaign_id": protocol["campaign_id"], "run_id": run_id,
        "card": CARD, "stage": "fixed_origin_confirmation", "track": TRACK,
        "reference_candidate": "R22_train_fold_tfidf_remarks",
        "reference_run_id": fixed["reference_run_id"],
        "development_run_id": fixed["development_result_path"].split("/batches/")[-1].split("/")[0],
        "screen_run_id": protocol["cards"][CARD]["screen_run_id"],
        "variant_run_id": fixed["R32_variant_run_id"],
        "origins": list(inputs["origins"]), "horizons": list(inputs["horizons"]),
        "fit_starts": len(fit_details), "additional_fit_starts_during_recovery": 0,
        "scored_saved_predictions": True, "recovered_without_refit": True,
        "blend_weight": {"R22": 0.5, "R32": 0.5},
        "training_policy": fixed["training_policy"], "tracks": {TRACK: score},
        "fit_details": fit_details, "component_manifest": component_manifest,
        "data_sha256": inputs["source"]["data_sha256"],
        "scoring_recovery": {
            "cause": "The first scoring attempt used a horizon-gate key name that differed from the preregistered campaign key.",
            "numeric_horizon_threshold_percent": gate["fixed_origin_horizon_regression_max_percent"],
            "original_fit_source_snapshot": str(audit["source_snapshot"].relative_to(campaign.ROOT)).replace("\\", "/"),
            "original_fit_source_sha256": audit["checkpoint"]["fit_source_sha256"],
            "saved_component_prediction_csvs_hash_verified": True,
            "candidate_reconstruction_matches_saved_partial_exactly": True,
            "fit_prediction_vector_hash_recomputed_from_csv": False,
        },
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }

    paired_path = run_output / "paired_predictions.csv"
    _atomic_csv(paired_path, paired)
    _atomic_json(result_path, result)

    source_files = [
        campaign.PROTOCOL_PATH, campaign.PACKAGE / "r33_fixed_origin.py",
        campaign.PACKAGE / "finalize_r33_fixed_origin.py",
        campaign.PACKAGE / "r32_comparable_r22.py", campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "runner.py", campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/bill_greedy/comparables.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
    ]
    prereg_gate = _read_json(audit["preregistered"])["cards"][CARD]["fixed_origin_confirmation"]["gates"]
    manifest = {
        "campaign_id": protocol["campaign_id"], "run_id": run_id,
        "stage": "fixed_origin_confirmation", "source_identity": inputs["source"],
        "input_hashes": {
            "r22_fixed_origin_manifest_sha256": _sha256(inputs["r22_dir"] / "run_manifest.json"),
            "r22_fixed_origin_result_sha256": _sha256(inputs["r22_result_path"]),
            "r22_fixed_origin_predictions_sha256": _sha256(inputs["r22_pairs_path"]),
            "r32_screen_result_sha256": _sha256(inputs["r32_result_path"]),
            "r33_screen_result_sha256": _sha256(inputs["r33_screen_result_path"]),
            "r33_development_result_sha256": _sha256(inputs["r33_dev_result_path"]),
            "preregistered_protocol_sha256": _sha256(audit["preregistered"]),
            "original_fit_source_snapshot_sha256": audit["checkpoint"]["fit_source_sha256"],
            "saved_partial_predictions_sha256": _sha256(run_output / "partial_off_market.csv"),
            "scoring_checkpoint_sha256": _sha256(run_output / "scoring_error_checkpoint.json"),
        },
        "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                         for path in source_files},
        "fit_starts": len(fit_details), "additional_fit_starts_during_recovery": 0,
        "fit_details": fit_details, "component_manifest": component_manifest,
        "recovered_without_refit": True,
        "registered_horizon_gate_alias": {
            "preregistered_key": "fixed_origin_horizon_mdape_regression_max_percent",
            "corrected_campaign_key": "fixed_origin_horizon_regression_max_percent",
            "preregistered_numeric_value": prereg_gate.get(
                "fixed_origin_horizon_regression_max_percent",
                prereg_gate.get("fixed_origin_horizon_mdape_regression_max_percent")),
            "used_numeric_value": gate["fixed_origin_horizon_regression_max_percent"],
        },
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _atomic_json(manifest_path, manifest)

    lines = [
        "# R33 off-market 固定起点确认", "",
        f"运行：`{run_id}`；R22 固定起点参考：`{fixed['reference_run_id']}`。",
        f"Learner fit starts：{len(fit_details)}；本次评分恢复新增 fits：0；R22/R32 固定各占 50%。",
        "评分从已保存并逐项核验哈希的组件预测恢复，没有重新训练。原评分中断是 horizon gate 字段名与协议字段不一致；预登记的 5% 数值门槛没有改变。",
        "训练严格早于起点；固定起点训练行读取前排除了 2025-05 和 2025-08。隐藏标签未读取。", "",
        "| 赛道 | R22 MdAPE | R33 MdAPE | 相对改善 | MAPE 变化 | 最差 horizon 退化 | 通过 |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| {TRACK} | {score['parent']['mdape_percent']:.5f}% | "
        f"{score['candidate']['mdape_percent']:.5f}% | "
        f"{score['relative_mdape_improvement_percent']:+.3f}% | "
        f"{score['mape_relative_change_percent']:+.3f}% | "
        f"{score['worst_horizon_mdape_regression_percent']:.3f}% | "
        f"{'通过' if score['passes_fixed_origin'] else '未通过'} |",
        "", "## 按起点", "",
        "| 起点 | n | R22 MdAPE | R33 MdAPE | R22 MAPE | R33 MAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for origin, values in score["by_origin"].items():
        lines.append(
            f"| {origin} | {values['candidate']['n']} | "
            f"{values['parent']['mdape_percent']:.5f}% | {values['candidate']['mdape_percent']:.5f}% | "
            f"{values['parent']['mape_percent']:.5f}% | {values['candidate']['mape_percent']:.5f}% |"
        )
    lines.extend(["", "## 按 horizon", "", "| 月数 | n | R22 MdAPE | R33 MdAPE | 相对 MdAPE 变化 |",
                  "|---:|---:|---:|---:|---:|"])
    for horizon, values in score["by_horizon"].items():
        lines.append(
            f"| {horizon} | {values['candidate']['n']} | "
            f"{values['parent']['mdape_percent']:.5f}% | {values['candidate']['mdape_percent']:.5f}% | "
            f"{values['relative_mdape_change_percent']:+.3f}% |"
        )
    lines.extend([
        "", "固定起点门槛：pooled MdAPE 不退化、最差 horizon MdAPE 退化不超过 5%、MAPE 退化不超过 1%。",
        "开发阶段的 off-market MdAPE 目标仍是严格低于 7%；本确认不代表目标已达到。预留月份与隐藏标签保持封存。", "",
    ])
    report_path = run_output / "research_report.md"
    temporary_report = report_path.with_suffix(".md.tmp")
    temporary_report.write_text("\n".join(lines), encoding="utf-8")
    temporary_report.replace(report_path)

    campaign._append({
        "event": "card_complete", "card": CARD, "stage": "fixed_origin_confirmation",
        "run_id": run_id, "fit_starts": len(fit_details),
        "additional_fit_starts_during_recovery": 0,
        "scored_saved_predictions": True, "recovered_without_refit": True,
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "passes_by_track": {TRACK: score["passes_fixed_origin"]},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    audit["r33"]._update_protocol(
        campaign, "fixed_origin_confirmation_complete", status="complete", run_id=run_id,
        result_path=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        result_sha256=_sha256(result_path), paired_predictions_sha256=_sha256(paired_path),
        fit_starts=len(fit_details), passes_fixed_origin=score["passes_fixed_origin"],
        metrics=score, recovered_without_refit=True,
        reserved_months_labels_opened=False, hidden_labels_opened=False,
    )
    next_action = (
        "R33 passed fixed-origin but off-market development MdAPE remains at or above 7%; register a distinct improvement and keep reserve labels sealed"
        if score["passes_fixed_origin"] else
        "R33 failed fixed-origin; retain R22 as the validated off-market candidate and register a distinct improvement; keep reserve labels sealed"
    )
    campaign._write_progress(
        "r33_fixed_origin_confirmation_complete", next_action=next_action,
        r33_fixed_origin_run_id=run_id,
        r33_fixed_origin_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=len(fit_details),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true", help="audit saved artifacts without fitting or scoring")
    modes.add_argument("--finalize", action="store_true", help="score audited saved predictions without fitting")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    if args.preflight:
        audit, paired, components = _reconstruct_and_verify(args.run_id)
        output = {
            "run_id": args.run_id, "verified_saved_rows": len(paired),
            "verified_fit_completions": len(audit["completes"]),
            "verified_origins": [row["origin"] for row in audit["origin_completes"]],
            "component_artifacts": len(components), "additional_fit_starts": 0,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
    else:
        output = finalize(args.run_id)
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
