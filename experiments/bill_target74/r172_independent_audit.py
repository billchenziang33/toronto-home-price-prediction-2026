"""Independent artifact audit for R172 screen and development runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "bill_target74_unbounded_20261024"
CARD = "R172_listaware_fair_loss"
PARENT_CARD = "R35_listaware_detached_r22_text_expert"
TRACK = "list_aware"
SOURCE_RELATIVE = "experiments/bill_target74/r172_listaware_fair_loss.py"
AUDIT_RELATIVE = "experiments/bill_target74/r172_independent_audit.py"
EXPECTED_SOURCE_RELATIVES = {
    SOURCE_RELATIVE,
    AUDIT_RELATIVE,
    "experiments/bill_target74/r22_remarks_tfidf.py",
    "experiments/bill_target74/r03_catboost.py",
    "experiments/bill_target74/r04_ablation.py",
    "experiments/bill_target74/r23_word_char_tfidf.py",
    "experiments/bill_target74/runner.py",
    "experiments/bill_greedy/runner.py",
    "experiments/bill_greedy/history.py",
    "experiments/bill_greedy/comparables.py",
    "data.py",
    "model_candidate_v2/features.py",
    "model_candidate_v2/winner_config.json",
    "model_candidate_v2/models/frozen_list_aware.json",
    "experiments/model_optimization_v2/learners.py",
    "experiments/model_optimization_v2/pipeline.py",
    "experiments/model_optimization_v2/folds.py",
    "experiments/model_optimization_v2/worker.py",
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261172


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _metric(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    _require(np.isfinite(actual).all() and (actual > 0).all(), "invalid sale prices in paired output")
    _require(np.isfinite(predicted).all() and (predicted > 0).all(), "invalid predictions in paired output")
    ape = np.abs(predicted - actual) / actual * 100.0
    return {"mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def _same_metric(saved: dict, fresh: dict, name: str) -> None:
    for key in ("mdape_percent", "mape_percent"):
        _require(
            np.isclose(float(saved[key]), fresh[key], rtol=0.0, atol=1e-10),
            f"{name} {key} differs: saved={saved[key]}, recomputed={fresh[key]}",
        )


def audit(run_id: str) -> dict:
    from experiments.bill_target74 import runner as campaign

    campaign_root = ROOT / "outputs" / "bill_target74" / CAMPAIGN_ID
    protocol_path = ROOT / "experiments" / "bill_target74" / "campaign_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _require(protocol.get("campaign_id") == CAMPAIGN_ID, "wrong campaign protocol")
    _require(_sha256(campaign.DATA) == protocol.get("data_sha256"),
             "active dataset hash differs from the registered campaign")
    card = protocol.get("cards", {}).get(CARD)
    _require(card is not None, "R172 card is missing from campaign protocol")
    _require(card.get("target_override") == "log_sold_over_list", "registered target is not log ratio")
    _require(
        card.get("model_config_overrides", {}).get(TRACK)
        == {"objective": "fair", "fair_c": 1.0},
        "registered LightGBM override is not Fair loss with fair_c=1.0",
    )

    output = campaign_root / "batches" / run_id
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_path = output / ("screen_results.json" if run_id.startswith("r172s_") else "development_results.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    paired_path = output / "paired_predictions.csv"
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    stage = result.get("stage")
    _require(stage in {"screen", "development"}, f"unsupported R172 stage {stage!r}")
    _require(result.get("campaign_id") == CAMPAIGN_ID, "result campaign mismatch")
    _require(result.get("run_id") == run_id == manifest.get("run_id"), "run ID mismatch")
    _require(result.get("card") == CARD and manifest.get("parent_card") == PARENT_CARD, "card/parent mismatch")
    _require(result.get("parent_run_id") == card.get("parent_run_id"), "parent run mismatch")
    _require(result.get("target_override") == "log_sold_over_list", "result target mismatch")
    _require(result.get("data_sha256") == protocol.get("data_sha256"), "result data hash mismatch")
    _require(manifest.get("source_identity", {}).get("data_sha256") == protocol.get("data_sha256"),
             "manifest data hash mismatch")
    _require(manifest.get("source_identity", {}).get("branch") == "bill", "manifest branch mismatch")
    _require(manifest.get("source_identity", {}).get("commit") == protocol.get("parent_commit"),
             "manifest commit mismatch")
    _require(result.get("bootstrap_seed") == BOOTSTRAP_SEED, "result bootstrap seed mismatch")
    _require(
        result.get("model_config_overrides", {}).get(TRACK)
        == {"objective": "fair", "fair_c": 1.0},
        "result model override mismatch",
    )
    for artifact in (result_path, paired_path):
        _require(artifact.is_file(), f"missing artifact: {artifact}")
    _require(manifest.get("result_sha256") == _sha256(result_path), "result hash mismatch")
    _require(manifest.get("paired_predictions_sha256") == _sha256(paired_path), "paired prediction hash mismatch")
    _require(manifest.get("target_override") == "log_sold_over_list", "manifest target mismatch")
    _require(
        manifest.get("model_config_overrides", {}).get(TRACK)
        == {"objective": "fair", "fair_c": 1.0},
        "manifest model override mismatch",
    )
    _require(manifest.get("parent_artifact_hashes") == card.get("parent_artifact_hashes"),
             "registered R35 artifact hashes changed")

    # Recheck both live sources and archived source snapshots against the manifest.
    code_hashes = manifest.get("code_sha256", {})
    _require(set(code_hashes) == EXPECTED_SOURCE_RELATIVES,
             "manifest source list is incomplete or contains unexpected files")
    for relative, digest in code_hashes.items():
        source_path = ROOT / Path(relative)
        _require(source_path.is_file(), f"missing registered source {relative}")
        _require(_sha256(source_path) == digest, f"source hash changed: {relative}")
        snapshot_name = manifest.get("source_snapshots", {}).get(relative)
        _require(snapshot_name is not None, f"source snapshot missing from manifest: {relative}")
        snapshot_path = output / "provenance" / snapshot_name
        _require(snapshot_path.is_file() and _sha256(snapshot_path) == digest,
                 f"source snapshot hash mismatch: {relative}")
    _require(manifest["code_sha256"].get(SOURCE_RELATIVE) == _sha256(ROOT / SOURCE_RELATIVE),
             "R172 source hash mismatch")
    _require(manifest["code_sha256"].get(AUDIT_RELATIVE) == _sha256(ROOT / AUDIT_RELATIVE),
             "auditor source hash mismatch")

    months = protocol["screen_months"] if stage == "screen" else protocol["development_months"]
    _require(result.get("months") == months, f"unexpected {stage} month set")
    _require(set(paired.month.astype(str)) == set(months), "paired rows contain unexpected months")
    _require(paired.track.eq(TRACK).all(), "paired rows contain another track")
    _require(not paired.duplicated(["month", "track", "ml_num"]).any(), "duplicate prediction keys")
    _require(paired.detached_route.notna().all(), "route flag missing")
    for flag in ("reserved_months_labels_opened", "hidden_labels_opened"):
        _require(result.get(flag) is False and manifest.get(flag) is False, f"{flag} is not false")
    _require(result.get("reserve_validation_eligible") is False, "reserve validation must remain ineligible")
    _require(manifest.get("reserve_validation_eligible") is False,
             "manifest must keep reserve validation ineligible")
    _require(not set(months).intersection({"2025-05", "2025-08"}), "restricted month present")
    safe_inputs = manifest.get("safe_input_artifacts", {})
    _require(set(safe_inputs) == set(months), "safe input artifacts do not exactly match run months")
    for month, info in safe_inputs.items():
        snapshot = output / "provenance" / info["snapshot"]
        _require(snapshot.is_file() and _sha256(snapshot) == info.get("sha256"),
                 f"frozen HGBR snapshot hash mismatch for {month}")
        _require(info.get("original_path", "").endswith(f"/frozen_hgbr/{TRACK}/{month}.csv"),
                 f"unexpected frozen HGBR source path for {month}")

    parent_result_rel = (
        card["result_path"] if stage == "screen" else card["development_result_path"]
    )
    parent_result_path = ROOT / Path(parent_result_rel)
    parent_rows_path = parent_result_path.parent / "paired_predictions.csv"
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    parent_manifest_path = parent_result_path.parent / "run_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    expected_parent_result_run_id = (
        card.get("parent_run_id") if stage == "screen" else card.get("parent_development_run_id")
    )
    _require(parent_result.get("run_id") == expected_parent_result_run_id,
             "R35 parent result/run mismatch")
    _require(parent_result.get("data_sha256") == protocol.get("data_sha256"),
             "R35 parent result data hash mismatch")
    _require(parent_manifest.get("source_identity", {}).get("data_sha256") == protocol.get("data_sha256"),
             "R35 parent manifest data hash mismatch")
    _require(parent_manifest.get("result_sha256") == _sha256(parent_result_path),
             "R35 parent result hash differs from its manifest")
    _require(parent_manifest.get("paired_predictions_sha256") == _sha256(parent_rows_path),
             "R35 parent prediction hash differs from its manifest")
    if stage == "screen":
        expected_hashes = {
            "screen_result_sha256": _sha256(parent_result_path),
            "screen_predictions_sha256": _sha256(parent_rows_path),
            "screen_manifest_sha256": _sha256(parent_manifest_path),
        }
    else:
        expected_hashes = {
            "development_result_sha256": _sha256(parent_result_path),
            "development_predictions_sha256": _sha256(parent_rows_path),
            "development_manifest_sha256": _sha256(parent_manifest_path),
        }
    for key, digest in expected_hashes.items():
        _require(card["parent_artifact_hashes"].get(key) == digest, f"R35 registered parent hash mismatch: {key}")
    parent_rows = pd.read_csv(parent_rows_path, dtype={"ml_num": str})
    parent_rows = parent_rows.loc[parent_rows.track.eq(TRACK)].copy()
    parent_rows["month"] = parent_rows.month.astype(str)
    paired["month"] = paired.month.astype(str)
    paired["detached_route"] = paired.detached_route.astype(bool)

    metrics_checked = {}
    for month in months:
        candidate_month = paired.loc[paired.month.eq(month)].copy()
        parent_month = parent_rows.loc[parent_rows.month.eq(month)].copy()
        _require(not parent_month.duplicated("ml_num").any(), f"duplicate R35 parent IDs in {month}")
        _require(len(candidate_month) == len(parent_month)
                 and set(candidate_month.ml_num.astype(str)) == set(parent_month.ml_num.astype(str)),
                 f"candidate row set differs from R35 parent in {month}")
        parent_month = parent_month.set_index(parent_month.ml_num.astype(str))
        aligned = parent_month.reindex(candidate_month.ml_num.astype(str))
        _require(not aligned.candidate_prediction.isna().any(), f"R35 rows missing for {month}")
        _require(
            np.array_equal(candidate_month.sold_price.to_numpy(float), aligned.sold_price.to_numpy(float)),
            f"sale labels differ from R35 in {month}",
        )
        _require(
            np.array_equal(candidate_month.parent_prediction.to_numpy(float),
                           aligned.candidate_prediction.to_numpy(float)),
            f"paired parent predictions differ from R35 in {month}",
        )
        route = candidate_month.detached_route.to_numpy(bool)
        _require(route.any(), f"no Detached homes in {month}")
        _require(
            np.array_equal(
                candidate_month.loc[~route, "candidate_prediction"].to_numpy(float),
                candidate_month.loc[~route, "parent_prediction"].to_numpy(float),
            ),
            f"non-Detached predictions changed in {month}",
        )

        component_path = output / f"detached_components_{month}.csv"
        _require(component_path.is_file(), f"missing Detached component audit rows for {month}")
        components = pd.read_csv(component_path, dtype={"ml_num": str}, float_precision="round_trip")
        route_rows = candidate_month.loc[route].copy()
        _require(not components.ml_num.astype(str).duplicated().any(), f"duplicate component IDs in {month}")
        _require(set(components.ml_num.astype(str)) == set(route_rows.ml_num.astype(str)),
                 f"Detached component IDs differ from paired route rows in {month}")
        components = components.set_index(components.ml_num.astype(str)).reindex(route_rows.ml_num.astype(str))
        _require(
            np.array_equal(components.sold_price.to_numpy(float), route_rows.sold_price.to_numpy(float)),
            f"component labels are misaligned in {month}",
        )
        listed = components.list_price.to_numpy(float)
        core = components.core_price_prediction.to_numpy(float)
        ratio = components.core_ratio_prediction.to_numpy(float)
        frozen = components.frozen_hgbr_prediction.to_numpy(float)
        _require(np.isfinite(listed).all() and (listed > 0).all(), f"invalid list prices in {month}")
        _require(np.isfinite(ratio).all() and (ratio > 0).all(), f"invalid ratio predictions in {month}")
        _require(np.isfinite(core).all() and (core > 0).all(), f"invalid core prices in {month}")
        _require(np.allclose(core / listed, ratio, rtol=0.0, atol=1e-12),
                 f"ratio-to-price arithmetic mismatch in {month}")
        cache_info = safe_inputs[month]
        cache = pd.read_csv(output / "provenance" / cache_info["snapshot"],
                            dtype={"ml_num": str}, float_precision="round_trip")
        _require(np.array_equal(cache.ml_num.astype(str).to_numpy(), candidate_month.ml_num.astype(str).to_numpy()),
                 f"frozen HGBR cache IDs differ from full evaluation rows in {month}")
        cache = cache.set_index(cache.ml_num.astype(str)).reindex(route_rows.ml_num.astype(str))
        _require(np.allclose(cache.raw_prediction.to_numpy(float), frozen, rtol=0.0, atol=1e-10),
                 f"frozen HGBR component values differ from preserved cache in {month}")
        expected_blend = np.round(0.25 * np.round(frozen) + 0.75 * core)
        _require(np.array_equal(expected_blend, components.blended_detached_prediction.to_numpy(float)),
                 f"R35 75/25 blend arithmetic mismatch in {month}")
        _require(
            np.array_equal(components.parent_prediction.to_numpy(float),
                           route_rows.parent_prediction.to_numpy(float)),
            f"component parent predictions are misaligned in {month}",
        )
        _require(
            np.array_equal(components.candidate_prediction.to_numpy(float),
                           route_rows.candidate_prediction.to_numpy(float)),
            f"Detached candidate predictions differ from paired output in {month}",
        )
        _require(
            np.array_equal(components.blended_detached_prediction.to_numpy(float),
                           components.candidate_prediction.to_numpy(float)),
            f"Detached candidate is not the registered blend in {month}",
        )

        fit_detail = next((item for item in result.get("fit_details", [])
                           if str(item.get("month")) == month), None)
        _require(fit_detail is not None, f"fit details missing for {month}")
        saved_prediction = fit_detail.get("saved_predictions", {})
        component_prediction_path = output / saved_prediction.get("path", "")
        _require(component_prediction_path.is_file(), f"saved core predictions missing for {month}")
        _require(saved_prediction.get("sha256") == _sha256(component_prediction_path),
                 f"saved core prediction hash mismatch for {month}")
        core_rows = pd.read_csv(component_prediction_path, dtype={"ml_num": str}, float_precision="round_trip")
        _require(np.array_equal(core_rows.ml_num.astype(str).to_numpy(), components.ml_num.astype(str).to_numpy()),
                 f"core prediction IDs differ from component rows in {month}")
        _require(np.allclose(core_rows.predicted_price.to_numpy(float), core, rtol=0.0, atol=1e-10),
                 f"core predictions differ from saved fit predictions in {month}")
        prediction_digest = hashlib.sha256(core_rows.predicted_price.to_numpy(dtype="float64").tobytes()).hexdigest()
        _require(fit_detail.get("prediction_sha256") == prediction_digest,
                 f"saved fit prediction digest mismatch in {month}")

        score = result["tracks"][TRACK]
        fresh_parent = _metric(candidate_month.sold_price.to_numpy(float),
                               candidate_month.parent_prediction.to_numpy(float))
        fresh_candidate = _metric(candidate_month.sold_price.to_numpy(float),
                                  candidate_month.candidate_prediction.to_numpy(float))
        _require(month in score.get("monthly", {}), f"saved monthly metrics missing {month}")
        _same_metric(score["monthly"][month]["parent"], fresh_parent, f"{month} parent")
        _same_metric(score["monthly"][month]["candidate"], fresh_candidate, f"{month} candidate")
        monthly_gain = 100.0 * (fresh_parent["mdape_percent"] - fresh_candidate["mdape_percent"]) / fresh_parent["mdape_percent"]
        _require(np.isclose(score["monthly"][month]["relative_mdape_improvement_percent"],
                            monthly_gain, rtol=0.0, atol=1e-10),
                 f"{month} relative MdAPE gain mismatch")
        metrics_checked[month] = {
            "rows": int(len(candidate_month)),
            "detached_rows": int(route.sum()),
            "parent_mdape": fresh_parent["mdape_percent"],
            "candidate_mdape": fresh_candidate["mdape_percent"],
        }

    score = result["tracks"][TRACK]
    fresh_parent = _metric(paired.sold_price.to_numpy(float), paired.parent_prediction.to_numpy(float))
    fresh_candidate = _metric(paired.sold_price.to_numpy(float), paired.candidate_prediction.to_numpy(float))
    _same_metric(score["parent"], fresh_parent, "pooled parent")
    _same_metric(score["candidate"], fresh_candidate, "pooled candidate")
    actual = paired.sold_price.to_numpy(float)
    parent_prediction = paired.parent_prediction.to_numpy(float)
    candidate_prediction = paired.candidate_prediction.to_numpy(float)
    parent_ape = np.abs(parent_prediction - actual) / actual * 100.0
    candidate_ape = np.abs(candidate_prediction - actual) / actual * 100.0
    relative_gain = 100.0 * (fresh_parent["mdape_percent"] - fresh_candidate["mdape_percent"]) / fresh_parent["mdape_percent"]
    mape_change = 100.0 * (fresh_candidate["mape_percent"] - fresh_parent["mape_percent"]) / fresh_parent["mape_percent"]
    monthly_gains = {}
    grouped = []
    for month in months:
        indices = np.flatnonzero(paired.month.eq(month).to_numpy())
        grouped.append(indices)
        parent_month = _metric(actual[indices], parent_prediction[indices])
        candidate_month = _metric(actual[indices], candidate_prediction[indices])
        monthly_gains[month] = 100.0 * (parent_month["mdape_percent"] - candidate_month["mdape_percent"]) / parent_month["mdape_percent"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_parent = []
        sampled_candidate = []
        for indices in grouped:
            picked = indices[rng.integers(0, len(indices), size=len(indices))]
            sampled_parent.append(parent_ape[picked])
            sampled_candidate.append(candidate_ape[picked])
        differences[replicate] = float(np.median(np.concatenate(sampled_candidate))
                                       - np.median(np.concatenate(sampled_parent)))
    lower95, upper95 = (float(value) for value in np.quantile(differences, [0.025, 0.975]))
    _require(np.isclose(score["relative_mdape_improvement_percent"], relative_gain, rtol=0.0, atol=1e-10),
             "relative MdAPE gain mismatch")
    _require(np.isclose(score["mape_relative_change_percent"], mape_change, rtol=0.0, atol=1e-10),
             "relative MAPE change mismatch")
    _require(score.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES, "bootstrap replicate count mismatch")
    _require(np.isclose(score["bootstrap_mdape_difference_percent"]["lower95"], lower95, rtol=0.0, atol=1e-10)
             and np.isclose(score["bootstrap_mdape_difference_percent"]["upper95"], upper95, rtol=0.0, atol=1e-10),
             "paired-bootstrap interval mismatch")
    months_improved = sum(gain > 0 for gain in monthly_gains.values())
    worst_regression = max(0.0, -min(monthly_gains.values()))
    gates = protocol["gates"]
    if stage == "screen":
        expected_gate_checks = {
            "relative_mdape_gain": relative_gain >= gates["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": months_improved == len(months),
            "mape_guard": mape_change <= gates["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": upper95 < 0.0,
        }
        expected_pass = all(expected_gate_checks.values())
        _require(score.get("qualifies_development") is expected_pass,
                 "screen score qualification differs from independently recomputed gates")
    else:
        development_gates_pass = bool(
            relative_gain >= gates["development_relative_mdape_improvement_percent"]
            and months_improved >= gates["development_months_improved_min"]
            and worst_regression <= gates["development_worst_month_regression_max_percent"]
            and mape_change <= gates["development_mape_regression_max_percent"]
            and upper95 < 0.0
        )
        expected_gate_checks = {
            "development_gates": development_gates_pass,
            "strict_mdape_target": fresh_candidate["mdape_percent"] < 4.0,
        }
        _require(score.get("qualifies_development") is development_gates_pass,
                 "development score qualification differs from recomputed gates")
        _require(score.get("target_met") is expected_gate_checks["strict_mdape_target"],
                 "strict target flag differs from recomputed MdAPE")
        expected_pass = development_gates_pass and expected_gate_checks["strict_mdape_target"]
    _require(result.get("gate_checks") == expected_gate_checks, "saved gate checks differ from independent recomputation")
    _require(result.get("qualifies_by_track", {}).get(TRACK) is expected_pass,
             "saved screen/development decision differs from independent recomputation")
    fit_details = result.get("fit_details", [])
    _require(len(fit_details) == len(months), "fit detail count does not match scheduled months")
    _require({str(item.get("month")) for item in fit_details} == set(months),
             "fit detail months do not match scheduled months")
    for detail in fit_details:
        _require(detail.get("actual_objective") == "fair", "fit used a non-Fair objective")
        _require(detail.get("target") == "log_sold_over_list", "fit used another target")
        _require(detail.get("fair_c") == 1.0, "fit detail fair_c is not 1.0")
        _require(detail.get("model_config", {}).get("fair_c") == 1.0, "model config fair_c is not 1.0")
        _require(detail.get("model_family") == "lightgbm", "fit family is not LightGBM")
        names = detail.get("feature_columns", [])
        _require("list_price" in names, "list_price feature is absent from fitted columns")
        _require(any(name.startswith("remarks_word_tfidf_") for name in names),
                 "R35 word TF-IDF features are absent")
        _require(not any(name.startswith("remarks_char_tfidf_") for name in names),
                 "R172 unexpectedly added character TF-IDF features")

    ledger_path = campaign_root / "fits.jsonl"
    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    run_events = [item for item in events if item.get("card") == CARD
                  and item.get("stage") == stage and item.get("run_id") == run_id]
    starts = [item for item in run_events if item.get("event") == "fit_start"]
    completes = [item for item in run_events if item.get("event") == "fit_complete"]
    _require(len(starts) == len(months), f"fit_start ledger count mismatch: {len(starts)}")
    _require(len(completes) == len(months), f"fit_complete ledger count mismatch: {len(completes)}")
    _require(all(item.get("status") == "complete" for item in completes), "one or more fits failed")
    _require({item.get("month") for item in starts} == set(months), "fit ledger month set mismatch")
    _require({item.get("fit_id") for item in starts} == {item.get("fit_id") for item in completes},
             "fit starts and completions do not pair by fit ID")
    _require({item.get("fit_id") for item in starts} == {item.get("fit_id") for item in fit_details},
             "fit details do not match ledger fit IDs")
    completion_by_id = {item.get("fit_id"): item for item in completes}
    detail_by_id = {item.get("fit_id"): item for item in fit_details}
    for fit_id, detail in detail_by_id.items():
        completion = completion_by_id[fit_id]
        _require(completion.get("month") == detail.get("month"), f"fit completion month mismatch: {fit_id}")
        _require(completion.get("actual_objective") == detail.get("actual_objective") == "fair",
                 f"fit completion objective mismatch: {fit_id}")
        _require(completion.get("target") == detail.get("target") == "log_sold_over_list",
                 f"fit completion target mismatch: {fit_id}")
        _require(completion.get("fair_c") == detail.get("fair_c") == 1.0,
                 f"fit completion fair_c mismatch: {fit_id}")
        _require(completion.get("model_family") == detail.get("model_family") == "lightgbm",
                 f"fit completion model family mismatch: {fit_id}")
        _require(completion.get("model_config") == detail.get("model_config"),
                 f"fit completion model config mismatch: {fit_id}")
        _require(completion.get("saved_predictions") == detail.get("saved_predictions"),
                 f"fit completion prediction artifact mismatch: {fit_id}")
    _require(all(item.get("expected_objective") == "fair" and item.get("expected_fair_c") == 1.0
                 and item.get("target_override") == "log_sold_over_list" for item in starts),
             "fit-start ledger lacks the expected target/objective/fair_c")

    _require(result.get("fit_starts") == len(months), "result fit count mismatch")
    _require(manifest.get("fit_starts") == len(months), "manifest fit count mismatch")
    _require(manifest.get("fit_details") == fit_details, "manifest fit details differ from result")
    _require(result.get("reserved_months_labels_opened") is False, "reserved labels opened flag is not false")
    summary = {
        "status": "passed",
        "run_id": run_id,
        "stage": stage,
        "rows": int(len(paired)),
        "fit_starts": len(starts),
        "objective": "fair",
        "fair_c": 1.0,
        "target": "log_sold_over_list",
        "pooled_parent_mdape_percent": fresh_parent["mdape_percent"],
        "pooled_candidate_mdape_percent": fresh_candidate["mdape_percent"],
        "months": metrics_checked,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    (output / "independent_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit saved R172 artifacts")
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(audit(args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
