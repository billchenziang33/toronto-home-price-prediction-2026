"""Independent audit for R156 saved screen/development artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


CARD = "R156_listaware_detached_listprice_median_experts"
PARENT_CARD = "R35_listaware_detached_r22_text_expert"
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
BANDS = ("low", "high")
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
RESERVED_MONTHS = set(campaign.PROTOCOL["reserved_months"])
MIN_TRAINING_ROWS_PER_BAND = 1000
MIN_EVALUATION_ROWS_PER_BAND = 100
SPECIALIST_WEIGHT = 0.75
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261156


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _same(actual, expected, path: str = "root") -> None:
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), f"{path}: key mismatch")
        for key in expected:
            _same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), f"{path}: length mismatch")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _same(left, right, f"{path}[{index}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        _require(isinstance(actual, (float, int)) and np.isclose(float(actual), float(expected), rtol=1e-12,
                                                                 atol=1e-12),
                 f"{path}: {actual!r} != {expected!r}")
    else:
        _require(actual == expected, f"{path}: {actual!r} != {expected!r}")


def _parent() -> dict:
    card = campaign.PROTOCOL["cards"].get(PARENT_CARD)
    _require(bool(card) and card.get("status") == "fixed_origin_confirmation_passed",
             "R35 is no longer the validated list-aware parent")
    result_path = campaign.ROOT / card["result_path"]
    rows_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    development_path = campaign.ROOT / card["development_result_path"]
    development_rows_path = development_path.parent / "paired_predictions.csv"
    development_manifest_path = development_path.parent / "run_manifest.json"
    for path in (result_path, rows_path, manifest_path, development_path,
                 development_rows_path, development_manifest_path):
        _require(path.is_file(), f"missing R35 artifact {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    _require(result.get("run_id") == card.get("run_id") == manifest.get("run_id"), "R35 screen run mismatch")
    _require(development.get("run_id") == card.get("development_run_id") == development_manifest.get("run_id"),
             "R35 development run mismatch")
    _require(manifest.get("result_sha256") == _sha256(result_path), "R35 screen result hash mismatch")
    _require(manifest.get("paired_predictions_sha256") == _sha256(rows_path), "R35 screen rows hash mismatch")
    _require(development_manifest.get("result_sha256") == _sha256(development_path),
             "R35 development result hash mismatch")
    _require(development_manifest.get("paired_predictions_sha256") == _sha256(development_rows_path),
             "R35 development rows hash mismatch")
    for saved in (result, development):
        _require(saved.get("reserved_months_labels_opened") is False
                 and saved.get("hidden_labels_opened") is False, "R35 parent reports restricted-label access")
    return {
        "card": card,
        "screen_result": result_path,
        "screen_rows": rows_path,
        "screen_manifest": manifest_path,
        "development_result": development_path,
        "development_rows": development_rows_path,
        "development_manifest": development_manifest_path,
    }


def _parent_hashes(parent: dict) -> dict[str, str]:
    return {
        "screen_result_sha256": _sha256(parent["screen_result"]),
        "screen_predictions_sha256": _sha256(parent["screen_rows"]),
        "screen_manifest_sha256": _sha256(parent["screen_manifest"]),
        "development_result_sha256": _sha256(parent["development_result"]),
        "development_predictions_sha256": _sha256(parent["development_rows"]),
        "development_manifest_sha256": _sha256(parent["development_manifest"]),
    }


def _train_eval_band(train: pd.DataFrame, homes: pd.DataFrame, month: str):
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy

    cutoff = pd.Period(month, freq="M").start_time
    dates = pd.to_datetime(train.date_sold, errors="raise")
    _require(not (dates >= cutoff).any(), f"training rows include {month} or future sales")
    _require(not dates.dt.to_period("M").astype(str).isin(RESERVED_MONTHS).any(),
             f"training rows contain reserved-month sales for {month}")
    safe_homes = hide_answers(homes, TRACK).reset_index(drop=True)
    train_detached = train.loc[greedy._specialist_mask(train, HOUSE_TYPE)].copy()
    eval_detached = safe_homes.loc[greedy._specialist_mask(safe_homes, HOUSE_TYPE)].copy()
    train_price = pd.to_numeric(train_detached.list_price, errors="raise").to_numpy(float)
    eval_price = pd.to_numeric(eval_detached.list_price, errors="raise").to_numpy(float)
    _require(np.isfinite(train_price).all() and (train_price > 0).all(), "invalid training list prices")
    _require(np.isfinite(eval_price).all() and (eval_price > 0).all(), "invalid evaluation list prices")
    threshold = float(np.median(train_price))
    train_band = np.where(train_price <= threshold, "low", "high")
    eval_band = np.where(eval_price <= threshold, "low", "high")
    return safe_homes, train_detached, eval_detached, threshold, train_band, eval_band


def _audit(run_id: str) -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as cache_helpers
    from experiments.bill_target74 import r04_ablation as identity_helpers
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    _require(run_id.startswith(("r156s_", "r156d_")), "unexpected R156 run ID")
    stage = "screen" if run_id.startswith("r156s_") else "development"
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    root = campaign.CAMPAIGN_ROOT / "batches" / run_id
    manifest_path = root / "run_manifest.json"
    result_path = root / ("screen_results.json" if stage == "screen" else "development_results.json")
    paired_path = root / "paired_predictions.csv"
    _require(all(path.is_file() for path in (manifest_path, result_path, paired_path)),
             "R156 run result, predictions, or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    card = campaign.PROTOCOL["cards"].get(CARD)
    _require(bool(card), "R156 registration card is missing")
    _require(result.get("campaign_id") == campaign.PROTOCOL["campaign_id"], "campaign ID mismatch")
    _require(result.get("run_id") == manifest.get("run_id") == run_id, "run ID mismatch")
    _require(result.get("card") == CARD and result.get("stage") == stage, "result card/stage mismatch")
    _require(result.get("months") == list(months) and result.get("bands") == list(BANDS),
             "run month or band set mismatch")
    _require(result.get("target") == "log_sold_over_list", "candidate target mismatch")
    _require(result.get("model_family") == "LightGBM", "candidate family mismatch")
    _require(result.get("bootstrap_seed") == BOOTSTRAP_SEED, "bootstrap seed mismatch")
    _require(result.get("data_sha256") == campaign.PROTOCOL["data_sha256"], "result data hash mismatch")
    _require(manifest.get("result_sha256") == _sha256(result_path), "result hash mismatch")
    _require(manifest.get("paired_predictions_sha256") == _sha256(paired_path), "paired predictions hash mismatch")
    _require(manifest.get("parent_card") == PARENT_CARD, "manifest parent mismatch")
    _require(manifest.get("parent_artifact_hashes") == card.get("parent_artifact_hashes"),
             "manifest parent artifact hashes differ from registered card")
    _require(manifest.get("fit_starts") == len(months) * len(BANDS) == result.get("fit_starts"),
             "fit count mismatch")
    for flags in (result, manifest):
        _require(flags.get("reserved_months_labels_opened") is False
                 and flags.get("hidden_labels_opened") is False, "R156 reports restricted-label access")
        _require(flags.get("reserve_validation_eligible") is False, "R156 reserve validation must stay ineligible")

    identity = identity_helpers._source_identity()
    _require(identity.get("data_sha256") == campaign.PROTOCOL["data_sha256"], "live data hash mismatch")
    _require(manifest.get("source_identity") == identity, "source identity changed since run")
    code_hashes = manifest.get("code_sha256", {})
    _require(code_hashes == card.get("code_sha256"), "manifest code hashes differ from preregistration")
    for relative, expected_hash in code_hashes.items():
        source_path = campaign.ROOT / relative
        _require(source_path.is_file() and _sha256(source_path) == expected_hash,
                 f"current source hash mismatch: {relative}")
        snapshot = root / "provenance" / manifest.get("source_snapshots", {}).get(relative, "")
        _require(snapshot.is_file() and _sha256(snapshot) == expected_hash,
                 f"source snapshot hash mismatch: {relative}")
    parent = _parent()
    _require(_parent_hashes(parent) == card.get("parent_artifact_hashes"), "live R35 parent changed")

    safe_inputs = manifest.get("safe_input_artifacts", {})
    _require(set(safe_inputs) == set(months), "safe HGBR snapshots do not match run months")
    for month, info in safe_inputs.items():
        snapshot = root / "provenance" / info["snapshot"]
        original = campaign.ROOT / info["original_path"]
        _require(snapshot.is_file() and _sha256(snapshot) == info.get("sha256"),
                 f"frozen HGBR snapshot hash mismatch for {month}")
        _require(original.is_file() and _sha256(original) == info.get("sha256"),
                 f"live frozen HGBR cache hash mismatch for {month}")
        _require(info.get("original_path", "").endswith(f"/frozen_hgbr/{TRACK}/{month}.csv"),
                 f"unexpected cache source for {month}")

    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "parent_prediction", "candidate_prediction",
                "detached_route", "price_band"}
    _require(required.issubset(paired.columns), "paired predictions lack required fields")
    paired["month"] = paired.month.astype(str)
    _require(set(paired.month) == set(months) and paired.track.eq(TRACK).all(), "paired month/track mismatch")
    _require(not paired.duplicated(["month", "track", "ml_num"]).any(), "duplicate paired prediction key")
    _require(np.isfinite(paired[["sold_price", "parent_prediction", "candidate_prediction"]].to_numpy(float)).all(),
             "non-finite paired value")
    _require((paired[["sold_price", "parent_prediction", "candidate_prediction"]].to_numpy(float) > 0).all(),
             "non-positive paired value")

    parent_rows_all = pd.read_csv(parent["screen_rows"] if stage == "screen" else parent["development_rows"],
                                  dtype={"ml_num": str})
    parent_rows_all = parent_rows_all.loc[parent_rows_all.track.eq(TRACK)].copy()
    if stage == "screen":
        expected_fit_count = len(months) * len(BANDS)
    else:
        expected_fit_count = len(months) * len(BANDS)
    details = result.get("fit_details", [])
    _require(len(details) == expected_fit_count, "fit detail count mismatch")
    detail_map = {(str(x["month"]), str(x["price_band"])): x for x in details}
    _require(set(detail_map) == {(month, band) for month in months for band in BANDS},
             "fit detail fold/band coverage mismatch")
    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [x for x in ledger if x.get("event") == "fit_start" and x.get("card") == CARD
              and x.get("run_id") == run_id]
    completions = [x for x in ledger if x.get("event") == "fit_complete" and x.get("card") == CARD
                   and x.get("run_id") == run_id]
    _require(len(starts) == len(completions) == expected_fit_count, "fit ledger starts/completions mismatch")
    _require(all(x.get("status") == "complete" for x in completions), "a registered fit failed")
    _require(all(x.get("reserved_months_labels_opened") is False and x.get("hidden_labels_opened") is False
                 for x in starts + completions), "fit ledger reports restricted-label access")
    start_ids = {x.get("fit_id") for x in starts}
    complete_ids = {x.get("fit_id") for x in completions}
    _require(start_ids == complete_ids == {x["fit_id"] for x in details}, "fit IDs do not reconcile")

    checked_months = {}
    for month in months:
        _, train, _, truth, heldout = greedy._load_fold(month)
        safe_homes = hide_answers(heldout, TRACK).reset_index(drop=True)
        safe_homes, train_detached, eval_detached, threshold, train_band, eval_band = _train_eval_band(
            train, heldout, month,
        )
        ids = safe_homes.ml_num.astype(str).to_numpy()
        month_parent = parent_rows_all.loc[parent_rows_all.month.astype(str).eq(month)].copy()
        month_parent = month_parent.set_index(month_parent.ml_num.astype(str), drop=False)
        _require(len(month_parent) == len(safe_homes) and set(month_parent.index) == set(ids),
                 f"R35 parent IDs mismatch for {month}")
        month_parent = month_parent.loc[ids]
        _require(np.array_equal(month_parent.ml_num.astype(str).to_numpy(), ids),
                 f"R35 parent order mismatch for {month}")
        _require(np.array_equal(month_parent.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)),
                 f"R35 labels mismatch for {month}")
        paired_month = paired.loc[paired.month.eq(month)].copy()
        paired_month = paired_month.set_index(paired_month.ml_num.astype(str), drop=False).loc[ids]
        parent_values = month_parent.candidate_prediction.to_numpy(float)
        _require(np.array_equal(paired_month.parent_prediction.to_numpy(float), parent_values),
                 f"paired parent predictions mismatch for {month}")
        _require(np.array_equal(paired_month.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)),
                 f"paired labels mismatch for {month}")
        detached_mask = np.asarray(greedy._specialist_mask(safe_homes, HOUSE_TYPE), dtype=bool)
        _require(np.array_equal(paired_month.detached_route.to_numpy(bool), detached_mask),
                 f"Detached route mask mismatch for {month}")
        expected_full_band = np.where(detached_mask, np.where(
            pd.to_numeric(safe_homes.list_price, errors="raise").to_numpy(float) <= threshold,
            "low", "high"), "non_detached")
        _require(np.array_equal(paired_month.price_band.astype(str).to_numpy(), expected_full_band),
                 f"held-out band assignment mismatch for {month}")

        component_path = root / f"detached_components_{month}.csv"
        _require(component_path.is_file(), f"missing component predictions for {month}")
        component = pd.read_csv(component_path, dtype={"ml_num": str})
        _require(set(component.ml_num.astype(str)) == set(eval_detached.ml_num.astype(str)),
                 f"component IDs do not cover Detached rows for {month}")
        component = component.set_index(component.ml_num.astype(str), drop=False).loc[
            eval_detached.ml_num.astype(str).to_numpy()
        ]
        eval_price = pd.to_numeric(eval_detached.list_price, errors="raise").to_numpy(float)
        expected_eval_band = np.where(eval_price <= threshold, "low", "high")
        _require(np.array_equal(component.price_band.astype(str).to_numpy(), expected_eval_band),
                 f"component band mismatch for {month}")
        _require(np.allclose(component.list_price.to_numpy(float), eval_price, rtol=0, atol=0),
                 f"component list price mismatch for {month}")
        _require(np.allclose(component.list_price_median_threshold.to_numpy(float), threshold, rtol=0, atol=0),
                 f"component threshold mismatch for {month}")
        _require(np.array_equal(component.parent_prediction.to_numpy(float), parent_values[detached_mask]),
                 f"component parent predictions mismatch for {month}")
        _require(np.array_equal(component.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)[detached_mask]),
                 f"component labels mismatch for {month}")

        cache_info = safe_inputs[month]
        cache = pd.read_csv(root / "provenance" / cache_info["snapshot"], dtype={"ml_num": str})
        _require(np.array_equal(cache.ml_num.astype(str).to_numpy(), ids), f"HGBR cache IDs mismatch for {month}")
        cache_predictions = cache.set_index(cache.ml_num.astype(str)).loc[
            eval_detached.ml_num.astype(str), "raw_prediction"
        ].to_numpy(float)
        _require(np.array_equal(component.frozen_hgbr_prediction.to_numpy(float), cache_predictions),
                 f"frozen HGBR component mismatch for {month}")
        from experiments.model_optimization_v2.pipeline import blend_prices
        rebuilt_blend = blend_prices(component.core_price_prediction.to_numpy(float), cache_predictions,
                                     challenger_weight=SPECIALIST_WEIGHT)
        _require(np.array_equal(component.blended_prediction.to_numpy(float), rebuilt_blend),
                 f"frozen blend mismatch for {month}")
        _require(np.array_equal(paired_month.candidate_prediction.to_numpy(float)[detached_mask], rebuilt_blend),
                 f"Detached candidate predictions mismatch for {month}")
        _require(np.array_equal(paired_month.candidate_prediction.to_numpy(float)[~detached_mask],
                                parent_values[~detached_mask]),
                 f"R156 changed non-Detached predictions for {month}")

        train_price = pd.to_numeric(train_detached.list_price, errors="raise").to_numpy(float)
        _require(float(np.median(train_price)) == threshold, f"training-only median mismatch for {month}")
        for band in BANDS:
            fit_detail = detail_map[(month, band)]
            expected_train = sorted(train_detached.loc[train_band == band, "ml_num"].astype(str).tolist())
            expected_eval = eval_detached.loc[eval_band == band, "ml_num"].astype(str).tolist()
            membership_path = campaign.ROOT / fit_detail["training_membership_path"]
            _require(membership_path.is_file(), f"missing fit membership file for {month}/{band}")
            _require(_sha256(membership_path) == fit_detail["training_membership_sha256"],
                     f"membership hash mismatch for {month}/{band}")
            membership = json.loads(membership_path.read_text(encoding="utf-8"))
            _require(membership.get("training_ids") == expected_train, f"training IDs mismatch for {month}/{band}")
            _require(membership.get("evaluation_ids") == expected_eval, f"evaluation IDs mismatch for {month}/{band}")
            _require(membership.get("threshold") == threshold, f"membership threshold mismatch for {month}/{band}")
            _require(membership.get("cutoff_exclusive") == pd.Period(month, freq="M").start_time.isoformat(),
                     f"membership cutoff mismatch for {month}/{band}")
            _require(membership.get("reserved_months_labels_opened") is False
                     and membership.get("hidden_labels_opened") is False,
                     f"membership record reports restricted access for {month}/{band}")
            _require(fit_detail.get("training_rows") == len(expected_train)
                     and fit_detail.get("evaluation_rows") == len(expected_eval),
                     f"fit support mismatch for {month}/{band}")
            _require(len(expected_train) >= MIN_TRAINING_ROWS_PER_BAND
                     and len(expected_eval) >= MIN_EVALUATION_ROWS_PER_BAND,
                     f"support below minimum for {month}/{band}")
            _require(fit_detail.get("model_family") == "lightgbm"
                     and fit_detail.get("model_config") == card.get("model_config")
                     and fit_detail.get("target") == "log_sold_over_list"
                     and fit_detail.get("actual_objective") == "regression",
                     f"model setup mismatch for {month}/{band}")
        checked_months[month] = {
            "rows": int(len(paired_month)),
            "detached_rows": int(detached_mask.sum()),
            "threshold": threshold,
            "bands": {band: {"training_rows": int(np.count_nonzero(train_band == band)),
                             "evaluation_rows": int(np.count_nonzero(eval_band == band))}
                      for band in BANDS},
            "non_detached_predictions_unchanged": True,
        }

    score_frame = paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
    fresh = r22.score_development_confirmation(
        score_frame, months=months, target_mdape_strictly_below=4.0,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        **({
            "minimum_relative_gain_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "minimum_months_improved": len(months),
            "maximum_worst_month_regression_percent": 0.0,
            "maximum_mape_regression_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        } if stage == "screen" else {})
    )
    _same(result["tracks"][TRACK], fresh, "track_metrics")
    if stage == "screen":
        fresh_gates = {
            "relative_mdape_gain": fresh["relative_mdape_improvement_percent"]
                                   >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": fresh["months_improved"] == len(months),
            "mape_guard": fresh["mape_relative_change_percent"]
                          <= campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": fresh["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
        }
    else:
        fresh_gates = {"development_gates": bool(fresh["qualifies_development"]),
                       "strict_mdape_target": bool(fresh["target_met"])}
    _same(result.get("gate_checks"), fresh_gates, "gate_checks")
    _require(result.get("qualifies_by_track", {}).get(TRACK) == all(fresh_gates.values()),
             "promotion result does not match recomputed gates")

    summary = {
        "status": "passed", "audit_tool_path": _relative(Path(__file__)),
        "audit_tool_sha256": _sha256(Path(__file__)), "run_id": run_id, "stage": stage,
        "rows": int(len(paired)), "fit_starts": len(starts),
        "target": "log_sold_over_list", "model_family": "lightgbm",
        "pooled_parent_mdape_percent": fresh["parent"]["mdape_percent"],
        "pooled_candidate_mdape_percent": fresh["candidate"]["mdape_percent"],
        "relative_mdape_improvement_percent": fresh["relative_mdape_improvement_percent"],
        "months": checked_months,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    (root / "independent_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit an R156 run")
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(_audit(args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
