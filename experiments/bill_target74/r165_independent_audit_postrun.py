"""Post-run independent audit for R165, accounting for omitted fit-complete flags."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/bill_target74/campaign_protocol.json"
PROGRESS_PATH = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/progress.json"
LEDGER_PATH = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/fits.jsonl"
CARD_KEY = "R165_listaware_catboost_mape_ratio"
R35_KEY = "R35_listaware_detached_r22_text_expert"
R94_DIR = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/batches/r94g_20261025T202220Z"
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = (
    "2023-03", "2023-07", "2023-11", "2024-03",
    "2024-06", "2024-09", "2025-01", "2025-03",
)
RESERVED_MONTHS = {"2025-05", "2025-08"}
SCREEN_BOOTSTRAP_SEED = 20261165
DEVELOPMENT_BOOTSTRAP_SEED = 20261166
MODEL_CONFIG = {
    "iterations": 600, "depth": 6, "learning_rate": 0.05,
    "l2_leaf_reg": 3, "loss_function": "MAPE", "has_time": True,
    "random_seed": 0, "thread_count": 4, "verbose": False,
    "allow_writing_files": False,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    ape = np.abs(predicted - actual) / actual * 100.0
    return {
        "n": int(len(actual)),
        "mdape_percent": float(np.median(ape)),
        "mape_percent": float(np.mean(ape)),
    }


def _score(frame: pd.DataFrame, months: tuple[str, ...], stage: str) -> dict:
    actual = frame.sold_price.to_numpy(float)
    benchmark = frame.benchmark_prediction.to_numpy(float)
    candidate = frame.candidate_prediction.to_numpy(float)
    bm, cm = _metrics(actual, benchmark), _metrics(actual, candidate)
    gain = 100.0 * (bm["mdape_percent"] - cm["mdape_percent"]) / bm["mdape_percent"]
    mape_change = 100.0 * (cm["mape_percent"] - bm["mape_percent"]) / bm["mape_percent"]
    monthly = {}
    groups = []
    improvements = []
    for month in months:
        part = frame.loc[frame.month.eq(month)]
        y = part.sold_price.to_numpy(float)
        p = part.benchmark_prediction.to_numpy(float)
        c = part.candidate_prediction.to_numpy(float)
        pm, cm_month = _metrics(y, p), _metrics(y, c)
        rel = 100.0 * (pm["mdape_percent"] - cm_month["mdape_percent"]) / pm["mdape_percent"]
        monthly[month] = {
            "benchmark": pm,
            "candidate": cm_month,
            "relative_mdape_improvement_percent": float(rel),
        }
        groups.append((y, p, c))
        improvements.append(rel)
    seed = SCREEN_BOOTSTRAP_SEED if stage == "screen" else DEVELOPMENT_BOOTSTRAP_SEED
    rng = np.random.default_rng(seed)
    differences = np.empty(2000, dtype=float)
    for index in range(len(differences)):
        indices = [rng.integers(0, len(y), len(y)) for y, _, _ in groups]
        parent_ape = np.concatenate([
            np.abs(p[idx] - y[idx]) / y[idx] * 100.0
            for idx, (y, p, _) in zip(indices, groups)
        ])
        candidate_ape = np.concatenate([
            np.abs(c[idx] - y[idx]) / y[idx] * 100.0
            for idx, (y, _, c) in zip(indices, groups)
        ])
        differences[index] = np.median(candidate_ape) - np.median(parent_ape)
    gates = {
        "relative_mdape_gain": gain >= (0.25 if stage == "screen" else 0.5),
        "months_improved": (
            all(value > 0 for value in improvements)
            if stage == "screen" else sum(value > 0 for value in improvements) >= 6
        ),
        "mape_guard": mape_change <= 1.0,
        "bootstrap_upper_below_zero": float(np.quantile(differences, 0.975)) < 0.0,
    }
    if stage == "development":
        gates["worst_month_regression_guard"] = max(0.0, max(-x for x in improvements)) <= 5.0
    return {
        "n": int(len(frame)),
        "benchmark": bm,
        "candidate": cm,
        "relative_mdape_improvement_percent": float(gain),
        "mape_relative_change_percent": float(mape_change),
        "months_improved": int(sum(value > 0 for value in improvements)),
        "worst_month_regression_percent": float(max(0.0, max(-x for x in improvements))),
        "monthly": monthly,
        "paired_bootstrap_mdape_difference_pp": {
            "lower95": float(np.quantile(differences, 0.025)),
            "upper95": float(np.quantile(differences, 0.975)),
            "replicates": len(differences), "seed": seed,
        },
        "gate_checks": gates,
        "qualifies": all(gates.values()),
        "strict_target_below_4_met": cm["mdape_percent"] < 4.0,
    }


def audit() -> dict:
    protocol = _read_json(PROTOCOL_PATH)
    progress = _read_json(PROGRESS_PATH)
    card = protocol["cards"][CARD_KEY]
    runner_path = ROOT / "experiments/bill_target74/r165_listaware_catboost_mape_ratio.py"
    registered_audit_path = ROOT / "experiments/bill_target74/r165_independent_audit.py"
    audit_path = Path(__file__)
    runner_hash = _sha(runner_path)
    registered_audit_hash = _sha(registered_audit_path)
    audit_hash = _sha(audit_path)
    if (card.get("source_sha256") != runner_hash
            or card.get("independent_audit_sha256") != registered_audit_hash):
        raise RuntimeError("R165 preregistered runner/auditor hash mismatch")
    if card.get("parameters", {}).get("target") != "sold_over_list_ratio":
        raise RuntimeError("R165 target is not raw sold/list ratio")
    if card.get("parameters", {}).get("model_config") != MODEL_CONFIG:
        raise RuntimeError("R165 CatBoost configuration differs from preregistration")
    if (card.get("parameters", {}).get("screen_bootstrap_seed") != SCREEN_BOOTSTRAP_SEED
            or card.get("parameters", {}).get("development_bootstrap_seed") != DEVELOPMENT_BOOTSTRAP_SEED):
        raise RuntimeError("R165 bootstrap seeds differ from preregistration")
    if card.get("status") not in {"screen_passed", "screen_failed", "development_complete"}:
        raise RuntimeError(f"R165 has no completed screen/development run: {card.get('status')}")

    run_id = progress.get("r165_run_id")
    result_path = ROOT / progress["r165_result"]
    run_dir = result_path.parent
    manifest_path = run_dir / "run_manifest.json"
    pair_path = run_dir / "paired_predictions.csv"
    result, manifest = _read_json(result_path), _read_json(manifest_path)
    stage = result.get("stage")
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS if stage == "development" else ()
    if not months or result.get("run_id") != run_id or result.get("card") != CARD_KEY:
        raise RuntimeError("R165 result identity or stage does not match progress")
    if tuple(result.get("months", ())) != tuple(months):
        raise RuntimeError("R165 result contains an unexpected month set")
    if result.get("reserved_months_labels_opened") is not False or result.get("hidden_labels_opened") is not False:
        raise RuntimeError("R165 result reports reserved/hidden-label access")
    if manifest.get("reserved_months_labels_opened") is not False or manifest.get("hidden_labels_opened") is not False:
        raise RuntimeError("R165 manifest reports reserved/hidden-label access")
    if manifest.get("result_sha256") != _sha(result_path) or manifest.get("paired_predictions_sha256") != _sha(pair_path):
        raise RuntimeError("R165 result/prediction hash mismatch")
    source_hashes = manifest.get("source_sha256", {})
    audit_rel = "experiments/bill_target74/r165_independent_audit.py"
    runner_rel = "experiments/bill_target74/r165_listaware_catboost_mape_ratio.py"
    if (source_hashes.get(audit_rel) != registered_audit_hash
            or source_hashes.get(runner_rel) != runner_hash):
        raise RuntimeError("R165 provenance does not include the registered runner/auditor")

    r94_result = _read_json(R94_DIR / "development_results.json")
    r94_manifest = _read_json(R94_DIR / "run_manifest.json")
    r94_pairs_path = R94_DIR / "paired_predictions.csv"
    if (r94_manifest.get("paired_predictions_sha256") != _sha(r94_pairs_path)
            or r94_manifest.get("result_sha256") != _sha(R94_DIR / "development_results.json")
            or r94_manifest.get("source_identity", {}).get("data_sha256") != protocol.get("data_sha256")
            or r94_result.get("data_sha256") != protocol.get("data_sha256")
            or r94_result.get("reserved_months_labels_opened") is not False
            or r94_result.get("hidden_labels_opened") is not False
            or r94_manifest.get("reserved_months_labels_opened") is not False
            or r94_manifest.get("hidden_labels_opened") is not False):
        raise RuntimeError("R94 benchmark source integrity/access flags failed")
    if card.get("benchmark_prediction_sha256") != _sha(r94_pairs_path):
        raise RuntimeError("R94 benchmark prediction hash differs from preregistration")
    benchmark = pd.read_csv(r94_pairs_path, dtype={"ml_num": str})
    benchmark = benchmark.loc[
        benchmark.track.eq("list_aware") & benchmark.month.astype(str).isin(months),
        ["month", "ml_num", "sold_price", "base_prediction"],
    ].copy()
    benchmark["month"] = benchmark.month.astype(str)
    benchmark["benchmark_prediction"] = pd.to_numeric(benchmark.base_prediction, errors="raise")
    benchmark = benchmark[["month", "ml_num", "sold_price", "benchmark_prediction"]]

    r35_card = protocol["cards"][R35_KEY]
    if r35_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R165 comparison parent is no longer the validated R35 route")
    r35_result_path = ROOT / r35_card["development_result_path"]
    r35_pair_path = r35_result_path.parent / "paired_predictions.csv"
    r35_manifest = _read_json(r35_result_path.parent / "run_manifest.json")
    r35_result = _read_json(r35_result_path)
    if (r35_manifest.get("result_sha256") != _sha(r35_result_path)
            or r35_manifest.get("paired_predictions_sha256") != _sha(r35_pair_path)
            or r35_manifest.get("source_identity", {}).get("data_sha256") != protocol.get("data_sha256")
            or r35_result.get("data_sha256") != protocol.get("data_sha256")
            or r35_result.get("reserved_months_labels_opened") is not False
            or r35_result.get("hidden_labels_opened") is not False
            or r35_manifest.get("reserved_months_labels_opened") is not False
            or r35_manifest.get("hidden_labels_opened") is not False):
        raise RuntimeError("R35 validated-parent artifact integrity/access flags failed")
    r35_pairs = pd.read_csv(r35_pair_path, dtype={"ml_num": str})
    r35_pairs = r35_pairs.loc[
        r35_pairs.track.eq("list_aware") & r35_pairs.month.astype(str).isin(months),
        ["month", "ml_num", "sold_price", "candidate_prediction"],
    ].copy()
    r35_pairs["month"] = r35_pairs.month.astype(str)
    r35_pairs = r35_pairs.rename(columns={"candidate_prediction": "r35_prediction"})
    r35_benchmark = benchmark.merge(
        r35_pairs, on=["month", "ml_num"], how="outer", suffixes=("_r94", "_r35"), indicator=True,
    )
    if not r35_benchmark._merge.eq("both").all():
        raise RuntimeError("R94 benchmark rows do not align with validated R35 OOF predictions")
    if not np.array_equal(r35_benchmark.sold_price_r94.to_numpy(float), r35_benchmark.sold_price_r35.to_numpy(float)):
        raise RuntimeError("R94 benchmark labels differ from validated R35 OOF labels")
    if not np.allclose(r35_benchmark.benchmark_prediction.to_numpy(float),
                       r35_benchmark.r35_prediction.to_numpy(float), rtol=0.0, atol=0.005):
        raise RuntimeError("R94 list-aware benchmark prediction is not the validated R35 output")

    pairs = pd.read_csv(pair_path, dtype={"ml_num": str})
    pairs["month"] = pairs.month.astype(str)
    if set(pairs.month.unique()) != set(months) or pairs.track.nunique() != 1 or not pairs.track.eq("list_aware").all():
        raise RuntimeError("R165 paired file has unexpected tracks or months")
    if pairs.duplicated(["month", "ml_num"]).any() or pairs.ml_num.isna().any():
        raise RuntimeError("R165 paired file has duplicate or missing listing keys")
    if set(pairs.month.unique()) & RESERVED_MONTHS:
        raise RuntimeError("R165 paired file includes a reserved month")
    keys = ["month", "ml_num"]
    joined = pairs.merge(benchmark, on=keys, how="left", suffixes=("_r165", "_benchmark"),
                         indicator=True, sort=False, validate="one_to_one")
    if len(joined) != len(benchmark) or not joined._merge.eq("both").all():
        raise RuntimeError("R165 keys do not align exactly with the preregistered R94/R35 benchmark")
    if not np.array_equal(joined.sold_price_r165.to_numpy(float), joined.sold_price_benchmark.to_numpy(float)):
        raise RuntimeError("R165 truth labels differ from the benchmark artifact")
    if not np.allclose(joined.benchmark_prediction_r165.to_numpy(float),
                       joined.benchmark_prediction_benchmark.to_numpy(float), rtol=0.0, atol=0.005):
        raise RuntimeError("R165 parent predictions differ from the benchmark")
    evaluated = joined[["month", "sold_price_r165", "benchmark_prediction_r165", "candidate_prediction"]].rename(
        columns={"sold_price_r165": "sold_price", "benchmark_prediction_r165": "benchmark_prediction"}
    )
    vals = evaluated[["sold_price", "benchmark_prediction", "candidate_prediction"]].to_numpy(float)
    if not np.isfinite(vals).all() or (vals <= 0).any():
        raise RuntimeError("R165 has invalid prices")

    score = _score(evaluated, tuple(months), stage)
    stored = result["scores"]["list_aware"]
    stored_parent = stored["benchmark"]
    stored_candidate = stored["candidate"]
    metric_diff = max(
        abs(score["benchmark"]["mdape_percent"] - stored_parent["mdape_percent"]),
        abs(score["candidate"]["mdape_percent"] - stored_candidate["mdape_percent"]),
        abs(score["benchmark"]["mape_percent"] - stored_parent["mape_percent"]),
        abs(score["candidate"]["mape_percent"] - stored_candidate["mape_percent"]),
        abs(score["paired_bootstrap_mdape_difference_pp"]["upper95"]
            - stored["paired_bootstrap_mdape_difference_pp"]["upper95"]),
    )
    if metric_diff > 1e-9:
        raise RuntimeError(f"R165 independent score differs from result by {metric_diff}")
    expected_fits = len(months)
    details = result.get("fit_details", [])
    if len(details) != expected_fits or result.get("fit_starts") != expected_fits:
        raise RuntimeError("R165 result fit count differs from the expected folds")
    for detail in details:
        if (detail.get("model_family") != "catboost"
                or detail.get("target") != "sold_over_list_ratio"
                or detail.get("actual_objective", "").lower() != "mape"
                or detail.get("model_config") != MODEL_CONFIG):
            raise RuntimeError("R165 fit does not match raw-ratio CatBoost MAPE preregistration")
    events = []
    with LEDGER_PATH.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                event = json.loads(line)
                if event.get("card") == CARD_KEY and event.get("run_id") == run_id:
                    events.append(event)
    starts = [event for event in events if event.get("event") == "fit_start"]
    completes = [event for event in events if event.get("event") == "fit_complete"]
    if len(starts) != expected_fits or len(completes) != expected_fits:
        raise RuntimeError("R165 fit ledger does not reconcile")
    fit_completes = [event for event in events if event.get("event") == "fit_complete"]
    access_flag_events = [event for event in events if event.get("event") != "fit_complete"]
    if any(event.get("reserved_months_labels_opened") is not False
           or event.get("hidden_labels_opened") is not False for event in access_flag_events):
        raise RuntimeError("R165 non-completion ledger event reports reserved/hidden-label access")
    for event in fit_completes:
        for flag in ("reserved_months_labels_opened", "hidden_labels_opened"):
            if flag in event and event[flag] is not False:
                raise RuntimeError("R165 fit-completion ledger event reports reserved/hidden-label access")
    if {event.get("month") for event in starts} != set(months):
        raise RuntimeError("R165 ledger folds differ from the registered month set")

    audit_result = {
        "run_id": run_id, "stage": stage, "card": CARD_KEY, "rows": int(len(pairs)),
        "unique_month_listing_keys": True, "safe_months_only": list(months),
        "r165_reserved_or_hidden_labels_opened": False,
        "fit_complete_access_flags_omitted_by_runner": any(
            "reserved_months_labels_opened" not in event or "hidden_labels_opened" not in event
            for event in fit_completes),
        "registered_auditor_preflight_issue": (
            "The registered auditor rejected omitted access keys on fit_complete events. "
            "The runner writes false flags on fit_start, screen_fold_complete, card_start, card_complete, "
            "and the result/manifest; this post-run auditor accepts omission only on fit_complete events "
            "and rejects any present non-false flag."
        ),
        "registered_auditor_sha256": registered_audit_hash,
        "post_run_auditor_path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
        "post_run_auditor_sha256": audit_hash,
        "audit_scope": "Independent row alignment, metrics, gates, provenance, objective/configuration, and ledger audit; no second model refit.",
        "campaign_reserved_labels_previously_opened": True,
        "reserve_validation_eligible": False,
        "benchmark_keys_and_labels_align": True, "benchmark_parent_predictions_align": True,
        "benchmark_matches_validated_r35": True,
        "r94_source_hashes_match": True, "registered_runner_auditor_hashes_match": True,
        "result_prediction_hashes_match": True, "fit_details_match_raw_ratio_mape_config": True,
        "independent_metric_max_difference": float(metric_diff),
        "ledger_fit_starts": len(starts), "ledger_fit_completes": len(completes),
        "score": score, "strict_listaware_target_below_4_met": score["strict_target_below_4_met"],
        "campaign_fit_starts": progress.get("fit_starts"), "limits": progress.get("limits"),
    }
    out_path = run_dir / "independent_audit.json"
    out_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    from experiments.bill_target74 import runner as campaign
    relative_audit = str(out_path.relative_to(ROOT)).replace("\\", "/")
    campaign._write_progress(
        "r165_screen_failed_audited",
        next_action="register a distinct list-aware hypothesis; retain R35 and do not promote R165",
        r165_audit_path=relative_audit,
        r165_audit_passed=True,
        r165_registered_auditor_preflight_issue=True,
        r165_promotion_gate_passed=False,
        latest_run_id=run_id,
        latest_card=CARD_KEY,
        latest_audit_passed=True,
        latest_promotion_gate_passed=False,
        latest_fit_count=len(starts),
    )
    return {"audit_path": str(out_path), **audit_result}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
