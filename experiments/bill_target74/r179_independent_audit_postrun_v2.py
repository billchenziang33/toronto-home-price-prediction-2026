"""Independent reconstruction audit for the R179 two-track mortgage-rate screen."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256

AUDIT_STAGE = "independent_audit_postrun_v2"
FOLD_COLUMNS = (
    "date_sold", "ml_num", "sold_price", "list_price", "date_listed", "house_area",
    "bedrooms", "washrooms", "parking_total", "latitude", "longitude", "house_style",
    "municipality", "community", "house_type_name", "lot_front", "lot_depth", "garage",
    "remarks",
)


def _ledger(campaign) -> list[dict]:
    return [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def _independent_rate_map(r179) -> dict[str, float]:
    path = r179.RATE_CSV_PATH
    if _sha256(path) != r179.RATE_CSV_SHA256:
        raise RuntimeError("R179 audit found a changed Bank of Canada CSV")
    table = pd.read_csv(path, parse_dates=["date"])
    periods = table.date.dt.to_period("M")
    values = pd.to_numeric(table.rate_percent, errors="coerce")
    if (len(table) != 77 or periods.duplicated().any()
            or periods.iloc[0] != pd.Period("2019-11", freq="M")
            or periods.iloc[-1] != pd.Period("2026-03", freq="M")
            or not (periods.astype("int64").diff().dropna() == 1).all()
            or not np.isfinite(values.to_numpy(float)).all() or (values <= 0).any()):
        raise RuntimeError("R179 audit found invalid external series coverage or values")
    return dict(zip(periods.astype(str), values.to_numpy(float)))


def _install_audit_features(r22, rate_map: dict[str, float], fit_rows: pd.DataFrame,
                            eval_rows: pd.DataFrame, track: str):
    from model_candidate_v2 import features as feature_module

    target = "off_market_features"
    original = getattr(feature_module, target)

    def with_lagged_rate(homes):
        base = original(homes).copy()
        dates = pd.to_datetime(homes.date_listed, errors="raise")
        if dates.isna().any():
            raise RuntimeError("R179 audit found missing listing dates")
        rate_month = (dates.dt.to_period("M") - 2).astype(str)
        rate = rate_month.map(rate_map)
        if rate.isna().any() or not np.isfinite(rate.to_numpy(float)).all():
            raise RuntimeError("R179 audit found a missing or non-finite lagged rate")
        base["boc_mortgage_rate_5y_lag2"] = rate.to_numpy(float)
        return base

    setattr(feature_module, target, with_lagged_rate)
    try:
        _, restore_text = r22._install_text_features(fit_rows, eval_rows, track)
    except BaseException:
        setattr(feature_module, target, original)
        raise

    def restore():
        try:
            restore_text()
        finally:
            setattr(feature_module, target, original)

    return restore


def _load_projected_fold(campaign, greedy, month: str):
    """Match the canonical safe fold while reading only model-required columns."""
    from experiments.model_optimization_v2.folds import build_month_fold

    month_end = (pd.Period(month, freq="M") + 1).start_time
    listings = pd.read_parquet(
        campaign.DATA, columns=list(FOLD_COLUMNS), filters=[("date_sold", "<", month_end)]
    )
    listings["date_sold"] = pd.to_datetime(listings.date_sold, errors="coerce")
    train, homes, truth = build_month_fold(listings, month, "off_market")
    heldout = listings.loc[listings.date_sold.dt.to_period("M").astype(str).eq(month)].copy()
    heldout["ml_num"] = heldout.ml_num.astype(str)
    heldout = heldout.set_index("ml_num").loc[homes.ml_num.astype(str)].reset_index()
    if not np.array_equal(homes.ml_num.astype(str).to_numpy(), truth.ml_num.astype(str).to_numpy()):
        raise RuntimeError(f"R179 projected fold rows and labels differ in {month}")
    return listings, train, homes, truth, heldout


def _rebuild_fold(campaign, r179, month: str, track: str, saved: pd.DataFrame,
                  parent_rows: pd.DataFrame, fit_details: list[dict], rate_map: dict[str, float]) -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_greedy import runner as greedy

    _, train, _, truth, heldout = _load_projected_fold(campaign, greedy, month)
    cutoff = pd.Period(month, freq="M").start_time
    train_dates = pd.to_datetime(train.date_sold, errors="raise")
    if train.empty or (train_dates >= cutoff).any():
        raise RuntimeError(f"R179 audit found non-past training rows in {month}")
    if train_dates.dt.to_period("M").astype(str).isin(("2025-05", "2025-08")).any():
        raise RuntimeError("R179 audit found a reserved month among training labels")
    homes = hide_answers(heldout, track)
    saved = saved.loc[saved.month.astype(str).eq(month) & saved.track.eq(track)].copy()
    parent_rows = parent_rows.loc[parent_rows.month.astype(str).eq(month)].copy()
    saved_i = saved.set_index(saved.ml_num.astype(str)).reindex(homes.ml_num.astype(str))
    parent_i = parent_rows.set_index(parent_rows.ml_num.astype(str)).reindex(homes.ml_num.astype(str))
    if (len(saved_i) != len(homes) or saved_i.isna().any().any() or parent_i.isna().any().any()
            or not np.array_equal(saved_i.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
            or not np.array_equal(parent_i.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
        raise RuntimeError(f"R179 audit found row or label misalignment for {track}/{month}")
    parent_prediction = parent_i.candidate_prediction.to_numpy(float)
    detached_eval = greedy._specialist_mask(homes, r179.HOUSE_TYPE)
    detached_train = greedy._specialist_mask(train, r179.HOUSE_TYPE)
    if int(detached_eval.sum()) < 100 or int(detached_train.sum()) < 500:
        raise RuntimeError(f"R179 audit found insufficient Detached support for {track}/{month}")

    from experiments.model_optimization_v2 import worker
    old_normalize = worker._normalize_categories
    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(r179.VARIANT)
    greedy.VARIANTS[r179.VARIANT] = {"model_config_overrides": {name: {} for name in r179.TRACKS}}
    fit_by_key = {(d["track"], d["month"], d["component"]): d for d in fit_details}
    audit_fits = []

    def normalize(train_x, eval_x, family):
        train_x, eval_x, categories = old_normalize(train_x, eval_x, family)
        for frame in (train_x, eval_x):
            if r179.RATE_COLUMN not in frame or not np.isfinite(
                    pd.to_numeric(frame[r179.RATE_COLUMN], errors="coerce").to_numpy(float)).all():
                raise RuntimeError("R179 audit normalization omitted or corrupted mortgage-rate feature")
        if list(train_x.columns) != list(eval_x.columns):
            raise RuntimeError("R179 audit found train/evaluation feature-column mismatch")
        return train_x, eval_x, categories

    worker._normalize_categories = normalize
    greedy._fit_record = lambda _parent_id, fit_fn: fit_fn()

    def fit_component(component: str, fit_rows: pd.DataFrame, eval_rows: pd.DataFrame,
                      base_variant: str | None = None):
        detail = fit_by_key.get((track, month, component))
        if detail is None:
            raise RuntimeError(f"R179 saved fit detail is missing for {track}/{month}/{component}")
        audit_fit_id = (f"{campaign.PROTOCOL['campaign_id']}:{r179.CARD}:{AUDIT_STAGE}:"
                        f"{track}:{month}:{component}:{detail['fit_id']}")
        campaign._append({"event": "fit_start", "card": r179.CARD, "stage": AUDIT_STAGE,
                          "run_id": detail["run_id"], "fit_id": audit_fit_id,
                          "track": track, "month": month, "component": component,
                          "training_rows": int(len(fit_rows)), "evaluation_rows": int(len(eval_rows)),
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        started = time.perf_counter()
        restore = _install_audit_features(r22, rate_map, fit_rows, eval_rows, track)
        try:
            fitted = greedy._fit_history_candidate(
                month, track, train, eval_rows, fit_rows=fit_rows,
                variant=r179.VARIANT, base_variant=base_variant,
                fit_tag=f"R179-audit-{track}-{month}-{component}",
            )
            if (r179.RATE_COLUMN not in fitted.get("feature_columns", [])
                    or list(fitted["feature_columns"]) != list(detail.get("feature_columns", []))):
                raise RuntimeError("R179 audit feature-column set/order differs from the recorded fit")
            audit_fits.append({"fit_id": audit_fit_id, "seconds": time.perf_counter() - started,
                               "status": "complete", "component": component})
            campaign._append({"event": "fit_complete", "card": r179.CARD,
                              "stage": AUDIT_STAGE, "run_id": detail["run_id"],
                              "fit_id": audit_fit_id, "track": track, "month": month,
                              "component": component, "status": "complete",
                              "seconds": time.perf_counter() - started,
                              "feature_columns_match": True,
                              "reserved_months_labels_opened": False, "hidden_labels_opened": False})
            return fitted
        except BaseException as exc:
            campaign._append({"event": "fit_complete", "card": r179.CARD,
                              "stage": AUDIT_STAGE, "run_id": detail["run_id"],
                              "fit_id": audit_fit_id, "track": track, "month": month,
                              "component": component, "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}",
                              "reserved_months_labels_opened": False, "hidden_labels_opened": False})
            raise
        finally:
            restore()

    try:
        if track == "off_market":
            base = fit_component("base", train.copy(), homes.copy(), "G06_lot_garage")
            expert = fit_component("detached_expert", train.loc[detached_train].copy(),
                                   homes.loc[detached_eval].copy(), "G06_lot_garage")
            rebuilt = greedy._blend_specialist_predictions(
                np.round(base["predictions"]), np.round(expert["predictions"]), homes,
                r179.HOUSE_TYPE, r179.SPECIALIST_WEIGHT)
        else:
            specialist = fit_component("detached_expert", train.loc[detached_train].copy(),
                                       homes.loc[detached_eval].copy())
            frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, track, train, homes)
            blended = (r179.SPECIALIST_WEIGHT * np.asarray(specialist["predictions"], dtype=float)
                       + (1.0 - r179.SPECIALIST_WEIGHT) * frozen_hgbr[detached_eval])
            rebuilt = greedy._overlay_specialist_predictions(
                parent_prediction, homes, blended, r179.HOUSE_TYPE)
            if not np.array_equal(rebuilt[~detached_eval], parent_prediction[~detached_eval]):
                raise RuntimeError("R179 audit found changed non-Detached List-aware predictions")
    finally:
        worker._normalize_categories = old_normalize
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(r179.VARIANT, None)
        else:
            greedy.VARIANTS[r179.VARIANT] = old_variant

    candidate = saved_i.candidate_prediction.to_numpy(float)
    difference = float(np.max(np.abs(rebuilt - candidate)))
    if difference != 0.0:
        raise RuntimeError(f"R179 audit prediction mismatch for {track}/{month}: {difference}")
    return {"track": track, "month": month, "training_cutoff_exclusive": cutoff.isoformat(),
            "training_rows": int(len(train)), "evaluation_rows": int(len(homes)),
            "detached_training_rows": int(detached_train.sum()),
            "detached_evaluation_rows": int(detached_eval.sum()),
            "audit_fit_count": len(audit_fits), "audit_fits": audit_fits,
            "feature_column_order_matches": True, "route_invariants_checked": True,
            "prediction_max_abs_diff": difference, "predictions_reconstructed": True}


def audit() -> dict:
    from experiments.bill_target74 import r04_ablation as identity
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r179_two_track_boc_mortgage_rate_lag2 as r179

    card = campaign.PROTOCOL["cards"].get(r179.CARD)
    if not card or card.get("status") not in {"screen_failed", "screen_passed"}:
        raise RuntimeError(f"R179 has no completed screen to audit: {None if not card else card.get('status')}")
    result_path = campaign.ROOT / card["result_path"]
    paired_path = campaign.ROOT / card["paired_predictions_path"]
    manifest_path = campaign.ROOT / card["manifest_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    rate_map = _independent_rate_map(r179)
    source_hashes = {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                     for path in r179._source_files()}
    checks: dict[str, object] = {
        "identity": result.get("card") == r179.CARD and result.get("run_id") == card.get("run_id")
                    and result.get("screen_months") == list(r179.SCREEN_MONTHS)
                    and manifest.get("run_id") == result.get("run_id"),
        "result_hash": manifest.get("result_sha256") == _sha256(result_path),
        "paired_hash": manifest.get("paired_predictions_sha256") == _sha256(paired_path),
        "data_identity": manifest.get("source_identity", {}).get("data_sha256") == identity._source_identity()["data_sha256"]
                         == campaign.PROTOCOL["data_sha256"],
        "external_data_identity": manifest.get("external_data", {}).get("csv_sha256") == r179.RATE_CSV_SHA256
                                  and manifest.get("external_data", {}).get("metadata_sha256") == _sha256(r179.RATE_META_PATH)
                                  and card.get("external_data", {}).get("csv_sha256") == r179.RATE_CSV_SHA256,
        "source_hashes": manifest.get("code_sha256") == source_hashes,
        "access_flags": result.get("reserved_months_labels_opened") is False
                        and result.get("hidden_labels_opened") is False
                        and result.get("reserve_validation_eligible") is False
                        and manifest.get("reserved_months_labels_opened") is False
                        and manifest.get("hidden_labels_opened") is False
                        and card.get("reserved_months_labels_opened") is False
                        and card.get("hidden_labels_opened") is False
                        and not (set(r179.SCREEN_MONTHS) & set(campaign.PROTOCOL["reserved_months"])),
        "paired_rows": (not paired.duplicated(["month", "track", "ml_num"]).any()
                        and set(paired.month.astype(str)) == set(r179.SCREEN_MONTHS)
                        and set(paired.track) == set(r179.TRACKS)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"R179 pre-reconstruction audit failed: {checks}")

    parents = {track: r179.parent_loader._verified_parent(campaign, track) for track in r179.TRACKS}
    fold_checks = []
    for track in r179.TRACKS:
        parent_rows = parents[track][4]
        candidate_details = [item for item in result["fit_details"] if item.get("track") == track]
        expected_components = 4 if track == "off_market" else 2
        if len(candidate_details) != expected_components:
            raise RuntimeError(f"R179 fit details have wrong count for {track}")
        for month in r179.SCREEN_MONTHS:
            fold_checks.append(_rebuild_fold(campaign, r179, month, track, paired,
                                             parent_rows, candidate_details, rate_map))

    recomputed_scores = {}
    for track in r179.TRACKS:
        frame = paired.loc[paired.track.eq(track),
                           ["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
        score = r22.score_development_confirmation(
            frame, months=r179.SCREEN_MONTHS,
            bootstrap_replicates=r179.BOOTSTRAP_REPLICATES, bootstrap_seed=r179.BOOTSTRAP_SEED,
            minimum_relative_gain_percent=0.25, minimum_months_improved=2,
            maximum_worst_month_regression_percent=0.0, maximum_mape_regression_percent=1.0)
        score["qualifies_screen"] = score.pop("qualifies_development")
        recomputed_scores[track] = score
    checks["metrics_recomputed"] = all(
        result.get("tracks", {}).get(track) == recomputed_scores[track]
        for track in r179.TRACKS)
    checks["fit_details"] = (result.get("fit_starts") == 6 and manifest.get("fit_starts") == 6
                              and len(result.get("fit_details", [])) == 6)
    events = [e for e in _ledger(campaign) if e.get("card") == r179.CARD and e.get("run_id") == result["run_id"]]
    candidate_starts = [e for e in events if e.get("event") == "fit_start" and e.get("stage") == "screen"]
    candidate_done = [e for e in events if e.get("event") == "fit_complete" and e.get("stage") == "screen"
                      and e.get("status") == "complete"]
    audit_starts = [e for e in events if e.get("event") == "fit_start" and e.get("stage") == AUDIT_STAGE]
    audit_done = [e for e in events if e.get("event") == "fit_complete" and e.get("stage") == AUDIT_STAGE
                  and e.get("status") == "complete"]
    checks["fit_ledger"] = len(candidate_starts) == len(candidate_done) == len(audit_starts) == len(audit_done) == 6
    checks["ledger_access_flags"] = not any(
        e.get("reserved_months_labels_opened") is True or e.get("hidden_labels_opened") is True for e in events)
    checks["prediction_reconstruction"] = all(f.get("predictions_reconstructed") is True for f in fold_checks)
    audit_passed = all(value is True for value in checks.values())
    audit_result = {"card": r179.CARD, "run_id": result["run_id"], "stage": "screen",
                    "audit_passed": audit_passed,
                    "promotion_gate_passed": audit_passed and any(s["qualifies_screen"] for s in recomputed_scores.values()),
                    "checks": checks, "fold_checks": fold_checks, "recomputed_scores": recomputed_scores,
                    "audit_attempt_stage": AUDIT_STAGE,
                    "audit_tool_sha256": _sha256(Path(__file__)),
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    audit_path = result_path.parent / "independent_audit.json"
    audit_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_hash = _sha256(audit_path)
    campaign._append({"event": "independent_audit_complete", "card": r179.CARD,
                      "run_id": result["run_id"], "stage": "screen",
                      "audit_passed": audit_passed,
                      "promotion_gate_passed": audit_result["promotion_gate_passed"],
                      "audit_path": str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "audit_sha256": audit_hash,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    card.update({"audit_path": str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                 "audit_sha256": audit_hash, "audit_passed": audit_passed,
                 "promotion_gate_passed": audit_result["promotion_gate_passed"],
                 "status": ("screen_passed_audited" if audit_result["promotion_gate_passed"]
                            else "screen_failed_audited")})
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._write_progress("r179_screen_audited_passed" if audit_result["promotion_gate_passed"]
                              else "r179_screen_audited_failed",
                              next_action=("register frozen eight-month development for the passing R179 track(s)"
                                           if audit_result["promotion_gate_passed"] else
                                           "retain R22/R35; register a distinct feature hypothesis, not a retune of R179"),
                              r179_run_id=result["run_id"], r179_audit_passed=audit_passed,
                              r179_promotion_gate_passed=audit_result["promotion_gate_passed"])
    return audit_result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
