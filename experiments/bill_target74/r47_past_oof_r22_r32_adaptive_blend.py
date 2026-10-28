"""R47: past-only blending of R22 with the R32 comparable-feature candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import r19_expert_weight as file_helpers
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r32_comparable_r22 as r32
from experiments.bill_target74 import r33_r22_r32_blend as r33_screen
from experiments.bill_target74 import r33_fixed_origin as r33_ids
from experiments.bill_target74 import r40_past_oof_r26_blend as weight_selection
from experiments.bill_target74 import r44_past_oof_r35_shrinkage as dates_helper
from experiments.bill_target74 import r22_remarks_tfidf as scoring
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy


CARD = "R47_past_oof_r22_r32_adaptive_blend"
TRACK = "off_market"
SEED_MONTHS = ("2022-06", "2022-11")
SCREEN_MONTHS = ("2023-03", "2024-06")
FIT_PLAN = tuple((month, component) for month in SEED_MONTHS for component in ("base", "expert"))
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
LIMIT_KEYS = (
    "campaign_fit_starts", "campaign_minutes", "exploration_fit_starts",
    "exploration_minutes", "final_reserve_fit_starts", "final_reserve_minutes",
    "batch_fit_starts", "batch_minutes", "single_fit_seconds",
)
R33_DEVELOPMENT_RUN = "bill_target74_r33_offmarket_development_20261025T030109Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _card_spec() -> dict:
    return {
        "status": "registered",
        "track": TRACK,
        "parent": "hash-verified R22 off-market development predictions",
        "variant": "R32 R22-text + seven strict-as-of comparable features",
        "source_runs": {
            "r22_seed_oof": "R40 strictly-past OOF rows (parent R22 predictions only)",
            "r32_seed_fit_months": list(SEED_MONTHS),
            "r32_screen_run_id": r33_ids.R32_SCREEN_RUN,
            "r33_development_run_id": R33_DEVELOPMENT_RUN,
        },
        "difference_from_prior_work": (
            "R33 fixed a 0.50 R32 share, passed its eight-month development gate at MdAPE "
            "7.07911%, then regressed 0.280% on previously used fixed-origin rows. R40's "
            "adaptive OOF blend used R26, and R41 used R26 by house type. R47 applies "
            "strict-past OOF weight selection to the distinct R32 comparable-feature candidate."
        ),
        "method": (
            "Fit R32 base and Detached expert only on the two safe 2022 seed folds; reuse "
            "R40's hash-verified R22 parent OOF predictions for those folds. Reconstruct R32 "
            "OOF predictions for all eight development months from the verified R32 screen "
            "rows and R33 saved component predictions. At every outer cutoff choose an R32 "
            "share from the 0.05 grid using only rows dated strictly before the cutoff; use "
            "minimum 1000 OOF rows, guard MAPE regression at 1%, break ties by lower MAPE "
            "then smaller share, and fall back to R22 when support or eligible weights are "
            "insufficient. Blend in price space and round once."
        ),
        "expected_fit_starts": len(FIT_PLAN),
        "seed_oof_months": list(SEED_MONTHS),
        "seed_oof_fit_plan": [{"month": month, "component": component} for month, component in FIT_PLAN],
        "screen_months_reused_as_past_oof": list(SCREEN_MONTHS),
        "development_confirmation": {
            "development_months": list(DEVELOPMENT_MONTHS),
            "target_mdape_strictly_below": 7.0,
            "relative_mdape_improvement_percent_min": 0.5,
            "months_improved_min": 6,
            "worst_month_regression_max_percent": 5.0,
            "mape_regression_max_percent": 1.0,
            "paired_bootstrap_upper95_below_zero": True,
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 20261026,
        },
        "weight_selection": {
            "grid": [float(x) for x in weight_selection.WEIGHTS],
            "minimum_past_oof_rows": int(weight_selection.MINIMUM_OOF_ROWS),
            "past_mape_regression_max_percent": 1.0,
            "cutoff": "date_sold strictly earlier than outer month start",
            "tie_break": ["lower MdAPE", "lower MAPE", "smaller R32 share"],
            "fallback": "R22 (R32 share 0.00) if support is insufficient or no grid share passes MAPE guard",
        },
        "fixed_origin_policy": (
            "R33 already scored the registered fixed-origin rows; any R47 re-score is descriptive "
            "only and cannot be represented as independent fixed-origin confirmation."
        ),
        "reserve_validation_eligible": False,
        "policy": (
            "Do not read or use 2025-05/08 or hidden labels. R47 uses development and safe 2022 "
            "OOF rows only. Preserve R22 unless all registered development gates pass."
        ),
    }


def register_card() -> dict:
    if any(campaign.PROTOCOL["limits"].get(key) is not None for key in LIMIT_KEYS):
        raise RuntimeError("R47 requires the active campaign's unbounded limits")
    cards = campaign.PROTOCOL["cards"]
    if CARD in cards:
        raise RuntimeError(f"R47 is already registered with status {cards[CARD].get('status')!r}")
    spec = _card_spec()
    cards[CARD] = spec
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                      encoding="utf-8")
    campaign._append({
        "event": "card_registered", "card": CARD, "track": TRACK,
        "expected_fit_starts": len(FIT_PLAN), "seed_oof_months": list(SEED_MONTHS),
        "development_months": list(DEVELOPMENT_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r47_registered",
        next_action="Verify the registered R32/R22 source artifacts and safe-fold support before fitting",
    )
    print(json.dumps({"card": CARD, "status": "registered",
                      "expected_fit_starts": len(FIT_PLAN)}, indent=2))
    return spec


def _verify_run_artifacts(run_id: str, result_name: str) -> tuple[Path, dict, dict, pd.DataFrame]:
    folder = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path = folder / result_name
    rows_path = folder / "paired_predictions.csv"
    manifest_path = folder / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, rows_path, manifest_path)):
        raise FileNotFoundError(f"run artifacts are incomplete for {run_id}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(rows_path)
            or manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result.get("reserved_months_labels_opened")
            or result.get("hidden_labels_opened")):
        raise RuntimeError(f"result, row, data, or label-safety verification failed for {run_id}")
    for source_key, snapshot in manifest.get("source_snapshots", {}).items():
        path = folder / "provenance" / snapshot
        if not path.is_file() or manifest.get("code_sha256", {}).get(source_key) != _sha256(path):
            raise RuntimeError(f"source snapshot hash failed for {run_id}/{source_key}")
    return folder, result, manifest, pd.read_csv(rows_path, dtype={"ml_num": str})


def _load_inputs() -> dict:
    from data import hide_answers

    if CARD not in campaign.PROTOCOL["cards"]:
        raise RuntimeError("R47 must be preregistered before examining candidate predictions")
    if any(campaign.PROTOCOL["limits"].get(key) is not None for key in LIMIT_KEYS):
        raise RuntimeError("active campaign is no longer unbounded")
    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R47 data identity differs from the campaign")

    # Verify the existing R40 OOF artifact used only to supply R22 seed predictions.
    r40_card = campaign.PROTOCOL["cards"]["R40_past_oof_r26_blend"]
    r40_path = campaign.ROOT / r40_card["result_path"]
    r40_folder, r40_result, r40_manifest, r40_rows = _verify_run_artifacts(
        r40_card["run_id"], "screen_results.json",
    )
    r40_oof_path = r40_folder / "past_oof_predictions.csv"
    if (r40_result.get("run_id") != r40_card["run_id"]
            or r40_manifest.get("past_oof_predictions_sha256") != _sha256(r40_oof_path)
            or r40_result.get("fit_starts") != 8):
        raise RuntimeError("R40 seed OOF artifact identity or hash failed")
    r40_oof = pd.read_csv(r40_oof_path, dtype={"ml_num": str})
    seed_parent = r40_oof.loc[r40_oof.month.astype(str).isin(SEED_MONTHS)].copy()
    if (set(seed_parent.month.astype(str)) != set(SEED_MONTHS)
            or seed_parent.duplicated(["month", "ml_num"]).any()
            or not {"parent_prediction", "sold_price", "date_sold"}.issubset(seed_parent.columns)):
        raise RuntimeError("R40 OOF rows do not provide the registered R22 seed predictions")

    # R33's development run contains exact saved base/expert component predictions for R32.
    dev_folder = campaign.CAMPAIGN_ROOT / "batches" / R33_DEVELOPMENT_RUN
    dev_result_path = dev_folder / "r33_off_market_development_results.json"
    dev_rows_path = dev_folder / "paired_predictions.csv"
    dev_manifest_path = dev_folder / "run_manifest.json"
    if not all(path.is_file() for path in (dev_result_path, dev_rows_path, dev_manifest_path)):
        raise FileNotFoundError("R33 off-market development artifacts are incomplete")
    dev_result = json.loads(dev_result_path.read_text(encoding="utf-8"))
    dev_manifest = json.loads(dev_manifest_path.read_text(encoding="utf-8"))
    if (dev_result.get("run_id") != R33_DEVELOPMENT_RUN
            or dev_result.get("track") != TRACK
            or dev_result.get("data_sha256") != source["data_sha256"]
            or dev_manifest.get("result_sha256") != _sha256(dev_result_path)
            or dev_manifest.get("paired_predictions_sha256") != _sha256(dev_rows_path)
            or dev_manifest.get("source_identity", {}).get("data_sha256") != source["data_sha256"]
            or dev_result.get("reserved_months_labels_opened")
            or dev_result.get("hidden_labels_opened")):
        raise RuntimeError("R33 development result identity, hash, or safety flags failed")
    dev_rows = pd.read_csv(dev_rows_path, dtype={"ml_num": str})
    dev_rows = dev_rows.loc[dev_rows.track.eq(TRACK)].copy()
    dev_rows["month"] = dev_rows.month.astype(str)
    dev_rows["ml_num"] = dev_rows.ml_num.astype(str)
    if (set(dev_rows.month.astype(str)) != set(DEVELOPMENT_MONTHS)
            or dev_rows.duplicated(["month", "ml_num"]).any()):
        raise RuntimeError("R33 development rows do not exactly cover the safe development months")

    # Verify R32 screen candidate predictions and both R33 component prediction hashes.
    r32_folder, r32_result, r32_manifest, r32_screen_rows = _verify_run_artifacts(
        r33_ids.R32_SCREEN_RUN, "r32_screen_results.json",
    )
    r32_screen_rows = r32_screen_rows.loc[r32_screen_rows.track.eq(TRACK)].copy()
    r32_screen_rows["month"] = r32_screen_rows.month.astype(str)
    r32_screen_rows["ml_num"] = r32_screen_rows.ml_num.astype(str)
    if (set(r32_screen_rows.month.astype(str)) != set(SCREEN_MONTHS)
            or r32_screen_rows.duplicated(["month", "ml_num"]).any()):
        raise RuntimeError("R32 screen rows do not exactly cover the registered screen months")
    component_map = {}
    for detail in dev_manifest.get("component_manifest", []):
        if detail.get("track") != TRACK:
            continue
        key = (str(detail.get("month")), str(detail.get("component")))
        if key in component_map:
            raise RuntimeError(f"R33 has duplicate saved R32 components for {key}")
        path = campaign.ROOT / detail.get("saved_prediction_path", "")
        if (not path.is_file()
                or detail.get("saved_prediction_sha256") != _sha256(path)
                or int(detail.get("neighbor_comparable_feature_count", -1)) != 7):
            raise RuntimeError(f"R33 saved R32 component failed verification: {key}")
        component_map[key] = (path, detail)
    expected_components = {
        (month, component)
        for month in DEVELOPMENT_MONTHS if month not in SCREEN_MONTHS
        for component in ("base", "expert")
    }
    if set(component_map) != expected_components:
        raise RuntimeError("R33 saved R32 component manifest differs from the registered development folds")

    metadata_months = tuple((*SEED_MONTHS, *DEVELOPMENT_MONTHS))
    metadata_rows = dates_helper._safe_date_metadata(metadata_months, campaign.DATA)
    dev_oof_parts = []
    support = {}
    for month in DEVELOPMENT_MONTHS:
        _, train, _, truth, heldout = greedy._load_fold(month)
        cutoff = pd.Period(month, freq="M").start_time
        if pd.to_datetime(train.date_sold, errors="raise").ge(cutoff).any():
            raise RuntimeError(f"R47 {month} training rows reach the outer cutoff")
        homes = hide_answers(heldout, TRACK).reset_index(drop=True)
        ids = homes.ml_num.astype(str).to_numpy()
        ref = dev_rows.loc[dev_rows.month.eq(month)].copy()
        if (ref.ml_num.astype(str).duplicated().any()
                or set(ref.ml_num.astype(str)) != set(ids)):
            raise RuntimeError(f"R33/R22 parent rows do not align to {month}")
        ref["ml_num"] = ref.ml_num.astype(str)
        ref = ref.set_index("ml_num").loc[ids].reset_index()
        if not np.array_equal(ref.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
            raise RuntimeError(f"R33/R22 labels do not align to {month}")
        parent_prediction = ref.parent_prediction.to_numpy(float)
        route_eval = greedy._specialist_mask(homes, "Detached")

        if month in SCREEN_MONTHS:
            alt = r32_screen_rows.loc[r32_screen_rows.month.eq(month)].copy()
            if (alt.ml_num.astype(str).duplicated().any()
                    or set(alt.ml_num.astype(str)) != set(ids)):
                raise RuntimeError(f"R32 screen rows do not align to {month}")
            alt["ml_num"] = alt.ml_num.astype(str)
            alt = alt.set_index("ml_num").loc[ids].reset_index()
            if not np.array_equal(alt.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
                raise RuntimeError(f"R32 screen labels do not align to {month}")
            variant_prediction = alt.candidate_prediction.to_numpy(float)
        else:
            base_path, _ = component_map[(month, "base")]
            expert_path, _ = component_map[(month, "expert")]
            base = pd.read_csv(base_path, dtype={"ml_num": str})
            expert = pd.read_csv(expert_path, dtype={"ml_num": str})
            if (base.ml_num.astype(str).duplicated().any()
                    or expert.ml_num.astype(str).duplicated().any()
                    or set(base.ml_num.astype(str)) != set(ids)
                    or set(expert.ml_num.astype(str)) != set(ids[route_eval])):
                raise RuntimeError(f"R32 base/expert component IDs do not align to {month}")
            base["ml_num"] = base.ml_num.astype(str)
            expert["ml_num"] = expert.ml_num.astype(str)
            base = base.set_index("ml_num").loc[ids].reset_index()
            expert = expert.set_index("ml_num").loc[ids[route_eval]].reset_index()
            variant_prediction = greedy._blend_specialist_predictions(
                np.round(base.predicted_price.to_numpy(float)),
                np.round(expert.predicted_price.to_numpy(float)),
                homes, "Detached", 0.75,
            )
        reconstructed = np.round((parent_prediction + variant_prediction) * 0.5)
        if not np.array_equal(reconstructed, ref.candidate_prediction.to_numpy(float)):
            raise RuntimeError(f"saved R32 components do not reconstruct the R33 50/50 output for {month}")
        dates = metadata_rows.loc[metadata_rows.month.astype(str).eq(month)].set_index("ml_num").loc[ids]
        if not dates.date_sold.dt.to_period("M").astype(str).eq(month).all():
            raise RuntimeError(f"R47 date metadata is outside {month}")
        dev_oof_parts.append(pd.DataFrame({
            "month": month, "track": TRACK, "ml_num": ids,
            "date_sold": pd.to_datetime(dates.date_sold.to_numpy()),
            "sold_price": truth.sold_price.to_numpy(float),
            "parent_prediction": parent_prediction,
            "variant_prediction": np.asarray(variant_prediction, dtype=float),
        }))
        route_train = greedy._specialist_mask(train, "Detached")
        support[month] = {
            "training_rows": int(len(train)),
            "detached_training_rows": int(route_train.sum()),
            "evaluation_rows": int(len(homes)),
            "detached_evaluation_rows": int(route_eval.sum()),
            "r32_variant_reconstructed_exactly": True,
        }

    seed_support = {}
    for month in SEED_MONTHS:
        _, train, _, truth, heldout = greedy._load_fold(month)
        cutoff = pd.Period(month, freq="M").start_time
        if pd.to_datetime(train.date_sold, errors="raise").ge(cutoff).any():
            raise RuntimeError(f"R47 seed training rows reach the {month} cutoff")
        homes = hide_answers(heldout, TRACK).reset_index(drop=True)
        ids = homes.ml_num.astype(str).to_numpy()
        parent_rows = seed_parent.loc[seed_parent.month.astype(str).eq(month)].copy()
        if (parent_rows.ml_num.astype(str).duplicated().any()
                or set(parent_rows.ml_num.astype(str)) != set(ids)):
            raise RuntimeError(f"R40 R22 parent OOF rows do not align to seed fold {month}")
        parent_rows["ml_num"] = parent_rows.ml_num.astype(str)
        parent_rows = parent_rows.set_index("ml_num").loc[ids].reset_index()
        if not np.array_equal(parent_rows.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
            raise RuntimeError(f"R40 R22 parent OOF labels do not align to seed fold {month}")
        date_row = metadata_rows.loc[metadata_rows.month.astype(str).eq(month)].set_index("ml_num").loc[ids]
        if not np.array_equal(pd.to_datetime(parent_rows.date_sold).to_numpy(),
                              pd.to_datetime(date_row.date_sold).to_numpy()):
            raise RuntimeError(f"R40 seed dates do not match safe date metadata for {month}")
        seed_support[month] = {
            "training_rows": int(len(train)),
            "detached_training_rows": int(greedy._specialist_mask(train, "Detached").sum()),
            "evaluation_rows": int(len(homes)),
            "detached_evaluation_rows": int(greedy._specialist_mask(homes, "Detached").sum()),
        }

    preflight = {
        "campaign_id": campaign.PROTOCOL["campaign_id"],
        "card": CARD,
        "screen_run_id": r33_ids.R33_SCREEN_RUN,
        "r32_screen_run_id": r33_ids.R32_SCREEN_RUN,
        "r33_development_run_id": R33_DEVELOPMENT_RUN,
        "r40_seed_oof_run_id": r40_result["run_id"],
        "development_months": list(DEVELOPMENT_MONTHS),
        "seed_oof_months": list(SEED_MONTHS),
        "expected_fit_starts": len(FIT_PLAN),
        "seed_support": seed_support,
        "development_support": support,
        "input_sha256": {
            "r40_result": _sha256(r40_path),
            "r40_oof": _sha256(r40_oof_path),
            "r33_dev_result": _sha256(dev_result_path),
            "r33_dev_rows": _sha256(dev_rows_path),
            "r33_dev_manifest": _sha256(dev_manifest_path),
            "r32_screen_result": _sha256(r32_folder / "r32_screen_results.json"),
            "r32_screen_rows": _sha256(r32_folder / "paired_predictions.csv"),
            "r32_screen_manifest": _sha256(r32_folder / "run_manifest.json"),
            **{f"r32_component_{month}_{component}": _sha256(path)
               for (month, component), (path, _) in component_map.items()},
        },
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    return {
        "preflight": preflight,
        "seed_parent": seed_parent,
        "development_oof": pd.concat(dev_oof_parts, ignore_index=True),
    }


def preflight() -> dict:
    details = _load_inputs()["preflight"]
    print(json.dumps(details, indent=2, sort_keys=True))
    return details


def register_stage() -> dict:
    details = _load_inputs()["preflight"]
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R47 card must be registered before the OOF preflight is registered")
    if card.get("stage_status") not in {None, "preflight_complete"}:
        raise RuntimeError(f"R47 stage already has status {card.get('stage_status')!r}")
    run_id = "bill_target74_r47_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    card.update({"stage_status": "registered", "run_id": run_id,
                 "preflight": details,
                 "registered_at_utc": datetime.now(timezone.utc).isoformat()})
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                      encoding="utf-8")
    campaign._append({
        "event": "card_stage_registered", "card": CARD,
        "stage": "past_oof_seed_and_development_blend", "run_id": run_id,
        "fit_starts_expected": len(FIT_PLAN), "fit_starts_completed": 0,
        "seed_oof_months": list(SEED_MONTHS),
        "development_months": list(DEVELOPMENT_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r47_registered",
        next_action="Fit four R32 seed OOF components, then run the registered adaptive development blend",
        r47_run_id=run_id)
    print(json.dumps({"run_id": run_id, "registered": True,
                      "expected_fit_starts": len(FIT_PLAN)}, indent=2))
    return card


def _fit_seed_oof(run_id: str, output: Path, source: dict, inputs: dict,
                  fit_details: list[dict], fit_ids: list[str]) -> pd.DataFrame:
    from data import hide_answers

    parent_rows = inputs["seed_parent"]
    frames = []
    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(r32.VARIANT)
    had_variant = r32.VARIANT in greedy.VARIANTS
    greedy.VARIANTS[r32.VARIANT] = dict(r32.VARIANT_SPEC)
    active: dict[str, object] = {}

    def record_fit(_fit_id_original, fit_fn):
        context = active.copy()
        fit_id = (f"{campaign.PROTOCOL['campaign_id']}:{CARD}:past_oof_seed:"
                  f"{context['month']}:{context['component']}:{run_id}")
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": "past_oof_seed",
            "run_id": run_id, "fit_id": fit_id, "track": TRACK,
            "month": context["month"], "component": context["component"],
            "training_rows": context["training_rows"],
            "evaluation_rows": context["evaluation_rows"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        fit_ids.append(fit_id)
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": "past_oof_seed",
                "run_id": run_id, "fit_id": fit_id, "month": context["month"],
                "component": context["component"], "status": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        path = output / "seed_components" / str(context["month"]) / f"{context['component']}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions = np.asarray(fitted["predictions"], dtype=float)
        pd.DataFrame({"ml_num": context["evaluation_ids"],
                      "predicted_price": predictions}).to_csv(path, index=False)
        detail = {
            "fit_id": fit_id, "month": context["month"], "component": context["component"],
            "seconds": time.perf_counter() - started,
            "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "feature_columns": fitted["feature_columns"],
            "neighbor_comparable_feature_count": sum(str(col).startswith("nn_")
                                                       for col in fitted["feature_columns"]),
            "prediction_sha256": fitted["prediction_sha256"],
            "saved_prediction_path": str(path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "saved_prediction_sha256": _sha256(path),
        }
        if detail["neighbor_comparable_feature_count"] != 7:
            raise RuntimeError(f"R32 comparable features are missing in {context['month']}/{context['component']}")
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "stage": "past_oof_seed",
            "run_id": run_id, **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in SEED_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if pd.to_datetime(train.date_sold, errors="raise").ge(cutoff).any():
                raise RuntimeError(f"R47 {month} seed train rows reach the fold cutoff")
            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            ids = homes.ml_num.astype(str).to_numpy()
            if not np.array_equal(ids, truth.ml_num.astype(str).to_numpy()):
                raise RuntimeError(f"R47 seed query IDs do not align to truth for {month}")
            route_train = greedy._specialist_mask(train, "Detached")
            route_eval = greedy._specialist_mask(homes, "Detached")
            expected = inputs["preflight"]["seed_support"][month]
            actual = {
                "training_rows": int(len(train)),
                "detached_training_rows": int(route_train.sum()),
                "evaluation_rows": int(len(homes)),
                "detached_evaluation_rows": int(route_eval.sum()),
            }
            if actual != expected:
                raise RuntimeError(f"R47 {month} support changed after preregistration: {actual}")

            components = {}
            for component, fit_rows, eval_rows in (
                ("base", train, homes),
                ("expert", train.loc[route_train].copy(), homes.loc[route_eval].copy()),
            ):
                transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
                active.update({
                    "month": month, "component": component,
                    "training_rows": int(len(fit_rows)),
                    "evaluation_rows": int(len(eval_rows)),
                    "evaluation_ids": eval_rows.ml_num.astype(str).to_numpy(),
                })
                try:
                    fitted = greedy._fit_history_candidate(
                        month, TRACK, train, eval_rows, fit_rows=fit_rows,
                        base_variant="G06_lot_garage", variant=r32.VARIANT,
                        fit_tag=f"R47-R32-{component}-{month}",
                    )
                finally:
                    restore()
                components[component] = fitted
            variant = greedy._blend_specialist_predictions(
                np.round(components["base"]["predictions"]),
                np.round(components["expert"]["predictions"]),
                homes, "Detached", 0.75,
            )
            ref = parent_rows.loc[parent_rows.month.astype(str).eq(month)].copy()
            if ref.ml_num.astype(str).duplicated().any() or set(ref.ml_num.astype(str)) != set(ids):
                raise RuntimeError(f"R40 R22 parent seed OOF rows changed for {month}")
            ref["ml_num"] = ref.ml_num.astype(str)
            ref = ref.set_index("ml_num").loc[ids].reset_index()
            if not np.array_equal(ref.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
                raise RuntimeError(f"R40 R22 parent seed labels changed for {month}")
            frames.append(pd.DataFrame({
                "month": month, "track": TRACK, "ml_num": ids,
                "date_sold": pd.to_datetime(heldout.date_sold.to_numpy(), errors="raise"),
                "sold_price": truth.sold_price.to_numpy(float),
                "parent_prediction": ref.parent_prediction.to_numpy(float),
                "variant_prediction": variant,
            }))
    finally:
        greedy._fit_record = old_fit_record
        if had_variant:
            greedy.VARIANTS[r32.VARIANT] = old_variant
        else:
            greedy.VARIANTS.pop(r32.VARIANT, None)
    return pd.concat(frames, ignore_index=True)


def _write_provenance(output: Path, inputs: dict) -> dict:
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(r22.__file__), Path(r32.__file__),
               Path(r33_screen.__file__), Path(weight_selection.__file__),
               Path(scoring.__file__), Path(campaign.__file__), Path(greedy.__file__),
               Path(dates_helper.__file__)]
    snapshots = {}
    hashes = {}
    for index, source in enumerate(sources):
        key = str(source.relative_to(campaign.ROOT)).replace("\\", "/")
        name = f"src_{index:02d}{source.suffix}.source"
        shutil.copy2(source, provenance / name)
        snapshots[key] = name
        hashes[key] = _sha256(source)
    return {"source_snapshots": snapshots, "code_sha256": hashes,
            "input_sha256": inputs["preflight"]["input_sha256"]}


def run() -> dict:
    inputs = _load_inputs()
    card = campaign.PROTOCOL["cards"].get(CARD)
    if (not card or card.get("status") != "registered"
            or card.get("stage_status") != "registered"
            or not card.get("run_id")):
        raise RuntimeError("R47 must be preregistered with verified source artifacts before fits")
    if card.get("preflight") != inputs["preflight"]:
        raise RuntimeError("R47 source hashes or support changed after preregistration")
    run_id = card["run_id"]
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    source = budget._source_identity()
    provenance = _write_provenance(output, inputs)
    fit_details: list[dict] = []
    fit_ids: list[str] = []
    campaign._append({
        "event": "card_start", "card": CARD,
        "stage": "past_oof_seed_and_development_blend", "run_id": run_id,
        "track": TRACK, "expected_fit_starts": len(FIT_PLAN),
        "seed_oof_months": list(SEED_MONTHS),
        "development_months": list(DEVELOPMENT_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r47_running",
        next_action="Fit R32 seed OOF models, then evaluate the past-only R22/R32 blend",
        r47_run_id=run_id)
    try:
        seed_oof = _fit_seed_oof(run_id, output, source, inputs, fit_details, fit_ids)
        dev_oof = inputs["development_oof"].copy()
        oof = pd.concat([seed_oof, dev_oof], ignore_index=True)
        oof["month"] = oof.month.astype(str)
        oof["ml_num"] = oof.ml_num.astype(str)
        if (oof.duplicated(["month", "ml_num"]).any()
                or set(oof.month.astype(str)) != set((*SEED_MONTHS, *DEVELOPMENT_MONTHS))):
            raise RuntimeError("R47 OOF dataset does not cover exactly its preregistered safe months")
        oof.to_csv(output / "past_oof_predictions.csv", index=False)

        selections = {}
        paired_parts = []
        for month in DEVELOPMENT_MONTHS:
            cutoff = pd.Period(month, freq="M").start_time
            past = oof.loc[pd.to_datetime(oof.date_sold, errors="raise").lt(cutoff)].copy()
            picked = weight_selection.select_weight(
                oof, cutoff,
                minimum_rows=int(card["weight_selection"]["minimum_past_oof_rows"]),
                max_mape_regression_percent=float(card["weight_selection"]["past_mape_regression_max_percent"]),
            )
            if len(past) != picked["oof_rows"] or (
                len(past) and not pd.to_datetime(past.date_sold, errors="raise").lt(cutoff).all()
            ):
                raise RuntimeError(f"R47 weight selection used non-past rows for {month}")
            if picked["reason"] not in {
                "selected_from_strictly_past_oof", "insufficient_past_oof_rows",
                "no_weight_passed_mape_guard",
            }:
                raise RuntimeError(f"unexpected R47 past-OOF selection reason for {month}")
            weight = float(picked["weight"])
            selections[month] = {**picked, "fallback_to_r22": picked["reason"] != "selected_from_strictly_past_oof"}
            rows = dev_oof.loc[dev_oof.month.astype(str).eq(month)].copy()
            candidate = weight_selection.blend(
                rows.parent_prediction.to_numpy(float),
                rows.variant_prediction.to_numpy(float), weight,
            )
            paired_parts.append(pd.DataFrame({
                "month": month, "track": TRACK, "ml_num": rows.ml_num.astype(str).to_numpy(),
                "date_sold": pd.to_datetime(rows.date_sold.to_numpy(), errors="raise"),
                "sold_price": rows.sold_price.to_numpy(float),
                "parent_prediction": rows.parent_prediction.to_numpy(float),
                "variant_prediction": rows.variant_prediction.to_numpy(float),
                "selected_r32_share": weight,
                "candidate_prediction": candidate,
            }))
        paired = pd.concat(paired_parts, ignore_index=True)
        if paired.duplicated(["month", "ml_num"]).any():
            raise RuntimeError("R47 paired development rows contain duplicate IDs")
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
            "track": TRACK, "parent": "R22", "variant": "R32",
            "seed_oof_months": list(SEED_MONTHS),
            "development_months": list(DEVELOPMENT_MONTHS),
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "selected_r32_share_by_month": selections,
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
            "source_identity": source, **provenance,
            "preflight": card["preflight"],
            "seed_oof_sha256": _sha256(output / "past_oof_predictions.csv"),
            "paired_predictions_sha256": _sha256(output / "paired_predictions.csv"),
            "result_sha256": _sha256(result_path),
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        _write_json(output / "run_manifest.json", manifest)
        report = [
            "# R47：R22 与 R32 的严格过去 OOF 自适应融合", "",
            f"Run：`{run_id}`。R22 为 off-market 父模型；R32 只增加七个严格 as-of 可比房源特征。",
            f"新增 seed OOF fits：{len(fit_details)}；八个月门槛通过：{qualifies}。",
            "每个月只使用该月开始日前的 R22/R32 OOF 行选择权重；样本少于 1,000 或 MAPE 守门无可用权重时回退 R22。价格空间融合后仅取整一次。",
            f"Pooled MdAPE：{metrics['parent']['mdape_percent']:.5f}% → "
            f"{metrics['candidate']['mdape_percent']:.5f}%；相对改善 "
            f"{metrics['relative_mdape_improvement_percent']:+.3f}%；MAPE 相对变化 "
            f"{metrics['mape_relative_change_percent']:+.3f}%；"
            f"{metrics['months_improved']}/8 月改善；bootstrap 上界 "
            f"{metrics['bootstrap_mdape_difference_percent']['upper95']:+.5f} 个百分点；"
            f"目标 <7%：{metrics['target_met']}。",
            "", "## 逐月权重与结果", "",
            "| 月份 | past-OOF 月份/行数 | R32 share | R22 MdAPE | 候选 MdAPE | 相对改善 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for month in DEVELOPMENT_MONTHS:
            picked = selections[month]
            row = metrics["monthly"][month]
            report.append(
                f"| {month} | {','.join(picked['oof_months']) or '无'} / {picked['oof_rows']} | "
                f"{picked['weight']:.2f} | {row['parent']['mdape_percent']:.5f}% | "
                f"{row['candidate']['mdape_percent']:.5f}% | "
                f"{row['relative_mdape_improvement_percent']:+.3f}% |"
            )
        report.extend([
            "", "这是一轮自适应开发分析，不会把已用过的固定起点结果称作独立确认。2025-05/08 保留月份未读取且不具备独立验证资格；隐藏标签未读取。", "",
        ])
        (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

        current = campaign.PROTOCOL["cards"][CARD]
        current.update({
            "status": "development_passed" if qualifies else "development_failed",
            "fit_starts": len(fit_details),
            "development_result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "development_metrics": metrics,
            "selected_r32_share_by_month": {month: float(selections[month]["weight"])
                                            for month in DEVELOPMENT_MONTHS},
            "qualifies_development": qualifies,
        })
        current["stage_status"] = "complete"
        current["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                          encoding="utf-8")
        campaign._append({
            "event": "card_complete", "card": CARD,
            "stage": "past_oof_seed_and_development_blend", "run_id": run_id,
            "fit_starts": len(fit_details),
            "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "qualifies_by_track": {TRACK: qualifies},
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        if qualifies:
            next_action = "R47 passed development gates; preserve the candidate and plan only a disclosed non-independent fixed-origin diagnostic"
            state = "r47_development_passed"
        else:
            next_action = "R47 failed; preserve R22 and register a distinct off-market or list-aware hypothesis"
            state = "r47_development_failed"
        campaign._write_progress(state, next_action=next_action,
            r47_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            r47_fit_starts=len(fit_details))
        print(
            f"R47: parent_mdape={metrics['parent']['mdape_percent']:.5f}% "
            f"candidate_mdape={metrics['candidate']['mdape_percent']:.5f}% "
            f"relative_gain={metrics['relative_mdape_improvement_percent']:+.3f}% "
            f"months_improved={metrics['months_improved']}/8 "
            f"bootstrap_upper95={metrics['bootstrap_mdape_difference_percent']['upper95']:+.5f} "
            f"target_met={metrics['target_met']} pass={qualifies} run={run_id}",
            flush=True,
        )
        return result
    except Exception as exc:
        failure = {
            "run_id": run_id, "card": CARD, "stage": "past_oof_seed_and_development_blend",
            "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_ids), "fit_completions": len(fit_details),
            "fit_ids": fit_ids, "fit_details": fit_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        _write_json(output / "failure_checkpoint.json", failure)
        current = campaign.PROTOCOL["cards"][CARD]
        current.update({"stage_status": "failed", "failure": failure["error"],
                        "fit_starts": len(fit_ids),
                        "failure_path": str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/")})
        campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2) + "\n",
                                          encoding="utf-8")
        campaign._append({
            "event": "card_failed", "card": CARD,
            "stage": "past_oof_seed_and_development_blend", "run_id": run_id,
            "fit_starts": len(fit_ids), "fit_completions": len(fit_details),
            "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress("r47_failed",
            next_action="Inspect R47 failure checkpoint and fit ledger before any retry",
            r47_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"))
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
