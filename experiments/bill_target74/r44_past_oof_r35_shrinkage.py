"""R44: choose a past-only shrinkage weight for the validated R35 candidate."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as r03
from experiments.bill_target74 import r04_ablation as data_helpers
from experiments.bill_target74 import r19_expert_weight as ledger
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r23_word_char_tfidf as scoring
from experiments.bill_target74 import r36_listaware_semidetached_text as r35_verifier
from experiments.bill_target74 import r40_past_oof_r26_blend as weight_selection
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy


CARD = "R44_past_oof_r35_blend"
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
SEED_MONTHS = ("2022-06", "2022-11")
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = tuple(r22.DEVELOPMENT_MONTHS)
MINIMUM_PAST_OOF_ROWS = 1000
SPECIALIST_WEIGHT = 0.75


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files() -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "r19_expert_weight.py",
        campaign.PACKAGE / "r35_fixed_origin.py",
        campaign.PACKAGE / "r36_listaware_semidetached_text.py",
        campaign.PACKAGE / "r40_past_oof_r26_blend.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
        campaign.ROOT / "experiments/model_optimization_v2/worker.py",
    ]


def _safe_date_metadata(months: tuple[str, ...], data_path: Path) -> pd.DataFrame:
    allowed = set((*SEED_MONTHS, *DEVELOPMENT_MONTHS))
    if not set(months).issubset(allowed) or set(months).intersection({"2025-05", "2025-08"}):
        raise RuntimeError("R44 requested date metadata outside its preregistered safe OOF months")
    frames = []
    for month in months:
        start = pd.Timestamp(month + "-01")
        stop = start + pd.DateOffset(months=1)
        frame = pd.read_parquet(
            data_path,
            engine="pyarrow",
            columns=["ml_num", "date_sold"],
            filters=[[("date_sold", ">=", start), ("date_sold", "<", stop)]],
        )
        frame["ml_num"] = frame.ml_num.astype(str)
        frame["date_sold"] = pd.to_datetime(frame.date_sold, errors="raise")
        if (frame.ml_num.duplicated().any()
                or not frame.date_sold.dt.to_period("M").astype(str).eq(month).all()):
            raise RuntimeError(f"R44 date metadata does not match the filtered month {month}")
        frame["month"] = month
        frames.append(frame[["month", "ml_num", "date_sold"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["month", "ml_num", "date_sold"]
    )


def _attach_oof_dates(rows: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if rows.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R44 OOF rows must be unique by month and listing ID")
    joined = rows.merge(metadata, on=["month", "ml_num"], how="left", validate="one_to_one",
                        indicator=True)
    if not joined._merge.eq("both").all():
        raise RuntimeError("R44 OOF IDs do not match the filtered safe-month date metadata")
    if not joined.date_sold.dt.to_period("M").astype(str).eq(joined.month.astype(str)).all():
        raise RuntimeError("R44 OOF dates do not match their recorded months")
    return joined.drop(columns="_merge")


def _verified_parent():
    parent = r35_verifier._verified_r35_parent(campaign)
    card = campaign.PROTOCOL["cards"].get(CARD, {})
    if card.get("parent_run_id") != parent["screen"]["run_id"]:
        raise RuntimeError("R44's registered parent does not match the verified R35 screen run")
    if card.get("variant_run_id") != parent["screen"]["run_id"]:
        raise RuntimeError("R44's registered R35 variant run does not match the verified artifact")
    return parent


def _read_r35_development_oof(parent: dict, data_path: Path) -> pd.DataFrame:
    rows = pd.read_csv(parent["development_rows_path"], dtype={"ml_num": str})
    rows = rows.loc[rows.track.eq(TRACK)].copy()
    required = {"month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"}
    if missing := required - set(rows.columns):
        raise RuntimeError(f"R35 development OOF rows lack columns: {sorted(missing)}")
    rows["month"] = rows.month.astype(str)
    if set(rows.month) != set(DEVELOPMENT_MONTHS):
        raise RuntimeError("R35 development OOF does not cover exactly the registered months")
    screen_rows = parent["screen_rows"].copy()
    for month in SCREEN_MONTHS:
        expected = rows.loc[rows.month.eq(month)].assign(ml_num=lambda x: x.ml_num.astype(str))
        observed = screen_rows.loc[screen_rows.month.eq(month)].assign(
            ml_num=lambda x: x.ml_num.astype(str)
        )
        if expected.ml_num.duplicated().any() or observed.ml_num.duplicated().any():
            raise RuntimeError(f"R35 has duplicate paired IDs in {month}")
        expected = expected.set_index("ml_num").sort_index()
        observed = observed.set_index("ml_num").sort_index()
        if (not expected.index.equals(observed.index)
                or not np.array_equal(expected.sold_price.to_numpy(float), observed.sold_price.to_numpy(float))
                or not np.array_equal(expected.parent_prediction.to_numpy(float),
                                      observed.parent_prediction.to_numpy(float))
                or not np.array_equal(expected.candidate_prediction.to_numpy(float),
                                      observed.candidate_prediction.to_numpy(float))):
            raise RuntimeError(f"R35 screen and development predictions differ for {month}")
    rows = rows[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]]
    rows = rows.rename(columns={"candidate_prediction": "variant_prediction"})
    metadata = _safe_date_metadata(DEVELOPMENT_MONTHS, data_path)
    return _attach_oof_dates(rows, metadata)


def _seed_oof(month: str, run_id: str, output: Path, data_sha256: str,
              fit_context: dict,
              campaign_module, greedy_module, r22_module, hide_answers,
              blend_prices) -> tuple[pd.DataFrame, dict]:
    listings, train, homes_raw, truth, heldout = greedy_module._load_fold(month)
    cutoff = pd.Period(month, freq="M").start_time
    if pd.to_datetime(train.date_sold, errors="raise").ge(cutoff).any():
        raise RuntimeError(f"R44 seed training rows reach the {month} cutoff")
    homes = hide_answers(heldout, TRACK).reset_index(drop=True)
    ids = homes.ml_num.astype(str).to_numpy()
    if not np.array_equal(ids, truth.ml_num.astype(str).to_numpy()):
        raise RuntimeError(f"R44 seed homes and labels do not align for {month}")
    date_by_id = heldout.assign(ml_num=heldout.ml_num.astype(str)).set_index("ml_num").loc[ids, "date_sold"]
    dates = pd.to_datetime(date_by_id.to_numpy(), errors="raise")
    if not (dates.to_period("M").astype(str) == month).all():
        raise RuntimeError(f"R44 seed evaluation includes rows outside {month}")

    detached_train = greedy_module._specialist_mask(train, HOUSE_TYPE)
    detached_eval = greedy_module._specialist_mask(homes, HOUSE_TYPE)
    registered = campaign_module.PROTOCOL["cards"][CARD]["seed_oof_route_support"][month]
    support = {
        "training_rows": int(len(train)),
        "detached_training_rows": int(detached_train.sum()),
        "evaluation_rows": int(len(homes)),
        "detached_evaluation_rows": int(detached_eval.sum()),
    }
    if support != registered:
        raise RuntimeError(f"R44 preregistered seed support changed for {month}: {support}")

    core_transformer, restore = r22_module._install_text_features(train, homes, TRACK)
    fit_context.update(month=month, component="R22_core", training_rows=len(train),
                       evaluation_rows=len(homes))
    try:
        core = greedy_module._fit_history_candidate(
            month, TRACK, train, homes, variant=r22_module.VARIANT,
            fit_tag=f"R44-R22-core-{month}", fit_rows=train,
        )
        core_feature_count = len(core_transformer.columns_ or [])
    finally:
        restore()

    fit_context.update(month=month, component="HGBR_baseline", training_rows=len(train),
                       evaluation_rows=len(homes))
    frozen_hgbr = greedy_module._frozen_hgbr_predictions(month, TRACK, train, homes)
    parent_prediction = blend_prices(core["predictions"], frozen_hgbr, challenger_weight=0.75)

    expert_train = train.loc[detached_train].copy()
    expert_homes = homes.loc[detached_eval].copy()
    expert_transformer, restore = r22_module._install_text_features(
        expert_train, expert_homes, TRACK,
    )
    fit_context.update(month=month, component="R35_detached_expert",
                       training_rows=len(expert_train), evaluation_rows=len(expert_homes))
    try:
        expert = greedy_module._fit_history_candidate(
            month, TRACK, train, expert_homes, fit_rows=expert_train,
            variant=r22_module.VARIANT, fit_tag=f"R44-R35-expert-{month}",
        )
        expert_feature_count = len(expert_transformer.columns_ or [])
    finally:
        restore()
    specialist_prediction = blend_prices(
        expert["predictions"], frozen_hgbr[detached_eval], challenger_weight=SPECIALIST_WEIGHT,
    )
    candidate_prediction = greedy_module._overlay_specialist_predictions(
        parent_prediction, homes, specialist_prediction, HOUSE_TYPE,
    )
    if not np.array_equal(candidate_prediction[~detached_eval], parent_prediction[~detached_eval]):
        raise RuntimeError(f"R44 changed non-Detached parent rows in {month}")
    frame = pd.DataFrame({
        "month": month,
        "track": TRACK,
        "ml_num": ids,
        "date_sold": dates,
        "sold_price": truth.sold_price.to_numpy(float),
        "parent_prediction": np.asarray(parent_prediction, dtype=float),
        "variant_prediction": np.asarray(candidate_prediction, dtype=float),
    })
    if (not np.isfinite(frame[["sold_price", "parent_prediction", "variant_prediction"]].to_numpy(float)).all()
            or (frame[["sold_price", "parent_prediction", "variant_prediction"]] <= 0.0).any().any()):
        raise RuntimeError(f"R44 seed predictions are invalid for {month}")
    frame.to_csv(output / "oof_seed" / f"{month}.csv", index=False)
    cache = greedy_module.OUTPUT / "frozen_hgbr" / TRACK / f"{month}.csv"
    if not cache.is_file():
        raise RuntimeError(f"R44 expected the frozen HGBR cache after its fit: {cache}")
    return frame, {
        "month": month, "source_month_cutoff": cutoff.isoformat(),
        "support": support, "data_sha256": data_sha256,
        "R22_core_text_feature_count": core_feature_count,
        "R35_expert_text_feature_count": expert_feature_count,
        "HGBR_cache_path": str(cache.relative_to(campaign_module.ROOT)).replace("\\", "/"),
        "HGBR_cache_sha256": _sha256(cache),
        "parent_prediction_sha256": hashlib.sha256(parent_prediction.astype("float64").tobytes()).hexdigest(),
        "variant_prediction_sha256": hashlib.sha256(candidate_prediction.astype("float64").tobytes()).hexdigest(),
    }


def run() -> dict:
    from data import hide_answers
    from experiments.model_optimization_v2.pipeline import blend_prices

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R44 must be preregistered before fitting")
    limit_keys = (
        "campaign_fit_starts", "campaign_minutes", "exploration_fit_starts",
        "exploration_minutes", "final_reserve_fit_starts", "final_reserve_minutes",
        "batch_fit_starts", "batch_minutes", "single_fit_seconds",
    )
    if any(campaign.PROTOCOL["limits"].get(key) is not None for key in limit_keys):
        raise RuntimeError("R44 requires the active campaign's user-authorized unbounded limits")
    parent = _verified_parent()
    source = data_helpers._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R44 data identity differs from the registered campaign")
    if int(card.get("expected_fit_starts", -1)) != 6:
        raise RuntimeError("R44 preregistration must expect exactly six seed-fit starts")
    dev_oof = _read_r35_development_oof(parent, campaign.DATA)
    run_id = "bill_target74_r44_past_oof_r35_blend_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    (output / "oof_seed").mkdir()
    greedy_run_output = output / "greedy_cache"
    greedy_run_output.mkdir()
    parent_files = [
        (parent["screen_path"], "r35_screen_results.json"),
        (parent["screen_rows_path"], "r35_screen_paired_predictions.csv"),
        (parent["screen_manifest_path"], "r35_screen_run_manifest.json"),
        (parent["development_path"], "r35_development_results.json"),
        (parent["development_rows_path"], "r35_development_paired_predictions.csv"),
        (parent["development_manifest_path"], "r35_development_run_manifest.json"),
        (parent["fixed_path"], "r35_fixed_origin_results.json"),
        (parent["fixed_manifest_path"], "r35_fixed_origin_run_manifest.json"),
    ]
    fixed_pairs = parent["fixed_path"].parent / "paired_predictions.csv"
    fixed_manifest = parent["fixed_manifest_path"]
    if not fixed_pairs.is_file():
        raise FileNotFoundError("verified R35 fixed-origin paired rows are missing")
    parent_files.append((fixed_pairs, "r35_fixed_origin_paired_predictions.csv"))
    for source_path, name in parent_files:
        shutil.copy2(source_path, provenance / name)
    dev_oof.to_csv(output / "past_oof_predictions.csv", index=False)

    code_hashes = {}
    source_snapshots = {}
    for index, path in enumerate(_source_files()):
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        code_hashes[key] = _sha256(path)
        snapshot_name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / snapshot_name)
        source_snapshots[key] = snapshot_name

    fit_context: dict[str, object] = {}
    fit_details: list[dict] = []
    fit_start_ids: list[str] = []
    seed_frames: list[pd.DataFrame] = []
    seed_details: list[dict] = []
    old_fit_record = greedy._fit_record
    old_greedy_output = greedy.OUTPUT

    def record_fit(parent_fit_id: str, fit_fn):
        context = fit_context.copy()
        if not context:
            raise RuntimeError("R44 fit context was not initialized")
        fit_id = (f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:past_oof_seed:"
                  f"{context['month']}:{context['component']}:{run_id}")
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": "past_oof_seed",
            "run_id": run_id, "fit_id": fit_id, "parent_fit_id": parent_fit_id,
            "track": TRACK, "month": context["month"], "component": context["component"],
            "training_rows": context["training_rows"], "evaluation_rows": context["evaluation_rows"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        fit_start_ids.append(fit_id)
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": "past_oof_seed",
                "run_id": run_id, "fit_id": fit_id, "track": TRACK,
                "month": context["month"], "component": context["component"],
                "status": "failed", "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        detail = {
            "fit_id": fit_id, "parent_fit_id": parent_fit_id,
            "month": context["month"], "track": TRACK,
            "component": context["component"], "seconds": time.perf_counter() - started,
            "training_rows": fitted.get("training_rows", context["training_rows"]),
            "evaluation_rows": fitted.get("evaluation_rows", context["evaluation_rows"]),
            "model_family": fitted.get("model_family"), "model_config": fitted.get("model_config"),
            "target": fitted.get("target"), "feature_columns": fitted.get("feature_columns", []),
            "prediction_sha256": fitted.get("prediction_sha256"),
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "stage": "past_oof_seed",
            "run_id": run_id, **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    campaign._append({
        "event": "card_start", "card": CARD, "stage": "past_oof_seed_and_screen",
        "run_id": run_id, "track": TRACK, "parent_run_id": card["parent_run_id"],
        "variant_run_id": card["variant_run_id"],
        "initial_parent_run_id": card["initial_parent_run_id"],
        "seed_months": list(SEED_MONTHS), "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": 6,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    greedy._fit_record = record_fit
    greedy.OUTPUT = greedy_run_output
    try:
        for month in SEED_MONTHS:
            seed_frame, detail = _seed_oof(
                month, run_id, output, source["data_sha256"], fit_context,
                campaign, greedy, r22, hide_answers, blend_prices,
            )
            seed_frames.append(seed_frame)
            seed_details.append(detail)
    except Exception as exc:
        checkpoint = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": "past_oof_seed", "status": "failed",
            "fit_starts": len(fit_start_ids), "fit_completions": len(fit_details),
            "fit_ids": fit_start_ids,
            "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (output / "failure_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "stage": "past_oof_seed", "status": "failed",
            "fit_starts": len(fit_start_ids), "fit_completions": len(fit_details),
            "error": checkpoint["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r44_attempt_failed",
            next_action="Inspect the R44 failure checkpoint; preserve all fit events and retry only under a new run_id after correcting the cause",
            r44_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        raise
    finally:
        greedy._fit_record = old_fit_record
        greedy.OUTPUT = old_greedy_output

    if len(fit_details) != 6:
        raise RuntimeError(f"R44 expected six seed fits, recorded {len(fit_details)}")
    seed_oof = pd.concat(seed_frames, ignore_index=True)
    oof = pd.concat([seed_oof, dev_oof], ignore_index=True)
    if oof.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R44 combined past-OOF rows contain duplicate month/ID pairs")
    oof.to_csv(output / "past_oof_predictions.csv", index=False)

    screen_rows = parent["screen_rows"].copy()
    screen_rows["month"] = screen_rows.month.astype(str)
    screen_rows["ml_num"] = screen_rows.ml_num.astype(str)
    paired_parts = []
    weight_details = {}
    for month in SCREEN_MONTHS:
        cutoff = pd.Timestamp(month + "-01")
        selection = weight_selection.select_weight(
            oof, cutoff, minimum_rows=MINIMUM_PAST_OOF_ROWS,
            max_mape_regression_percent=1.0,
        )
        subset = screen_rows.loc[screen_rows.month.eq(month)].copy()
        if subset.empty or subset.ml_num.duplicated().any():
            raise RuntimeError(f"R35 screen source rows are empty or duplicated for {month}")
        weight = float(selection["weight"])
        candidate = weight_selection.blend(
            subset.parent_prediction.to_numpy(float),
            subset.candidate_prediction.to_numpy(float),
            weight,
        )
        paired_parts.append(pd.DataFrame({
            "month": month, "track": TRACK, "ml_num": subset.ml_num.to_numpy(),
            "sold_price": subset.sold_price.to_numpy(float),
            "parent_prediction": subset.candidate_prediction.to_numpy(float),
            "candidate_prediction": candidate,
            "initial_parent_prediction": subset.parent_prediction.to_numpy(float),
            "r22_prediction": subset.parent_prediction.to_numpy(float),
            "r35_prediction": subset.candidate_prediction.to_numpy(float),
            "r35_share": weight,
        }))
        weight_details[month] = selection

    paired = pd.concat(paired_parts, ignore_index=True)
    if set(paired.month.astype(str)) != set(SCREEN_MONTHS):
        raise RuntimeError("R44 paired screen rows do not cover the registered months")
    score = scoring._track_result(paired)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "track": TRACK,
        "parent_run_id": parent["screen"]["run_id"],
        "initial_parent_run_id": card["initial_parent_run_id"],
        "variant_run_id": parent["screen"]["run_id"],
        "seed_months": list(SEED_MONTHS), "screen_months": list(SCREEN_MONTHS),
        "minimum_past_oof_rows": MINIMUM_PAST_OOF_ROWS,
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "seed_oof_details": seed_details,
        "selected_weights_by_screen_month": weight_details,
        "tracks": {TRACK: score},
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
        "reserve_incident_id": campaign.PROTOCOL.get("reserve_label_access_incident", {}).get("incident_id"),
        "safe_oof_months_read": list((*SEED_MONTHS, *DEVELOPMENT_MONTHS)),
        "data_sha256": source["data_sha256"],
    }
    result_path = output / "screen_results.json"
    ledger._write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "source_snapshots": source_snapshots,
        "parent_run_id": parent["screen"]["run_id"],
        "initial_parent_run_id": card["initial_parent_run_id"],
        "parent_screen_result_sha256": _sha256(parent["screen_path"]),
        "parent_screen_rows_sha256": _sha256(parent["screen_rows_path"]),
        "parent_development_result_sha256": _sha256(parent["development_path"]),
        "parent_development_rows_sha256": _sha256(parent["development_rows_path"]),
        "parent_fixed_result_sha256": _sha256(parent["fixed_path"]),
        "parent_fixed_rows_sha256": _sha256(fixed_pairs),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "seed_oof_sha256": _sha256(output / "past_oof_predictions.csv"),
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
        "safe_oof_months_read": list((*SEED_MONTHS, *DEVELOPMENT_MONTHS)),
    }
    ledger._write_json(output / "run_manifest.json", manifest)
    (output / "research_report.md").write_text(
        "# R44：R35 的过去 OOF 权重收缩\n\n"
        f"Run：`{run_id}`；比较基线为已通过固定起点确认的 R35 run `{parent['screen']['run_id']}`。\n\n"
        "每个筛选月的 R35 权重只用严格早于该月的 R22/R35 OOF 行选择，使用 0.05 网格，并约束 OOF MAPE 相对 R22 的退化不超过 1%。筛选候选是在 R22 与 R35 已保存价格预测之间做价格空间线性混合；筛选标签只用于评分，不参与选权重。\n\n"
        f"筛选通过：{score['qualifies_screen']}；R35 父模型 MdAPE：{score['parent']['mdape_percent']:.5f}%；"
        f"收缩候选 MdAPE：{score['candidate']['mdape_percent']:.5f}%；"
        f"相对改善：{score['relative_mdape_improvement_percent']:+.3f}%；"
        f"MAPE 相对变化：{score['mape_relative_change_percent']:+.3f}%。\n\n"
        f"初始 R22 基线 MdAPE：{score['initial_parent']['metrics']['mdape_percent']:.5f}%；"
        f"候选相对 R22 的 MdAPE 改善：{score['initial_parent']['relative_mdape_improvement_percent']:+.3f}%。\n\n"
        f"各月 R35 权重：{json.dumps({month: details['weight'] for month, details in weight_details.items()}, sort_keys=True)}。\n\n"
        "本轮只读预登记 OOF 月份和两个安全筛选月；2025-05/08 与隐藏标签未读取。\n",
        encoding="utf-8",
    )
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": {TRACK: score["qualifies_screen"]},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    campaign.PROTOCOL["cards"][CARD].update({
        "status": "screen_passed" if score["qualifies_screen"] else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_screen": score["qualifies_screen"],
        "screen_metrics": {
            "parent_mdape_percent": score["parent"]["mdape_percent"],
            "candidate_mdape_percent": score["candidate"]["mdape_percent"],
            "relative_mdape_improvement_percent": score["relative_mdape_improvement_percent"],
            "mape_relative_change_percent": score["mape_relative_change_percent"],
            "monthly_relative_mdape_improvement_percent": {
                month: values["relative_mdape_improvement_percent"]
                for month, values in score["monthly"].items()
            },
            "selected_weight_by_month": {
                month: details["weight"] for month, details in weight_details.items()
            },
        },
    })
    ledger._write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._write_progress(
        "r44_screen_complete",
        next_action=("R44 passed its two-month screen; register eight-month development confirmation"
                     if score["qualifies_screen"]
                     else "R44 failed its fixed screen; preserve R35 and register a distinct list-aware hypothesis"),
        latest_fit_count=len(fit_details),
        r44_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
    )
    return result


if __name__ == "__main__":
    result = run()
    score = result["tracks"][TRACK]
    print(
        f"R44: parent_mdape={score['parent']['mdape_percent']:.5f}% "
        f"candidate_mdape={score['candidate']['mdape_percent']:.5f}% "
        f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
        f"pass={score['qualifies_screen']} run={result['run_id']}",
        flush=True,
    )
