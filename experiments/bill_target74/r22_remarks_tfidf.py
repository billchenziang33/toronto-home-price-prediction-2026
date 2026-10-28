from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R22_train_fold_tfidf_remarks"
VARIANT = "R22_train_fold_tfidf_remarks"
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = ("2023-03", "2023-07", "2023-11", "2024-03",
                      "2024-06", "2024-09", "2025-01", "2025-03")
TRACKS = ("off_market", "list_aware")
MAX_FEATURES = 256
MIN_DF = 25
MAX_DF = 0.95
NGRAM_RANGE = (1, 2)


class FoldTfidf:
    """Train-only TF-IDF vocabulary for a single model component and fold."""

    def __init__(self, *, max_features: int = MAX_FEATURES, min_df: int | float = MIN_DF,
                 max_df: int | float = MAX_DF):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        )
        self.columns_: list[str] | None = None

    @staticmethod
    def _texts(values) -> pd.Series:
        texts = pd.Series(values, copy=False).fillna("").astype(str)
        if texts.ndim != 1:
            raise ValueError("remarks must be a one-dimensional sequence")
        return texts

    def fit_transform(self, remarks, *, index=None) -> pd.DataFrame:
        texts = self._texts(remarks)
        matrix = self.vectorizer.fit_transform(texts.tolist())
        self.columns_ = [f"remarks_tfidf_{position:03d}" for position in range(matrix.shape[1])]
        values = matrix.toarray()
        frame = pd.DataFrame(values, index=index, columns=self.columns_)
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("training TF-IDF features contain non-finite values")
        return frame

    def transform(self, remarks, *, index=None) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("TF-IDF vocabulary must be fitted on training remarks before transform")
        texts = self._texts(remarks)
        matrix = self.vectorizer.transform(texts.tolist())
        if matrix.shape[1] != len(self.columns_):
            raise RuntimeError("evaluation TF-IDF vocabulary does not match the training vocabulary")
        frame = pd.DataFrame(matrix.toarray(), index=index, columns=self.columns_)
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("evaluation TF-IDF features contain non-finite values")
        return frame

    @property
    def vocabulary_(self) -> dict[str, int]:
        return dict(self.vectorizer.vocabulary_)


def _install_text_features(fit_rows: pd.DataFrame, eval_rows: pd.DataFrame, track: str):
    """Patch one feature builder for one fit, restoring it in the returned cleanup."""
    from model_candidate_v2 import features as frozen_features

    if "remarks" not in fit_rows or "remarks" not in eval_rows:
        raise ValueError("training and evaluation rows must both contain listing remarks")
    if track not in TRACKS:
        raise ValueError(f"unknown track: {track!r}")
    attribute = "off_market_features" if track == "off_market" else "list_aware_features"
    original = getattr(frozen_features, attribute)
    transformer = FoldTfidf()
    fit_identity, eval_identity = id(fit_rows), id(eval_rows)

    def wrapped(homes):
        base = original(homes)
        if id(homes) == fit_identity:
            text = transformer.fit_transform(homes.remarks, index=base.index)
        elif id(homes) == eval_identity:
            text = transformer.transform(homes.remarks, index=base.index)
        else:
            raise RuntimeError("unexpected row frame passed to the fold-scoped TF-IDF feature builder")
        return pd.concat([base, text], axis=1)

    setattr(frozen_features, attribute, wrapped)

    def restore() -> None:
        setattr(frozen_features, attribute, original)

    return transformer, restore


def _fit_id(run_id: str, track: str, month: str, component: str) -> str:
    return f"bill_target74:{CARD}:{track}:{month}:{component}:{run_id}"


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def _mape_relative_change(parent_mape: float, candidate_mape: float) -> float:
    if parent_mape == 0.0:
        return 0.0 if candidate_mape == 0.0 else float("inf")
    return 100.0 * (candidate_mape - parent_mape) / parent_mape


def score_development_confirmation(
    paired: pd.DataFrame,
    *,
    months: tuple[str, ...] = DEVELOPMENT_MONTHS,
    target_mdape_strictly_below: float | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261024,
    minimum_relative_gain_percent: float = 0.5,
    minimum_months_improved: int = 6,
    maximum_worst_month_regression_percent: float = 5.0,
    maximum_mape_regression_percent: float = 1.0,
) -> dict:
    required = {"month", "sold_price", "parent_prediction", "candidate_prediction"}
    if missing := required - set(paired.columns):
        raise ValueError(f"development paired predictions are missing columns: {sorted(missing)}")
    if not months or len(set(months)) != len(months):
        raise ValueError("development months must be a nonempty unique sequence")
    if bootstrap_replicates < 100:
        raise ValueError("at least 100 paired bootstrap replicates are required")
    if minimum_months_improved < 0 or minimum_months_improved > len(months):
        raise ValueError("minimum_months_improved is outside the development window")
    values = paired[["sold_price", "parent_prediction", "candidate_prediction"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("development prices must be finite and positive")
    month_values = paired.month.astype(str)
    if set(month_values.unique()) != set(months):
        raise ValueError("development paired rows do not exactly cover the registered months")
    if "ml_num" in paired and paired.duplicated(["month", "ml_num"]).any():
        raise ValueError("development paired rows must be unique by month and ml_num")

    actual = paired.sold_price.to_numpy(dtype=float)
    parent_prediction = paired.parent_prediction.to_numpy(dtype=float)
    candidate_prediction = paired.candidate_prediction.to_numpy(dtype=float)
    parent_ape = 100.0 * np.abs(parent_prediction - actual) / actual
    candidate_ape = 100.0 * np.abs(candidate_prediction - actual) / actual
    parent_metrics = _metrics(actual, parent_prediction)
    candidate_metrics = _metrics(actual, candidate_prediction)
    relative_gain = 100.0 * (parent_metrics["mdape_percent"] - candidate_metrics["mdape_percent"]) / parent_metrics["mdape_percent"]
    mape_change = _mape_relative_change(parent_metrics["mape_percent"], candidate_metrics["mape_percent"])
    monthly = {}
    grouped_indices = []
    for month in months:
        indices = np.flatnonzero(month_values.eq(month).to_numpy())
        grouped_indices.append(indices)
        month_parent = _metrics(actual[indices], parent_prediction[indices])
        month_candidate = _metrics(actual[indices], candidate_prediction[indices])
        gain = 100.0 * (month_parent["mdape_percent"] - month_candidate["mdape_percent"]) / month_parent["mdape_percent"]
        monthly[month] = {
            "n": int(len(indices)), "parent": month_parent, "candidate": month_candidate,
            "relative_mdape_improvement_percent": float(gain),
        }

    rng = np.random.default_rng(bootstrap_seed)
    differences = np.empty(bootstrap_replicates, dtype=float)
    for replicate in range(bootstrap_replicates):
        sampled_parent = []
        sampled_candidate = []
        for indices in grouped_indices:
            picked = indices[rng.integers(0, len(indices), size=len(indices))]
            sampled_parent.append(parent_ape[picked])
            sampled_candidate.append(candidate_ape[picked])
        differences[replicate] = float(np.median(np.concatenate(sampled_candidate))
                                       - np.median(np.concatenate(sampled_parent)))
    lower95, upper95 = (float(value) for value in np.quantile(differences, [0.025, 0.975]))
    gains = [row["relative_mdape_improvement_percent"] for row in monthly.values()]
    months_improved = sum(gain > 0.0 for gain in gains)
    worst_regression = max(0.0, -min(gains))
    qualifies = bool(
        relative_gain >= minimum_relative_gain_percent
        and months_improved >= minimum_months_improved
        and worst_regression <= maximum_worst_month_regression_percent
        and mape_change <= maximum_mape_regression_percent
        and upper95 < 0.0
    )
    result = {
        "n": int(len(paired)), "months": list(months), "parent": parent_metrics,
        "candidate": candidate_metrics,
        "relative_mdape_improvement_percent": float(relative_gain),
        "mape_relative_change_percent": float(mape_change), "monthly": monthly,
        "months_improved": int(months_improved),
        "worst_month_regression_percent": float(worst_regression),
        "bootstrap_mdape_difference_percent": {"lower95": lower95, "upper95": upper95},
        "bootstrap_replicates": bootstrap_replicates,
        "qualifies_development": qualifies,
    }
    if target_mdape_strictly_below is not None:
        result["target_mdape_strictly_below"] = float(target_mdape_strictly_below)
        result["target_met"] = bool(candidate_metrics["mdape_percent"] < target_mdape_strictly_below)
    return result


def development_fit_plan(tracks: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """Return the fixed component-fit schedule for tracks that passed screening."""
    if not tracks or len(set(tracks)) != len(tracks):
        raise ValueError("development tracks must be nonempty and unique")
    unknown = set(tracks) - set(TRACKS)
    if unknown:
        raise ValueError(f"unknown development track(s): {sorted(unknown)}")
    plan = []
    for track in tracks:
        components = ("base", "expert") if track == "off_market" else ("core",)
        plan.extend((track, month, component)
                    for month in DEVELOPMENT_MONTHS for component in components)
    return plan


def source_sha256_variants(source: bytes) -> set[str]:
    """Hash raw and newline-normalized forms of source snapshots."""
    normalized = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    crlf = normalized.replace(b"\n", b"\r\n")
    return {hashlib.sha256(value).hexdigest() for value in (source, normalized, crlf)}


def fixed_origin_fit_plan(tracks: tuple[str, ...], origins: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """Return fit components needed for reserve-matched parent/candidate origins."""
    if not tracks or len(set(tracks)) != len(tracks) or set(tracks) - set(TRACKS):
        raise ValueError("fixed-origin tracks must be nonempty, unique, and known")
    if not origins or len(set(origins)) != len(origins):
        raise ValueError("fixed origins must be nonempty and unique")
    plan = []
    for origin in origins:
        pd.Timestamp(origin)
        for track in tracks:
            components = (("parent_base", "parent_expert", "candidate_base", "candidate_expert")
                          if track == "off_market" else
                          ("parent_core", "candidate_core", "shared_hgbr"))
            plan.extend((origin, track, component) for component in components)
    return plan


def fixed_origin_training_intervals(origin: str) -> tuple[tuple[str | None, str], ...]:
    """Return half-open train intervals before origin with reserved months omitted."""
    cutoff = pd.Timestamp(origin)
    if cutoff.day != 1:
        raise ValueError("fixed origins must fall on the first day of a month")
    cursor: pd.Timestamp | None = None
    intervals: list[tuple[str | None, str]] = []
    for month in ("2025-05", "2025-08"):
        start = pd.Timestamp(month + "-01")
        end = start + pd.DateOffset(months=1)
        if start >= cutoff:
            continue
        if cursor is None or cursor < start:
            intervals.append((None if cursor is None else str(cursor.date()), str(start.date())))
        cursor = max(cursor, end) if cursor is not None else end
    if cursor is None or cursor < cutoff:
        intervals.append((None if cursor is None else str(cursor.date()), str(cutoff.date())))
    return tuple(intervals)


def load_fixed_origin_training(data_path, origin: str) -> pd.DataFrame:
    """Read only pre-origin row intervals, keeping reserved sale labels off disk reads."""
    cutoff = pd.Timestamp(origin)
    chunks = []
    for start, end in fixed_origin_training_intervals(origin):
        filters = [("date_sold", "<", pd.Timestamp(end))]
        if start is not None:
            filters.insert(0, ("date_sold", ">=", pd.Timestamp(start)))
        chunks.append(pd.read_parquet(data_path, filters=filters))
    if not chunks:
        raise ValueError(f"no training interval exists before fixed origin {origin}")
    train = pd.concat(chunks, ignore_index=True)
    train["date_sold"] = pd.to_datetime(train.date_sold, errors="raise")
    if train.empty or (train.date_sold >= cutoff).any():
        raise ValueError(f"fixed-origin training rows must be nonempty and strictly before {origin}")
    reserved = train.date_sold.dt.to_period("M").astype(str).isin(("2025-05", "2025-08"))
    if reserved.any():
        raise RuntimeError("reserved-month labels entered fixed-origin training")
    return train


def score_fixed_origin_confirmation(
    paired: pd.DataFrame,
    *,
    origins: tuple[str, ...] = ("2024-06-01", "2025-12-01"),
    horizons: tuple[int, ...] = (1, 2, 3, 4),
    maximum_pooled_mdape_regression_percent: float = 0.0,
    maximum_horizon_mdape_regression_percent: float = 5.0,
    maximum_mape_regression_percent: float = 1.0,
) -> dict:
    required = {"origin", "horizon", "sold_price", "parent_prediction", "candidate_prediction"}
    if missing := required - set(paired.columns):
        raise ValueError(f"fixed-origin paired predictions are missing columns: {sorted(missing)}")
    if not origins or not horizons or set(paired.origin.astype(str)) != set(origins):
        raise ValueError("fixed-origin rows must exactly cover the registered origins")
    if set(paired.horizon.astype(int)) != set(horizons):
        raise ValueError("fixed-origin rows must exactly cover the registered horizons")
    coverage = paired.assign(origin=paired.origin.astype(str), horizon=paired.horizon.astype(int))
    expected_cells = {(origin, horizon) for origin in origins for horizon in horizons}
    actual_cells = set(zip(coverage.origin, coverage.horizon))
    if actual_cells != expected_cells:
        raise ValueError("fixed-origin paired rows must cover every origin/horizon cell")
    if {"track", "ml_num", "month"}.issubset(paired.columns):
        if paired.duplicated(["origin", "track", "month", "ml_num"]).any():
            raise ValueError("fixed-origin IDs must be unique within origin, track, and month")
    values = paired[["sold_price", "parent_prediction", "candidate_prediction"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("fixed-origin prices must be finite and positive")

    def metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float | int]:
        from experiments.bill_target74 import runner as campaign

        return campaign._metrics(frame.sold_price.to_numpy(float),
                                 frame[prediction_column].to_numpy(float))

    def relative_change(parent_value: float, candidate_value: float) -> float:
        if parent_value == 0.0:
            return 0.0 if candidate_value == 0.0 else float("inf")
        return 100.0 * (candidate_value - parent_value) / parent_value

    parent = metrics(paired, "parent_prediction")
    candidate = metrics(paired, "candidate_prediction")
    relative_improvement = -relative_change(parent["mdape_percent"], candidate["mdape_percent"])
    mape_change = relative_change(parent["mape_percent"], candidate["mape_percent"])
    by_origin = {}
    for origin in origins:
        part = paired.loc[paired.origin.astype(str).eq(origin)]
        by_origin[origin] = {"parent": metrics(part, "parent_prediction"),
                             "candidate": metrics(part, "candidate_prediction")}
    by_horizon = {}
    for horizon in horizons:
        part = paired.loc[paired.horizon.astype(int).eq(horizon)]
        pmetrics, cmetrics = metrics(part, "parent_prediction"), metrics(part, "candidate_prediction")
        change = relative_change(pmetrics["mdape_percent"], cmetrics["mdape_percent"])
        by_horizon[str(horizon)] = {"parent": pmetrics, "candidate": cmetrics,
                                    "relative_mdape_change_percent": float(change)}
    worst_horizon_regression = max(0.0, max(row["relative_mdape_change_percent"]
                                             for row in by_horizon.values()))
    passes = bool(relative_improvement >= -maximum_pooled_mdape_regression_percent
                  and worst_horizon_regression <= maximum_horizon_mdape_regression_percent
                  and mape_change <= maximum_mape_regression_percent)
    return {"n": int(len(paired)), "origins": list(origins), "horizons": list(horizons),
            "parent": parent, "candidate": candidate,
            "relative_mdape_improvement_percent": float(relative_improvement),
            "mape_relative_change_percent": float(mape_change), "by_origin": by_origin,
            "by_horizon": by_horizon,
            "worst_horizon_mdape_regression_percent": float(worst_horizon_regression),
            "passes_fixed_origin": passes}


def _verified_screen_result(campaign, expected_run_id: str | None = None) -> tuple[dict, Path, dict]:
    """Load an R22 screen only after checking its result and paired-row hashes."""
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    relative = progress.get("r22_result")
    if not relative:
        raise RuntimeError("campaign progress does not identify a completed R22 screen")
    result_path = campaign.ROOT / relative
    if not result_path.is_file():
        raise FileNotFoundError(f"R22 screen result is missing: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = result_path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paired_path = result_path.parent / "paired_predictions.csv"
    if result.get("card") != CARD or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]:
        raise RuntimeError("R22 screen identity does not match the active campaign")
    if expected_run_id and result.get("run_id") != expected_run_id:
        raise RuntimeError(f"requested R22 screen {expected_run_id!r} differs from latest completed screen")
    if (manifest.get("run_id") != result.get("run_id")
            or manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)):
        raise RuntimeError("R22 screen result or paired prediction hash does not match its manifest")
    if manifest.get("fit_starts") != result.get("fit_starts"):
        raise RuntimeError("R22 screen fit count differs between result and manifest")
    source_key = "experiments/bill_target74/r22_remarks_tfidf.py"
    source_copy = result_path.parent / "provenance" / "r22_remarks_tfidf_screen_source.py"
    expected_source_hash = manifest.get("code_sha256", {}).get(source_key)
    if (not source_copy.is_file() or not expected_source_hash
            or expected_source_hash not in source_sha256_variants(source_copy.read_bytes())):
        raise RuntimeError("R22 screen source snapshot does not match the run manifest")
    return result, result_path, manifest


def _save_component_predictions(output, track: str, month: str, component: str,
                                ids, predictions: np.ndarray) -> dict:
    frame = pd.DataFrame({"ml_num": pd.Series(ids, dtype=str).to_numpy(),
                          "predicted_price": np.asarray(predictions, dtype=float)})
    if (frame.ml_num.duplicated().any() or not np.isfinite(frame.predicted_price).all()
            or (frame.predicted_price <= 0.0).any()):
        raise ValueError(f"invalid {track}/{month}/{component} component predictions")
    path = output / "component_predictions" / track / month / f"{component}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {"path": str(path.relative_to(output)).replace("\\", "/"),
            "sha256": _sha256(path), "rows": len(frame)}


def run_r22_development_confirmation(screen_run_id: str | None = None) -> dict:
    """Confirm only screen-qualified R22 tracks on the frozen eight-month window."""
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74.r03_catboost import _verified_hgbr_cache
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    screen, screen_path, screen_manifest = _verified_screen_result(campaign, screen_run_id)
    eligible_tracks = tuple(track for track in TRACKS
                            if screen.get("tracks", {}).get(track, {}).get("qualifies_screen") is True)
    if not eligible_tracks:
        raise RuntimeError("no R22 track passed its frozen two-month screen")
    source = budget._source_identity()
    run_id = "bill_target74_r22_development_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    plan = development_fit_plan(eligible_tracks)
    campaign._append({"event": "card_start", "card": CARD, "stage": "development_confirmation",
                      "run_id": run_id, "screen_run_id": screen["run_id"],
                      "tracks": list(eligible_tracks), "development_months": list(DEVELOPMENT_MONTHS),
                      "expected_fit_starts": len(plan),
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})

    fit_details = []
    component_manifest = []
    rows_by_track = {track: [] for track in eligible_tracks}
    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    fit_context: dict[str, object] = {}

    def record_fit(_original_fit_id: str, fit_fn):
        if not fit_context:
            raise RuntimeError("R22 development fit context was not initialized")
        track, month, component = (fit_context[key] for key in ("track", "month", "component"))
        fit_id = _fit_id(run_id, track, month, component)
        campaign._append({"event": "fit_start", "card": CARD, "stage": "development_confirmation",
                          "run_id": run_id, "fit_id": fit_id, "track": track,
                          "month": month, "component": component,
                          "data_sha256": source["data_sha256"]})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD,
                              "stage": "development_confirmation", "run_id": run_id,
                              "fit_id": fit_id, "track": track, "month": month,
                              "component": component, "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        seconds = time.perf_counter() - started
        saved = _save_component_predictions(output, track, month, component,
                                            fit_context["evaluation_ids"], fitted["predictions"])
        detail = {"fit_id": fit_id, "track": track, "month": month, "component": component,
                  "seconds": seconds, "model_family": fitted["model_family"],
                  "model_config": fitted["model_config"], "target": fitted["target"],
                  "training_rows": fitted["training_rows"],
                  "evaluation_rows": fitted["evaluation_rows"],
                  "feature_columns": fitted["feature_columns"],
                  "text_feature_count": sum(name.startswith("remarks_tfidf_")
                                             for name in fitted["feature_columns"]),
                  "prediction_sha256": fitted["prediction_sha256"],
                  "saved_predictions": saved}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD,
                          "stage": "development_confirmation", "run_id": run_id, **detail,
                          "status": "complete"})
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: _verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in DEVELOPMENT_MONTHS:
            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R22 development training data include sales at or after cutoff: {month}")
            for track in eligible_tracks:
                homes = hide_answers(heldout, track)
                parent_id = campaign.PROTOCOL["parent_by_track"][track]
                parent, parent_path = budget._check_parent_artifact(
                    greedy, month, track, homes, truth, parent_id)
                if track == "off_market":
                    specialist = greedy._specialist_mask(train, "Detached")
                    eval_specialist = greedy._specialist_mask(homes, "Detached")
                    fit_jobs = [
                        ("base", train, homes),
                        ("expert", train.loc[specialist].copy(), homes.loc[eval_specialist].copy()),
                    ]
                    components = {}
                    for component, fit_rows, eval_rows in fit_jobs:
                        transformer, restore = _install_text_features(fit_rows, eval_rows, track)
                        fit_context.update(track=track, month=month, component=component,
                                           evaluation_ids=eval_rows.ml_num.astype(str).tolist())
                        try:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant="G06_lot_garage",
                                variant=VARIANT, fit_tag=f"R22-{component}-{month}", fit_rows=fit_rows)
                        finally:
                            restore()
                        components[component] = fitted
                        component_manifest.append({
                            "track": track, "month": month, "component": component,
                            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                            "target": fitted["target"], "training_rows": fitted["training_rows"],
                            "evaluation_rows": fitted["evaluation_rows"],
                            "text_feature_count": len(transformer.columns_ or []),
                            "vocabulary_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                            "prediction_sha256": fitted["prediction_sha256"],
                        })
                    candidate_prediction = greedy._blend_specialist_predictions(
                        np.round(components["base"]["predictions"]),
                        np.round(components["expert"]["predictions"]), homes, "Detached", 0.75)
                else:
                    transformer, restore = _install_text_features(train, homes, track)
                    fit_context.update(track=track, month=month, component="core",
                                       evaluation_ids=homes.ml_num.astype(str).tolist())
                    try:
                        core = greedy._fit_history_candidate(
                            month, track, train, homes, variant=VARIANT,
                            fit_tag=f"R22-core-{month}", fit_rows=train)
                    finally:
                        restore()
                    frozen = greedy._frozen_hgbr_predictions(month, track, train, homes)
                    candidate_prediction = blend_prices(core["predictions"], frozen, challenger_weight=0.75)
                    component_manifest.append({
                        "track": track, "month": month, "component": "core",
                        "model_family": core["model_family"], "model_config": core["model_config"],
                        "target": core["target"], "training_rows": core["training_rows"],
                        "evaluation_rows": core["evaluation_rows"],
                        "text_feature_count": len(transformer.columns_ or []),
                        "vocabulary_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                        "prediction_sha256": core["prediction_sha256"],
                    })

                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": np.asarray(candidate_prediction, dtype=float)})
                rows_by_track[track].append(helpers._candidate_frame(
                    month, track, truth, parent, candidate))
                campaign._append({"event": "development_fold_complete", "card": CARD,
                                  "stage": "development_confirmation", "run_id": run_id,
                                  "track": track, "month": month, "parent_id": parent_id,
                                  "parent_csv_sha256": _sha256(parent_path),
                                  "date_cutoff_exclusive": cutoff.isoformat(),
                                  "reserved_months_labels_opened": False,
                                  "hidden_labels_opened": False})
                partial = pd.concat(rows_by_track[track], ignore_index=True)
                partial.to_csv(output / f"partial_{track}.csv", index=False)
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "stage": "development_confirmation",
                   "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(fit_details),
                   "fit_details": fit_details,
                   "completed_components": [f"{row['track']}/{row['month']}/{row['component']}"
                                             for row in component_manifest],
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD,
                          "stage": "development_confirmation", "run_id": run_id,
                          "fit_starts": len(fit_details), "error": failure["error"]})
        campaign._write_progress("r22_development_confirmation_failed",
                                 next_action="audit R22 development checkpoint and fit ledger before resuming",
                                 r22_development_failure=str(
                                     (output / "failure_checkpoint.json").relative_to(campaign.ROOT)
                                 ).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record
        greedy._frozen_hgbr_predictions = old_hgbr

    paired = pd.concat([frame for frames in rows_by_track.values() for frame in frames],
                       ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    gates = campaign.PROTOCOL["gates"]
    track_results = {}
    for track, rows in rows_by_track.items():
        paired_track = pd.concat(rows, ignore_index=True)
        target = campaign.PROTOCOL["initial_development_metrics"][track]["target_mdape_strictly_below"]
        track_results[track] = score_development_confirmation(
            paired_track, target_mdape_strictly_below=target,
            bootstrap_replicates=2000, bootstrap_seed=20261024,
            minimum_relative_gain_percent=gates["development_relative_mdape_improvement_percent"],
            minimum_months_improved=gates["development_months_improved_min"],
            maximum_worst_month_regression_percent=gates["development_worst_month_regression_max_percent"],
            maximum_mape_regression_percent=gates["development_mape_regression_max_percent"],
        )
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "stage": "development_confirmation", "screen_run_id": screen["run_id"],
              "parents": campaign.PROTOCOL["parent_by_track"],
              "development_months": list(DEVELOPMENT_MONTHS), "eligible_tracks": list(eligible_tracks),
              "fit_starts": len(fit_details), "tracks": track_results,
              "fit_details": fit_details, "component_manifest": component_manifest,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r22_development_results.json"
    _write_json(result_path, result)
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r22_remarks_tfidf.py",
                    campaign.PACKAGE / "r13_lgbm_bagging.py", campaign.PACKAGE / "r03_catboost.py",
                    campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "runner.py",
                    campaign.ROOT / "model_candidate_v2/features.py",
                    campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/bill_greedy/history.py",
                    campaign.ROOT / "experiments/model_optimization_v2/learners.py"]
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "stage": "development_confirmation", "source_identity": source,
                "screen_run_id": screen["run_id"],
                "screen_result_sha256": _sha256(screen_path),
                "screen_paired_predictions_sha256": screen_manifest["paired_predictions_sha256"],
                "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                                for path in source_files},
                "data_sha256": source["data_sha256"], "fit_starts": len(fit_details),
                "fit_details": fit_details, "component_manifest": component_manifest,
                "paired_predictions_sha256": _sha256(paired_path),
                "result_sha256": _sha256(result_path),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    _write_json(output / "run_manifest.json", manifest)
    lines = ["# R22：八个月开发确认", "",
             f"运行：`{run_id}`；筛选运行：`{screen['run_id']}`。",
             f"本轮 {len(fit_details)} 次 learner fit；仅进入筛选通过的轨道：{', '.join(eligible_tracks)}。", "",
             "| 赛道 | 父 MdAPE | 候选 MdAPE | 相对改善 | MAPE变化 | 逐月改善 | 95%配对区间上界 | 开发门槛 | 目标 |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for track, score in track_results.items():
        gains = ", ".join(f"{month} {row['relative_mdape_improvement_percent']:+.2f}%"
                           for month, row in score["monthly"].items())
        target = score["target_mdape_strictly_below"]
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | {gains} | "
            f"{score['bootstrap_mdape_difference_percent']['upper95']:+.5f} pp | "
            f"{'通过' if score['qualifies_development'] else '未通过'} | "
            f"{'已达到' if score['target_met'] else '未达到'} <{target:.2f}% |"
        )
    lines.extend(["", "逐房配对预测、组件预测、完整组件配置、fit账本与文件哈希均保存在本目录。",
                  "Reserved 2025-05/08 与隐藏标签未读取；通过八个月开发门槛仍不代表已通过固定起点或保留集确认。", ""])
    (output / "research_report.md").write_text("\n".join(lines), encoding="utf-8")
    qualifies = {track: score["qualifies_development"] for track, score in track_results.items()}
    targets_met = {track: score["target_met"] for track, score in track_results.items()}
    campaign._append({"event": "card_complete", "card": CARD,
                      "stage": "development_confirmation", "run_id": run_id,
                      "fit_starts": len(fit_details), "result_path": str(result_path.relative_to(campaign.ROOT)),
                      "qualifies_by_track": qualifies, "targets_met_by_track": targets_met,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress(
        "r22_development_confirmation_complete",
        next_action="run fixed-origin confirmation for R22 tracks passing development gates; "
                    "otherwise retain parents and register distinct candidates",
        r22_development_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=len(fit_details),
    )
    return result


def _verified_development_result(campaign, expected_run_id: str | None = None) -> tuple[dict, Path, dict]:
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    relative = progress.get("r22_development_result")
    if not relative:
        raise RuntimeError("campaign progress does not identify completed R22 development confirmation")
    result_path = campaign.ROOT / relative
    if not result_path.is_file():
        raise FileNotFoundError(f"R22 development result is missing: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = result_path.parent / "run_manifest.json"
    paired_path = result_path.parent / "paired_predictions.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("card") != CARD or result.get("stage") != "development_confirmation"
            or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]):
        raise RuntimeError("R22 development result identity does not match the active campaign")
    if expected_run_id and result.get("run_id") != expected_run_id:
        raise RuntimeError(f"requested development run {expected_run_id!r} differs from latest completed run")
    if (manifest.get("run_id") != result.get("run_id")
            or manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("fit_starts") != result.get("fit_starts")):
        raise RuntimeError("R22 development result or paired predictions fail manifest verification")
    return result, result_path, manifest


def _fixed_origin_fit_id(campaign, run_id: str, origin: str,
                         track: str, component: str) -> str:
    return (f"{campaign.PROTOCOL['campaign_id']}:{CARD}:fixed_origin:"
            f"{origin}:{track}:{component}:{run_id}")


def run_r22_fixed_origin_confirmation(development_run_id: str | None = None) -> dict:
    """Run reserve-safe fixed-origin comparisons for R22 development-qualified tracks."""
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    development, development_path, development_manifest = _verified_development_result(
        campaign, development_run_id)
    eligible_tracks = tuple(track for track in TRACKS
                            if development.get("tracks", {}).get(track, {}).get("qualifies_development") is True)
    if not eligible_tracks:
        raise RuntimeError("no R22 track passed eight-month development confirmation")
    origins = tuple(campaign.PROTOCOL["fixed_origins"])
    plan = fixed_origin_fit_plan(eligible_tracks, origins)
    source = budget._source_identity()
    run_id = "bill_target74_r22_fixedorigin_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "stage": "fixed_origin_confirmation",
                      "run_id": run_id, "development_run_id": development["run_id"],
                      "tracks": list(eligible_tracks), "origins": list(origins),
                      "horizons": 4, "expected_fit_starts": len(plan),
                      "training_policy": "retrain parent and candidate on identical pre-origin rows; "
                                         "omit sealed 2025-05 and 2025-08 sale-month rows before reading them",
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})

    fit_details = []
    component_manifest = []
    paired_parts = {track: [] for track in eligible_tracks}
    old_fit_record = greedy._fit_record
    fit_context: dict[str, object] = {}

    def record_fit(_original_fit_id: str, fit_fn):
        if not fit_context:
            raise RuntimeError("R22 fixed-origin fit context was not initialized")
        origin, track, component = (str(fit_context[key]) for key in ("origin", "track", "component"))
        eval_ids = fit_context["evaluation_ids"]
        fit_id = _fixed_origin_fit_id(campaign, run_id, origin, track, component)
        campaign._append({"event": "fit_start", "card": CARD,
                          "stage": "fixed_origin_confirmation", "run_id": run_id,
                          "fit_id": fit_id, "track": track, "origin": origin,
                          "component": component, "data_sha256": source["data_sha256"]})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD,
                              "stage": "fixed_origin_confirmation", "run_id": run_id,
                              "fit_id": fit_id, "track": track, "origin": origin,
                              "component": component, "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        seconds = time.perf_counter() - started
        saved = _save_component_predictions(output, track, origin[:7], component,
                                            eval_ids, fitted["predictions"])
        detail = {"fit_id": fit_id, "track": track, "origin": origin, "component": component,
                  "seconds": seconds, "model_family": fitted.get("model_family"),
                  "model_config": fitted.get("model_config"), "target": fitted.get("target"),
                  "training_rows": fitted.get("training_rows"),
                  "evaluation_rows": fitted.get("evaluation_rows"),
                  "feature_columns": fitted.get("feature_columns", []),
                  "text_feature_count": sum(name.startswith("remarks_tfidf_")
                                             for name in fitted.get("feature_columns", [])),
                  "prediction_sha256": fitted["prediction_sha256"], "saved_predictions": saved}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD,
                          "stage": "fixed_origin_confirmation", "run_id": run_id,
                          **detail, "status": "complete"})
        return fitted

    greedy._fit_record = record_fit
    try:
        for origin_text in origins:
            origin = pd.Timestamp(origin_text)
            end = origin + pd.DateOffset(months=4)
            train = load_fixed_origin_training(campaign.DATA, origin_text)
            heldout = pd.read_parquet(
                campaign.DATA,
                filters=[("date_sold", ">=", origin), ("date_sold", "<", end)],
            )
            heldout["date_sold"] = pd.to_datetime(heldout.date_sold, errors="raise")
            heldout["ml_num"] = heldout.ml_num.astype(str)
            if heldout.empty or not heldout.ml_num.is_unique:
                raise RuntimeError(f"fixed-origin evaluation rows are invalid at {origin_text}")
            expected_months = {(origin.to_period("M") + offset).strftime("%Y-%m")
                               for offset in range(4)}
            heldout_months = set(heldout.date_sold.dt.to_period("M").astype(str))
            if heldout_months != expected_months:
                raise RuntimeError(f"fixed-origin evaluation horizon is incomplete at {origin_text}")

            for track in eligible_tracks:
                homes = hide_answers(heldout, track).reset_index(drop=True)
                eval_ids = homes.ml_num.astype(str).tolist()
                labels = heldout.sold_price.to_numpy(float)
                if track == "off_market":
                    detached_train = greedy._specialist_mask(train, "Detached")
                    detached_homes = greedy._specialist_mask(homes, "Detached")
                    if not detached_train.any() or not detached_homes.any():
                        raise RuntimeError(f"Detached component lacks rows at {origin_text}")
                    models = {}
                    for role, component, fit_rows, eval_rows, variant, text_features in [
                        ("parent", "parent_base", train, homes, "G06_lot_garage", False),
                        ("parent", "parent_expert", train.loc[detached_train].copy(),
                         homes.loc[detached_homes].copy(), "G18_detached_specialist", False),
                        ("candidate", "candidate_base", train, homes, VARIANT, True),
                        ("candidate", "candidate_expert", train.loc[detached_train].copy(),
                         homes.loc[detached_homes].copy(), VARIANT, True),
                    ]:
                        fit_context.update(origin=origin_text, track=track, component=component,
                                           evaluation_ids=eval_rows.ml_num.astype(str).tolist())
                        if text_features:
                            transformer, restore = _install_text_features(fit_rows, eval_rows, track)
                        else:
                            transformer, restore = None, lambda: None
                        try:
                            fitted = greedy._fit_history_candidate(
                                origin.strftime("%Y-%m"), track, train, eval_rows, cutoff=origin,
                                fit_tag=f"R22-{role}-{component}-{origin_text}",
                                variant=None if not text_features and role == "parent" else
                                        ("G18_detached_specialist" if component == "parent_expert" else
                                         (VARIANT if text_features else "G06_lot_garage")),
                                base_variant="G06_lot_garage", fit_rows=fit_rows)
                        finally:
                            restore()
                        models[component] = fitted
                        component_manifest.append({
                            "track": track, "origin": origin_text, "role": role,
                            "component": component, "model_family": fitted["model_family"],
                            "model_config": fitted["model_config"], "target": fitted["target"],
                            "training_rows": fitted["training_rows"],
                            "evaluation_rows": fitted["evaluation_rows"],
                            "text_feature_count": len(transformer.columns_ or []) if transformer else 0,
                            "vocabulary_terms": (transformer.vectorizer.get_feature_names_out().tolist()
                                                  if transformer else []),
                            "prediction_sha256": fitted["prediction_sha256"],
                        })
                    parent_prediction = greedy._blend_specialist_predictions(
                        np.round(models["parent_base"]["predictions"]),
                        np.round(models["parent_expert"]["predictions"]), homes, "Detached", 0.75)
                    candidate_prediction = greedy._blend_specialist_predictions(
                        np.round(models["candidate_base"]["predictions"]),
                        np.round(models["candidate_expert"]["predictions"]), homes, "Detached", 0.75)
                else:
                    fit_context.update(origin=origin_text, track=track, component="parent_core",
                                       evaluation_ids=eval_ids)
                    parent_core = greedy._fit_history_candidate(
                        origin.strftime("%Y-%m"), track, train, homes, cutoff=origin,
                        fit_tag=f"R22-parent-core-{origin_text}")
                    component_manifest.append({"track": track, "origin": origin_text,
                                               "role": "parent", "component": "parent_core",
                                               "model_family": parent_core["model_family"],
                                               "model_config": parent_core["model_config"],
                                               "target": parent_core["target"],
                                               "training_rows": parent_core["training_rows"],
                                               "evaluation_rows": parent_core["evaluation_rows"],
                                               "text_feature_count": 0, "vocabulary_terms": [],
                                               "prediction_sha256": parent_core["prediction_sha256"]})
                    fit_context.update(origin=origin_text, track=track, component="candidate_core",
                                       evaluation_ids=eval_ids)
                    transformer, restore = _install_text_features(train, homes, track)
                    try:
                        candidate_core = greedy._fit_history_candidate(
                            origin.strftime("%Y-%m"), track, train, homes, cutoff=origin,
                            variant=VARIANT, fit_tag=f"R22-candidate-core-{origin_text}")
                    finally:
                        restore()
                    component_manifest.append({"track": track, "origin": origin_text,
                                               "role": "candidate", "component": "candidate_core",
                                               "model_family": candidate_core["model_family"],
                                               "model_config": candidate_core["model_config"],
                                               "target": candidate_core["target"],
                                               "training_rows": candidate_core["training_rows"],
                                               "evaluation_rows": candidate_core["evaluation_rows"],
                                               "text_feature_count": len(transformer.columns_ or []),
                                               "vocabulary_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                                               "prediction_sha256": candidate_core["prediction_sha256"]})
                    fit_context.update(origin=origin_text, track=track, component="shared_hgbr",
                                       evaluation_ids=eval_ids)
                    hgbr = greedy._fit_baseline_component(
                        origin.strftime("%Y-%m"), track, train, homes, "HGBR",
                        fit_tag=f"R22-shared-HGBR-{origin_text}")
                    component_manifest.append({"track": track, "origin": origin_text,
                                               "role": "shared", "component": "shared_hgbr",
                                               "model_family": hgbr.get("model_family"),
                                               "model_config": hgbr.get("model_config"),
                                               "target": hgbr.get("target"),
                                               "training_rows": hgbr.get("training_rows"),
                                               "evaluation_rows": hgbr.get("evaluation_rows"),
                                               "text_feature_count": 0, "vocabulary_terms": [],
                                               "prediction_sha256": hgbr["prediction_sha256"]})
                    parent_prediction = blend_prices(parent_core["predictions"],
                                                    hgbr["predictions"], challenger_weight=0.75)
                    candidate_prediction = blend_prices(candidate_core["predictions"],
                                                       hgbr["predictions"], challenger_weight=0.75)

                parent_frame = pd.DataFrame({"ml_num": eval_ids, "sold_price": labels,
                                             "predicted_price": parent_prediction})
                candidate_frame = pd.DataFrame({"ml_num": eval_ids, "sold_price": labels,
                                                "predicted_price": candidate_prediction})
                truth = pd.DataFrame({"ml_num": eval_ids, "sold_price": labels})
                if (not np.isfinite(np.asarray(parent_prediction, dtype=float)).all()
                        or not np.isfinite(np.asarray(candidate_prediction, dtype=float)).all()):
                    raise RuntimeError(f"invalid fixed-origin predictions for {origin_text}/{track}")
                joined = helpers._candidate_frame(origin_text[:7], track, truth,
                                                  parent_frame, candidate_frame)
                joined["origin"] = origin_text
                joined["month"] = heldout.date_sold.dt.to_period("M").astype(str).to_numpy()
                joined["horizon"] = greedy._month_horizons(heldout.date_sold, origin)
                paired_parts[track].append(joined)
                campaign._append({"event": "fixed_origin_complete", "card": CARD,
                                  "stage": "fixed_origin_confirmation", "run_id": run_id,
                                  "track": track, "origin": origin_text,
                                  "training_rows": int(len(train)), "evaluation_rows": int(len(heldout)),
                                  "training_intervals": [list(interval) for interval in
                                                         fixed_origin_training_intervals(origin_text)],
                                  "reserved_months_labels_opened": False,
                                  "hidden_labels_opened": False})
                pd.concat(paired_parts[track], ignore_index=True).to_csv(
                    output / f"partial_{track}.csv", index=False)
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "stage": "fixed_origin_confirmation",
                   "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(fit_details),
                   "fit_details": fit_details,
                   "completed_components": [f"{row['track']}/{row['origin']}/{row['component']}"
                                             for row in component_manifest],
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD,
                          "stage": "fixed_origin_confirmation", "run_id": run_id,
                          "fit_starts": len(fit_details), "error": failure["error"]})
        campaign._write_progress("r22_fixed_origin_confirmation_failed",
                                 next_action="audit R22 fixed-origin checkpoint and fit ledger before resuming",
                                 r22_fixed_origin_failure=str(
                                     (output / "failure_checkpoint.json").relative_to(campaign.ROOT)
                                 ).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record

    track_results = {}
    for track, parts in paired_parts.items():
        track_results[track] = score_fixed_origin_confirmation(
            pd.concat(parts, ignore_index=True), origins=origins,
            maximum_pooled_mdape_regression_percent=campaign.PROTOCOL["gates"][
                "fixed_origin_pooled_mdape_regression_max_percent"],
            maximum_horizon_mdape_regression_percent=campaign.PROTOCOL["gates"][
                "fixed_origin_horizon_regression_max_percent"],
            maximum_mape_regression_percent=campaign.PROTOCOL["gates"][
                "fixed_origin_mape_regression_max_percent"],
        )
    paired = pd.concat([part for parts in paired_parts.values() for part in parts], ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "stage": "fixed_origin_confirmation",
              "development_run_id": development["run_id"],
              "origins": list(origins), "horizons": [1, 2, 3, 4],
              "eligible_tracks": list(eligible_tracks), "fit_starts": len(fit_details),
              "training_policy": "retrain parent and candidate on identical pre-origin rows; "
                                 "omit sealed 2025-05 and 2025-08 sale-month rows before reading them",
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "tracks": track_results, "fit_details": fit_details,
              "component_manifest": component_manifest, "data_sha256": source["data_sha256"]}
    result_path = output / "r22_fixed_origin_results.json"
    _write_json(result_path, result)
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r22_remarks_tfidf.py",
                    campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "runner.py", campaign.ROOT / "model_candidate_v2/features.py",
                    campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/bill_greedy/history.py",
                    campaign.ROOT / "experiments/model_optimization_v2/learners.py"]
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "stage": "fixed_origin_confirmation", "source_identity": source,
                "development_run_id": development["run_id"],
                "development_result_sha256": _sha256(development_path),
                "development_paired_predictions_sha256": development_manifest[
                    "paired_predictions_sha256"],
                "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                                for path in source_files},
                "fit_starts": len(fit_details), "fit_details": fit_details,
                "component_manifest": component_manifest,
                "paired_predictions_sha256": _sha256(paired_path),
                "result_sha256": _sha256(result_path),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    _write_json(output / "run_manifest.json", manifest)
    lines = ["# R22：固定起点确认", "",
             f"运行：`{run_id}`；八个月开发运行：`{development['run_id']}`。",
             f"Learner fit starts：{len(fit_details)}。两个起点的 parent 与候选均在相同训练行上重训。",
             "2025-12 起点训练查询按日期区间排除 2025-05 和 2025-08；两个月标签没有读取。", "",
             "| 赛道 | Parent MdAPE | R22 MdAPE | 相对改善 | MAPE变化 | 最差 horizon 退化 | 通过 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for track, score in track_results.items():
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | "
            f"{score['worst_horizon_mdape_regression_percent']:.3f}% | "
            f"{'通过' if score['passes_fixed_origin'] else '未通过'} |"
        )
    lines.extend(["", "逐起点和逐 horizon 指标、逐房预测、所有 fit 配置/预测、源码/数据哈希保存在本目录。",
                  "此固定起点结果只验证两条轨道的泛化；它不代替目标阈值或一次性保留集。", ""])
    (output / "research_report.md").write_text("\n".join(lines), encoding="utf-8")
    passes = {track: score["passes_fixed_origin"] for track, score in track_results.items()}
    campaign._append({"event": "card_complete", "card": CARD,
                      "stage": "fixed_origin_confirmation", "run_id": run_id,
                      "fit_starts": len(fit_details), "result_path": str(result_path.relative_to(campaign.ROOT)),
                      "passes_by_track": passes,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress(
        "r22_fixed_origin_confirmation_complete",
        next_action="register a distinct R22-informed candidate; reserve months stay sealed until both "
                    "development targets are met",
        r22_fixed_origin_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=len(fit_details),
    )
    return result


def _report(result: dict) -> str:
    lines = [
        "# R22：加入训练折拟合的 remarks TF-IDF 特征", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        f"唯一改动：以训练折 remarks 拟合最多 {MAX_FEATURES} 个英文 unigram/bigram TF-IDF 列（min_df={MIN_DF}）；筛选折只 transform，不参与词表或 IDF 拟合。",
        "Off-market 重训 G06 base 与 Detached G18 specialist；list-aware 重训 G01 LightGBM core，并使用已核验的冻结 HGBR 按原 75/25 权重混合。", "",
        "## 双月筛选", "",
        "| 赛道 | 父 MdAPE | 候选 MdAPE | 相对改善 | MAPE变化 | 两月均改善 | 通过筛选 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for track, score in result["tracks"].items():
        both = all(row["relative_mdape_improvement_percent"] > 0.0
                    for row in score["monthly"].values())
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | {'是' if both else '否'} | "
            f"{'是' if score['qualifies_screen'] else '否'} |"
        )
    lines.extend(["", "## 逐月结果", "", "| 赛道 | 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 | 父 MAPE | 候选 MAPE |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, row in score["monthly"].items():
            lines.append(
                f"| {track} | {month} | {row['parent']['mdape_percent']:.5f}% | "
                f"{row['candidate']['mdape_percent']:.5f}% | "
                f"{row['relative_mdape_improvement_percent']:+.3f}% | "
                f"{row['parent']['mape_percent']:.5f}% | {row['candidate']['mape_percent']:.5f}% |"
            )
    lines.extend(["", "## 拟合记录", "", "| 赛道 | 月份 | 组件 | 训练行 | 文本特征列 | fit秒数 | 目标 |",
                  "|---|---|---|---:|---:|---:|---|"])
    for fit in result["fit_details"]:
        lines.append(f"| {fit['track']} | {fit['month']} | {fit['component']} | {fit['training_rows']} | "
                     f"{fit['text_feature_count']} | {fit['seconds']:.1f} | {fit['target']} |")
    lines.extend(["", "## 运行边界", "",
                  f"Learner fit starts：{result['fit_starts']}。词表/IDF 逐组件只从该 fit 的训练文本拟合；不读筛选文本标签，不使用 `days_on_market`、外部 AVM 或预留/隐藏标签。",
                  "只有 pooled MdAPE 相对改善至少 0.25%、两个筛选月分别改善且 MAPE 恶化不超过 1% 的轨道才进入八个月开发确认；否则保留 G19/G01。", "",
                  "- 逐房配对预测：`paired_predictions.csv`",
                  "- 预测分数和组件清单：`r22_screen_results.json`",
                  "- 来源与文件哈希：`run_manifest.json`",
                  "- Campaign 事件账本：`../../fits.jsonl`", ""])
    return "\n".join(lines)


def run_r22_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74.r03_catboost import _verified_hgbr_cache
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    source = budget._source_identity()
    if campaign.PROTOCOL["cards"].get(CARD, {}).get("status") != "registered":
        raise RuntimeError("R22 must be registered in campaign_protocol.json before fitting")
    run_id = "bill_target74_r22_remarks_tfidf_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "tracks": list(TRACKS),
                      "features": {"max_features": MAX_FEATURES, "min_df": MIN_DF,
                                   "max_df": MAX_DF, "ngram_range": list(NGRAM_RANGE),
                                   "token_pattern": r"(?u)\b[a-zA-Z]{2,}\b"},
                      "expected_fit_starts": 6, "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False})
    fit_details = []
    component_manifest = []
    rows_by_track = {track: [] for track in TRACKS}
    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    fit_context: dict[str, str] = {}

    def record_fit(_original_fit_id: str, fit_fn):
        if not fit_context:
            raise RuntimeError("R22 fit context was not initialized")
        track, month, component = (fit_context[key] for key in ("track", "month", "component"))
        fit_id = _fit_id(run_id, track, month, component)
        campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "track": track, "month": month,
                          "component": component, "data_sha256": source["data_sha256"]})
        started = time.perf_counter()
        try:
            result = fit_fn()
        except Exception as exc:
            seconds = time.perf_counter() - started
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "track": track, "month": month,
                              "component": component, "status": "failed", "seconds": seconds,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        seconds = time.perf_counter() - started
        detail = {"fit_id": fit_id, "track": track, "month": month, "component": component,
                  "seconds": seconds, "model_family": result["model_family"],
                  "model_config": result["model_config"], "target": result["target"],
                  "training_rows": result["training_rows"],
                  "evaluation_rows": result["evaluation_rows"],
                  "feature_columns": result["feature_columns"],
                  "text_feature_count": sum(name.startswith("remarks_tfidf_")
                                             for name in result["feature_columns"]),
                  "prediction_sha256": result["prediction_sha256"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "track": track, "month": month,
                          "component": component, "status": "complete", "seconds": seconds,
                          "prediction_sha256": result["prediction_sha256"],
                          "training_rows": result["training_rows"]})
        return result

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: _verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in SCREEN_MONTHS:
            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R22 training data include sales at or after cutoff: {month}")
            for track in TRACKS:
                homes = hide_answers(heldout, track)
                parent_id = campaign.PROTOCOL["parent_by_track"][track]
                parent, parent_path = budget._check_parent_artifact(
                    greedy, month, track, homes, truth, parent_id)
                if track == "off_market":
                    specialist = greedy._specialist_mask(train, "Detached")
                    eval_specialist = greedy._specialist_mask(homes, "Detached")
                    fit_jobs = [
                        ("base", train, homes, "G06_lot_garage"),
                        ("expert", train.loc[specialist].copy(), homes.loc[eval_specialist].copy(),
                         "G06_lot_garage"),
                    ]
                    fitted_components = {}
                    for component, fit_rows, eval_rows, base_variant in fit_jobs:
                        transformer, restore = _install_text_features(fit_rows, eval_rows, track)
                        fit_context.update(track=track, month=month, component=component)
                        try:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant=base_variant,
                                variant=VARIANT, fit_tag=f"R22-{component}-{month}", fit_rows=fit_rows,
                            )
                        finally:
                            restore()
                        fitted_components[component] = fitted
                        component_manifest.append({
                            "track": track, "month": month, "component": component,
                            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                            "target": fitted["target"], "training_rows": fitted["training_rows"],
                            "evaluation_rows": fitted["evaluation_rows"],
                            "text_feature_count": len(transformer.columns_ or []),
                            "vocabulary_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                            "prediction_sha256": fitted["prediction_sha256"],
                        })
                    base_prediction = np.round(fitted_components["base"]["predictions"])
                    expert_prediction = np.round(fitted_components["expert"]["predictions"])
                    candidate_prediction = greedy._blend_specialist_predictions(
                        base_prediction, expert_prediction, homes, "Detached", 0.75)
                else:
                    transformer, restore = _install_text_features(train, homes, track)
                    fit_context.update(track=track, month=month, component="core")
                    try:
                        core = greedy._fit_history_candidate(
                            month, track, train, homes, variant=VARIANT,
                            fit_tag=f"R22-core-{month}", fit_rows=train,
                        )
                    finally:
                        restore()
                    frozen = greedy._frozen_hgbr_predictions(month, track, train, homes)
                    candidate_prediction = blend_prices(core["predictions"], frozen, challenger_weight=0.75)
                    component_manifest.append({
                        "track": track, "month": month, "component": "core",
                        "model_family": core["model_family"], "model_config": core["model_config"],
                        "target": core["target"], "training_rows": core["training_rows"],
                        "evaluation_rows": core["evaluation_rows"],
                        "text_feature_count": len(transformer.columns_ or []),
                        "vocabulary_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                        "prediction_sha256": core["prediction_sha256"],
                    })

                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": np.asarray(candidate_prediction, dtype=float)})
                rows_by_track[track].append(helpers._candidate_frame(month, track, truth, parent, candidate))
                campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                                  "track": track, "month": month, "parent_id": parent_id,
                                  "parent_csv_sha256": _sha256(parent_path),
                                  "date_cutoff_exclusive": cutoff.isoformat()})
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
                   "fit_starts": len(fit_details), "fit_details": fit_details,
                   "completed_components": [f"{row['track']}/{row['month']}/{row['component']}"
                                             for row in component_manifest],
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "fit_starts": len(fit_details), "error": failure["error"]})
        campaign._write_progress("r22_remarks_tfidf_failed",
                                 next_action="inspect R22 failure checkpoint and fit ledger before retry",
                                 r22_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record
        greedy._frozen_hgbr_predictions = old_hgbr

    paired = pd.concat([frame for track_frames in rows_by_track.values() for frame in track_frames],
                       ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {track: helpers._track_result(pd.concat(rows, ignore_index=True))
              for track, rows in rows_by_track.items()}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "parents": campaign.PROTOCOL["parent_by_track"],
              "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
              "tracks": tracks, "fit_details": fit_details,
              "component_manifest": component_manifest,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r22_screen_results.json"
    _write_json(result_path, result)
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r22_remarks_tfidf.py",
                    campaign.PACKAGE / "r13_lgbm_bagging.py", campaign.PACKAGE / "r03_catboost.py",
                    campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "runner.py",
                    campaign.ROOT / "model_candidate_v2/features.py",
                    campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/bill_greedy/history.py",
                    campaign.ROOT / "experiments/model_optimization_v2/learners.py"]
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "source_identity": source,
                "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                                for path in source_files},
                "data_sha256": source["data_sha256"], "fit_starts": len(fit_details),
                "fit_details": fit_details, "component_manifest": component_manifest,
                "paired_predictions_sha256": _sha256(paired_path),
                "result_sha256": _sha256(result_path),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    _write_json(output / "run_manifest.json", manifest)
    (output / "research_report.md").write_text(_report(result), encoding="utf-8")
    qualifies = {track: score["qualifies_screen"] for track, score in tracks.items()}
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "result_path": str(result_path.relative_to(campaign.ROOT)),
                      "qualifies_by_track": qualifies, "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False})
    next_action = ("run sequential eight-month confirmation for passing R22 tracks"
                   if any(qualifies.values()) else
                   "R22 did not qualify; preserve G19/G01 and register a distinct candidate")
    campaign._write_progress("r22_remarks_tfidf_screen_complete", next_action=next_action,
                             r22_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=len(fit_details))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run R22 screen or fixed development confirmation")
    parser.add_argument("--development", action="store_true",
                        help="run eight-month confirmation for tracks passing the saved R22 screen")
    parser.add_argument("--screen-run-id", default=None,
                        help="require the most recent screen result to have this run ID")
    parser.add_argument("--fixed-origins", action="store_true",
                        help="run both reserve-safe fixed-origin confirmations")
    parser.add_argument("--development-run-id", default=None,
                        help="require the latest development result to have this run ID")
    args = parser.parse_args()
    if args.fixed_origins and args.development:
        parser.error("choose either --development or --fixed-origins")
    result = (run_r22_fixed_origin_confirmation(args.development_run_id) if args.fixed_origins
              else run_r22_development_confirmation(args.screen_run_id) if args.development
              else run_r22_screen())
    if args.fixed_origins:
        for track, score in result["tracks"].items():
            print(f"{track}: parent MdAPE={score['parent']['mdape_percent']:.5f}% "
                  f"candidate={score['candidate']['mdape_percent']:.5f}% "
                  f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
                  f"mape_change={score['mape_relative_change_percent']:+.3f}% "
                  f"fixed_origin_pass={score['passes_fixed_origin']}", flush=True)
        from experiments.bill_target74 import runner as campaign
        print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}",
              flush=True)
        return
    if args.development:
        for track, score in result["tracks"].items():
            print(f"{track}: parent MdAPE={score['parent']['mdape_percent']:.5f}% "
                  f"candidate={score['candidate']['mdape_percent']:.5f}% "
                  f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
                  f"mape_change={score['mape_relative_change_percent']:+.3f}% "
                  f"development_pass={score['qualifies_development']} "
                  f"target_met={score['target_met']}", flush=True)
        from experiments.bill_target74 import runner as campaign
        print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}",
              flush=True)
        return
    for track, score in result["tracks"].items():
        print(f"{track}: parent MdAPE={score['parent']['mdape_percent']:.5f}% "
              f"candidate={score['candidate']['mdape_percent']:.5f}% "
              f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
              f"mape_change={score['mape_relative_change_percent']:+.3f}% "
              f"screen_pass={score['qualifies_screen']}", flush=True)
    from experiments.bill_target74 import runner as campaign
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
