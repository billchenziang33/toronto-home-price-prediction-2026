"""Scoring and correctly paired uncertainty calculations for optimization runs."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np
import pandas as pd

PAIR_KEYS = ["ml_num", "track", "month"]


def validate_pairing(baseline: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return rows in matching key order; reject duplicate, missing, or changed targets."""
    needed = set(PAIR_KEYS + ["sold_price", "predicted_price"])
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = needed - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")
        if frame.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{label} contains duplicate home/month keys")
    left = baseline.set_index(PAIR_KEYS).sort_index()
    right = candidate.set_index(PAIR_KEYS).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("baseline and candidate home/month keys differ")
    if not np.array_equal(left["sold_price"].to_numpy(), right["sold_price"].to_numpy()):
        raise ValueError("baseline and candidate targets differ")
    if not np.isfinite(left["sold_price"].to_numpy()).all() or (left["sold_price"] <= 0).any():
        raise ValueError("sold_price must be finite and positive")
    if not np.isfinite(left["predicted_price"].to_numpy()).all() or (left["predicted_price"] <= 0).any():
        raise ValueError("baseline predictions must be finite and positive")
    if not np.isfinite(right["predicted_price"].to_numpy()).all() or (right["predicted_price"] <= 0).any():
        raise ValueError("candidate predictions must be finite and positive")
    return left.reset_index(), right.reset_index()


def ape_percent(predicted: Iterable[float], actual: Iterable[float]) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if predicted.shape != actual.shape or not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("predictions and targets must be matching finite arrays")
    if (predicted <= 0).any() or (actual <= 0).any():
        raise ValueError("predictions and targets must be positive")
    return 100.0 * np.abs(predicted - actual) / actual


def score(predicted: Iterable[float], actual: Iterable[float]) -> dict[str, float | int]:
    ape = ape_percent(predicted, actual)
    signed = 100.0 * (np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)) / actual
    return {"n": int(len(ape)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape)), "median_signed_percent": float(np.median(signed))}


def paired_mdape_delta(baseline: pd.DataFrame, candidate: pd.DataFrame) -> float:
    left, right = validate_pairing(baseline, candidate)
    base = ape_percent(left.predicted_price, left.sold_price)
    new = ape_percent(right.predicted_price, right.sold_price)
    return float(np.median(new) - np.median(base))


def _bootstrap_values(base: np.ndarray, new: np.ndarray, repeats: int, seed: int, chunk_size: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for start in range(0, repeats, chunk_size):
        count = min(chunk_size, repeats - start)
        indices = rng.integers(0, len(base), size=(count, len(base)))
        # Recompute the difference of the two medians on the same resampled homes.
        values[start:start + count] = np.median(new[indices], axis=1) - np.median(base[indices], axis=1)
    return values


def paired_mdape_bootstrap(baseline: pd.DataFrame, candidate: pd.DataFrame,
                           repeats: int = 1000, seed: int = 0) -> dict[str, float | int | list[float]]:
    left, right = validate_pairing(baseline, candidate)
    base = ape_percent(left.predicted_price, left.sold_price)
    new = ape_percent(right.predicted_price, right.sold_price)
    samples = _bootstrap_values(base, new, repeats, seed)
    return {"n": int(len(base)), "delta_percentage_points": float(np.median(new) - np.median(base)),
            "ci95_percentage_points": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
            "repeats": int(repeats), "seed": int(seed)}


def monthly_scores(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    required = {"month", "predicted_price", "sold_price"}
    if required - set(frame.columns):
        raise ValueError(f"predictions are missing columns: {sorted(required - set(frame.columns))}")
    return {str(month): score(group.predicted_price, group.sold_price)
            for month, group in frame.groupby("month", sort=True)}


def stable_fingerprint(config: dict, data_hash: str, code_hashes: dict[str, str]) -> str:
    payload = {"config": config, "data_sha256": data_hash, "code_sha256": code_hashes}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

