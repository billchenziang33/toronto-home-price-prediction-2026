from __future__ import annotations

import numpy as np
import pandas as pd


def _score(frame: pd.DataFrame) -> dict:
    actual = frame.sold_price.to_numpy(dtype=float)
    prediction = frame.predicted_price.to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all() or (actual <= 0).any() or (prediction <= 0).any():
        raise ValueError("paired evaluation needs finite positive actual and predicted prices")
    ape = 100.0 * np.abs(prediction - actual) / actual
    return {"n": len(frame), "mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def _paired_arrays(parent: pd.DataFrame, candidate: pd.DataFrame):
    keys = ["month", "ml_num"]
    required = set(keys + ["sold_price", "predicted_price"])
    for label, frame in [("parent", parent), ("candidate", candidate)]:
        if required - set(frame.columns):
            raise ValueError(f"{label} is missing paired evaluation columns")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} contains duplicate month/home keys")
    left = parent.set_index(keys).sort_index()
    right = candidate.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("parent and candidate home/month keys do not align")
    if not np.array_equal(left.sold_price.to_numpy(), right.sold_price.to_numpy()):
        raise ValueError("parent and candidate targets do not align")
    return left.reset_index(), right.reset_index()


def _bootstrap_ci(parent_ape: np.ndarray, candidate_ape: np.ndarray, repeats: int, seed: int,
                  chunk_size: int = 100) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=float)
    for start in range(0, repeats, chunk_size):
        count = min(chunk_size, repeats - start)
        index = rng.integers(0, len(parent_ape), size=(count, len(parent_ape)))
        samples[start:start + count] = np.median(candidate_ape[index], axis=1) - np.median(parent_ape[index], axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def _month_cluster_ci(parent: pd.DataFrame, candidate: pd.DataFrame, repeats: int, seed: int = 1) -> list[float]:
    months = sorted(parent.month.astype(str).unique())
    if len(months) < 2:
        return [float("nan"), float("nan")]
    parent_by_month = {}
    candidate_by_month = {}
    for month in months:
        p = parent.loc[parent.month.astype(str).eq(month)]
        c = candidate.loc[candidate.month.astype(str).eq(month)]
        parent_by_month[month] = 100.0 * np.abs(p.predicted_price.to_numpy(float) - p.sold_price.to_numpy(float)) / p.sold_price.to_numpy(float)
        candidate_by_month[month] = 100.0 * np.abs(c.predicted_price.to_numpy(float) - c.sold_price.to_numpy(float)) / c.sold_price.to_numpy(float)
    rng = np.random.default_rng(seed)
    picked = rng.integers(0, len(months), size=(repeats, len(months)))
    deltas = []
    for indices in picked:
        p = np.concatenate([parent_by_month[months[i]] for i in indices])
        c = np.concatenate([candidate_by_month[months[i]] for i in indices])
        deltas.append(float(np.median(c) - np.median(p)))
    return [float(value) for value in np.percentile(deltas, [2.5, 97.5])]


def evaluate_increment(parent: pd.DataFrame, candidate: pd.DataFrame, repeats: int = 10000,
                       seed: int = 0) -> dict:
    p, c = _paired_arrays(parent, candidate)
    p_score, c_score = _score(p), _score(c)
    p_month = {month: _score(frame) for month, frame in p.groupby("month", sort=True)}
    c_month = {month: _score(frame) for month, frame in c.groupby("month", sort=True)}
    monthly = {str(month): (p_month[month]["mdape_percent"] - c_month[month]["mdape_percent"]) /
               p_month[month]["mdape_percent"] for month in p_month}
    p_ape = 100.0 * np.abs(p.predicted_price.to_numpy(float) - p.sold_price.to_numpy(float)) / p.sold_price.to_numpy(float)
    c_ape = 100.0 * np.abs(c.predicted_price.to_numpy(float) - c.sold_price.to_numpy(float)) / c.sold_price.to_numpy(float)
    relative = (p_score["mdape_percent"] - c_score["mdape_percent"]) / p_score["mdape_percent"]
    mape_change = (c_score["mape_percent"] - p_score["mape_percent"]) / p_score["mape_percent"]
    paired_ci = _bootstrap_ci(p_ape, c_ape, repeats, seed)
    month_ci = _month_cluster_ci(p, c, repeats, seed + 1)
    worst_regression = max((max(0.0, -value) for value in monthly.values()), default=0.0)
    passes = bool(relative >= 0.01 and sum(value > 0 for value in monthly.values()) >= 3 and
                   worst_regression <= 0.05 and mape_change <= 0.01 and paired_ci[1] < 0.0)
    return {"n": p_score["n"], "parent": p_score, "candidate": c_score,
            "relative_mdape_improvement": relative, "mape_relative_change": mape_change,
            "monthly_relative_improvement": monthly,
            "months_improved": sum(value > 0 for value in monthly.values()),
            "worst_month_relative_regression": worst_regression,
            "paired_home_bootstrap": {"ci95_percentage_points": paired_ci, "repeats": repeats, "seed": seed},
            "month_cluster_sensitivity": {"ci95_percentage_points": month_ci, "clusters": len(monthly),
                                          "repeats": repeats, "seed": seed + 1},
            "passes_metrics": passes}
