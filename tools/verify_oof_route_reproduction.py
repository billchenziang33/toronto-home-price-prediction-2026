"""Recompute the saved R52 and R151 safe OOF postprocessing from packaged assets.

This audit uses only the five local safe OOF assets. It does not fit learners,
open the training parquet, or access reserved-month or hidden-test labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_submission_assets as assets


def _same_keys(left: pd.DataFrame, right: pd.DataFrame, label: str) -> pd.DataFrame:
    keys = ["month", "ml_num"]
    for frame in (left, right):
        frame["month"] = frame.month.astype(str)
        frame["ml_num"] = frame.ml_num.astype(str)
        if frame.duplicated(keys).any():
            raise RuntimeError(f"{label} contains duplicate month/listing keys")
    joined = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise RuntimeError(f"{label} row keys do not match")
    return joined.drop(columns="_merge")


def _scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {"rows": int(len(ape)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def verify_oof_route_reproduction(output_dir: str | Path = assets.OUTPUT) -> dict:
    output = Path(output_dir).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("OOF route assets must be inside the repository")
    verified = assets.verify_assets(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    reserved = set(manifest["reserved_months"])
    months = sorted(manifest["development_months"])

    r151_base = pd.read_csv(output / "off_market_r151_safe_oof.csv",
                            dtype={"ml_num": str})
    r151_saved = pd.read_csv(output / "off_market_r151_candidate_oof.csv",
                             dtype={"ml_num": str})
    r48_r50 = pd.read_csv(output / "off_market_r48_r50_weight_history.csv",
                          dtype={"ml_num": str})
    list_aware = pd.read_csv(output / "list_aware_r97_safe_oof.csv",
                             dtype={"ml_num": str})
    for label, frame in (("R151 base", r151_base), ("R151 route", r151_saved),
                         ("R48/R50 history", r48_r50), ("R97 OOF", list_aware)):
        observed = set(frame.month.astype(str))
        if observed.intersection(reserved):
            raise RuntimeError(f"{label} contains a registered reserved month")
        if not observed.issubset(set(months) | {"2022-06", "2022-11"}):
            raise RuntimeError(f"{label} contains a month outside the registered safe folds")

    # Recreate R52's second-stage blend. Its share is selected only from earlier
    # R48/R50 OOF rows, using the exact registered selector and rounding rule.
    r52_rows = 0
    r52_months = {}
    for month in months:
        current = r48_r50.loc[r48_r50.month.astype(str).eq(month)].copy()
        earlier = r48_r50.loc[r48_r50.month.astype(str).lt(month)].copy()
        if set(earlier.month.astype(str)).intersection({month, *reserved}):
            raise RuntimeError(f"R52 selector history crosses the cutoff for {month}")
        if len(earlier) >= assets.r52.MIN_PAST_ROWS:
            share, selection = assets.r52._select_share(earlier)
        else:
            share, selection = 0.0, {
                "past_rows": int(len(earlier)),
                "fallback": "fewer than minimum strict-past rows",
            }
        candidate = np.round((1.0 - share) * current.r48_prediction.to_numpy(float)
                             + share * current.r50_prediction.to_numpy(float))
        expected = r151_base.loc[
            r151_base.month.astype(str).eq(month), ["month", "ml_num", "prediction_r52"]
        ].copy()
        paired = _same_keys(
            current[["month", "ml_num"]].copy().assign(recomputed=candidate),
            expected.rename(columns={"prediction_r52": "saved"}),
            f"R52 {month}",
        )
        if not np.array_equal(paired.recomputed.to_numpy(float), paired.saved.to_numpy(float)):
            raise RuntimeError(f"R52 predictions do not reproduce exactly for {month}")
        r52_rows += len(paired)
        r52_months[month] = {"rows": len(paired), "r50_share": float(share),
                             "past_oof_rows": int(len(earlier)),
                             "fallback": selection.get("fallback")}

    # Reapply R151's local spatial residual correction to each month from only
    # strictly earlier safe OOF labels. Query rows contain features only.
    r151_rows = 0
    r151_months = {}
    for month in months:
        history = r151_base.loc[r151_base.month.astype(str).lt(month)].copy()
        current = r151_base.loc[r151_base.month.astype(str).eq(month)].copy()
        cutoff = pd.Timestamp(month + "-01")
        if not history.empty and not pd.to_datetime(history.date_sold).lt(cutoff).all():
            raise RuntimeError(f"R151 history crosses the cutoff for {month}")
        query = current[["house_type_name", "latitude", "longitude"]].copy()
        correction, _diagnostics = assets.r151._fit_correction(history, query, cutoff)
        recomputed = np.rint(current.prediction_r52.to_numpy(float) * np.exp(correction))
        saved = r151_saved.loc[r151_saved.month.astype(str).eq(month)].copy()
        paired = _same_keys(
            current[["month", "ml_num"]].copy().assign(
                recomputed=recomputed, recomputed_correction=correction),
            saved[["month", "ml_num", "candidate_prediction", "correction_log"]].rename(
                columns={"candidate_prediction": "saved", "correction_log": "saved_correction"}
            ),
            f"R151 {month}",
        )
        if not np.array_equal(paired.recomputed.to_numpy(float), paired.saved.to_numpy(float)):
            raise RuntimeError(f"R151 predictions do not reproduce exactly for {month}")
        if not np.allclose(paired.recomputed_correction.to_numpy(float),
                           paired.saved_correction.to_numpy(float), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"R151 corrections do not reproduce for {month}")
        r151_rows += len(paired)
        r151_months[month] = {"rows": len(paired), "past_oof_rows": int(len(history)),
                              "mdape_percent": _scores(
                                  current.sold_price.to_numpy(float), recomputed
                              )["mdape_percent"]}

    route_labels = r151_saved.sold_price.to_numpy(float)
    route_predictions = r151_saved.candidate_prediction.to_numpy(float)
    return {
        "status": "verified",
        "asset_count": verified["asset_count"],
        "r52": {"exact_prediction_rows": r52_rows, "months": r52_months},
        "r151": {"exact_prediction_rows": r151_rows, "months": r151_months,
                 "pooled_metrics": _scores(route_labels, route_predictions)},
        "reserved_labels_read": False,
        "hidden_labels_read": False,
        "training_parquet_opened": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify_oof_route_reproduction(), indent=2, ensure_ascii=False))
