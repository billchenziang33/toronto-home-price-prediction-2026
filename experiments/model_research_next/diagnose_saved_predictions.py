from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.model_research_next.diagnostics import compute_paired_residuals, summarize_residuals


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "outputs/model_research_fresh"
OUT = ROOT / "outputs/model_research_next/residual_diagnostics"
MONTHS = ("2024-10", "2024-12", "2025-01", "2025-03")
COMPARISONS = {
    ("off_market", "Gamma"): ("C0", "GAMMA_PRICE"),
    ("list_aware", "Q50"): ("C0", "Q50"),
    ("list_aware", "CatBoost_MAE"): ("C0", "CB_MAE"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predictions(track: str, candidate: str) -> pd.DataFrame:
    frames = []
    for month in MONTHS:
        path = RUN_ROOT / "predictions/development/pipelines" / track / f"{candidate}__{month}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if not set(["ml_num", "month", "sold_price", "predicted_price"]).issubset(frame.columns):
            raise ValueError(f"unexpected prediction schema in {path}")
        frames.append(frame[["ml_num", "month", "sold_price", "predicted_price"]])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/.hf_cache/hub/datasets--Baliyang--SSU_data_dash/snapshots/96a258439e2a85f650699be8339e52d8ac5614b7/listings.parquet"
    source_hash = _sha256(source)
    expected_hash = json.loads((RUN_ROOT / "freeze_manifest.json").read_text(encoding="utf-8"))["data_sha256"]
    if source_hash != expected_hash:
        raise ValueError("training source hash differs from the completed fresh-fold run")

    # This audit reads only labeled development records before 2025-04-01. It
    # never opens the reserved May/August 2025 folds or hidden-test files.
    source_rows = pd.read_parquet(
        source,
        columns=["ml_num", "date_sold", "sold_price", "house_type_name", "house_style", "municipality", "community"],
        filters=[("date_sold", "<", pd.Timestamp("2025-04-01"))],
    )
    used_ids = set()
    feature_columns = ["house_type_name", "house_style", "municipality", "community"]
    results = {}
    for (track, label), (parent_name, candidate_name) in COMPARISONS.items():
        parent = _predictions(track, parent_name)
        candidate = _predictions(track, candidate_name)
        used_ids.update(parent.ml_num.astype(str))
        eval_rows = parent[["ml_num", "month", "sold_price"]].copy()
        eval_rows["month"] = eval_rows.month.astype(str)
        if eval_rows.duplicated(["month", "ml_num"]).any():
            raise ValueError(f"duplicate scored homes for {track}/{label}")
        keys = set(eval_rows.ml_num.astype(str))
        homes = source_rows.loc[source_rows.ml_num.astype(str).isin(keys)].copy()
        if homes.ml_num.astype(str).duplicated().any():
            raise ValueError("source listings contain duplicate IDs")
        if not set(keys).issubset(set(homes.ml_num.astype(str))):
            raise ValueError("source feature rows do not cover all saved predictions")
        source_keyed = homes.assign(ml_num=homes.ml_num.astype(str)).set_index("ml_num")
        eval_keyed = eval_rows.set_index("ml_num")
        source_month = pd.to_datetime(source_keyed.loc[eval_keyed.index, "date_sold"]).dt.to_period("M").astype(str)
        if not np.array_equal(source_month.to_numpy(), eval_keyed.month.to_numpy()):
            raise ValueError("source sale months do not match saved prediction months")
        if not np.array_equal(source_keyed.loc[eval_keyed.index, "sold_price"].to_numpy(dtype=float),
                              eval_keyed.sold_price.to_numpy(dtype=float)):
            raise ValueError("source sale prices do not match saved prediction targets")
        # Sale-price bands are descriptive diagnostics formed only after the
        # outer-fold predictions were saved. They are never model inputs.
        price = pd.to_numeric(homes.sold_price, errors="coerce")
        edges = np.unique(price.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float))
        homes["actual_price_band"] = pd.cut(price, bins=[-np.inf, *edges, np.inf],
                                                labels=[f"Q{i+1}" for i in range(len(edges)+1)],
                                                include_lowest=True).astype(str)
        features = homes[["ml_num", *feature_columns, "actual_price_band"]].copy()
        rows = compute_paired_residuals(parent, candidate, features)
        report = summarize_residuals(
            rows,
            group_columns=("month", "actual_price_band", "house_type_name", "house_style", "municipality", "community"),
            min_group_size=100,
        )
        results[f"{track}/{label}"] = report
        rows.to_csv(OUT / f"{track}_{label}_paired_residuals.csv", index=False)

    audit = {
        "data_sha256": source_hash,
        "prediction_source": "outputs/model_research_fresh/predictions/development/pipelines/",
        "months": list(MONTHS),
        "source_filter_exclusive": "2025-04-01",
        "source_columns_read": ["ml_num", "date_sold", "sold_price", *feature_columns],
        "actual_price_band_policy": "post-hoc descriptive segmentation only; never used as model feature or calibration input",
        "min_segment_n": 100,
        "results": results,
    }
    (OUT / "residual_diagnostics.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    summary_lines = ["# Saved-prediction residual diagnostics", "",
                     "Paired on month and listing ID using only the four already-used development folds. Sale-price bands and property/geography splits are descriptive and were not used to fit a model.", ""]
    for key, report in results.items():
        overall = report["overall"]
        summary_lines.append(f"## {key}")
        summary_lines.append("")
        summary_lines.append(
            f"n={overall['n']:,}; MdAPE {overall['parent_mdape_percent']:.4f}% → {overall['candidate_mdape_percent']:.4f}% "
            f"({overall['relative_mdape_improvement']:+.2%}); MAPE {overall['parent_mape_percent']:.4f}% → "
            f"{overall['candidate_mape_percent']:.4f}% ({overall['relative_mape_change']:+.2%}); "
            f"median signed error {overall['parent_median_signed_error_percent']:+.3f}% → "
            f"{overall['candidate_median_signed_error_percent']:+.3f}%.")
        summary_lines.append("")
        for group_name in ("month", "actual_price_band", "house_type_name", "house_style", "municipality"):
            cells = report["by_group"].get(group_name, {})
            if not cells:
                continue
            sorted_cells = sorted(cells.items(), key=lambda item: item[1]["relative_mdape_improvement"], reverse=True)
            summary_lines.append(f"- `{group_name}`: " + "; ".join(
                f"{name} n={cell['n']}, MdAPE change {cell['relative_mdape_improvement']:+.1%}, MAPE change {cell['relative_mape_change']:+.1%}"
                for name, cell in sorted_cells[:12]))
        summary_lines.append("")
    (OUT / "residual_diagnostics.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote {OUT / 'residual_diagnostics.md'} using {len(used_ids):,} paired homes across {len(MONTHS)} months.")


if __name__ == "__main__":
    main()
