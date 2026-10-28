from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r168_listaware_r97_past_blend as registered
from experiments.bill_target74 import runner as campaign


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-10):
        raise RuntimeError(f"{label} differs: stored={actual}, recomputed={expected}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape)), "p90_ape_percent": float(np.quantile(ape, 0.9)),
            "ape_over_20_percent": float(100.0 * np.mean(ape > 20.0))}


def _choose_alpha(history: pd.DataFrame) -> tuple[float, dict]:
    months = sorted(history.month.unique().tolist())
    if len(history) < registered.MIN_HISTORY_ROWS or len(months) < registered.MIN_HISTORY_MONTHS:
        return 0.0, {"past_rows": len(history), "past_months": months,
                     "alpha": 0.0, "fallback": "insufficient_past_support"}
    actual = history.sold_price.to_numpy(float)
    p35 = history.r35_prediction.to_numpy(float)
    p97 = history.r97_prediction.to_numpy(float)
    baseline = _metrics(actual, p35)
    choices = []
    for alpha in registered.ALPHA_GRID:
        score = _metrics(actual, np.rint((1.0 - alpha) * p35 + alpha * p97))
        mape_delta = 100.0 * (score["mape_percent"] - baseline["mape_percent"]) / baseline["mape_percent"]
        if mape_delta <= registered.MAPE_GUARD_PERCENT + 1e-12:
            choices.append((score["mdape_percent"], score["mape_percent"], alpha, mape_delta))
    if not choices:
        return 0.0, {"past_rows": len(history), "past_months": months,
                     "alpha": 0.0, "fallback": "no_mape_feasible_alpha"}
    mdape, mape, alpha, mape_delta = min(choices, key=lambda item: (item[0], item[1], item[2]))
    return alpha, {"past_rows": len(history), "past_months": months, "alpha": alpha,
                   "fallback": None, "selected_past_mdape_percent": float(mdape),
                   "selected_past_mape_percent": float(mape),
                   "selected_past_mape_change_percent": float(mape_delta)}


def _bootstrap(actual: np.ndarray, candidate: np.ndarray, comparator: np.ndarray) -> dict:
    rng = np.random.default_rng(registered.BOOTSTRAP_SEED)
    ca = 100.0 * np.abs(candidate - actual) / actual
    pa = 100.0 * np.abs(comparator - actual) / actual
    diff = np.empty(registered.BOOTSTRAP_REPLICATES, dtype=float)
    for i in range(len(diff)):
        idx = rng.integers(0, len(actual), size=len(actual))
        diff[i] = np.median(ca[idx]) - np.median(pa[idx])
    return {"lower95_pp": float(np.quantile(diff, 0.025)),
            "upper95_pp": float(np.quantile(diff, 0.975)),
            "replicates": registered.BOOTSTRAP_REPLICATES, "seed": registered.BOOTSTRAP_SEED}


def audit(run_id: str) -> dict:
    batch = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path = batch / "development_results.json"
    manifest_path = batch / "run_manifest.json"
    pair_path = batch / "paired_predictions.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered_card = campaign.PROTOCOL["cards"].get(registered.CARD)
    if result.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise RuntimeError("R168 run identity mismatch")
    if registered_card is None or registered_card.get("status") != "development_complete":
        raise RuntimeError("R168 registered card is not in development_complete state")
    if registered_card.get("source_sha256") != sha256(Path(registered.__file__)):
        raise RuntimeError("runner hash differs from registered card")
    if registered_card.get("audit_source_sha256") != sha256(Path(__file__)):
        raise RuntimeError("auditor hash differs from registered card")
    for path, key in ((result_path, "result_sha256"), (pair_path, "paired_predictions_sha256")):
        if sha256(path) != manifest[key]:
            raise RuntimeError(f"{key} mismatch")
    if sha256(Path(registered.__file__)) != manifest["source_sha256"]:
        raise RuntimeError("runner source hash differs from manifest")
    if sha256(Path(__file__)) != manifest["audit_source_sha256"]:
        raise RuntimeError("auditor source hash differs from manifest")
    expected_inputs = registered.input_identity()
    if manifest.get("input_files") != expected_inputs or registered_card.get("inputs") != expected_inputs:
        raise RuntimeError("input hashes or registered paths differ from the immutable card")
    for name, item in expected_inputs.items():
        snapshot = batch / "provenance" / f"{name}_paired_predictions.csv"
        if not snapshot.is_file() or sha256(snapshot) != item["sha256"]:
            raise RuntimeError(f"{name} provenance snapshot hash mismatch")
    if result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened"):
        raise RuntimeError("R168 result reports restricted-label access")
    if manifest.get("reserved_months_labels_opened") or manifest.get("hidden_labels_opened"):
        raise RuntimeError("R168 manifest reports restricted-label access")

    src = registered.load_inputs()
    saved = pd.read_csv(pair_path, dtype={"month": str, "track": str, "ml_num": str})
    key = ["month", "track", "ml_num"]
    if saved.duplicated(key).any() or src.duplicated(key).any():
        raise RuntimeError("duplicate paired keys")
    if set(saved.month.unique()) != set(registered.MONTHS) or set(saved.month.unique()) & {"2025-05", "2025-08"}:
        raise RuntimeError("saved R168 rows exceed the allowed development months")
    if len(saved) != len(src) or not saved[key].equals(src[key]):
        raise RuntimeError("saved predictions do not preserve exact source row order and keys")
    for col in ("sold_price", "r35_prediction", "r97_prediction"):
        if not np.array_equal(saved[col].to_numpy(float), src[col].to_numpy(float)):
            raise RuntimeError(f"saved {col} differs from source inputs")

    expected_fits = []
    for month in registered.MONTHS:
        current_idx = saved.month.eq(month)
        history = saved.loc[saved.month.lt(month)].copy()
        alpha, detail = _choose_alpha(history)
        month_rows = saved.loc[current_idx]
        observed_alpha = month_rows.selected_alpha.to_numpy(float)
        if not np.all(observed_alpha == alpha):
            raise RuntimeError(f"{month} alpha was not constant or was selected incorrectly")
        expected = np.rint((1.0 - alpha) * month_rows.r35_prediction.to_numpy(float)
                           + alpha * month_rows.r97_prediction.to_numpy(float))
        if not np.array_equal(expected, month_rows.r168_prediction.to_numpy(float)):
            raise RuntimeError(f"{month} predictions do not reconstruct from its past-only alpha")
        if len(history) >= registered.MIN_HISTORY_ROWS and history.month.nunique() >= registered.MIN_HISTORY_MONTHS:
            expected_fits.append({"month": month, "alpha": alpha, "past_rows": len(history),
                                  "past_months": sorted(history.month.unique().tolist())})

    actual = saved.sold_price.to_numpy(float)
    candidate = saved.r168_prediction.to_numpy(float)
    comps = {"R35": saved.r35_prediction.to_numpy(float), "R97": saved.r97_prediction.to_numpy(float)}
    stored_scores = result["scores"]
    pooled = {name: _metrics(actual, pred) for name, pred in {**comps, "R168": candidate}.items()}
    for name, metrics in pooled.items():
        for metric in ("n", "mdape_percent", "mape_percent", "p90_ape_percent", "ape_over_20_percent"):
            _assert_close(stored_scores["pooled"][name][metric], metrics[metric], f"pooled {name} {metric}")
    for comparator_name, comparator in comps.items():
        comparison = stored_scores["comparisons"][comparator_name]
        gain = 100.0 * (pooled[comparator_name]["mdape_percent"] - pooled["R168"]["mdape_percent"]) / pooled[comparator_name]["mdape_percent"]
        mape_change = 100.0 * (pooled["R168"]["mape_percent"] - pooled[comparator_name]["mape_percent"]) / pooled[comparator_name]["mape_percent"]
        _assert_close(comparison["relative_mdape_improvement_percent"], gain, f"{comparator_name} MdAPE gain")
        _assert_close(comparison["mape_relative_change_percent"], mape_change, f"{comparator_name} MAPE delta")
        ci = _bootstrap(actual, candidate, comparator)
        for metric, value in ci.items():
            _assert_close(comparison["paired_bootstrap_mdape_difference"][metric], value,
                          f"{comparator_name} bootstrap {metric}")
        for month, group in saved.groupby("month", sort=True):
            before = _metrics(group.sold_price.to_numpy(float), group[f"{comparator_name.lower()}_prediction"].to_numpy(float))
            after = _metrics(group.sold_price.to_numpy(float), group.r168_prediction.to_numpy(float))
            expected_gain = 100.0 * (before["mdape_percent"] - after["mdape_percent"]) / before["mdape_percent"]
            _assert_close(comparison["monthly"][str(month)]["relative_mdape_improvement_percent"], expected_gain,
                          f"{comparator_name}/{month} monthly gain")
        monthly_gains = [
            100.0 * (
                _metrics(group.sold_price.to_numpy(float), group[f"{comparator_name.lower()}_prediction"].to_numpy(float))["mdape_percent"]
                - _metrics(group.sold_price.to_numpy(float), group.r168_prediction.to_numpy(float))["mdape_percent"]
            ) / _metrics(group.sold_price.to_numpy(float), group[f"{comparator_name.lower()}_prediction"].to_numpy(float))["mdape_percent"]
            for _, group in saved.groupby("month", sort=True)
        ]
        expected_gates = {
            "relative_mdape_improvement": gain >= 0.25,
            "months_improved": sum(value > 0 for value in monthly_gains) >= 6,
            "worst_month_regression": max([0.0] + [-value for value in monthly_gains]) <= 5.0,
            "mape_regression": mape_change <= 1.0,
            "paired_bootstrap_upper_below_zero": ci["upper95_pp"] < 0.0,
            "strict_target": pooled["R168"]["mdape_percent"] < registered.TARGET,
        }
        if comparison.get("gates") != expected_gates or comparison.get("qualifies") != all(expected_gates.values()):
            raise RuntimeError(f"{comparator_name} development gate mismatch")
    if result["scores"].get("qualifies_development") != all(
        row["qualifies"] for row in result["scores"]["comparisons"].values()
    ):
        raise RuntimeError("pooled development decision does not match comparator gates")

    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    run_rows = [row for row in ledger if row.get("card") == registered.CARD and row.get("run_id") == run_id]
    starts = [row for row in run_rows if row.get("event") == "fit_start"]
    completions = [row for row in run_rows if row.get("event") == "fit_complete"]
    if len(starts) != len(expected_fits) or len(completions) != len(starts):
        raise RuntimeError("calibration fit ledger counts do not reconcile")
    if any(row.get("reserved_months_labels_opened") or row.get("hidden_labels_opened") for row in run_rows):
        raise RuntimeError("ledger contains a restricted-label access flag")
    if {row.get("month") for row in starts} != {row["month"] for row in expected_fits}:
        raise RuntimeError("fit ledger months differ from supported calibration folds")

    audit_result = {"run_id": run_id, "card": registered.CARD, "status": "passed",
                    "row_count": len(saved), "development_months": list(registered.MONTHS),
                    "source_hashes_verified": True, "row_and_label_alignment_verified": True,
                    "past_only_weights_reconstructed": True, "predictions_reconstructed": True,
                    "metrics_and_bootstrap_recomputed": True,
                    "fit_starts": len(starts), "fit_completions": len(completions),
                    "restricted_label_access": False}
    audit_path = batch / "independent_audit.json"
    audit_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card = campaign.PROTOCOL["cards"][registered.CARD]
    card.update({"audit_status": "passed", "audit_path": rel(audit_path), "audit_sha256": sha256(audit_path)})
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({"event": "independent_audit_complete", "card": registered.CARD, "run_id": run_id,
                      "status": "passed", "audit_path": rel(audit_path), "audit_sha256": sha256(audit_path),
                      "fit_starts": len(starts), "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False})
    passed = bool(stored_scores["qualifies_development"])
    campaign._write_progress("r168_audit_passed" if passed else "r168_audited_no_improvement",
                             next_action=("run locked fixed-origin confirmation for R168" if passed
                                          else "retain R35/R97; explore a distinct list-aware feature or learner"),
                             latest_run_id=run_id, latest_card=registered.CARD,
                             latest_audit_passed=True, latest_promotion_gate_passed=passed)
    return audit_result


def rel(path: Path) -> str:
    return path.relative_to(campaign.ROOT).as_posix()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(audit(args.run_id), indent=2, ensure_ascii=False))
