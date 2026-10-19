from __future__ import annotations

DEVELOPMENT_MONTHS = ("2024-10", "2024-12", "2025-01", "2025-03")
FINAL_MONTHS = ("2025-05", "2025-08")
CANDIDATE_ORDER = ("Q50", "CB_MAE", "GAMMA_PRICE")
TRACKS = ("off_market", "list_aware")


def _month_ordinal(month: str) -> int:
    try:
        year_text, month_text = month.split("-")
        year, number = int(year_text), int(month_text)
        if not 1 <= number <= 12:
            raise ValueError
        return year * 12 + number
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid YYYY-MM month: {month!r}") from exc


def validate_fold_split(development_months, final_months, previously_evaluated) -> None:
    development = tuple(development_months)
    final = tuple(final_months)
    used = set(previously_evaluated)
    if not development or not final:
        raise ValueError("development and final folds must both be nonempty")
    for name, months in (("development", development), ("final", final)):
        ordinals = [_month_ordinal(month) for month in months]
        if ordinals != sorted(set(ordinals)):
            raise ValueError(f"{name} months must be unique and chronological")
        reused = set(months).intersection(used)
        if reused:
            raise ValueError(f"months were previously evaluated: {sorted(reused)}")
    if set(development).intersection(final):
        raise ValueError("development and final folds overlap")
    if _month_ordinal(final[0]) <= _month_ordinal(development[-1]):
        raise ValueError("final months must be after all development months")


def candidate_spec(candidate_id: str, track: str, c0_config: dict) -> dict:
    if candidate_id not in CANDIDATE_ORDER:
        raise ValueError(f"unknown candidate: {candidate_id}")
    if track not in TRACKS:
        raise ValueError(f"unknown track: {track}")

    if candidate_id == "Q50":
        config = dict(c0_config)
        config.update(objective="quantile", alpha=0.5)
        family, target_mode = "lightgbm", "frozen_log_target"
    elif candidate_id == "CB_MAE":
        config = {
            "loss_function": "MAE",
            "iterations": 600,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3,
            "thread_count": 4,
            "random_seed": 0,
            "verbose": False,
            "allow_writing_files": False,
        }
        family, target_mode = "catboost", "frozen_log_target"
    else:
        config = dict(c0_config)
        config["objective"] = "gamma"
        family, target_mode = "lightgbm", "positive_dollar_price"

    return {
        "candidate_id": candidate_id,
        "track": track,
        "model_family": family,
        "model_config": config,
        "target_mode": target_mode,
        "target_override": None if target_mode == "positive_dollar_price" else (
            "log_sold_over_list" if track == "list_aware" else "log_sold_price"
        ),
        "method_id": "M13" if target_mode == "positive_dollar_price" else "C0",
        "list_blend_weight": 0.75 if track == "list_aware" else 1.0,
        "postprocessing": "rounded candidate; for list-aware, round HGBR dollar prediction, blend in dollar space at frozen weight, then round",
    }


def _gate(report: dict, *, min_months_improved: int) -> dict:
    ci = report.get("paired_home_bootstrap", {}).get("ci95_percentage_points", [float("nan"), float("nan")])
    criteria = {
        "mdape_improvement_at_least_1pct": report.get("relative_mdape_improvement", float("-inf")) >= 0.01,
        "minimum_months_improved": report.get("months_improved", 0) >= min_months_improved,
        "worst_month_regression_at_most_5pct": report.get("worst_month_relative_regression", float("inf")) <= 0.05,
        "mape_relative_regression_at_most_1pct": report.get("mape_relative_change", float("inf")) <= 0.01,
        "paired_home_bootstrap_upper_below_zero": len(ci) == 2 and ci[1] < 0.0,
    }
    return {"passed": all(criteria.values()), "criteria": criteria}


def development_gate(report: dict) -> dict:
    return _gate(report, min_months_improved=3)


def final_gate(report: dict) -> dict:
    return _gate(report, min_months_improved=2)
