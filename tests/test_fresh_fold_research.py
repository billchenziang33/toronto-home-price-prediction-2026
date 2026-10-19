from __future__ import annotations

import pytest

from experiments.model_research_fresh.candidates import (
    DEVELOPMENT_MONTHS,
    FINAL_MONTHS,
    validate_fold_split,
    candidate_spec,
    development_gate,
    final_gate,
)
from experiments.model_research_fresh.audit_months import build_audit


def test_frozen_development_and_final_months_are_not_previous_validation_months():
    used = {
        "2025-02", "2025-04", "2025-06", "2025-07", "2025-09", "2025-10",
        "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05",
    }

    validate_fold_split(DEVELOPMENT_MONTHS, FINAL_MONTHS, used)

    assert DEVELOPMENT_MONTHS == ("2024-10", "2024-12", "2025-01", "2025-03")
    assert FINAL_MONTHS == ("2025-05", "2025-08")


def test_fold_split_rejects_reused_or_nonchronological_months():
    with pytest.raises(ValueError, match="previously evaluated"):
        validate_fold_split(("2025-06",), ("2025-08",), {"2025-06"})
    with pytest.raises(ValueError, match="after all development months"):
        validate_fold_split(("2025-03",), ("2025-01",), set())


def test_candidate_specs_use_three_fixed_distinct_loss_pipelines():
    c0 = {"objective": "regression", "n_estimators": 1000, "num_leaves": 31,
          "learning_rate": 0.05, "min_child_samples": 100, "reg_lambda": 5,
          "random_state": 0, "n_jobs": 4, "deterministic": True, "force_col_wise": True}

    q50 = candidate_spec("Q50", "off_market", c0)
    cat_mae = candidate_spec("CB_MAE", "list_aware", c0)
    gamma = candidate_spec("GAMMA_PRICE", "list_aware", c0)

    assert q50["model_family"] == "lightgbm"
    assert q50["model_config"]["objective"] == "quantile"
    assert q50["model_config"]["alpha"] == 0.5
    assert q50["target_mode"] == "frozen_log_target"
    assert cat_mae["model_family"] == "catboost"
    assert cat_mae["model_config"]["loss_function"] == "MAE"
    assert cat_mae["target_mode"] == "frozen_log_target"
    assert gamma["model_family"] == "lightgbm"
    assert gamma["model_config"]["objective"] == "gamma"
    assert gamma["target_mode"] == "positive_dollar_price"
    assert cat_mae["list_blend_weight"] == gamma["list_blend_weight"] == 0.75


def test_development_gate_requires_mdape_month_stability_mape_and_paired_ci():
    report = {"relative_mdape_improvement": 0.02, "months_improved": 3,
              "worst_month_relative_regression": 0.04, "mape_relative_change": 0.005,
              "paired_home_bootstrap": {"ci95_percentage_points": [-0.1, -0.01]}}

    assert development_gate(report)["passed"]
    report["mape_relative_change"] = 0.011
    assert not development_gate(report)["passed"]


def test_final_gate_requires_both_fresh_months_to_improve_and_paired_ci():
    report = {"relative_mdape_improvement": 0.015, "months_improved": 2,
              "worst_month_relative_regression": 0.0, "mape_relative_change": 0.005,
              "paired_home_bootstrap": {"ci95_percentage_points": [-0.1, -0.01]}}

    assert final_gate(report)["passed"]
    report["months_improved"] = 1
    assert not final_gate(report)["passed"]


def test_real_data_audit_confirms_fresh_development_rows_without_opening_final_labels():
    audit = build_audit(final_stage=False)

    assert audit["data_sha256"] == "18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf"
    assert audit["allowed_data_cutoff_exclusive"] == "2025-04-01"
    assert set(audit["fold_counts"]) == set(DEVELOPMENT_MONTHS)
    assert {month: value["evaluation_rows"] for month, value in audit["fold_counts"].items()} == {
        "2024-10": 2451,
        "2024-12": 1345,
        "2025-01": 1339,
        "2025-03": 1931,
    }
    assert audit["quality"]["unique_ml_num"]
    assert audit["quality"]["missing_sale_date"] == 0
    assert audit["quality"]["missing_sale_price"] == 0
