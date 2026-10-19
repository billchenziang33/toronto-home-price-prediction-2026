import pandas as pd
import pytest

from experiments.model_research_next.methods import (
    apply_oof_calibration,
    apply_oof_prediction_band_calibration,
    fit_oof_blend_weight,
    fit_oof_calibration,
    fit_oof_prediction_band_calibration,
    validate_fold_plan,
)
from experiments.model_research_next.diagnostics import compute_paired_residuals, summarize_residuals


SCREEN_MONTHS = ("2022-03", "2022-10")
CONFIRMATION_MONTHS = (
    "2023-03",
    "2023-07",
    "2023-11",
    "2024-03",
    "2024-06",
    "2024-09",
)
FINAL_MONTHS = ("2025-05", "2025-08")


def test_fold_plan_accepts_audited_disjoint_chronological_months():
    validate_fold_plan(
        SCREEN_MONTHS,
        CONFIRMATION_MONTHS,
        FINAL_MONTHS,
        previously_evaluated={"2025-02", "2025-04", "2025-06"},
    )


def test_fold_plan_rejects_a_month_used_by_any_prior_experiment():
    with pytest.raises(ValueError, match="previously evaluated"):
        validate_fold_plan(
            SCREEN_MONTHS,
            CONFIRMATION_MONTHS,
            FINAL_MONTHS,
            previously_evaluated={"2022-10"},
        )


def test_oof_calibrator_ignores_rows_on_or_after_outer_cutoff():
    oof = pd.DataFrame(
        {
            "ml_num": ["a", "b", "future"],
            "date_sold": ["2022-03-10", "2022-03-20", "2023-04-01"],
            "sold_price": [110.0, 110.0, 1_000_000.0],
            "predicted_price": [100.0, 100.0, 1.0],
            "house_type_name": ["Condo Apt", "Condo Apt", "Detached"],
        }
    )

    fitted = fit_oof_calibration(oof, cutoff="2023-03-01", group_columns=("house_type_name",))

    assert fitted["n_oof"] == 2
    assert fitted["max_oof_date"] == "2022-03-20"
    assert fitted["global_factor"] == pytest.approx(1.05)
    assert "Detached" not in fitted["group_factors"]


def test_oof_calibrator_shrinks_small_groups_and_applies_group_factor():
    oof = pd.DataFrame(
        {
            "ml_num": ["a", "b", "c", "d"],
            "date_sold": ["2022-03-01", "2022-03-02", "2022-03-03", "2022-03-04"],
            "sold_price": [104.0, 104.0, 98.0, 98.0],
            "predicted_price": [100.0, 100.0, 100.0, 100.0],
            "house_type_name": ["Detached", "Detached", "Condo Apt", "Condo Apt"],
        }
    )
    fitted = fit_oof_calibration(
        oof,
        cutoff="2023-01-01",
        group_columns=("house_type_name",),
        pseudocount=2,
        min_group=2,
    )
    predictions = apply_oof_calibration(
        [100.0, 100.0],
        pd.DataFrame({"house_type_name": ["Detached", "Unknown"]}),
        fitted,
    )

    assert predictions[0] > predictions[1]
    assert 0.95 <= predictions[0] / 100.0 <= 1.05
    assert predictions[1] == pytest.approx(100.0 * fitted["global_factor"])


def test_blend_weight_uses_only_aligned_oof_rows_and_is_deterministic():
    oof = pd.DataFrame(
        {
            "month": ["2022-03", "2022-03", "2022-10"],
            "ml_num": ["a", "b", "c"],
            "sold_price": [200.0, 100.0, 300.0],
            "parent_prediction": [100.0, 100.0, 100.0],
            "challenger_prediction": [200.0, 100.0, 300.0],
        }
    )

    weight = fit_oof_blend_weight(oof, weights=(0.0, 0.25, 0.5, 0.75, 1.0))

    assert weight == 1.0


def test_blend_weight_rejects_duplicate_or_misaligned_home_month_keys():
    oof = pd.DataFrame(
        {
            "month": ["2022-03", "2022-03"],
            "ml_num": ["a", "a"],
            "sold_price": [100.0, 100.0],
            "parent_prediction": [100.0, 100.0],
            "challenger_prediction": [90.0, 90.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        fit_oof_blend_weight(oof)


def test_paired_residual_summary_reports_month_and_segment_changes():
    parent = pd.DataFrame(
        {
            "month": ["2022-03", "2022-03", "2022-10", "2022-10"],
            "ml_num": ["a", "b", "c", "d"],
            "sold_price": [100.0, 100.0, 200.0, 200.0],
            "predicted_price": [110.0, 110.0, 220.0, 220.0],
        }
    )
    candidate = parent.copy()
    candidate["predicted_price"] = [105.0, 105.0, 230.0, 230.0]
    features = pd.DataFrame(
        {
            "ml_num": ["a", "b", "c", "d"],
            "house_type_name": ["Condo", "Condo", "Detached", "Detached"],
        }
    )

    rows = compute_paired_residuals(parent, candidate, features)
    report = summarize_residuals(rows, group_columns=("month", "house_type_name"), min_group_size=1)

    assert report["overall"]["n"] == 4
    assert report["by_group"]["month"]["2022-03"]["relative_mdape_improvement"] == pytest.approx(0.5)
    assert report["by_group"]["house_type_name"]["Detached"]["relative_mdape_improvement"] == pytest.approx(-0.5)


def test_paired_residuals_reject_unaligned_home_months():
    base = pd.DataFrame(
        {"month": ["2022-03"], "ml_num": ["a"], "sold_price": [100.0], "predicted_price": [100.0]}
    )
    candidate = base.assign(ml_num="b")
    features = pd.DataFrame({"ml_num": ["a", "b"], "house_type_name": ["Condo", "Detached"]})

    with pytest.raises(ValueError, match="align"):
        compute_paired_residuals(base, candidate, features)


def test_oof_calibration_can_require_and_limit_past_months():
    oof = pd.DataFrame(
        {
            "month": ["2022-02", "2022-03", "2022-10", "2022-11"],
            "ml_num": ["a", "b", "c", "d"],
            "date_sold": ["2022-02-10", "2022-03-10", "2022-10-10", "2022-11-10"],
            "sold_price": [110.0, 120.0, 130.0, 140.0],
            "predicted_price": [100.0, 100.0, 100.0, 100.0],
        }
    )

    fitted = fit_oof_calibration(
        oof,
        cutoff="2022-12-01",
        max_months=2,
        min_oof_months=2,
        clip=(0.1, 2.0),
    )

    assert fitted["n_oof"] == 2
    assert fitted["oof_months"] == ["2022-10", "2022-11"]
    assert fitted["global_factor"] == pytest.approx((1.3 * 1.4) ** 0.5)


def test_prediction_band_calibration_uses_past_oof_edges_and_applies_them():
    oof = pd.DataFrame(
        {
            "month": ["2022-02"] * 4 + ["2023-03"],
            "ml_num": ["a", "b", "c", "d", "future"],
            "date_sold": ["2022-02-01", "2022-02-02", "2022-02-03", "2022-02-04", "2023-03-01"],
            "sold_price": [60.0, 110.0, 170.0, 240.0, 1_000_000.0],
            "predicted_price": [50.0, 100.0, 150.0, 200.0, 1.0],
        }
    )

    fitted = fit_oof_prediction_band_calibration(
        oof,
        cutoff="2023-03-01",
        group_columns=(),
        min_group=1,
        pseudocount=0,
    )
    calibrated = apply_oof_prediction_band_calibration(
        [50.0, 150.0],
        pd.DataFrame(index=[0, 1]),
        fitted,
    )

    assert fitted["max_oof_date"] == "2022-02-04"
    assert len(fitted["band_edges"]) == 3
    assert calibrated[0] > 50.0
    assert calibrated[1] > 150.0
