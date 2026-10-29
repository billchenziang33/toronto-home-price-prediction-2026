import pytest

import submission_models
from models import make_model
from submission_models import ListAwareR97Model, OffMarketR151Model


def test_model_factory_rejects_unknown_track_instead_of_silently_using_off_market():
    with pytest.raises(ValueError, match="unknown track"):
        make_model("list_aware_typo")


def test_model_factory_selects_frozen_candidate_route_for_each_track():
    off_market = make_model("off_market")
    list_aware = make_model("list_aware")

    assert isinstance(off_market, OffMarketR151Model)
    assert isinstance(list_aware, ListAwareR97Model)
    assert off_market.route == "R52"
    assert list_aware.route == "R97"


def test_direct_submission_fit_is_logged_and_waits_for_desktop_friendly_runtime(monkeypatch):
    run_fit = getattr(submission_models, "_run_submission_fit", None)
    assert callable(run_fit), "direct model fits need the same ledger and runtime gate as wrapped fits"

    events = []
    waits = []
    monkeypatch.setattr(submission_models, "_append_fit_record", events.append)
    monkeypatch.setenv("BILL_DESKTOP_FRIENDLY_ACTIVE", "1")
    from experiments.bill_target74 import desktop_friendly_runtime as runtime
    monkeypatch.setattr(runtime, "wait_before_fit", waits.append)
    context = {"route": "R97", "component": "CatBoost_residual_meta",
               "track": "list_aware", "cutoff": "2026-01-01",
               "training_rows": 5000, "evaluation_rows": 10,
               "run_id": "test-run-17"}

    result = run_fit(context, lambda: {"model_family": "catboost"})

    assert result["model_family"] == "catboost"
    assert [event["event"] for event in events] == ["fit_start", "fit_complete"]
    assert events[0]["run_id"] == "test-run-17"
    assert events[-1]["status"] == "complete"
    assert waits == ["submission:R97:CatBoost_residual_meta"]


def test_rowwise_r22_r32_blend_applies_hierarchical_weight_per_home():
    blend = getattr(submission_models, "_blend_by_row_weight", None)
    assert callable(blend), "R48/R50 require a validated per-home blend helper"

    result = blend([100.0, 200.0, 300.0], [200.0, 400.0, 600.0], [0.0, 0.5, 1.0])

    assert result.tolist() == [100.0, 300.0, 600.0]
