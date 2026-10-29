import importlib.util
from pathlib import Path

import pytest


def load_builder():
    path = Path(__file__).parents[1] / "tools" / "build_submission_assets.py"
    assert path.is_file(), "submission asset builder is missing"
    spec = importlib.util.spec_from_file_location("submission_asset_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_month_validation_accepts_registered_safe_development_months():
    builder = load_builder()
    allowed = {"2023-03", "2023-07"}
    # Each month naturally appears once per OOF row; row multiplicity is not a duplicate-month error.
    assert builder.validate_oof_months(
        ["2023-03", "2023-03", "2023-07"], allowed, {"2025-05"}
    ) == allowed


def test_month_validation_rejects_a_reserved_month_even_if_allowlisted():
    builder = load_builder()
    with pytest.raises(ValueError, match="reserved"):
        builder.validate_oof_months(["2025-05"], {"2025-05"}, {"2025-05"})


def test_month_validation_rejects_unregistered_months():
    builder = load_builder()
    with pytest.raises(ValueError, match="unexpected"):
        builder.validate_oof_months(["2024-01"], {"2023-03"}, {"2025-05"})


def test_packaged_assets_verify_without_loading_source_data():
    builder = load_builder()
    report = builder.verify_assets()
    assert report["status"] == "verified"
    assert report["asset_count"] == 5
