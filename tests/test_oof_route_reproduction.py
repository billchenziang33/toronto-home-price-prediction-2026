import importlib.util
from pathlib import Path


def load_verifier():
    path = Path(__file__).parents[1] / "tools" / "verify_oof_route_reproduction.py"
    assert path.is_file(), "OOF route reproduction verifier is missing"
    spec = importlib.util.spec_from_file_location("oof_route_reproduction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r52_and_r151_oof_routes_reproduce_from_packaged_safe_assets():
    report = load_verifier().verify_oof_route_reproduction()
    assert report["status"] == "verified"
    assert report["r52"]["exact_prediction_rows"] == 15642
    assert report["r151"]["exact_prediction_rows"] == 15642
    assert report["reserved_labels_read"] is False
    assert report["hidden_labels_read"] is False
