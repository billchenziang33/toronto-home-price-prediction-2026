from __future__ import annotations

import numpy as np


def blend_prices(challenger, frozen, challenger_weight: float):
    """Match the retrospective benchmark: round frozen prices, blend, round result."""
    challenger = np.asarray(challenger, dtype=float)
    frozen = np.asarray(frozen, dtype=float)
    if challenger.shape != frozen.shape or not 0.0 <= challenger_weight <= 1.0:
        raise ValueError("blend inputs must align and challenger_weight must be in [0, 1]")
    if not np.isfinite(challenger).all() or not np.isfinite(frozen).all():
        raise ValueError("blend inputs must be finite")
    frozen_rounded = np.round(frozen)
    result = np.round((1.0 - challenger_weight) * frozen_rounded + challenger_weight * challenger)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("blended prices must be finite and positive")
    return result


def apply_multiplier(prediction, factor: float):
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("calibration factor must be finite and positive")
    result = np.asarray(prediction, dtype=float) * factor
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("calibrated predictions must be finite and positive")
    return result
