import numpy as np

from run import _paired_mdape_delta_ci


def test_paired_mdape_interval_resamples_difference_of_route_medians():
    baseline = np.tile([0.0, 100.0, 100.0, 100.0, 100.0], 100)
    candidate = np.tile([10.0, 90.0, 100.0, 110.0, 90.0], 100)

    lower, upper = _paired_mdape_delta_ci(baseline, candidate, repeats=1000, seed=0)

    # Candidate MdAPE is 90 and baseline MdAPE is 100. The paired per-row
    # differences have median 0, so an interval for that different statistic
    # would not be centered on the table's -10 percentage-point estimand.
    assert np.median(candidate) - np.median(baseline) == -10.0
    assert upper < 0.0
    assert lower <= -10.0 <= upper
