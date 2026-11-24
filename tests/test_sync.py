import numpy as np
import pytest
from helpers import activity_from_enu, winding_enu

from stride.resample import _seconds
from stride.sync import ClockSync


def pair(offset_s, drift_ppm, noise_m=2.0, duration=3600, rate_b=1.0):
    true_a = np.arange(0, duration, 1.0)
    ref = activity_from_enu(true_a, true_a, winding_enu(true_a), "watch", noise_m, seed=1)
    true_b = np.arange(3.3, duration - 5, 1.0 / rate_b)
    clock_b = true_b + offset_s + drift_ppm * 1e-6 * true_b
    other = activity_from_enu(true_b, clock_b, winding_enu(true_b), "phone", noise_m, seed=2)
    return ref, other


@pytest.mark.parametrize("offset", [-42.6, 0.0, 7.3, 187.25])
def test_constant_offset_is_recovered(offset):
    ref, other = pair(offset, 0.0)
    result = ClockSync().estimate(ref, other)
    assert result.offset_s == pytest.approx(offset, abs=0.1)
    assert abs(result.drift_ppm) < 20
    assert result.residual_m < 5


def test_drift_is_recovered_and_correction_aligns_clocks():
    ref, other = pair(12.0, 300.0)
    sync = ClockSync()
    result = sync.estimate(ref, other)
    assert result.drift_ppm == pytest.approx(300.0, abs=30)
    fixed = sync.apply(other, result)
    true_b = np.arange(3.3, 3595, 1.0)
    error = _seconds(fixed.samples["t"]) - _seconds(ref.samples["t"].iloc[[0]]) - true_b
    assert np.abs(error).max() < 0.15


def test_sparse_device_still_syncs():
    ref, other = pair(-9.5, 0.0, rate_b=0.25)
    assert ClockSync().estimate(ref, other).offset_s == pytest.approx(-9.5, abs=0.15)


def test_no_shared_positions_gives_none():
    ref, other = pair(5.0, 0.0)
    other.samples["lat"] = np.nan
    assert ClockSync().estimate(ref, other) is None
