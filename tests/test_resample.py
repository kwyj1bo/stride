import numpy as np
import pandas as pd
import pytest

from stride.resample import Resampler
from stride.schema import CHANNELS, Activity


def make(seconds, **channels):
    t = pd.to_datetime(np.array(seconds) * 1e6, unit="us", utc=True) + pd.Timedelta("1000D")
    frame = pd.DataFrame({"t": t})
    for name in CHANNELS:
        frame[name] = channels.get(name, np.full(len(seconds), np.nan))
    return Activity("a", "d", frame)


def test_grid_is_aligned_to_whole_seconds_and_interpolates_linearly():
    activity = make([0.4, 2.4, 4.4], speed_mps=[1.0, 3.0, 5.0])
    out = Resampler(rate_hz=1.0).run(activity).samples
    assert (out["t"].dt.microsecond == 0).all()
    assert len(out) == 4  # whole seconds 1..4 inside [0.4, 4.4]
    np.testing.assert_allclose(out["speed_mps"], [1.6, 2.6, 3.6, 4.6])


def test_no_interpolation_across_gaps():
    activity = make([0, 1, 2, 30, 31], hr_bpm=[100, 101, 102, 130, 131])
    out = Resampler(rate_hz=1.0, max_gap_s=10).run(activity).samples
    hr = out["hr_bpm"].to_numpy()
    assert hr[:3].tolist() == [100, 101, 102]
    assert np.isnan(hr[3:30]).all()
    assert hr[30:].tolist() == [130, 131]


def test_channels_with_missing_values_interpolate_from_their_own_samples():
    activity = make([0, 1, 2, 3], hr_bpm=[100, np.nan, np.nan, 103], speed_mps=[1, 2, 3, 4])
    out = Resampler(rate_hz=1.0).run(activity).samples
    np.testing.assert_allclose(out["hr_bpm"], [100, 101, 102, 103])


def test_rejects_unsorted_input():
    with pytest.raises(ValueError, match="strictly increasing"):
        Resampler().run(make([0, 2, 1], hr_bpm=[1, 2, 3]))


def test_half_hertz_grid():
    out = Resampler(rate_hz=0.5).run(make([0, 1, 2, 3, 4], hr_bpm=[0, 1, 2, 3, 4])).samples
    np.testing.assert_allclose(out["hr_bpm"], [0, 2, 4])
