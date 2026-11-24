"""Resampling onto a common time grid, so channels from different devices line up sample
for sample.

Values are linearly interpolated in time, but never across a recording gap: a grid point
whose nearest real samples are further apart than max_gap_s gets NaN, because inventing a
straight line through a 2-minute GPS dropout would fake data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stride.schema import CHANNELS, Activity


@dataclass(frozen=True)
class Resampler:
    rate_hz: float = 1.0
    max_gap_s: float = 10.0

    def run(self, activity: Activity) -> Activity:
        samples = activity.samples
        seconds = _seconds(samples["t"])
        if len(seconds) and np.any(np.diff(seconds) <= 0):
            raise ValueError("resampling needs strictly increasing timestamps; clean first")
        grid = self.grid(seconds)
        out = pd.DataFrame({"t": pd.to_datetime(grid, unit="s", utc=True).as_unit("ns")})
        for name in CHANNELS:
            out[name] = self.interpolate(seconds, samples[name].to_numpy(float), grid)
        return Activity(
            activity.activity_id,
            activity.device_id,
            out,
            activity.sport,
            activity.source_path,
            [*activity.notes, f"resampled to {self.rate_hz:g} Hz"],
        )

    def grid(self, seconds: np.ndarray) -> np.ndarray:
        """Grid points aligned to whole multiples of the period, so grids from different
        devices coincide exactly."""
        if not len(seconds):
            return np.array([])
        step = 1.0 / self.rate_hz
        start = np.ceil(seconds[0] / step) * step
        stop = np.floor(seconds[-1] / step) * step
        return np.arange(round((stop - start) / step) + 1) * step + start

    def interpolate(self, t: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(values)
        t, values = t[valid], values[valid]
        out = np.full(len(grid), np.nan)
        if len(t) < 2:
            return out
        out[:] = np.interp(grid, t, values)
        right = np.clip(np.searchsorted(t, grid, side="left"), 1, len(t) - 1)
        gap = t[right] - t[right - 1]
        exact = np.isin(grid, t)
        outside = (grid < t[0]) | (grid > t[-1])
        out[((gap > self.max_gap_s) & ~exact) | outside] = np.nan
        return out


def _seconds(times: pd.Series) -> np.ndarray:
    """UTC seconds as float. Microsecond precision is plenty and keeps doubles exact."""
    delta = pd.to_datetime(times, utc=True) - pd.Timestamp(0, tz="UTC")
    return (delta // pd.Timedelta("1us")).to_numpy(dtype=np.int64) / 1e6
