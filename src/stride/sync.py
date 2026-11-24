"""Clock synchronisation between two devices that recorded the same session.

Device clocks disagree: a phone may be seconds off, a cheap watch drifts. Comparing
sample-by-sample needs one time base. We estimate how far device B's clock is ahead of
reference A by sliding B's GPS track in time until it lies on top of A's track.

offset(t) = a + b * (t - t0): a is the offset at the middle of the overlap, b the drift
(seconds of error per second; 1e-6 is 1 ppm). Corrected B time = t_B - offset(t_B).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stride.geo import LocalFrame
from stride.resample import Resampler, _seconds
from stride.schema import Activity


@dataclass
class SyncResult:
    offset_s: float  # B clock minus A clock at t0
    drift_ppm: float
    t0: float  # middle of the overlap, UTC seconds
    residual_m: float  # RMS distance between the aligned tracks
    windows: int  # windows used for the drift fit

    def offset_at(self, seconds: np.ndarray) -> np.ndarray:
        return self.offset_s + self.drift_ppm * 1e-6 * (seconds - self.t0)


class ClockSync:
    def __init__(
        self,
        max_lag_s: float = 300.0,
        window_s: float = 600.0,
        min_overlap_s: float = 120.0,
        max_drift_ppm: float = 500.0,
    ) -> None:
        self.max_lag_s = max_lag_s
        self.window_s = window_s
        self.min_overlap_s = min_overlap_s
        self.max_drift_ppm = max_drift_ppm

    def estimate(self, ref: Activity, other: Activity) -> SyncResult | None:
        """None when the two tracks do not share enough positions to compare."""
        tracks = _Tracks(ref, other)
        if tracks.empty:
            return None
        grid = tracks.ref_t
        if grid[-1] - grid[0] < self.min_overlap_s:
            return None
        coarse = self._search(tracks, grid, -self.max_lag_s, self.max_lag_s, 1.0)
        if coarse is None:
            return None
        lag = self._search(tracks, grid, coarse - 1.5, coarse + 1.5, 0.1)
        return self._fit_drift(tracks, grid, lag)

    def apply(self, activity: Activity, result: SyncResult) -> Activity:
        seconds = _seconds(activity.samples["t"])
        corrected = seconds - result.offset_at(seconds)
        samples = activity.samples.copy()
        samples["t"] = _to_times(corrected)
        note = f"clock corrected by {result.offset_s:+.2f} s, drift {result.drift_ppm:+.1f} ppm"
        return Activity(
            activity.activity_id,
            activity.device_id,
            samples,
            activity.sport,
            activity.source_path,
            [*activity.notes, note],
        )

    def _fit_drift(self, tracks, grid, lag) -> SyncResult:
        """The whole-overlap lag is the offset at the overlap's middle. Drift comes from
        re-estimating the lag in windows and fitting a line, and is kept only if the slope
        is clearly larger than its own standard error: GPS error is correlated over tens of
        seconds, so on a short session window lags wobble by a few tenths of a second and
        an unchecked fit would report hundreds of ppm of drift that is not there."""
        centres, lags = [], []
        for start in np.arange(grid[0], grid[-1], self.window_s):
            window = grid[(grid >= start) & (grid < start + self.window_s)]
            if len(window) < 2 or window[-1] - window[0] < self.window_s / 2:
                continue
            local = self._search(tracks, window, lag - 1.0, lag + 1.0, 0.1)
            if local is not None:
                centres.append(window.mean())
                lags.append(local)
        t0 = float(grid.mean())
        drift = _significant_slope(np.array(centres) - t0, np.array(lags), self.max_drift_ppm)
        residual = np.sqrt(max(tracks.cost(grid, lag), 0.0))  # noise correction can dip below 0
        return SyncResult(float(lag), drift * 1e6, t0, float(residual), len(lags))

    def _search(self, tracks, grid, lo, hi, step) -> float | None:
        """Lag with the lowest cost on a grid, refined by a parabola through the best
        point and its two neighbours, so the answer is not limited to the grid step."""
        lags = np.arange(lo, hi + step / 2, step)
        costs = np.array([tracks.cost(grid, lag) for lag in lags])
        if np.isnan(costs).all():
            return None
        i = int(np.nanargmin(costs))
        if 0 < i < len(lags) - 1 and not np.isnan(costs[[i - 1, i + 1]]).any():
            left, mid, right = costs[i - 1], costs[i], costs[i + 1]
            curvature = left - 2 * mid + right
            if curvature > 0:
                return float(lags[i] + 0.5 * step * (left - right) / curvature)
        return float(lags[i])


_INTERP = Resampler(max_gap_s=10.0)


class _Tracks:
    """Both devices' positions in one ENU frame, with interpolation for B."""

    MIN_PAIRS = 30

    def __init__(self, ref: Activity, other: Activity) -> None:
        a, b = ref.samples, other.samples
        lat = np.concatenate([a["lat"], b["lat"]])
        lon = np.concatenate([a["lon"], b["lon"]])
        self.empty = np.isnan(a["lat"]).all() or np.isnan(b["lat"]).all()
        if self.empty:
            return
        frame = LocalFrame.at_first_fix(lat, lon)
        self.ref_t, self.ref_xy = _valid_track(ref, frame)
        self.other_t, self.other_xy = _valid_track(other, frame)
        self.empty = len(self.ref_t) < self.MIN_PAIRS or len(self.other_t) < self.MIN_PAIRS
        if not self.empty:
            self.other_noise2 = _white_noise_variance(self.other_xy)

    def cost(self, grid: np.ndarray, lag: float) -> float:
        """Mean squared distance between A at time g and B at its clock reading g + lag.
        Plain least squares: trimming the worst pairs looked robust but made the cost jump
        as pairs crossed the cut, which biased the lag. Bad fixes are removed by cleaning."""
        query = grid + lag
        inside = (query >= self.other_t[0]) & (query <= self.other_t[-1])
        if inside.sum() < self.MIN_PAIRS:
            return np.nan
        ex = _INTERP.interpolate(self.other_t, self.other_xy[:, 0], query[inside])
        ny = _INTERP.interpolate(self.other_t, self.other_xy[:, 1], query[inside])
        ref_idx = np.searchsorted(self.ref_t, grid[inside])
        d2 = (ex - self.ref_xy[ref_idx, 0]) ** 2 + (ny - self.ref_xy[ref_idx, 1]) ** 2
        # Interpolating halfway between two noisy fixes averages their noise, so raw d2
        # would favour lags that land between samples. Remove that expected term.
        d2 = d2 - self.other_noise2 * _interp_variance_factor(self.other_t, query[inside])
        d2 = d2[~np.isnan(d2)]
        return float(d2.mean()) if len(d2) >= self.MIN_PAIRS else np.nan


def _significant_slope(x: np.ndarray, y: np.ndarray, max_ppm: float) -> float:
    """Least-squares slope if it is at least three standard errors from zero and
    physically plausible; otherwise 0."""
    if len(x) < 3:
        return 0.0
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    spread = np.sum((x - x.mean()) ** 2)
    if spread <= 0:
        return 0.0
    stderr = np.sqrt(np.sum(residuals**2) / (len(x) - 2) / spread)
    if abs(slope) < 3 * stderr or abs(slope) * 1e6 > max_ppm:
        return 0.0
    return float(slope)


def _white_noise_variance(xy: np.ndarray) -> float:
    """Variance of the sample-to-sample (white) position noise, both axes together.
    For white noise e, the second difference e[i-1] - 2e[i] + e[i+1] has variance 6 var(e);
    a smooth path adds almost nothing to it. The MAD keeps outliers from inflating it."""
    second = np.diff(xy, n=2, axis=0)
    if len(second) < 10:
        return 0.0
    mad = np.median(np.abs(second - np.median(second, axis=0)), axis=0)
    return float(((1.4826 * mad) ** 2 / 6.0).sum())


def _interp_variance_factor(t: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Noise variance of a linear interpolation relative to one sample: (1-f)^2 + f^2."""
    right = np.clip(np.searchsorted(t, query, side="right"), 1, len(t) - 1)
    span = t[right] - t[right - 1]
    f = np.clip((query - t[right - 1]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    return (1 - f) ** 2 + f**2


def _valid_track(activity: Activity, frame: LocalFrame) -> tuple[np.ndarray, np.ndarray]:
    s = activity.samples
    seconds = _seconds(s["t"])
    enu = frame.to_enu(s["lat"], s["lon"])
    valid = ~np.isnan(s["lat"].to_numpy(float)) & ~np.isnan(s["lon"].to_numpy(float))
    return seconds[valid], enu[valid, :2]


def _to_times(seconds: np.ndarray) -> pd.Series:
    micros = np.round(seconds * 1e6).astype(np.int64)
    return pd.Series(pd.to_datetime(micros, unit="us", utc=True).as_unit("ns"))
