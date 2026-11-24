"""Validation rules. Each sample rule returns a boolean mask over the samples it flags;
activity rules look at the recording as a whole.

Rules run on normalized data in the order the device wrote it, before any cleaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stride.geo import LocalFrame

# Fastest plausible ground speed per sport, m/s. Above this, a sample is not real.
MAX_SPEED = {"running": 12.0, "walking": 4.0, "hiking": 4.0, "cycling": 30.0, "biking": 30.0}
DEFAULT_MAX_SPEED = 30.0


@dataclass
class Context:
    """Everything rules need, computed once per activity."""

    seconds: np.ndarray
    samples: pd.DataFrame
    sport: str | None
    nominal_hz: float
    enu: np.ndarray  # (n, 3); NaN rows where there is no fix

    @classmethod
    def build(cls, samples: pd.DataFrame, sport: str | None, nominal_hz: float) -> Context:
        delta = pd.to_datetime(samples["t"], utc=True) - pd.Timestamp(0, tz="UTC")
        seconds = (delta // pd.Timedelta("1us")).to_numpy(dtype=np.int64) / 1e6
        lat, lon = samples["lat"].to_numpy(float), samples["lon"].to_numpy(float)
        if np.isnan(lat).all():
            enu = np.full((len(samples), 3), np.nan)
        else:
            enu = LocalFrame.at_first_fix(lat, lon).to_enu(lat, lon)
            enu[np.isnan(lat) | np.isnan(lon)] = np.nan
        return cls(seconds, samples, sport, nominal_hz, enu)

    @property
    def max_speed(self) -> float:
        return MAX_SPEED.get((self.sport or "").lower(), DEFAULT_MAX_SPEED)

    def channel(self, name: str) -> np.ndarray:
        return self.samples[name].to_numpy(float)


class SampleRule:
    name = ""
    severity = "warning"  # "warning" keeps the sample; "error" removes the value in cleaning

    def check(self, ctx: Context) -> np.ndarray:
        raise NotImplementedError


class NonMonotonic(SampleRule):
    name, severity = "non_monotonic", "error"

    def check(self, ctx):
        """A sample whose time is earlier than the latest time seen before it."""
        running_max = np.maximum.accumulate(ctx.seconds)
        mask = np.zeros(len(ctx.seconds), bool)
        mask[1:] = ctx.seconds[1:] < running_max[:-1]
        return mask


class DuplicateTimestamp(SampleRule):
    name, severity = "duplicate_timestamp", "error"

    def check(self, ctx):
        return pd.Series(ctx.seconds).duplicated(keep="first").to_numpy()


class TimestampGap(SampleRule):
    """The first sample after a pause longer than five nominal periods (at least 10 s)."""

    name, severity = "timestamp_gap", "warning"

    def check(self, ctx):
        limit = max(10.0, 5.0 / ctx.nominal_hz)
        ordered = np.sort(ctx.seconds)
        gap_after = ordered[np.flatnonzero(np.diff(ordered) > limit) + 1]
        return np.isin(ctx.seconds, gap_after) & ~DuplicateTimestamp().check(ctx)


class GpsJump(SampleRule):
    """Fixes that sit further from the last trusted fix than the sport allows.

    Each fix is compared with the last accepted one: it is a jump if reaching it needs more
    than max_speed plus a 20 m allowance for normal GPS noise. After five jumps in a row the
    track is accepted as moved (the device really relocated, e.g. after a car ride), so
    one bad anchor cannot flag the rest of the recording.
    """

    name, severity = "gps_jump", "error"
    NOISE_M = 20.0
    REANCHOR_AFTER = 5

    def check(self, ctx):
        order = np.argsort(ctx.seconds, kind="stable")
        mask = np.zeros(len(order), bool)
        anchor, streak = None, []
        for i in order:
            if np.isnan(ctx.enu[i, 0]):
                continue
            if anchor is None:
                anchor = i
                continue
            dt = max(ctx.seconds[i] - ctx.seconds[anchor], 0.0)
            dist = np.hypot(*(ctx.enu[i, :2] - ctx.enu[anchor, :2]))
            if dist <= ctx.max_speed * dt + self.NOISE_M:
                anchor, streak = i, []
                continue
            streak.append(i)
            if len(streak) > self.REANCHOR_AFTER:
                anchor, streak = i, []
            else:
                mask[i] = True
        return mask


class ImpossibleSpeed(SampleRule):
    name, severity = "impossible_speed", "error"

    def check(self, ctx):
        speed = ctx.channel("speed_mps")
        return (speed < 0) | (speed > ctx.max_speed)


class HeartRateRange(SampleRule):
    name, severity = "hr_out_of_range", "error"

    def check(self, ctx):
        hr = ctx.channel("hr_bpm")
        return (hr < 30) | (hr > 230)


class ElevationSpike(SampleRule):
    """Altitude more than 40 m away from the median of its 11-sample neighbourhood."""

    name, severity = "elevation_spike", "error"

    def check(self, ctx):
        order = np.argsort(ctx.seconds, kind="stable")
        ele = pd.Series(ctx.channel("ele_m")[order])
        median = ele.rolling(11, center=True, min_periods=3).median()
        spike = ((ele - median).abs() > 40.0).to_numpy()
        mask = np.zeros(len(order), bool)
        mask[order] = spike
        return mask


class DistanceDecrease(SampleRule):
    name, severity = "distance_decrease", "error"

    def check(self, ctx):
        order = np.argsort(ctx.seconds, kind="stable")
        dist = ctx.channel("distance_m")[order]
        valid = ~np.isnan(dist)
        running_max = np.fmax.accumulate(np.where(valid, dist, -np.inf))
        drop = np.zeros(len(dist), bool)
        drop[1:] = valid[1:] & (dist[1:] < running_max[:-1] - 0.5)
        mask = np.zeros(len(order), bool)
        mask[order] = drop
        return mask


SAMPLE_RULES: tuple[SampleRule, ...] = (
    NonMonotonic(),
    DuplicateTimestamp(),
    TimestampGap(),
    GpsJump(),
    ImpossibleSpeed(),
    HeartRateRange(),
    ElevationSpike(),
    DistanceDecrease(),
)
