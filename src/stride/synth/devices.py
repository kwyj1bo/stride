"""Simulated recording devices.

Each device samples the truth on its own clock ticks, with its own clock error, GPS error
and channel set, and returns samples in the units its registry profile declares.

Clock model: reading = true + offset + drift * (true - session start). A device samples at
whole ticks of its own clock, so the true sample times are found by inverting that model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stride.geo import LocalFrame
from stride.registry import DeviceProfile
from stride.synth.truth import DT, TruthTrack


@dataclass(frozen=True)
class DeviceModel:
    device_id: str
    period_s: tuple[float, float]  # min and max seconds between samples
    channels: tuple[str, ...]
    offset_sd_s: float  # clock offset spread
    max_drift_ppm: float
    gps_ar_sd_m: float  # slowly varying GPS error
    gps_tau_s: float
    gps_white_sd_m: float  # sample-to-sample GPS error
    ele_sd_m: float = 2.0
    odometer_sd: float = 0.004  # relative error of the device's own distance


MODELS = {
    "watch_fit": DeviceModel(
        "watch_fit",
        (1, 1),
        ("lat", "lon", "ele", "hr", "cadence", "speed", "distance"),
        0.3,
        10,
        2.0,
        40,
        0.5,
    ),
    "phone_gpx": DeviceModel(
        "phone_gpx", (1, 1), ("lat", "lon", "ele"), 20.0, 30, 3.5, 25, 1.5, ele_sd_m=4.0
    ),
    "bike_tcx": DeviceModel(
        "bike_tcx",
        (1, 8),
        ("lat", "lon", "ele", "hr", "cadence", "speed", "distance"),
        1.0,
        50,
        2.5,
        40,
        0.8,
    ),
    "app_csv": DeviceModel(
        "app_csv",
        (2, 2),
        ("lat", "lon", "ele", "hr", "cadence", "speed", "distance"),
        6.0,
        100,
        3.0,
        30,
        1.2,
        ele_sd_m=3.0,
    ),
}


@dataclass
class DeviceRecording:
    device_id: str
    samples: pd.DataFrame  # raw channels in device units, "t" = device clock in UTC
    offset_s: float
    drift_ppm: float
    true_seconds: np.ndarray  # true time of each sample, seconds since session start


class SimulatedDevice:
    def __init__(self, model: DeviceModel, profile: DeviceProfile) -> None:
        self.model = model
        self.profile = profile

    def record(
        self, rng: np.random.Generator, truth: TruthTrack, start_s: float, stop_s: float
    ) -> DeviceRecording:
        offset = float(rng.normal(0, self.model.offset_sd_s))
        drift = float(rng.uniform(-1, 1) * self.model.max_drift_ppm)
        ticks = self._ticks(rng, start_s + offset, stop_s + offset)
        true_s = (ticks - offset) / (1 + drift * 1e-6)  # invert the clock model
        keep = (true_s >= 0) & (true_s <= truth.seconds[-1])
        ticks, true_s = ticks[keep], true_s[keep]
        values = truth.at(true_s)
        samples = pd.DataFrame({"t": _utc(truth.start_utc + ticks)})
        for channel, column in self._channels(rng, truth, values, true_s).items():
            if channel in self.model.channels:
                samples[channel] = column
        return DeviceRecording(self.model.device_id, samples, offset, drift, true_s)

    def _ticks(self, rng, start, stop) -> np.ndarray:
        """Whole seconds of the device clock. Smart recording varies the spacing."""
        lo, hi = self.model.period_s
        first = np.ceil(start / lo) * lo
        if lo == hi:
            return np.arange(first, stop, lo)
        gaps = rng.integers(int(lo), int(hi) + 1, size=int((stop - start) / lo) + 2)
        ticks = first + np.concatenate([[0], np.cumsum(gaps)])
        return ticks[ticks < stop].astype(float)

    def _channels(self, rng, truth, values, true_s) -> dict[str, np.ndarray]:
        n = len(true_s)
        error = np.stack([self._gps_error(rng, true_s) for _ in range(2)], axis=1)
        enu = np.stack(
            [
                values["e"] + error[:, 0],
                values["n"] + error[:, 1],
                values["u"] + rng.normal(0, self.model.ele_sd_m, n),
            ],
            axis=1,
        )
        lat, lon, ele = LocalFrame(*truth.origin).to_geodetic(enu)
        scale = 1 + rng.normal(0, self.model.odometer_sd)
        distance = (values["distance"] - values["distance"][0]) * scale
        speed = np.clip(values["speed"] * scale + rng.normal(0, 0.05, n), 0, None)
        return {
            "lat": lat, "lon": lon,
            "ele": ele / self.profile.factor("elevation"),
            "hr": np.round(values["hr"] + rng.normal(0, 1.0, n)),
            "cadence": np.round(values["cadence"]),
            "speed": speed / self.profile.factor("speed"),
            "distance": distance / self.profile.factor("distance"),
        }  # fmt: skip

    def _gps_error(self, rng, true_s) -> np.ndarray:
        """AR(1) error correlated over gps_tau_s, plus independent white noise."""
        m = self.model
        dt = np.diff(true_s, prepend=true_s[0] - 1)
        a = np.exp(-np.maximum(dt, DT) / m.gps_tau_s)
        shocks = rng.normal(0, 1, len(true_s)) * m.gps_ar_sd_m * np.sqrt(1 - a * a)
        slow = np.empty(len(true_s))
        slow[0] = rng.normal(0, m.gps_ar_sd_m)
        for i in range(1, len(slow)):
            slow[i] = a[i] * slow[i - 1] + shocks[i]
        return slow + rng.normal(0, m.gps_white_sd_m, len(true_s))


def _utc(seconds: np.ndarray) -> pd.Series:
    micros = np.round(seconds * 1e6).astype(np.int64)
    return pd.Series(pd.to_datetime(micros, unit="us", utc=True).as_unit("ns"))
