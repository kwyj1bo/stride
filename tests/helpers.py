"""Builders for small in-memory activities used across tests."""

import numpy as np
import pandas as pd

from stride.geo import LocalFrame
from stride.schema import CHANNELS, Activity

ORIGIN = (19.076, 72.8777, 10.0)


def straight_run(n=300, speed=3.0, hz=1.0, sport="running"):
    """A run due east at constant speed, sampled at hz, with every channel filled."""
    seconds = np.arange(n) / hz
    east = speed * seconds
    enu = np.stack([east, np.zeros(n), np.zeros(n)], axis=1)
    lat, lon, ele = LocalFrame(*ORIGIN).to_geodetic(enu)
    t = pd.Timestamp("2026-11-01T01:00:00Z") + pd.to_timedelta(seconds, unit="s")
    frame = pd.DataFrame({"t": t.as_unit("ns")})
    values = {
        "lat": lat,
        "lon": lon,
        "ele_m": ele,
        "hr_bpm": np.full(n, 140.0),
        "cadence_spm": np.full(n, 170.0),
        "speed_mps": np.full(n, speed),
        "distance_m": east,
    }
    for name in CHANNELS:
        frame[name] = values[name]
    return Activity("a1", "phone_gpx", frame, sport)


def winding_enu(seconds):
    """A loop with changing speed, so every moment of the track is distinctive."""
    angle = 2 * np.pi * seconds / 900.0
    radius = 400.0 + 80.0 * np.sin(seconds / 70.0)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(seconds)], 1)


def activity_from_enu(true_seconds, clock_seconds, enu, device="d", noise_m=0.0, seed=0):
    """Positions at true times, stamped with the device's clock readings."""
    rng = np.random.default_rng(seed)
    enu = enu + rng.normal(0, noise_m, enu.shape) * [1, 1, 0]
    lat, lon, ele = LocalFrame(*ORIGIN).to_geodetic(enu)
    t = pd.Timestamp("2026-11-01T01:00:00Z") + pd.to_timedelta(clock_seconds, unit="s")
    frame = pd.DataFrame({"t": t.as_unit("ns")})
    for name in CHANNELS:
        frame[name] = np.nan
    frame["lat"], frame["lon"], frame["ele_m"] = lat, lon, ele
    return Activity(f"{device}-1", device, frame, "running")
