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
