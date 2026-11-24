"""Garmin Training Center (TCX) reader."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from stride.io._common import child, child_float, descendant_float, iter_local, parse_times
from stride.schema import RawActivity

_CHANNELS = ("lat", "lon", "ele", "distance", "hr", "cadence", "speed")


def read_tcx(path: str | Path) -> RawActivity:
    root = ET.parse(path).getroot()
    points = [p for p in iter_local(root, "Trackpoint") if child(p, "Time") is not None]
    if not points:
        raise ValueError(f"{path}: no track points with a time")
    rows = {name: np.empty(len(points)) for name in _CHANNELS}
    times = []
    for i, point in enumerate(points):
        times.append(child(point, "Time").text)
        rows["lat"][i] = child_float(point, "Position", "LatitudeDegrees")
        rows["lon"][i] = child_float(point, "Position", "LongitudeDegrees")
        rows["ele"][i] = child_float(point, "AltitudeMeters")
        rows["distance"][i] = child_float(point, "DistanceMeters")
        rows["hr"][i] = child_float(point, "HeartRateBpm", "Value")
        rows["cadence"][i] = child_float(point, "Cadence")
        rows["speed"][i] = descendant_float(point, "Speed")
    samples = pd.DataFrame({"t": parse_times(times), **rows})
    return RawActivity(samples, "tcx", str(path), sport=_sport(root))


def _sport(root: ET.Element) -> str | None:
    activity = next(iter_local(root, "Activity"), None)
    sport = activity.get("Sport") if activity is not None else None
    return sport.lower() if sport else None
