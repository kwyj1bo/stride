"""GPX 1.1 reader. Heart rate and cadence come from the Garmin TrackPointExtension."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from stride.io._common import child, child_float, descendant_float, iter_local, parse_times
from stride.schema import RawActivity


def read_gpx(path: str | Path) -> RawActivity:
    root = ET.parse(path).getroot()
    points = list(iter_local(root, "trkpt"))
    if not points:
        raise ValueError(f"{path}: no track points")
    times = []
    rows = {name: np.empty(len(points)) for name in ("lat", "lon", "ele", "hr", "cadence")}
    for i, point in enumerate(points):
        time = child(point, "time")
        if time is None or not time.text:
            raise ValueError(f"{path}: track point {i} has no time")
        times.append(time.text)
        rows["lat"][i] = float(point.get("lat"))
        rows["lon"][i] = float(point.get("lon"))
        rows["ele"][i] = child_float(point, "ele")
        rows["hr"][i] = descendant_float(point, "hr")
        rows["cadence"][i] = descendant_float(point, "cad")
    samples = pd.DataFrame({"t": parse_times(times), **rows})
    return RawActivity(samples, "gpx", str(path), sport=_sport(root))


def _sport(root: ET.Element) -> str | None:
    track = next(iter_local(root, "trk"), None)
    kind = child(track, "type") if track is not None else None
    return kind.text.strip().lower() if kind is not None and kind.text else None
