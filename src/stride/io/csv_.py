"""CSV reader. Column names and the time format come from the device profile."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stride.io._common import parse_times
from stride.registry import DeviceProfile
from stride.schema import RawActivity


def read_csv(path: str | Path, profile: DeviceProfile) -> RawActivity:
    table = pd.read_csv(path)
    mapping = profile.csv_columns
    missing = [header for header in mapping.values() if header not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} for device {profile.device_id}")
    samples = pd.DataFrame({"t": _times(table[mapping["t"]], profile)})
    for channel, header in mapping.items():
        if channel != "t":
            samples[channel] = pd.to_numeric(table[header], errors="coerce").to_numpy(float)
    sport = None
    if "Sport" in table.columns and len(table):
        sport = str(table["Sport"].iloc[0]).lower()
    return RawActivity(samples, "csv", str(path), sport=sport)


def _times(column: pd.Series, profile: DeviceProfile) -> pd.Series:
    values = column.astype(str).tolist()
    if profile.csv_time_format:
        return pd.Series(pd.to_datetime(values, format=profile.csv_time_format))
    return parse_times(values)
