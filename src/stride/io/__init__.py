"""Readers for each supported file format."""

from __future__ import annotations

from pathlib import Path

from stride.io.csv_ import read_csv
from stride.io.fit import read_fit
from stride.io.gpx import read_gpx
from stride.io.tcx import read_tcx
from stride.registry import DeviceProfile
from stride.schema import RawActivity


def read(path: str | Path, profile: DeviceProfile) -> RawActivity:
    """Parse one file with the reader its device profile names."""
    if profile.format == "gpx":
        return read_gpx(path)
    if profile.format == "tcx":
        return read_tcx(path)
    if profile.format == "fit":
        return read_fit(path)
    if profile.format == "csv":
        return read_csv(path, profile)
    raise ValueError(f"no reader for format {profile.format!r}")
