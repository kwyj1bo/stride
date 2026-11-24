import math
from pathlib import Path

import pandas as pd
import pytest

from stride.io import read
from stride.io._common import parse_times
from stride.registry import Registry

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def registry():
    return Registry.load()


def test_gpx_reads_points_extensions_and_mixed_offsets(registry):
    raw = read(DATA / "run.gpx", registry.get("phone_gpx"))
    s = raw.samples
    assert raw.sport == "running"
    assert len(s) == 3
    assert s["hr"].tolist()[:2] == [120, 122]
    assert math.isnan(s["hr"].iloc[2])
    # +05:30 local time is the same instant as 01:00:02Z
    assert s["t"].iloc[2] == pd.Timestamp("2026-11-01T01:00:02Z")


def test_tcx_skips_points_without_time_and_reads_speed(registry):
    raw = read(DATA / "ride.tcx", registry.get("bike_tcx"))
    s = raw.samples
    assert raw.sport == "biking"
    assert len(s) == 2
    assert s["speed"].iloc[0] == 6.5
    assert s["distance"].tolist() == [0.0, 26.0]
    assert math.isnan(s["hr"].iloc[1])


def test_csv_uses_profile_columns_and_keeps_naive_local_time(registry):
    raw = read(DATA / "app.csv", registry.get("app_csv"))
    s = raw.samples
    assert s["t"].dt.tz is None
    assert s["t"].iloc[0] == pd.Timestamp("2026-11-01 06:30:00")
    assert s["distance"].iloc[1] == pytest.approx(0.0034)
    assert math.isnan(s["hr"].iloc[2])
    assert raw.sport == "running"


def test_parse_times_rejects_mixed_naive_and_aware():
    with pytest.raises(ValueError, match="mixes"):
        parse_times(["2026-11-01T01:00:00Z", "2026-11-01T01:00:01"])
