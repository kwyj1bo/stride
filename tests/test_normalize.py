from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stride.io import read
from stride.normalize import Normalizer
from stride.registry import Registry
from stride.schema import CHANNELS, RawActivity

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def registry():
    return Registry.load()


def test_imperial_local_time_csv_becomes_si_utc(registry):
    profile = registry.get("app_csv")
    activity = Normalizer(profile).run(read(DATA / "app.csv", profile), "a1")
    s = activity.samples
    assert list(s.columns) == ["t", *CHANNELS]
    # 06:30 in Asia/Kolkata (UTC+5:30) is 01:00 UTC
    assert s["t"].iloc[0] == pd.Timestamp("2026-11-01T01:00:00Z")
    assert s["ele_m"].iloc[0] == pytest.approx(32.8 * 0.3048)
    assert s["speed_mps"].iloc[0] == pytest.approx(6.0 * 0.44704)
    assert s["distance_m"].iloc[1] == pytest.approx(0.0034 * 1609.344)
    assert any("Asia/Kolkata" in note for note in activity.notes)


def test_order_and_duplicates_are_kept_for_validation(registry):
    t = pd.to_datetime(["2026-11-01T01:00:02Z", "2026-11-01T01:00:01Z", "2026-11-01T01:00:01Z"])
    raw = RawActivity(pd.DataFrame({"t": t, "hr": [1.0, 2.0, 3.0]}), "gpx")
    s = Normalizer(registry.get("phone_gpx")).run(raw, "a").samples
    assert s["hr_bpm"].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(s["lat"]).all()


def test_nonexistent_local_times_are_dropped_with_a_note():
    registry = Registry.from_dict(
        {
            "devices": {
                "ny": {
                    "name": "x",
                    "format": "csv",
                    "timezone": "America/New_York",
                    "csv_columns": {"t": "Time"},
                }
            }
        }
    )
    # 02:30 on 8 March 2026 does not exist in New York (clocks jump 02:00 -> 03:00)
    t = pd.to_datetime(["2026-03-08 01:59:00", "2026-03-08 02:30:00", "2026-03-08 03:01:00"])
    raw = RawActivity(pd.DataFrame({"t": t, "hr": [1.0, 2.0, 3.0]}), "csv")
    activity = Normalizer(registry.get("ny")).run(raw, "a")
    assert activity.samples["hr_bpm"].tolist() == [1.0, 3.0]
    assert any("dropped 1" in note for note in activity.notes)
