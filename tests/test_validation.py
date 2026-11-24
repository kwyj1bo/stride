import numpy as np
import pytest
from helpers import straight_run

from stride.clean import Cleaner
from stride.geo import LocalFrame
from stride.validation import Validator


def flagged(report, rule):
    return np.flatnonzero(report.flags[rule]).tolist()


def test_clean_run_passes_with_full_quality():
    report = Validator(nominal_hz=1.0).run(straight_run())
    assert report.passed
    assert report.quality["validity"] == 1.0
    assert report.quality["rate_ratio"] == pytest.approx(1.0)
    assert report.quality["completeness_hr_bpm"] == 1.0


def test_each_injected_fault_is_flagged_at_its_sample():
    activity = straight_run()
    s = activity.samples
    lat, lon, _ = LocalFrame(*s.loc[50, ["lat", "lon", "ele_m"]]).to_geodetic(
        np.array([[400.0, 300.0, 0.0]])
    )
    s.loc[50, ["lat", "lon"]] = [lat[0], lon[0]]  # 500 m teleport for one sample
    s.loc[60, "hr_bpm"] = 250
    s.loc[70, "ele_m"] += 120
    s.loc[80, "speed_mps"] = 45
    s.loc[90, "distance_m"] = 10
    s.loc[100, "t"] = s.loc[99, "t"]  # repeated timestamp
    s.loc[[150, 151]] = s.loc[[151, 150]].to_numpy()  # out-of-order pair
    activity.samples = s.drop(index=range(200, 230)).reset_index(drop=True)
    report = Validator(nominal_hz=1.0).run(activity)
    assert flagged(report, "gps_jump") == [50]
    assert flagged(report, "hr_out_of_range") == [60]
    assert flagged(report, "elevation_spike") == [70]
    assert flagged(report, "impossible_speed") == [80]
    assert flagged(report, "distance_decrease") == [90]
    assert flagged(report, "duplicate_timestamp") == [100]
    assert flagged(report, "non_monotonic") == [151]
    assert flagged(report, "timestamp_gap") == [200]  # first sample after the removed block
    assert not report.passed


def test_multi_point_detour_is_flagged_but_a_real_relocation_is_accepted():
    activity = straight_run()
    s = activity.samples
    frame = LocalFrame(*s.loc[0, ["lat", "lon", "ele_m"]])
    for i in range(40, 43):  # 3-sample excursion that comes back
        lat, lon, _ = frame.to_geodetic(np.array([[3.0 * i, 800.0, 0.0]]))
        s.loc[i, ["lat", "lon"]] = [lat[0], lon[0]]
    for i in range(150, 300):  # the device is driven 5 km north and keeps recording
        lat, lon, _ = frame.to_geodetic(np.array([[3.0 * i, 5000.0, 0.0]]))
        s.loc[i, ["lat", "lon"]] = [lat[0], lon[0]]
    report = Validator(nominal_hz=1.0).run(activity)
    # The detour is flagged. The relocation is not: forwards it looks like five jumps,
    # backwards the five fixes before it do, and the two passes never agree.
    assert flagged(report, "gps_jump") == [40, 41, 42]


@pytest.mark.parametrize(
    "scale, registered, expected",
    [
        (1 / 1609.344, {"distance": "m"}, "looks like mi"),
        (1 / 1000.0, {"distance": "m"}, "looks like km"),
        (1000 / 1609.344, {"distance": "km"}, "looks like mi"),
    ],
)
def test_unit_mix_up_is_named(scale, registered, expected):
    activity = straight_run()
    activity.samples["distance_m"] *= scale
    findings = Validator(1.0, registered).run(activity).unit_findings
    assert len(findings) == 1
    assert expected in findings[0].cause


def test_speed_registered_as_mps_but_written_in_kmh():
    activity = straight_run()
    activity.samples["speed_mps"] *= 3.6
    findings = Validator(1.0, {"speed": "m/s"}).run(activity).unit_findings
    assert [f.channel for f in findings] == ["speed_mps"]
    assert "looks like km/h" in findings[0].cause


def test_cleaner_sorts_dedupes_and_blanks_only_the_bad_channel():
    activity = straight_run(n=50)
    s = activity.samples
    s.loc[10, "hr_bpm"] = 300
    s.loc[20, "t"] = s.loc[19, "t"]
    s.loc[[30, 31]] = s.loc[[31, 30]].to_numpy()
    report = Validator(nominal_hz=1.0).run(activity)
    cleaned = Cleaner().run(activity, report).samples
    assert len(cleaned) == 49
    assert cleaned["t"].is_monotonic_increasing and cleaned["t"].is_unique
    row = cleaned[cleaned["t"] == s.loc[10, "t"]].iloc[0]
    assert np.isnan(row["hr_bpm"]) and not np.isnan(row["lat"])


def test_slipped_jump_does_not_flag_the_good_fixes_after_it():
    # A drift away in 28 m steps never breaks the 32 m limit, so the forward pass trusts
    # it, then sees the return to the true track as a jump. The backward pass disagrees.
    activity = straight_run()
    s = activity.samples
    frame = LocalFrame(*s.loc[0, ["lat", "lon", "ele_m"]])
    for k, i in enumerate(range(100, 103), start=1):
        lat, lon, _ = frame.to_geodetic(np.array([[3.0 * i, 28.0 * k, 0.0]]))
        s.loc[i, ["lat", "lon"]] = [lat[0], lon[0]]
    report = Validator(nominal_hz=1.0).run(activity)
    assert not set(flagged(report, "gps_jump")) & set(range(103, 110))
