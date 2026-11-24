import numpy as np
import pandas as pd
from helpers import straight_run

from stride.fuse import Fuser, group_sessions


def shifted(activity, seconds, device):
    activity.samples["t"] = activity.samples["t"] + pd.Timedelta(seconds=seconds)
    activity.device_id = device
    return activity


def test_overlapping_recordings_form_one_session_and_later_ones_another():
    a = shifted(straight_run(n=600), 0, "watch")
    b = shifted(straight_run(n=500), 30, "phone")
    c = shifted(straight_run(n=100), 5000, "watch")
    sessions = group_sessions([c, b, a])
    assert [[m.device_id for m in s] for s in sessions] == [["watch", "phone"], ["watch"]]


def test_each_channel_comes_from_the_most_trusted_device_with_fallback():
    watch = straight_run(n=10)
    watch.device_id = "watch"
    watch.samples["hr_bpm"] = np.nan
    watch.samples.loc[5, "lat"] = np.nan
    phone = straight_run(n=10)
    phone.device_id = "phone"
    phone.samples["hr_bpm"] = 150.0
    phone.samples["lat"] += 0.001
    quality = {
        "watch": {"validity": 1.0, "completeness_lat": 0.9, "completeness_hr_bpm": 0.0},
        "phone": {"validity": 0.8, "completeness_lat": 1.0, "completeness_hr_bpm": 1.0},
    }
    fused = Fuser().run("s1", [watch, phone], quality)
    s = fused.activity.samples
    assert (s["hr_bpm"] == 150.0).all()
    # watch wins position (0.9 * 1.0 > 1.0 * 0.8) except where it has no fix
    assert s.loc[4, "lat"] == watch.samples.loc[4, "lat"]
    assert s.loc[5, "lat"] == phone.samples.loc[5, "lat"]
    assert fused.provenance["lat"] == {"phone": 0.1, "watch": 0.9}
    assert fused.provenance["hr_bpm"] == {"phone": 1.0}
