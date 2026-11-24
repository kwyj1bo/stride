"""Cleaning: apply a validation report. Samples are put in time order, repeated timestamps
are dropped, and every value an error rule flagged is blanked in the channel it concerns,
so a bad GPS fix loses its position but keeps its heart rate."""

from __future__ import annotations

import numpy as np

from stride.schema import Activity
from stride.validation import ValidationReport

# rule -> channels it invalidates
BLANKS = {
    "gps_jump": ("lat", "lon"),
    "impossible_speed": ("speed_mps",),
    "hr_out_of_range": ("hr_bpm",),
    "elevation_spike": ("ele_m",),
    "distance_decrease": ("distance_m",),
}


class Cleaner:
    def run(self, activity: Activity, report: ValidationReport) -> Activity:
        samples = activity.samples.copy()
        for rule, channels in BLANKS.items():
            if rule in report.flags:
                mask = report.flags[rule].to_numpy()
                samples.loc[mask, list(channels)] = np.nan
        samples = samples.sort_values("t", kind="stable")
        samples = samples.drop_duplicates("t", keep="first").reset_index(drop=True)
        removed = len(activity.samples) - len(samples)
        blanked = {r: report.count(r) for r in BLANKS if r in report.flags and report.count(r)}
        notes = [*activity.notes, f"cleaning removed {removed} samples, blanked {blanked}"]
        return Activity(
            activity.activity_id,
            activity.device_id,
            samples,
            activity.sport,
            activity.source_path,
            notes,
        )
