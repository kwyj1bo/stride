"""Group recordings of the same session and fuse them into one activity.

Fusion works channel by channel: for every second, take the value from the most
trustworthy device that has one. Trust is the channel's completeness times the device's
validity score, so a watch with clean GPS supplies position while a chest strap paired to
the phone supplies heart rate. Provenance (which device supplied how much of each channel)
is kept with the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stride.resample import _seconds
from stride.schema import CHANNELS, Activity


@dataclass
class FusedActivity:
    activity: Activity
    provenance: dict[str, dict[str, float]] = field(default_factory=dict)  # channel -> share


class Fuser:
    def run(
        self, session_id: str, members: list[Activity], quality: dict[str, dict[str, float]]
    ) -> FusedActivity:
        """members: resampled activities on the same grid; quality: device_id -> metrics."""
        grid = _union_grid(members)
        aligned = {m.device_id: _on_grid(m, grid) for m in members}
        out = pd.DataFrame({"t": pd.to_datetime(grid * 1e6, unit="us", utc=True).as_unit("ns")})
        provenance = {}
        for name in CHANNELS:
            order = sorted(aligned, key=lambda d: -_trust(quality.get(d, {}), name))
            values = np.full(len(grid), np.nan)
            source = np.full(len(grid), "", dtype=object)
            for device in order:
                fill = np.isnan(values) & ~np.isnan(aligned[device][name])
                values[fill] = aligned[device][name][fill]
                source[fill] = device
            out[name] = values
            provenance[name] = _shares(source)
        sport = next((m.sport for m in members if m.sport), None)
        note = f"fused from {sorted(aligned)}"
        return FusedActivity(Activity(session_id, "fused", out, sport, notes=[note]), provenance)


def group_sessions(activities: list[Activity], tolerance_s: float = 300.0) -> list[list[Activity]]:
    """Chain recordings whose time spans overlap (allowing for clock offsets up to
    tolerance_s) into sessions. Sessions come back ordered by start time."""
    spans = sorted(
        (
            (_seconds(a.samples["t"]).min(), _seconds(a.samples["t"]).max(), a)
            for a in activities
            if len(a.samples)
        ),
        key=lambda span: span[0],
    )
    sessions: list[list[Activity]] = []
    end = -np.inf
    for start, stop, activity in spans:
        if sessions and start <= end + tolerance_s:
            sessions[-1].append(activity)
            end = max(end, stop)
        else:
            sessions.append([activity])
            end = stop
    return sessions


def _trust(metrics: dict[str, float], channel: str) -> float:
    return metrics.get(f"completeness_{channel}", 0.0) * metrics.get("validity", 0.0)


def _union_grid(members: list[Activity]) -> np.ndarray:
    seconds = np.concatenate([_seconds(m.samples["t"]) for m in members])
    return np.unique(np.round(seconds, 6))


def _on_grid(activity: Activity, grid: np.ndarray) -> dict[str, np.ndarray]:
    seconds = np.round(_seconds(activity.samples["t"]), 6)
    index = np.searchsorted(grid, seconds)
    out = {}
    for name in CHANNELS:
        column = np.full(len(grid), np.nan)
        column[index] = activity.samples[name].to_numpy(float)
        out[name] = column
    return out


def _shares(source: np.ndarray) -> dict[str, float]:
    filled = source[source != ""]
    if not len(filled):
        return {}
    devices, counts = np.unique(filled.astype(str), return_counts=True)
    return {d: round(float(c) / len(source), 4) for d, c in zip(devices, counts, strict=True)}
