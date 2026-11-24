"""Normalization: device units to SI, device-local time to UTC, raw channel names to
canonical ones.

Sample order and duplicates are left alone on purpose. Validation needs to see the data
as the device wrote it; cleaning happens after validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stride.registry import DeviceProfile
from stride.schema import CHANNELS, RAW_TO_CANONICAL, Activity, RawActivity

# raw channel -> unit quantity from the device profile
_UNIT_OF = {"ele": "elevation", "distance": "distance", "speed": "speed"}


class Normalizer:
    def __init__(self, profile: DeviceProfile) -> None:
        self.profile = profile

    def run(self, raw: RawActivity, activity_id: str) -> Activity:
        notes: list[str] = []
        times, dropped = self._to_utc(raw.samples["t"])
        if dropped:
            notes.append(
                f"dropped {dropped} samples whose local time does not exist or is ambiguous"
            )
        keep = times.notna().to_numpy()
        samples = pd.DataFrame({"t": times[keep].reset_index(drop=True)})
        for raw_name, name in RAW_TO_CANONICAL.items():
            samples[name] = self._channel(raw.samples, raw_name)[keep]
        self._note_units(notes)
        return Activity(
            activity_id=activity_id,
            device_id=self.profile.device_id,
            samples=samples[["t", *CHANNELS]],
            sport=raw.sport,
            source_path=raw.source_path,
            notes=notes,
        )

    def _to_utc(self, times: pd.Series) -> tuple[pd.Series, int]:
        times = pd.Series(pd.to_datetime(times))
        if times.dt.tz is None:
            # Timestamps with no offset are in the device's local zone. A daylight-saving
            # change makes some local times ambiguous or missing; those become NaT.
            times = times.dt.tz_localize(self.profile.timezone, ambiguous="NaT", nonexistent="NaT")
        dropped = int(times.isna().sum())
        return times.dt.tz_convert("UTC").dt.as_unit("ns"), dropped

    def _channel(self, samples: pd.DataFrame, raw_name: str) -> np.ndarray:
        if raw_name not in samples:
            return np.full(len(samples), np.nan)
        values = samples[raw_name].to_numpy(dtype=float)
        quantity = _UNIT_OF.get(raw_name)
        return values * self.profile.factor(quantity) if quantity else values

    def _note_units(self, notes: list[str]) -> None:
        converted = {q: u for q, u in self.profile.units.items() if self.profile.factor(q) != 1.0}
        if converted:
            notes.append(f"converted units to SI from {converted}")
        if self.profile.timezone != "UTC":
            notes.append(f"read naive timestamps as {self.profile.timezone}")
