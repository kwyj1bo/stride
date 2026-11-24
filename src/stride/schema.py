"""The data shapes that move through the pipeline.

Parsers emit a RawActivity: values in the source device's units, timestamps as the file
stores them. Normalization turns that into the canonical frame below: SI units, UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"

# Measurement channels, in canonical (SI) units after normalization.
CHANNELS = ("lat", "lon", "ele_m", "hr_bpm", "cadence_spm", "speed_mps", "distance_m")

# Channels a parser may emit, before unit conversion.
RAW_CHANNELS = ("lat", "lon", "ele", "hr", "cadence", "speed", "distance")

RAW_TO_CANONICAL = dict(zip(RAW_CHANNELS, CHANNELS, strict=True))

CANONICAL_COLUMNS = ("activity_id", "device_id", "t", *CHANNELS)


@dataclass
class RawActivity:
    """One recording as parsed from one file."""

    samples: pd.DataFrame  # column "t" plus any of RAW_CHANNELS
    source_format: str
    source_path: str = ""
    sport: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "t" not in self.samples.columns:
            raise ValueError("samples need a 't' column")
        unknown = set(self.samples.columns) - {"t", *RAW_CHANNELS}
        if unknown:
            raise ValueError(f"unknown raw channels: {sorted(unknown)}")


@dataclass
class Activity:
    """One recording in canonical form: SI units, UTC timestamps, sorted by time."""

    activity_id: str
    device_id: str
    samples: pd.DataFrame  # column "t" (UTC) plus CHANNELS
    sport: str | None = None
    source_path: str = ""
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        frame = self.samples.copy()
        frame.insert(0, "device_id", self.device_id)
        frame.insert(0, "activity_id", self.activity_id)
        return frame[list(CANONICAL_COLUMNS)]


def empty_channels(n: int) -> dict[str, np.ndarray]:
    return {name: np.full(n, np.nan) for name in CHANNELS}
