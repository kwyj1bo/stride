"""Turn decoded FIT record messages into a RawActivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stride.io.fit_protocol import FIT_EPOCH_OFFSET, TIMESTAMP_FIELD, FitDecoder
from stride.schema import RawActivity

MESG_RECORD = 20
MESG_SPORT = 12
MESG_SESSION = 18

SEMICIRCLES_TO_DEG = 180.0 / 2**31
SPORTS = {1: "running", 2: "cycling", 5: "swimming", 11: "walking", 17: "hiking"}

# record field number -> (raw channel, scale, offset); value = raw / scale - offset
RECORD_FIELDS = {
    0: ("lat", 1 / SEMICIRCLES_TO_DEG, 0.0),
    1: ("lon", 1 / SEMICIRCLES_TO_DEG, 0.0),
    2: ("ele", 5.0, 500.0),
    3: ("hr", 1.0, 0.0),
    4: ("cadence", 1.0, 0.0),
    5: ("distance", 100.0, 0.0),
    6: ("speed", 1000.0, 0.0),
}
# Newer devices write 32-bit "enhanced" fields; they win over the 16-bit ones.
ENHANCED_FIELDS = {78: ("ele", 5.0, 500.0), 73: ("speed", 1000.0, 0.0)}


def read_fit(path: str | Path) -> RawActivity:
    decoder = FitDecoder(Path(path).read_bytes(), keep={MESG_RECORD, MESG_SPORT, MESG_SESSION})
    messages = decoder.messages()
    records = [v for num, v in messages if num == MESG_RECORD and v.get(TIMESTAMP_FIELD)]
    if not records:
        raise ValueError(f"{path}: no record messages with a timestamp")
    seconds = np.array([r[TIMESTAMP_FIELD] for r in records], dtype=np.int64)
    samples = pd.DataFrame({"t": pd.to_datetime(seconds + FIT_EPOCH_OFFSET, unit="s", utc=True)})
    for channel, values in _channels(records).items():
        samples[channel] = values
    return RawActivity(samples, "fit", str(path), sport=_sport(messages))


def _channels(records: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for fields in (RECORD_FIELDS, ENHANCED_FIELDS):
        for number, (channel, scale, offset) in fields.items():
            raw = np.array([_num(r.get(number)) for r in records])
            if np.all(np.isnan(raw)):
                continue
            values = raw / scale - offset
            if channel in out:  # enhanced field: fill only where it has a value
                values = np.where(np.isnan(values), out[channel], values)
            out[channel] = values
    return out


def _sport(messages: list[tuple[int, dict]]) -> str | None:
    for number, values in messages:
        field = 0 if number == MESG_SPORT else 5 if number == MESG_SESSION else None
        if field is not None and values.get(field) is not None:
            return SPORTS.get(values[field], f"sport_{values[field]}")
    return None


def _num(value) -> float:
    return float("nan") if value is None else float(value)
