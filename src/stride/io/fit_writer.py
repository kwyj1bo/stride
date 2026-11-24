"""Minimal FIT encoder: writes file_id, sport and record messages.

The synthetic benchmark uses it to produce real binary watch files, so the FIT reader is
tested on the same path as a file copied off a device.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd

from stride.io.fit import SEMICIRCLES_TO_DEG, SPORTS
from stride.io.fit_protocol import FIT_EPOCH_OFFSET, crc16

_SPORT_IDS = {name: num for num, name in SPORTS.items()}

# (field number, size, base type, struct code)
_FILE_ID = [(0, 1, 0x00, "B"), (4, 4, 0x86, "I")]
_SPORT = [(0, 1, 0x00, "B")]
_RECORD = [
    (253, 4, 0x86, "I"),
    (0, 4, 0x85, "i"),
    (1, 4, 0x85, "i"),
    (78, 4, 0x86, "I"),
    (3, 1, 0x02, "B"),
    (4, 1, 0x02, "B"),
    (5, 4, 0x86, "I"),
    (73, 4, 0x86, "I"),
]
_INVALID = {"B": 0xFF, "I": 0xFFFFFFFF, "i": 0x7FFFFFFF}


def write_fit(path: str | Path, samples: pd.DataFrame, sport: str = "running") -> None:
    """Write samples (column "t" in UTC plus lat, lon, ele, hr, cadence, distance, speed in
    SI units) as a FIT activity file."""
    times = _fit_seconds(samples["t"])
    body = bytearray()
    body += _definition(0, 0, _FILE_ID) + _data(0, _FILE_ID, [4, int(times[0])])
    body += _definition(1, 12, _SPORT) + _data(1, _SPORT, [_SPORT_IDS.get(sport, 0)])
    body += _definition(2, 20, _RECORD)
    for i, t in enumerate(times):
        body += _data(2, _RECORD, _record_values(samples, i, int(t)))
    header = struct.pack("<BBHI4s", 14, 0x20, 2132, len(body), b".FIT")
    header += struct.pack("<H", crc16(header))
    content = header + bytes(body)
    Path(path).write_bytes(content + struct.pack("<H", crc16(content)))


def _record_values(samples: pd.DataFrame, i: int, t: int) -> list:
    def get(name):
        value = samples[name].iloc[i] if name in samples else np.nan
        return None if pd.isna(value) else float(value)

    return [
        t,
        _scaled(get("lat"), 1 / SEMICIRCLES_TO_DEG, 0),
        _scaled(get("lon"), 1 / SEMICIRCLES_TO_DEG, 0),
        _scaled(get("ele"), 5, 500),
        _scaled(get("hr"), 1, 0),
        _scaled(get("cadence"), 1, 0),
        _scaled(get("distance"), 100, 0),
        _scaled(get("speed"), 1000, 0),
    ]


def _scaled(value: float | None, scale: float, offset: float) -> int | None:
    return None if value is None else int(round((value + offset) * scale))


def _fit_seconds(times: pd.Series) -> np.ndarray:
    # pandas may store times in ns, us or s, so divide by a Timedelta, not a constant
    seconds = (pd.to_datetime(times, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
    return seconds.to_numpy(dtype=np.int64) - FIT_EPOCH_OFFSET


def _definition(local: int, global_number: int, fields: list) -> bytes:
    out = struct.pack("<BBBHB", 0x40 | local, 0, 0, global_number, len(fields))
    for number, size, base_type, _ in fields:
        out += struct.pack("<BBB", number, size, base_type)
    return out


def _data(local: int, fields: list, values: list) -> bytes:
    out = struct.pack("<B", local)
    for (_, _, _, code), value in zip(fields, values, strict=True):
        out += struct.pack("<" + code, _INVALID[code] if value is None else value)
    return out
