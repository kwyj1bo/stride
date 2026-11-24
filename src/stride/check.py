"""Independent checks on a written snapshot, run before it is released.

The checker reads only what is on disk. It does not reuse pipeline objects, so a bug in
the pipeline cannot hide itself from the check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stride.schema import CANONICAL_COLUMNS
from stride.snapshot import TABLES, table_hash

RANGES = {
    "lat": (-90, 90), "lon": (-180, 180), "hr_bpm": (30, 230),
    "speed_mps": (0, 30), "cadence_spm": (0, 300), "ele_m": (-500, 9000),
}  # fmt: skip


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


class SnapshotChecker:
    def __init__(self, folder: str | Path, min_validity: float = 0.9) -> None:
        self.folder = Path(folder)
        self.min_validity = min_validity

    def run(self) -> list[Check]:
        manifest = json.loads((self.folder / "manifest.json").read_text())
        tables = {name: pd.read_parquet(self.folder / f"{name}.parquet") for name in TABLES}
        checks = [self._hash(name, frame, manifest) for name, frame in tables.items()]
        checks.append(self._columns(tables))
        for name, key in (("recordings", "activity_id"), ("sessions", "session_id")):
            checks.append(self._time_order(name, tables[name], key))
        checks.append(self._grid(tables["recordings"], manifest["config"]["rate_hz"]))
        checks.append(self._ranges(tables))
        checks.append(self._quality(manifest))
        return checks

    def _hash(self, name, frame, manifest) -> Check:
        ok = table_hash(frame) == manifest["content"][name]
        return Check(f"{name}_content_hash", ok, "" if ok else "data differs from the manifest")

    def _columns(self, tables) -> Check:
        rec = tuple(tables["recordings"].columns)
        ses = tuple(tables["sessions"].columns)
        expected_sessions = ("session_id", *CANONICAL_COLUMNS[1:])
        ok = rec == CANONICAL_COLUMNS and ses == expected_sessions
        return Check("schema_columns", ok, "" if ok else f"recordings {rec}, sessions {ses}")

    def _time_order(self, name, frame, key) -> Check:
        if frame["t"].isna().any():
            return Check(f"{name}_time_order", False, "missing timestamps")
        bad = [
            k
            for k, group in frame.groupby(key, sort=False)
            if not group["t"].is_monotonic_increasing or not group["t"].is_unique
        ]
        return Check(f"{name}_time_order", not bad, f"not strictly increasing: {bad[:5]}")

    def _grid(self, frame, rate_hz) -> Check:
        step_us = round(1e6 / rate_hz)
        micros = (frame["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1us")
        off_grid = int((micros % step_us != 0).sum())
        return Check("on_resample_grid", off_grid == 0, f"{off_grid} samples off the grid")

    def _ranges(self, tables) -> Check:
        problems = []
        for name, frame in tables.items():
            for column, (lo, hi) in RANGES.items():
                values = frame[column].to_numpy(float)
                out = int(((values < lo) | (values > hi)).sum())
                if out:
                    problems.append(f"{name}.{column}: {out} outside [{lo}, {hi}]")
        return Check("value_ranges", not problems, "; ".join(problems))

    def _quality(self, manifest) -> Check:
        validity = manifest["quality"]["mean_validity"]
        failed = manifest["counts"]["failed"]
        ok = validity >= self.min_validity and failed == 0
        return Check("quality_gate", ok, f"mean validity {validity:.4f}, failed files {failed}")


def all_passed(checks: list[Check]) -> bool:
    return all(c.passed for c in checks) and bool(np.size(checks))
