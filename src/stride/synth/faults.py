"""Fault injection with labels, so the benchmark knows exactly which samples are bad.

Every row carries the set of faults injected into it. Operations that move rows (inserting
a duplicate, swapping a pair, cutting a block) carry the labels along, so the final labels
match the row order the file is written in. Faults are placed in separate slots at least
20 rows apart, so one fault never changes how another is detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stride.geo import LocalFrame
from stride.registry import DeviceProfile

SLOT_SPACING = 25
EDGE = 15

# fault -> (probability that a recording gets it, max occurrences)
FAULT_RATES = {
    "gps_jump": (0.5, 3),
    "elevation_spike": (0.3, 2),
    "hr_out_of_range": (0.3, 2),
    "impossible_speed": (0.3, 2),
    "distance_decrease": (0.2, 1),
    "duplicate_timestamp": (0.3, 2),
    "non_monotonic": (0.3, 2),
    "timestamp_gap": (0.3, 1),
}
STRUCTURAL = ("timestamp_gap", "duplicate_timestamp", "non_monotonic")
NEEDS = {
    "elevation_spike": "ele", "hr_out_of_range": "hr", "impossible_speed": "speed",
    "distance_decrease": "distance",
}  # fmt: skip


@dataclass
class FaultedRecording:
    samples: pd.DataFrame
    labels: list[set[str]]  # per row, in final order
    injected: dict[str, int] = field(default_factory=dict)


class FaultInjector:
    def __init__(self, profile: DeviceProfile, origin: tuple[float, float, float]) -> None:
        self.profile = profile
        self.frame = LocalFrame(*origin)

    def run(self, rng: np.random.Generator, samples: pd.DataFrame) -> FaultedRecording:
        rows = samples.reset_index(drop=True).copy()
        labels: list[set[str]] = [set() for _ in range(len(rows))]
        plan = self._plan(rng, rows)
        injected: dict[str, int] = {}
        # Value edits first, while every slot still points at its planned row. Then the
        # operations that cut, add or move rows, from the last slot backwards, so a change
        # never shifts a slot that is still waiting.
        for fault, slot in plan:
            if fault not in STRUCTURAL:
                getattr(self, f"_{fault}")(rng, rows, labels, slot)
                injected[fault] = injected.get(fault, 0) + 1
        for fault, slot in sorted((p for p in plan if p[0] in STRUCTURAL), key=lambda p: -p[1]):
            rows, labels = getattr(self, f"_{fault}")(rng, rows, labels, slot)
            injected[fault] = injected.get(fault, 0) + 1
        return FaultedRecording(rows.reset_index(drop=True), labels, injected)

    def _plan(self, rng, rows) -> list[tuple[str, int]]:
        slots = list(range(EDGE, len(rows) - EDGE - 4, SLOT_SPACING))
        rng.shuffle(slots)
        plan = []
        for fault, (probability, most) in FAULT_RATES.items():
            if fault in NEEDS and NEEDS[fault] not in rows:
                continue
            if rng.random() < probability:
                for _ in range(rng.integers(1, most + 1)):
                    if slots:
                        plan.append((fault, slots.pop()))
        return plan

    def _gps_jump(self, rng, rows, labels, i):
        length = int(rng.integers(1, 4))
        # Sizes start below what the rule can see at higher speeds or sparse sampling, so
        # recall reflects real limits rather than a fault catalogue built to be caught.
        angle, size = rng.uniform(0, 2 * np.pi), rng.uniform(40, 1000)
        for j in range(i, i + length):
            enu = self.frame.to_enu(rows.at[j, "lat"], rows.at[j, "lon"])
            enu = enu.reshape(1, 3) + [size * np.cos(angle), size * np.sin(angle), 0]
            lat, lon, _ = self.frame.to_geodetic(enu)
            rows.at[j, "lat"], rows.at[j, "lon"] = float(lat[0]), float(lon[0])
            labels[j].add("gps_jump")

    def _elevation_spike(self, rng, rows, labels, i):
        rows.at[i, "ele"] += (
            rng.choice([-1, 1]) * rng.uniform(20, 250) / self.profile.factor("elevation")
        )
        labels[i].add("elevation_spike")

    def _hr_out_of_range(self, rng, rows, labels, i):
        rows.at[i, "hr"] = float(rng.choice([0, 12, 240, 250]))  # 255 means "no value" in FIT
        labels[i].add("hr_out_of_range")

    def _impossible_speed(self, rng, rows, labels, i):
        rows.at[i, "speed"] = rng.uniform(15, 90) / self.profile.factor("speed")
        labels[i].add("impossible_speed")

    def _distance_decrease(self, rng, rows, labels, i):
        drop = rng.uniform(50, 500) / self.profile.factor("distance")
        rows.at[i, "distance"] = max(0.0, rows.at[i - 1, "distance"] - drop)
        labels[i].add("distance_decrease")

    def _timestamp_gap(self, rng, rows, labels, i):
        """Cut a block lasting 5-150 s; the first sample after it is labelled."""
        seconds = (rows["t"] - rows.at[i, "t"]).dt.total_seconds().to_numpy()
        end = int(np.searchsorted(seconds, rng.uniform(5, 150)))
        end = min(max(end, i + 1), len(rows) - 1)
        rows = rows.drop(index=range(i, end)).reset_index(drop=True)
        labels = labels[:i] + labels[end:]
        labels[i].add("timestamp_gap")
        return rows, labels

    def _duplicate_timestamp(self, rng, rows, labels, i):
        """Repeat row i with slightly different values; the repeat is labelled."""
        copy = rows.iloc[[i]].copy()
        for column in ("lat", "lon"):
            if column in copy:
                copy[column] += rng.normal(0, 2e-6)
        rows = pd.concat([rows.iloc[: i + 1], copy, rows.iloc[i + 1 :]], ignore_index=True)
        labels = labels[: i + 1] + [{"duplicate_timestamp"}] + labels[i + 1 :]
        return rows, labels

    def _non_monotonic(self, rng, rows, labels, i):
        """Swap rows i and i+1; the earlier-time row now comes second and is labelled."""
        order = list(range(len(rows)))
        order[i], order[i + 1] = order[i + 1], order[i]
        rows = rows.iloc[order].reset_index(drop=True)
        labels = [labels[j] for j in order]
        labels[i + 1] = labels[i + 1] | {"non_monotonic"}
        return rows, labels
