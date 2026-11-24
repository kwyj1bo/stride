"""Activity-level check that the registry's units match what the device really wrote.

The device's own distance and speed channels are compared with what the GPS track says.
If registry says unit R but the device wrote unit A, normalized values come out scaled by
factor(R) / factor(A). A ratio near 1 is fine; a ratio near one of those known factors
names the likely mistake. The pipeline reports this instead of silently rescaling: the
fix belongs in the registry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stride import units
from stride.geo import path_length
from stride.validation.rules import Context


@dataclass
class UnitFinding:
    channel: str
    ratio: float
    cause: str


class UnitCheck:
    MIN_TRACK_M = 200.0  # a shorter track gives an unstable ratio
    TOLERANCE = 0.06  # relative match to a known unit-mix-up ratio
    AGREEMENT = 0.15  # ratios this close to 1 are fine: GPS noise inflates track length

    def __init__(self, registered: dict[str, str]) -> None:
        self.registered = registered

    def check(self, ctx: Context) -> list[UnitFinding]:
        order = np.argsort(ctx.seconds, kind="stable")
        enu = ctx.enu[order]
        gps_m = path_length(enu)
        if gps_m < self.MIN_TRACK_M:
            return []
        ratios = {
            "distance": self._distance_ratio(ctx, order, gps_m),
            "speed": self._speed_ratio(ctx, order, enu, gps_m),
        }
        findings = []
        for quantity, ratio in ratios.items():
            finding = self._classify(quantity, ratio)
            if finding is not None:
                findings.append(finding)
        return findings

    def _distance_ratio(self, ctx, order, gps_m) -> float | None:
        dist = ctx.channel("distance_m")[order]
        dist = dist[~np.isnan(dist)]
        if len(dist) < 2:
            return None
        return (dist.max() - dist.min()) / gps_m

    def _speed_ratio(self, ctx, order, enu, gps_m) -> float | None:
        speed = ctx.channel("speed_mps")[order]
        fixes = ctx.seconds[order][~np.isnan(enu[:, 0])]
        has_speed = ~np.isnan(speed)
        if has_speed.sum() < 10 or len(fixes) < 10 or fixes[-1] <= fixes[0]:
            return None
        # Time-weighted mean, so devices that record less often when steady are not biased.
        t = ctx.seconds[order][has_speed]
        mean_speed = np.trapezoid(speed[has_speed], t) / (t[-1] - t[0])
        return float(mean_speed / (gps_m / (fixes[-1] - fixes[0])))

    def _classify(self, quantity: str, ratio: float | None) -> UnitFinding | None:
        if ratio is None or not np.isfinite(ratio) or abs(ratio - 1) <= self.AGREEMENT:
            return None
        channel = f"{quantity}_{'m' if quantity == 'distance' else 'mps'}"
        registered = self.registered.get(quantity, "m" if quantity == "distance" else "m/s")
        reg_factor = units.factor(quantity, registered)
        for actual, act_factor in units.TABLES[quantity].items():
            if (
                actual != registered
                and abs(ratio / (reg_factor / act_factor) - 1) <= self.TOLERANCE
            ):
                cause = f"registered as {registered}, but the data looks like {actual}"
                return UnitFinding(channel, float(ratio), cause)
        return UnitFinding(channel, float(ratio), "device value disagrees with the GPS track")
