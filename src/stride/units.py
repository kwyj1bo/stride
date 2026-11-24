"""Unit conversion to SI. Each table maps a unit name to its factor into the base unit."""

from __future__ import annotations

DISTANCE_TO_M = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "ft": 0.3048}
SPEED_TO_MPS = {"m/s": 1.0, "km/h": 1000.0 / 3600.0, "mph": 1609.344 / 3600.0}
ELEVATION_TO_M = {"m": 1.0, "ft": 0.3048}

TABLES = {"distance": DISTANCE_TO_M, "speed": SPEED_TO_MPS, "elevation": ELEVATION_TO_M}


def factor(quantity: str, unit: str) -> float:
    """Multiplier that converts `unit` into the SI unit for `quantity`."""
    table = TABLES[quantity]
    if unit not in table:
        raise ValueError(f"unknown {quantity} unit {unit!r}; known: {sorted(table)}")
    return table[unit]
