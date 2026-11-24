"""Device registry: what each recording device writes, and in which units and time zone.

Every input file belongs to one registered device. The registry is the single place that
says "this watch stores distance in miles" or "this app writes local time without a zone",
so the pipeline never guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from stride import units

FORMATS = ("gpx", "tcx", "fit", "csv")
DEFAULT_UNITS = {"distance": "m", "speed": "m/s", "elevation": "m"}


@dataclass(frozen=True)
class DeviceProfile:
    device_id: str
    name: str
    format: str
    timezone: str = "UTC"  # zone for timestamps that carry no offset
    nominal_hz: float = 1.0
    units: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UNITS))
    csv_columns: dict[str, str] = field(default_factory=dict)  # raw channel -> CSV header
    csv_time_format: str | None = None

    def validate(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"{self.device_id}: format must be one of {FORMATS}")
        for quantity, unit in self.units.items():
            units.factor(quantity, unit)
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"{self.device_id}: unknown timezone {self.timezone!r}") from error
        if self.nominal_hz <= 0:
            raise ValueError(f"{self.device_id}: nominal_hz must be positive")
        if self.format == "csv" and "t" not in self.csv_columns:
            raise ValueError(f"{self.device_id}: csv devices must map the 't' column")

    def factor(self, quantity: str) -> float:
        return units.factor(quantity, self.units.get(quantity, DEFAULT_UNITS[quantity]))


class Registry:
    def __init__(self, profiles: list[DeviceProfile]) -> None:
        self._profiles: dict[str, DeviceProfile] = {}
        for profile in profiles:
            profile.validate()
            if profile.device_id in self._profiles:
                raise ValueError(f"duplicate device id {profile.device_id!r}")
            self._profiles[profile.device_id] = profile

    @classmethod
    def load(cls, path: str | Path | None = None) -> Registry:
        """Load a registry file, or the built-in one when no path is given."""
        if path is None:
            text = resources.files("stride").joinpath("devices.yaml").read_text()
        else:
            text = Path(path).read_text()
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, data: dict) -> Registry:
        profiles = []
        for device_id, spec in (data.get("devices") or {}).items():
            spec = dict(spec)
            spec["units"] = {**DEFAULT_UNITS, **spec.get("units", {})}
            profiles.append(DeviceProfile(device_id=device_id, **spec))
        return cls(profiles)

    def get(self, device_id: str) -> DeviceProfile:
        if device_id not in self._profiles:
            raise KeyError(f"device {device_id!r} is not in the registry")
        return self._profiles[device_id]

    def __contains__(self, device_id: str) -> bool:
        return device_id in self._profiles

    def __iter__(self):
        return iter(self._profiles.values())

    def __len__(self) -> int:
        return len(self._profiles)
