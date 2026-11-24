"""Coordinate frames. GPS gives latitude/longitude on the WGS84 ellipsoid; analysis wants
metres. LocalFrame converts to East-North-Up (ENU) metres around a reference point, through
Earth-centred Earth-fixed (ECEF) coordinates, with no flat-earth shortcut.
"""

from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0  # semi-major axis, metres
WGS84_F = 1 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # first eccentricity squared


class LocalFrame:
    """East-North-Up frame with its origin at (lat0, lon0, h0)."""

    def __init__(self, lat0: float, lon0: float, h0: float = 0.0) -> None:
        self.origin = geodetic_to_ecef(lat0, lon0, h0)
        phi, lam = np.radians(lat0), np.radians(lon0)
        # Rows are the east, north and up unit vectors expressed in ECEF.
        self.rotation = np.array(
            [
                [-np.sin(lam), np.cos(lam), 0.0],
                [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
                [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
            ]
        )

    @classmethod
    def at_first_fix(cls, lat, lon, h=None) -> LocalFrame:
        """Frame anchored at the first sample that has a position."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        valid = np.flatnonzero(~np.isnan(lat) & ~np.isnan(lon))
        if not len(valid):
            raise ValueError("no valid position to anchor the frame")
        i = valid[0]
        h0 = 0.0 if h is None or np.isnan(np.asarray(h, float)[i]) else float(np.asarray(h)[i])
        return cls(float(lat[i]), float(lon[i]), h0)

    def to_enu(self, lat, lon, h=None) -> np.ndarray:
        """(n, 3) array of east, north, up metres. Missing height counts as h0."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        h = np.full_like(lat, self._h0()) if h is None else np.asarray(h, float)
        h = np.where(np.isnan(h), self._h0(), h)
        ecef = np.stack(geodetic_to_ecef(lat, lon, h), axis=-1)
        return (ecef - np.array(self.origin)) @ self.rotation.T

    def to_geodetic(self, enu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ecef = np.asarray(enu, float) @ self.rotation + np.array(self.origin)
        return ecef_to_geodetic(ecef[..., 0], ecef[..., 1], ecef[..., 2])

    def _h0(self) -> float:
        return float(ecef_to_geodetic(*self.origin)[2])


def geodetic_to_ecef(lat, lon, h):
    phi, lam = np.radians(lat), np.radians(lon)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(phi) ** 2)  # prime vertical radius
    x = (n + h) * np.cos(phi) * np.cos(lam)
    y = (n + h) * np.cos(phi) * np.sin(lam)
    z = (n * (1 - WGS84_E2) + h) * np.sin(phi)
    return x, y, z


def ecef_to_geodetic(x, y, z, iterations: int = 5):
    """Inverse of geodetic_to_ecef by fixed-point iteration on latitude. Five rounds
    reach sub-millimetre accuracy for points near the surface."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    lam = np.arctan2(y, x)
    p = np.hypot(x, y)
    phi = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(iterations):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(phi) ** 2)
        h = p / np.cos(phi) - n
        phi = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(phi) ** 2)
    h = p / np.cos(phi) - n
    return np.degrees(phi), np.degrees(lam), h


def path_length(enu: np.ndarray, use_height: bool = False) -> float:
    """Sum of straight segments between consecutive valid points."""
    points = np.asarray(enu, float)[:, : 3 if use_height else 2]
    points = points[~np.isnan(points).any(axis=1)]
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def haversine_m(lat1, lon1, lat2, lon2, radius: float = 6371008.8) -> np.ndarray:
    """Great-circle distance on a sphere of the mean Earth radius. Error up to ~0.5%."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))
