"""Ground-truth workouts: what really happened, sampled at 10 Hz.

Speed and heading follow mean-reverting random processes (Ornstein-Uhlenbeck), so routes
wind and pace varies like a real session. Rides include traffic stops. Heart rate follows
effort with a lag. Everything is seeded, so a seed always gives the same workout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT = 0.1  # seconds

# (lat, lon, height): Mumbai, Pune, Bengaluru, Delhi, Chennai, Kolkata
ORIGINS = (
    (19.0760, 72.8777, 14.0),
    (18.5204, 73.8567, 560.0),
    (12.9716, 77.5946, 920.0),
    (28.6139, 77.2090, 216.0),
    (13.0827, 80.2707, 6.0),
    (22.5726, 88.3639, 9.0),
)

SPORTS = {
    # mean speed range m/s, speed noise sd, cadence range
    "running": ((2.4, 3.8), 0.25, (160, 182)),
    "cycling": ((5.5, 9.0), 0.9, (78, 96)),
}


@dataclass
class TruthTrack:
    sport: str
    origin: tuple[float, float, float]
    start_utc: float  # UNIX seconds
    seconds: np.ndarray  # since start, every DT
    enu: np.ndarray  # (n, 3) metres around origin
    speed: np.ndarray
    distance: np.ndarray  # horizontal, cumulative
    hr: np.ndarray
    cadence: np.ndarray

    def at(self, seconds: np.ndarray) -> dict[str, np.ndarray]:
        """Every channel linearly interpolated at the given true times."""
        out = {
            name: np.interp(seconds, self.seconds, getattr(self, name))
            for name in ("speed", "distance", "hr", "cadence")
        }
        for axis, name in enumerate(("e", "n", "u")):
            out[name] = np.interp(seconds, self.seconds, self.enu[:, axis])
        return out


def make_truth(
    rng: np.random.Generator, sport: str, duration_s: float, start_utc: float
) -> TruthTrack:
    (lo, hi), speed_sd, (cad_lo, cad_hi) = SPORTS[sport]
    n = int(duration_s / DT) + 1
    seconds = np.arange(n) * DT
    speed = _speed(rng, n, rng.uniform(lo, hi), speed_sd, sport)
    heading = rng.uniform(0, 2 * np.pi) + np.cumsum(_ou(rng, n, 0.0, 0.025, 25.0)) * DT
    step = speed * DT
    east = np.concatenate([[0.0], np.cumsum(step[:-1] * np.cos(heading[:-1]))])
    north = np.concatenate([[0.0], np.cumsum(step[:-1] * np.sin(heading[:-1]))])
    up = _terrain(rng, seconds)
    distance = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(east), np.diff(north)))])
    hr = _heart_rate(rng, speed, seconds)
    cadence = np.where(speed > 0.3, rng.uniform(cad_lo, cad_hi) + _ou(rng, n, 0, 2.0, 30.0), 0)
    origin = ORIGINS[rng.integers(len(ORIGINS))]
    return TruthTrack(
        sport,
        origin,
        start_utc,
        seconds,
        np.stack([east, north, up], 1),
        speed,
        distance,
        hr,
        cadence,
    )


def _ou(rng, n, mean, sd, tau) -> np.ndarray:
    """Ornstein-Uhlenbeck process with stationary sd and correlation time tau."""
    a = np.exp(-DT / tau)
    noise = rng.normal(0, sd * np.sqrt(1 - a * a), n)
    out = np.empty(n)
    out[0] = mean + rng.normal(0, sd)
    for i in range(1, n):
        out[i] = mean + a * (out[i - 1] - mean) + noise[i]
    return out


def _speed(rng, n, mean, sd, sport) -> np.ndarray:
    speed = np.clip(_ou(rng, n, mean, sd, 60.0), 0.3 * mean, None)
    if sport == "cycling":  # traffic stops with 8 s slow-down and speed-up ramps
        for _ in range(rng.integers(0, 5)):
            start = rng.integers(int(60 / DT), max(int(60 / DT) + 1, n - int(120 / DT)))
            stop = int(rng.uniform(15, 60) / DT)
            ramp = int(8 / DT)
            shape = np.concatenate(
                [np.linspace(1, 0, ramp), np.zeros(stop), np.linspace(0, 1, ramp)]
            )
            end = min(start + len(shape), n)
            speed[start:end] *= shape[: end - start]
    return speed


def _terrain(rng, seconds) -> np.ndarray:
    up = np.zeros_like(seconds)
    for _ in range(3):
        amplitude, period = rng.uniform(2, 20), rng.uniform(300, 1800)
        up += amplitude * np.sin(2 * np.pi * seconds / period + rng.uniform(0, 2 * np.pi))
    return up


def _heart_rate(rng, speed, seconds) -> np.ndarray:
    """First-order response (30 s) to effort, plus slow cardiac drift over the session."""
    effort = speed / speed.mean()
    target = (
        rng.uniform(95, 110) + rng.uniform(35, 55) * effort + seconds / 60 * rng.uniform(0, 0.3)
    )
    hr = np.empty_like(target)
    hr[0] = target[0] - 20
    a = np.exp(-DT / 30.0)
    for i in range(1, len(hr)):
        hr[i] = a * hr[i - 1] + (1 - a) * target[i]
    return np.clip(hr, 60, 200)
