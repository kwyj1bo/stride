"""Analysis helpers on canonical samples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stride.geo import LocalFrame


def track_distance_m(samples: pd.DataFrame, smooth_samples: int = 5) -> float:
    """Horizontal distance along the GPS track.

    Summing raw fix-to-fix steps overstates distance, because every random GPS error adds
    length. A short centred moving average over positions removes most of that while
    keeping turns. Missing fixes are skipped, so a gap counts as a straight line.
    """
    lat, lon = samples["lat"].to_numpy(float), samples["lon"].to_numpy(float)
    valid = ~np.isnan(lat) & ~np.isnan(lon)
    if valid.sum() < 2:
        return 0.0
    enu = LocalFrame.at_first_fix(lat, lon).to_enu(lat[valid], lon[valid])[:, :2]
    if smooth_samples > 1:
        enu = pd.DataFrame(enu).rolling(smooth_samples, center=True, min_periods=1).mean()
        enu = enu.to_numpy()
    return float(np.linalg.norm(np.diff(enu, axis=0), axis=1).sum())


def raw_distance_m(samples: pd.DataFrame) -> float:
    """Fix-to-fix distance in the order the samples come, with no cleaning or smoothing:
    what a naive script gets from the file."""
    return track_distance_m(samples, smooth_samples=1)
