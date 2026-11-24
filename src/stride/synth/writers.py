"""Writers for GPX, TCX and CSV, used to produce test and benchmark files in each device's
own format. (FIT has its own encoder in stride.io.fit_writer.)

Input samples use raw channel names (lat, lon, ele, hr, cadence, speed, distance) in the
device's units, with "t" in UTC.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

from stride.registry import DeviceProfile

_TPE_OPEN = "<gpxtpx:TrackPointExtension>"
_TPE_CLOSE = "</gpxtpx:TrackPointExtension>"


def write_gpx(path: str | Path, samples: pd.DataFrame, sport: str = "running") -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="stride-synth" xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">',
        f"<trk><type>{escape(sport)}</type><trkseg>",
    ]
    for row in samples.itertuples(index=False):
        if pd.isna(row.lat) or pd.isna(row.lon):
            continue  # GPX cannot store a point without a position
        inner = _tag("ele", row.ele, ".2f") + f"<time>{_iso(row.t)}</time>"
        ext = _tag("gpxtpx:hr", row.hr, ".0f") + _tag("gpxtpx:cad", row.cadence, ".0f")
        if ext:
            inner += f"<extensions>{_TPE_OPEN}{ext}{_TPE_CLOSE}</extensions>"
        lines.append(f'<trkpt lat="{row.lat:.7f}" lon="{row.lon:.7f}">{inner}</trkpt>')
    lines.append("</trkseg></trk></gpx>")
    Path(path).write_text("\n".join(lines))


def write_tcx(path: str | Path, samples: pd.DataFrame, sport: str = "Biking") -> None:
    start = _iso(samples["t"].iloc[0])
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<TrainingCenterDatabase "
        'xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2" '
        'xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2">',
        f'<Activities><Activity Sport="{escape(sport)}"><Id>{start}</Id>',
        f'<Lap StartTime="{start}"><Track>',
    ]
    for row in samples.itertuples(index=False):
        body = f"<Time>{_iso(row.t)}</Time>"
        if not (pd.isna(row.lat) or pd.isna(row.lon)):
            body += (
                f"<Position><LatitudeDegrees>{row.lat:.7f}</LatitudeDegrees>"
                f"<LongitudeDegrees>{row.lon:.7f}</LongitudeDegrees></Position>"
            )
        body += _tag("AltitudeMeters", row.ele, ".2f") + _tag("DistanceMeters", row.distance, ".2f")
        if not pd.isna(row.hr):
            body += f"<HeartRateBpm><Value>{row.hr:.0f}</Value></HeartRateBpm>"
        body += _tag("Cadence", row.cadence, ".0f")
        if not pd.isna(row.speed):
            speed = f"<ns3:Speed>{row.speed:.3f}</ns3:Speed>"
            body += f"<Extensions><ns3:TPX>{speed}</ns3:TPX></Extensions>"
        lines.append(f"<Trackpoint>{body}</Trackpoint>")
    lines.append("</Track></Lap></Activity></Activities></TrainingCenterDatabase>")
    Path(path).write_text("\n".join(lines))


def write_csv(
    path: str | Path, samples: pd.DataFrame, profile: DeviceProfile, sport: str = "Running"
) -> None:
    """CSV with the profile's headers. Times are written in the profile's zone with no
    offset, the way many apps export them."""
    local = pd.to_datetime(samples["t"], utc=True).dt.tz_convert(profile.timezone)
    fmt = profile.csv_time_format or "%Y-%m-%dT%H:%M:%S"
    table = pd.DataFrame({profile.csv_columns["t"]: local.dt.strftime(fmt)})
    for channel, header in profile.csv_columns.items():
        if channel != "t":
            table[header] = (
                samples[channel].round(7)
                if channel in ("lat", "lon")
                else samples[channel].round(4)
            )
    table["Sport"] = sport
    table.to_csv(path, index=False)


def _tag(name: str, value, fmt: str) -> str:
    return "" if pd.isna(value) else f"<{name}>{value:{fmt}}</{name}>"


def _iso(t) -> str:
    """UTC with millisecond precision; whole seconds drop the fraction, like most exports."""
    t = pd.Timestamp(t).tz_convert("UTC")
    text = t.strftime("%Y-%m-%dT%H:%M:%S")
    if t.microsecond:
        text += f".{t.microsecond // 1000:03d}"
    return text + "Z"
