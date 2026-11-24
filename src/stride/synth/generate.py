"""Generate a labelled multi-device dataset: real files in each device's format, plus the
ground truth needed to score the pipeline (true tracks, clock errors, injected faults).

Layout:
  <out>/inputs/<device_id>/<session>.<ext>   files the pipeline ingests
  <out>/truth/manifest.json                  per-file clock error, faults and labels
  <out>/truth/<session>.parquet              true track at 1 Hz
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stride.io.fit_writer import write_fit
from stride.registry import Registry
from stride.schema import RAW_CHANNELS
from stride.synth.devices import MODELS, SimulatedDevice
from stride.synth.faults import FaultInjector
from stride.synth.truth import TruthTrack, make_truth
from stride.synth.writers import write_csv, write_gpx, write_tcx

FIRST_START = 1793500200  # 2026-11-01 06:00 IST, a whole second
EXTENSIONS = {"fit": "fit", "gpx": "gpx", "tcx": "tcx", "csv": "csv"}

# Which devices record each sport, with the chance each one is present.
LINEUPS = {
    "running": {"watch_fit": 1.0, "phone_gpx": 0.7, "app_csv": 0.5},
    "cycling": {"bike_tcx": 1.0, "phone_gpx": 0.6, "watch_fit": 0.4},
}
DURATION_MIN = {"running": (20, 60), "cycling": (30, 90)}
# Chance that a file is written in different units than its registry entry says.
MISLABEL = {"app_csv": ({"distance": "km", "speed": "km/h"}, 0.10),
            "bike_tcx": ({"speed": "km/h"}, 0.08)}  # fmt: skip


class DatasetGenerator:
    def __init__(self, seed: int, sessions: int, registry: Registry | None = None) -> None:
        self.seed = seed
        self.sessions = sessions
        self.registry = registry or Registry.load()

    def write(self, out: str | Path) -> dict:
        out = Path(out)
        (out / "truth").mkdir(parents=True, exist_ok=True)
        manifest = {"seed": self.seed, "sessions": []}
        for number in range(self.sessions):
            rng = np.random.default_rng([self.seed, number])
            manifest["sessions"].append(self._session(rng, number, out))
        (out / "truth" / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return manifest

    def _session(self, rng, number: int, out: Path) -> dict:
        sport = "running" if rng.random() < 0.6 else "cycling"
        lo, hi = DURATION_MIN[sport]
        start = FIRST_START + number * 86400 + int(rng.integers(0, 3600))
        truth = make_truth(rng, sport, rng.uniform(lo, hi) * 60, start)
        name = f"s{number:04d}"
        _truth_frame(truth).to_parquet(out / "truth" / f"{name}.parquet", index=False)
        files = []
        for device_id, chance in LINEUPS[sport].items():
            if chance >= 1.0 or rng.random() < chance:
                files.append(self._file(rng, truth, name, device_id, out))
        return {
            "session": name,
            "sport": sport,
            "start_utc": start,
            "duration_s": float(truth.seconds[-1]),
            "files": files,
        }

    def _file(self, rng, truth: TruthTrack, name: str, device_id: str, out: Path) -> dict:
        profile = self.registry.get(device_id)
        written_units, mislabeled = self._units(rng, device_id, profile)
        writer_profile = dataclasses.replace(profile, units={**profile.units, **written_units})
        device = SimulatedDevice(MODELS[device_id], writer_profile)
        start = rng.uniform(0, 60)
        recording = device.record(rng, truth, start, truth.seconds[-1] - rng.uniform(0, 60))
        faulted = FaultInjector(writer_profile, truth.origin).run(rng, recording.samples)
        path = out / "inputs" / device_id / f"{name}.{EXTENSIONS[profile.format]}"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_file(path, faulted.samples, writer_profile, truth.sport)
        labels = {}
        for row, faults in enumerate(faulted.labels):
            for fault in faults:
                labels.setdefault(fault, []).append(row)
        return {
            "path": str(path.relative_to(out / "inputs")),
            "device_id": device_id,
            "offset_s": recording.offset_s,
            "drift_ppm": recording.drift_ppm,
            "rows": len(faulted.samples),
            "labels": labels,
            "unit_mislabel": sorted(written_units) if mislabeled else [],
        }

    def _units(self, rng, device_id, profile) -> tuple[dict, bool]:
        if device_id in MISLABEL:
            units, chance = MISLABEL[device_id]
            if rng.random() < chance:
                return units, True
        return {}, False

    def _write_file(self, path, samples, profile, sport) -> None:
        samples = samples.reindex(columns=["t", *RAW_CHANNELS])  # absent channels -> NaN
        if profile.format == "fit":
            write_fit(path, samples, sport)
        elif profile.format == "gpx":
            write_gpx(path, samples, sport)
        elif profile.format == "tcx":
            write_tcx(path, samples, "Biking" if sport == "cycling" else "Running")
        else:
            write_csv(path, samples, profile, sport.capitalize())


def _truth_frame(truth: TruthTrack) -> pd.DataFrame:
    every = int(round(1 / (truth.seconds[1] - truth.seconds[0])))
    return pd.DataFrame({
        "seconds": truth.seconds[::every],
        "t_utc": truth.start_utc + truth.seconds[::every],
        "e": truth.enu[::every, 0], "n": truth.enu[::every, 1], "u": truth.enu[::every, 2],
        "distance": truth.distance[::every], "hr": truth.hr[::every],
        "lat0": truth.origin[0], "lon0": truth.origin[1], "h0": truth.origin[2],
    })  # fmt: skip
