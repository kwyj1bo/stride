"""End-to-end pipeline: files on disk -> cleaned, synced, fused sessions.

Input layout is one folder per registered device:  <input>/<device_id>/<any file>.
A file that fails to parse is recorded as a failure and the run continues.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from stride import io
from stride.clean import Cleaner
from stride.fuse import FusedActivity, Fuser, group_sessions
from stride.normalize import Normalizer
from stride.registry import Registry
from stride.resample import Resampler
from stride.schema import Activity
from stride.sync import ClockSync, SyncResult
from stride.validation import ValidationReport, Validator


@dataclass(frozen=True)
class PipelineConfig:
    rate_hz: float = 1.0
    max_gap_s: float = 10.0
    max_clock_lag_s: float = 300.0
    session_tolerance_s: float = 300.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Recording:
    path: str
    activity: Activity  # cleaned; later synced and resampled
    report: ValidationReport
    sync: SyncResult | None = None


@dataclass
class Session:
    session_id: str
    reference: str
    fused: FusedActivity
    members: list[Recording]


@dataclass
class PipelineResult:
    recordings: list[Recording] = field(default_factory=list)
    sessions: list[Session] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)  # path -> error
    inputs: list[tuple[Path, str]] = field(default_factory=list)  # (path, device_id)


class Pipeline:
    def __init__(self, registry: Registry, config: PipelineConfig | None = None) -> None:
        self.registry = registry
        self.config = config or PipelineConfig()
        self.resampler = Resampler(self.config.rate_hz, self.config.max_gap_s)
        self.sync = ClockSync(max_lag_s=self.config.max_clock_lag_s)

    def run(self, input_dir: str | Path) -> PipelineResult:
        result = PipelineResult(inputs=self.discover(Path(input_dir)))
        for path, device_id in result.inputs:
            try:
                result.recordings.append(self.prepare(path, device_id, Path(input_dir)))
            except Exception as error:  # noqa: BLE001 - one bad file must not stop the run
                result.failures[str(path.relative_to(input_dir))] = (
                    f"{type(error).__name__}: {error}"
                )
        by_activity = {r.activity.activity_id: r for r in result.recordings}
        groups = group_sessions(
            [r.activity for r in result.recordings], self.config.session_tolerance_s
        )
        for number, group in enumerate(groups, start=1):
            members = [by_activity[a.activity_id] for a in group]
            result.sessions.append(self.fuse(f"session-{number:04d}", members))
        return result

    def discover(self, input_dir: Path) -> list[tuple[Path, str]]:
        """Every file under a registered device folder, in a stable order."""
        found = []
        for folder in sorted(p for p in input_dir.iterdir() if p.is_dir()):
            if folder.name not in self.registry:
                continue
            found += [(f, folder.name) for f in sorted(folder.iterdir()) if f.is_file()]
        return found

    def prepare(self, path: Path, device_id: str, root: Path) -> Recording:
        """Parse, normalize, validate and clean one file."""
        profile = self.registry.get(device_id)
        raw = io.read(path, profile)
        activity_id = f"{device_id}/{path.stem}"
        normalized = Normalizer(profile).run(raw, activity_id)
        report = Validator(profile.nominal_hz, profile.units).run(normalized)
        cleaned = Cleaner().run(normalized, report)
        return Recording(str(path.relative_to(root)), cleaned, report)

    def fuse(self, session_id: str, members: list[Recording]) -> Session:
        """Sync every member's clock to the reference, resample, then fuse."""
        # Most trusted positions first; on a tie, the denser recording.
        reference = max(members, key=lambda r: (_position_trust(r.report), r.report.n_samples))
        for member in members:
            if member is not reference:
                member.sync = self.sync.estimate(reference.activity, member.activity)
                if member.sync is not None:
                    member.activity = self.sync.apply(member.activity, member.sync)
            member.activity = self.resampler.run(member.activity)
        quality = {m.activity.device_id: m.report.quality for m in members}
        fused = Fuser().run(session_id, [m.activity for m in members], quality)
        return Session(session_id, reference.activity.device_id, fused, members)


def recordings_frame(result: PipelineResult) -> pd.DataFrame:
    frames = [r.activity.to_frame() for r in result.recordings]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sessions_frame(result: PipelineResult) -> pd.DataFrame:
    frames = [s.fused.activity.to_frame() for s in result.sessions]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).rename(columns={"activity_id": "session_id"})


def _position_trust(report: ValidationReport) -> float:
    return report.quality.get("completeness_lat", 0.0) * report.quality.get("validity", 0.0)
