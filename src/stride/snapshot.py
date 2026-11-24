"""Versioned dataset snapshots.

A snapshot is a folder with the data (Parquet), the reports and a manifest. Its id is a
hash of the data content, the pipeline config and the input file hashes, so the same
inputs and settings always give the same id, and any change gives a new one.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stride import __version__
from stride.pipeline import PipelineConfig, PipelineResult, recordings_frame, sessions_frame
from stride.schema import SCHEMA_VERSION

TABLES = ("recordings", "sessions")


class SnapshotWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, result: PipelineResult, config: PipelineConfig, input_dir: Path) -> Path:
        tables = {"recordings": recordings_frame(result), "sessions": sessions_frame(result)}
        manifest = build_manifest(result, config, input_dir, tables)
        folder = self.root / "snapshots" / manifest["snapshot_id"]
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_parquet(folder / f"{name}.parquet", index=False)
        (folder / "reports.json").write_text(json.dumps(build_reports(result), indent=2))
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._record(manifest)
        return folder

    def _record(self, manifest: dict) -> None:
        """Append to the snapshot index, newest last, without duplicating a rerun."""
        index_path = self.root / "snapshots" / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else []
        if not any(entry["snapshot_id"] == manifest["snapshot_id"] for entry in index):
            index.append({k: manifest[k] for k in ("snapshot_id", "created_at", "counts")})
        index_path.write_text(json.dumps(index, indent=2))


def build_manifest(
    result: PipelineResult, config: PipelineConfig, input_dir: Path, tables: dict[str, pd.DataFrame]
) -> dict:
    inputs = [
        {"path": str(p.relative_to(input_dir)), "device_id": d, "sha256": file_sha256(p)}
        for p, d in result.inputs
    ]
    content = {name: table_hash(frame) for name, frame in tables.items()}
    identity = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": __version__,
        "config": config.as_dict(),
        "inputs": inputs,
        "content": content,
    }
    snapshot_id = sha256_json(identity)[:12]
    return {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **identity,
        "counts": {
            "files": len(result.inputs),
            "failed": len(result.failures),
            "recordings": len(result.recordings),
            "sessions": len(result.sessions),
            **{f"{name}_rows": len(frame) for name, frame in tables.items()},
        },
        "quality": dataset_quality(result),
    }


def build_reports(result: PipelineResult) -> dict:
    return {
        "recordings": [
            r.report.summary() | {"path": r.path, "notes": r.activity.notes}
            for r in result.recordings
        ],
        "sessions": [
            {
                "session_id": s.session_id,
                "reference": s.reference,
                "members": [m.activity.activity_id for m in s.members],
                "sync": {m.activity.activity_id: vars(m.sync) for m in s.members if m.sync},
                "provenance": s.fused.provenance,
            }
            for s in result.sessions
        ],
        "failures": result.failures,
    }


def dataset_quality(result: PipelineResult) -> dict:
    reports = [r.report for r in result.recordings]
    issues: dict[str, int] = {}
    for report in reports:
        for rule in report.flags.columns:
            issues[rule] = issues.get(rule, 0) + report.count(rule)
    n = max(len(reports), 1)
    return {
        "recordings_passed": sum(r.passed for r in reports),
        "recordings_with_unit_findings": sum(bool(r.unit_findings) for r in reports),
        "mean_validity": round(sum(r.quality["validity"] for r in reports) / n, 6),
        "flagged_samples_by_rule": issues,
    }


def table_hash(frame: pd.DataFrame) -> str:
    """Hash of the table's values and column names, independent of Parquet encoding."""
    digest = hashlib.sha256(",".join(map(str, frame.columns)).encode())
    if len(frame):
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
