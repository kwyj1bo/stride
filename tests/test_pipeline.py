import json

import numpy as np
import pandas as pd
import pytest
from helpers import activity_from_enu, winding_enu

from stride.check import SnapshotChecker, all_passed
from stride.cli import main
from stride.diff import diff_snapshots
from stride.io.fit_writer import write_fit
from stride.pipeline import Pipeline, PipelineConfig
from stride.registry import Registry
from stride.snapshot import SnapshotWriter
from stride.synth.writers import write_csv, write_gpx

KM_PER_MI = 1.609344


def raw_frame(activity, true_seconds):
    """Canonical test activity -> raw columns in SI units, with true distance and speed."""
    s = activity.samples
    fine = np.arange(true_seconds[0], true_seconds[-1] + 0.05, 0.1)
    step = np.linalg.norm(np.diff(winding_enu(fine)[:, :2], axis=0), axis=1)
    distance = np.interp(true_seconds, fine, np.concatenate([[0], np.cumsum(step)]))
    return pd.DataFrame(
        {
            "t": s["t"],
            "lat": s["lat"],
            "lon": s["lon"],
            "ele": s["ele_m"],
            "hr": 140.0,
            "cadence": 170.0,
            "speed": np.gradient(distance, true_seconds),
            "distance": distance,
        }
    )


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "in"
    for device in ("watch_fit", "phone_gpx", "app_csv", "unknown_device"):
        (root / device).mkdir(parents=True)
    true_a = np.arange(0, 1800, 1.0)
    watch = activity_from_enu(true_a, true_a, winding_enu(true_a), "watch_fit", 2.0, seed=1)
    write_fit(root / "watch_fit" / "run.fit", raw_frame(watch, true_a))
    true_b = np.arange(2.0, 1790, 1.0)
    phone = activity_from_enu(true_b, true_b + 4.0, winding_enu(true_b), "phone_gpx", 2.0, seed=2)
    raw = raw_frame(phone, true_b)
    raw.loc[300, "lat"] += 0.01  # a 1.1 km GPS jump
    write_gpx(root / "phone_gpx" / "run.gpx", raw)
    true_c = np.arange(0, 1780, 2.0)
    app = activity_from_enu(true_c, true_c, winding_enu(true_c), "app_csv", 2.0, seed=3)
    imperial = raw_frame(app, true_c)
    imperial["distance"] /= 1609.344
    imperial["speed"] /= 0.44704
    imperial["ele"] /= 0.3048
    write_csv(root / "app_csv" / "run.csv", imperial, Registry.load().get("app_csv"))
    (root / "watch_fit" / "broken.fit").write_bytes(b"not a fit file at all")
    (root / "unknown_device" / "ignored.gpx").write_text("<gpx/>")
    return root


def test_pipeline_end_to_end(inputs):
    result = Pipeline(Registry.load()).run(inputs)
    assert sorted(result.failures) == ["watch_fit/broken.fit"]
    assert len(result.recordings) == 3 and len(result.sessions) == 1
    phone = next(r for r in result.recordings if r.activity.device_id == "phone_gpx")
    assert phone.report.count("gps_jump") == 1
    app = next(r for r in result.recordings if r.activity.device_id == "app_csv")
    assert not app.report.unit_findings  # imperial units read correctly
    session = result.sessions[0]
    assert session.reference == "watch_fit"
    assert phone.sync.offset_s == pytest.approx(4.0, abs=0.15)
    assert abs(app.sync.offset_s) < 0.6  # CSV keeps whole seconds only


def test_snapshot_is_reproducible_checkable_and_diffable(inputs, tmp_path):
    config = PipelineConfig()
    first = SnapshotWriter(tmp_path / "out").write(
        Pipeline(Registry.load()).run(inputs), config, inputs
    )
    second = SnapshotWriter(tmp_path / "out").write(
        Pipeline(Registry.load()).run(inputs), config, inputs
    )
    assert first == second  # same inputs and settings -> same snapshot id
    checks = SnapshotChecker(first, min_validity=0.5).run()
    assert [c.name for c in checks if not c.passed] == ["quality_gate"]  # one file failed to parse
    (inputs / "watch_fit" / "broken.fit").unlink()
    third = SnapshotWriter(tmp_path / "out").write(
        Pipeline(Registry.load()).run(inputs), config, inputs
    )
    assert all_passed(SnapshotChecker(third, min_validity=0.5).run())
    diff = diff_snapshots(first, third)
    assert diff["inputs_removed"] == ["watch_fit/broken.fit"]
    assert diff["identical_data"]
    index = json.loads((tmp_path / "out" / "snapshots" / "index.json").read_text())
    assert len(index) == 2


def test_tampered_snapshot_fails_its_hash_check(inputs, tmp_path):
    folder = SnapshotWriter(tmp_path).write(
        Pipeline(Registry.load()).run(inputs), PipelineConfig(), inputs
    )
    table = pd.read_parquet(folder / "sessions.parquet")
    table.loc[0, "hr_bpm"] = 99
    table.to_parquet(folder / "sessions.parquet", index=False)
    failed = [c.name for c in SnapshotChecker(folder).run() if not c.passed]
    assert "sessions_content_hash" in failed


def test_cli_ingest_check_show(inputs, tmp_path, capsys):
    (inputs / "watch_fit" / "broken.fit").unlink()
    assert main(["ingest", str(inputs), "--out", str(tmp_path / "o")]) == 0
    snapshot = next((tmp_path / "o" / "snapshots").glob("*/"))
    assert main(["check", str(snapshot), "--min-validity", "0.5"]) == 0
    assert main(["show", str(snapshot)]) == 0
    out = capsys.readouterr().out
    assert "phone_gpx/run.gpx" in out and "sync session-0001" in out
