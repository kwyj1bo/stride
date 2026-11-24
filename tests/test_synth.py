import hashlib

import numpy as np
import pandas as pd
import pytest

from stride.registry import Registry
from stride.synth.bench import Benchmark
from stride.synth.devices import MODELS, SimulatedDevice
from stride.synth.faults import FaultInjector
from stride.synth.generate import DatasetGenerator
from stride.synth.truth import make_truth


def digest(folder):
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_same_seed_writes_identical_files(tmp_path):
    DatasetGenerator(5, 3).write(tmp_path / "a")
    DatasetGenerator(5, 3).write(tmp_path / "b")
    assert digest(tmp_path / "a" / "inputs") == digest(tmp_path / "b" / "inputs")
    DatasetGenerator(6, 3).write(tmp_path / "c")
    assert digest(tmp_path / "a" / "inputs") != digest(tmp_path / "c" / "inputs")


def test_device_samples_on_its_own_clock_ticks():
    rng = np.random.default_rng(0)
    truth = make_truth(rng, "running", 600, 1793500200)
    device = SimulatedDevice(MODELS["app_csv"], Registry.load().get("app_csv"))
    rec = device.record(rng, truth, 10, 590)
    clock = (rec.samples["t"] - pd.Timestamp(1793500200, unit="s", tz="UTC")).dt.total_seconds()
    assert np.allclose(clock % 2, 0)  # 0.5 Hz device ticks on even clock seconds
    expected = rec.true_seconds + rec.offset_s + rec.drift_ppm * 1e-6 * rec.true_seconds
    assert np.allclose(clock, expected, atol=1e-5)
    # imperial units, as the registry says
    assert rec.samples["distance"].iloc[-1] == pytest.approx(
        (
            truth.at(rec.true_seconds[[-1]])["distance"][0]
            - truth.at(rec.true_seconds[[0]])["distance"][0]
        )
        / 1609.344,
        rel=0.03,
    )


def test_labels_follow_rows_through_cuts_inserts_and_swaps():
    rng = np.random.default_rng(1)
    truth = make_truth(rng, "running", 1200, 1793500200)
    profile = Registry.load().get("watch_fit")
    rec = SimulatedDevice(MODELS["watch_fit"], profile).record(rng, truth, 0, 1200)
    for seed in range(20):
        out = FaultInjector(profile, truth.origin).run(np.random.default_rng(seed), rec.samples)
        t = out.samples["t"].reset_index(drop=True)
        assert len(out.labels) == len(out.samples)
        for i, faults in enumerate(out.labels):
            if "duplicate_timestamp" in faults:
                assert t[i] == t[i - 1]
            if "non_monotonic" in faults:
                assert t[i] < t[i - 1]
            if "timestamp_gap" in faults:
                assert (t[i] - t[i - 1]).total_seconds() > 1


def test_small_benchmark_runs_and_is_reproducible(tmp_path):
    result = Benchmark(11, 3, tmp_path).run()
    assert result.reproducible
    assert result.dataset["failed_files"] == 0
    assert result.detection["all"]["precision"] > 0.9
    assert result.distance["pipeline_abs_error_pct"]["median"] < 2.0
