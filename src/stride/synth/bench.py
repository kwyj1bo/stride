"""Benchmark: generate a labelled dataset, run the pipeline, score it against the truth.

Every number in docs/results.md comes from this module, with a fixed seed.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stride import io
from stride.metrics import raw_distance_m, track_distance_m
from stride.normalize import Normalizer
from stride.pipeline import Pipeline, PipelineConfig, PipelineResult
from stride.registry import Registry
from stride.resample import _seconds
from stride.snapshot import SnapshotWriter
from stride.synth.faults import FAULT_RATES
from stride.synth.generate import DatasetGenerator


@dataclass
class BenchResult:
    dataset: dict = field(default_factory=dict)
    detection: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    sync: dict = field(default_factory=dict)
    distance: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)
    reproducible: bool = False


class Benchmark:
    def __init__(self, seed: int, sessions: int, work: str | Path) -> None:
        self.seed = seed
        self.sessions = sessions
        self.work = Path(work)
        self.registry = Registry.load()

    def run(self) -> BenchResult:
        truth = DatasetGenerator(self.seed, self.sessions, self.registry).write(self.work)
        inputs = self.work / "inputs"
        pipeline = Pipeline(self.registry, PipelineConfig())
        started = time.perf_counter()
        result = pipeline.run(inputs)
        elapsed = time.perf_counter() - started
        files = {f["path"]: f | {"session": s} for s in truth["sessions"] for f in s["files"]}
        bench = BenchResult()
        bench.dataset = self._dataset(truth, result)
        bench.detection = self._detection(result, files)
        bench.units = self._units(result, files)
        bench.sync = self._sync(result, files)
        bench.distance = self._distance(result, files, inputs)
        rows = sum(f["rows"] for f in files.values())
        bench.performance = {
            "pipeline_seconds": round(elapsed, 2),
            "files_per_second": round(len(files) / elapsed, 1),
            "samples_per_second": round(rows / elapsed),
            "machine": f"{platform.machine()} {platform.system()}, "
            f"Python {platform.python_version()}",
        }
        bench.reproducible = self._reproducible(pipeline, result, inputs)
        return bench

    def _dataset(self, truth, result: PipelineResult) -> dict:
        files = [f for s in truth["sessions"] for f in s["files"]]
        return {
            "seed": self.seed,
            "sessions": len(truth["sessions"]),
            "files": len(files),
            "files_by_device": pd.Series([f["device_id"] for f in files]).value_counts().to_dict(),
            "samples": sum(f["rows"] for f in files),
            "hours": round(sum(s["duration_s"] for s in truth["sessions"]) / 3600, 1),
            "failed_files": len(result.failures),
        }

    def _detection(self, result: PipelineResult, files: dict) -> dict:
        """Sample-level precision and recall per rule. Files written in the wrong units are
        scored separately (their speeds are legitimately flagged as impossible)."""
        counts = {rule: {"tp": 0, "fp": 0, "fn": 0} for rule in FAULT_RATES}
        for recording in result.recordings:
            truth = files[recording.path]
            if truth["unit_mislabel"]:
                continue
            for rule, c in counts.items():
                predicted = set(np.flatnonzero(recording.report.flags[rule]).tolist())
                actual = set(truth["labels"].get(rule, []))
                c["tp"] += len(predicted & actual)
                c["fp"] += len(predicted - actual)
                c["fn"] += len(actual - predicted)
        for c in counts.values():
            c.update(_precision_recall(c["tp"], c["fp"], c["fn"]))
        total = {k: sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn")}
        return {"by_rule": counts, "all": total | _precision_recall(**total)}

    def _units(self, result: PipelineResult, files: dict) -> dict:
        tp = fp = fn = 0
        for recording in result.recordings:
            flagged = bool(recording.report.unit_findings)
            actual = bool(files[recording.path]["unit_mislabel"])
            tp += flagged and actual
            fp += flagged and not actual
            fn += actual and not flagged
        return {"tp": tp, "fp": fp, "fn": fn} | _precision_recall(tp, fp, fn)

    def _sync(self, result: PipelineResult, files: dict) -> dict:
        offset_errors, drift_errors, failed = [], [], 0
        for session in result.sessions:
            ref = next(m for m in session.members if m.activity.device_id == session.reference)
            ref_truth = files[ref.path]
            start = ref_truth["session"]["start_utc"]
            for member in session.members:
                if member is ref:
                    continue
                if member.sync is None:
                    failed += 1
                    continue
                t = files[member.path]
                tau = member.sync.t0 - start - ref_truth["offset_s"]
                d_drift = t["drift_ppm"] - ref_truth["drift_ppm"]
                true_offset = t["offset_s"] - ref_truth["offset_s"] + d_drift * 1e-6 * tau
                offset_errors.append(member.sync.offset_s - true_offset)
                drift_errors.append(member.sync.drift_ppm - d_drift)
        errors = np.abs(np.array(offset_errors))
        return {
            "pairs": len(offset_errors),
            "failed": failed,
            "offset_abs_error_s": _quantiles(errors),
            "within_0_25_s": round(float((errors <= 0.25).mean()), 4) if len(errors) else None,
            "within_0_5_s": round(float((errors <= 0.5).mean()), 4) if len(errors) else None,
            "drift_abs_error_ppm": _quantiles(np.abs(np.array(drift_errors))),
        }

    def _distance(self, result: PipelineResult, files: dict, inputs: Path) -> dict:
        """Relative distance error per session: fused pipeline output vs a naive sum over
        the raw reference file, both against the true distance over the same span."""
        pipeline_err, naive_err = [], []
        for session in result.sessions:
            ref = next(m for m in session.members if m.activity.device_id == session.reference)
            truth_file = files[ref.path]
            track = pd.read_parquet(
                self.work / "truth" / f"{truth_file['session']['session']}.parquet"
            )
            fused = session.fused.activity.samples
            pipeline_err.append(_rel(track_distance_m(fused), _true_distance(track, fused)))
            profile = self.registry.get(ref.activity.device_id)
            raw = Normalizer(profile).run(io.read(inputs / ref.path, profile), "raw").samples
            naive_err.append(_rel(raw_distance_m(raw), _true_distance(track, raw)))
        return {
            "sessions": len(pipeline_err),
            "pipeline_abs_error_pct": _quantiles(np.abs(pipeline_err) * 100),
            "naive_abs_error_pct": _quantiles(np.abs(naive_err) * 100),
        }

    def _reproducible(self, pipeline: Pipeline, first: PipelineResult, inputs: Path) -> bool:
        out = self.work / "snapshots-check"
        a = SnapshotWriter(out).write(first, pipeline.config, inputs)
        b = SnapshotWriter(out).write(pipeline.run(inputs), pipeline.config, inputs)
        return a.name == b.name


def _true_distance(track: pd.DataFrame, samples: pd.DataFrame) -> float:
    """True distance between the first and last sample times. Device clocks are within
    seconds of true time, which moves both ends equally and barely changes the span."""
    seconds = _seconds(samples["t"].dropna())
    lo, hi = seconds.min(), seconds.max()
    d = np.interp([lo, hi], track["t_utc"], track["distance"])
    return float(d[1] - d[0])


def _rel(measured: float, true: float) -> float:
    return (measured - true) / true if true > 0 else np.nan


def _precision_recall(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if not len(values):
        return {}
    return {
        "median": round(float(np.median(values)), 4),
        "mean": round(float(values.mean()), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "max": round(float(values.max()), 4),
    }


def save(result: BenchResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.__dict__, indent=2, default=float))
