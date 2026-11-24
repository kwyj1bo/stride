"""Validation: run every rule on a normalized activity and collect the findings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stride.schema import CHANNELS, Activity
from stride.validation.rules import SAMPLE_RULES, Context, SampleRule
from stride.validation.units_check import UnitCheck, UnitFinding


@dataclass
class ValidationReport:
    activity_id: str
    device_id: str
    n_samples: int
    flags: pd.DataFrame  # one boolean column per sample rule, rows in device order
    unit_findings: list[UnitFinding] = field(default_factory=list)
    quality: dict[str, float] = field(default_factory=dict)

    def count(self, rule: str) -> int:
        return int(self.flags[rule].sum())

    @property
    def error_mask(self) -> np.ndarray:
        """Samples flagged by any error-severity rule."""
        errors = [r.name for r in SAMPLE_RULES if r.severity == "error"]
        return self.flags[errors].any(axis=1).to_numpy()

    @property
    def passed(self) -> bool:
        return not self.unit_findings and not self.error_mask.any()

    def summary(self) -> dict:
        return {
            "activity_id": self.activity_id,
            "device_id": self.device_id,
            "n_samples": self.n_samples,
            "issues": {name: self.count(name) for name in self.flags.columns if self.count(name)},
            "unit_findings": [vars(f) for f in self.unit_findings],
            "quality": self.quality,
            "passed": self.passed,
        }


class Validator:
    def __init__(
        self,
        nominal_hz: float,
        units: dict[str, str] | None = None,
        rules: tuple[SampleRule, ...] = SAMPLE_RULES,
    ) -> None:
        self.nominal_hz = nominal_hz
        self.units = units or {}
        self.rules = rules

    def run(self, activity: Activity) -> ValidationReport:
        ctx = Context.build(activity.samples, activity.sport, self.nominal_hz)
        flags = pd.DataFrame({rule.name: rule.check(ctx) for rule in self.rules})
        report = ValidationReport(
            activity.activity_id,
            activity.device_id,
            len(activity.samples),
            flags,
            UnitCheck(self.units).check(ctx, exclude=self._bad_positions(flags)),
        )
        report.quality = quality_metrics(activity.samples, ctx.seconds, report, self.nominal_hz)
        return report

    @staticmethod
    def _bad_positions(flags: pd.DataFrame) -> np.ndarray | None:
        return flags["gps_jump"].to_numpy() if "gps_jump" in flags else None


def quality_metrics(
    samples: pd.DataFrame, seconds: np.ndarray, report: ValidationReport, nominal_hz: float
) -> dict[str, float]:
    """Completeness: share of samples with each channel present. Validity: share with no
    error flag. Rate ratio: achieved sample rate over the device's nominal one."""
    n = max(len(samples), 1)
    metrics = {f"completeness_{c}": float(samples[c].notna().sum() / n) for c in CHANNELS}
    metrics["validity"] = float(1 - report.error_mask.mean()) if len(samples) else 0.0
    span = float(seconds.max() - seconds.min()) if len(seconds) > 1 else 0.0
    unique = len(np.unique(seconds))
    metrics["rate_ratio"] = float((unique - 1) / span / nominal_hz) if span else 0.0
    return metrics
