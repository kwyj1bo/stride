"""Render benchmark results as Markdown."""

from __future__ import annotations

from stride.synth.bench import BenchResult


def to_markdown(r: BenchResult, command: str) -> str:
    d, s, dist, perf = r.dataset, r.sync, r.distance, r.performance
    lines = [
        "# Benchmark results",
        "",
        f"Reproduce with `{command}`. The dataset is synthetic: ground truth is known exactly,",
        "so every number below is checked against it. See the README for how it is generated.",
        "",
        "## Dataset",
        "",
        f"- {d['sessions']} sessions, {d['hours']} hours, {d['files']} files, "
        f"{d['samples']:,} samples",
        "- Files by device: " + ", ".join(f"{k} {v}" for k, v in d["files_by_device"].items()),
        f"- Files that failed to parse: {d['failed_files']}",
        "",
        "## Fault detection (sample level)",
        "",
        "Files written in the wrong units are scored in the unit-check section instead.",
        "",
        "| Rule | Injected | Found | False alarms | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rule, c in r.detection["by_rule"].items():
        lines.append(
            f"| {rule} | {c['tp'] + c['fn']} | {c['tp']} | {c['fp']} | "
            f"{c['precision']:.3f} | {c['recall']:.3f} |"
        )
    a = r.detection["all"]
    lines += [
        f"| **all** | {a['tp'] + a['fn']} | {a['tp']} | {a['fp']} | "
        f"{a['precision']:.3f} | {a['recall']:.3f} |",
        "",
        "Missed faults are mostly ones injected below a rule's threshold on purpose (for",
        "example a 40 m GPS jump on a bike computer that records every 8 s).",
        "",
        "## Unit check (file level)",
        "",
        "- Files written in different units than their registry entry: "
        f"{r.units['tp'] + r.units['fn']}",
        f"- Flagged: {r.units['tp']}, missed: {r.units['fn']}, false alarms: {r.units['fp']}",
        f"- Precision {r.units['precision']:.3f}, recall {r.units['recall']:.3f}",
        "",
        "## Clock sync",
        "",
        f"- Device pairs synced: {s['pairs']} (failed: {s['failed']})",
        f"- Offset error: median {s['offset_abs_error_s'].get('median')} s, "
        f"p95 {s['offset_abs_error_s'].get('p95')} s, max {s['offset_abs_error_s'].get('max')} s",
        f"- Within 0.25 s: {s['within_0_25_s']:.1%}; within 0.5 s: {s['within_0_5_s']:.1%}",
        f"- Drift error: median {s['drift_abs_error_ppm'].get('median')} ppm, "
        f"p95 {s['drift_abs_error_ppm'].get('p95')} ppm",
        "",
        "## Distance",
        "",
        "Absolute error of session distance against the true distance over the same span.",
        "",
        "| Method | Median | Mean | p95 |",
        "|---|---:|---:|---:|",
        _row("Pipeline (cleaned, fused, 5 s smoothing)", dist["pipeline_abs_error_pct"]),
        _row("Naive (raw fixes summed, no cleaning)", dist["naive_abs_error_pct"]),
        "",
        "## Performance and reproducibility",
        "",
        f"- Pipeline time: {perf['pipeline_seconds']} s ({perf['files_per_second']} files/s, "
        f"{perf['samples_per_second']:,} samples/s) on one core ({perf.get('machine', 'n/a')})",
        f"- Two runs gave the same snapshot id: {'yes' if r.reproducible else 'no'}",
        "",
    ]
    return "\n".join(lines)


def _row(name: str, q: dict) -> str:
    return f"| {name} | {q['median']:.2f}% | {q['mean']:.2f}% | {q['p95']:.2f}% |"
