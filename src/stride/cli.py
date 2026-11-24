"""Command line: stride ingest | check | diff | show."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stride.check import SnapshotChecker, all_passed
from stride.diff import diff_snapshots
from stride.pipeline import Pipeline, PipelineConfig
from stride.registry import Registry
from stride.snapshot import SnapshotWriter


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stride", description=__doc__)
    commands = parser.add_subparsers(required=True)

    ingest = commands.add_parser("ingest", help="run the pipeline and write a snapshot")
    ingest.add_argument("input", type=Path, help="folder with one subfolder per device")
    ingest.add_argument("--out", type=Path, default=Path("out"))
    ingest.add_argument("--registry", type=Path, help="device registry YAML (default: built-in)")
    ingest.add_argument("--rate", type=float, default=1.0, help="resample rate in Hz")
    ingest.set_defaults(handler=_ingest)

    check = commands.add_parser("check", help="validate a snapshot before release")
    check.add_argument("snapshot", type=Path)
    check.add_argument("--min-validity", type=float, default=0.9)
    check.set_defaults(handler=_check)

    diff = commands.add_parser("diff", help="compare two snapshots")
    diff.add_argument("old", type=Path)
    diff.add_argument("new", type=Path)
    diff.set_defaults(handler=_diff)

    show = commands.add_parser("show", help="print a snapshot's issues and sync results")
    show.add_argument("snapshot", type=Path)
    show.set_defaults(handler=_show)

    _add_extra_commands(commands)
    return parser


def _add_extra_commands(commands) -> None:
    """Commands that live in optional modules (synthetic data, benchmark)."""
    try:
        from stride.synth.cli import register
    except ImportError:
        return
    register(commands)


def _ingest(args) -> int:
    config = PipelineConfig(rate_hz=args.rate)
    pipeline = Pipeline(Registry.load(args.registry), config)
    result = pipeline.run(args.input)
    folder = SnapshotWriter(args.out).write(result, config, args.input)
    manifest = json.loads((folder / "manifest.json").read_text())
    print(f"snapshot {manifest['snapshot_id']} -> {folder}")
    print(json.dumps({"counts": manifest["counts"], "quality": manifest["quality"]}, indent=2))
    return 0


def _check(args) -> int:
    checks = SnapshotChecker(args.snapshot, args.min_validity).run()
    for c in checks:
        print(f"{'PASS' if c.passed else 'FAIL'}  {c.name}  {c.detail if not c.passed else ''}")
    return 0 if all_passed(checks) else 1


def _diff(args) -> int:
    print(json.dumps(diff_snapshots(args.old, args.new), indent=2))
    return 0


def _show(args) -> int:
    reports = json.loads((args.snapshot / "reports.json").read_text())
    for rec in reports["recordings"]:
        status = "ok  " if rec["passed"] else "FLAG"
        units = "; ".join(f["cause"] for f in rec["unit_findings"])
        print(f"{status} {rec['path']:<40} {rec['issues'] or ''} {units}")
    for session in reports["sessions"]:
        for member, sync in session["sync"].items():
            print(
                f"sync {session['session_id']} {member}: offset {sync['offset_s']:+.2f} s, "
                f"drift {sync['drift_ppm']:+.1f} ppm, residual {sync['residual_m']:.1f} m"
            )
    for path, error in reports["failures"].items():
        print(f"FAIL {path}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
