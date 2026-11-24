"""CLI commands for the synthetic dataset and the benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from stride.synth.bench import Benchmark, save
from stride.synth.generate import DatasetGenerator
from stride.synth.report import to_markdown


def register(commands) -> None:
    synth = commands.add_parser("synth", help="write a labelled multi-device dataset")
    synth.add_argument("--seed", type=int, default=7)
    synth.add_argument("--sessions", type=int, default=20)
    synth.add_argument("--out", type=Path, default=Path("data/synth"))
    synth.set_defaults(handler=_synth)

    bench = commands.add_parser("bench", help="score the pipeline on a synthetic dataset")
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--sessions", type=int, default=200)
    bench.add_argument("--work", type=Path, default=Path("data/bench"))
    bench.add_argument("--json", type=Path, help="write the raw results here")
    bench.add_argument("--markdown", type=Path, help="write the results table here")
    bench.set_defaults(handler=_bench)


def _synth(args) -> int:
    manifest = DatasetGenerator(args.seed, args.sessions).write(args.out)
    files = sum(len(s["files"]) for s in manifest["sessions"])
    print(f"wrote {len(manifest['sessions'])} sessions, {files} files to {args.out}")
    return 0


def _bench(args) -> int:
    result = Benchmark(args.seed, args.sessions, args.work).run()
    for path in (args.json, args.markdown):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
    if args.json:
        save(result, args.json)
    command = f"stride bench --seed {args.seed} --sessions {args.sessions}"
    if args.markdown:
        args.markdown.write_text(to_markdown(result, command))
    print(json.dumps(result.__dict__, indent=2, default=float))
    return 0
