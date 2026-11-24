"""Compare two snapshot manifests: what changed in inputs, settings, size and quality."""

from __future__ import annotations

import json
from pathlib import Path


def diff_snapshots(old: str | Path, new: str | Path) -> dict:
    a = json.loads((Path(old) / "manifest.json").read_text())
    b = json.loads((Path(new) / "manifest.json").read_text())
    old_inputs = {i["path"]: i["sha256"] for i in a["inputs"]}
    new_inputs = {i["path"]: i["sha256"] for i in b["inputs"]}
    return {
        "from": a["snapshot_id"],
        "to": b["snapshot_id"],
        "identical_data": a["content"] == b["content"],
        "inputs_added": sorted(new_inputs.keys() - old_inputs.keys()),
        "inputs_removed": sorted(old_inputs.keys() - new_inputs.keys()),
        "inputs_changed": sorted(
            p for p in old_inputs.keys() & new_inputs.keys() if old_inputs[p] != new_inputs[p]
        ),
        "config_changed": {
            k: [a["config"].get(k), b["config"].get(k)]
            for k in a["config"].keys() | b["config"].keys()
            if a["config"].get(k) != b["config"].get(k)
        },
        "counts_delta": {k: b["counts"][k] - a["counts"].get(k, 0) for k in b["counts"]},
        "mean_validity": [a["quality"]["mean_validity"], b["quality"]["mean_validity"]],
    }
