"""Fetch one pinned public checkpoint and persist a complete file-hash manifest."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from tools.benchmarks.gliclass_mixed.evidence import atomic_json, digest
from tools.benchmarks.gliclass_mixed.validate import validate
from tools.benchmarks.gliclass_mixed.workload import MODEL, REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time_ns()
    snapshot = Path(
        snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "spm.model"])
    )
    files = {
        str(path.relative_to(snapshot)): {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }
    if not any(name.endswith(".safetensors") for name in files):
        raise AssertionError("Pinned snapshot has no safe tensor weights")
    atomic_json(
        args.output / "manifest.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "snapshot": str(snapshot),
            "started_time_ns": started,
            "finished_time_ns": time.time_ns(),
            "files": files,
        },
    )
    validate(snapshot, args.output / "workload-validation.json")


if __name__ == "__main__":
    main()
