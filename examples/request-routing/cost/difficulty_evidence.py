"""Pinned public GEPA difficulty experiment, including frozen policy and usage."""

from pathlib import Path

import study

REVISION = "dc85e8a7e4d70c53b4680febdda8fab9e66d9c0a"
FOLDER = "evaluations/2026-10-17/request-routing-gepa-difficulty-v2"
MANIFEST_SHA256 = "215755d02ca2be5d81ea08d98391f2745d0928e81de8466e46e91a27feceb0d7"
FILES = (
    "README.md",
    "plan.json",
    "selection.json",
    "candidates.json",
    "scores.jsonl",
    "calls.jsonl",
    "router.json",
    "results.json",
    "runtime.json",
    "historical-prices.json",
    "pilot-note.json",
    *(f"reflection-{i:02d}.json" for i in range(1, 17)),
)


def manifest_entries(body: bytes) -> dict[str, tuple[str, int]]:
    return study.manifest_entries(body, digest=MANIFEST_SHA256, files=FILES)


def verify_evidence(directory: Path) -> None:
    study.verify_evidence(directory, digest=MANIFEST_SHA256, files=FILES)
