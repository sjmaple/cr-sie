"""Pinned public SIE Cloud GEPA workshop run (separate from the original studies)."""

from pathlib import Path

import study

REVISION = "dfe4f4b19c4900c70518e4b4d58768cd5b769582"
FOLDER = "evaluations/2026-10-16/request-routing-gepa-workshop-v1"
MANIFEST_SHA256 = "e5e0f5e5606513572f93dd1ed13a497a74cec90021b81cbfa082f8b8067b2971"
FILES = (
    "README.md",
    "plan.json",
    "selection.json",
    "candidates.json",
    "scores.jsonl",
    "calls.jsonl",
    "router.json",
    "results.json",
    *(f"reflection-{i:02d}.json" for i in range(1, 7)),
)


def manifest_entries(body: bytes) -> dict[str, tuple[str, int]]:
    return study.manifest_entries(body, digest=MANIFEST_SHA256, files=FILES)


def verify_evidence(directory: Path) -> None:
    study.verify_evidence(directory, digest=MANIFEST_SHA256, files=FILES)
