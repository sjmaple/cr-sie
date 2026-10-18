"""Pinned public GEPA handler experiment and reserved cost-routing audit."""

from pathlib import Path

import study

REVISION = "54bda087d98fa2bd7f1883f194cd534276a904b2"
FOLDER = "evaluations/2026-10-16/request-routing-gepa-improvement-v1"
MANIFEST_SHA256 = "9577ae8e6317352a6f63f308e651e88854efd399b913a76829ab05c5f8754b8e"
FILES = (
    "README.md",
    "CLINC150-LICENSE.txt",
    "data.json",
    "plan.json",
    "selection.json",
    "candidates.json",
    "scores.jsonl",
    "calls.jsonl",
    "handlers.json",
    "results.json",
    "predictions.json",
    "uncertainty.json",
    *(f"reflection-{i:02d}.json" for i in range(1, 17)),
    *(
        f"cost-diagnostic/{name}"
        for name in (
            "plan.json",
            "selection.json",
            "candidates.json",
            "scores.jsonl",
            "calls.jsonl",
            "router.json",
            "results.json",
            "uncertainty.json",
        )
    ),
)


def manifest_entries(body: bytes) -> dict[str, tuple[str, int]]:
    return study.manifest_entries(body, digest=MANIFEST_SHA256, files=FILES)


def verify_evidence(directory: Path) -> None:
    study.verify_evidence(directory, digest=MANIFEST_SHA256, files=FILES)
