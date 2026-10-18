"""Download and verify the independent-label difficulty-policy replay files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fetch
import study

REVISION = "002087323da76e925235e0c6acb12f402bf8fd9e"
FOLDER = "evaluations/2026-10-17/request-routing-difficulty-policy-v1"
MANIFEST_SHA256 = "894d168a197116963c1e55a6e3baac1988ffc89d97460bbe6532a5b91ff5f7e2"
FILES = (
    "README.md",
    "REPORT.md",
    "main/data.json",
    "main/frozen.json",
    "main/results.json",
    "main/predictions.json",
    "main/learning-curve.png",
    "main/learning-curve.svg",
    "main/teacher-order-invalid.json",
    "main/dev-embeddings/plan.json",
    "main/dev-embeddings/vectors.jsonl.gz",
    "main/test-embeddings/plan.json",
    "main/test-embeddings/vectors.jsonl.gz",
    "main/test-reranker/scores.jsonl",
    "teacher/data.json",
    "teacher/frozen.json",
    "teacher/teacher-results.json",
    "teacher/test-teacher/teacher-scores.jsonl",
    "source/metadata.json",
    "source/README.md",
    "source/train.parquet",
    "source/eval.parquet",
)


def manifest_entries(body: bytes) -> dict[str, tuple[str, int]]:
    return study.manifest_entries(body, digest=MANIFEST_SHA256, files=FILES)


def verify_evidence(directory: Path) -> None:
    study.verify_evidence(directory, digest=MANIFEST_SHA256, files=FILES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evidence/difficulty-policy"))
    args = parser.parse_args()
    # Reuse the downloader's atomic staging, size limits and pinned-hash checks.
    fetch.fetch(args.output, source=sys.modules[__name__])
    print(f"Verified {len(FILES)} replay files at {REVISION} in {args.output}")
    print(f"Full audit/source snapshot: https://huggingface.co/datasets/{study.DATASET}/tree/{REVISION}/{FOLDER}")


if __name__ == "__main__":
    main()
