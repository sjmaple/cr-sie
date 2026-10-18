"""Download the pinned paired-answer routing experiment and its actual receipts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fetch
import study

REVISION = "fc8ac96c8492c30945a133dd6669013ba559e494"
FOLDER = "evaluations/2026-10-18/request-routing-economics-v1"
MANIFEST_SHA256 = "ed092969dee37049fae8cd9571483a2dcfda0a7194b9bcb9563a96c6f9568336"
FILES = (
    "README.md",
    "plan.json",
    "inputs.json",
    "answer-settings.json",
    "answers-development.jsonl",
    "answers-test.jsonl",
    "gold-for-independent-scoring-only.json",
    "teacher-input.json",
    "teacher-reviews.jsonl",
    "teacher-provenance.json",
    "routing-frozen.json",
    "review-routing-frozen.json",
    "gold-routing-frozen.json",
    "evaluation-plan.json",
    "economic-results.json",
    "economic-results-unadjusted.json",
    "grading-syntax-audit.json",
    "economic-summary.json",
    "collection-audit.json",
    "economic-results.png",
    "economic-results.svg",
    "router-receipts.jsonl",
    "router-embeddings/plan.json",
    "router-embeddings/vectors.jsonl.gz",
    "router-embeddings/collection.json",
)


def manifest_entries(body):
    return study.manifest_entries(body, digest=MANIFEST_SHA256, files=FILES)


def verify_evidence(directory):
    study.verify_evidence(directory, digest=MANIFEST_SHA256, files=FILES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evidence/economics"))
    args = parser.parse_args()
    fetch.fetch(args.output, source=sys.modules[__name__])
    print(f"Verified paired answers, teacher labels, frozen policies and credit receipts at {REVISION}.")


if __name__ == "__main__":
    main()
