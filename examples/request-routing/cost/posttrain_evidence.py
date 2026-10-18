"""Download verified post-training evidence; native model weights are optional."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

import fetch
import study

REVISION = "85c9d93a73d5559a6c69a22145f5f86f6b450fbc"
FOLDER = "evaluations/2026-10-18/request-routing-course-posttrain-v2"
MANIFEST_SHA256 = "d0e6635f6bfddeb97495a1a93f869a29817314350f233c5de65ea2d0d8b6cb74"
FILES = (
    "README.md",
    "report.json",
    "plan.json",
    "frozen.json",
    "selection.json",
    "train.jsonl",
    "validation.jsonl",
    "questions.json",
    "preparation.json",
    "reference.json",
    "confirmation.json",
    "confirmation-ridge.json",
    "evaluation-reference.json",
    "evaluation-confirmation.json",
    "base-history.json",
    "full-history.json",
    "lora-history.json",
    "full-export-parity.json",
    "lora-export-parity.json",
    "lora-merge-parity.json",
    "sie-serving.json",
    "sie-latency-confirmation.json",
    "course-results.png",
    "course-results.svg",
    "confirmation-embeddings/plan.json",
    "confirmation-embeddings/vectors.jsonl.gz",
    "confirmation-embeddings/collection.json",
    "confirmation-embeddings/calls.jsonl",
    "verification/clean-verification.json",
    "verification/cloud-smoke.json",
    "verification/local-smoke.json",
    "verification/local-decision-serving.json",
)
CHECKPOINT_FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
)


def manifest_entries(body):
    return study.manifest_entries(body, digest=MANIFEST_SHA256, files=FILES)


def verify_evidence(directory):
    study.verify_evidence(directory, digest=MANIFEST_SHA256, files=FILES)


def verify_file(path, expected):
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    if (h.hexdigest(), size) != expected:
        raise study.EvidenceError(f"{path.name}: size or SHA-256 differs from the pinned manifest")


def weights(directory, variant):
    names = tuple(f"{variant}/{name}" for name in CHECKPOINT_FILES)
    entries = study.manifest_entries((directory / study.MANIFEST).read_bytes(), digest=MANIFEST_SHA256, files=names)
    for name, expected in entries.items():
        path = directory / name
        if path.exists():
            verify_file(path, expected)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # Streaming keeps an 843 MB checkpoint from becoming an 843 MB RAM buffer.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                request = urllib.request.Request(
                    fetch.url(name, source=sys.modules[__name__]),
                    headers={"User-Agent": "sie-examples/request-routing"},
                )
                with urllib.request.urlopen(request, timeout=300) as response:
                    total = 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        total += len(chunk)
                        if total > expected[1]:
                            raise study.EvidenceError("Checkpoint exceeds its declared size")
                        handle.write(chunk)
                handle.flush()
                verify_file(temporary, expected)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Verified {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evidence/posttrain"))
    parser.add_argument("--weights", choices=("full", "lora", "both"))
    args = parser.parse_args()
    fetch.fetch(args.output, source=sys.modules[__name__])
    if args.weights:
        for variant in ("full", "lora") if args.weights == "both" else (args.weights,):
            weights(args.output, variant)
    print(f"Verified {len(FILES)} replay files at {REVISION} in {args.output}")
    print(f"Public evidence: https://huggingface.co/datasets/{study.DATASET}/tree/{REVISION}/{FOLDER}")


if __name__ == "__main__":
    main()
