"""Check the current model catalog before a fresh benefit-label collection phase."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import benefit_collect
from benefit_data import checked_training
from data import digest, file_hash, write
from serving import SingleSendClient, error_receipt


def collect(directory, arm, phase, **options):
    directory = Path(directory)
    _, _, settings = checked_training(directory)
    if arm not in settings["arms"] or phase not in benefit_collect.PHASES:
        raise ValueError("Choose a declared solver arm and collection phase")
    expected = settings["arms"][arm]
    folder = directory / "catalog-preflights" / arm
    sequence = len(list(folder.glob(f"{phase}-*.json"))) + 1
    path = folder / f"{phase}-{sequence:04d}.json"
    record = {
        "format": "sie-benefit-catalog-preflight-v1",
        "arm": arm,
        "phase": phase,
        "model": expected["model"],
        "expected_revision": expected["revision"],
        "plan_sha256": file_hash(directory / "plan.json"),
        "settings_sha256": digest(settings),
        "endpoint_sha256": digest(os.environ.get("SIE_BASE_URL", "https://api.superlinked.com")),
        "wrapper_source_sha256": file_hash(Path(__file__)),
        "collector_source_sha256": file_hash(Path(benefit_collect.__file__)),
        "scope": "Prospective catalog name/revision check for this phase; not an assertion of identical serving runtime",
    }
    try:
        with SingleSendClient(timeout_s=60, read_timeout_s=60) as client:
            metadata = client.get_model(expected["model"])
        name = metadata.get("name", metadata.get("id"))
        revision = metadata.get("revision")
        matched = name == expected["model"] and revision == expected["revision"]
        record.update(status="matched" if matched else "mismatch", observed_name=name, observed_revision=revision)
    except Exception as error:  # Persist only sanitized metadata failures before re-raising.
        write(path, record | {"status": "failed", "error": error_receipt(error)})
        raise
    write(path, record)
    if not matched:
        raise ValueError("Current model catalog differs from the prepared solver; no generation was submitted")
    return benefit_collect.collect(directory, arm, phase, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("cheap", "strong"))
    parser.add_argument("--phase", required=True, choices=benefit_collect.PHASES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--max-credits", type=float)
    parser.add_argument("--self-hosted", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-unknown", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--runtime-receipt", type=Path)
    result = collect(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in result.items() if key != "receipt_sha256"}, sort_keys=True))


if __name__ == "__main__":
    main()
