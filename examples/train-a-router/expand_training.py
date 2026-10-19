"""Expand the original training prefix using the same pinned source and labels.

This creates development inputs only. Validation, calibration and the exposed
regression test remain unchanged; no model predictions enter the expansion.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data import (
    LABELS,
    QUESTION,
    SOURCE_HASHES,
    SOURCE_REVISION,
    audit,
    digest,
    duplicate_components,
    file_hash,
    interleave,
    partition,
    render_request,
    write,
)


def expand(source, evidence, output):
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(output)
    dataset = json.loads((evidence / "data.json").read_text())
    questions = json.loads((evidence / "questions.json").read_text())
    audit(dataset)
    if (
        dataset["source_revision"] != SOURCE_REVISION
        or dataset["protocol"]["name"] != "contest-relative-grouped-v2"
        or questions != QUESTION
    ):
        raise ValueError("Expected the original pinned dataset and question")
    source_rows = []
    for name, expected in SOURCE_HASHES.items():
        if file_hash(source / name) != expected:
            raise ValueError(f"Pinned source differs: {name}")
        source_split = name.split(".")[0]
        source_rows.extend(
            dict(row, id=f"e2h-amc-{source_split}-{i}")
            for i, row in enumerate(pq.read_table(source / name).to_pylist())
        )

    selected = []
    removed = Counter()
    for component in duplicate_components(source_rows):
        members = [source_rows[i] for i in component]
        splits = {partition(row["year"]) for row in members}
        if "train" not in splits:
            continue
        if splits != {"train"}:
            removed["cross_partition_family_rows"] += len(members)
            continue
        row = min(members, key=lambda r: digest(["representative-v2", r["id"]]))
        tag = row["tag"]
        selected.append(
            {
                "id": row["id"],
                "request": render_request(tag, row["problem"]),
                "label": int(row["rating"] >= dataset["label_thresholds"][tag]),
                "contest": tag,
                "year": row["year"],
                "family": digest(sorted(r["id"] for r in members)),
                "paper_group": f"{row['contest']}:{row['year']}",
            }
        )
        removed["within_training_family_duplicates"] += len(members) - 1

    training = interleave(selected)
    original = dataset["splits"]["train"]
    if len(training) <= len(original) or training[: len(original)] != original:
        raise ValueError("Expanded training must preserve the complete original prefix")
    expanded_dataset = dict(dataset, splits=dict(dataset["splits"], train=training))
    audit(expanded_dataset)
    lines = [
        json.dumps({"state": row["request"], "questions": questions, "expected": {"difficulty": LABELS[row["label"]]}})
        + "\n"
        for row in training
    ]
    if "".join(lines[: len(original)]) != (evidence / "train.jsonl").read_text():
        raise ValueError("Original training JSONL differs from its dataset/question definition")

    output.mkdir(parents=True)
    path = output / "train-expanded.jsonl"
    with path.open("x") as handle:
        handle.writelines(lines)
    record = {
        "source_revision": SOURCE_REVISION,
        "source_hashes": SOURCE_HASHES,
        "original_dataset_sha256": digest(dataset),
        "original_train_jsonl_sha256": file_hash(evidence / "train.jsonl"),
        "questions_sha256": file_hash(evidence / "questions.json"),
        "train_expanded_jsonl_sha256": file_hash(path),
        "expanded_rows_sha256": digest(training),
        "original_prefix_rows": len(original),
        "training_rows": len(training),
        "additional_rows": len(training) - len(original),
        "training_years": [min(r["year"] for r in training), max(r["year"] for r in training)],
        "label_thresholds": dataset["label_thresholds"],
        "label_counts": dict(Counter(r["label"] for r in training)),
        "contest_counts": dict(Counter(r["contest"] for r in training)),
        "removed": dict(removed),
        "ids": [r["id"] for r in training],
        "families": [r["family"] for r in training],
        "protocol": (
            "All eligible pre-2018 duplicate-family representatives; unchanged training-population medians, "
            "hash-selected representatives and round-robin label/contest ordering. "
            "Original validation, calibration and test populations remain untouched. "
            "The existing test is an exposed regression benchmark, not a new prospective holdout."
        ),
    }
    write(output / "training-expansion-audit.json", record)
    print(json.dumps({k: record[k] for k in ("training_rows", "additional_rows", "train_expanded_jsonl_sha256")}))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing the two pinned Parquet files")
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are preserved")
    args = parser.parse_args()
    expand(args.source, args.evidence, args.output)


if __name__ == "__main__":
    main()
