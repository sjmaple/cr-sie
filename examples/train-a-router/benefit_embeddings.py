"""Reuse recorded training vectors and collect only additional eligible inputs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import serving
import solver
from benefit_expanded import checked_training
from data import digest, file_hash, write


def new_vectors(training, output, *, self_hosted=False):
    training, output = Path(training), Path(output)
    plan, inputs, _ = checked_training(training)
    new_ids = set(plan["new_train_ids"])
    rows = [row for row in inputs["train"] if row["id"] in new_ids]
    if not rows:
        raise ValueError("This population needs no new embedding calls")
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "source-binding.json",
        {
            "training_plan_sha256": file_hash(training / "plan.json"),
            "inputs_sha256": digest(rows),
            "source_sha256": file_hash(Path(__file__)),
            "max_seconds": 900,
            "max_observed_credits": 5000,
            "automatic_retries": False,
        },
    )
    serving.collect(rows, output / "additional", self_hosted=self_hosted)


def _lines(rows, directory):
    serving.vectors(rows, directory)
    selected = {row["id"] for row in rows}
    result = {}
    with gzip.open(directory / "vectors.jsonl.gz", "rb") as handle:
        for line in handle:
            identity = json.loads(line)["id"]
            if identity in selected:
                result[identity] = line
    return result


def combine(training, reused, output, additional=None):
    training, reused, output = Path(training), Path(reused), Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan, inputs, _ = checked_training(training)
    rows, new_ids = inputs["train"], set(plan["new_train_ids"])
    reused_rows = [row for row in rows if row["id"] not in new_ids]
    lines = _lines(reused_rows, reused)
    sources = {"reused": reused}
    if new_ids:
        if additional is None:
            raise ValueError("Additional eligible rows need their completed embedding collection")
        additional = Path(additional)
        binding = solver.read(additional / "source-binding.json")
        new_rows = [row for row in rows if row["id"] in new_ids]
        if (
            binding["training_plan_sha256"] != file_hash(training / "plan.json")
            or binding["inputs_sha256"] != digest(new_rows)
            or binding["source_sha256"] != file_hash(Path(__file__))
        ):
            raise ValueError("New embedding collection belongs to different source inputs")
        lines.update(_lines(new_rows, additional / "additional"))
        sources["additional"] = additional / "additional"
    elif additional is not None:
        raise ValueError("No additional collection belongs to the cleaned original population")
    hashes = {
        key: {name: file_hash(directory / name) for name in ("plan.json", "collection.json", "vectors.jsonl.gz")}
        for key, directory in sources.items()
    }
    write(
        output / "plan.json",
        {
            "model": serving.MODEL,
            "revision": serving.REVISION,
            "instruction": serving.INSTRUCTION,
            "rows_sha256": digest(rows),
            "ids": [row["id"] for row in rows],
            "input_sha256": {row["id"]: digest(row["request"]) for row in rows},
            "is_query": True,
            "dimensions": 2560,
            "training_plan_sha256": file_hash(training / "plan.json"),
            "source_sha256": file_hash(Path(__file__)),
            "source_collections_sha256": hashes,
            "reused_vector_bytes_unchanged": True,
        },
    )
    with gzip.open(output / "vectors.jsonl.gz", "wb") as handle:
        for row in rows:
            handle.write(lines[row["id"]])
    fresh = solver.read(sources["additional"] / "collection.json") if "additional" in sources else None
    write(
        output / "collection.json",
        {
            "reused_rows": len(reused_rows),
            "additional_rows": len(new_ids),
            "additional_observed_credits": fresh["credits"] if fresh else 0,
            "reused_credit_allocation": "Not apportioned: original requests may share batches with other source rows",
            "source_collections_sha256": hashes,
        },
    )
    serving.vectors(rows, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--training", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--self-hosted", action="store_true")
    merge = commands.add_parser("combine")
    for name in ("training", "reused", "output"):
        merge.add_argument(f"--{name}", type=Path, required=True)
    merge.add_argument("--additional", type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        new_vectors(args.training, args.output, self_hosted=args.self_hosted)
    else:
        combine(args.training, args.reused, args.output, args.additional)


if __name__ == "__main__":
    main()
