"""Recompute a completed GEPA run from saved scores and grades, with no inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import learning
import learning_evidence
import score
import study
from cloud import digest


def verify(actual, expected, path="result") -> None:
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{path}: keys differ")
        for key in expected:
            verify(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{path}: lengths differ")
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            verify(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{path}: number differs")
    elif actual != expected:
        raise ValueError(f"{path}: value differs")


def replay(directory: Path, s: study.Study) -> dict:
    plan = json.loads((directory / "plan.json").read_text())
    selection = json.loads((directory / "selection.json").read_text())
    expected = json.loads((directory / "results.json").read_text())
    groups = learning.split(s, len(plan["splits"]["initial"]))
    verify({name: [s.ids[i] for i in indices] for name, indices in groups.items()}, plan["splits"], "splits")
    if plan["study_revision"] != study.REVISION:
        raise ValueError("base study revision differs")
    rows = study.read_jsonl(directory / "scores.jsonl")
    records = {(row["candidate"], row["id"]): row for row in rows}
    if len(records) != len(rows):
        raise ValueError("duplicate recorded scores")
    candidates = [{**c, "instruction": plan["seed_candidate"]["instruction"]} for c in selection["candidates"]]
    verify(candidates[0], plan["seed_candidate"], "seed")

    def evaluate(candidate, fit, test):
        values, usd = np.zeros(len(s.ids)), np.zeros(len(s.ids))
        for i in fit + test:
            row = records[(digest(candidate), s.ids[i])]
            v, cost = row["score"], row["router_usd"]
            if not math.isfinite(v) or not 0 <= v <= 1 or not math.isfinite(cost) or cost < 0:
                raise ValueError("invalid recorded score/cost")
            values[i], usd[i] = v, cost
        return learning.evaluate(s.pairs[plan["pair"]], values, usd, fit, test)

    validation = [evaluate(c, groups["initial"], groups["validation"]) for c in candidates]
    validation = [{k: v for k, v in row.items() if k in score.FIELDS} for row in validation]
    verify(validation, selection["validation"], "validation")
    eligible = [i for i in range(1, len(candidates)) if learning.improvement(validation[i], validation[0])]
    chosen = min(eligible, key=lambda i: validation[i]["cost_per_1k"]) if eligible else 0
    verify(chosen, selection["chosen"], "chosen")
    heldout = {
        name: evaluate(candidates[i], groups["threshold"], groups["test"])
        for name, i in (("static", 0), ("updated", chosen))
    }
    reranker = {"promoted": chosen != 0, "chosen": chosen, "validation": validation, "heldout": heldout}
    verify(reranker, expected["reranker"], "reranker")
    embeddings = learning.embedding_experiment(s, plan["pair"], groups)
    verify(embeddings, expected["embedding"], "embedding")
    calls = study.read_jsonl(directory / "calls.jsonl")
    verify(sum(row.get("credits", 0) for row in calls), expected["cloud"]["observed_credits"], "credits")
    verify(len(calls), expected["cloud"]["requests"], "requests")
    failed = sum(row["status"] != "success" for row in calls)
    print(f"{'Router':<28} {'$/1,000':>8} {'Saving':>8} {'Quality':>8} {'Cheap':>8}")
    for name, row in (
        (f"kNN static ({len(groups['initial'])} bank)", embeddings["heldout"]["static"]),
        ("kNN expanded challenger", embeddings["heldout"]["challenger"]),
        ("kNN after update gate", embeddings["heldout"][embeddings["deployed"]]),
        ("Reranker static", heldout["static"]),
        ("Reranker after update gate", heldout["updated"]),
    ):
        learning.report_row(name, row)
    print(f"Promoted: embedding={embeddings['promoted']}, reranker={chosen != 0}")
    print(f"Verified splits, validation selection, final thresholds, held-out metrics and {len(calls)} receipts.")
    print(f"Observed credits: {expected['cloud']['observed_credits']:g}; calls with unknown billing: {failed}.")
    print(expected["cost_note"])
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="replay your own run; default is the hash-verified public run")
    args = parser.parse_args()
    directory = args.run or study.EVIDENCE / "learning"
    if args.run is None:
        learning_evidence.verify_evidence(directory)
    replay(directory, study.load_study(study.EVIDENCE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
