"""Explicit labeled-data control, separate from the no-gold teacher recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import economics
import economics_policy
from cloud import digest


def train(directory, reference, development_limit=None):
    data = economics.read(directory / "inputs.json")
    settings = economics.read(directory / "answer-settings.json")
    gold = economics.read(directory / "gold-for-independent-scoring-only.json")
    if digest(gold) != economics.read(directory / "plan.json")["gold_sha256"]:
        raise ValueError("Answer keys changed")
    # Only development rows are graded here. Test outcomes never enter fitting.
    rows = data["train"][:development_limit] + data["validation"][:development_limit]
    records = economics.answer_records(directory, {r["id"]: r["request"] for r in rows}, settings["arms"])
    targets = {}
    for row in rows:
        grades = {}
        for alias in ("cheap", "strong"):
            answer = records[(row["id"], settings[f"{alias}_arm"])]["answer"]
            measured = economics.grade(answer, gold[row["id"]], row["request"])
            if not measured["reference_parsed"]:
                raise ValueError("Unparseable development answer key")
            grades[alias] = int(measured["correct"])
        # On the shared optimizer's reference scale, strong=1. Switching to
        # cheap changes quality by q_cheap-q_strong, including cheap-only wins.
        targets[row["id"]] = 1 + grades["cheap"] - grades["strong"]
    return economics_policy.train(
        directory,
        reference,
        development_limit=development_limit,
        supervised_reference=targets,
        label_source="gold_benefit",
        label_provenance={"gold_sha256": digest(gold), "development_only": True},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("evidence/difficulty-policy"))
    parser.add_argument("--development-limit", type=int, choices=(64, 128, 256, 512))
    args = parser.parse_args()
    print(json.dumps(train(args.run, args.reference, args.development_limit), indent=2))
