"""Retain two expanded training rows after explicit symbolic-answer review.

This amendment was made after training answers were collected, before fitting
the expanded models. It does not change the training population, learner,
calibration, or test grader. Original references and automatic grades remain
in each adjudication so the source correction is visible and reproducible.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import benefit_expanded
import solver
from benefit_expanded_gold_v2 import grade_expanded
from data import digest, file_hash, write
from solver_answers import extract_final_answer

REVIEW = {
    "e2h-amc-eval-1621": {
        "original": "2001^{2002}",
        "canonical": "2001^{2002}",
        "answers": {"2001^{2002}": True, "2001^{2002^{2003^{2004}}}": False},
        "proof": "The four nested logarithms are defined iff the third logarithm is positive, the second is greater than1, the first is greater than2002, and x>2001^2002. Since2001>1 and2002^(2003^2004)>2002, the predicted power tower is strictly larger than the boundary. No huge integer or floating-point approximation is needed.",
        "source_correction": False,
    },
    "e2h-amc-eval-1775": {
        "original": r"\sqrt{\sqrt{4^{2012}+1}}",
        "canonical": r"\sqrt{4^{2012}+1}",
        "answers": {r"\sqrt{2^{4024}+1}": True, "2^{2012}": False},
        "proof": "The nth tent-map iterate has2^n segments of width2^-n and vertical displacement1. Total length is2^n*sqrt(2^-2n+1)=sqrt(4^n+1). The source solution derives this expression but its final boxed answer adds an erroneous extra square root. At n=2012,4^2012=2^4024 exactly, so the strong answer is equal;2^2012 has square4^2012, strictly less than4^2012+1, so the cheap answer is unequal.",
        "source_correction": True,
    },
}


def reviewed_grade(identity, original, response):
    automatic = grade_expanded(identity, original, response)
    if identity not in REVIEW:
        if automatic["status"] in ("gold_unparseable", "grading_error"):
            raise ValueError(f"Unreviewed grading failure: {identity}")
        return automatic
    rule = REVIEW[identity]
    final = re.sub(r"\s+", "", extract_final_answer(response))
    if original != rule["original"] or final not in rule["answers"]:
        raise ValueError(f"Symbolic training adjudication does not cover this answer: {identity}")
    return automatic | {
        "correct": rule["answers"][final],
        "status": "graded",
        "training_adjudication": {
            "identity": identity,
            "original_reference": original,
            "canonical_reference": rule["canonical"],
            "proof": rule["proof"],
            "source_correction": rule["source_correction"],
            "automatic_grade": automatic,
            "reviewed_after_training_collection": True,
        },
    }


def grade(directory, output=None):
    directory = Path(directory)
    output = Path(output) if output else directory / "grades-train.json"
    if output.exists():
        raise FileExistsError(output)
    plan, inputs, settings = benefit_expanded.checked_training(directory)
    gold = solver.read(directory / "gold-for-scoring-only.json")
    if digest(gold) != plan["gold_sha256"] or plan["grader"] != benefit_expanded.grader_metadata():
        raise ValueError("Expanded references or frozen grading dependency changed")
    collection = directory / "answers-train"
    summary = solver.read(collection / "collection.json")
    binding = solver.read(collection / "collection-plan.json")
    solver.collection_binding(binding, directory, "train", inputs["train"], settings, None, True)
    if summary["status"] != "complete" or summary["completed"] != 2 * len(inputs["train"]) or summary["failed"]:
        raise ValueError("Expanded paired collection is incomplete")
    rows, reviewed = [], []
    for row in inputs["train"]:
        outcome = {"id": row["id"]}
        for arm in settings["arms"]:
            path = collection / f"{row['id']}-{arm}.json"
            record = solver.read(path)
            solver.validate_record(record, row, arm, settings)
            if record["status"] != "complete":
                raise ValueError("Incomplete expanded training answer")
            measured = reviewed_grade(row["id"], gold[row["id"]], record["answer"])
            outcome.update(
                {
                    f"{arm}_correct": measured["correct"],
                    f"{arm}_credits": record["credits"],
                    f"{arm}_grade": measured,
                    f"{arm}_record_sha256": file_hash(path),
                }
            )
            if "training_adjudication" in measured:
                reviewed.append({"id": row["id"], "arm": arm, **measured["training_adjudication"]})
        rows.append(outcome)
    if {row["id"] for row in reviewed} != set(REVIEW) or len(reviewed) != 2 * len(REVIEW):
        raise ValueError("The fixed expanded-training review population differs")
    result = {
        "split": "train",
        "plan_sha256": file_hash(directory / "plan.json"),
        "collection_plan_sha256": file_hash(collection / "collection-plan.json"),
        "collection_summary_sha256": file_hash(collection / "collection.json"),
        "policies_frozen_sha256": None,
        **benefit_expanded.source_hashes(),
        "gold_review_sha256": plan["gold_review_sha256"],
        "training_gold_rules_sha256": plan["training_gold_rules_sha256"],
        "post_collection_training_adjudication": {
            "source_sha256": file_hash(Path(__file__)),
            "review_sha256": digest(REVIEW),
            "scope": "Two expanded-only training rows; no exclusions, no calibration/test changes, and no learner changes. Parent grader hashes identify the retained automatic protocol, augmented explicitly by this review.",
            "records": reviewed,
        },
        "rows": rows,
    }
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    grade(args.directory, args.output)


if __name__ == "__main__":
    main()
