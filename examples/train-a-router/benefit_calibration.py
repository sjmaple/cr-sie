"""Apply the amended grader to saved calibration answers without reading test answers.

The original collection and reports remain unchanged. A separate snapshot
records that this amendment happened after collection, before the new study's
test outcome access. Replaying published results makes no unseen-test claim.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import benefit
import benefit_expanded_gold_v2 as grading
import solver
import solver_policies
from data import digest, file_hash, write


def grade_calibration(directory):
    directory = Path(directory)
    plan, inputs, settings = solver.checked_inputs(directory)
    gold = solver.read(directory / "gold-for-scoring-only.json")
    if digest(gold) != plan["gold_sha256"]:
        raise ValueError("Original source references changed")
    collection = directory / "answers-calibration"
    summary = solver.read(collection / "collection.json")
    binding = solver.read(collection / "collection-plan.json")
    selected = inputs["calibration"]
    if (
        summary["status"] != "complete"
        or summary["completed"] != 2 * len(selected)
        or summary["failed"] != 0
        or binding["ids"] != [row["id"] for row in selected]
        or binding["plan_sha256"] != file_hash(directory / "plan.json")
        or binding["settings_sha256"] != digest(settings)
        or binding["split"] != "calibration"
    ):
        raise ValueError("Need the complete calibration population under its original settings")
    solver.check_review(directory, binding["ids"])
    rows = []
    for row in selected:
        outcome = {"id": row["id"]}
        for arm in ("cheap", "strong"):
            path = collection / f"{row['id']}-{arm}.json"
            answer = solver.read(path)
            solver.validate_record(answer, row, arm, settings)
            if answer["status"] != "complete":
                raise ValueError("Calibration answer is incomplete")
            measured = grading.grade_expanded(row["id"], gold[row["id"]], answer["answer"])
            if measured["status"] in ("gold_unparseable", "grading_error"):
                raise ValueError("Amended grader cannot safely evaluate a calibration answer")
            outcome.update(
                {
                    f"{arm}_correct": measured["correct"],
                    f"{arm}_credits": answer["credits"],
                    f"{arm}_grade": measured,
                    f"{arm}_record_sha256": file_hash(path),
                }
            )
        rows.append(outcome)
    return {
        "split": "calibration",
        "plan_sha256": file_hash(directory / "plan.json"),
        "collection_plan_sha256": file_hash(collection / "collection-plan.json"),
        "collection_summary_sha256": file_hash(collection / "collection.json"),
        "policies_frozen_sha256": None,
        "grader_source_sha256": file_hash(Path(__file__).with_name("solver_answers.py")),
        "gold_review_source_sha256": file_hash(Path(__file__).with_name("solver_gold.py")),
        "gold_review_sha256": file_hash(directory / "gold-review.json"),
        **grading.outcome_grader_metadata(),
        "amendment_builder_source_sha256": file_hash(Path(__file__)),
        "rows": rows,
    }


def amend(directory, output, *, attest_test_outcomes_unopened=False, replay=False):
    if bool(attest_test_outcomes_unopened) == bool(replay):
        raise ValueError("Choose either a fresh-study test-access assertion or published-result replay")
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    old_path = directory / "policies-frozen.json"
    old = solver.read(old_path)
    solver_policies.check_seal(old, solver_policies.FROZEN_FORMAT)
    if old["plan_sha256"] != file_hash(directory / "plan.json"):
        raise ValueError("Original collection plan differs from its historical freeze")
    graded = grade_calibration(directory)
    keys = ("id", "cheap_correct", "strong_correct", "cheap_credits", "strong_credits")
    rows = [{key: row[key] for key in keys} for row in graded["rows"]]
    if [row["id"] for row in rows] != [row["id"] for row in old["calibration_rows"]]:
        raise ValueError("Amended calibration population differs")
    output.mkdir(parents=True)
    shutil.copyfile(directory / "plan.json", output / "plan.json")
    write(output / "grades-calibration.json", graded)
    amended = {key: value for key, value in old.items() if key != "sha256"}
    amended.update(
        {
            "calibration_rows": rows,
            "calibration_grades_sha256": file_hash(output / "grades-calibration.json"),
            **grading.outcome_grader_metadata(),
            "amendment_builder_source_sha256": file_hash(Path(__file__)),
            "selections": solver_policies._selections(
                rows, old["source_scores"], old["embedding_calibration_credits"], old["credit_usd"]
            ),
            "amendment": {
                "created_at": benefit.now(),
                "original_frozen_sha256": file_hash(old_path),
                "original_calibration_grades_sha256": old["calibration_grades_sha256"],
                "reason": "Training-only teacher audit exposed a source error and approximate numeric comparison; use the same amended exactness protocol for all new-study splits.",
                "boundary": "public_regression_reproduction" if replay else "after_test_collection_before_test_outcome_access",
                "attest_test_outcomes_unopened": bool(attest_test_outcomes_unopened),
                "legacy_selections": "Recomputed only to keep the compatibility snapshot internally consistent; new outcome models choose their own cutoffs after training.",
            },
            "test_access": "This calibration-only amendment does not read or authorize solver test outcomes",
        }
    )
    write(output / "policies-frozen.json", solver_policies.seal(amended))
    return {
        "n": len(rows),
        "correct": {arm: sum(row[f"{arm}_correct"] for row in rows) for arm in ("cheap", "strong")},
        "changed_labels": {
            arm: [
                row["id"]
                for row, prior in zip(rows, old["calibration_rows"], strict=True)
                if row[f"{arm}_correct"] != prior[f"{arm}_correct"]
            ]
            for arm in ("cheap", "strong")
        },
        "frozen_sha256": file_hash(output / "policies-frozen.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--attest-test-outcomes-unopened", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            amend(
                args.collection,
                args.output,
                attest_test_outcomes_unopened=args.attest_test_outcomes_unopened,
                replay=args.replay,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
