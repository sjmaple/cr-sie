"""Open the common solver test after the main and teacher studies are frozen.

The original answer recordings and historical grading protocol stay intact.
This command writes a separate result using the shared amended grader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import benefit
import benefit_expanded_gold_v2 as grading
import benefit_outcomes as outcomes
import benefit_outcomes_native as native
import source_review
import teacher_outcomes as teacher
from data import digest, file_hash, write


def validate_access(head_runs, native_runs, calibration, teacher_run=None):
    if len(head_runs) != 2 or len(native_runs) != 2:
        raise ValueError("Freeze both training sizes for both router designs before test access")
    calibration = Path(calibration)
    calibration_sha = file_hash(calibration)
    records, populations, sizes = [], [], []
    for head_path, native_path in zip(head_runs, native_runs, strict=True):
        head_path, native_path = Path(head_path), Path(native_path)
        head_frozen, head_model = outcomes.checked_frozen(head_path)
        plan, trained, native_frozen = native.checked_frozen(native_path, head_path)
        head = benefit.checked(outcomes.read(head_path / "test-access.json"), outcomes.ACCESS_FORMAT)
        adapted = benefit.checked(outcomes.read(native_path / "test-access.json"), native.ACCESS_FORMAT)
        if not (
            head["frozen_sha256"] == file_hash(head_path / "frozen.json")
            and head["selected_model_sha256"] == head_frozen["selected_model_sha256"]
            and head["plan_sha256"] == head_model["plan_file_sha256"]
            and head["source_code_sha256"] == outcomes.code_hashes()
            and adapted["native_frozen_sha256"] == file_hash(native_path / "frozen.json")
            and adapted["head_frozen_sha256"] == head["frozen_sha256"]
            and adapted["head_plan_sha256"] == plan["head_plan_sha256"] == head["plan_sha256"]
            and adapted["checkpoint_manifest_sha256"] == digest(trained["checkpoints"])
            and adapted["source_code_sha256"] == native.code_hashes()
            and native_frozen["selected_embedding_sha256"] == head["selected_model_sha256"]
            and head_model["plan"]["original_frozen_sha256"] == calibration_sha
            and all(
                head_frozen["grading_provenance"].get(key) == value
                for key, value in grading.outcome_grader_metadata().items()
            )
        ):
            raise ValueError("Test access does not bind the frozen models, calibration and current grader")
        populations.append(head_model["plan"]["populations"]["test"])
        sizes.append(len(head_model["plan"]["populations"]["train"]))
        records.append(
            {
                "training_rows": sizes[-1],
                "head_plan_sha256": head["plan_sha256"],
                "head_frozen_sha256": head["frozen_sha256"],
                "native_frozen_sha256": adapted["native_frozen_sha256"],
                "head_access_sha256": file_hash(head_path / "test-access.json"),
                "native_access_sha256": file_hash(native_path / "test-access.json"),
            }
        )
    if populations[0] != populations[1] or sizes[0] >= sizes[1]:
        raise ValueError("Need the same test population and increasing original/expanded training sizes")
    if teacher_run is None:
        raise ValueError("Freeze the declared teacher-label control before opening the common test")
    teacher_run = Path(teacher_run)
    teacher_frozen, teacher_model = teacher.checked_frozen(teacher_run)
    teacher_access = benefit.checked(outcomes.read(teacher_run / "test-access.json"), teacher.ACCESS_FORMAT)
    reference = teacher_model["plan"]["reference_plan"]
    if not (
        teacher_access["frozen_sha256"] == file_hash(teacher_run / "frozen.json")
        and teacher_access["selected_model_sha256"] == teacher_frozen["selected_model_sha256"]
        and teacher_access["plan_sha256"] == teacher_model["plan_file_sha256"]
        and teacher_access["source_code_sha256"] == teacher.code_hashes()
        and reference["populations"]["test"] == populations[0]
        and len(reference["populations"]["train"]) == sizes[1]
        and teacher_model["plan"]["reference_plan_file_sha256"] == records[1]["head_plan_sha256"]
        and reference["original_frozen_sha256"] == calibration_sha
        and teacher_frozen["grading_provenance"] == head_frozen["grading_provenance"]
    ):
        raise ValueError("Teacher control differs from the common frozen evaluation and expanded training population")
    records.append(
        {
            "design": "teacher_embedding",
            "training_rows": len(teacher_model["head"]["training_ids"]),
            "frozen_sha256": teacher_access["frozen_sha256"],
            "access_sha256": file_hash(teacher_run / "test-access.json"),
            "plan_sha256": teacher_access["plan_sha256"],
        }
    )
    return records


def grade(directory, calibration, head_runs, native_runs, output, *, teacher_run=None, source_review_path=None):
    directory, calibration, output = Path(directory), Path(calibration), Path(output)
    if output.exists():
        raise FileExistsError(output)
    access = validate_access(head_runs, native_runs, calibration, teacher_run)
    if source_review_path is None:
        raise ValueError("The explicit source-review manifest is required")
    review_metadata = source_review.metadata(source_review_path)
    snapshot = outcomes.read(calibration)
    if snapshot["plan_sha256"] != file_hash(directory / "plan.json") or snapshot["amendment"][
        "original_frozen_sha256"
    ] != file_hash(directory / "policies-frozen.json"):
        raise ValueError("Amended calibration does not belong to this original paired collection")
    if any(snapshot.get(key) != value for key, value in review_metadata.items()):
        raise ValueError("Test reference review differs from frozen calibration")
    access_path = output.with_name(output.stem + "-access.json")
    write(
        access_path,
        {
            "format": "sie-reviewed-outcome-all-model-access-v1",
            "split": "test",
            "created_at": benefit.now(),
            "statement": "All four main studies and the teacher-label control are frozen and their access receipts checked before this command opens solver test outcomes. Collection happened earlier.",
            "studies": access,
            "calibration_sha256": file_hash(calibration),
            "original_solver_plan_sha256": file_hash(directory / "plan.json"),
            "grading_command_sha256": file_hash(Path(__file__)),
            **grading.outcome_grader_metadata(),
            **review_metadata,
        },
    )
    result = source_review.grade_collection(directory, "test", source_review_path, test_access_path=access_path)
    result.update(
        {
            "policies_frozen_sha256": file_hash(calibration),
            **grading.outcome_grader_metadata(),
            "grading_command_sha256": file_hash(Path(__file__)),
            "all_studies_access_sha256": file_hash(access_path),
        }
    )
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--heads", type=Path, nargs=2, required=True, metavar=("ORIGINAL", "EXPANDED"))
    parser.add_argument("--native", type=Path, nargs=2, required=True, metavar=("ORIGINAL", "EXPANDED"))
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = grade(
        args.collection,
        args.calibration,
        args.heads,
        args.native,
        args.output,
        teacher_run=args.teacher,
        source_review_path=args.source_review,
    )
    print(
        json.dumps(
            {
                "n": len(result["rows"]),
                "correct": {arm: sum(row[f"{arm}_correct"] for row in result["rows"]) for arm in ("cheap", "strong")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
