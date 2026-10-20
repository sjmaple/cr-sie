"""Compare independent teacher labels with benchmark labels on matched arithmetic rows."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import arithmetic_answers
import arithmetic_optimizer
import arithmetic_study as study
import arithmetic_teacher
import teacher_format
from data import digest, file_hash, write
from serving import vectors
from solver_answers import extract_final_answer

PROTOCOL = {
    "label_source": "Independent question-only teacher answers; never substitute benchmark references.",
    "eligibility": (
        "Completed teacher answer satisfying its saved numeric or general answer contract and successful self-comparison. "
        "This mask uses only question and teacher reference; apply the identical mask to both heads."
    ),
    "learner": "Unchanged two logistic heads, original source-family folds, fixed penalty grid and conditional cost formula.",
    "loss_evaluation": arithmetic_optimizer.EVALUATION,
    "unresolved_solver_comparison": "Fail the comparison; never drop a row or silently label an unresolved answer wrong.",
    "teacher_accuracy": "Separate benchmark-reference audit, not an eligibility filter or a training target.",
    "scope": "Training labels only; calibration and test remain benchmark-supervised.",
}


def grade_teacher(question, reference, response):
    """Treat an unqualified clock hour as a concise answer, with no gold input."""
    final = extract_final_answer(response)
    expected, actual = arithmetic_answers.clock_value(reference), arithmetic_answers.clock_value(final)
    if (
        arithmetic_answers.clock_question(question)
        and expected
        and actual
        and expected["period"] is not None
        and actual["period"] is None
        and actual["hour"] <= 12
    ):
        short = f"{expected['hour']}:{expected['minute']:02d}"
        return arithmetic_answers.grade_reference(question, short, response) | {
            "teacher_clock_reference": reference,
            "normalization": "Bare clock-hour prediction leaves period unspecified; preserve explicit period conflicts.",
        }
    return arithmetic_answers.grade_reference(question, reference, response)


def numeric_reference(reference):
    """Validate the declared numeric-only answer without any benchmark input."""
    return len(reference) <= 128 and arithmetic_teacher.protocol_for("numeric").numeric_answer(reference)


def label_pair(question, reference, answers, *, answer_format="general"):
    """Construct targets without receiving any benchmark reference or identity."""
    try:
        self_grade = grade_teacher(question, reference, r"\boxed{" + reference + "}")
        if answer_format == "numeric":
            unit = {
                "unit_status": "numeric_requested_units"
                if numeric_reference(reference)
                else "numeric_contract_invalid",
                "reference_only": True,
            }
        elif answer_format != "general":
            raise ValueError("Unknown teacher answer format")
        elif arithmetic_answers.clock_question(question) and arithmetic_answers.clock_value(reference):
            unit = {"unit_status": "declared_clock"}
        else:
            unit = teacher_format.normalize_final(question, reference)
    except (ValueError, TypeError, ArithmeticError, TimeoutError) as error:
        return False, {"error_type": type(error).__name__, "reference_only": True}, {}
    usable = bool(
        self_grade["status"] == "graded"
        and self_grade["correct"]
        and unit["unit_status"] not in {"unknown_requested_unit", "incompatible", "numeric_contract_invalid"}
    )
    grades = {}
    if usable:
        for arm in study.outcomes.ARMS:
            try:
                grades[arm] = grade_teacher(question, reference, answers[arm])
            except (ValueError, TypeError, ArithmeticError, TimeoutError) as error:
                grades[arm] = {"correct": False, "status": "grading_error", "error_type": type(error).__name__}
    return usable, {"self_grade": self_grade, "unit_contract": unit}, grades


def _label_item(item):
    identity, question, reference, answers, benchmark_reference = item[:5]
    answer_format = item[5] if len(item) == 6 else "general"
    usable, contract, grades = label_pair(question, reference, answers, answer_format=answer_format)
    try:
        audit = arithmetic_answers.grade_reference(question, benchmark_reference, r"\boxed{" + reference + "}")
    except (ValueError, TypeError, ArithmeticError, TimeoutError) as error:
        audit = {"status": "grading_error", "correct": False, "error_type": type(error).__name__}
    return {"id": identity, "usable": usable, "contract": contract, "grades": grades, "audit": audit}


def checked_closure(path, directory, inputs, collection):
    """Accept an explicit end to acquisition without rewriting its receipts."""
    path, directory = Path(path), Path(directory)
    closure = study.read(path)
    ids = [row["id"] for row in inputs]
    completed = [identity for identity in ids if identity in collection["completed_attempts"]]
    study.require(
        closure.get("format") == "sie-arithmetic-teacher-closure-v1"
        and closure.get("plan_sha256") == file_hash(directory / "plan.json")
        and closure.get("collection_sha256") == file_hash(directory / "collection.json")
        and closure.get("completed_ids") == completed
        and closure.get("missing_ids") == [identity for identity in ids if identity not in completed]
        and closure.get("physical_attempts") == collection["physical_attempts"]
        and closure.get("acquisition_stopped") is True
        and closure.get("additional_requests_authorized") is False
        and closure.get("eligibility_basis")
        == "Teacher completion and reference-only parsing; no solver or gold outcomes."
        and isinstance(closure.get("reason"), str)
        and bool(closure["reason"].strip()),
        "Teacher acquisition closure does not match the preserved collection",
    )
    return closure


def teacher_records(directory, inputs, *, closure_path=None):
    directory = Path(directory)
    plan, collection = (study.read(directory / name) for name in ("plan.json", "collection.json"))
    protocol = arithmetic_teacher.protocol_for(plan.get("answer_format", "general"))
    study.require(
        plan["format"] == arithmetic_teacher.FORMAT
        and plan["settings"] == protocol.SETTINGS
        and plan["ids"] == [row["id"] for row in inputs]
        and plan["questions_sha256"] == digest(inputs)
        and plan["question_only"] is True
        and plan["reference_answers_received"] is False
        and plan["solver_answers_received"] is False
        and collection["plan_sha256"] == file_hash(directory / "plan.json"),
        "Need the settled teacher collection for the exact question-only training population",
    )
    if closure_path is not None:
        checked_closure(closure_path, directory, inputs, collection)
    study.require(
        not collection["pending"] or closure_path is not None,
        "Teacher collection has pending references; complete acquisition or bind an explicit closure receipt",
    )
    completed, missing = {}, []
    for row in inputs:
        history = arithmetic_teacher.attempt_history(directory, row)
        valid = [(path, record) for path, _, record in history if record and record["status"] == "complete"]
        if not valid:
            study.require(row["id"] not in collection["completed_attempts"], "Teacher completion mapping differs")
            missing.append(row["id"])
            continue
        path, record = valid[0]
        study.require(
            collection["completed_attempts"].get(row["id"]) == str(path.relative_to(directory)),
            "Use the first schema-valid completed teacher answer, independent of correctness",
        )
        completed[row["id"]] = {
            "answer": record["answer"],
            "record_sha256": file_hash(path / "records" / f"{row['id']}.json"),
            "raw_response_sha256": record["raw_response_sha256"],
        }
    study.require(len(completed) == collection["complete"], "Teacher collection count differs")
    return completed, missing, plan, collection


def fit(source, run, embeddings, answers, teacher, *, workers=8, teacher_closure=None):
    source, run, embeddings, answers, teacher = map(Path, (source, run, embeddings, answers, teacher))
    study.unopened(run)
    for name in ("teacher-models.json", "teacher-labels.json", "frozen.json"):
        if (run / name).exists():
            raise FileExistsError(run / name)
    study.require(type(workers) is int and 1 <= workers <= 16, "Use 1..16 grading processes")
    study.checked_plan(source, run)
    context = study.training(source, run, 2000)
    records, missing, teacher_plan, collection = teacher_records(
        teacher, context["inputs"], closure_path=teacher_closure
    )
    source_hashes = {
        "arithmetic_labels.py": file_hash(Path(__file__)),
        "arithmetic_answers.py": file_hash(Path(arithmetic_answers.__file__)),
        "arithmetic_optimizer.py": file_hash(Path(arithmetic_optimizer.__file__)),
        "arithmetic_teacher.py": file_hash(Path(arithmetic_teacher.__file__)),
        Path(arithmetic_teacher.protocol_for(teacher_plan.get("answer_format", "general")).__file__).name: file_hash(
            Path(arithmetic_teacher.protocol_for(teacher_plan.get("answer_format", "general")).__file__)
        ),
        "teacher_format.py": file_hash(Path(teacher_format.__file__)),
    }
    design = {
        "format": "sie-arithmetic-teacher-comparison-plan-v1",
        "protocol": PROTOCOL,
        "study_plan_sha256": file_hash(run / "plan.json"),
        "teacher_plan_sha256": file_hash(teacher / "plan.json"),
        "teacher_collection_sha256": file_hash(teacher / "collection.json"),
        "teacher_closure_sha256": file_hash(teacher_closure) if teacher_closure is not None else None,
        "answer_format": teacher_plan.get("answer_format", "general"),
        "source_sha256": source_hashes,
    }
    write(run / "teacher-comparison-plan.json", design)
    study.checked_source(source, include_gold=True)
    gold = study.read(source / "gold-for-scoring-only.json")
    jobs = []
    for row, benchmark in zip(context["inputs"], context["rows"], strict=True):
        identity = row["id"]
        if identity not in records:
            continue
        actual = {}
        for arm in study.outcomes.ARMS:
            path = answers / f"{identity}-{arm}.json"
            study.require(file_hash(path) == benchmark[f"{arm}_record_sha256"], "Paired training answer changed")
            actual[arm] = study.read(path)["answer"]
        jobs.append(
            (
                identity,
                row["request"],
                records[identity]["answer"],
                actual,
                gold[identity],
                teacher_plan.get("answer_format", "general"),
            )
        )
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        result = list(executor.map(_label_item, jobs, chunksize=8) if executor else map(_label_item, jobs))
    finally:
        if executor:
            executor.shutdown()
    unresolved = [
        {"id": item["id"], "arm": arm, "grade": grade}
        for item in result
        if item["usable"]
        for arm, grade in item["grades"].items()
        if grade["status"] in study.UNRESOLVED
    ]
    write(
        run / "teacher-labels.json",
        {
            **design,
            "format": "sie-arithmetic-teacher-labels-v1",
            "teacher_records": records,
            "missing_teacher_ids": missing,
            "rows": result,
            "unresolved_solver_comparisons": unresolved,
        },
    )
    study.require(not unresolved, "Unresolved teacher/solver comparison; retain all rows and review before fitting")
    eligible = {item["id"]: item for item in result if item["usable"]}
    positions = [i for i, row in enumerate(context["inputs"]) if row["id"] in eligible]
    study.require(bool(positions), "No usable independent teacher references")
    inputs = [context["inputs"][i] for i in positions]
    baseline_rows = [context["rows"][i] for i in positions]
    teacher_rows = [
        {
            "id": row["id"],
            **{f"{arm}_correct": bool(eligible[row["id"]]["grades"][arm]["correct"]) for arm in study.outcomes.ARMS},
        }
        for row in inputs
    ]
    matrix = vectors(inputs, embeddings)
    groups = [context["groups"][i] for i in positions]
    folds = [context["folds"][i] for i in positions]
    costs = [[row[f"{arm}_credits"] for arm in study.outcomes.ARMS] for row in baseline_rows]
    heads = {}
    for name, rows in zip(study.OPTIONAL_METHODS, (teacher_rows, baseline_rows), strict=True):
        print(f"Fit {name} on {len(rows)} matched source-family rows", flush=True)
        heads[name] = arithmetic_optimizer.fit_head(
            matrix,
            rows,
            groups,
            costs,
            fold_assignments=folds,
            progress=lambda value: print(value, flush=True),
        )
    audit_rows = [item for item in result if item["audit"]["status"] == "graded"]
    summary = {
        "teacher_model": teacher_plan["settings"]["model"],
        "training_pool": len(context["inputs"]),
        "completed_references": len(records),
        "missing_references": len(missing),
        "teacher_closure_sha256": design["teacher_closure_sha256"],
        "usable": len(inputs),
        "excluded": [{"id": identity, "reason": "missing_teacher_answer"} for identity in missing]
        + [
            {"id": item["id"], "reason": "reference_not_independently_gradeable", "contract": item["contract"]}
            for item in result
            if not item["usable"]
        ],
        "teacher_correct": sum(item["audit"]["correct"] for item in audit_rows),
        "teacher_audit_gradeable": len(audit_rows),
        "teacher_audit_unresolved": len(result) - len(audit_rows),
        "label_agreement_by_arm": {
            arm: sum(
                a[f"{arm}_correct"] == b[f"{arm}_correct"] for a, b in zip(teacher_rows, baseline_rows, strict=True)
            )
            / len(inputs)
            for arm in study.outcomes.ARMS
        },
        "signed_benefit_agreement": sum(
            int(a["strong_correct"]) - int(a["cheap_correct"]) == int(b["strong_correct"]) - int(b["cheap_correct"])
            for a, b in zip(teacher_rows, baseline_rows, strict=True)
        )
        / len(inputs),
        "known_teacher_label_cost_usd": collection["known_list_price_usd"],
        "budget_exposure": collection["budget"],
        "cost_scope": collection["cost_scope"],
        "scope": PROTOCOL["scope"],
    }
    output = {
        "format": "sie-arithmetic-teacher-models-v1",
        "study_plan_sha256": file_hash(run / "plan.json"),
        "training_grades_sha256": file_hash(run / "grades-train.json"),
        "teacher_comparison_plan_sha256": file_hash(run / "teacher-comparison-plan.json"),
        "teacher_labels_sha256": file_hash(run / "teacher-labels.json"),
        "source_sha256": source_hashes,
        "heads": heads,
        "summary": summary,
        "embedding_files_sha256": {
            name: file_hash(embeddings / name) for name in ("plan.json", "vectors.jsonl.gz", "collection.json")
        },
    }
    write(run / "teacher-models.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "run", "embeddings", "answers", "teacher"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--teacher-closure", type=Path)
    args = parser.parse_args()
    result = fit(
        args.source,
        args.run,
        args.embeddings,
        args.answers,
        args.teacher,
        workers=args.workers,
        teacher_closure=args.teacher_closure,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
