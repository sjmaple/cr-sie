"""Regrade a saved teacher-reference audit on CPU, without any model requests.

The teacher solves only the question. Its answer becomes a reference for each
saved solver answer through question-only formatting rules. Benchmark labels
are independently recomputed from the original source gold using the same
frozen outcome grader; old numeric labels are used only to describe changes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import benefit_data
import solver
from benefit_expanded_gold_v2 import grade_expanded, grade_teacher_reference, outcome_grader_metadata
from data import digest, file_hash, write

_UNUSABLE_REFERENCE = {"gold_unparseable", "grading_error"}


def read(path):
    return json.loads(Path(path).read_text())


def _usage_cost(record, prices):
    usage = record["usage"]
    details = usage.get("input_tokens_details") or {}
    counts = {
        "input": usage["input_tokens"],
        "output": usage["output_tokens"],
        "reasoning": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
        "cached_input": details.get("cached_tokens", 0),
        "cache_write_input": details.get("cache_write_tokens", 0),
    }
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Teacher usage has invalid token counts")
    uncached = counts["input"] - counts["cached_input"] - counts["cache_write_input"]
    if uncached < 0:
        raise ValueError("Teacher cache counts exceed input tokens")
    units = {
        "input": uncached,
        "cached_input": counts["cached_input"],
        "input_cache_write": counts["cache_write_input"],
        "output": counts["output"],
    }
    amount = sum(Decimal(str(prices[name])) * number for name, number in units.items()) / 1_000_000
    if abs(amount - Decimal(str(record["list_price_estimate_usd"]))) > Decimal("0.000000001"):
        raise ValueError("Teacher record token-price calculation differs")
    return counts, amount


def _changes(old, rows):
    if old is None:
        return None
    previous = {row["id"]: row for row in old["rows"]}
    if len(previous) != len(rows) or set(previous) != {row["id"] for row in rows}:
        raise ValueError("Previous report sample differs")
    changed = []
    for row in rows:
        prior, fields = previous[row["id"]], {}
        if prior["teacher_correct"] != row["teacher_correct"]:
            fields["teacher_correct"] = {"old": prior["teacher_correct"], "new": row["teacher_correct"]}
        for arm in ("cheap", "strong"):
            for name in ("reference_correct", "teacher_label", "usable"):
                before, after = prior["arms"][arm][name], row["arms"][arm][name]
                if before != after:
                    fields[f"{arm}.{name}"] = {"old": before, "new": after}
        if fields:
            changed.append({"id": row["id"], "changes": fields})
    return {
        "old_teacher_correct": old["teacher_correct"],
        "new_teacher_correct": sum(row["teacher_correct"] for row in rows),
        "changed_questions": len(changed),
        "rows": changed,
    }


def summarize(teacher: Path, training: Path, previous_report: Path | None = None):
    """Verify saved receipts, then recompute every label in the declared sample."""
    teacher, training = Path(teacher), Path(training)
    plan, collection = read(teacher / "plan.json"), read(teacher / "collection.json")
    training_plan, training_inputs, settings = benefit_data.checked_training(training)
    gold = read(training / "gold-for-scoring-only.json")
    old_grades = read(training / "grades-train.json")
    original_rows = {row["id"]: row for row in old_grades["rows"]}
    input_rows = {row["id"]: row for row in training_inputs["train"]}
    identities = [row["id"] for row in plan["inputs"]]
    n = len(identities)
    if (
        plan.get("format") != "sie-teacher-reference-audit-v1"
        or plan.get("n") != n
        or n == 0
        or len(set(identities)) != n
        or plan.get("teacher_receives_gold") is not False
        or plan.get("teacher_receives_solver_answers") is not False
        or plan.get("fit_router_on_this_pilot") is not False
        or plan["settings"]["model"] != plan["model"]
        or plan["training_inputs_sha256"] != file_hash(training / "inputs.json")
        or plan["baseline_data_sha256"] != training_plan["baseline_data_sha256"]
        or any(input_rows.get(row["id"]) != row for row in plan["inputs"])
    ):
        raise ValueError("Teacher plan does not match the question-only training sample")
    if (
        collection.get("status") != "complete"
        or any(collection.get(key) != n for key in ("n", "returned", "complete"))
        or collection["plan_sha256"] != file_hash(teacher / "plan.json")
        or set(collection["record_hashes"]) != set(identities)
    ):
        raise ValueError("Teacher collection is incomplete or differs from its plan")
    if (
        digest(gold) != training_plan["gold_sha256"]
        or set(gold) != set(input_rows)
        or old_grades.get("split") != "train"
        or old_grades["plan_sha256"] != file_hash(training / "plan.json")
        or old_grades["collection_plan_sha256"] != file_hash(training / "answers-train/collection-plan.json")
        or old_grades["collection_summary_sha256"] != file_hash(training / "answers-train/collection.json")
        or len(original_rows) != len(old_grades["rows"])
        or not set(identities) <= set(original_rows)
    ):
        raise ValueError("Saved training source or paired-answer provenance differs")
    previous_report = Path(previous_report) if previous_report else None
    old = read(previous_report) if previous_report else None
    if old and (
        old["provenance"]["plan_sha256"] != file_hash(teacher / "plan.json")
        or old["provenance"]["collection_sha256"] != file_hash(teacher / "collection.json")
        or old["provenance"]["training_grades_sha256"] != file_hash(training / "grades-train.json")
    ):
        raise ValueError("Previous audit report has different source evidence")

    rows, tokens, cost = [], Counter(), Decimal(0)
    paired_hashes = {}
    for row in plan["inputs"]:
        identity = row["id"]
        teacher_path = teacher / "records" / f"{identity}.json"
        if file_hash(teacher_path) != collection["record_hashes"][identity]:
            raise ValueError("Teacher raw record changed")
        record = read(teacher_path)
        body = plan["settings"] | {"input": row["request"]}
        if (
            record.get("id") != identity
            or record.get("request_sha256") != digest(body)
            or record.get("status") != "complete"
            or record.get("response_status") != "completed"
            or record.get("returned_model") != plan["model"]
            or not isinstance(record.get("answer"), str)
            or read_json_answer(record["output_text"]) != record["answer"]
        ):
            raise ValueError("Teacher answer differs from its request or returned JSON")
        usage, charge = _usage_cost(record, plan["prices_usd_per_million"])
        tokens.update(usage)
        cost += charge
        teacher_answer = record["answer"]
        teacher_grade = grade_expanded(identity, gold[identity], r"\boxed{" + teacher_answer + "}")
        if teacher_grade["status"] in _UNUSABLE_REFERENCE:
            raise ValueError("Reviewed benchmark reference cannot be graded")
        self_grade = grade_teacher_reference(row["request"], teacher_answer, r"\boxed{" + teacher_answer + "}")
        reference_usable = self_grade["status"] not in _UNUSABLE_REFERENCE and self_grade["correct"]
        outcome = {
            "id": identity,
            "teacher_correct": teacher_grade["correct"],
            "teacher_grade": teacher_grade,
            "teacher_answer": teacher_answer,
            "teacher_reference_usable": reference_usable,
            "teacher_reference_self_grade": self_grade,
            "arms": {},
        }
        for arm in ("cheap", "strong"):
            path = training / "answers-train" / f"{identity}-{arm}.json"
            expected_hash = original_rows[identity][f"{arm}_record_sha256"]
            if file_hash(path) != expected_hash:
                raise ValueError("Saved paired solver answer changed")
            paired_hashes[f"{identity}-{arm}"] = expected_hash
            answer = read(path)
            solver.validate_record(answer, row, arm, settings)
            if answer["status"] != "complete":
                raise ValueError("Saved paired solver answer is incomplete")
            reference = grade_expanded(identity, gold[identity], answer["answer"])
            if reference["status"] in _UNUSABLE_REFERENCE:
                raise ValueError("Reviewed benchmark reference cannot grade a paired answer")
            derived = grade_teacher_reference(row["request"], teacher_answer, answer["answer"])
            usable = reference_usable and derived["status"] not in _UNUSABLE_REFERENCE
            outcome["arms"][arm] = {
                "reference_correct": reference["correct"],
                "reference_grade": reference,
                "teacher_label": derived["correct"] if usable else None,
                "derived_grade": derived,
                "usable": usable,
            }
        rows.append(outcome)
    if abs(cost - Decimal(str(collection["list_price_estimate_usd"]))) > Decimal("0.000000001"):
        raise ValueError("Teacher collection cost differs from saved token receipts")
    if old and abs(cost - Decimal(str(old["list_price_estimate_usd"]))) > Decimal("0.000000001"):
        raise ValueError("Regrading changed the teacher acquisition cost")

    arms = {}
    for arm in ("cheap", "strong"):
        observed = [row["arms"][arm] for row in rows]
        covered = [value for value in observed if value["usable"]]
        arms[arm] = {
            "n": n,
            "reference_correct": sum(value["reference_correct"] for value in observed),
            "usable_teacher_labels": len(covered),
            "coverage": len(covered) / n,
            "labels_agree": sum(value["teacher_label"] == value["reference_correct"] for value in covered),
            "teacher_false_correct": sum(
                value["teacher_label"] and not value["reference_correct"] for value in covered
            ),
            "teacher_false_incorrect": sum(
                not value["teacher_label"] and value["reference_correct"] for value in covered
            ),
            "disagreement_ids": [
                row["id"]
                for row in rows
                if row["arms"][arm]["usable"]
                and row["arms"][arm]["teacher_label"] != row["arms"][arm]["reference_correct"]
            ],
        }
    covered = [row for row in rows if all(value["usable"] for value in row["arms"].values())]
    signed_disagreements = [
        row["id"]
        for row in covered
        if (int(row["arms"]["strong"]["teacher_label"]) - int(row["arms"]["cheap"]["teacher_label"]))
        != (int(row["arms"]["strong"]["reference_correct"]) - int(row["arms"]["cheap"]["reference_correct"]))
    ]
    return {
        "format": "sie-teacher-reference-audit-result-v2",
        "n": n,
        "teacher_model": plan["model"],
        "teacher_correct": sum(row["teacher_correct"] for row in rows),
        "teacher_accuracy": sum(row["teacher_correct"] for row in rows) / n,
        "teacher_accuracy_denominator": n,
        "arms": arms,
        "teacher_reference_coverage": {
            "usable": sum(row["teacher_reference_usable"] for row in rows),
            "n": n,
            "unusable_ids": [row["id"] for row in rows if not row["teacher_reference_usable"]],
        },
        "signed_benefit_labels": {
            "covered": len(covered),
            "n": n,
            "agree": len(covered) - len(signed_disagreements),
            "disagreement_ids": signed_disagreements,
        },
        "teacher_incorrect_ids": [row["id"] for row in rows if not row["teacher_correct"]],
        "list_price_estimate_usd": float(cost),
        "acquisition_cost_unchanged": True,
        "tokens": dict(tokens),
        "raw_to_amended": _changes(old, rows),
        "provenance": {
            "plan_sha256": file_hash(teacher / "plan.json"),
            "collection_sha256": file_hash(teacher / "collection.json"),
            "teacher_record_hashes": collection["record_hashes"],
            "paired_record_hashes": paired_hashes,
            "training_plan_sha256": file_hash(training / "plan.json"),
            "training_inputs_sha256": file_hash(training / "inputs.json"),
            "original_gold_file_sha256": file_hash(training / "gold-for-scoring-only.json"),
            "original_gold_sha256": digest(gold),
            "source": training_plan["source"],
            "original_training_grades_sha256": file_hash(training / "grades-train.json"),
            "previous_report_sha256": file_hash(previous_report) if previous_report else None,
            "audit_source_sha256": file_hash(Path(__file__)),
            **benefit_data.source_hashes(),
            **outcome_grader_metadata(),
        },
        "limitations": [
            "This is a fixed training-only sample, not an unseen-test guarantee or a theoretical teacher quality ceiling.",
            "Teacher responses were collected without benchmark gold or paired solver responses. Reference-label construction uses only question context and the teacher/solver finals.",
            "Benchmark labels are freshly recomputed from raw solver responses under the amended grader; old labels are not ground truth for this audit.",
            "Reference coverage measures parseability, not correctness. Unusable references remain in the teacher-accuracy denominator and are excluded explicitly from label-agreement denominators.",
            "The teacher audit exposed source and grading issues; independent source checks preceded this amended report, and the original report is preserved.",
            "No router has been trained on these teacher labels. This audit alone does not establish their downstream routing performance.",
            "Token-price cost is a historical list-price estimate from the saved acquisition plan, not an invoice. Regrading makes no model requests.",
        ],
        "rows": rows,
    }


def read_json_answer(value):
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or set(parsed) != {"answer"} or not isinstance(parsed["answer"], str):
        raise ValueError("Teacher output is not its declared answer schema")
    return parsed["answer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True, help="Saved teacher plan, collection and records")
    parser.add_argument(
        "--training", type=Path, required=True, help="Original prepared training data and paired answers"
    )
    parser.add_argument("--previous-report", type=Path, help="Defaults to the preserved teacher/report.json")
    parser.add_argument("--output", type=Path, help="Defaults to a new teacher/report-v2.json; never overwritten")
    args = parser.parse_args()
    output = args.output or args.teacher / "report-v2.json"
    if output.exists():
        raise FileExistsError(output)
    previous = args.previous_report or args.teacher / "report.json"
    result = summarize(args.teacher, args.training, previous)
    write(output, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("rows", "provenance", "limitations", "raw_to_amended")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
