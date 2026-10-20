import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import arithmetic_labels as labels


class ArithmeticLabelTests(unittest.TestCase):
    def test_explicit_closure_preserves_incomplete_collection_and_exact_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = [{"id": "one", "request": "How many?"}, {"id": "two", "request": "How much?"}]
            plan = {
                "format": labels.arithmetic_teacher.FORMAT,
                "settings": labels.arithmetic_teacher.protocol_for("numeric").SETTINGS,
                "answer_format": "numeric",
                "ids": ["one", "two"],
                "questions_sha256": labels.digest(inputs),
                "question_only": True,
                "reference_answers_received": False,
                "solver_answers_received": False,
            }
            (directory / "plan.json").write_text(json.dumps(plan))
            attempt = directory / "attempts/one/attempt-01"
            (attempt / "records").mkdir(parents=True)
            record = {"status": "complete", "answer": "7", "raw_response_sha256": "a" * 64}
            (attempt / "records/one.json").write_text(json.dumps(record))
            collection = {
                "plan_sha256": labels.file_hash(directory / "plan.json"),
                "complete": 1,
                "completed_attempts": {"one": "attempts/one/attempt-01"},
                "pending": ["two"],
                "physical_attempts": 2,
            }
            collection_path = directory / "collection.json"
            collection_path.write_text(json.dumps(collection))
            original_bytes = collection_path.read_bytes()
            closure = {
                "format": "sie-arithmetic-teacher-closure-v1",
                "plan_sha256": labels.file_hash(directory / "plan.json"),
                "collection_sha256": labels.file_hash(collection_path),
                "completed_ids": ["one"],
                "missing_ids": ["two"],
                "physical_attempts": 2,
                "acquisition_stopped": True,
                "additional_requests_authorized": False,
                "eligibility_basis": "Teacher completion and reference-only parsing; no solver or gold outcomes.",
                "reason": "Provider quota exhausted; preserve all completed references.",
            }
            closure_path = directory / "closure.json"
            closure_path.write_text(json.dumps(closure))
            with self.assertRaisesRegex(ValueError, "pending references"):
                labels.teacher_records(directory, inputs)
            with patch.object(
                labels.arithmetic_teacher,
                "attempt_history",
                side_effect=lambda _, row: [(attempt, {}, record)] if row["id"] == "one" else [],
            ):
                records, missing, _, _ = labels.teacher_records(directory, inputs, closure_path=closure_path)
            self.assertEqual(list(records), ["one"])
            self.assertEqual(missing, ["two"])
            self.assertEqual(collection_path.read_bytes(), original_bytes)
            closure["missing_ids"] = []
            closure_path.write_text(json.dumps(closure))
            with self.assertRaisesRegex(ValueError, "closure does not match"):
                labels.teacher_records(directory, inputs, closure_path=closure_path)

    def test_teacher_targets_do_not_fall_back_to_benchmark_reference(self):
        item = labels._label_item(
            (
                "training",
                "How many apples remain?",
                "7",
                {"cheap": r"\boxed{7}", "strong": r"\boxed{6}"},
                "6",
            )
        )
        self.assertTrue(item["usable"])
        self.assertTrue(item["grades"]["cheap"]["correct"])
        self.assertFalse(item["grades"]["strong"]["correct"])
        self.assertFalse(item["audit"]["correct"])

    def test_unusable_teacher_reference_masks_both_without_comparing_solvers(self):
        with patch.object(
            labels, "grade_teacher", return_value={"correct": False, "status": "gold_unparseable"}
        ) as grade:
            usable, _, comparisons = labels.label_pair(
                "How many apples remain?",
                "Unknown",
                {
                    "cheap": r"\boxed{2}",
                    "strong": r"\boxed{3}",
                },
            )
        self.assertFalse(usable)
        self.assertEqual(comparisons, {})
        self.assertEqual(grade.call_count, 1)

    def test_solver_comparison_exception_cannot_change_reference_eligibility(self):
        with patch.object(
            labels,
            "grade_teacher",
            side_effect=[
                {"correct": True, "status": "graded"},
                ValueError("unsupported"),
                {"correct": False, "status": "graded"},
            ],
        ):
            usable, _, comparisons = labels.label_pair(
                "How many apples remain?",
                "2",
                {
                    "cheap": r"\boxed{odd notation}",
                    "strong": r"\boxed{3}",
                },
            )
        self.assertTrue(usable)
        self.assertEqual(comparisons["cheap"]["status"], "grading_error")
        self.assertFalse(comparisons["strong"]["correct"])

    def test_clock_teacher_accepts_bare_hour_but_preserves_explicit_period_conflict(self):
        question = "What time should the bus depart?"
        self.assertTrue(labels.grade_teacher(question, "2:00 pm", r"\boxed{2}")["correct"])
        self.assertTrue(labels.grade_teacher(question, "2:00 pm", r"\boxed{14:00}")["correct"])
        self.assertFalse(labels.grade_teacher(question, "2:00 pm", r"\boxed{2:00 am}")["correct"])

    def test_wrong_unit_reference_does_not_become_usable(self):
        usable, _, _ = labels.label_pair(
            "Walk 12 miles in 3 hours. How many hours did it take?",
            "3 miles",
            {"cheap": r"\boxed{3}", "strong": r"\boxed{3 hours}"},
        )
        self.assertFalse(usable)

    def test_benchmark_audit_failure_cannot_change_teacher_targets(self):
        with patch.object(
            labels.arithmetic_answers,
            "grade_reference",
            side_effect=[
                {"correct": True, "status": "graded"},
                {"correct": True, "status": "graded"},
                {"correct": False, "status": "graded"},
                ValueError("unsupported benchmark"),
            ],
        ):
            item = labels._label_item(
                (
                    "training",
                    "How many apples remain?",
                    "7",
                    {"cheap": r"\boxed{7}", "strong": r"\boxed{6}"},
                    "unsupported source",
                )
            )
        self.assertTrue(item["usable"])
        self.assertEqual(item["audit"]["status"], "grading_error")
        self.assertTrue(item["grades"]["cheap"]["correct"])

    def test_numeric_contract_rejects_units_prose_and_nonfinite_values(self):
        for value in ("23", "-3.5", "1/4", ".5"):
            self.assertTrue(labels.numeric_reference(value), value)
        for value in ("$23", "23 pens", "23 or 24", "NaN", "1/0", "1e6", "1 000"):
            self.assertFalse(labels.numeric_reference(value), value)

    def test_numeric_reference_is_eligible_without_inferred_requested_unit(self):
        usable, contract, grades = labels.label_pair(
            "An agent buys ads. How much did he spend on buying all the ads?",
            "1375",
            {"cheap": r"\boxed{1375}", "strong": r"\boxed{1374}"},
            answer_format="numeric",
        )
        self.assertTrue(usable)
        self.assertEqual(contract["unit_contract"]["unit_status"], "numeric_requested_units")
        self.assertTrue(grades["cheap"]["correct"])
        self.assertFalse(grades["strong"]["correct"])

    def test_numeric_contract_failure_never_uses_solver_answers(self):
        with patch.object(labels, "grade_teacher", return_value={"correct": True, "status": "graded"}) as grade:
            usable, _, grades = labels.label_pair(
                "How many apples remain?",
                "7 apples",
                {"cheap": r"\boxed{7}", "strong": r"\boxed{7}"},
                answer_format="numeric",
            )
        self.assertFalse(usable)
        self.assertEqual(grades, {})
        self.assertEqual(grade.call_count, 1)


if __name__ == "__main__":
    unittest.main()
