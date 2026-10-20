"""Teacher targets retain independent provenance; no network or real outcomes."""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import benefit
import teacher_outcomes as teacher
from data import file_hash, write


class QuestionOnlyLabelsTests(unittest.TestCase):
    def test_source_accuracy_audit_preserves_symbolic_proof_and_unresolved_coverage(self):
        rule = {
            "original": "old",
            "canonical": "proved",
            "proof": "independent derivation",
            "answers": {"wrong": False},
        }
        item = {
            "id": "synthetic",
            "semantic_kind": "reviewed_symbolic_training",
            "original_reference": "old",
            "reference": "proved",
        }
        with patch.dict(teacher.benefit_training_review.REVIEW, {"synthetic": rule}):
            self.assertTrue(teacher.audit_teacher_reference(item, "proved")["correct"])
            self.assertFalse(teacher.audit_teacher_reference(item, "wrong")["correct"])
            self.assertEqual(teacher.audit_teacher_reference(item, "unsupported")["status"], "grading_error")
            with self.assertRaises(ValueError):
                teacher.audit_teacher_reference(item | {"reference": "changed"}, "proved")
        summary = teacher.teacher_accuracy_summary(
            [
                {"id": "correct", "teacher_correct": True, "teacher_accuracy_gradeable": True},
                {"id": "wrong", "teacher_correct": False, "teacher_accuracy_gradeable": True},
                {"id": "unknown", "teacher_correct": False, "teacher_accuracy_gradeable": False},
            ]
        )
        self.assertEqual(summary["teacher_accuracy_denominator"], 2)
        self.assertEqual(summary["teacher_accuracy_unresolved_ids"], ["unknown"])
        self.assertEqual(summary["teacher_confirmed_correct_fraction_lower_bound"], 1 / 3)

    def test_missing_reference_is_allowed_only_for_source_excluded_questions(self):
        acquisition = {"inputs": [{"id": "training"}, {"id": "excluded"}]}
        collection = {
            "n": 2,
            "complete": 1,
            "returned": 1,
            "status": "incomplete",
            "record_hashes": {"training": "a" * 64},
            "missing_ids": ["excluded"],
        }
        review = {"records": [{"id": "training", "decision": "retain"}, {"id": "excluded", "decision": "exclude"}]}
        rows = [{"id": "training"}]
        result = teacher.checked_teacher_coverage(acquisition, collection, rows, review)
        self.assertEqual(result["missing_ids"], ["excluded"])
        for changed in (collection | {"missing_ids": []}, collection | {"returned": 2}):
            with self.assertRaises(ValueError):
                teacher.checked_teacher_coverage(acquisition, changed, rows, review)
        review["records"][1]["decision"] = "retain"
        with self.assertRaises(ValueError):
            teacher.checked_teacher_coverage(acquisition, collection, rows, review)

    def test_teacher_can_disagree_with_the_benchmark_without_fallback(self):
        usable, _, grades = teacher.label_pair("What is 2+2?", "5", {"cheap": r"\boxed{5}", "strong": r"\boxed{4}"})
        self.assertTrue(usable)
        self.assertTrue(grades["cheap"]["correct"])
        self.assertFalse(grades["strong"]["correct"])

    def test_ratio_and_count_formatting_come_only_from_the_question(self):
        for question, reference, response in (
            ("What is the ratio of the sides?", "3:4", r"\boxed{3/4}"),
            ("How many games were played?", "96", r"\boxed{96\text{games}}"),
            ("What percentage is shaded?", "25", r"\boxed{25\%}"),
        ):
            with self.subTest(question=question):
                usable, _, grades = teacher.label_pair(question, reference, dict.fromkeys(teacher.ARMS, response))
                self.assertTrue(usable)
                self.assertTrue(all(value["correct"] for value in grades.values()))

    def test_unusable_reference_never_becomes_a_label(self):
        with patch.object(
            teacher, "grade_teacher_reference", return_value={"correct": False, "status": "gold_unparseable"}
        ):
            usable, _, _ = teacher.label_pair("question", "ambiguous", dict.fromkeys(teacher.ARMS, "answer"))
        self.assertFalse(usable)

    def test_exact_small_difference_does_not_become_false_equality(self):
        usable, _, grades = teacher.label_pair(
            "Give the exact value.", "1/16", {"cheap": r"\boxed{1/16+1/2^{2000}}", "strong": r"\boxed{1/16}"}
        )
        self.assertTrue(usable)
        self.assertFalse(grades["cheap"]["correct"])
        self.assertTrue(grades["strong"]["correct"])

    def test_unknown_reference_units_are_masked_without_looking_at_solver_answers(self):
        question = "Find x."
        teacher_final = r"7\text{ widgets}"
        for answers in (
            dict.fromkeys(teacher.ARMS, r"\boxed{7\text{ widgets}}"),
            dict.fromkeys(teacher.ARMS, r"\boxed{8}"),
        ):
            usable, self_grade, _ = teacher.label_pair(question, teacher_final, answers)
            self.assertFalse(usable)
            self.assertEqual(self_grade["reference_unit_contract"]["unit_status"], "unknown_requested_unit")
        usable, _, _ = teacher.label_pair("How many widgets?", teacher_final, dict.fromkeys(teacher.ARMS, r"\boxed{8}"))
        self.assertTrue(usable)

    def test_prediction_parse_failure_does_not_select_training_rows(self):
        replies = [
            {"correct": True, "status": "graded"},
            {"correct": False, "status": "prediction_unparseable"},
            {"correct": False, "status": "unit_contract_unresolved"},
        ]
        with patch.object(teacher, "grade_teacher_reference", side_effect=replies):
            usable, _, grades = teacher.label_pair("question", "7", dict.fromkeys(teacher.ARMS, "answer"))
        self.assertTrue(usable)
        self.assertFalse(any(value["correct"] for value in grades.values()))
        with self.assertRaisesRegex(ValueError, "Unresolved teacher comparison"):
            teacher.require_comparable_targets("synthetic", usable, grades)
        grades["strong"]["status"] = "no_final_answer"
        teacher.require_comparable_targets("synthetic", usable, grades)


class LabelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.ids = [f"row{i}" for i in range(20)]
        groups = [f"paper{i % 10}" for i in range(20)]
        self.plan = {
            "teacher_plan_sha256": "a" * 64,
            "benchmark_training_grades_sha256": "b" * 64,
            "outcome_grading": {"source": "v2"},
            "design": {
                "training_ids": self.ids,
                "training_paper_groups": groups,
                "training_folds": benefit.group_folds(groups),
            },
        }
        write(self.run / "plan.json", self.plan)
        rows = [
            {
                "id": identity,
                "usable": True,
                "cheap_correct": i % 2 == 0,
                "strong_correct": i % 3 == 0,
                "teacher_self_grade": {
                    "correct": True,
                    "status": "graded",
                    "reference_unit_contract": {"unit_status": "none"},
                },
                **{f"{arm}_record_sha256": f"{i:064x}" for arm in teacher.ARMS},
            }
            for i, identity in enumerate(self.ids)
        ]
        for row in rows:
            row.update({f"{arm}_grade": {"correct": row[f"{arm}_correct"], "status": "graded"} for arm in teacher.ARMS})
        benchmark = copy.deepcopy(rows)
        for row in benchmark:
            row["cheap_correct"] = not row["cheap_correct"]
            row["strong_correct"] = not row["strong_correct"]
        self.labels = {
            "format": teacher.LABEL_FORMAT,
            "status": "complete",
            "n": len(self.ids),
            "usable_rows": len(self.ids),
            "eligible_ids": self.ids,
            "unusable_ids": [],
            "eligibility_rule": teacher.COMPARISON["eligibility"],
            "plan_sha256": file_hash(self.run / "plan.json"),
            "teacher_plan_sha256": "a" * 64,
            "benchmark_training_grades_sha256": "b" * 64,
            "reference_source": "independent_question_only_teacher",
            "benchmark_values_used_for_teacher_targets": False,
            "outcome_grading": self.plan["outcome_grading"],
            "rows": rows,
            "matched_benchmark_rows": benchmark,
        }

    def save_labels(self, document):
        (self.run / "teacher-labels.json").unlink(missing_ok=True)
        write(self.run / "teacher-labels.json", benefit.seal(document))

    def make_unusable(self, document, index):
        document["rows"][index]["usable"] = False
        document["rows"][index]["teacher_self_grade"] = {
            "correct": False,
            "status": "gold_unparseable",
            "reference_unit_contract": {"unit_status": "none"},
        }
        document["usable_rows"] -= 1
        document["eligible_ids"] = [row["id"] for row in document["rows"] if row["usable"]]
        document["unusable_ids"] = [row["id"] for row in document["rows"] if not row["usable"]]

    def test_full_and_explicit_matched_subset_are_accepted(self):
        self.save_labels(self.labels)
        self.assertEqual(teacher.checked_labels(self.run, self.plan)["usable_rows"], 20)
        subset = copy.deepcopy(self.labels)
        self.make_unusable(subset, 3)
        self.save_labels(subset)
        result = teacher.checked_labels(self.run, self.plan)
        self.assertEqual(result["usable_rows"], 19)
        matched = teacher._matched(result, self.plan["design"])
        self.assertEqual([row["id"] for row in matched["teacher"]], [row["id"] for row in matched["benchmark"]])
        self.assertEqual(matched["folds"], [v for i, v in enumerate(self.plan["design"]["training_folds"]) if i != 3])

    def test_silent_subset_or_benchmark_fallback_is_rejected(self):
        for change in (
            {"status": "incomplete"},
            {"usable_rows": 19},
            {"unusable_ids": ["row0"]},
            {"reference_source": "benchmark_reference"},
            {"benchmark_values_used_for_teacher_targets": True},
            {"teacher_plan_sha256": "b" * 64},
            {"rows": self.labels["rows"][:-1]},
            {"rows": list(reversed(self.labels["rows"]))},
            {"matched_benchmark_rows": list(reversed(self.labels["matched_benchmark_rows"]))},
        ):
            with self.subTest(change=list(change)):
                self.save_labels(self.labels | change)
                with self.assertRaises(ValueError):
                    teacher.checked_labels(self.run, self.plan)

    def test_usable_mask_cannot_depend_on_solver_correctness(self):
        value = copy.deepcopy(self.labels)
        self.make_unusable(value, 3)
        value["rows"][3]["teacher_self_grade"] = {
            "correct": True,
            "status": "graded",
            "reference_unit_contract": {"unit_status": "none"},
        }
        self.save_labels(value)
        with self.assertRaises(ValueError):
            teacher.checked_labels(self.run, self.plan)

    def test_replay_enforces_the_independent_reference_unit_mask(self):
        value = copy.deepcopy(self.labels)
        self.make_unusable(value, 3)
        value["rows"][3]["teacher_self_grade"] = {
            "correct": True,
            "status": "graded",
            "reference_unit_contract": {"unit_status": "unknown_requested_unit"},
        }
        self.save_labels(value)
        self.assertEqual(teacher.checked_labels(self.run, self.plan)["usable_rows"], 19)
        value["rows"][3]["teacher_self_grade"]["reference_unit_contract"]["unit_status"] = "none"
        self.save_labels(value)
        with self.assertRaises(ValueError):
            teacher.checked_labels(self.run, self.plan)

    def test_existing_access_or_outputs_block_mutations(self):
        for name in ("test-access.json", "report.json", "selected-model.json"):
            path = self.run / name
            path.write_text("{}")
            with self.assertRaises(FileExistsError):
                teacher.unopened(self.run, "selected-model.json")
            path.unlink()

    def test_fit_uses_identical_subset_features_folds_costs_and_separate_labels(self):
        embeddings, training = self.run / "vectors", self.run / "training"
        for path in (
            embeddings / "plan.json",
            embeddings / "vectors.jsonl.gz",
            training / "cost-proxy.json",
            training / "grades-train.json",
            self.run / "prices.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic")
        self.plan |= {
            "training_embeddings_sha256": {
                name: file_hash(embeddings / name) for name in ("plan.json", "vectors.jsonl.gz")
            },
            "cost_proxy_sha256": file_hash(training / "cost-proxy.json"),
            "benchmark_training_grades_sha256": file_hash(training / "grades-train.json"),
            "price_snapshot_sha256": file_hash(self.run / "prices.json"),
            "reference_plan": {},
            "teacher_model": "synthetic-teacher",
        }
        labels = copy.deepcopy(self.labels)
        labels["benchmark_training_grades_sha256"] = self.plan["benchmark_training_grades_sha256"]
        self.make_unusable(labels, 3)
        self.save_labels(labels)
        costs = [[10 + i % 2, 50 + i % 3] for i in range(20)]
        matrix = np.random.default_rng(89).normal(size=(20, 2))
        captures = []
        real_fit_head = teacher.outcomes.fit_head

        def fit_head(x, rows, groups, passed_costs, *, progress, fold_assignments):
            captures.append((x.copy(), rows, groups, passed_costs, fold_assignments))
            return real_fit_head(x, rows, groups, passed_costs, progress=progress, fold_assignments=fold_assignments)

        def synthetic_logistic(x, targets, penalty):
            del penalty
            frequency = float(targets.mean())
            return {
                "weights": [0.0] * x.shape[1],
                "bias": math.log(frequency / (1 - frequency)),
                "optimizer": {"success": True},
            }

        protocol = teacher.outcomes.PROTOCOL | {"dimensions": 2}
        with (
            patch.object(teacher.outcomes, "PROTOCOL", protocol),
            patch.object(teacher, "checked_plan", return_value=self.plan),
            patch.object(teacher, "checked_labels", return_value=labels),
            patch.object(teacher, "_training_files", return_value=[{"id": v, "request": "question"} for v in self.ids]),
            patch.object(teacher.outcomes, "_training", return_value=([], {"rows": labels["matched_benchmark_rows"]})),
            patch.object(teacher.outcomes, "cost_proxy", return_value=(costs, {"must_not_reuse": True}, {})),
            patch.object(teacher, "vectors", return_value=matrix),
            patch.object(teacher.outcomes, "fit_head", side_effect=fit_head),
            patch.object(teacher.outcomes, "fit_logistic", side_effect=synthetic_logistic),
        ):
            selected = teacher.fit(self.run, training, embeddings, self.run / "prices.json")
            self.assertEqual(selected["training_rows"], 19)
            self.assertEqual(len(captures), 2)
            np.testing.assert_array_equal(captures[0][0], captures[1][0])
            self.assertEqual(captures[0][2:], captures[1][2:])
            self.assertNotEqual(captures[0][1], captures[1][1])
            self.assertEqual(selected["head"]["training_ids"], selected["matched_benchmark_control"]["training_ids"])
            self.assertNotEqual(
                selected["head"]["conditional_cost_statistics"],
                selected["matched_benchmark_control"]["conditional_cost_statistics"],
            )
            write(self.run / "selected-model.json", selected)
            self.assertEqual(teacher.checked_model(self.run), selected)
            corrupted = copy.deepcopy(selected)
            corrupted["matched_benchmark_control"]["targets"][0][0] ^= 1
            (self.run / "selected-model.json").unlink()
            write(self.run / "selected-model.json", benefit.seal(corrupted))
            with self.assertRaises(ValueError):
                teacher.checked_model(self.run)


if __name__ == "__main__":
    unittest.main()
