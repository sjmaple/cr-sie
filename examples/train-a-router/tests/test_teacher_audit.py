"""Synthetic saved-response tests; no API calls or real evaluation outcomes."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import teacher_audit
from data import digest, file_hash, write


class TeacherAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.teacher, self.training = self.root / "teacher", self.root / "training"
        self.identity = "e2h-amc-train-1"
        self.row = {"id": self.identity, "request": "What is one?"}
        inputs, gold = {"train": [self.row]}, {self.identity: "1"}
        self.settings = {"arms": {arm: {"model": arm, "revision": "pinned"} for arm in ("cheap", "strong")}}
        self.training_plan = {
            "gold_sha256": digest(gold),
            "baseline_data_sha256": "baseline",
            "source": {"source_hashes": {"source": "pinned"}},
        }
        for name, value in (
            ("inputs.json", inputs),
            ("gold-for-scoring-only.json", gold),
            ("plan.json", self.training_plan),
            ("answers-train/collection-plan.json", {}),
            ("answers-train/collection.json", {}),
        ):
            write(self.training / name, value)
        record_hashes = {}
        for arm, answer in (("cheap", "0"), ("strong", "1")):
            record = {
                "id": self.identity,
                "arm": arm,
                "request_sha256": digest(self.row["request"]),
                "settings_sha256": digest(self.settings),
                "model": arm,
                "catalog_revision": "pinned",
                "status": "complete",
                "calls": [{"billing": "self_hosted"}],
                "answer": r"\boxed{" + answer + "}",
                "observed_credits": 0,
                "credits": None,
                "has_unknown_charges": False,
            }
            path = self.training / "answers-train" / f"{self.identity}-{arm}.json"
            write(path, record)
            record_hashes[f"{arm}_record_sha256"] = file_hash(path)
        write(
            self.training / "grades-train.json",
            {
                "split": "train",
                "plan_sha256": file_hash(self.training / "plan.json"),
                "collection_plan_sha256": file_hash(self.training / "answers-train/collection-plan.json"),
                "collection_summary_sha256": file_hash(self.training / "answers-train/collection.json"),
                "rows": [{"id": self.identity, "cheap_correct": True, "strong_correct": False, **record_hashes}],
            },
        )
        self.plan = {
            "format": "sie-teacher-reference-audit-v1",
            "n": 1,
            "model": "teacher",
            "inputs": [self.row],
            "teacher_receives_gold": False,
            "teacher_receives_solver_answers": False,
            "fit_router_on_this_pilot": False,
            "settings": {"model": "teacher", "input": "<question>"},
            "training_inputs_sha256": file_hash(self.training / "inputs.json"),
            "baseline_data_sha256": "baseline",
            "prices_usd_per_million": {"input": 10, "cached_input": 1, "input_cache_write": 12.5, "output": 50},
        }
        write(self.teacher / "plan.json", self.plan)
        record = {
            "id": self.identity,
            "request_sha256": digest(self.plan["settings"] | {"input": self.row["request"]}),
            "status": "complete",
            "response_status": "completed",
            "returned_model": "teacher",
            "answer": "1",
            "output_text": '{"answer":"1"}',
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "list_price_estimate_usd": 0.00011,
        }
        write(self.teacher / "records" / f"{self.identity}.json", record)
        write(
            self.teacher / "collection.json",
            {
                "status": "complete",
                "n": 1,
                "returned": 1,
                "complete": 1,
                "plan_sha256": file_hash(self.teacher / "plan.json"),
                "record_hashes": {self.identity: file_hash(self.teacher / "records" / f"{self.identity}.json")},
                "list_price_estimate_usd": 0.00011,
            },
        )
        old_arm = {"reference_correct": True, "teacher_label": True, "usable": True}
        write(
            self.teacher / "report.json",
            {
                "teacher_correct": 1,
                "list_price_estimate_usd": 0.00011,
                "provenance": {
                    "plan_sha256": file_hash(self.teacher / "plan.json"),
                    "collection_sha256": file_hash(self.teacher / "collection.json"),
                    "training_grades_sha256": file_hash(self.training / "grades-train.json"),
                },
                "rows": [
                    {
                        "id": self.identity,
                        "teacher_correct": True,
                        "arms": {"cheap": dict(old_arm), "strong": dict(old_arm)},
                    }
                ],
            },
        )
        self.check = patch.object(
            teacher_audit.benefit_data, "checked_training", return_value=(self.training_plan, inputs, self.settings)
        )
        self.check.start()
        self.addCleanup(self.check.stop)

    def report(self):
        return teacher_audit.summarize(self.teacher, self.training, self.teacher / "report.json")

    def change(self, path, update):
        value = json.loads(path.read_text())
        update(value)
        path.write_text(json.dumps(value))

    def test_regrades_actual_answers_and_preserves_original_report(self):
        original = (self.teacher / "report.json").read_bytes()
        report = self.report()
        self.assertEqual(report["teacher_correct"], 1)
        self.assertEqual(report["teacher_accuracy_denominator"], 1)
        self.assertEqual(report["arms"]["cheap"]["reference_correct"], 0)
        self.assertEqual(report["arms"]["strong"]["reference_correct"], 1)
        self.assertEqual(report["signed_benefit_labels"]["agree"], 1)
        self.assertEqual(report["list_price_estimate_usd"], 0.00011)
        self.assertEqual(report["raw_to_amended"]["changed_questions"], 1)
        self.assertEqual((self.teacher / "report.json").read_bytes(), original)

    def test_teacher_record_tampering_is_rejected(self):
        path = self.teacher / "records" / f"{self.identity}.json"
        self.change(path, lambda value: value.update(answer="2"))
        with self.assertRaisesRegex(ValueError, "raw record changed"):
            self.report()

    def test_paired_answer_tampering_is_rejected(self):
        path = self.training / "answers-train" / f"{self.identity}-cheap.json"
        self.change(path, lambda value: value.update(answer=r"\boxed{1}"))
        with self.assertRaisesRegex(ValueError, "paired solver answer changed"):
            self.report()

    def test_unparseable_teacher_keeps_accuracy_denominator_and_removes_label_coverage(self):
        with patch.object(
            teacher_audit, "grade_teacher_reference", return_value={"correct": False, "status": "gold_unparseable"}
        ):
            report = self.report()
        self.assertEqual(report["teacher_accuracy_denominator"], 1)
        self.assertEqual(report["teacher_reference_coverage"]["usable"], 0)
        self.assertEqual(report["arms"]["cheap"]["usable_teacher_labels"], 0)
        self.assertEqual(report["signed_benefit_labels"]["covered"], 0)
        self.assertIsNone(report["rows"][0]["arms"]["cheap"]["teacher_label"])

    def test_existing_output_refused_before_regrading(self):
        output = self.teacher / "report-v2.json"
        output.write_text("preserve")
        with (
            patch("sys.argv", ["teacher_audit.py", "--teacher", str(self.teacher), "--training", str(self.training)]),
            patch.object(teacher_audit, "summarize") as summarize,
            self.assertRaises(FileExistsError),
        ):
            teacher_audit.main()
        summarize.assert_not_called()
        self.assertEqual(output.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
