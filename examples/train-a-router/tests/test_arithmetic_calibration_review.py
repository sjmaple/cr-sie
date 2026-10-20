import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import arithmetic_calibration_review as review


class CalibrationReviewTests(unittest.TestCase):
    def fixture(self):
        flag = {
            "id": "example",
            "arm": "strong",
            "status": "unit_contract_unresolved",
            "question": "How much of each?",
            "reference": "30",
            "final_answer": r"30 \text{ of each}",
            "record_sha256": "a" * 64,
        }
        row = {
            "id": "example",
            "strong_correct": False,
            "strong_record_sha256": "a" * 64,
            "strong_grade": {"correct": False, "status": flag["status"], "final_answer": flag["final_answer"]},
        }
        decision = {
            "id": "example",
            "arm": "strong",
            "record_sha256": "a" * 64,
            "question_sha256": review.text_hash(flag["question"]),
            "reference_sha256": review.text_hash("30"),
            "final_answer_sha256": review.text_hash(flag["final_answer"]),
            "correct": True,
            "reason": "The equal-share arithmetic gives 30 for each requested category.",
        }
        return row, flag, decision

    def test_exact_unresolved_review_retains_original_result(self):
        row, flag, decision = self.fixture()
        original = copy.deepcopy(row["strong_grade"])
        review.apply_decision(row, flag, decision, "30")
        self.assertTrue(row["strong_correct"])
        self.assertEqual(row["strong_grade"]["base_grade"], original)

    def test_resolved_or_changed_record_is_not_reviewable(self):
        for key, value in (("record_sha256", "b" * 64), ("final_answer_sha256", "c" * 64), ("correct", 1)):
            row, flag, decision = self.fixture()
            decision[key] = value
            with self.assertRaises(ValueError):
                review.apply_decision(row, flag, decision, "30")
        row, flag, decision = self.fixture()
        row["strong_grade"]["status"] = "graded"
        with self.assertRaises(ValueError):
            review.apply_decision(row, flag, decision, "30")

    def test_repeat_validation_preserves_first_scan_and_uses_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "plan.json").write_text("{}")
            original = run / "unresolved-calibration.json"
            original.write_text('{"cases": ["reviewed"]}')
            manifest = {"decisions": [{}], "original_scan_sha256": review.file_hash(original)}
            before = original.read_bytes()

            def base_grade(source, scratch, collection, prices, split, workers):
                self.assertNotEqual(scratch, run)
                self.assertEqual((scratch / "plan.json").read_bytes(), (run / "plan.json").read_bytes())
                (scratch / "unresolved-calibration.json").write_bytes(before)
                raise ValueError("1 unresolved comparisons; retain every row and review the complete list")

            with patch.object(review.study, "grade", side_effect=base_grade):
                actual = review.verified_scan(run, run, run, run, manifest, 1)
            self.assertEqual(actual, original)
            self.assertEqual(original.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
