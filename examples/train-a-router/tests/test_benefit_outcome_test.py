import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benefit_calibration
import benefit_outcome_test


class OutcomeAccessTests(unittest.TestCase):
    def test_partial_study_set_never_opens_answers(self):
        with tempfile.TemporaryDirectory() as temporary, patch("solver.grade") as raw_grader:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "both training sizes"):
                benefit_outcome_test.grade(root, root / "calibration.json", [root], [root], root / "grades.json")
            raw_grader.assert_not_called()
            self.assertFalse((root / "grades-access.json").exists())

    def test_invalid_frozen_model_never_opens_answers(self):
        with tempfile.TemporaryDirectory() as temporary, patch("solver.grade") as raw_grader:
            root = Path(temporary)
            calibration = root / "calibration.json"
            calibration.write_text("{}")
            with (
                patch("benefit_outcomes.checked_frozen", side_effect=ValueError("model changed")),
                self.assertRaisesRegex(ValueError, "model changed"),
            ):
                benefit_outcome_test.grade(root, calibration, [root, root], [root, root], root / "grades.json")
            raw_grader.assert_not_called()
            self.assertFalse((root / "grades-access.json").exists())

    def test_calibration_replay_cannot_claim_a_fresh_test(self):
        with patch("benefit_calibration.grade_calibration") as raw_grader:
            for fresh, replay in ((False, False), (True, True)):
                with self.assertRaisesRegex(ValueError, "either a fresh-study"):
                    benefit_calibration.amend("missing", "also-missing", attest_test_outcomes_unopened=fresh, replay=replay)
            raw_grader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
