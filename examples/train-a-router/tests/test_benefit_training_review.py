"""Check that symbolic review cannot silently relabel other predictions."""

import unittest

from benefit_training_review import REVIEW, reviewed_grade


class TrainingReviewTests(unittest.TestCase):
    def test_monotonic_power_comparison(self):
        identity = "e2h-amc-eval-1621"
        for final, correct in REVIEW[identity]["answers"].items():
            result = reviewed_grade(identity, REVIEW[identity]["original"], r"\boxed{" + final + "}")
            self.assertIs(result["correct"], correct)
            self.assertEqual(result["status"], "graded")

    def test_tent_map_length_exact_square(self):
        self.assertEqual(4**2012, 2**4024)
        self.assertLess((2**2012) ** 2, 4**2012 + 1)
        identity = "e2h-amc-eval-1775"
        for final, correct in REVIEW[identity]["answers"].items():
            result = reviewed_grade(identity, REVIEW[identity]["original"], r"\boxed{" + final + "}")
            self.assertIs(result["correct"], correct)
            self.assertTrue(result["training_adjudication"]["source_correction"])

    def test_unreviewed_reference_or_answer_rejected(self):
        identity = "e2h-amc-eval-1775"
        with self.assertRaises(ValueError):
            reviewed_grade(identity, "5", "7")
        with self.assertRaises(ValueError):
            reviewed_grade(identity, REVIEW[identity]["original"], "7")


if __name__ == "__main__":
    unittest.main()
