"""Exact arithmetic and question-only teacher semantics, without model outputs."""

import unittest
from fractions import Fraction
from math import comb
from unittest.mock import patch

import benefit_expanded_gold as previous
import benefit_expanded_gold_v2 as amended
from data import digest


class ExactArithmeticTests(unittest.TestCase):
    def test_equivalent_rational_forms_are_exact(self):
        expected = Fraction(1 + 2**1005, 2**1007) ** 2
        for value in (
            r"\left(\frac{1+2^{1005}}{2^{1007}}\right)^2",
            r"\frac{1}{16}(1+\frac{1}{2^{1005}})^2",
            r"\frac{1}{16}+2^{-1008}+2^{-2014}",
            r"p=\frac{(1+2^{1005})^2}{2^{2014}}",
        ):
            with self.subTest(value=value):
                self.assertEqual(amended.exact_rational(value), expected)

    def test_nearby_values_cannot_match_by_numeric_tolerance(self):
        expected = r"\frac{1}{16}(1+2^{-1005})^2"
        for wrong in (
            "0.0625",
            r"\frac{1}{16}",
            r"\frac{1}{16}(1+2^{-1006})^2",
            r"\frac{1}{16}+2^{-1008}",
            r"\frac{1}{16}(1-2^{-1005})^2",
        ):
            self.assertFalse(amended.exact_comparison(expected, wrong)["correct"], wrong)
        self.assertFalse(amended.exact_comparison("1/3", "0.333333333333333333333333333333")["correct"])
        self.assertTrue(amended.exact_comparison("1/8", ".125")["correct"])

    def test_precedence_decimal_percent_and_braced_arguments(self):
        for expression, expected in (
            ("-2^2", -4),
            ("(-2)^2", 4),
            ("2^3^2", 512),
            ("2^-3", Fraction(1, 8)),
            (r"\frac34", Fraction(3, 4)),
            (r"\frac 34", Fraction(3, 4)),
            (r"\dfrac{12}{16}", Fraction(3, 4)),
            ("2(1+3)", 8),
            (r"75\%", Fraction(3, 4)),
            ("0.1+0.2", Fraction(3, 10)),
            (r"\sqrt{\frac{4}{9}}", Fraction(2, 3)),
            (r"x=\frac34", Fraction(3, 4)),
        ):
            self.assertEqual(amended.exact_rational(expression), expected, expression)

    def test_unsupported_expressions_and_resource_bounds_fail_closed(self):
        for expression in (
            "__import__('os').system('bad')",
            "2=3",
            "1/0",
            "0^-1",
            "0^0",
            "2^1000000000",
            "2^(2^20)",
            "2^(1/2)",
            r"\sqrt2",
            r"\pi",
            "x+1",
            "1,2",
            "3:4",
            "0.1.2",
            "1e-6",
            "",
            "(" * 100 + "1" + ")" * 100,
            "1" * 4097,
        ):
            self.assertIsNone(amended.exact_rational(expression), expression[:60])

    def test_guard_is_generic_and_overrides_a_mocked_tolerant_fallback(self):
        with patch.object(previous, "grade_expanded", return_value={"correct": True, "status": "graded"}):
            result = amended.grade_expanded("synthetic", "1/3", r"\boxed{0.3333333333333}")
        self.assertFalse(result["correct"])
        self.assertEqual(result["exact_numeric_guard"], "applied")

    def test_mixed_numbers_and_digit_spacing_do_not_become_products_or_concatenation(self):
        for expression in (r"1\frac{1}{5}", r"1\frac15", "1 1/5", "1 000", r"1\,1/5", r"\frac1 5"):
            self.assertIsNone(amended.exact_rational(expression), expression)
        self.assertEqual(amended.exact_rational(r"2*\frac12"), 1)
        self.assertEqual(amended.exact_rational(r"2\cdot\frac12"), 1)
        for original in (r"1\frac{1}{5}", "1 1/5"):
            response = r"\boxed{1.2}"
            result = amended.grade_expanded("synthetic", original, response)
            self.assertEqual(result["correct"], previous.grade_expanded("synthetic", original, response)["correct"])
            self.assertIn("unsupported", result["exact_numeric_guard"])
        self.assertTrue(amended.grade_expanded("synthetic", r"1\frac{1}{5}", r"\boxed{1.2}")["correct"])

    def test_symbolic_fallback_and_existing_approximation_are_explicit(self):
        result = amended.grade_expanded("synthetic", r"\sqrt2", r"\boxed{\sqrt{8}/2}")
        self.assertTrue(result["correct"])
        self.assertIn("unsupported", result["exact_numeric_guard"])
        scientific = amended.grade_expanded("synthetic", r"10^{-6}", r"\boxed{1e-6}")
        self.assertIn("unsupported", scientific["exact_numeric_guard"])
        original, _, _ = amended.benefit_data.RULES["e2h-amc-eval-1984"]
        result = amended.grade_expanded("e2h-amc-eval-1984", original, r"\boxed{82.9\%}")
        self.assertTrue(result["correct"])
        self.assertIn("non-exact", result["exact_numeric_guard"])


class SourceAmendmentTests(unittest.TestCase):
    def grade(self, identity, answer):
        return amended.grade_expanded(identity, amended.RULES[identity][0], r"\boxed{" + answer + "}")

    def test_pentagon_binomial_count_and_independent_walk(self):
        counts = [1, 0, 0, 0, 0]
        for _ in range(10):
            counts = [counts[(i - 1) % 5] + counts[(i + 1) % 5] for i in range(5)]
        ways = sum(comb(10, k) for k in range(11) if (2 * k - 10) % 5 == 0)
        self.assertEqual(ways, counts[0])
        self.assertEqual(Fraction(ways, sum(counts)), Fraction(127, 512))
        self.assertTrue(self.grade("e2h-amc-eval-1181", r"\frac{127}{512}")["correct"])
        self.assertFalse(self.grade("e2h-amc-eval-1181", r"\sqrt{\frac{127}{512}}")["correct"])

    def test_chessboard_independent_integer_recurrence(self):
        counts = [1] + [0] * 7
        for _ in range(2012):
            counts = [counts[(i - 1) % 8] + counts[(i + 1) % 8] for i in range(8)]
        observed = Fraction(counts[0], sum(counts)) ** 2
        self.assertEqual(observed, Fraction(1 + 2**1005, 2**1007) ** 2)
        self.assertGreater(observed, Fraction(1, 16))
        self.assertTrue(self.grade("e2h-amc-eval-776", r"\frac1{16}(1+2^{-1005})^2")["correct"])
        self.assertFalse(self.grade("e2h-amc-eval-776", "0.0625")["correct"])
        self.assertFalse(self.grade("e2h-amc-eval-776", "p")["correct"])

    def test_ratio_and_count_units_without_altering_numeric_value(self):
        for answer in ("3:4", "6:8", "3/4"):
            self.assertTrue(self.grade("e2h-amc-eval-2735", answer)["correct"])
        for answer in ("4:3", "3:0", "3:4:5"):
            self.assertFalse(self.grade("e2h-amc-eval-2735", answer)["correct"])
        self.assertEqual(2 * 2 * comb(6, 2) + 6 * 6, 96)
        for answer in ("96", "96 games", r"96\text{ games}", r"96\text{games}", "96games", "96 conference games"):
            self.assertTrue(self.grade("e2h-amc-eval-911", answer)["correct"], answer)
        for answer in ("192 games", "96 teams", "96 games per team", "96+1 games"):
            self.assertFalse(self.grade("e2h-amc-eval-911", answer)["correct"], answer)

    def test_review_binds_both_versions_and_new_source_correction(self):
        gold = {identity: rule[0] for identity, rule in amended.RULES.items()}
        sources = {
            identity: {"answer": value, "problem": "Bound source", "solution": "Exact proof"}
            for identity, value in gold.items()
        }
        corrected, _, review = amended.review_expanded(gold, sources)
        self.assertTrue(review["ready"])
        self.assertEqual(corrected["e2h-amc-eval-1181"], r"\frac{127}{512}")
        self.assertEqual(review["canonical_gold_sha256"], digest(corrected))
        self.assertEqual(review["outcome_grader_protocol_sha256"], digest(amended.OUTCOME_PROTOCOL))
        self.assertEqual(len(review["previous_review_source_sha256"]), 64)
        self.assertEqual(sum(row["source_answer_correction"] for row in review["correction_records"]), 1)
        with self.assertRaisesRegex(ValueError, "original answer"):
            amended.grade_expanded("e2h-amc-eval-911", "95", r"\boxed{96}")


class TeacherContextTests(unittest.TestCase):
    def grade(self, question, reference, answer):
        return amended.grade_teacher_reference(question, reference, r"\boxed{" + answer + "}")

    def test_percentage_normalization_uses_request_not_reference_value(self):
        for question, teacher, observed in (
            ("What is the percent of yellow tint?", r"40\%", "40"),
            ("What percentage of males listen?", "75", r"75\%"),
            ("By what percent did the price increase?", "17", r"17\%"),
        ):
            self.assertTrue(self.grade(question, teacher, observed)["correct"])
        self.assertFalse(self.grade("What percentage of males listen?", "75", r"0.75\%")["correct"])
        self.assertFalse(self.grade("25 percent of them are red. How many are red?", "5", r"5\%")["correct"])
        self.assertTrue(self.grade("What fraction is red?", r"25\%", "1/4")["correct"])

    def test_ratio_context_and_game_suffix_are_limited(self):
        self.assertTrue(self.grade("What is the ratio of x to y?", "3:4", "3/4")["correct"])
        self.assertFalse(self.grade("What is the ratio of x to y?", "3:4", "4/3")["correct"])
        self.assertTrue(self.grade("How many conference games are scheduled?", "96 games", "96")["correct"])
        self.assertFalse(self.grade("How many teams are scheduled?", "96", "96 games")["correct"])

    def test_teacher_reference_never_uses_known_gold_or_row_rules(self):
        with patch.object(amended, "RULES", {}), patch.object(amended, "REASONS", {}):
            result = self.grade("What is the probability?", "1/2", "1/2")
        self.assertTrue(result["correct"])
        self.assertEqual(result["reference_mode"], "teacher_final_and_question_context_only")
        self.assertFalse(self.grade("What is the probability?", "1/3", "0.333333333333")["correct"])

    def test_no_intermediate_or_missing_final_answer_is_selected(self):
        result = amended.grade_teacher_reference(
            "What percentage?", "40", "We derive 40 while checking another quantity."
        )
        self.assertFalse(result["correct"])
        self.assertEqual(result["status"], "no_final_answer")
        result = amended.grade_teacher_reference("How many games?", "96 games", "\\boxed{96}\nFinal answer: 97 games")
        self.assertFalse(result["correct"])


if __name__ == "__main__":
    unittest.main()
