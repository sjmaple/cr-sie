"""Source-only expanded-answer semantics; no solver outputs or model calls."""

import importlib.util
import unittest
from decimal import Decimal, localcontext
from itertools import combinations, permutations
from math import comb, factorial

import benefit_data
import benefit_expanded_gold as expanded
from data import digest


@unittest.skipUnless(importlib.util.find_spec("math_verify"), "Install the evaluation dependency group")
class ExpandedGoldTests(unittest.TestCase):
    def grade(self, identity, answer):
        return expanded.grade_expanded(identity, expanded.RULES[identity][0], r"\boxed{" + answer + "}")

    def test_every_reviewed_canonical_answer_and_original_binding(self):
        self.assertFalse(set(expanded.RULES) & set(benefit_data.RULES))
        for identity, (original, canonical, _) in expanded.RULES.items():
            with self.subTest(identity=identity):
                self.assertTrue(self.grade(identity, canonical)["correct"])
                with self.assertRaisesRegex(ValueError, "original answer"):
                    expanded.grade_expanded(identity, original + " changed", canonical)

    def test_original_training_rules_delegate_unchanged(self):
        for identity, (original, canonical, _) in benefit_data.RULES.items():
            response = r"\boxed{" + canonical + "}"
            with self.subTest(identity=identity):
                self.assertEqual(
                    expanded.grade_expanded(identity, original, response),
                    benefit_data.grade_training(identity, original, response),
                )

    def test_missing_zero_and_secret_santa_corrections(self):
        for identity, correct, wrong in (
            ("e2h-amc-train-774", "400000", "4000"),
            ("e2h-amc-eval-870", "400000", "400"),
            ("e2h-amc-eval-1456", "1008016", "1816"),
            ("e2h-amc-eval-1412", "0.96040", "0.96039"),
        ):
            self.assertTrue(self.grade(identity, correct)["correct"])
            self.assertFalse(self.grade(identity, wrong)["correct"])

    def test_ratios_accept_equivalent_fraction_without_losing_direction(self):
        identity = "e2h-amc-eval-1533"
        for answer in (r"6:\pi", r"12:2\pi", r"\frac{6}{\pi}", r"\frac{1350}{225\pi}"):
            self.assertTrue(self.grade(identity, answer)["correct"], answer)
        for answer in (r"\pi:6", "6:0", "6:3", "6", "6:1:2"):
            self.assertFalse(self.grade(identity, answer)["correct"], answer)
        self.assertTrue(self.grade("e2h-amc-eval-2110", "4:2")["correct"])

    def test_morning_time_is_not_a_ratio_or_evening_time(self):
        identity = "e2h-amc-eval-910"
        for answer in ("11", "11:0", "11:00", "11 AM", "11:00 a.m.", r"\text{11:00 AM}"):
            self.assertTrue(self.grade(identity, answer)["correct"], answer)
        for answer in ("11 PM", "23:00", "11:60", "22:0", "11/0", "10:59", "1.1"):
            self.assertFalse(self.grade(identity, answer)["correct"], answer)

    def test_division_symbol_and_words_not_missing_choice_letter(self):
        identity = "e2h-amc-eval-946"
        for answer in (r"\div", "÷", "/", "division", r"\text{division operator}"):
            self.assertTrue(self.grade(identity, answer)["correct"], answer)
        for answer in ("A", "2", "+", "multiplication", r"\frac63"):
            self.assertFalse(self.grade(identity, answer)["correct"], answer)

    def test_both_officially_accepted_seating_interpretations(self):
        identity = "e2h-amc-eval-2670"
        for answer in ("2", str(2 * factorial(15) ** 2), r"2\cdot15 !^{2}", r"2(15!)^2", r"\{2,2(15!)^2\}"):
            self.assertTrue(self.grade(identity, answer)["correct"], answer)
        for answer in ("1", "4", "15!", "30!", r"2\cdot(14!)^2", "3=2"):
            self.assertFalse(self.grade(identity, answer)["correct"], answer)

    def test_plus_minus_requires_every_solution_and_keeps_tuple_order(self):
        self.assertTrue(self.grade("e2h-amc-eval-2564", r"\frac12,\frac{-1\pm\sqrt{13}}4")["correct"])
        self.assertFalse(self.grade("e2h-amc-eval-2564", r"\frac12,\frac{-1+\sqrt{13}}4")["correct"])
        self.assertTrue(self.grade("e2h-amc-train-965", r"(-1006,\pm1006\sqrt3)")["correct"])
        self.assertFalse(self.grade("e2h-amc-train-965", r"(\pm1006\sqrt3,-1006)")["correct"])
        self.assertFalse(self.grade("e2h-amc-eval-799", r"\{(3,1),(\frac13,3)\}")["correct"])

    def test_named_answers_preserve_order(self):
        identity = "e2h-amc-eval-1138"
        for answer in ("MLS", "M,L,S", "medium, large, small", "Medium > Large > Small"):
            self.assertTrue(self.grade(identity, answer)["correct"], answer)
        for answer in ("SML", "small, large, medium", "M*S*L"):
            self.assertFalse(self.grade(identity, answer)["correct"], answer)
        for answer in ("Friday", "friday", "day 5", "fifth day"):
            self.assertTrue(self.grade("e2h-amc-eval-2777", answer)["correct"], answer)
        self.assertFalse(self.grade("e2h-amc-eval-2777", "Thursday")["correct"])

    def test_percentage_context_does_not_change_counts_or_fractions(self):
        for identity, (_, canonical, kind) in expanded.RULES.items():
            if kind == "percentage":
                self.assertTrue(self.grade(identity, canonical.removesuffix(r"\%"))["correct"])
        self.assertFalse(self.grade("e2h-amc-eval-558", r"0.125\%")["correct"])
        # This question asks a fraction, even though the source prints25%.
        self.assertTrue(expanded.grade_expanded("e2h-amc-train-944", r"25\%", r"\boxed{\frac14}")["correct"])
        self.assertFalse(expanded.grade_expanded("e2h-amc-train-944", r"25\%", r"\boxed{25}")["correct"])
        self.assertFalse(expanded.grade_expanded("e2h-amc-eval-64", "5", r"\boxed{5\%}")["correct"])
        self.assertFalse(self.grade("e2h-amc-eval-1782", "26.6666667")["correct"])

    def test_source_invalid_questions_cannot_be_collected_or_graded(self):
        for identity, rule in expanded.EXCLUSIONS.items():
            self.assertEqual(len(rule["problem_sha256"]), 64)
            self.assertEqual(len(rule["solution_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "excluded before collection"):
                expanded.grade_expanded(identity, rule["original_gold"], r"\boxed{1}")
            with self.assertRaisesRegex(ValueError, "source-invalid"):
                expanded.review_expanded({identity: rule["original_gold"]}, {})

    def test_review_binds_metadata_and_keeps_unknown_ambiguity_visible(self):
        identity = "e2h-amc-train-774"
        source = {"answer": "400,0", "problem": "Reviewed source question", "solution": "Reviewed derivation"}
        corrected, _, review = expanded.review_expanded({identity: "400,0"}, {identity: source})
        self.assertTrue(review["ready"])
        self.assertEqual(corrected[identity], "400000")
        self.assertEqual(review["correction_records"][0]["problem_sha256"], digest(source["problem"]))
        self.assertEqual(review["canonical_gold_sha256"], digest(corrected))
        self.assertEqual(len(review["expanded_gold_review_source_sha256"]), 64)
        _, _, unknown = expanded.review_expanded({"unknown": "(1,2)"}, {})
        self.assertFalse(unknown["ready"])
        self.assertEqual(unknown["unresolved_ids"], ["unknown"])


class IndependentSourceProofTests(unittest.TestCase):
    @staticmethod
    def santa_counts(n):
        edges = ((0, 1), (0, 2), (1, 0), (2, 0))
        valid = 0
        for count in range(5):
            for chosen in combinations(edges, count):
                rows = {left for left, _ in chosen}
                columns = {right for _, right in chosen}
                if len(rows) != count or len(columns) != count:
                    continue
                diagonal_constraints = n - len(rows | columns)
                completions = sum(
                    (-1) ** j * comb(diagonal_constraints, j) * factorial(n - count - j)
                    for j in range(diagonal_constraints + 1)
                )
                valid += (-1) ** count * completions
        all_derangements = sum((-1) ** j * comb(n, j) * factorial(n - j) for j in range(n + 1))
        return valid, all_derangements

    def test_secret_santa_count_matches_small_exhaustive_permutations(self):
        for n in range(3, 8):
            derangements = [p for p in permutations(range(n)) if all(value != i for i, value in enumerate(p))]
            good = sum(p[0] not in (1, 2) and p[1] != 0 and p[2] != 0 for p in derangements)
            self.assertEqual(self.santa_counts(n), (good, len(derangements)))

    def test_exact101_person_probability_changes_fifth_decimal(self):
        valid, total = self.santa_counts(101)
        with localcontext() as context:
            context.prec = 60
            rounded = (Decimal(valid) / total).quantize(Decimal("0.00001"))
        self.assertEqual(str(rounded), "0.96040")
        self.assertEqual(expanded.RULES["e2h-amc-eval-1412"][1], str(rounded))


if __name__ == "__main__":
    unittest.main()
