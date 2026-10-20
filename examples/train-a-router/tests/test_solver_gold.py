"""Exercise source-only adjudications without consulting model outputs."""

import importlib.util
import unittest

from solver_gold import RULES, grade_reviewed, reviewed_gold


@unittest.skipUnless(importlib.util.find_spec("math_verify"), "Install the evaluation dependency group")
class ReviewedGoldTests(unittest.TestCase):
    def grade(self, identity, final):
        return grade_reviewed(identity, RULES[identity][0], "\\boxed{" + final + "}")["correct"]

    def test_all_declared_rules_are_gradeable_without_excluding_rows(self):
        original = {identity: rule[0] for identity, rule in RULES.items()} | {"ordinary": "42"}
        corrected, audit = reviewed_gold(original)
        self.assertTrue(audit["ready"])
        self.assertEqual(set(corrected), set(original))
        self.assertEqual(audit["n"], len(RULES) + 1)
        self.assertEqual(audit["excluded"], [])
        self.assertEqual(set(audit["approved_ids"]), set(RULES))
        self.assertEqual(sum(r["source_answer_correction"] for r in audit["correction_records"]), 2)

    def test_source_corrections_are_exact_and_bound_to_original_text(self):
        self.assertTrue(self.grade("e2h-amc-eval-1694", "1010000"))
        self.assertFalse(self.grade("e2h-amc-eval-1694", "1100"))
        self.assertTrue(self.grade("e2h-amc-eval-620", "21000"))
        self.assertFalse(self.grade("e2h-amc-eval-620", "210"))
        with self.assertRaisesRegex(ValueError, "original source"):
            grade_reviewed("e2h-amc-eval-620", "changed", r"\boxed{21000}")
        with self.assertRaisesRegex(ValueError, "Source gold changed"):
            reviewed_gold({"e2h-amc-eval-620": "changed"})

    def test_ratio_equivalence_does_not_turn_time_into_a_ratio(self):
        for identity, positive, negative in [
            ("e2h-amc-train-684", ["1:2", "2:4", "0.5"], ["2:1", "1:0"]),
            ("e2h-amc-eval-1815", ["2:3", "4:6", r"\frac23"], ["3:2"]),
            ("e2h-amc-eval-2906", ["4:1", "8:2", "4"], ["1:4"]),
        ]:
            for value in positive:
                self.assertTrue(self.grade(identity, value), value)
            for value in negative:
                self.assertFalse(self.grade(identity, value), value)

    def test_clock_supports_equivalent_time_notations(self):
        for value in ["6:0", "6:00", "6 pm", "6:00 PM", "18:00", r"6\text{ p.m.}", "6", "6:00 in the evening"]:
            self.assertTrue(self.grade("e2h-amc-train-977", value), value)
        for value in ["6 am", "7:00 pm", "18:30", "6:60", "24:00", "1.8:00", ".6", "6:00 in the morning"]:
            self.assertFalse(self.grade("e2h-amc-train-977", value), value)

    def test_progression_accepts_only_the_complete_discrete_set(self):
        positive = [
            r"0,3,6,9,\ldots,6048",
            r"\{0,3,\dots,6048\}",
            r"\{3k:k=0,1,\ldots,2016\}",
            r"\{3k\mid k\in\mathbb{Z},0\leq k\leq2016\}",
            r"\{3n:0\leq n<2017,n\in\mathbb{Z}\}",
            r"\{n\in\mathbb{Z}:0\le n\le6048,3\mid n\}",
            "all multiples of 3 from 0 to 6048 inclusive",
            "all multiples of 3 from 0 to 6048",
        ]
        negative = [
            "0",
            "{0,3,6,9,6048}",
            r"0,4,8,\ldots,6048",
            r"0,3,\ldots,6045",
            r"\{3k:0\leq k\leq2016\}",
            r"\{3k:k=0,1,\ldots,2015\}",
            "[0,6048]",
            r"\{n\in\mathbb{Z}:0<n\le6048,3\mid n\}",
            r"\{n\in\mathbb{R}:0\le n\le6048,3\mid n\}",
            r"\{n\in\mathbb{Z}:0\le n\le6048,0\mid n\}",
            "multiples of 3 between 0 and 6048",
        ]
        for value in positive:
            self.assertTrue(self.grade("e2h-amc-eval-2738", value), value)
        for value in negative:
            self.assertFalse(self.grade("e2h-amc-eval-2738", value), value)

    def test_ordered_coordinates_and_unordered_solution_sets(self):
        self.assertTrue(self.grade("e2h-amc-train-945", "(7,9)"))
        self.assertFalse(self.grade("e2h-amc-train-945", "(9,7)"))
        self.assertTrue(self.grade("e2h-amc-eval-769", "15,7"))
        self.assertFalse(self.grade("e2h-amc-eval-769", "7"))
        self.assertTrue(self.grade("e2h-amc-eval-111", r"\{(17,12),(2,2),(12,17)\}"))
        self.assertFalse(self.grade("e2h-amc-eval-111", r"\{(17,12),(2,2)\}"))

    def test_unknown_ambiguous_gold_is_not_automatically_approved(self):
        _, audit = reviewed_gold({"new": "(1,2)"})
        self.assertFalse(audit["ready"])
        self.assertIn("new", audit["needs_review"])
        self.assertEqual(grade_reviewed("ordinary", "42", "No final answer.")["status"], "no_final_answer")

    def test_percentage_units_are_bound_to_questions_requesting_percentages(self):
        for identity, (_, canonical, kind) in RULES.items():
            if kind != "percentage":
                continue
            number = canonical.removesuffix(r"\%")
            self.assertTrue(self.grade(identity, number), identity)
            self.assertTrue(self.grade(identity, canonical), identity)
            self.assertTrue(self.grade(identity, "x=" + number), identity)
            self.assertFalse(self.grade(identity, "0.01" + canonical), identity)
        # This source question contains percentage data but asks for ball count.
        self.assertFalse(grade_reviewed("e2h-amc-eval-2529", "50", r"\boxed{50\%}")["correct"])


if __name__ == "__main__":
    unittest.main()
