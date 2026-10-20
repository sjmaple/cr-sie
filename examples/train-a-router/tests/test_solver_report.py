"""Check paired downstream quality, real cost accounting, and calibration selection."""

import copy
import unittest

import numpy as np

from solver_report import select_cutoff, summarize, summarize_quality


def fixture():
    # Both right, strong only, cheap only, both wrong. The two solvers have equal
    # overall accuracy; cheap-only wins must not disappear from the quality gap.
    return [
        {"id": "a", "cheap_correct": True, "strong_correct": True, "cheap_credits": 10, "strong_credits": 100},
        {"id": "b", "cheap_correct": False, "strong_correct": True, "cheap_credits": 20, "strong_credits": 200},
        {"id": "c", "cheap_correct": True, "strong_correct": False, "cheap_credits": 30, "strong_credits": 300},
        {"id": "d", "cheap_correct": False, "strong_correct": False, "cheap_credits": 40, "strong_credits": 400},
    ]


class SolverReportTests(unittest.TestCase):
    def test_quality_only_matches_paired_quality_without_inventing_prices(self):
        rows = fixture()
        priced = summarize(rows, [False, True, False, False])
        for row in rows:
            row["cheap_credits"] = None
            row["strong_credits"] = None
        quality = summarize_quality(rows, [False, True, False, False])
        self.assertEqual(quality["cost_status"], "unknown")
        for key in ("n", "answer_accuracy", "gap_vs_strong_pp", "regressions", "rescues", "cheap_share"):
            self.assertEqual(quality[key], priced[key])
        for key, interval in quality["paired_bootstrap_95ci"].items():
            np.testing.assert_allclose(interval, priced["paired_bootstrap_95ci"][key])
        self.assertFalse(any("usd" in key or "savings" in key for key in quality))
        with self.assertRaises((TypeError, ValueError)):
            summarize(rows, "always_strong")
        with self.assertRaises((TypeError, ValueError)):
            select_cutoff(rows, [0, 1, 2, 3], 1)

    def test_quality_only_requires_valid_outcomes_and_alignment_but_not_credit_fields(self):
        rows = [{key: value for key, value in row.items() if "credits" not in key} for row in fixture()]
        self.assertEqual(summarize_quality(rows, "always_strong")["gap_vs_strong_pp"], 0)
        self.assertEqual(summarize_quality(rows, "always_cheap")["regressions"], 1)
        for bad in ([], rows + [rows[0]], [dict(row, strong_correct=1) for row in rows]):
            with self.assertRaises((TypeError, ValueError)):
                summarize_quality(bad, "always_strong")
        with self.assertRaises(ValueError):
            summarize_quality(rows, [True])

    def test_rescues_reduce_the_gap_and_costs_use_each_selected_solver(self):
        result = summarize(fixture(), [False, True, False, False], router_credits=[1, 2, 3, 4])
        self.assertEqual(result["answer_accuracy"], 0.75)
        self.assertEqual(result["always_strong_answer_accuracy"], 0.5)
        self.assertEqual(result["gap_vs_strong_pp"], -25)
        self.assertEqual((result["regressions"], result["rescues"]), (0, 1))
        self.assertEqual(result["cheap_share"], 0.75)
        self.assertAlmostEqual(result["total_solver_cost_usd_per1000"], 0.7)
        self.assertAlmostEqual(result["always_strong_solver_cost_usd_per1000"], 2.5)
        self.assertAlmostEqual(result["gross_solver_savings_usd_per1000"], 1.8)
        self.assertAlmostEqual(result["gross_solver_savings_pct"], 72)
        self.assertAlmostEqual(result["router_cost_usd_per1000"], 0.025)
        self.assertAlmostEqual(result["net_total_cost_usd_per1000"], 0.725)
        self.assertAlmostEqual(result["net_savings_usd_per1000"], 1.775)
        self.assertAlmostEqual(result["net_savings_pct"], 71)
        self.assertAlmostEqual(result["break_even_router_usd_per1000"], 1.8)

    def test_regressions_and_rescues_offset_without_clipping(self):
        result = summarize(fixture(), "always_cheap")
        self.assertEqual((result["regressions"], result["rescues"]), (1, 1))
        self.assertEqual(result["gap_vs_strong_pp"], 0)
        low, high = result["paired_bootstrap_95ci"]["gap_vs_strong_pp"]
        self.assertLess(low, 0)
        self.assertGreater(high, 0)
        self.assertNotIn("net_total_cost_usd_per1000", result)
        self.assertNotIn("net_savings_pct", result["paired_bootstrap_95ci"])

    def test_explicit_baselines_bypass_router_but_uniform_learned_routes_pay(self):
        for mode, value in (("always_strong", True), ("always_cheap", False)):
            with self.subTest(mode=mode):
                baseline = summarize(fixture(), mode, router_credits=1000)
                learned = summarize(fixture(), [value] * 4, router_credits=1000)
                self.assertEqual(baseline["router_cost_usd_per1000"], 0)
                self.assertFalse(baseline["router_overhead_applied"])
                self.assertAlmostEqual(learned["router_cost_usd_per1000"], 10)
                self.assertTrue(learned["router_overhead_applied"])
        strong = summarize(fixture(), "always_strong", router_credits=1000)
        self.assertEqual(strong["paired_bootstrap_95ci"]["gap_vs_strong_pp"], [0, 0])
        self.assertEqual(strong["paired_bootstrap_95ci"]["net_savings_usd_per1000"], [0, 0])

    def test_cost_increase_and_zero_baseline_have_honest_savings(self):
        rows = fixture()
        for row in rows:
            row["strong_credits"] = 0
        result = summarize(rows, "always_cheap", router_credits=0)
        self.assertLess(result["gross_solver_savings_usd_per1000"], 0)
        self.assertLess(result["break_even_router_usd_per1000"], 0)
        self.assertIsNone(result["gross_solver_savings_pct"])
        self.assertIsNone(result["paired_bootstrap_95ci"]["net_savings_pct"])

    def test_bootstrap_is_paired_reproducible_and_does_not_mutate_inputs(self):
        rows = fixture()
        original = copy.deepcopy(rows)
        routes = {"d": False, "c": False, "b": True, "a": False}
        first = summarize(rows, routes, router_credits=0)
        self.assertEqual(first, summarize(rows, [False, True, False, False], router_credits=[0] * 4))
        self.assertEqual(rows, original)
        # Independent implementation of the fixed-seed paired resampling.
        indices = np.random.default_rng(89).integers(0, 4, (2000, 4))
        correct = np.array([True, True, True, False])
        strong = np.array([True, True, False, False])
        gap = 100 * (strong[indices].mean(axis=1) - correct[indices].mean(axis=1))
        np.testing.assert_array_equal(
            first["paired_bootstrap_95ci"]["gap_vs_strong_pp"], np.quantile(gap, [0.025, 0.975])
        )

    def test_bad_measurements_and_alignment_fail_closed(self):
        invalid_rows = [[], fixture() + [fixture()[0]]]
        for key, value in (
            ("id", ""),
            ("cheap_correct", 1),
            ("strong_correct", None),
            ("cheap_credits", -1),
            ("strong_credits", float("nan")),
            ("cheap_credits", float("inf")),
            ("strong_credits", True),
        ):
            rows = fixture()
            rows[0][key] = value
            invalid_rows.append(rows)
        rows = fixture()
        del rows[0]["cheap_credits"]
        invalid_rows.append(rows)
        for rows in invalid_rows:
            with self.subTest(rows=rows), self.assertRaises((TypeError, ValueError)):
                summarize(rows, "always_strong")
        for routes in ([True], [0, 1, 0, 1], {"a": True}, "unknown"):
            with self.subTest(routes=routes), self.assertRaises(ValueError):
                summarize(fixture(), routes)
        for overhead in (-1, float("inf"), float("nan"), True, [0], [0, 0, 0, -1]):
            with self.subTest(overhead=overhead), self.assertRaises((TypeError, ValueError)):
                summarize(fixture(), "always_strong", router_credits=overhead)
        for rate in (-1, float("nan"), True):
            with self.subTest(rate=rate), self.assertRaises((TypeError, ValueError)):
                summarize(fixture(), "always_strong", credit_usd=rate)

    def test_selection_uses_quality_target_and_keeps_score_ties_together(self):
        rows = fixture()
        rows[2]["cheap_correct"] = False  # Cheap now loses one answer with no rescue.
        selection = select_cutoff(rows, [0.1, 0.8, 0.1, 0.2], 0)
        self.assertEqual(selection["mode"], "threshold")
        self.assertEqual(selection["cutoff"], 0.8)
        self.assertEqual(selection["calibration_routes"], [False, True, False, False])
        self.assertEqual(selection["calibration"]["gap_vs_strong_pp"], 0)
        looser = select_cutoff(rows, [0.1, 0.8, 0.1, 0.2], 25)
        self.assertEqual(looser["mode"], "always_cheap")
        self.assertIsNone(looser["cutoff"])
        self.assertEqual(looser["calibration"]["gap_vs_strong_pp"], 25)

    def test_selection_bypasses_uncompetitive_router_and_all_strong_cost_ties(self):
        rows = fixture()
        rows[2]["cheap_correct"] = False
        selection = select_cutoff(rows, [0.1, 0.8, 0.1, 0.2], 0, router_credits=1000)
        self.assertEqual(selection["mode"], "always_strong")
        self.assertEqual(selection["calibration"]["router_cost_usd_per1000"], 0)
        for row in rows:
            row["cheap_credits"] = row["strong_credits"]
        tied = select_cutoff(rows, [0.1, 0.8, 0.1, 0.2], 100, router_credits=0)
        self.assertEqual(tied["mode"], "always_strong")

    def test_selection_includes_cheap_only_wins_and_actual_per_row_cost(self):
        selection = select_cutoff(fixture(), {"d": 0.2, "b": 0.8, "c": 0.1, "a": 0.1}, 0, router_credits=1000)
        self.assertEqual(selection["mode"], "always_cheap")
        self.assertEqual(selection["calibration"]["gap_vs_strong_pp"], 0)
        self.assertEqual(selection["calibration"]["regressions"], 1)
        self.assertEqual(selection["calibration"]["rescues"], 1)
        self.assertEqual(selection["calibration"]["router_cost_usd_per1000"], 0)
        rows = fixture()
        rows[2]["cheap_correct"] = False
        # For this problem strong is less expensive; sending it to cheap cannot
        # be assumed to reduce cost merely because more requests use cheap.
        rows[3]["cheap_credits"] = 1000
        cheapest = select_cutoff(rows, [0.1, 0.8, 0.1, 0.5], 0)
        self.assertEqual(cheapest["cutoff"], 0.5)
        self.assertEqual(cheapest["calibration_routes"], [False, True, False, True])

    def test_selection_rejects_nonfinite_scores_wrong_ids_and_invalid_target(self):
        for scores in ([0, 1], [0, 1, 2, float("nan")], [0, 1, 2, float("inf")], [True] * 4, {"a": 0}):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                select_cutoff(fixture(), scores, 1)
        for target in (-1, 101, float("nan"), True):
            with self.subTest(target=target), self.assertRaises((TypeError, ValueError)):
                select_cutoff(fixture(), [0, 1, 2, 3], target)

    def test_equivalent_threshold_ties_choose_the_lower_cutoff(self):
        rows = fixture()
        rows[2]["cheap_correct"] = False
        rows[3]["cheap_credits"] = rows[3]["strong_credits"]
        selection = select_cutoff(rows, [0.1, 0.8, 0.1, 0.5], 0)
        self.assertEqual(selection["cutoff"], 0.5)
        self.assertEqual(selection["calibration_routes"], [False, True, False, True])

    def test_numpy_scores_and_routes_must_be_one_dimensional(self):
        for values in (np.array(1), np.zeros((4, 1))):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                select_cutoff(fixture(), values, 1)
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                summarize(fixture(), values)


if __name__ == "__main__":
    unittest.main()
