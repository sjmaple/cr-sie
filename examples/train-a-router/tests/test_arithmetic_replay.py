import unittest

import arithmetic_replay as replay


class PublicReplayTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "easy", "cheap_correct": True, "strong_correct": True, "cheap_credits": 1, "strong_credits": 10},
            {"id": "hard", "cheap_correct": False, "strong_correct": True, "cheap_credits": 1, "strong_credits": 10},
        ]
        self.model = {
            "ids": {"test": ["easy", "hard"]},
            "scores": {"test": [0, 2]},
            "router_credits": {"test": [1, 1]},
            "policies": {"1": {"mode": "threshold", "cutoff": 1}},
            "method": "embedding_2000", "design": "embedding", "training_rows": 2000,
        }

    def test_good_decisions_save_money_after_embedding_cost(self):
        metrics = replay.method_report(self.model, self.rows)["policies"]["1"]["metrics"]
        self.assertEqual(metrics["answer_accuracy"], 1)
        self.assertEqual(metrics["gap_vs_strong_pp"], 0)
        self.assertAlmostEqual(metrics["gross_solver_savings_pct"], 45)
        self.assertAlmostEqual(metrics["net_savings_pct"], 35)

    def test_static_policy_skips_embedding_charge(self):
        self.model["policies"]["1"] = {"mode": "always_strong", "cutoff": None}
        metrics = replay.method_report(self.model, self.rows)["policies"]["1"]["metrics"]
        self.assertEqual(metrics["net_savings_pct"], 0)
        self.assertEqual(metrics["router_cost_usd_per1000"], 0)

    def test_reordered_outcomes_cannot_be_used_for_saved_decisions(self):
        with self.assertRaisesRegex(ValueError, "question order"):
            replay.method_report(self.model, list(reversed(self.rows)))


if __name__ == "__main__":
    unittest.main()
