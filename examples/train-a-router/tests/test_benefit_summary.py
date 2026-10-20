"""Comparison rendering must retain every declared policy on identical answers."""

import copy
import unittest

import benefit_summary


class BenefitSummaryTests(unittest.TestCase):
    def setUp(self):
        metric = {
            "n": 2,
            "answer_accuracy": 0.5,
            "always_strong_answer_accuracy": 0.5,
            "gap_vs_strong_pp": 0,
            "total_solver_cost_usd_per1000": 4,
            "gross_solver_savings_pct": 20,
        }
        baseline = metric | {"net_total_cost_usd_per1000": 5, "net_savings_pct": 0}
        policies = {str(target): {"metrics": metric, "mode": "threshold", "cutoff": 0.2} for target in (1, 0, 2, 5)}
        shared = {
            "n": 2,
            "ids": ["a", "b"],
            "test_grades_sha256": "a" * 64,
            "primary_max_quality_loss_pp": 1,
            "training_rows": 511,
        }
        self.embedding = shared | {
            "format": "sie-solver-benefit-report-v1",
            "policies": policies,
            "baselines": {"always_strong": baseline, "always_cheap": baseline},
        }
        self.native = shared | {
            "format": "sie-solver-benefit-native-report-v1",
            "models": {"base": policies, "lora": policies},
            "selected_epoch": 3,
        }
        self.control = shared | {"format": "sie-benefit-competition-control-report-v1", "policies": policies}

    def test_every_target_is_rendered_and_unknown_native_cost_is_explicit(self):
        result = benefit_summary.combine(self.embedding, self.native, self.control)
        rendered = benefit_summary.markdown(result)
        for target in (1, 0, 2, 5):
            self.assertIn(f"calibration target: {target} percentage point(s)", rendered)
        self.assertIn("$4.0000 + routing hosting", rendered)
        self.assertIn("20.00% before routing hosting", rendered)
        self.assertEqual(rendered.count("Laya + LoRA, 511 examples"), 4)

    def test_reordered_problems_or_different_grades_cannot_be_combined(self):
        for replacement in ({"ids": ["b", "a"]}, {"test_grades_sha256": "b" * 64}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                benefit_summary.combine(self.embedding, self.native | replacement, self.control)

    def test_hiding_a_declared_target_is_rejected(self):
        control = copy.deepcopy(self.control)
        del control["policies"]["5"]
        with self.assertRaisesRegex(ValueError, "Every predeclared"):
            benefit_summary.combine(self.embedding, self.native, control)


if __name__ == "__main__":
    unittest.main()
