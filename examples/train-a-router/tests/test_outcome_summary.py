"""Do not hide policies, mix populations, or present unpriced hosting as free."""

import copy
import unittest

import outcome_summary


class OutcomeSummaryTests(unittest.TestCase):
    def setUp(self):
        gross = {
            "n": 2,
            "answer_accuracy": 0.5,
            "always_strong_answer_accuracy": 0.5,
            "gap_vs_strong_pp": 0,
            "total_solver_cost_usd_per1000": 4,
            "always_strong_solver_cost_usd_per1000": 5,
            "gross_solver_savings_usd_per1000": 1,
            "gross_solver_savings_pct": 20,
            "break_even_router_usd_per1000": 1,
            "paired_bootstrap_95ci": {
                "gap_vs_strong_pp": [-1, 1],
                "gross_solver_savings_pct": [10, 30],
            },
        }
        net = gross | {
            "router_cost_usd_per1000": 0.01,
            "net_total_cost_usd_per1000": 4.01,
            "net_savings_usd_per1000": 0.99,
            "net_savings_pct": 19.8,
            "paired_bootstrap_95ci": gross["paired_bootstrap_95ci"] | {"net_savings_pct": [9.8, 29.8]},
        }
        baseline = net | {
            "total_solver_cost_usd_per1000": 5,
            "gross_solver_savings_usd_per1000": 0,
            "gross_solver_savings_pct": 0,
            "break_even_router_usd_per1000": 0,
            "router_cost_usd_per1000": 0,
            "net_total_cost_usd_per1000": 5,
            "net_savings_usd_per1000": 0,
            "net_savings_pct": 0,
            "paired_bootstrap_95ci": {
                "gap_vs_strong_pp": [0, 0],
                "gross_solver_savings_pct": [0, 0],
                "net_savings_pct": [0, 0],
            },
        }

        def policies(metric):
            return {str(target): {"metrics": metric, "mode": "threshold", "cutoff": 0.2} for target in (1, 0, 2, 5)}

        shared = {
            "n": 2,
            "ids": ["a", "b"],
            "test_grades_sha256": "a" * 64,
            "primary_max_quality_loss_pp": 1,
        }
        self.heads, self.natives = [], []
        for size in (510, 2560):
            self.heads.append(
                shared
                | {
                    "format": "sie-solver-outcome-report-v1",
                    "training_rows": size,
                    "policies": policies(net),
                    "baselines": {"always_strong": baseline, "always_cheap": baseline},
                }
            )
            self.natives.append(
                shared
                | {
                    "format": "sie-solver-outcome-native-report-v1",
                    "training_rows": size,
                    "models": {"base": policies(gross), "lora": policies(gross)},
                    "selected_epoch": 2,
                }
            )

    def test_all_sizes_and_targets_and_unknown_cost_are_visible(self):
        report = outcome_summary.combine(self.heads, self.natives)
        self.assertEqual(len(report["methods"]), 6)
        rendered = outcome_summary.markdown(report)
        for target in (1, 0, 2, 5):
            self.assertIn(f"calibration target: {target} percentage point(s)", rendered)
        self.assertIn("$40.000 + hosting", rendered)
        self.assertIn("$40.100", rendered)
        self.assertIn("2,560 examples", rendered)

    def test_primary_money_table_separates_solver_router_and_hosting_allowance(self):
        rendered = outcome_summary.markdown(outcome_summary.combine(self.heads, self.natives))
        self.assertIn("per **10,000 requests**", rendered)
        self.assertIn("$40.000 | $0.100 | $40.100 | $10.000 | $9.900 API savings", rendered)
        self.assertIn(
            "$40.000 | hosting unpriced | $40.000 + hosting | $10.000 | hosting must cost < $10.000", rendered
        )
        self.assertIn("+0.00 [-1.00, +1.00]", rendered)
        self.assertIn("19.80% [9.80, 29.80] (API)", rendered)
        self.assertIn("20.00% [10.00, 30.00] (solver only; hosting unpriced)", rendered)
        self.assertIn("shared-paper dependence", rendered)
        self.assertIn("Training, teacher labels", rendered)

    def test_negative_savings_are_not_presented_as_a_positive_hosting_budget(self):
        metrics = self.natives[0]["models"]["lora"]["1"]["metrics"] | {
            "break_even_router_usd_per1000": -0.2,
            "gross_solver_savings_usd_per1000": -0.2,
        }
        rendered = "\n".join(outcome_summary.cost_table([("Synthetic", metrics)]))
        self.assertIn("already $2.000 more before hosting", rendered)
        self.assertNotIn("hosting must cost < $-", rendered)

    def test_static_bypass_is_visible_and_does_not_add_unknown_hosting(self):
        native = copy.deepcopy(self.natives)
        for policy in native[0]["models"]["base"].values():
            policy.update(mode="always_strong", cutoff=None, metrics=self.heads[0]["baselines"]["always_strong"])
        rendered = outcome_summary.markdown(outcome_summary.combine(self.heads, native))
        row = next(line for line in rendered.splitlines() if "router skipped)" in line and "$" in line)
        self.assertIn("always 27B; router skipped", row)
        self.assertIn("$50.000 | $0.000 | $50.000 | $0.000 | $0.000 API savings", row)
        self.assertNotIn("hosting", row)

    def test_missing_interval_is_visible_instead_of_invented(self):
        metric = self.heads[0]["baselines"]["always_strong"] | {"paired_bootstrap_95ci": {}}
        rendered = "\n".join(outcome_summary.quality_table([("Synthetic", metric)]))
        self.assertIn("interval unavailable", rendered)

    def test_different_populations_or_training_sizes_are_rejected(self):
        for difference in ({"ids": ["b", "a"]}, {"training_rows": 511}, {"test_grades_sha256": "b" * 64}):
            native = [self.natives[0], self.natives[1] | difference]
            with self.subTest(difference=difference), self.assertRaises(ValueError):
                outcome_summary.combine(self.heads, native)

    def test_missing_secondary_target_is_rejected(self):
        heads = copy.deepcopy(self.heads)
        del heads[1]["policies"]["5"]
        with self.assertRaisesRegex(ValueError, "all four"):
            outcome_summary.combine(heads, self.natives)

    def test_native_unknown_hosting_cannot_be_reported_as_net_savings(self):
        native = copy.deepcopy(self.natives)
        native[0]["models"]["lora"]["1"]["metrics"]["net_savings_pct"] = 20
        with self.assertRaisesRegex(ValueError, "known/unknown"):
            outcome_summary.combine(self.heads, native)

    def test_teacher_comparison_retains_label_source(self):
        teacher = self.heads[0] | {
            "format": "sie-solver-outcome-teacher-report-v1",
            "teacher_model": "declared-teacher",
        }
        report = outcome_summary.combine(self.heads, self.natives, teacher)
        self.assertEqual(report["methods"]["embedding_teacher"]["label_source"], "declared-teacher")
        self.assertIn("510 teacher-labeled examples", outcome_summary.markdown(report))


if __name__ == "__main__":
    unittest.main()
