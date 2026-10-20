import unittest

import numpy as np

import arithmetic_optimizer as stable
import benefit_outcomes as original


class ArithmeticOptimizerTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(89)
        self.matrix = generator.normal(size=(80, 4))
        self.targets = (self.matrix[:, 0] + 0.4 * generator.normal(size=80) > 0).astype(float)

    def test_same_convex_solution_and_translation_invariance(self):
        expected = original.fit_logistic(self.matrix, self.targets, 0.1)
        actual = stable.fit_logistic(self.matrix, self.targets, 0.1)
        shifted = stable.fit_logistic(self.matrix + 13, self.targets, 0.1)
        np.testing.assert_allclose(
            original.probability(self.matrix, expected),
            original.probability(self.matrix, actual),
            atol=1e-7,
            rtol=0,
        )
        np.testing.assert_allclose(
            original.probability(self.matrix, actual),
            original.probability(self.matrix + 13, shifted),
            atol=1e-7,
            rtol=0,
        )
        self.assertTrue(actual["optimizer"]["success"])

    def test_rejects_nonbinary_and_missing_class(self):
        for targets in (np.full(80, 0.5), np.ones(80)):
            with self.assertRaises(ValueError):
                stable.fit_logistic(self.matrix, targets, 0.1)

    def test_head_preserves_candidate_grid_folds_targets_and_costs(self):
        rows = [
            {"id": str(i), "cheap_correct": bool(self.targets[i]), "strong_correct": bool(self.targets[-i - 1])}
            for i in range(80)
        ]
        groups = [f"family-{i // 2}" for i in range(80)]
        folds = [(i // 2) % 5 for i in range(80)]
        costs = [[50.0 + i, 200.0 + i] for i in range(80)]
        head = stable.fit_head(self.matrix, rows, groups, costs, fold_assignments=folds)
        self.assertEqual(head["folds"], folds)
        self.assertEqual(head["training_cost_proxies"], costs)
        self.assertEqual([item["penalty"] for item in head["cv"]], original.PROTOCOL["penalties"])
        self.assertEqual(
            head["conditional_cost_statistics"], original.conditional_costs(np.array(head["targets"]), costs)
        )
        winner = min(head["cv"], key=lambda item: (item["summed_binary_logloss"], -item["penalty"]))
        self.assertEqual(head["penalty"], winner["penalty"])
        with self.assertRaises(ValueError):
            stable.fit_head(self.matrix, rows, ["one-family"] * 80, costs, fold_assignments=folds)


if __name__ == "__main__":
    unittest.main()
