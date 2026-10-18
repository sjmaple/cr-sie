"""Exact cross-validation, policy serving, and held-out isolation for the learned router."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_learning import fixture

import learn
import learning
import study


class LearnedRouterTests(unittest.TestCase):
    def test_analytic_leave_one_out_matches_refitting_every_fold(self):
        rng = np.random.default_rng(3)
        x = study.normalise(rng.normal(size=(15, 6)))
        y = rng.normal(size=15)
        for penalty in learn.PENALTIES:
            _, _, loo = learn.fit(x, y, penalty)
            brute = []
            for i in range(len(x)):
                keep = np.arange(len(x)) != i
                weights, bias, _ = learn.fit(x[keep], y[keep], penalty)
                brute.append(float(x[i] @ weights + bias))
            np.testing.assert_allclose(loo, brute, atol=1e-11, rtol=1e-11)

    def test_policy_uses_normalized_vectors_and_rejects_nonfinite_parameters(self):
        policy = {
            "type": learn.POLICY_TYPE,
            "model": study.EMBEDDER,
            "pair": "openai",
            "weights": [0.5, 2],
            "bias": 0.2,
            "tau": 1.2,
        }
        learn.validate_policy(policy, "openai")
        self.assertAlmostEqual(learn.policy_score(policy, [3, 4]), 2.1)
        with self.assertRaises(ValueError):
            learn.policy_score(policy, [1, 2, 3])
        for change in ({"bias": float("nan")}, {"weights": [float("inf")]}, {"tau": True}):
            with self.assertRaises(ValueError):
                learn.validate_policy(policy | change, "openai")

    def test_test_outcomes_do_not_select_model_and_embedding_cost_does_not_change(self):
        s = fixture()
        groups = learning.split(s, 4)
        real_metrics = learn.metrics
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)

            def guarded(o, values, usd, tau, indices):
                if set(indices) & set(groups["test"]):
                    self.assertTrue((path / "selection.json").exists())
                return real_metrics(o, values, usd, tau, indices)

            with patch.object(learn, "metrics", side_effect=guarded):
                results = learn.experiment(s, path, initial_size=4)
            selection = json.loads((path / "selection.json").read_text())
            s.pairs["openai"].q_cheap[groups["test"]] = 0
            with patch.object(learn, "metrics", side_effect=guarded):
                learn.experiment(s, path, initial_size=4)
            self.assertEqual(selection, json.loads((path / "selection.json").read_text()))
            row = results["openai"]
            self.assertEqual(row["static"]["embedding_per_1k"], row["updated"]["embedding_per_1k"])
            for variant in ("static", "updated"):
                self.assertAlmostEqual(
                    row[variant]["cost_per_1k"], row[variant]["embedding_per_1k"] + row[variant]["generation_per_1k"]
                )


if __name__ == "__main__":
    unittest.main()
