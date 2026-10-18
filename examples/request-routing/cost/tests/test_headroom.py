"""Check the diagnostic's optimization math and isolation from held-out labels."""

import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_learning import fixture

import headroom
import reranker


class HeadroomTests(unittest.TestCase):
    def test_sweep_keeps_ties_together_and_matches_direct_routing(self):
        values = np.array([0.2, 0.8, 0.8, 0.1])
        outcomes = {
            "q_cheap": np.array([0.0, 1.0, 0.0, 1.0]),
            "q_strong": np.ones(4),
            "usd_cheap": np.array([1.0, 2.0, 3.0, 4.0]),
            "usd_strong": np.array([3.0, 4.0, 2.0, 8.0]),
        }
        costs = np.array([0.01, 0.02, 0.03, 0.04])
        points = headroom.sweep(outcomes, values, costs)
        self.assertEqual([p["tau"] for p in points], [None, 0.8, 0.2, 0.1])
        for point in points:
            direct = headroom.apply_threshold(outcomes, values, costs, point["tau"])
            for key in ("cost_per_1k", "quality_vs_strong"):
                self.assertAlmostEqual(point[key], direct[key])

    def test_oracle_matches_exhaustive_fractional_vertices(self):
        # Includes cheap better, strong cheaper, and equal-cost answers.
        o = {
            "q_cheap": np.array([0.3, 0.9, 0.2, 0.5]),
            "q_strong": np.array([1.0, 0.5, 0.7, 0.8]),
            "usd_cheap": np.array([1.0, 2.0, 4.0, 3.0]),
            "usd_strong": np.array([4.0, 5.0, 2.0, 3.0]),
        }
        for target in (0.5, 0.95, 1.0, 1.05):
            feasible = []
            for fractional in range(4):
                others = [i for i in range(4) if i != fractional]
                for bits in itertools.product((0.0, 1.0), repeat=3):
                    p = np.zeros(4)
                    p[others] = bits
                    base = np.sum(p * o["q_cheap"] + (1 - p) * o["q_strong"])
                    delta = o["q_cheap"][fractional] - o["q_strong"][fractional]
                    boundary = (target * o["q_strong"].sum() - base) / delta
                    for value in (0.0, 1.0, boundary):
                        if not 0 <= value <= 1:
                            continue
                        p[fractional] = value
                        q = np.sum(p * o["q_cheap"] + (1 - p) * o["q_strong"])
                        if q >= target * o["q_strong"].sum() - 1e-12:
                            feasible.append(np.mean(p * o["usd_cheap"] + (1 - p) * o["usd_strong"]) * 1000)
            self.assertAlmostEqual(headroom.oracle(o, target)["cost_per_1k"], min(feasible))

    def test_validation_and_test_labels_do_not_change_fits(self):
        s = fixture()
        groups = {"train": list(range(12)), "validation": list(range(12, 16)), "test": list(range(16, len(s.ids)))}
        recording = reranker.Recording(np.linspace(0, 1, len(s.ids)), np.zeros(len(s.ids)), {}, {})

        def signatures(result):
            return {
                pair: [(r["fit_tau"], r["ridge_fit"], r["cost_fit"]) for r in row["runs"]]
                for pair, row in result["pairs"].items()
            }

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(headroom.learning, "split", return_value=groups),
            patch.object(headroom.reranker, "load", return_value=recording),
        ):
            kwargs = {"seeds": (0,), "sizes": (4, 12)}
            before = headroom.experiment(s, Path(temp), **kwargs)
            for o in s.pairs.values():
                o.q_cheap[groups["test"]] = 1000
                o.q_strong[groups["test"]] = 1000
                o.usd_strong[groups["test"]] = 1000
            after_test = headroom.experiment(s, Path(temp), **kwargs)
            self.assertEqual(before, after_test)
            for o in s.pairs.values():
                o.q_cheap[groups["validation"]] = 0.123
                o.usd_strong[groups["validation"]] = 100
            after_validation = headroom.experiment(s, Path(temp), **kwargs)
            self.assertEqual(signatures(before), signatures(after_validation))


if __name__ == "__main__":
    unittest.main()
