"""Check split isolation and statistical operations independently of model scores."""

import copy
import unittest

import numpy as np

from data import audit, duplicate_components
from finetune import cutoff as native_cutoff
from policies import cutoff, ridge


class ProtocolTests(unittest.TestCase):
    def test_families_include_number_substitutions(self):
        text = "Alice has {} apples in a basket and Bob gives her three more apples in the morning"
        rows = [{"problem": text.format(n)} for n in (12, 24)]
        rows.append({"problem": "Find the roots of the polynomial x squared minus five"})
        self.assertEqual(duplicate_components(rows), [[0, 1], [2]])

    def test_audit_rejects_cross_split_families(self):
        splits = {}
        for split, year in zip(("train", "validation", "calibration", "test"), (2017, 2018, 2019, 2020)):
            splits[split] = [
                {
                    "id": f"{split}{i}",
                    "request": f"{split} text {i}",
                    "label": i,
                    "family": f"{split}{i}",
                    "paper_group": str(year),
                    "year": year,
                }
                for i in (0, 1)
            ]
        audit({"splits": splits})
        bad = copy.deepcopy(splits)
        bad["test"][0]["family"] = "train0"
        with self.assertRaisesRegex(ValueError, "leakage"):
            audit({"splits": bad})
        bad = copy.deepcopy(splits)
        bad["test"][0]["year"] = 2017
        with self.assertRaisesRegex(ValueError, "year"):
            audit({"splits": bad})

    def test_ridge_matches_primal_solution(self):
        rng = np.random.default_rng(13)
        x, y = rng.normal(size=(20, 4)), rng.integers(0, 2, 20)
        centered = x - x.mean(axis=0)
        expected = np.linalg.solve(centered.T @ centered + 0.3 * np.eye(4), centered.T @ (y - y.mean()))
        weights, bias = ridge(x, y, 0.3)
        np.testing.assert_allclose(weights, expected)
        self.assertAlmostEqual(bias, y.mean() - x.mean(axis=0) @ expected)

    def test_calibration_handles_imbalance(self):
        values, labels = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 1, 0])
        self.assertAlmostEqual(cutoff(values, labels), 0.25)
        self.assertAlmostEqual(native_cutoff(values, labels), 0.25)
        with self.assertRaises(ValueError):
            cutoff(values, np.zeros(4))


if __name__ == "__main__":
    unittest.main()
