"""Confirmation must freeze without test outcomes; teacher replies must align."""

import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from test_learning import fixture

import confirm
import teacher_pilot


class ConfirmationTests(unittest.TestCase):
    def test_policy_freeze_does_not_depend_on_test_grades(self):
        original = fixture()
        altered = copy.deepcopy(original)
        groups = {"train": list(range(16)), "validation": list(range(16, 20)), "test": list(range(20, 30))}
        for outcomes in altered.pairs.values():
            outcomes.q_cheap[20:] = 0
            outcomes.q_strong[20:] = 1
            outcomes.usd_strong[20:] = 100
        with (
            patch.object(confirm.learning, "split", return_value=groups),
            patch.object(confirm.encoder_comparison, "load_vectors", side_effect=lambda s, p: s),
        ):
            self.assertEqual(confirm.frozen_policies(original, None), confirm.frozen_policies(altered, None))

    def test_paired_bootstrap_preserves_identical_quality_and_halved_cost(self):
        s = fixture()
        indices = list(range(20))
        before = (np.ones(20), np.ones(20))
        after = (np.ones(20), np.full(20, 0.5))
        result = confirm.bootstrap(s, indices, s.pairs["openai"], before, after)
        self.assertEqual(result["cost_reduction_pct_95ci"], [50, 50])
        self.assertEqual(result["quality_change_pp_95ci"], [0, 0])

    def test_teacher_rejects_missing_or_reordered_labels_and_boolean_scores(self):
        rows = [{"id": i, "difficulty": 1, "escalation_probability": 50, "reason": "Familiar task"} for i in ("a", "b")]
        self.assertEqual(teacher_pilot.parse(json.dumps({"labels": rows}), ["a", "b"]), rows)
        self.assertEqual(teacher_pilot.parse(json.dumps(rows), ["a", "b"]), rows)
        with self.assertRaises(ValueError):
            teacher_pilot.parse(json.dumps({"labels": rows[::-1]}), ["a", "b"])
        rows[0]["difficulty"] = True
        with self.assertRaises(ValueError):
            teacher_pilot.parse(json.dumps({"labels": rows}), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
