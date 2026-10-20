"""Transfer scoring must use inputs only and keep the existing policies."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import generalization_score as scoring
import solver
from data import digest, file_hash, write


class GeneralizationScoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.study, self.run, self.head = (self.root / name for name in ("study", "native", "head"))
        self.rows = [{"id": "a", "request": "Problem A"}, {"id": "b", "request": "Problem B"}]
        write(self.study / "inputs.json", self.rows)
        write(
            self.study / "source-plan.json",
            {
                "format": "sie-arithmetic-transfer-source-v1",
                "n": 2,
                "inputs_sha256": digest(self.rows),
                "settings_sha256": digest(solver.SETTINGS),
                "files_sha256": {"inputs.json": file_hash(self.study / "inputs.json")},
            },
        )
        write(self.run / "frozen.json", {"fixture": "native"})
        write(self.head / "frozen.json", {"fixture": "head"})
        self.costs = {
            f"{arm}_{outcome}": {"mean_credits_proxy": cost}
            for arm, cost in (("cheap", 1), ("strong", 5))
            for outcome in ("correct", "wrong")
        }

    def test_native_scoring_reads_no_gold_and_serializes_numpy_margins(self):
        policies = {"1": {"mode": "threshold", "cutoff": 0.03}}
        frozen = {"models": {variant: {"policies": policies} for variant in ("base", "lora")}}
        trained = {"conditional_cost_statistics": self.costs, "checkpoints": {"fixture": "bound"}}
        with (
            patch.object(scoring.native, "checked_frozen", return_value=({}, trained, frozen)) as checked,
            patch.object(
                scoring.native, "checkpoint_margins", return_value=np.array([[1.0, 2.0], [2.0, 1.0]])
            ) as infer,
        ):
            result = scoring.score_native(self.run, self.head, self.study, self.root / "scores.json")
        self.assertEqual(infer.call_count, 2)
        self.assertTrue(checked.call_args.kwargs["weights"])
        self.assertEqual(result["models"]["lora"]["policies"], policies)
        self.assertEqual(result, json.loads((self.root / "scores.json").read_text()))
        self.assertFalse((self.study / "gold-for-scoring-only.json").exists())

    def test_input_change_rejected_before_native_inference(self):
        (self.study / "inputs.json").write_text(json.dumps(self.rows[::-1]))
        with patch.object(scoring.native, "checked_frozen") as checked, self.assertRaises(ValueError):
            scoring.score_native(self.run, self.head, self.study, self.root / "scores.json")
        checked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
