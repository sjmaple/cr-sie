import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import arithmetic_controls as controls


class ArithmeticControlTests(unittest.TestCase):
    def test_probabilities_must_be_finite_and_unmodified(self):
        statistics = {
            "cheap": {"incorrect": 10, "correct": 2},
            "strong": {"incorrect": 20, "correct": 8},
        }
        for values in ([[float("nan"), 0.9]], [[0.5, 1.01]], [[0.5]], [[0.5, 0.9], [0.2, 0.8]]):
            with self.assertRaisesRegex(ValueError, "finite, unchanged"):
                controls.scores(values, statistics, 1)

    def test_separate_controls_preserve_core_and_bind_access(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            run, output = root / "study", root / "controls"
            run.mkdir()
            inputs = {
                "calibration": [{"id": "cal1"}, {"id": "cal2"}],
                "test": [{"id": "test1"}, {"id": "test2"}],
            }
            probabilities = [[0.8, 0.9], [0.2, 0.6]]
            native = {
                "models": {
                    "reference_base": {
                        "trained_sha256": "b" * 64,
                        "splits": {
                            split: {"ids": [row["id"] for row in rows], "probabilities": probabilities}
                            for split, rows in inputs.items()
                        },
                    }
                }
            }
            for name, value in {
                "plan.json": {},
                "native-scores.json": native,
                "grades-calibration.json": {"rows": inputs["calibration"]},
                "native-500/trained.json": {"size": 500},
                "native-2000/trained.json": {"size": 2000},
            }.items():
                path = run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            core = {
                "study_plan_sha256": controls.file_hash(run / "plan.json"),
                "calibration_grades_sha256": controls.file_hash(run / "grades-calibration.json"),
                "model_artifacts_sha256": {"native-scores.json": controls.file_hash(run / "native-scores.json")},
                "source_code_sha256": {"unchanged": "source"},
                "ids": {split: [row["id"] for row in rows] for split, rows in inputs.items()},
            }
            (run / "frozen.json").write_text(json.dumps(core))
            original = (run / "frozen.json").read_bytes()
            stack.enter_context(patch.object(controls.study, "checked_plan", return_value=({}, inputs, {})))
            stack.enter_context(patch.object(controls.study, "code_hashes", return_value={"unchanged": "source"}))
            stack.enter_context(
                patch.object(
                    controls.study,
                    "checked_native",
                    side_effect=lambda source, run, size: {"conditional_cost_statistics": {"scale": size}},
                )
            )
            stack.enter_context(
                patch.object(
                    controls.study.outcomes,
                    "normalized_scores",
                    side_effect=lambda values, statistics: (
                        (values[:, 1] - values[:, 0]) / statistics["scale"],
                        None,
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    controls.study.solver_report,
                    "select_cutoff",
                    return_value={"mode": "always_strong", "cutoff": None},
                )
            )
            result = controls.freeze(root, run, output)
            self.assertEqual((run / "frozen.json").read_bytes(), original)
            self.assertEqual(result["methods"]["base_arithmetic_500"]["training_rows"], 0)
            self.assertEqual(result["methods"]["base_arithmetic_2000"]["cost_training_rows"], 2000)
            self.assertNotEqual(
                result["methods"]["base_arithmetic_500"]["scores"],
                result["methods"]["base_arithmetic_2000"]["scores"],
            )
            controls.checked_frozen(root, run, output)
            with self.assertRaises(FileExistsError):
                controls.freeze(root, run, output)
            (run / "test-access.json").write_text(
                json.dumps({"frozen_sha256": controls.file_hash(run / "frozen.json")})
            )
            access = controls.authorize_test(root, run, output)
            self.assertEqual(access["frozen_sha256"], controls.file_hash(output / "frozen.json"))
            result["methods"]["base_arithmetic_500"]["scores"]["test"][0] += 0.5
            (output / "frozen.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "score formula changed"):
                controls.checked_frozen(root, run, output)


if __name__ == "__main__":
    unittest.main()
