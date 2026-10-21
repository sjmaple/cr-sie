import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import paired_calibration
import paired_training as core
import reproduction


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data, self.run = self.root / "data", self.root / "run"
        self.rows = [{"id": "a", "source": "mmlupro", "family": "family-a", "task": "biology", "role": "final"}]
        self.plan = {"pair_spec_sha256": "2" * 64}
        self.write(self.run / "plan.json", self.plan)
        for name in ("recordings/grades-train.json", "recordings/grades-calibration.json", "fees/projection.json"):
            self.write(self.data / name, {"fixture": name})
        for method in ("embedding", "native", "topic"):
            self.write(self.run / f"{method}-scores.json", {"fixture": method})

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def bundle(self):
        provenance = {
            key: "1" * 64 for key in (
                "source_plan_sha256", "acquisition_protocol_sha256", "grader_sha256",
                "calibration_sha256", "calibration_source_sha256", "utility_source_sha256",
            )
        }
        provenance.update(
            plan_sha256=core.sha(self.run / "plan.json"),
            pair_spec_sha256=self.plan["pair_spec_sha256"],
            training_grades_sha256=core.sha(self.data / "recordings/grades-train.json"),
            calibration_grades_sha256=core.sha(self.data / "recordings/grades-calibration.json"),
            fee_projection_sha256=core.sha(self.data / "fees/projection.json"),
        )
        policies = {}
        for method in ("embedding", "base", "lora", "topic", "topic_metadata_diagnostic", "random"):
            for target in paired_calibration.TARGETS:
                policies[f"{method}@{target}pp"] = {
                    "kind": "random" if method == "random" else "learned",
                    "method": method,
                    "target_gap_pp": target,
                    "strong_probabilities": [0],
                    "router_usd": [0],
                    "metadata_assisted": method == "topic_metadata_diagnostic",
                }
        for arm in ("cheap", "strong"):
            policies[f"always_{arm}"] = {
                "kind": "always", "target_gap_pp": None,
                "strong_probabilities": [int(arm == "strong")], "router_usd": [0],
            }
        return {
            **provenance,
            "format": "routerbench-frozen-final-routing-v1",
            "frozen_before_final_model_outcomes": True,
            "final_outcomes_accessed": False,
            "ids": ["a"],
            "source_rows": [{key: value for key, value in self.rows[0].items() if key != "role"}],
            "protocol": paired_calibration.PROTOCOL,
            "scores_sha256": {method: core.sha(self.run / f"{method}-scores.json") for method in ("embedding", "native", "topic")},
            "cost_scope": "fixture USD",
            "policies": policies,
        }

    def fixtures(self):
        original = self.bundle()
        original_path = self.data / "reference/frozen/frozen-final-routes.json"
        self.write(original_path, original)
        grades = {
            "format": "routerbench-paired-grades-v1", "split": "final",
            "source_plan_sha256": original["source_plan_sha256"],
            "pair_spec_sha256": original["pair_spec_sha256"],
            "cost_scope": original["cost_scope"],
            "parent_acquisition_protocol_sha256": original["acquisition_protocol_sha256"],
            "parent_grader_sha256": original["grader_sha256"],
            "frozen_final_routing_sha256": core.sha(original_path),
            "acquisition_protocol_sha256": "3" * 64,
            "grader_sha256": "4" * 64,
            "rows": [{"id": "a", "dataset": "mmlupro", "family": "family-a", "task": "biology"}],
        }
        self.write(self.data / "recordings/grades-final.json", grades)
        changed = self.bundle()
        changed["policies"]["embedding@1pp"]["strong_probabilities"] = [1]

        def freeze(*args):
            self.write(Path(args[-1]) / "frozen-final-routes.json", changed)
            return changed

        with patch.object(core, "checked_plan", return_value=(self.plan, self.rows)):
            with patch.object(paired_calibration, "freeze", side_effect=freeze) as fitter:
                envelope = reproduction.calibrate(self.run, self.data, self.run / "frozen")
        return envelope, fitter.call_args.args

    def test_public_calibration_is_retrospective_without_reading_final_grades(self):
        envelope, arguments = self.fixtures()
        self.assertFalse(envelope["frozen_before_final_model_outcomes"])
        self.assertFalse(envelope["final_outcomes_accessed_during_calibration"])
        self.assertNotIn(self.data / "recordings/grades-final.json", arguments)
        with self.assertRaisesRegex(ValueError, "timing boundary"):
            paired_calibration.checked_frozen_bundle(envelope, self.rows)

    def test_refit_evaluation_retains_original_grade_binding(self):
        self.fixtures()
        final_path = self.data / "recordings/grades-final.json"
        original_bytes = final_path.read_bytes()
        with patch.object(core, "checked_plan", return_value=(self.plan, self.rows)):
            with patch.object(reproduction.paired_report, "evaluate", return_value={"questions": 1}):
                result = reproduction.evaluate(self.data, self.run, self.root / "report.json")
        report = core.read(result["output"])
        self.assertFalse(report["new_blind_evaluation"])
        self.assertTrue(report["test_outcomes_already_published"])
        self.assertNotEqual(
            report["provenance"]["generated_routes_sha256"],
            report["provenance"]["original_frozen_routing_sha256"],
        )
        self.assertEqual(final_path.read_bytes(), original_bytes)

    def test_bad_original_lineage_blocks_refit_evaluation(self):
        self.fixtures()
        final_path = self.data / "recordings/grades-final.json"
        grades = core.read(final_path)
        grades["frozen_final_routing_sha256"] = "9" * 64
        self.write(final_path, grades)
        with patch.object(core, "checked_plan", return_value=(self.plan, self.rows)):
            with patch.object(reproduction.paired_report, "evaluate") as evaluate:
                with self.assertRaisesRegex(ValueError, "frozen parent lineage"):
                    reproduction.evaluate(self.data, self.run, self.root / "report.json")
        evaluate.assert_not_called()

    def test_changed_internal_routes_are_rejected(self):
        self.fixtures()
        generated = self.run / "frozen" / reproduction.INTERNAL_ROUTES
        self.write(generated, self.bundle())
        with self.assertRaisesRegex(ValueError, "route artifact changed"):
            reproduction.checked_routes(self.run / "frozen/frozen-final-routes.json", self.rows)

    def test_reference_controls_require_identical_plan_and_preserve_exact_bytes(self):
        reference = self.data / "reference"
        self.write(reference / "plan.json", {"different": True})
        with patch.object(core, "checked_plan", return_value=(self.plan, self.rows)):
            with self.assertRaisesRegex(ValueError, "identical study plan"):
                reproduction.reuse_controls(self.data, self.run)
        self.assertFalse((self.run / "comparison-reuse.json").exists())
        self.write(reference / "plan.json", self.plan)
        bundle = self.bundle()
        for method in ("native", "topic"):
            name = "native/trained.json" if method == "native" else "topic-model.json"
            key = "trained_sha256" if method == "native" else "model_sha256"
            self.write(reference / name, {"comparison": method})
            self.write(reference / f"{method}-scores.json", {key: core.sha(reference / name)})
            bundle["scores_sha256"][method] = core.sha(reference / f"{method}-scores.json")
            (self.run / f"{method}-scores.json").unlink()
        self.write(reference / "frozen/frozen-final-routes.json", bundle)
        with patch.object(core, "checked_plan", return_value=(self.plan, self.rows)):
            result = reproduction.reuse_controls(self.data, self.run)
        self.assertFalse(result["fitting_performed"])
        self.assertFalse(result["weights_copied"])
        for name, digest in result["copied_sha256"].items():
            self.assertEqual(core.sha(reference / name), digest)


if __name__ == "__main__":
    unittest.main()
