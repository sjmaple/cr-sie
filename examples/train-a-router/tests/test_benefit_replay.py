"""Student reproduction boundaries without GPUs, downloads or real outcomes."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import benefit_replay as replay
from data import file_hash, write


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study, self.run = self.root / "study", self.root / "student"
        write(self.study / "manifest.json", {"synthetic": True})
        for name in replay.PLAN_NAMES:
            write(self.study / "benefit" / name / "plan.json", {"historical": name, "synthetic": True})
        self.sources = {
            "root": self.study,
            "baseline": self.study / "benefit/baseline",
            "training": self.study / "benefit/training",
            **{name: self.study / "benefit" / name for name in replay.PLAN_NAMES},
        }
        self.stack = self.enterContext(ExitStack())
        self.stack.enter_context(patch.object(replay, "reference", return_value=self.sources))
        self.stack.enter_context(patch.object(replay.benefit, "_checked_plan"))
        self.stack.enter_context(patch.object(replay.benefit_control, "checked_plan"))
        self.stack.enter_context(patch.object(replay.benefit_native, "checked_plan"))

    def prepared(self):
        return replay.prepare(self.study, self.run)

    def fitted_embedding(self):
        self.prepared()
        with (
            patch.object(replay.benefit, "freeze", return_value={"synthetic": "head"}),
            patch.object(replay.benefit_control, "freeze", return_value={"synthetic": "control"}),
        ):
            return replay.fit_embedding(self.run, self.study)

    def test_prepare_keeps_historical_bytes_and_creates_no_attestation(self):
        result = self.prepared()
        self.assertEqual(result["evaluation_mode"], replay.MODE)
        for name in replay.PLAN_NAMES:
            self.assertEqual(
                (self.run / name / "plan.json").read_bytes(),
                (self.sources[name] / "plan.json").read_bytes(),
            )
            self.assertFalse((self.run / name / "test-access.json").exists())
        with self.assertRaises(FileExistsError):
            replay.prepare(self.study, self.run)

    def test_changed_historical_plan_or_false_access_receipt_is_rejected(self):
        self.prepared()
        (self.run / "head/plan.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Historical plan"):
            replay.checked_run(self.run, self.study)
        (self.run / "head/plan.json").write_bytes((self.sources["head"] / "plan.json").read_bytes())
        write(self.run / "head/test-access.json", {"should_not_exist": True})
        with self.assertRaisesRegex(ValueError, "unopened-test"):
            replay.checked_run(self.run, self.study)

    def test_cpu_fit_receipt_binds_outputs_and_refuses_repeat_before_fit(self):
        result = self.fitted_embedding()
        self.assertEqual(result["evaluation_mode"], replay.MODE)
        self.assertEqual(result["files_sha256"]["head/frozen.json"], file_hash(self.run / "head/frozen.json"))
        with patch.object(replay.benefit, "freeze") as fit:
            with self.assertRaises(FileExistsError):
                replay.fit_embedding(self.run, self.study)
            fit.assert_not_called()
        (self.run / "head/frozen.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            replay.completed_stage(self.run, "fit-embedding")

    def test_partial_native_run_refuses_download_or_training(self):
        self.prepared()
        write(self.run / "native/training-start.json", {"synthetic": True})
        with patch.object(replay.benefit_native, "train") as train:
            with self.assertRaises(FileExistsError):
                replay.fit_native(self.run, self.study, "mps")
            train.assert_not_called()

    def test_native_stage_uses_existing_api_without_authorizing_test(self):
        self.prepared()

        def train(run, *_args, **kwargs):
            self.assertEqual(kwargs, {"device": "cuda", "local_files_only": True})
            write(run / "frozen.json", {"synthetic": "native"})

        with (
            patch.object(replay.benefit_native, "train", side_effect=train),
            patch.object(replay.benefit_native, "checked_frozen"),
            patch.object(replay.benefit, "authorize_test") as head_access,
            patch.object(replay.benefit_native, "authorize_test") as native_access,
        ):
            result = replay.fit_native(self.run, self.study, "cuda", local_files_only=True)
        self.assertEqual(result["evaluation_mode"], replay.MODE)
        self.assertFalse((self.run / "head/frozen.json").exists())
        head_access.assert_not_called()
        native_access.assert_not_called()

    def test_cost_accounting_keeps_native_unknown_and_static_router_free_of_model_calls(self):
        rows = [
            {"id": "a", "cheap_correct": False, "strong_correct": True, "cheap_credits": 2, "strong_credits": 10},
            {"id": "b", "cheap_correct": True, "strong_correct": True, "cheap_credits": 2, "strong_credits": 10},
        ]
        selections = {str(target): {"mode": "threshold", "cutoff": 0.5} for target in replay.benefit.TARGETS}
        selections["0"] = {"mode": "always_strong", "cutoff": None}
        with patch.object(replay.solver_report, "BOOTSTRAP_SAMPLES", 20):
            native = replay.measured(rows, [1, -1], selections, None, 1e-5)
            embeddings = replay.measured(rows, [1, -1], selections, [1, 1], 1e-5)
        self.assertNotIn("net_savings_pct", native["1"]["metrics"])
        self.assertAlmostEqual(embeddings["1"]["metrics"]["net_savings_pct"], 30)
        self.assertAlmostEqual(native["1"]["metrics"]["gross_solver_savings_pct"], 40)
        self.assertEqual(embeddings["0"]["metrics"]["router_cost_usd_per1000"], 0)

    def test_evaluation_cannot_read_outcomes_before_completed_native_stage(self):
        self.fitted_embedding()
        with patch.object(replay.solver_policies, "_grade_rows") as read_outcomes:
            with self.assertRaises(FileNotFoundError):
                replay.evaluate(self.run, self.study)
            read_outcomes.assert_not_called()

    def test_public_evaluation_labels_exposure_and_retains_all_targets(self):
        self.fitted_embedding()
        write(self.run / "native/frozen.json", {"synthetic": True})
        replay.stage_done(self.run, "fit-native", ("native/frozen.json",))
        write(self.study / "grades-test.json", {"synthetic": True})
        rows = [
            {"id": "a", "cheap_correct": False, "strong_correct": True, "cheap_credits": 2, "strong_credits": 10},
            {"id": "b", "cheap_correct": True, "strong_correct": True, "cheap_credits": 2, "strong_credits": 10},
        ]
        selections = {str(target): {"mode": "threshold", "cutoff": 0.5} for target in replay.benefit.TARGETS}
        provenance = {"synthetic": "grader"}
        plan = {
            "populations": {"test": [{"id": row["id"], "request_sha256": "synthetic"} for row in rows]},
            "original_solver_plan_sha256": "synthetic-plan",
            "original_frozen_sha256": "synthetic-freeze",
        }
        head = {
            "plan": plan,
            "grading_provenance": provenance,
            "credit_usd": 1e-5,
            "training_rows": 2,
            "scores": {"test": [1, -1]},
            "selections": selections,
        }
        control = {"scores": {"test": [1, -1]}, "selections": selections}
        native = {
            "grading_provenance": provenance,
            "selected_epoch": 1,
            "models": {name: {"scores": {"test": [1, -1]}, "policies": selections} for name in ("base", "lora")},
        }
        with (
            patch.object(replay.benefit, "_checked_frozen", return_value=head),
            patch.object(replay.benefit_control, "checked_frozen", return_value=(control, head)),
            patch.object(replay.benefit_native, "checked_frozen", return_value=(plan, native)),
            patch.object(
                replay.solver_policies,
                "_grade_rows",
                return_value={"rows": rows, "policies_frozen_sha256": "synthetic-freeze"},
            ),
            patch.object(replay.solver_policies, "_grading_provenance", return_value=provenance),
            patch.object(replay.solver_policies, "_credits", return_value=([1, 1], "synthetic-credits")),
            patch.object(replay.solver_report, "BOOTSTRAP_SAMPLES", 20),
        ):
            result = replay.evaluate(self.run, self.study)
        self.assertEqual(result["evaluation_mode"], replay.MODE)
        self.assertEqual(set(result["policies"]), {"base", "lora", "embedding", "competition"})
        self.assertTrue(all(set(policies) == {"0", "1", "2", "5"} for policies in result["policies"].values()))
        self.assertIn(replay.MODE, (self.run / "evaluation.md").read_text())
        self.assertEqual(json.loads((self.run / "evaluation.json").read_text()), result)
        self.assertFalse(list(self.run.rglob("test-access.json")))
        with self.assertRaises(FileExistsError):
            replay.evaluate(self.run, self.study)


if __name__ == "__main__":
    unittest.main()
