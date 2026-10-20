"""Soft-outcome training, paper selection, and native test-access contracts."""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_benefit as fixtures

import benefit
import benefit_native as native
from data import file_hash


def reseal(document):
    return benefit.seal({key: value for key, value in document.items() if key != "sha256"})


def history(targets, winner=1):
    result = []
    for epoch in range(native.PROTOCOL["selection_epochs"] + 1):
        scores = np.asarray(targets) * (0.6 if epoch == winner else 0.0)
        margins = 2 * np.arctanh(scores)
        result.append(
            {
                "epoch": epoch,
                "margins": margins.tolist(),
                "benefit_mse": float(np.mean((scores - targets) ** 2)),
                "train_loss": None if epoch == 0 else 0.5,
            }
        )
    return result


class NativeSoftTargetTests(unittest.TestCase):
    def test_soft_targets_keep_rescues_and_both_types_of_tie(self):
        inputs = [{"request": f"Problem {i}"} for i in range(4)]
        rows = native.soft_rows(inputs, [-1, 0, 0, 1])
        self.assertEqual(
            [row["gold"]["route"]["probabilities"] for row in rows],
            [
                {"cheap": 1.0, "strong": 0.0},
                {"cheap": 0.5, "strong": 0.5},
                {"cheap": 0.5, "strong": 0.5},
                {"cheap": 0.0, "strong": 1.0},
            ],
        )
        self.assertEqual(list(rows[0]["questions"]["route"]["criteria"]), ["cheap", "strong"])
        for targets in ([0], [0, 0, 0, 2], [0, 0, 0, np.nan]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                native.soft_rows(inputs, targets)

    def test_canonical_records_produce_signed_benefit_without_argmax(self):
        records = [(0, [0, margin], target, 2) for margin, target in ((-1000, [1, 0]), (0, [0.5, 0.5]), (1000, [0, 1]))]
        margins, scores = native.records_to_scores(records, [-1, 0, 1])
        np.testing.assert_array_equal(margins, [-1000, 0, 1000])
        np.testing.assert_array_equal(scores, [-1, 0, 1])
        for bad in (
            [(0, [0, 1], [1, 0], 2)],
            [(0, [0, np.nan], [0.5, 0.5], 2)],
            [(0, [0, 1], [0.5, 0.5], 3)],
        ):
            with self.subTest(records=bad), self.assertRaises(ValueError):
                native.records_to_scores(bad, [0])

    def test_epoch_selection_recomputes_soft_mse_and_includes_epoch_zero(self):
        targets = np.array([-1, 0, 1])
        complete = history(targets, winner=3)
        self.assertEqual(native.selected_epoch(complete, targets), 3)
        self.assertEqual(native.selected_epoch(history(targets, winner=0), targets), 0)
        self.assertEqual(native.selected_epoch(history(np.zeros(3)), np.zeros(3)), 0)
        complete[2]["benefit_mse"] = 0
        with self.assertRaisesRegex(ValueError, "MSE"):
            native.selected_epoch(complete, targets)
        with self.assertRaisesRegex(ValueError, "every declared"):
            native.selected_epoch(history(targets)[:-1], targets)
        with self.assertRaisesRegex(ValueError, "signed paired"):
            native.selected_epoch(history(targets), [0.5, 0, 1])

    def test_inference_rejects_missing_local_weights_before_optional_imports(self):
        with self.assertRaisesRegex(ValueError, "local complete"):
            native.score_checkpoint(Path(__file__).parent / "no-such-checkpoint", [{"request": "x"}], "cpu")


class NativeLifecycleTests(unittest.TestCase):
    def setUp(self):
        # Reuse a complete public synthetic experiment, not real solver outcomes.
        self.fixture = fixtures.BenefitLifecycleTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.root / "native"
        self.head = self.fixture.run
        fixtures.save(self.run / "plan.json", native.prepare(self.head))

    def preflight(self):
        f = self.fixture
        return native.preflight(
            self.run, self.head, f.baseline, f.original, f.training_plan, f.training_inputs, f.training_grades
        )

    def freeze(self):
        context = self.preflight()
        self.fixture.freeze()
        targets = context["targets"][context["held"]]
        selection_history = history(targets)
        chosen = native.selected_epoch(selection_history, targets)
        start = {
            "plan_sha256": file_hash(self.run / "plan.json"),
            "training_artifacts_sha256": context["training_artifacts_sha256"],
            "eager_flags": native.PROTOCOL["eager_flags"],
            "device": "mps",
            "runtime": native.PROTOCOL["runtime_profiles"]["mps"],
            "versions": native.PROTOCOL["versions"],
        }
        fixtures.save(self.run / "training-start.json", start)
        models = {}
        for variant in ("base", "lora"):
            destination = self.run / variant
            destination.mkdir()
            (destination / "model.safetensors").write_bytes(b"synthetic weights; never loaded")
            margins = {
                split: [-2 if i % 4 < 2 else 2 for i in range(len(context[split]))] for split in ("calibration", "test")
            }
            scores = {split: native.benefit_scores(values).tolist() for split, values in margins.items()}
            models[variant] = {
                "checkpoint_files_sha256": {"model.safetensors": file_hash(destination / "model.safetensors")},
                "parameters": 100,
                "margins": margins,
                "scores": scores,
                "max_merge_margin_error": 0.0,
                "max_export_margin_error": 0.0,
                "policies": native._calibrated(context["original"]["calibration_rows"], scores["calibration"], 1e-5),
            }
        frozen = benefit.seal(
            {
                "format": native.FROZEN_FORMAT,
                "plan_sha256": file_hash(self.run / "plan.json"),
                "head_plan_sha256": file_hash(self.head / "plan.json"),
                "training_start_sha256": file_hash(self.run / "training-start.json"),
                "training_artifacts_sha256": context["training_artifacts_sha256"],
                "training_grade_provenance": context["training_grade_provenance"],
                "selection_history": selection_history,
                "validation_signed_targets": targets.tolist(),
                "selected_epoch": chosen,
                "refit_losses": [0.5] * chosen,
                "grading_provenance": self.fixture.grading,
                "calibration_rows": context["original"]["calibration_rows"],
                "calibration_grades_sha256": context["original"]["calibration_grades_sha256"],
                "credit_usd": 1e-5,
                "models": models,
            }
        )
        fixtures.save(self.run / "frozen.json", frozen)
        return frozen

    def test_prepare_uses_only_one_training_paper_fold_and_preflight_leaves_test_marker_unopened(self):
        context = self.preflight()
        plan = context["plan"]
        self.assertEqual(plan["training_pool_rows"], 20)
        self.assertEqual(plan["training_rows"], 19)
        fitting, held = set(plan["selection_fit_ids"]), set(plan["selection_validation_ids"])
        self.assertFalse(fitting & held)
        self.assertEqual(fitting | held, {row["id"] for row in self.fixture.eligible_rows})
        by_id = {row["id"]: row for row in self.fixture.eligible_rows}
        self.assertFalse(
            {by_id[ident]["paper_group"] for ident in fitting} & {by_id[ident]["paper_group"] for ident in held}
        )
        marker = fixtures.save(self.fixture.original_dir / "grades-test.json", {"held": True})
        original_read = benefit.read

        def guarded(path):
            self.assertNotEqual(Path(path), marker)
            return original_read(path)

        with patch.object(benefit, "read", side_effect=guarded):
            self.preflight()

    def test_shared_access_refuses_new_training_before_optional_model_work(self):
        fixtures.save(self.head / "test-access.json", {"access": "already opened"})
        with self.assertRaisesRegex(ValueError, "already authorized"):
            native.prepare(self.head)
        f = self.fixture
        with patch.object(native, "_train_gpu") as model_work:
            with self.assertRaisesRegex(ValueError, "already authorized"):
                native.train(
                    self.run,
                    self.head,
                    f.baseline,
                    f.original,
                    f.training_plan,
                    f.training_inputs,
                    f.training_grades,
                    device="mps",
                )
            model_work.assert_not_called()

    def test_training_review_provenance_mismatch_and_repeated_attempt_fail(self):
        grades = benefit.read(self.fixture.training_grades)
        fixtures.save(self.fixture.training_grades, grades | {"training_gold_rules_sha256": "a" * 64})
        with self.assertRaisesRegex(ValueError, "review"):
            self.preflight()
        fixtures.save(self.fixture.training_grades, grades)
        fixtures.save(self.run / "training-start.json", {"previous": "attempt"})
        with self.assertRaises(FileExistsError):
            self.preflight()

    def test_freeze_recomputes_selection_scoring_and_calibration(self):
        original = self.freeze()
        native.checked_frozen(self.run, self.head, weights=True)
        mutations = (
            lambda item: item.update(selected_epoch=8),
            lambda item: item["selection_history"][1].update(benefit_mse=100.0),
            lambda item: item["models"]["lora"]["scores"]["test"].__setitem__(0, 0.0),
            lambda item: item["models"]["lora"]["policies"]["1"].update(cutoff=100.0),
            lambda item: item["models"]["lora"].update(max_export_margin_error=0.031),
        )
        for mutate in mutations:
            changed = copy.deepcopy(original)
            mutate(changed)
            fixtures.save(self.run / "frozen.json", reseal(changed))
            with self.subTest(change=mutate), self.assertRaises(ValueError):
                native.checked_frozen(self.run, self.head)
        fixtures.save(self.run / "frozen.json", original)
        start = benefit.read(self.run / "training-start.json")
        start["runtime"] = {**start["runtime"], "gradient_checkpointing": True}
        fixtures.save(self.run / "training-start.json", start)
        original["training_start_sha256"] = file_hash(self.run / "training-start.json")
        fixtures.save(self.run / "frozen.json", reseal(original))
        with self.assertRaises(ValueError):
            native.checked_frozen(self.run, self.head)

    def test_weights_optional_for_metadata_replay_but_checked_when_requested(self):
        self.freeze()
        (self.run / "lora" / "model.safetensors").write_bytes(b"changed weights")
        native.checked_frozen(self.run, self.head)
        with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
            native.authorize_test(self.run, self.head, verify_weights=True)
        (self.run / "lora" / "model.safetensors").unlink()
        access = native.authorize_test(self.run, self.head)
        self.assertEqual(access["checkpoint_verification"], "manifest-only")
        self.assertEqual(
            access["checkpoint_manifest_sha256"],
            native.checkpoint_manifest_hash(benefit.read(self.run / "frozen.json")),
        )

    def test_report_requires_both_access_receipts_before_reading_test_grades(self):
        self.freeze()
        grade_path = self.fixture.original_dir / "grades-test.json"
        with patch.object(native.solver_policies, "_grade_rows") as reader:
            with self.assertRaises(FileNotFoundError):
                native.report(self.run, self.head, grade_path)
            reader.assert_not_called()
            fixtures.save(self.run / "test-access.json", native.authorize_test(self.run, self.head))
            with self.assertRaises(FileNotFoundError):
                native.report(self.run, self.head, grade_path)
            reader.assert_not_called()
        access = benefit.authorize_test(self.head)
        fixtures.save(self.head / "test-access.json", access)
        access["plan_sha256"] = "f" * 64
        fixtures.save(self.head / "test-access.json", reseal(access))
        with patch.object(native.solver_policies, "_grade_rows") as reader:
            with self.assertRaisesRegex(ValueError, "test access"):
                native.report(self.run, self.head, grade_path)
            reader.assert_not_called()

    def test_report_preserves_cheap_rescue_negative_gap_and_unknown_native_cost(self):
        self.freeze()
        fixtures.save(self.run / "test-access.json", native.authorize_test(self.run, self.head))
        fixtures.save(self.head / "test-access.json", benefit.authorize_test(self.head))
        rows = fixtures.outcomes("test", 8)
        rows[0]["strong_correct"] = False  # Cheap-only success is a real rescue.
        path = fixtures.save(
            self.fixture.original_dir / "grades-test.json",
            {
                "split": "test",
                "plan_sha256": file_hash(self.fixture.original_plan),
                "policies_frozen_sha256": file_hash(self.fixture.original),
                **self.fixture.grading,
                "rows": rows,
            },
        )
        result = native.report(self.run, self.head, path)
        self.assertEqual(result["ids"], [row["id"] for row in rows])
        self.assertEqual(result["training_rows"], 19)
        for variant in ("base", "lora"):
            primary = result["models"][variant]["1"]
            self.assertEqual(primary["mode"], "threshold")
            self.assertEqual(primary["metrics"]["gap_vs_strong_pp"], -12.5)
            self.assertEqual(primary["metrics"]["rescues"], 1)
            self.assertFalse(primary["metrics"]["router_overhead_known"])
            self.assertNotIn("router_cost_usd_per1000", primary["metrics"])
            self.assertNotIn("net_cost_usd_per1000", primary["metrics"])


if __name__ == "__main__":
    unittest.main()
