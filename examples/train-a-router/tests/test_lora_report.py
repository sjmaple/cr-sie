"""A self-consistent run must still match the frozen study it is reported under."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from finetune import BASE, LORA_RECIPES, REVISION, auc, cutoff, lora_targets, metrics, sha, write
from lora_report import check_evaluation_lifecycle, check_study_run, comparison, recipe_means


class StudyBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.labels = [0, 0, 1, 1]
        self.study = {
            "base": BASE,
            "revision": REVISION,
            "recipes": copy.deepcopy(LORA_RECIPES),
            "epochs": 8,
            "export_dtype": "float32",
            "data_sha256": {
                size: {
                    "train.jsonl": value * 64,
                    "validation.jsonl": "c" * 64,
                    "calibration.jsonl": "d" * 64,
                    "questions.json": "e" * 64,
                }
                for size, value in (("512", "a"), ("expanded", "b"))
            },
        }
        recipe = LORA_RECIPES["original"]
        self.plan = {
            "base": BASE,
            "revision": REVISION,
            "recipe": "original",
            "validation_metric": "auc",
            "export_dtype": "float32",
            "variants": ["lora"],
            "config": {"seed": 89, "epochs": 8},
            "data_sha256": self.study["data_sha256"]["expanded"],
            "lora": {**recipe, "alpha": 2 * recipe["rank"], "targets": lora_targets(recipe)},
        }
        self.history = []
        for epoch in range(9):
            scores = np.array([0.0, 1.0, 2.0, 3.0] if epoch == 1 else [0.0, 2.0, 1.0, 3.0])
            row = metrics(scores, np.array(self.labels), 0.0)
            row.update(
                epoch=epoch,
                auc=auc(scores, self.labels),
                cross_entropy=float(np.mean(np.logaddexp(0, scores) - np.array(self.labels) * scores)),
            )
            self.history.append(row)
        selected = self.history[1]
        cal_scores = np.array([-0.2, 0.1, 0.8, 0.9])
        threshold = cutoff(cal_scores, self.labels)
        self.calibration = metrics(cal_scores, np.array(self.labels), threshold)
        self.saved = {
            "seed": 89,
            "epoch": selected["epoch"],
            "validation_auc": selected["auc"],
            "validation_cross_entropy": selected["cross_entropy"],
            "validation_accuracy_at_zero": selected["accuracy"],
            "threshold": threshold,
            "checkpoint_sha256": "f" * 64,
        }
        self.receipt = self.seal()

    def seal(self):
        write(self.directory / "plan.json", self.plan)
        write(self.directory / "lora-history.json", self.history)
        write(self.directory / "lora-calibration.json", self.calibration)
        write(self.directory / "lora-merge-parity.json", {"max_margin_error": 0.0001})
        write(self.directory / "lora-export-parity.json", {"max_margin_error": 0.0001})
        write(
            self.directory / "frozen.json",
            {"plan_sha256": sha(self.directory / "plan.json"), "selected": {"lora": self.saved}},
        )
        return {"frozen_sha256": sha(self.directory / "frozen.json"), **self.saved}

    def check(self, receipt=None):
        return check_study_run(
            self.directory,
            self.receipt if receipt is None else receipt,
            self.study,
            "original",
            89,
            "expanded",
            self.labels,
            self.labels,
        )

    def test_valid_expanded_run(self):
        self.assertEqual(self.check(), 1.0)

    def test_substituting_self_consistent_run_breaks_stage_binding(self):
        self.saved["checkpoint_sha256"] = "0" * 64
        self.seal()
        with self.assertRaisesRegex(ValueError, "stage was frozen"):
            self.check()

    def test_self_consistent_wrong_training_size_is_rejected(self):
        self.plan["data_sha256"] = self.study["data_sha256"]["512"]
        receipt = self.seal()
        with self.assertRaisesRegex(ValueError, "development inputs"):
            self.check(receipt)

    def test_self_consistent_wrong_seed_is_rejected(self):
        self.plan["config"]["seed"] = self.saved["seed"] = 137
        receipt = self.seal()
        with self.assertRaisesRegex(ValueError, "seed"):
            self.check(receipt)

    def test_self_consistent_wrong_recipe_is_rejected(self):
        self.plan["recipe"] = "slow"
        receipt = self.seal()
        with self.assertRaisesRegex(ValueError, "recipe"):
            self.check(receipt)

    def test_self_consistent_wrong_export_precision_is_rejected(self):
        self.plan["export_dtype"] = "float16"
        receipt = self.seal()
        with self.assertRaisesRegex(ValueError, "recipe, seed or development inputs"):
            self.check(receipt)

    def test_failed_export_parity_is_rejected(self):
        write(self.directory / "lora-export-parity.json", {"max_margin_error": 0.04})
        with self.assertRaisesRegex(ValueError, "parity check"):
            self.check()

    def test_expanded_checkpoint_must_win_validation_selection(self):
        other = self.history[2]
        self.saved.update(
            epoch=other["epoch"],
            validation_auc=other["auc"],
            validation_cross_entropy=other["cross_entropy"],
            validation_accuracy_at_zero=other["accuracy"],
        )
        receipt = self.seal()
        with self.assertRaisesRegex(ValueError, "validation selection"):
            self.check(receipt)

    def test_history_cannot_omit_epoch_zero(self):
        self.history = self.history[1:]
        self.seal()
        with self.assertRaisesRegex(ValueError, "including epoch zero"):
            self.check()

    def test_history_cannot_use_other_validation_labels(self):
        self.history[0]["labels"] = [0, 1, 0, 1]
        self.seal()
        with self.assertRaisesRegex(ValueError, "validation labels"):
            self.check()

    def test_validation_auc_is_recomputed_from_scores(self):
        self.history[0]["auc"] = 0.99
        self.seal()
        with self.assertRaisesRegex(ValueError, "validation AUC"):
            self.check()

    def test_calibration_cutoff_cannot_be_replaced(self):
        self.saved["threshold"] = 99.0
        self.calibration = metrics(np.array(self.calibration["scores"]), np.array(self.labels), 99.0)
        receipt = self.seal()
        with self.assertRaisesRegex(ValueError, "separate calibration"):
            self.check(receipt)


class StatisticalTests(unittest.TestCase):
    def test_report_uses_exact_driver_mean(self):
        values = [0.5450405531179364, 0.5051190001329611, 0.7427868634490095]
        self.assertEqual(recipe_means({"original": values}), {"original": sum(values) / 3})
        self.assertEqual(recipe_means({"original": values})["original"], 0.5976488055666357)

    def test_duplicating_seed_does_not_create_more_test_observations(self):
        rows = [{"label": label, "paper_group": group} for label, group in zip((0, 0, 1, 1), ("a", "a", "b", "b"))]
        before, after = [0, 1, 1, 0], [0, 0, 1, 0]
        single = comparison(rows, before, after)
        repeated = comparison(rows, [before] * 3, [after] * 3)
        self.assertEqual(single, repeated)
        self.assertEqual(single["accuracy_gain_pp"], 25.0)
        self.assertIn("4 problems / 2 competition-year groups", single["method"])


class EvaluationLifecycleTests(unittest.TestCase):
    """Selection is checked separately; these fixtures model its completed lifecycle."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.study, self.evidence = root / "study", root / "evidence"
        self.environment = {"TORCH_COMPILE_DISABLE": "1", "TORCHDYNAMO_DISABLE": "1"}
        for file in (
            "train.jsonl",
            "validation.jsonl",
            "calibration.jsonl",
            "questions.json",
            "test.json",
            "manifest.json",
        ):
            write(self.evidence / file, {})
        self.study_plan = {
            "training_source_sha256": "a" * 64,
            "data_sha256": {
                "512": {
                    file: sha(self.evidence / file)
                    for file in ("train.jsonl", "validation.jsonl", "calibration.jsonl", "questions.json")
                }
            },
        }
        self.choice = {"recipe": "original"}
        write(self.study / "study-plan.json", self.study_plan)
        write(self.study / "all-frozen.json", {})
        names = [f"{stage}/seed-{seed}" for stage in ("screening/original", "expanded") for seed in (89, 137, 211)]
        self.first = names[0]
        self.records, checkpoints, references = {}, {}, {}
        for name in names:
            directory = self.study / name
            write(directory / "plan.json", {})
            write(
                directory / "frozen.json",
                {
                    "plan_sha256": sha(directory / "plan.json"),
                    "selected": {"lora": {"epoch": 1, "threshold": 0.5, "checkpoint_sha256": "f" * 64}},
                },
            )
            row = {"epoch": 1, "scores": [-1.0, 0.49, 0.8, 1.2], "labels": [0, 0, 1, 1]}
            write(directory / "lora-history.json", [row])
            write(directory / "lora-calibration.json", row)
            write(directory / "evaluation-test.json", {})
            self.records[name] = {
                "runtime_environment": self.environment.copy(),
                "frozen_sha256": sha(directory / "frozen.json"),
                "splits": {
                    split: {
                        "scores": row["scores"].copy(),
                        "labels": row["labels"].copy(),
                        "max_margin_error": 0.0,
                        "decision_changes": 0,
                    }
                    for split in ("validation", "calibration")
                },
            }
            checkpoints[name] = {
                "frozen_sha256": sha(directory / "frozen.json"),
                "plan_sha256": sha(directory / "plan.json"),
                "checkpoint_files_sha256": {
                    file: "f" * 64
                    for file in (
                        "model.safetensors",
                        "rl_agent_config.json",
                        "encoder/config.json",
                        "tokenizer/tokenizer.json",
                        "tokenizer/tokenizer_config.json",
                    )
                },
            }
            references[name] = {file: sha(directory / file) for file in ("lora-history.json", "lora-calibration.json")}
        common = {
            "runtime_environment": self.environment.copy(),
            "all_frozen_sha256": sha(self.study / "all-frozen.json"),
            "study_plan_sha256": sha(self.study / "study-plan.json"),
            "training_source_sha256": self.study_plan["training_source_sha256"],
            "evaluation_source_sha256": "b" * 64,
            "runs": checkpoints,
            "device": "cuda",
            "timeout_seconds": 1800,
        }
        self.parity_plan = copy.deepcopy(common) | {
            "max_margin_error": 0.03,
            "require_unchanged_frozen_cutoff_decisions": True,
            "development_data_sha256": self.study_plan["data_sha256"]["512"],
            "references": references,
        }
        self.evaluation_plan = copy.deepcopy(common) | {
            "selected_recipe": "original",
            "test_sha256": sha(self.evidence / "test.json"),
            "questions_sha256": sha(self.evidence / "questions.json"),
            "evidence_manifest_sha256": sha(self.evidence / "manifest.json"),
        }
        self.seal()

    def seal(self):
        for name, record in self.records.items():
            write(self.study / name / "runtime-parity.json", record)
        write(self.study / "runtime-parity-plan.json", self.parity_plan)
        write(
            self.study / "runtime-parity-complete.json",
            {
                "runtime_parity_plan_sha256": sha(self.study / "runtime-parity-plan.json"),
                "parity_receipts_sha256": {
                    name: sha(self.study / name / "runtime-parity.json") for name in self.records
                },
            },
        )
        self.evaluation_plan.update(
            runtime_parity_plan_sha256=sha(self.study / "runtime-parity-plan.json"),
            runtime_parity_complete_sha256=sha(self.study / "runtime-parity-complete.json"),
        )
        write(self.study / "evaluation-plan.json", self.evaluation_plan)
        write(
            self.study / "evaluation-complete.json",
            {
                "evaluation_plan_sha256": sha(self.study / "evaluation-plan.json"),
                "evaluations_sha256": {name: sha(self.study / name / "evaluation-test.json") for name in self.records},
            },
        )

    def check(self):
        return check_evaluation_lifecycle(self.study, self.evidence, self.study_plan, self.choice)

    def test_complete_chain_replays_without_checkpoint_weights(self):
        self.assertFalse(list(self.study.rglob("model.safetensors")))
        result = self.check()
        self.assertEqual(result["unique_evaluated_runs"], 6)
        self.assertTrue(result["lifecycle_verified"])
        self.assertEqual(result["maximum_development_margin_error"], 0.0)

    def test_changed_runtime_flags_rejected_even_with_consistent_hashes(self):
        for plan in (self.parity_plan, self.evaluation_plan):
            plan["runtime_environment"]["TORCHDYNAMO_DISABLE"] = "0"
        self.seal()
        with self.assertRaisesRegex(ValueError, "runtime, frozen source"):
            self.check()

    def test_relaxed_parity_protocol_rejected(self):
        self.parity_plan["max_margin_error"] = 0.05
        self.seal()
        with self.assertRaisesRegex(ValueError, "parity protocol"):
            self.check()

    def test_missing_seed_rejected(self):
        del self.parity_plan["runs"][self.first]
        self.seal()
        with self.assertRaisesRegex(ValueError, "declared run set"):
            self.check()

    def test_checkpoint_hash_substitution_rejected(self):
        for plan in (self.parity_plan, self.evaluation_plan):
            plan["runs"][self.first]["checkpoint_files_sha256"]["model.safetensors"] = "0" * 64
        self.seal()
        with self.assertRaisesRegex(ValueError, "checkpoint, reference"):
            self.check()

    def test_changed_reference_file_rejected(self):
        write(self.study / self.first / "lora-history.json", [])
        with self.assertRaisesRegex(ValueError, "checkpoint, reference"):
            self.check()

    def test_score_drift_is_recomputed_instead_of_trusting_summary(self):
        self.records[self.first]["splits"]["calibration"]["scores"][0] = -0.9
        self.seal()
        with self.assertRaisesRegex(ValueError, "preserve development margins"):
            self.check()

    def test_small_margin_with_changed_cutoff_decision_is_rejected(self):
        measured = self.records[self.first]["splits"]["validation"]
        measured["scores"][1] = 0.51
        measured["max_margin_error"] = 0.51 - 0.49
        measured["decision_changes"] = 1
        self.seal()
        with self.assertRaisesRegex(ValueError, "preserve development margins"):
            self.check()

    def test_changed_calibration_labels_are_rejected(self):
        self.records[self.first]["splits"]["calibration"]["labels"] = [1, 0, 1, 1]
        self.seal()
        with self.assertRaisesRegex(ValueError, "development population"):
            self.check()

    def test_modified_prediction_file_breaks_completion_binding(self):
        write(self.study / self.first / "evaluation-test.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "prediction receipt"):
            self.check()

    def test_changed_evaluation_plan_breaks_completion_chain(self):
        self.evaluation_plan["device"] = "cpu"
        write(self.study / "evaluation-plan.json", self.evaluation_plan)
        with self.assertRaisesRegex(ValueError, "lifecycle hash chain"):
            self.check()

    def test_incomplete_marker_rejected_even_with_complete_receipts(self):
        write(self.study / "runtime-parity-incomplete.json", {"error_type": "TimeoutError"})
        with self.assertRaisesRegex(ValueError, "Incomplete attempts"):
            self.check()


if __name__ == "__main__":
    unittest.main()
