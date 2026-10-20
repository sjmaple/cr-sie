"""Synthetic outcome learning, paper separation and pre-test freeze checks."""

import gzip
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import benefit
import solver_policies
from data import digest, file_hash
from serving import INSTRUCTION, MODEL, REVISION


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, allow_nan=False) + "\n")
    return path


def outcomes(split, count):
    return [
        {
            "id": f"{split}-{i}",
            "cheap_correct": i % 4 < 2,
            "strong_correct": True,
            "cheap_credits": 10,
            "strong_credits": 100,
        }
        for i in range(count)
    ]


class BenefitLearningTests(unittest.TestCase):
    def test_signed_targets_preserve_cheap_only_rescues_and_both_failures(self):
        rows = [
            {"id": str(i), "cheap_correct": cheap, "strong_correct": strong}
            for i, (cheap, strong) in enumerate(((True, False), (True, True), (False, False), (False, True)))
        ]
        np.testing.assert_array_equal(benefit.signed_targets(rows), [-1, 0, 0, 1])
        rows[0]["cheap_correct"] = 1
        with self.assertRaises(TypeError):
            benefit.signed_targets(rows)

    def test_cv_excludes_every_held_out_paper_and_refits_all_training_rows(self):
        rows = outcomes("train", 20)
        matrix = np.array([[i, i % 4, 1] for i in range(20)], dtype=float)
        groups = [f"paper-{i // 4}" for i in range(20)]
        calls = []
        ridge = benefit.policies.ridge

        def record(x, y, penalty):
            calls.append((x.copy(), y.copy(), penalty))
            return ridge(x, y, penalty)

        with patch.object(benefit.policies, "ridge", side_effect=record):
            head = benefit.fit_head(matrix, rows, groups)
        self.assertEqual(len(calls), 4 * 5 + 1)
        folds = np.array(head["folds"])
        targets = benefit.signed_targets(rows)
        for index, penalty in enumerate(benefit.PROTOCOL["penalties"]):
            for fold in range(5):
                x, y, observed_penalty = calls[5 * index + fold]
                np.testing.assert_array_equal(x, matrix[folds != fold])
                np.testing.assert_array_equal(y, targets[folds != fold])
                self.assertEqual(observed_penalty, penalty)
                fit_groups = {groups[int(row[0])] for row in x}
                held_groups = {group for group, assignment in zip(groups, folds, strict=True) if assignment == fold}
                self.assertFalse(fit_groups & held_groups)
        np.testing.assert_array_equal(calls[-1][0], matrix)
        np.testing.assert_array_equal(calls[-1][1], targets)
        self.assertEqual(head["penalty"], calls[-1][2])
        self.assertEqual(benefit.group_folds(groups), head["folds"])

    def test_exact_cv_ties_choose_stronger_regularization_and_predictions_stay_finite(self):
        rows = outcomes("train", 10)
        for row in rows:
            row["cheap_correct"] = row["strong_correct"]
        head = benefit.fit_head(np.zeros((10, 3)), rows, [f"paper-{i // 2}" for i in range(10)])
        self.assertEqual(head["penalty"], 10)
        self.assertEqual(head["weights"], [0, 0, 0])
        self.assertEqual(head["bias"], 0)
        with self.assertRaisesRegex(ValueError, "five distinct"):
            benefit.fit_head(np.zeros((10, 3)), rows, ["one-paper"] * 10)
        with self.assertRaisesRegex(ValueError, "finite"):
            benefit.fit_head(np.full((10, 3), np.nan), rows, [f"paper-{i // 2}" for i in range(10)])


class BenefitLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline, self.original_dir, self.training, self.run = (
            self.root / name for name in ("baseline", "original", "training", "head")
        )
        # Exercise the full lifecycle on small vectors without downloading models.
        protocol = patch.dict(
            benefit.PROTOCOL,
            {"training_pool_rows": 20, "training_rows": 19, "excluded_training_ids": ["train-0"], "dimensions": 3},
        )
        protocol.start()
        self.addCleanup(protocol.stop)
        verify = patch.object(benefit.evidence, "verify")
        verify.start()
        self.addCleanup(verify.stop)
        self.dataset = {
            "splits": {
                split: [
                    {
                        "id": f"{split}-{i}",
                        "request": f"{split} problem number {i}",
                        "family": f"{split}-family-{i}",
                        "paper_group": f"{split}-paper-{i // 4}",
                        "label": i % 2,
                        "year": {"train": 2017, "validation": 2018, "calibration": 2019, "test": 2020}[split],
                    }
                    for i in range(count)
                ]
                for split, count in (("train", 20), ("validation", 4), ("calibration", 8), ("test", 8))
            }
        }
        save(self.baseline / "data.json", self.dataset)
        save(self.baseline / "manifest.json", {})
        for directory, splits in (
            ("development-embeddings", ("train", "validation", "calibration")),
            ("test-embeddings", ("test",)),
        ):
            rows = [row for split in splits for row in self.dataset["splits"][split]]
            path = self.baseline / directory
            save(
                path / "plan.json",
                {
                    "model": MODEL,
                    "revision": REVISION,
                    "instruction": INSTRUCTION,
                    "ids": [row["id"] for row in rows],
                    "input_sha256": {row["id"]: digest(row["request"]) for row in rows},
                },
            )
            with gzip.open(path / "vectors.jsonl.gz", "wt") as handle:
                for row in rows:
                    index = int(row["id"].rsplit("-", 1)[1])
                    handle.write(
                        json.dumps({"id": row["id"], "vector": [1, index % 4 - 1.5, 0.01 * (index // 4)]}) + "\n"
                    )
        self.original_plan = save(self.original_dir / "plan.json", {"settings_sha256": "1" * 64})
        score_splits = {split: self.dataset["splits"][split] for split in ("calibration", "test")}
        scores = solver_policies.seal(
            {
                "format": solver_policies.SCORE_FORMAT,
                "dataset_sha256": digest(self.dataset),
                "ids": {split: [row["id"] for row in rows] for split, rows in score_splits.items()},
                "input_sha256": {
                    split: {row["id"]: digest(row["request"]) for row in rows} for split, rows in score_splits.items()
                },
                "stages": {
                    name: {
                        "family": "embedding" if name.startswith("ridge") else "native",
                        "original_threshold": 0.5,
                        "scores": {split: [i % 2 for i in range(len(rows))] for split, rows in score_splits.items()},
                    }
                    for name in solver_policies.STAGES
                },
            }
        )
        self.grading = {
            "grader_source_sha256": "2" * 64,
            "gold_review_source_sha256": "3" * 64,
            "gold_review_sha256": "4" * 64,
        }
        self.original = save(
            self.original_dir / "policies-frozen.json",
            solver_policies.seal(
                {
                    "format": solver_policies.FROZEN_FORMAT,
                    "analysis_mode": "cost",
                    "plan_sha256": file_hash(self.original_plan),
                    "source_scores": scores,
                    "source_code_sha256": {
                        name: file_hash(Path(benefit.__file__).with_name(name))
                        for name in ("solver_policies.py", "solver_report.py")
                    },
                    "calibration_grades_sha256": "5" * 64,
                    "calibration_rows": outcomes("calibration", 8),
                    "embedding_calibration_credits": [2] * 8,
                    "embedding_calibration_credits_sha256": "6" * 64,
                    "credit_usd": 1e-5,
                    **self.grading,
                }
            ),
        )
        self.eligible_rows = self.dataset["splits"]["train"][1:]
        self.eligibility = save(
            self.training / "eligibility.json",
            {
                "format": "sie-benefit-training-eligibility-v1",
                "original_pool_ids": [row["id"] for row in self.dataset["splits"]["train"]],
                "eligible_train_ids": [row["id"] for row in self.eligible_rows],
                "exclusions": [{"id": "train-0", "reason": "Missing choices leave the source question ambiguous"}],
                "source_only": True,
                "decided_before_training_outcomes": True,
                "source_revision": benefit.SOURCE_REVISION,
                "source_hashes": benefit.SOURCE_HASHES,
                "dataset_sha256": digest(self.dataset),
                "training_gold_review_source_sha256": "9" * 64,
            },
        )
        inputs = {"train": [{key: row[key] for key in ("id", "request")} for row in self.eligible_rows]}
        self.training_inputs = save(self.training / "inputs.json", inputs)
        self.training_plan = save(
            self.training / "plan.json",
            {
                "dataset_sha256": digest(self.dataset),
                "settings_sha256": "1" * 64,
                "inputs_sha256": digest(inputs),
                "source_counts": {"train": 20},
                "counts": {"train": 19},
                "eligible_train_ids": [row["id"] for row in self.eligible_rows],
                "excluded_train_ids": ["train-0"],
                "eligibility_sha256": file_hash(self.eligibility),
                **self.grading,
                "gold_review_sha256": "7" * 64,
                "training_gold_rules_sha256": "8" * 64,
                "training_gold_review_source_sha256": "9" * 64,
            },
        )
        self.training_grades = save(
            self.training / "grades-train.json",
            {
                "split": "train",
                "plan_sha256": file_hash(self.training_plan),
                **self.grading,
                "gold_review_sha256": "7" * 64,
                "training_gold_rules_sha256": "8" * 64,
                "training_gold_review_source_sha256": "9" * 64,
                "rows": outcomes("train", 20)[1:],
            },
        )
        plan = benefit.prepare(self.baseline, self.original, self.eligibility, attest_test_outcomes_unopened=True)
        save(self.run / "plan.json", plan)

    def freeze(self):
        frozen = benefit.freeze(
            self.run, self.baseline, self.original, self.training_plan, self.training_inputs, self.training_grades
        )
        save(self.run / "frozen.json", frozen)
        return frozen

    def test_freeze_reads_training_and_calibration_only_even_with_held_test_marker(self):
        marker = save(self.original_dir / "grades-test.json", {"held": "test outcomes remain unopened"})
        read = benefit.read
        opened = []

        def guarded(path):
            opened.append(Path(path))
            if Path(path) == marker:
                self.fail("Freeze attempted to read test answers")
            return read(path)

        with patch.object(benefit, "read", side_effect=guarded):
            frozen = self.freeze()
        self.assertNotIn(marker, opened)
        self.assertEqual(frozen["head"]["training_ids"], [row["id"] for row in self.eligible_rows])
        self.assertEqual(frozen["training_pool_rows"], 20)
        self.assertEqual(frozen["training_rows"], 19)
        self.assertEqual(len(frozen["plan"]["populations"]["train_pool"]), 20)
        self.assertEqual(frozen["plan"]["eligibility_sha256"], file_hash(self.eligibility))
        self.assertEqual(frozen["head"]["folds"], frozen["plan"]["training_folds"])
        self.assertEqual(tuple(frozen["selections"]), ("1", "0", "2", "5"))
        self.assertEqual(frozen["training_grade_provenance"]["training_gold_rules_sha256"], "8" * 64)
        self.assertEqual(frozen["training_artifacts_sha256"]["grades"], file_hash(self.training_grades))
        self.assertEqual(len(frozen["scores"]["test"]), 8)
        self.assertFalse((self.run / "test-access.json").exists())

    def test_changed_training_ids_settings_and_review_hashes_fail_before_fit(self):
        original_plan, original_grades = benefit.read(self.training_plan), benefit.read(self.training_grades)
        changes = [
            ("grades", {"rows": list(reversed(original_grades["rows"]))}),
            ("grades", {"grader_source_sha256": "a" * 64}),
            ("grades", {"training_gold_rules_sha256": None}),
            ("grades", {"training_gold_rules_sha256": "e" * 64}),
            ("grades", {"gold_review_sha256": "e" * 64}),
            ("plan", {"settings_sha256": "b" * 64}),
            ("plan", {"eligibility_sha256": "c" * 64}),
            ("plan", {"excluded_train_ids": ["train-0", "train-1"]}),
            ("plan", {"training_gold_review_source_sha256": "e" * 64}),
        ]
        for kind, change in changes:
            with self.subTest(kind=kind, change=list(change)):
                save(self.training_plan, original_plan)
                save(self.training_grades, original_grades)
                save(
                    self.training_plan if kind == "plan" else self.training_grades,
                    (original_plan if kind == "plan" else original_grades) | change,
                )
                if kind == "plan":
                    save(self.training_grades, original_grades | {"plan_sha256": file_hash(self.training_plan)})
                with patch.object(benefit, "fit_head") as fit:
                    with self.assertRaises(ValueError):
                        self.freeze()
                    fit.assert_not_called()

    def test_modified_cached_features_are_rejected_before_fitting(self):
        path = self.baseline / "development-embeddings/vectors.jsonl.gz"
        with path.open("ab") as handle:
            handle.write(b"changed")
        with patch.object(benefit, "fit_head") as fit:
            with self.assertRaisesRegex(ValueError, "vectors changed"):
                self.freeze()
            fit.assert_not_called()

    def test_report_cannot_open_test_grades_without_matching_access_receipt(self):
        self.freeze()
        test = self.root / "must-not-open-test.json"
        read = benefit.read

        def guarded(path):
            if Path(path) == test:
                self.fail("Test opened before access chain was validated")
            return read(path)

        with patch.object(benefit, "read", side_effect=guarded):
            with self.assertRaises(FileNotFoundError):
                benefit.report(self.run, test)
            access = benefit.authorize_test(self.run)
            access.pop("sha256")
            save(self.run / "test-access.json", benefit.seal(access | {"frozen_sha256": "a" * 64}))
            with self.assertRaisesRegex(ValueError, "Test-access receipt"):
                benefit.report(self.run, test)

    def test_report_keeps_frozen_policy_when_test_quality_loss_exceeds_target(self):
        frozen = self.freeze()
        save(self.run / "test-access.json", benefit.authorize_test(self.run))
        rows = outcomes("test", 8)
        for row in rows:
            row["cheap_correct"] = False
        grades = save(
            self.root / "test-grades.json",
            {
                "split": "test",
                "plan_sha256": file_hash(self.original_plan),
                "policies_frozen_sha256": file_hash(self.original),
                **self.grading,
                "rows": rows,
            },
        )
        select = benefit.solver_report.select_cutoff

        def only_calibration(rows, *args, **kwargs):
            self.assertTrue(all(row["id"].startswith("calibration-") for row in rows))
            return select(rows, *args, **kwargs)

        with patch.object(benefit.solver_report, "select_cutoff", side_effect=only_calibration):
            measured = benefit.report(self.run, grades)
        self.assertEqual(measured["policies"]["1"]["cutoff"], frozen["selections"]["1"]["cutoff"])
        self.assertGreater(measured["policies"]["1"]["metrics"]["gap_vs_strong_pp"], 1)
        self.assertNotIn("net_total_cost_usd_per1000", measured["policies"]["1"]["metrics"])
        credits = save(
            self.root / "test-router-credits.json",
            {
                "plan_sha256": file_hash(self.original_plan),
                "model": MODEL,
                "revision": REVISION,
                "batch_size": 1,
                "test": [{"id": row["id"], "credits": 2} for row in rows],
            },
        )
        priced = benefit.report(self.run, grades, credits)["policies"]["1"]["metrics"]
        self.assertAlmostEqual(priced["router_cost_usd_per1000"], 0.02)
        self.assertAlmostEqual(priced["net_total_cost_usd_per1000"], priced["total_solver_cost_usd_per1000"] + 0.02)
        save(grades, benefit.read(grades) | {"gold_review_sha256": "c" * 64})
        with self.assertRaisesRegex(ValueError, "grading experiment"):
            benefit.report(self.run, grades)

    def test_resealed_cutoff_tamper_and_second_freeze_are_rejected(self):
        frozen = self.freeze()
        with self.assertRaises(FileExistsError):
            self.freeze()
        frozen.pop("sha256")
        frozen["selections"]["1"]["cutoff"] += 1
        save(self.run / "frozen.json", benefit.seal(frozen))
        with self.assertRaisesRegex(ValueError, "calibration-only selection"):
            benefit.authorize_test(self.run)

    def test_prepare_requires_unopened_test_attestation(self):
        with self.assertRaisesRegex(ValueError, "explicit assertion"):
            benefit.prepare(self.baseline, self.original, self.eligibility, attest_test_outcomes_unopened=False)

    def test_eligibility_cannot_drop_more_rows_reorder_or_follow_training_outcomes(self):
        original = benefit.read(self.eligibility)
        for change in (
            {"source_only": False},
            {"decided_before_training_outcomes": False},
            {"eligible_train_ids": original["eligible_train_ids"][1:]},
            {"eligible_train_ids": list(reversed(original["eligible_train_ids"]))},
            {"exclusions": original["exclusions"] + [{"id": "train-1", "reason": "Additional drop"}]},
            {"source_revision": "other"},
        ):
            with self.subTest(change=list(change)):
                save(self.eligibility, original | change)
                with self.assertRaisesRegex(ValueError, "Eligibility"):
                    benefit.prepare(self.baseline, self.original, self.eligibility, attest_test_outcomes_unopened=True)

    def test_existing_cli_output_is_preserved_before_loading_data(self):
        output = self.run / "plan.json"
        before = output.read_bytes()
        args = [
            "benefit.py",
            "prepare",
            "--run",
            str(self.run),
            "--baseline",
            "absent",
            "--original-frozen",
            "absent",
            "--eligibility",
            "absent",
            "--attest-test-outcomes-unopened",
        ]
        with patch("sys.argv", args), patch.object(benefit, "prepare") as prepare, redirect_stdout(io.StringIO()):
            with self.assertRaises(FileExistsError):
                benefit.main()
            prepare.assert_not_called()
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
