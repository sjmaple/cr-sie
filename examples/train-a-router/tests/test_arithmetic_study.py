import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import arithmetic_study as study
from data import digest, file_hash, write


def replace(path, value):
    path.unlink(missing_ok=True)
    write(path, value)


class ArithmeticStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.run = self.root / "source", self.root / "run"
        self.counts = {"train": 10, "train_small": 5, "calibration": 2, "test": 2}
        self.count_patch = patch.object(study, "COUNTS", self.counts)
        self.count_patch.start()
        self.code_patch = patch.object(
            study,
            "code_hashes",
            return_value={
                name: "a" * 64
                for name in ("arithmetic_study.py", "arithmetic_answers.py", "teacher_format.py", "solver_answers.py")
            },
        )
        self.code_patch.start()
        self.inputs = {
            split: [{"id": f"{split}-{i}", "request": f"How many apples are in box {i}?"} for i in range(n)]
            for split, n in self.counts.items()
            if split != "train_small"
        }
        self.metadata = {
            split: [{"id": row["id"], "family": row["id"], "original_split": "train"} for row in rows]
            for split, rows in self.inputs.items()
        }
        write(self.source / "inputs.json", self.inputs)
        write(self.source / "rows.json", self.metadata)
        write(self.source / "train-small-ids.json", [row["id"] for row in self.inputs["train"][:5]])
        write(
            self.source / "gold-for-scoring-only.json",
            {row["id"]: "3" for rows in self.inputs.values() for row in rows},
        )
        self.reseal_source()
        self.embedding, self.native = self.root / "embedding", self.root / "native"
        for directory, names in (
            (self.embedding, ("selected-model.json", "frozen.json")),
            (self.native, ("trained.json", "frozen.json")),
        ):
            for name in names:
                write(directory / name, {})
        study.prepare(self.source, self.run, self.embedding, self.native)

    def tearDown(self):
        self.count_patch.stop()
        self.code_patch.stop()
        self.temporary.cleanup()

    def reseal_source(self):
        replace(
            self.source / "source-plan.json",
            {
                "format": study.SOURCE_FORMAT,
                "counts": self.counts,
                "exposed_source_plan_sha256": "b" * 64,
                "files_sha256": {
                    name: file_hash(self.source / name)
                    for name in ("inputs.json", "rows.json", "train-small-ids.json", "gold-for-scoring-only.json")
                },
            },
        )

    def test_source_mutation_and_cross_role_family_are_rejected(self):
        changed = study.read(self.source / "inputs.json")
        changed["train"][0]["request"] = "Different question"
        replace(self.source / "inputs.json", changed)
        with self.assertRaisesRegex(ValueError, "source changed"):
            study.checked_source(self.source)
        replace(self.source / "inputs.json", self.inputs)
        self.metadata["calibration"][0]["family"] = self.metadata["train"][0]["family"]
        replace(self.source / "rows.json", self.metadata)
        self.reseal_source()
        with self.assertRaisesRegex(ValueError, "family crosses"):
            study.checked_source(self.source)

    def test_nested_training_cannot_split_a_family(self):
        self.metadata["train"][5]["family"] = self.metadata["train"][0]["family"]
        replace(self.source / "rows.json", self.metadata)
        self.reseal_source()
        with self.assertRaisesRegex(ValueError, "splits a source family"):
            study.checked_source(self.source)

    def test_prepared_fold_assignment_and_subset_cannot_change(self):
        plan = study.read(self.run / "plan.json")
        plan["training_folds"][0] = (plan["training_folds"][0] + 1) % 5
        replace(self.run / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "folds changed"):
            study.checked_plan(self.source, self.run)

    def test_fresh_test_cannot_be_read_before_model_freeze(self):
        with self.assertRaises(FileNotFoundError) as failure:
            study.grade(self.source, self.run, self.root / "missing-answers", self.root / "prices", "test")
        self.assertEqual(Path(failure.exception.filename), self.run / "test-access.json")

    def test_all_unresolved_arms_and_all_physical_calls_are_preserved(self):
        record = {
            "answer": r"\boxed{3}",
            "status": "complete",
            "has_unknown_charges": False,
            "calls": [
                {"tokens": {"prompt_tokens": 10, "completion_tokens": 4}},
                {"tokens": {"prompt_tokens": 15, "completion_tokens": 2}},
            ],
        }
        rate = {"input_tokens": {"numerator": 1, "denominator": 1}, "output_tokens": {"numerator": 2, "denominator": 1}}
        with patch.object(
            study.arithmetic_answers,
            "grade_reference",
            side_effect=[
                {"correct": False, "status": "unit_contract_unresolved"},
                {"correct": False, "status": "grading_error"},
            ],
        ):
            row, flags = study._grade_pair(
                (
                    {"id": "one", "request": "Question"},
                    {"family": "f"},
                    "3",
                    {arm: record for arm in study.outcomes.ARMS},
                    {arm: "c" * 64 for arm in study.outcomes.ARMS},
                    {arm: rate for arm in study.outcomes.ARMS},
                )
            )
        self.assertEqual(len(flags), 2)
        self.assertEqual(row["cheap_credits"], 37)
        self.assertEqual(len(row["strong_cost_calls"]), 2)

    def test_grading_replays_verified_records_without_optional_acquisition_files(self):
        collection = self.root / "answers"
        manifest = {}
        for question in self.inputs["train"]:
            for arm in study.outcomes.ARMS:
                path = collection / f"{question['id']}-{arm}.json"
                write(path, {"answer": r"\boxed{3}"})
                manifest[path.name] = file_hash(path)
        write(
            collection / "collection-plan.json",
            {
                "source_plan_sha256": "b" * 64,
                "settings_sha256": digest(study.solver.SETTINGS),
                "records_sha256": manifest,
            },
        )
        write(
            collection / "collection.json",
            {
                "status": "complete",
                "completed": 20,
                "missing": 0,
                "unknown_execution_attempts": 0,
                "receipt_sha256": {"strong/model.yaml": "f" * 64},
            },
        )
        prices = self.root / "prices.json"
        write(prices, {})
        with (
            patch.object(study.solver, "validate_record"),
            patch.object(study.benefit_costs, "rates_from_snapshot", return_value=({"cheap": {}, "strong": {}}, [])),
            patch.object(study.benefit_costs, "physical_costs", return_value=[{"credits_proxy": 3}]),
            patch.object(
                study.arithmetic_answers, "grade_reference", return_value={"status": "graded", "correct": True}
            ),
        ):
            grades = study.grade(self.source, self.run, collection, prices, "train", workers=1)
        self.assertEqual(len(grades["rows"]), 10)
        self.assertEqual(grades["acquisition_receipt_mapping_sha256"], digest({"strong/model.yaml": "f" * 64}))
        self.assertTrue(all(row["cheap_correct"] and row["strong_correct"] for row in grades["rows"]))

    def test_embedding_overhead_is_allocated_once_per_batch(self):
        directory = self.root / "embeddings"
        write(
            directory / "collection.json", {"calls": [{"ids": ["a", "b"], "credits": 5}, {"ids": ["c"], "credits": 2}]}
        )
        self.assertEqual(study.overhead(directory, ["c", "b", "a"]), [2, 2.5, 2.5])
        replace(
            directory / "collection.json", {"calls": [{"ids": ["a", "b"], "credits": 5}, {"ids": ["b"], "credits": 2}]}
        )
        with self.assertRaisesRegex(ValueError, "multiple batches"):
            study.overhead(directory, ["a"])

    def test_native_context_preserves_shared_labels_and_family_fold(self):
        rows = [
            {"id": "a", "cheap_correct": True, "strong_correct": False, "cheap_credits": 2, "strong_credits": 5},
            {"id": "b", "cheap_correct": False, "strong_correct": True, "cheap_credits": 4, "strong_credits": 7},
        ]
        grades = {"prices_sha256": "d" * 64, "cost_scope": "token proxy", "rows": rows}
        write(self.run / "grades-train.json", grades)
        context = {
            "inputs": [{"id": "a", "request": "A"}, {"id": "b", "request": "B"}],
            "rows": rows,
            "folds": [0, 1],
            "grades": grades,
        }
        with (
            patch.object(study, "training", return_value=context),
            patch.object(study.native, "_train_cuda", return_value={"complete": True}) as trainer,
        ):
            study.train_native(self.source, self.run, 500)
        passed = trainer.call_args.args[2]
        self.assertEqual(passed["targets"].tolist(), [[1, 0], [0, 1]])
        self.assertEqual(passed["held"].tolist(), [True, False])
        self.assertEqual(passed["plan"]["selection_validation_ids"], ["a"])
        self.assertEqual(passed["costs"], [[2, 5], [4, 7]])


if __name__ == "__main__":
    unittest.main()
