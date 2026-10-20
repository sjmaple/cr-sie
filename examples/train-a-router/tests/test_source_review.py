"""Source-review boundaries use synthetic collections, never held-out answers."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import solver
import solver_policies
import source_review
from data import SOURCE_HASHES, SOURCE_REVISION, digest, file_hash, render_request, write


class SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def collection(self, split="train"):
        directory = self.root / split
        directory.mkdir()
        questions = {
            "first": "What is one plus one?",
            "excluded": "Choose the missing diagram.",
            "last": "What is two plus two?",
        }
        references = {"first": "2", "excluded": "A", "last": "4"}
        rows = [
            {"id": identity, "request": render_request("SYNTHETIC", question)}
            for identity, question in questions.items()
        ]
        inputs = {split: rows}
        write(directory / "inputs.json", inputs)
        write(directory / "settings.json", solver.SETTINGS)
        write(directory / "gold-for-scoring-only.json", references)
        write(directory / "gold-review.json", {"synthetic": True})
        write(
            directory / "plan.json",
            {
                "inputs_sha256": digest(inputs),
                "settings_sha256": digest(solver.SETTINGS),
                "gold_sha256": digest(references),
            },
        )
        frozen = solver_policies.seal(
            {
                "format": solver_policies.FROZEN_FORMAT,
                "calibration_rows": [{"id": row["id"]} for row in rows],
                "embedding_calibration_credits": [1, 20, 3],
            }
        )
        write(directory / "policies-frozen.json", frozen)
        answers = directory / ("answers-" + split)
        answers.mkdir()
        write(answers / "collection.json", {"status": "complete", "completed": 6, "failed": 0, "missing": 0})
        write(
            answers / "collection-plan.json",
            {
                "ids": [row["id"] for row in rows],
                "split": split,
                "plan_sha256": file_hash(directory / "plan.json"),
                "settings_sha256": digest(solver.SETTINGS),
                "policies_frozen_sha256": file_hash(directory / "policies-frozen.json") if split == "test" else None,
                "self_hosted": False,
            },
        )
        # Excluded answer files intentionally do not exist: grading must not open them.
        for row in (rows[0], rows[2]):
            for arm in solver.SETTINGS["arms"]:
                record = solver.initial_record(row, arm, solver.SETTINGS) | {
                    "status": "complete",
                    "answer": r"\boxed{" + references[row["id"]] + "}",
                    "has_unknown_charges": False,
                    "credits": 2,
                    "observed_credits": 2,
                    "calls": [{"billing": "cloud", "receipt": {"credits": 2}}],
                }
                write(answers / f"{row['id']}-{arm}.json", record)
        review = {
            "format": source_review.FORMAT,
            "protocol": source_review.PROTOCOL,
            "finalized": True,
            "source_revision": SOURCE_REVISION,
            "source_hashes": SOURCE_HASHES,
            "solver_test_outcomes_accessed": False,
            "training_outcomes_already_available": True,
            "records": [
                {
                    "id": row["id"],
                    "split": split,
                    "question": questions[row["id"]],
                    "question_sha256": digest(questions[row["id"]]),
                    "request_sha256": digest(row["request"]),
                    "decision": "exclude" if row["id"] == "excluded" else "retain",
                    "reason": "Synthetic missing input" if row["id"] == "excluded" else "Synthetic valid input",
                    "original_reference": references[row["id"]],
                    "reference": references[row["id"]],
                    "semantic_kind": "scalar",
                }
                for row in rows
            ],
        }
        review_path = directory / "source-review.json"
        write(review_path, review)
        return directory, rows, review, review_path

    def test_excluded_answers_are_never_opened_or_graded(self):
        directory, _rows, _review, review_path = self.collection()
        real_read = source_review.read

        def guarded_read(path):
            self.assertNotIn("excluded-", Path(path).name)
            return real_read(path)

        with (
            patch.object(source_review, "read", side_effect=guarded_read),
            patch.object(
                source_review.teacher_format, "grade_reference", return_value={"correct": True, "status": "graded"}
            ) as grader,
        ):
            result = source_review.grade_collection(directory, "train", review_path)
        self.assertEqual([row["id"] for row in result["rows"]], ["first", "last"])
        self.assertEqual(result["collected_ids"], ["first", "excluded", "last"])
        self.assertEqual(result["excluded_ids"], ["excluded"])
        self.assertEqual(grader.call_count, 4)
        self.assertEqual(result["source_review_sha256"], file_hash(review_path))

    def test_split_request_text_and_question_mismatches_are_rejected(self):
        _directory, rows, review, _path = self.collection()
        with self.assertRaisesRegex(ValueError, "different split"):
            source_review.eligible_rows(rows, "test", review)
        spoofed = copy.deepcopy(rows)
        spoofed[0]["request_sha256"] = digest(spoofed[0]["request"])
        spoofed[0]["request"] += " Altered question."
        with self.assertRaisesRegex(ValueError, "text differs"):
            source_review.eligible_rows(spoofed, "train", review)
        changed = copy.deepcopy(review)
        changed["records"][0]["question"] = "A different problem"
        changed["records"][0]["question_sha256"] = digest("A different problem")
        with self.assertRaisesRegex(ValueError, "question differs"):
            source_review.eligible_rows(rows, "train", changed)

    def test_draft_or_nonhex_request_hash_cannot_enter_training(self):
        _directory, rows, review, _path = self.collection()
        draft = review | {"finalized": False}
        with self.assertRaisesRegex(ValueError, "pinned data"):
            source_review.eligible_rows(rows, "train", draft)
        changed = copy.deepcopy(review)
        changed["records"][0]["request_sha256"] = "z" * 64
        with self.assertRaisesRegex(ValueError, "Invalid source"):
            source_review.checked_review(changed)

    def access(self, directory, review_path):
        return {
            "format": "sie-reviewed-outcome-all-model-access-v1",
            "split": "test",
            "source_review_sha256": file_hash(review_path),
            "original_solver_plan_sha256": file_hash(directory / "plan.json"),
            "calibration_sha256": "a" * 64,
            "studies": [{"frozen_sha256": "b" * 64}],
        }

    def test_test_access_refusal_happens_before_opening_answers(self):
        directory, _rows, _review, review_path = self.collection("test")
        real_read = source_review.read
        opened_answers = []

        def guarded_read(path):
            if "answers-test" in Path(path).parts:
                opened_answers.append(str(path))
                self.fail("Test answers opened before access was validated")
            return real_read(path)

        access_path = directory / "access.json"
        valid = self.access(directory, review_path)
        invalid = [
            None,
            {},
            valid | {"source_review_sha256": "0" * 64},
            valid | {"studies": []},
            valid | {"split": "train"},
        ]
        with patch.object(source_review, "read", side_effect=guarded_read):
            for receipt in invalid:
                path = None
                if receipt is not None:
                    access_path.write_text(json.dumps(receipt))
                    path = access_path
                with self.assertRaisesRegex(ValueError, "test-access"):
                    source_review.grade_collection(directory, "test", review_path, test_access_path=path)
        self.assertEqual(opened_answers, [])

    def test_bound_test_access_allows_only_reviewed_population(self):
        directory, _rows, _review, review_path = self.collection("test")
        access = directory / "access.json"
        write(access, self.access(directory, review_path))
        with patch.object(
            source_review.teacher_format, "grade_reference", return_value={"correct": True, "status": "graded"}
        ):
            result = source_review.grade_collection(directory, "test", review_path, test_access_path=access)
        self.assertEqual(result["test_access_sha256"], file_hash(access))
        self.assertEqual([row["id"] for row in result["rows"]], ["first", "last"])

    def test_changed_special_reference_requires_explicit_generic_correction(self):
        _directory, _rows, review, _path = self.collection()
        row = review["records"][0]
        row.update({"semantic_kind": "clock_time", "reference": "A changed reference"})
        with (
            patch.object(source_review, "canonical_reference", return_value=("old", "clock_time")),
            self.assertRaisesRegex(ValueError, "source_correction"),
        ):
            source_review.checked_review(review)
        row["semantic_kind"] = "source_correction"
        with patch.object(
            source_review.teacher_format, "grade_reference", return_value={"correct": True, "status": "graded"}
        ) as grader:
            source_review.grade_reference_row(
                row["id"], row["question"], row["original_reference"], "synthetic", review
            )
        self.assertEqual(grader.call_args.args[1], "A changed reference")

    def test_calibration_costs_follow_verified_source_order(self):
        directory, _rows, _review, review_path = self.collection("calibration")
        with patch.object(
            source_review.teacher_format, "grade_reference", return_value={"correct": True, "status": "graded"}
        ):
            output = self.root / "amended"
            source_review.calibration_snapshot(directory, output, review_path)
            self.assertEqual(
                source_review.read(output / "policies-frozen.json")["embedding_calibration_credits"], [1, 3]
            )
            old = source_review.read(directory / "policies-frozen.json")
            old["calibration_rows"] = list(reversed(old["calibration_rows"]))
            old.pop("sha256")
            (directory / "policies-frozen.json").write_text(json.dumps(solver_policies.seal(old)))
            with self.assertRaisesRegex(ValueError, "cost order"):
                source_review.calibration_snapshot(directory, self.root / "bad-amendment", review_path)
            self.assertFalse((self.root / "bad-amendment").exists())

    def test_review_cannot_change_between_grading_and_provenance_receipt(self):
        directory, _rows, review, review_path = self.collection()

        def mutate_review(*_args):
            review["records"][0]["reason"] = "Changed while grading"
            review_path.write_text(json.dumps(review))
            return {"correct": True, "status": "graded"}

        with (
            patch.object(source_review.teacher_format, "grade_reference", side_effect=mutate_review),
            self.assertRaisesRegex(ValueError, "changed during grading"),
        ):
            source_review.grade_collection(directory, "train", review_path)


if __name__ == "__main__":
    unittest.main()
