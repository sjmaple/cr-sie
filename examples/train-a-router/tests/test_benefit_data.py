"""Source-only grading and input-boundary checks; no solver outputs or calls."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import benefit_data as benefit
import solver
from data import digest, file_hash, render_request, write


@unittest.skipUnless(importlib.util.find_spec("math_verify"), "Install the evaluation dependency group")
class TrainingReviewTests(unittest.TestCase):
    def grade(self, identity, answer):
        return benefit.grade_training(identity, benefit.RULES[identity][0], "\\boxed{" + answer + "}")

    def test_all_canonical_rules_grade_and_source_text_is_bound(self):
        for identity, (original, canonical, _) in benefit.RULES.items():
            with self.subTest(identity=identity):
                self.assertTrue(self.grade(identity, canonical)["correct"])
                with self.assertRaisesRegex(ValueError, "original source"):
                    benefit.grade_training(identity, original + " changed", canonical)
        self.assertNotIn("e2h-amc-eval-661", benefit.RULES)

    def test_two_clear_source_transcription_corrections(self):
        for identity, good, wrong in [
            ("e2h-amc-train-966", "251750000", "2517500"),
            ("e2h-amc-eval-2430", r"\frac27", r"\sqrt{\frac27}"),
        ]:
            self.assertTrue(self.grade(identity, good)["correct"])
            self.assertFalse(self.grade(identity, wrong)["correct"])

    def test_prime_approximation_uses_independent_count_and_declared_source_score(self):
        count, logarithm = benefit.prime_compositions()
        self.assertEqual(logarithm, benefit.RULES["e2h-amc-eval-2451"][1])
        # Even the subset using just2 and3 exceeds the supplied614.519 bits.
        subset = [1] + [0] * 2013
        for index in range(1, 2014):
            subset[index] = sum(subset[index - part] for part in (2, 3) if part <= index)
        self.assertGreater(subset[-1].bit_length(), 615)
        self.assertGreater(int(count), subset[-1])
        for answer in ("1129.418", "1129", "1130", "1120", "1140"):
            self.assertTrue(self.grade("e2h-amc-eval-2451", answer)["correct"], answer)
        for answer in ("614.519", "1100", "1150", "0", "-1129", "1/0", "x", "8=1129"):
            self.assertFalse(self.grade("e2h-amc-eval-2451", answer)["correct"], answer)

    def test_approximate_percent_accepts_source_rounding_and_correctly_stated_precision(self):
        for answer in ("80", r"80\%", "83", r"82.9\%", "82.93", r"\frac{3400}{41}\%", "3400/41"):
            self.assertTrue(self.grade("e2h-amc-eval-1984", answer)["correct"], answer)
        for answer in ("82", "82.5", "82.94", "80.1", r"0.83\%", "8=80", "0", "100"):
            self.assertFalse(self.grade("e2h-amc-eval-1984", answer)["correct"], answer)

    def test_percentage_questions_do_not_change_percentage_data_into_answers(self):
        for identity, (_, answer, kind) in benefit.RULES.items():
            if kind == "percentage":
                self.assertTrue(self.grade(identity, answer)["correct"])
                self.assertTrue(self.grade(identity, answer.removesuffix(r"\%"))["correct"])
        self.assertFalse(benefit.grade_training("e2h-amc-eval-1935", "37", r"\boxed{37\%}")["correct"])
        self.assertFalse(self.grade("e2h-amc-eval-759", r"0.75\%")["correct"])

    def test_compact_plus_minus_expands_both_ordered_complex_points(self):
        identity = "e2h-amc-train-230"
        self.assertTrue(self.grade(identity, benefit.RULES[identity][0])["correct"])
        self.assertTrue(self.grade(identity, benefit.RULES[identity][1])["correct"])
        self.assertFalse(self.grade(identity, r"(1,2),(4,2),(\frac52,2+\frac32 i)")["correct"])
        self.assertFalse(self.grade(identity, r"(2,1),(4,2),(\frac52,2\pm\frac32 i)")["correct"])
        self.assertFalse(self.grade(identity, r"(1,2),(4,2),(\frac52,2\pm\frac32 i),99")["correct"])

    def test_unreviewed_ambiguity_stays_visible(self):
        sources = {"new": {"problem": "Unknown meaning", "solution": "Unknown"}}
        _, _, review = benefit.review_training({"new": "(1,2)"}, sources)
        self.assertFalse(review["ready"])
        self.assertEqual(review["unresolved_ids"], ["new"])


class TrainingBoundaryTests(unittest.TestCase):
    def test_exact_source_join_exposes_only_id_and_request(self):
        source = {"problem": "What is2+2?", "answer": "4", "tag": "AMC12", "year": 2000, "solution": "Add."}
        row = {
            "id": "e2h-amc-train-0",
            "request": render_request(source["tag"], source["problem"]),
            "contest": source["tag"],
            "year": source["year"],
        }
        dataset = {"splits": {"train": [row]}}
        inputs, gold, _ = benefit.join_training(dataset, {"train": [source]})
        self.assertEqual(set(inputs["train"][0]), {"id", "request"})
        self.assertEqual(gold, {row["id"]: "4"})
        row["request"] += " Changed"
        with self.assertRaisesRegex(ValueError, "source/request join"):
            benefit.join_training(dataset, {"train": [source]})

    def fixture(self, directory):
        ids = ["synthetic-" + str(index) for index in range(511)]
        excluded = list(benefit.EXCLUSIONS)
        inputs = {"train": [{"id": identity, "request": "A synthetic problem."} for identity in ids]}
        eligibility = {
            "format": "sie-benefit-training-eligibility-v1",
            "source_only": True,
            "decided_before_training_outcomes": True,
            "original_pool_ids": ids[:20] + excluded + ids[20:],
            "eligible_train_ids": ids,
            "exclusions": [{"id": identity, **benefit.EXCLUSIONS[identity]} for identity in excluded],
        }
        write(directory / "inputs.json", inputs)
        write(directory / "settings.json", solver.SETTINGS)
        write(directory / "eligibility.json", eligibility)
        write(directory / "gold-preflight.json", {"synthetic": True})
        write(directory / "training-gold-adjudications.json", {"synthetic": True})
        write(
            directory / "gold-review.json",
            {"ready": True, "approved_ids": ids, "preflight_sha256": file_hash(directory / "gold-preflight.json")},
        )
        plan = {
            "format": "sie-benefit-training-source-v1",
            "source_counts": {"train": 512},
            "counts": {"train": 511},
            "eligible_train_ids": ids,
            "excluded_train_ids": excluded,
            "inputs_sha256": digest(inputs),
            "settings_sha256": digest(solver.SETTINGS),
            "eligibility_sha256": file_hash(directory / "eligibility.json"),
            "gold_review_sha256": file_hash(directory / "gold-review.json"),
            "training_gold_rules_sha256": file_hash(directory / "training-gold-adjudications.json"),
            **benefit.source_hashes(),
        }
        write(directory / "plan.json", plan)
        return plan

    def test_collection_validation_needs_no_answer_key_and_rejects_input_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            # Neither gold-for-scoring-only.json nor model answers exist.
            benefit.checked_training(directory)
            inputs = json.loads((directory / "inputs.json").read_text())
            inputs["train"][0]["request"] = "Tampered"
            (directory / "inputs.json").write_text(json.dumps(inputs))
            with self.assertRaisesRegex(ValueError, "inputs or settings changed"):
                benefit.checked_training(directory)

    def test_collection_rejects_changed_review_and_runtime_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            with (
                mock.patch.object(benefit, "source_hashes", return_value={"grader_source_sha256": "0" * 64}),
                self.assertRaisesRegex(ValueError, "source changed"),
            ):
                benefit.checked_training(directory)
            (directory / "eligibility.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "review changed"):
                benefit.checked_training(directory)

    def test_grade_refuses_overwriting_before_reading_any_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "grades-train.json"
            output.write_text("preserve")
            with self.assertRaises(FileExistsError):
                benefit.grade(directory)
            self.assertEqual(output.read_text(), "preserve")

    def completed_fixture(self, directory):
        plan = self.fixture(directory)
        inputs = benefit.read(directory / "inputs.json")
        gold = {row["id"]: "42" for row in inputs["train"]}
        write(directory / "gold-for-scoring-only.json", gold)
        plan.update({"gold_sha256": digest(gold), "grader": benefit.grader_metadata()})
        (directory / "plan.json").write_text(json.dumps(plan))
        row = inputs["train"][0]
        collection = directory / "answers-train"
        write(
            collection / "collection-plan.json",
            {
                "ids": [row["id"]],
                "split": "train",
                "plan_sha256": file_hash(directory / "plan.json"),
                "settings_sha256": digest(solver.SETTINGS),
                "policies_frozen_sha256": None,
                "self_hosted": False,
            },
        )
        write(collection / "collection.json", {"status": "complete", "completed": 2, "failed": 0})
        for arm in solver.SETTINGS["arms"]:
            answer = solver.initial_record(row, arm, solver.SETTINGS) | {
                "status": "complete",
                "observed_credits": 1,
                "credits": 1,
                "has_unknown_charges": False,
                "answer": r"\boxed{42}" if arm == "cheap" else r"\boxed{41}",
                "calls": [{"billing": "cloud", "receipt": {"credits": 1}}],
            }
            write(collection / f"{row['id']}-{arm}.json", answer)
        return row, plan

    @unittest.skipUnless(importlib.util.find_spec("math_verify"), "Install the evaluation dependency group")
    def test_grade_preserves_negative_benefit_and_binds_every_provenance_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row, plan = self.completed_fixture(directory)
            with redirect_stdout(io.StringIO()):
                benefit.grade(directory)
            grades = benefit.read(directory / "grades-train.json")
            self.assertEqual(grades["split"], "train")
            self.assertEqual(grades["plan_sha256"], file_hash(directory / "plan.json"))
            self.assertEqual(grades["training_gold_rules_sha256"], plan["training_gold_rules_sha256"])
            self.assertEqual(grades["gold_review_source_sha256"], benefit.source_hashes()["gold_review_source_sha256"])
            self.assertTrue(grades["rows"][0]["cheap_correct"])
            self.assertFalse(grades["rows"][0]["strong_correct"])
            path = directory / "answers-train" / f"{row['id']}-cheap.json"
            self.assertEqual(grades["rows"][0]["cheap_record_sha256"], file_hash(path))

    @unittest.skipUnless(importlib.util.find_spec("math_verify"), "Install the evaluation dependency group")
    def test_grade_rejects_wrong_request_and_unbound_answer_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row, _ = self.completed_fixture(directory)
            path = directory / "answers-train" / f"{row['id']}-cheap.json"
            answer = benefit.read(path)
            answer["request_sha256"] = "0" * 64
            path.write_text(json.dumps(answer))
            with self.assertRaisesRegex(ValueError, "frozen request"):
                benefit.grade(directory)
            (directory / "gold-for-scoring-only.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Answer key|answer key"):
                benefit.grade(directory)


if __name__ == "__main__":
    unittest.main()
