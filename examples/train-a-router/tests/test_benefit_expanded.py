"""Expanded inference gates bind source review without reading answer keys."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benefit_expanded as expanded
import solver
from data import digest, file_hash, write


class ExpandedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.inputs = {"train": [{"id": "old0", "request": "One"}, {"id": "new0", "request": "Two"}]}
        self.eligibility = {
            "source_only": True,
            "decided_before_new_training_outcomes": True,
            "reused_training_outcomes_already_available": True,
            "original_pool_ids": ["old0", "old1", "new0"],
            "eligible_train_ids": ["old0", "new0"],
            "exclusions": [{"id": "old1", "reason": "Source-only template overlap"}],
        }
        write(self.directory / "inputs.json", self.inputs)
        write(self.directory / "settings.json", solver.SETTINGS)
        write(self.directory / "source-leakage-review.json", {"source_only": True})
        self.eligibility["leakage_review_sha256"] = file_hash(self.directory / "source-leakage-review.json")
        write(self.directory / "eligibility.json", self.eligibility)
        write(self.directory / "reuse.json", {"ids": ["old0"]})
        write(self.directory / "training-rows.json", [])
        write(self.directory / "expansion-audit.json", {})
        write(self.directory / "training-gold-adjudications.json", {"answer_bearing": True})
        # The inference gate must hash this file, not parse its answer-bearing contents.
        (self.directory / "gold-preflight.json").write_bytes(b"not valid JSON or an answer key")
        write(
            self.directory / "gold-review.json",
            {
                "preflight_sha256": file_hash(self.directory / "gold-preflight.json"),
                "ready": True,
                "approved_ids": ["old0", "new0"],
            },
        )
        self.plan = {
            "format": expanded.FORMAT,
            "inputs_sha256": digest(self.inputs),
            "settings_sha256": digest(solver.SETTINGS),
            "counts": {"train": 2},
            "source_counts": {"train": 3},
            "eligible_train_ids": ["old0", "new0"],
            "excluded_train_ids": ["old1"],
            "new_train_ids": ["new0"],
            "grader_source_sha256": "a" * 64,
        }
        for key, name in (
            ("eligibility_sha256", "eligibility.json"),
            ("leakage_review_sha256", "source-leakage-review.json"),
            ("reuse_sha256", "reuse.json"),
            ("training_rows_sha256", "training-rows.json"),
            ("expansion_audit_sha256", "expansion-audit.json"),
            ("training_gold_rules_sha256", "training-gold-adjudications.json"),
            ("gold_review_sha256", "gold-review.json"),
        ):
            self.plan[key] = file_hash(self.directory / name)
        write(self.directory / "plan.json", self.plan)
        mock = patch.object(expanded, "source_hashes", return_value={"grader_source_sha256": "a" * 64})
        mock.start()
        self.addCleanup(mock.stop)

    def replace_eligibility(self):
        (self.directory / "eligibility.json").unlink()
        write(self.directory / "eligibility.json", self.eligibility)
        self.plan["eligibility_sha256"] = file_hash(self.directory / "eligibility.json")
        (self.directory / "plan.json").unlink()
        write(self.directory / "plan.json", self.plan)

    def test_inference_needs_no_answer_key_or_parsed_preflight(self):
        _, inputs, _ = expanded.checked_training(self.directory)
        self.assertEqual(inputs, self.inputs)
        self.assertFalse((self.directory / "gold-for-scoring-only.json").exists())

    def test_revised_eligibility_must_acknowledge_existing_outcomes(self):
        self.eligibility.pop("reused_training_outcomes_already_available")
        self.eligibility["decided_before_training_outcomes"] = True
        self.replace_eligibility()
        with self.assertRaisesRegex(ValueError, "eligibility"):
            expanded.checked_training(self.directory)

    def test_resealed_duplicate_population_is_rejected(self):
        self.eligibility["original_pool_ids"] = ["old0", "old0", "new0"]
        self.replace_eligibility()
        with self.assertRaisesRegex(ValueError, "eligibility"):
            expanded.checked_training(self.directory)

    def test_leakage_receipt_bytes_are_required(self):
        (self.directory / "source-leakage-review.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            expanded.checked_training(self.directory)


class SourceDecisionTests(unittest.TestCase):
    def setUp(self):
        self.gold = {identity: "1" for identity in ("old", "new1", "new2", "invalid", "leak")}
        self.sources = {identity: {"problem": identity, "solution": "Solution"} for identity in self.gold}
        self.leakage = {
            "format": "sie-expanded-source-leakage-review-v1",
            "source_only": True,
            "solver_outcomes_read_for_decisions": False,
            "decided_before_new_training_outcomes": True,
            "reused_training_outcomes_already_available": True,
            "exclusions": {"leak": {"reason": "Held-out question template"}},
            "duplicate_groups": [["new1", "new2"], ["old", "new2"]],
            "candidate_reviews": [],
        }
        self.mandatory = {"invalid": {"original_gold": "1", "reason": "Missing choices"}}
        mock = patch.object(expanded.benefit_data, "EXCLUSIONS", {})
        mock.start()
        self.addCleanup(mock.stop)

    def test_connected_duplicates_keep_original_before_new_rows(self):
        excluded, decisions = expanded.reviewed_exclusions(
            self.gold, self.sources, {"old"}, self.leakage, self.mandatory
        )
        self.assertEqual(set(excluded), {"new1", "new2", "invalid", "leak"})
        self.assertEqual(decisions[0]["retained_id"], "old")

    def test_new_duplicate_representative_uses_fixed_digest(self):
        _, decisions = expanded.reviewed_exclusions(self.gold, self.sources, set(), self.leakage, self.mandatory)
        expected = min(("old", "new1", "new2"), key=lambda identity: digest(["expanded-source-duplicate-v1", identity]))
        self.assertEqual(decisions[0]["retained_id"], expected)

    def test_changed_source_cannot_reuse_review(self):
        self.leakage["candidate_reviews"] = [{"source_problem_sha256": {"old": digest("Different question")}}]
        with self.assertRaisesRegex(ValueError, "problem differs"):
            expanded.reviewed_exclusions(self.gold, self.sources, {"old"}, self.leakage, self.mandatory)


if __name__ == "__main__":
    unittest.main()
