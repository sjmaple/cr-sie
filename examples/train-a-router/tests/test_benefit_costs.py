import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benefit_costs
import solver
from data import digest, file_hash, write


class CostProxyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.answers = self.root / "answers-train"
        self.answers.mkdir()
        self.inputs = {
            "train": [{"id": identity, "request": "Synthetic " + identity} for identity in ("first", "second")]
        }
        write(self.root / "inputs.json", self.inputs)
        write(self.root / "settings.json", solver.SETTINGS)
        write(
            self.root / "plan.json", {"inputs_sha256": digest(self.inputs), "settings_sha256": digest(solver.SETTINGS)}
        )
        write(
            self.answers / "collection-plan.json",
            {
                "ids": [row["id"] for row in self.inputs["train"]],
                "split": "train",
                "plan_sha256": file_hash(self.root / "plan.json"),
                "settings_sha256": digest(solver.SETTINGS),
                "policies_frozen_sha256": None,
                "self_hosted": True,
            },
        )
        write(self.answers / "collection.json", {"status": "complete", "completed": 4, "failed": 0})
        rows = []
        for identity, correct in (("first", True), ("second", False)):
            row = {"id": identity}
            for arm, model in benefit_costs.MODELS.items():
                call = {
                    "tokens": {
                        "prompt_tokens": 1 if arm == "cheap" else 5,
                        "completion_tokens": 1 if arm == "cheap" else 5,
                    },
                    "finish_reason": "stop",
                    "billing": "self_hosted",
                }
                record = {
                    "id": identity,
                    "arm": arm,
                    "model": model,
                    "status": "complete",
                    "has_unknown_charges": False,
                    "calls": [call, call] if arm == "cheap" else [call],
                    "request_sha256": digest("Synthetic " + identity),
                    "settings_sha256": digest(solver.SETTINGS),
                    "catalog_revision": solver.SETTINGS["arms"][arm]["revision"],
                    "answer": "Synthetic answer",
                    "observed_credits": 0,
                    "credits": None,
                }
                path = self.answers / f"{identity}-{arm}.json"
                write(path, record)
                row[f"{arm}_record_sha256"] = file_hash(path)
                row[f"{arm}_correct"] = correct
            rows.append(row)
        write(
            self.root / "grades-train.json",
            {
                "split": "train",
                "plan_sha256": file_hash(self.root / "plan.json"),
                "collection_summary_sha256": file_hash(self.answers / "collection.json"),
                "collection_plan_sha256": file_hash(self.answers / "collection-plan.json"),
                "rows": rows,
            },
        )
        items = []
        for arm, model in benefit_costs.MODELS.items():
            for unit, rate in (
                ("input_tokens", 2 if arm == "cheap" else 4),
                ("output_tokens", 3 if arm == "cheap" else 6),
            ):
                items.append(
                    {
                        "operation": "generate",
                        "model": model,
                        "profile": "default",
                        "unit": unit,
                        "region": "us",
                        "currency": "usd",
                        "usd_per_unit": {"numerator": str(rate), "denominator": "1000000"},
                    }
                )
        self.prices = self.root / "prices.json"
        write(
            self.prices,
            {"currency": "usd", "pricing_version": "fixture-v1", "rate_book_sha256": "a" * 64, "items": items},
        )

    def test_round_each_physical_call_and_include_finalizer(self):
        result = benefit_costs.build(self.root, self.prices)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["rows"][0]["cheap_credits_proxy"], 2)
        self.assertEqual(result["rows"][0]["strong_credits_proxy"], 5)
        self.assertEqual(result["conditional_means"]["cheap_wrong"]["mean_credits_proxy"], 2)
        self.assertEqual(result["price_source"]["sha256"], file_hash(self.prices))

    def test_changed_raw_answer_is_rejected(self):
        path = self.answers / "first-cheap.json"
        record = benefit_costs.read(path)
        record["calls"][0]["tokens"]["completion_tokens"] += 1
        path.unlink()
        write(path, record)
        with self.assertRaisesRegex(ValueError, "changed after grading"):
            benefit_costs.build(self.root, self.prices)

    def test_ambiguous_or_missing_tokens_are_rejected(self):
        rates, _ = benefit_costs.rates_from_snapshot(benefit_costs.read(self.prices))
        record = benefit_costs.read(self.answers / "first-cheap.json")
        record["has_unknown_charges"] = True
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            benefit_costs.physical_costs(record, rates["cheap"])
        record["has_unknown_charges"] = False
        record["calls"][0]["tokens"]["completion_tokens"] = None
        with self.assertRaisesRegex(ValueError, "integer token"):
            benefit_costs.physical_costs(record, rates["cheap"])

    def test_ambiguous_price_rows_are_rejected(self):
        prices = benefit_costs.read(self.prices)
        prices["items"].append(dict(prices["items"][0]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            benefit_costs.rates_from_snapshot(prices)

    def filtered_fixture(self):
        """Add one collected/excluded row without creating readable answer files."""
        self.inputs["train"].insert(1, {"id": "excluded", "request": "Source missing required information"})
        (self.root / "inputs.json").write_text(json.dumps(self.inputs))
        plan = benefit_costs.read(self.root / "plan.json") | {"inputs_sha256": digest(self.inputs)}
        (self.root / "plan.json").write_text(json.dumps(plan))
        binding = benefit_costs.read(self.answers / "collection-plan.json") | {
            "ids": ["first", "excluded", "second"],
            "plan_sha256": file_hash(self.root / "plan.json"),
        }
        (self.answers / "collection-plan.json").write_text(json.dumps(binding))
        summary = benefit_costs.read(self.answers / "collection.json") | {"completed": 6}
        (self.answers / "collection.json").write_text(json.dumps(summary))
        grades = benefit_costs.read(self.root / "grades-train.json") | {
            "plan_sha256": file_hash(self.root / "plan.json"),
            "collection_plan_sha256": file_hash(self.answers / "collection-plan.json"),
            "collection_summary_sha256": file_hash(self.answers / "collection.json"),
            "collected_ids": ["first", "excluded", "second"],
            "excluded_ids": ["excluded"],
        }
        path = self.root / "reviewed-grades.json"
        write(path, grades)
        return path

    def test_filtered_grades_price_only_retained_rows_and_preserve_default_file(self):
        original_hash = file_hash(self.root / "grades-train.json")
        grades = self.filtered_fixture()
        real_read = benefit_costs.read

        def reject_excluded(path):
            self.assertNotIn("excluded-", Path(path).name)
            return real_read(path)

        with patch.object(benefit_costs, "read", side_effect=reject_excluded):
            result = benefit_costs.build(self.root, self.prices, grades_path=grades)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["collected_ids"], ["first", "excluded", "second"])
        self.assertEqual(result["excluded_ids"], ["excluded"])
        self.assertEqual(
            result["conditional_means"]["cheap_correct"], {"n": 1, "sum_credits_proxy": 2, "mean_credits_proxy": 2}
        )
        self.assertEqual(result["conditional_means"]["strong_wrong"]["mean_credits_proxy"], 5)
        self.assertEqual(result["training_grades_sha256"], file_hash(grades))
        self.assertEqual(file_hash(self.root / "grades-train.json"), original_hash)

    def test_filtered_partition_and_complete_original_count_are_required(self):
        grades = self.filtered_fixture()
        original = benefit_costs.read(grades)
        cases = (
            original | {"excluded_ids": []},
            original | {"excluded_ids": ["excluded", "excluded"]},
            original | {"collected_ids": ["first", "second"]},
            original | {"rows": list(reversed(original["rows"]))},
        )
        for invalid in cases:
            grades.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "partition"):
                benefit_costs.build(self.root, self.prices, grades_path=grades)
        grades.write_text(json.dumps(original))
        summary = benefit_costs.read(self.answers / "collection.json") | {"completed": 4}
        (self.answers / "collection.json").write_text(json.dumps(summary))
        original["collection_summary_sha256"] = file_hash(self.answers / "collection.json")
        grades.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "complete paired collection"):
            benefit_costs.build(self.root, self.prices, grades_path=grades)

    def test_resealed_raw_record_with_wrong_request_is_rejected(self):
        path = self.answers / "first-cheap.json"
        record = benefit_costs.read(path) | {"request_sha256": digest("Another question")}
        path.write_text(json.dumps(record))
        grades_path = self.root / "grades-train.json"
        grades = benefit_costs.read(grades_path)
        grades["rows"][0]["cheap_record_sha256"] = file_hash(path)
        grades_path.write_text(json.dumps(grades))
        with self.assertRaisesRegex(ValueError, "frozen request"):
            benefit_costs.build(self.root, self.prices)


if __name__ == "__main__":
    unittest.main()
