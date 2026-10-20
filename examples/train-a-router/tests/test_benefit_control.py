"""Synthetic checks for the fixed competition-only control; no model outputs."""

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import benefit_control as control
from data import digest, file_hash


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


class MetadataControlTests(unittest.TestCase):
    def test_publication_downloader_pins_are_not_part_of_the_model_code_contract(self):
        self.assertNotIn("evidence.py", control.CODE_FILES)
        self.assertIn("solver_report.py", control.CODE_FILES)

    def test_fixed_shrinkage_preserves_negative_benefit_and_uses_global_unknown_fallback(self):
        model = control.fit_metadata(["A", "A", "B"], [1, 1, -1])
        self.assertEqual(model["smoothing"], 20)
        self.assertEqual(model["global_mean"], 1 / 3)
        self.assertAlmostEqual(model["competitions"]["A"]["score"], (2 + 20 / 3) / 22)
        self.assertAlmostEqual(model["competitions"]["B"]["score"], (-1 + 20 / 3) / 21)
        self.assertEqual(control.predict(model, ["new"]), [1 / 3])
        self.assertGreater(control.predict(model, ["A"])[0], control.predict(model, ["B"])[0])

    def test_invalid_labels_and_missing_metadata_are_not_coerced(self):
        for contests, targets in (([], []), (["A"], [True]), (["A"], [2]), (["A"], [1.0]), ([""], [1]), (["A"], [])):
            with self.subTest(contests=contests, targets=targets), self.assertRaises(ValueError):
                control.fit_metadata(contests, targets)


class ControlLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline, self.head, self.run = (self.root / name for name in ("baseline", "head", "control"))
        self.dataset = {"splits": {}}
        for split, count, year in (
            ("train", 10, 2017),
            ("validation", 4, 2018),
            ("calibration", 4, 2019),
            ("test", 4, 2020),
        ):
            self.dataset["splits"][split] = [
                {
                    "id": f"{split}-{index}",
                    "request": f"{split} problem {index}",
                    "contest": "A" if index % 2 == 0 else "B",
                    "label": index % 2,
                    "year": year,
                    "family": f"{split}-family-{index}",
                    "paper_group": f"{split}-paper-{index // 2}",
                }
                for index in range(count)
            ]
        save(self.baseline / "data.json", self.dataset)
        save(self.baseline / "manifest.json", {})
        self.head_plan = {
            "dataset_sha256": digest(self.dataset),
            "eligibility_sha256": "1" * 64,
            "original_solver_plan_sha256": "2" * 64,
            "original_frozen_sha256": "3" * 64,
            "populations": {
                split: [
                    {"id": row["id"], "request_sha256": digest(row["request"])} for row in self.dataset["splits"][split]
                ]
                for split in ("train", "calibration", "test")
            },
        }
        save(self.head / "plan.json", self.head_plan)
        self.grading = {
            "grader_source_sha256": "4" * 64,
            "gold_review_source_sha256": "5" * 64,
            "gold_review_sha256": "6" * 64,
        }
        self.head_frozen = {
            "training_pool_rows": 10,
            "plan": self.head_plan,
            "plan_file_sha256": file_hash(self.head / "plan.json"),
            "head": {"training_ids": [row["id"] for row in self.dataset["splits"]["train"]], "targets": [1, -1] * 5},
            "calibration_rows": self.outcomes("calibration"),
            "credit_usd": 1e-5,
            "calibration_grades_sha256": "7" * 64,
            "grading_provenance": self.grading,
        }
        for replacement in (
            patch.object(control.evidence, "verify"),
            patch.object(control.benefit, "_checked_plan", side_effect=lambda plan: plan),
            patch.object(
                control.benefit, "_checked_frozen", side_effect=lambda head: control.read(Path(head) / "frozen.json")
            ),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        save(self.run / "plan.json", control.prepare(self.head, self.baseline))

    def outcomes(self, split):
        return [
            {
                "id": row["id"],
                "cheap_correct": index % 2 == 1,
                "strong_correct": True,
                "cheap_credits": 10,
                "strong_credits": 100,
            }
            for index, row in enumerate(self.dataset["splits"][split])
        ]

    def freeze(self):
        save(self.head / "frozen.json", self.head_frozen)
        frozen = control.freeze(self.run, self.head, self.baseline)
        save(self.run / "frozen.json", frozen)
        return frozen

    def access(self, *, seconds=-0):
        access = control.benefit.seal(
            {
                "format": control.benefit.ACCESS_FORMAT,
                "authorized_at": (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat(),
                "frozen_sha256": file_hash(self.head / "frozen.json"),
                "plan_sha256": file_hash(self.head / "plan.json"),
                "source_code_sha256": control.benefit.code_hashes(),
            }
        )
        save(self.head / "test-access.json", access)

    def test_preparation_reads_metadata_only_and_refuses_post_fit_registration(self):
        self.assertFalse((self.head / "frozen.json").exists())
        plan = control.read(self.run / "plan.json")
        self.assertEqual(set(plan["populations"]["train"][0]), {"id", "contest"})
        self.assertEqual(plan["head_plan_sha256"], file_hash(self.head / "plan.json"))
        save(self.head / "frozen.json", self.head_frozen)
        with self.assertRaisesRegex(ValueError, "before the embedding head"):
            control.prepare(self.head, self.baseline)

    def test_freeze_fits_only_training_targets_and_calibrates_with_no_router_api_cost(self):
        marker = save(self.root / "must-not-open-test-grades.json", {"held": True})
        read = control.read

        def guarded(path):
            self.assertNotEqual(Path(path), marker)
            return read(path)

        with patch.object(control, "read", side_effect=guarded):
            frozen = self.freeze()
        self.assertEqual(frozen["model"]["n"], 10)
        self.assertEqual(frozen["model"]["global_mean"], 0)
        self.assertEqual(frozen["scores"]["test"], [0.2, -0.2, 0.2, -0.2])
        self.assertEqual(frozen["selections"]["1"]["mode"], "threshold")
        self.assertEqual(frozen["selections"]["1"]["calibration"]["router_cost_usd_per1000"], 0)

    def test_late_freeze_and_metadata_changes_are_rejected(self):
        save(self.head / "frozen.json", self.head_frozen)
        self.access()
        with self.assertRaises(FileExistsError):
            control.freeze(self.run, self.head, self.baseline)
        (self.head / "test-access.json").unlink()
        self.dataset["splits"]["train"][0]["contest"] = "Changed"
        save(self.baseline / "data.json", self.dataset)
        with self.assertRaisesRegex(ValueError, "dataset"):
            control.freeze(self.run, self.head, self.baseline)

    def test_report_validates_access_before_opening_grades_and_never_retunes_test(self):
        frozen = self.freeze()
        test_path = self.root / "test-grades.json"
        with self.assertRaises(FileNotFoundError):
            control.report(self.run, self.head, self.baseline, test_path)
        self.access(seconds=-60)
        with self.assertRaisesRegex(ValueError, "before matching head test access"):
            control.report(self.run, self.head, self.baseline, test_path)
        self.access()
        rows = self.outcomes("test")
        for row in rows:
            row["cheap_correct"] = False
        save(
            test_path,
            {
                "split": "test",
                "plan_sha256": self.head_plan["original_solver_plan_sha256"],
                "policies_frozen_sha256": self.head_plan["original_frozen_sha256"],
                **self.grading,
                "rows": rows,
            },
        )
        measured = control.report(self.run, self.head, self.baseline, test_path)
        self.assertEqual(measured["policies"]["1"]["cutoff"], frozen["selections"]["1"]["cutoff"])
        self.assertEqual(measured["policies"]["1"]["metrics"]["gap_vs_strong_pp"], 50)
        self.assertEqual(measured["policies"]["1"]["metrics"]["router_cost_usd_per1000"], 0)
        save(test_path, control.read(test_path) | {"gold_review_sha256": "a" * 64})
        with self.assertRaisesRegex(ValueError, "paired solver experiment"):
            control.report(self.run, self.head, self.baseline, test_path)

    def test_resealed_control_model_or_cutoff_tamper_is_rejected(self):
        frozen = self.freeze()
        for field in ("model", "selections"):
            modified = json.loads(json.dumps(frozen))
            modified.pop("sha256")
            if field == "model":
                modified["model"]["smoothing"] = 1
            else:
                modified["selections"]["1"]["cutoff"] = -1
            save(self.run / "frozen.json", control.benefit.seal(modified))
            with self.assertRaisesRegex(ValueError, "fixed training/calibration"):
                control.checked_frozen(self.run, self.head, self.baseline)


if __name__ == "__main__":
    unittest.main()
