"""Exercise calibration/test separation and provenance with small public fixtures."""

import gzip
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import solver_policies as routing
from data import LABELS, digest, file_hash
from serving import INSTRUCTION, MODEL, REVISION


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")
    return path


class SolverPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline, self.study, self.policies, self.run = (
            self.root / name for name in ("baseline", "study", "policies", "run")
        )
        self.dataset = {
            "splits": {
                split: [{"id": f"{split}{i}", "request": f"{split} problem {i}", "label": i % 2} for i in range(size)]
                for split, size in (("train", 512), ("calibration", 4), ("test", 4))
            }
        }
        save(self.baseline / "data.json", self.dataset)
        save(self.baseline / "test.json", self.dataset["splits"]["test"])
        save(self.baseline / "manifest.json", {})
        for name in ("manifest.json", "all-frozen.json", "evaluation-plan.json", "evaluation-complete.json"):
            save(self.study / name, {})
        (self.baseline / "calibration.jsonl").write_text(
            "".join(
                json.dumps({"state": row["request"], "expected": {"difficulty": LABELS[row["label"]]}}) + "\n"
                for row in self.dataset["splits"]["calibration"]
            )
        )
        for root, variant in (
            (self.baseline / "native/seed-89", "base"),
            (self.study / "screening/original/seed-89", "lora"),
            (self.study / "expanded/seed-89", "lora"),
        ):
            self.native(root, variant)
        for split, directory in (("calibration", "development-embeddings"), ("test", "test-embeddings")):
            rows = self.dataset["splits"][split]
            directory = self.baseline / directory
            save(
                directory / "plan.json",
                {
                    "model": MODEL,
                    "revision": REVISION,
                    "instruction": INSTRUCTION,
                    "ids": [row["id"] for row in rows],
                    "input_sha256": {row["id"]: digest(row["request"]) for row in rows},
                },
            )
            with gzip.open(directory / "vectors.jsonl.gz", "wt") as handle:
                for i, row in enumerate(rows):
                    vector = [1 if i == 1 else -1] + [0] * 2559
                    handle.write(json.dumps({"id": row["id"], "vector": vector}) + "\n")
        suite = {"dataset_sha256": digest(self.dataset), "policies": {}}
        for size in (64, 512):
            policy = {
                "format": "sie-contest-relative-router-v2",
                "dataset_sha256": digest(self.dataset),
                "method": "ridge",
                "encoder": MODEL,
                "revision": REVISION,
                "instruction": INSTRUCTION,
                "labels": size,
                "train_ids": [row["id"] for row in self.dataset["splits"]["train"][:size]],
                "development_vectors_sha256": file_hash(self.baseline / "development-embeddings/vectors.jsonl.gz"),
                "weights": [1] + [0] * 2559,
                "bias": 0,
                "threshold": 0,
            }
            policy["sha256"] = digest(policy)
            save(self.policies / f"ridge{size}.json", policy)
            suite["policies"][f"ridge{size}"] = policy["sha256"]
        save(self.policies / "frozen.json", suite)
        save(
            self.baseline / "embedding-evaluation.json",
            {
                "dataset_sha256": digest(self.dataset),
                "ids": [row["id"] for row in self.dataset["splits"]["test"]],
                "predictions": {f"ridge{size}": [0, 1, 0, 0] for size in (64, 512)},
            },
        )
        # Only the external manifest pins are mocked. Every internal file hash,
        # input join, native binding, vector load and ridge calculation is real.
        self.verify = patch.object(routing.evidence, "verify")
        self.verify.start()
        self.addCleanup(self.verify.stop)
        self.scores = routing.load_stage_scores(self.baseline, self.study, self.policies)
        inputs = {
            split: [{"id": row["id"], "request": row["request"]} for row in self.dataset["splits"][split]]
            for split in ("calibration", "test")
        }
        save(self.run / "inputs.json", inputs)
        self.plan = save(
            self.run / "plan.json",
            {
                "dataset_sha256": digest(self.dataset),
                "inputs_sha256": digest(inputs),
                "counts": {split: len(rows) for split, rows in inputs.items()},
            },
        )
        self.calibration = save(self.run / "grades-calibration.json", self.grades("calibration"))

    def native(self, directory, variant):
        plan = save(
            directory / "plan.json",
            {
                "config": {"seed": 89},
                "data_sha256": {"calibration.jsonl": file_hash(self.baseline / "calibration.jsonl")},
            },
        )
        frozen = save(
            directory / "frozen.json",
            {
                "plan_sha256": file_hash(plan),
                "selected": {variant: {"seed": 89, "threshold": 0.5, "checkpoint_sha256": "1" * 64}},
            },
        )
        measured = {
            "scores": [0.1, 0.9, 0.2, 0.3],
            "labels": [0, 1, 0, 1],
            "predictions": [0, 1, 0, 0],
        }
        save(directory / f"{variant}-calibration.json", measured)
        save(
            directory / "evaluation-test.json",
            {
                "frozen_sha256": file_hash(frozen),
                "data_sha256": file_hash(self.baseline / "test.json"),
                "ids": [row["id"] for row in self.dataset["splits"]["test"]],
                "models": {variant: measured},
            },
        )

    def grades(self, split):
        # The useful calibration cutoff fails on test; the report must retain it.
        cheap = [True, False, False, False] if split == "calibration" else [True, True, False, False]
        strong = [True, True, False, False] if split == "calibration" else [True] * 4
        return {
            "split": split,
            "plan_sha256": file_hash(self.plan),
            "grader_source_sha256": "2" * 64,
            "gold_review_source_sha256": "6" * 64,
            "gold_review_sha256": "7" * 64,
            "rows": [
                {
                    "id": row["id"],
                    "cheap_correct": cheap[i],
                    "strong_correct": strong[i],
                    "cheap_credits": 10,
                    "strong_credits": 100,
                }
                for i, row in enumerate(self.dataset["splits"][split])
            ],
        }

    def frozen_and_test(self, credits=None):
        frozen = routing.freeze_calibrations(self.plan, self.calibration, self.scores, credits)
        path = save(self.run / "policies-frozen.json", frozen)
        grades = self.grades("test") | {"policies_frozen_sha256": file_hash(path)}
        return path, save(self.run / "grades-test.json", grades)

    def credits(self, split, amount=2):
        return save(
            self.root / f"credits-{split}.json",
            {
                "plan_sha256": file_hash(self.plan),
                "model": MODEL,
                "revision": REVISION,
                "batch_size": 1,
                split: [{"id": identity, "credits": amount} for identity in self.scores["ids"][split]],
            },
        )

    def test_load_joins_five_stages_with_real_scores_and_file_hashes(self):
        self.assertEqual(tuple(self.scores["stages"]), routing.STAGES)
        self.assertEqual(self.scores["stages"]["base"]["scores"]["calibration"], [0.1, 0.9, 0.2, 0.3])
        self.assertEqual(self.scores["stages"]["ridge64"]["scores"]["test"], [-1, 1, -1, -1])
        self.assertEqual(len(self.scores["source_artifacts_sha256"]), 28)
        self.assertEqual(
            self.scores["source_artifacts_sha256"]["study/expanded/seed-89/frozen.json"],
            file_hash(self.study / "expanded/seed-89/frozen.json"),
        )

    def test_native_calibration_order_and_model_binding_are_checked(self):
        path = self.baseline / "calibration.jsonl"
        original = path.read_text()
        path.write_text("\n".join(reversed(original.splitlines())) + "\n")
        with self.assertRaisesRegex(ValueError, "calibration order"):
            routing.load_stage_scores(self.baseline, self.study, self.policies)
        path.write_text(original)
        plan = self.study / "expanded/seed-89/plan.json"
        save(plan, routing.read(plan) | {"config": {"seed": 137}})
        with self.assertRaisesRegex(ValueError, "frozen model"):
            routing.load_stage_scores(self.baseline, self.study, self.policies)

    def test_ridge_weights_cannot_change_under_frozen_hash(self):
        path = self.policies / "ridge64.json"
        policy = routing.read(path)
        policy["weights"][0] = -1
        save(path, policy)
        with self.assertRaisesRegex(ValueError, "Frozen policy changed"):
            routing.load_stage_scores(self.baseline, self.study, self.policies)

    def test_freeze_uses_only_calibration_and_keeps_all_declared_targets(self):
        read = routing.read
        opened = []

        def guarded(path):
            opened.append(Path(path).name)
            if Path(path).name == "grades-test.json":
                self.fail("Freeze opened solver test answers")
            return read(path)

        with patch.object(routing, "read", side_effect=guarded):
            frozen = routing.freeze_calibrations(self.plan, self.calibration, self.scores)
        self.assertNotIn("grades-test.json", opened)
        self.assertEqual(frozen["plan_sha256"], file_hash(self.plan))
        self.assertEqual(frozen["calibration_grades_sha256"], file_hash(self.calibration))
        for name in routing.GRADING_HASHES:
            self.assertEqual(frozen[name], self.grades("calibration")[name])
        self.assertEqual(frozen["source_scores"]["ids"]["test"], self.scores["ids"]["test"])
        for stage in routing.STAGES:
            self.assertEqual(list(frozen["selections"][stage]), ["1", "0", "2", "5"])
            self.assertEqual(frozen["selections"][stage]["1"]["calibration_routes"], [False, True, False, False])

    def test_report_applies_frozen_routes_even_when_test_loss_exceeds_target(self):
        frozen, grades = self.frozen_and_test()
        select = routing.solver_report.select_cutoff

        def only_calibration(rows, *args, **kwargs):
            self.assertEqual([row["id"] for row in rows], self.scores["ids"]["calibration"])
            return select(rows, *args, **kwargs)

        with patch.object(routing.solver_report, "select_cutoff", side_effect=only_calibration):
            report = routing.report(frozen, grades)
        self.assertEqual(report["baselines"]["always_strong"]["answer_accuracy"], 1)
        for stage in routing.STAGES:
            result = report["stages"][stage]["calibrated_cost_policies"]["1"]
            self.assertEqual(result["metrics"]["gap_vs_strong_pp"], 50)
            self.assertEqual(result["metrics"]["regressions"], 2)
            self.assertEqual(result["mode"], "threshold")
        self.assertNotIn("net_total_cost_usd_per1000", report["stages"]["base"]["original_difficulty_policy"])

    def test_optional_encoder_costs_apply_to_ridge_only(self):
        frozen, grades = self.frozen_and_test(self.credits("calibration"))
        report = routing.report(frozen, grades, self.credits("test", amount=3))
        for name in ("ridge64", "ridge512"):
            measured = report["stages"][name]["original_difficulty_policy"]
            self.assertAlmostEqual(measured["router_cost_usd_per1000"], 0.03)
        self.assertNotIn("router_cost_usd_per1000", report["stages"]["lora2581"]["original_difficulty_policy"])
        for baseline in report["baselines"].values():
            self.assertEqual(baseline["router_cost_usd_per1000"], 0)

    def test_self_hosted_quality_only_freezes_original_cutoffs_without_cost_selection(self):
        calibration = routing.read(self.calibration)
        for row in calibration["rows"]:
            row["cheap_credits"] = row["strong_credits"] = None
        save(self.calibration, calibration)
        with self.assertRaises((ValueError, TypeError)):
            routing.freeze_calibrations(self.plan, self.calibration, self.scores)
        with patch.object(
            routing.solver_report, "select_cutoff", side_effect=AssertionError("Must not optimize unknown cost")
        ):
            frozen = routing.freeze_calibrations(self.plan, self.calibration, self.scores, quality_only=True)
            self.assertEqual(frozen["analysis_mode"], "quality_only")
            self.assertEqual(frozen["selections"], {})
            self.assertIsNone(frozen["credit_usd"])
            frozen_path = save(self.run / "policies-frozen.json", frozen)
            grades = self.grades("test") | {"policies_frozen_sha256": file_hash(frozen_path)}
            for row in grades["rows"]:
                row["cheap_credits"] = row["strong_credits"] = None
            test_path = save(self.run / "grades-test.json", grades)
            measured = routing.report(frozen_path, test_path)
        for name, stage in measured["stages"].items():
            self.assertEqual(stage["original_threshold"], self.scores["stages"][name]["original_threshold"])
            self.assertEqual(stage["calibrated_cost_policies"], {})
            self.assertEqual(stage["original_difficulty_policy"]["cost_status"], "unknown")
            self.assertFalse(any("usd" in key or "savings" in key for key in stage["original_difficulty_policy"]))
        self.assertEqual(measured["baselines"]["always_strong"]["answer_accuracy"], 1)
        self.assertIsNone(measured["primary_max_quality_loss_pp"])

    def test_quality_only_cannot_be_combined_with_partial_pricing(self):
        credits = self.credits("calibration")
        with self.assertRaisesRegex(ValueError, "does not accept priced router"):
            routing.freeze_calibrations(self.plan, self.calibration, self.scores, credits, quality_only=True)

    def test_bad_calibration_population_or_inputs_cannot_freeze(self):
        original = routing.read(self.calibration)
        for rows in (original["rows"][:-1], list(reversed(original["rows"]))):
            save(self.calibration, original | {"rows": rows})
            with self.assertRaisesRegex(ValueError, "every expected ID"):
                routing.freeze_calibrations(self.plan, self.calibration, self.scores)
        save(self.calibration, original)
        inputs = routing.read(self.run / "inputs.json")
        inputs["test"][0]["request"] += " changed"
        save(self.run / "inputs.json", inputs)
        with self.assertRaisesRegex(ValueError, "plan inputs"):
            routing.freeze_calibrations(self.plan, self.calibration, self.scores)

    def test_test_grades_bind_plan_split_ids_grader_and_precollection_freeze(self):
        frozen, grades = self.frozen_and_test()
        original = routing.read(grades)
        changes = [
            {"plan_sha256": "3" * 64},
            {"split": "calibration"},
            {"grader_source_sha256": "4" * 64},
            {"gold_review_source_sha256": "8" * 64},
            {"gold_review_sha256": "9" * 64},
            {"policies_frozen_sha256": "5" * 64},
            {"rows": list(reversed(original["rows"]))},
        ]
        for change in changes:
            with self.subTest(change=list(change)):
                save(grades, original | change)
                with self.assertRaises(ValueError):
                    routing.report(frozen, grades)

    def test_freeze_rejects_missing_or_invalid_grading_provenance(self):
        original = routing.read(self.calibration)
        for name in routing.GRADING_HASHES:
            for value in (None, "", "g" * 64, 123):
                with self.subTest(name=name, value=value):
                    modified = original.copy()
                    if value is None:
                        modified.pop(name)
                    else:
                        modified[name] = value
                    save(self.calibration, modified)
                    with self.assertRaisesRegex(ValueError, "Grading provenance"):
                        routing.freeze_calibrations(self.plan, self.calibration, self.scores)

    def test_report_requires_review_provenance_in_both_freeze_and_test_grades(self):
        frozen, grades = self.frozen_and_test()
        original_frozen, original_grades = routing.read(frozen), routing.read(grades)
        for name in routing.GRADING_HASHES:
            with self.subTest(name=name, missing="test"):
                modified = original_grades.copy()
                modified.pop(name)
                save(grades, modified)
                with self.assertRaisesRegex(ValueError, "Grading provenance"):
                    routing.report(frozen, grades)
            with self.subTest(name=name, missing="frozen"):
                modified = original_frozen.copy()
                modified.pop("sha256")
                modified.pop(name)
                save(frozen, routing.seal(modified))
                save(grades, original_grades | {"policies_frozen_sha256": file_hash(frozen)})
                with self.assertRaisesRegex(ValueError, "Grading provenance"):
                    routing.report(frozen, grades)
            save(frozen, original_frozen)
            save(grades, original_grades)

    def test_changed_selection_rejected_even_if_document_is_resealed(self):
        frozen, grades = self.frozen_and_test()
        modified = routing.read(frozen)
        modified["selections"]["base"]["1"]["cutoff"] = 0.1
        save(frozen, modified)
        with self.assertRaisesRegex(ValueError, "content hash"):
            routing.report(frozen, grades)
        modified.pop("sha256")
        save(frozen, routing.seal(modified))
        save(grades, routing.read(grades) | {"policies_frozen_sha256": file_hash(frozen)})
        with self.assertRaisesRegex(ValueError, "Frozen calibration policy"):
            routing.report(frozen, grades)

    def test_router_credit_population_model_and_batch_scope_are_checked(self):
        path = self.credits("calibration")
        original = routing.read(path)
        for change in (
            {"batch_size": 16},
            {"revision": "other"},
            {"plan_sha256": "0" * 64},
            {"calibration": list(reversed(original["calibration"]))},
            {"calibration": [{"id": row["id"], "credits": -1} for row in original["calibration"]]},
        ):
            with self.subTest(change=list(change)):
                save(path, original | change)
                with self.assertRaises(ValueError):
                    routing.freeze_calibrations(self.plan, self.calibration, self.scores, path)

    def test_freeze_refuses_after_test_collection_started(self):
        (self.run / "answers-test").mkdir()
        with self.assertRaisesRegex(ValueError, "after test-answer collection"):
            routing.freeze_calibrations(self.plan, self.calibration, self.scores)

    def test_cli_preserves_existing_output_before_loading_evidence(self):
        output = save(self.root / "existing.json", {"keep": True})
        args = ["solver_policies.py", "freeze"]
        for name in ("plan", "calibration-grades", "baseline", "study", "policies"):
            args.extend([f"--{name}", "absent"])
        args.extend(["--output", str(output)])
        with patch("sys.argv", args), patch.object(routing, "load_stage_scores") as load:
            with self.assertRaises(FileExistsError):
                routing.main()
            load.assert_not_called()
        self.assertEqual(routing.read(output), {"keep": True})


class SolverDisplayTests(unittest.TestCase):
    def fixture(self, *, quality_only=False):
        rows = [
            {"id": str(i), "cheap_correct": cheap, "strong_correct": strong, "cheap_credits": 10, "strong_credits": 100}
            for i, (cheap, strong) in enumerate(((True, True), (False, True), (True, False), (False, False)))
        ]
        routes = [False, True, False, False]
        summarize = routing.solver_report.summarize_quality if quality_only else routing.solver_report.summarize
        native = summarize(rows, routes)
        embedding = native if quality_only else summarize(rows, routes, router_credits=2.5)
        return {
            "analysis_mode": "quality_only" if quality_only else "cost",
            "n": len(rows),
            "primary_max_quality_loss_pp": None if quality_only else 1,
            "baselines": {
                name: summarize(rows, name) if quality_only else summarize(rows, name, router_credits=0)
                for name in ("always_strong", "always_cheap")
            },
            "stages": {
                name: {
                    "original_difficulty_policy": embedding if name.startswith("ridge") else native,
                    "calibrated_cost_policies": {}
                    if quality_only
                    else {"1": {"metrics": embedding if name.startswith("ridge") else native}},
                }
                for name in routing.STAGES
            },
        }

    def test_cost_display_preserves_negative_gap_rescues_and_unpriced_native_overhead(self):
        result = self.fixture()
        original = json.dumps(result, sort_keys=True)
        output = io.StringIO()
        with redirect_stdout(output):
            routing.display(result)
        rendered = output.getvalue()
        native = [line.split("|") for line in rendered.splitlines() if line.startswith("Base Laya")]
        self.assertEqual([value.strip() for value in native[0]], ["Base Laya", "75.00", "-25.00", "75.00", "0", "1"])
        self.assertEqual(
            [value.strip() for value in native[1]], ["Base Laya", "0.3250", "67.50", "unpriced", "unpriced", "unpriced"]
        )
        baseline = [line.split("|") for line in rendered.splitlines() if line.startswith("Always strong")]
        self.assertEqual(
            [value.strip() for value in baseline[1]],
            ["Always strong", "1.0000", "0.00", "0.0000", "1.0000", "0.00"],
        )
        ridge = [line.split("|") for line in rendered.splitlines() if line.startswith("Qwen ridge 64")]
        self.assertEqual(
            [value.strip() for value in ridge[1]], ["Qwen ridge 64", "0.3250", "67.50", "0.0250", "0.3500", "65.00"]
        )
        self.assertIn("Regressions / rescues", rendered)
        self.assertIn("Gross savings exclude router costs", rendered)
        self.assertIn("1pp calibration loss target (test results)", rendered)
        self.assertIn("targets are not test guarantees", rendered)
        self.assertEqual(json.dumps(result, sort_keys=True), original)

    def test_quality_only_display_has_no_dollar_values_or_cost_policies(self):
        output = io.StringIO()
        with redirect_stdout(output):
            routing.display(self.fixture(quality_only=True))
        rendered = output.getvalue()
        self.assertIn("Costs unknown", rendered)
        self.assertIn("-25.00", rendered)
        self.assertIn("Rescues", rendered)
        self.assertNotIn("$", rendered)
        self.assertNotIn("Cost policies selected", rendered)
        self.assertNotIn("Gross save", rendered)

    def test_report_cli_displays_summary_and_preserves_json_document(self):
        result = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            args = [
                "solver_policies.py",
                "report",
                "--frozen",
                "frozen.json",
                "--test-grades",
                "grades.json",
                "--output",
                str(output),
            ]
            text = io.StringIO()
            with patch("sys.argv", args), patch.object(routing, "report", return_value=result), redirect_stdout(text):
                routing.main()
            self.assertEqual(routing.read(output), result)
            self.assertIn("Answer quality on 4 paired test requests", text.getvalue())
            self.assertEqual(
                json.loads(text.getvalue().splitlines()[-1]), {"output": str(output), "sha256": file_hash(output)}
            )


if __name__ == "__main__":
    unittest.main()
