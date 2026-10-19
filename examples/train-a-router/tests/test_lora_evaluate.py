"""Evaluation must not expose test inputs before freezing or repeat a shared stage."""

import argparse
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data import write
from finetune import sha
from lora_evaluate import (
    EAGER_ENV,
    check_float32,
    check_runtime_parity,
    checkpoint_manifest,
    evaluation_runs,
    margin_parity,
    run,
)


def parity_fixture(directory):
    write(directory / "plan.json", {})
    write(directory / "frozen.json", {"selected": {"lora": {"epoch": 1, "threshold": 0.5}}})
    reference = {"epoch": 1, "scores": [-1.0, 0.49, 0.8, 1.2], "labels": [0, 0, 1, 1]}
    write(directory / "lora-history.json", [reference])
    write(directory / "lora-calibration.json", reference)
    return {
        "runtime_environment": EAGER_ENV,
        "frozen_sha256": sha(directory / "frozen.json"),
        "splits": {
            split: margin_parity(reference["scores"], reference["scores"], 0.5) | {"labels": reference["labels"]}
            for split in ("validation", "calibration")
        },
    }


class EvaluationGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(
            study=self.root / "study", evidence=self.root / "evidence", device="cuda", timeout_seconds=1800
        )
        self.args.study.mkdir()
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)

    def test_incomplete_study_never_opens_test_evidence_or_starts_inference(self):
        with (
            patch("lora_evaluate.verify_evidence") as evidence,
            patch("lora_evaluate.subprocess.run") as inference,
            self.assertRaises(FileNotFoundError),
        ):
            run(self.args)
        evidence.assert_not_called()
        inference.assert_not_called()

    def test_changed_training_source_is_rejected_before_test_access(self):
        frozen = ({"training_source_sha256": "0" * 64}, {"recipe": "original"}, {}, {})
        with (
            patch("lora_evaluate.check_study", return_value=frozen),
            patch("lora_evaluate.verify_evidence") as evidence,
            patch("lora_evaluate.subprocess.run") as inference,
            self.assertRaisesRegex(ValueError, "same finetune.py"),
        ):
            run(self.args)
        evidence.assert_not_called()
        inference.assert_not_called()

    def test_existing_observations_are_preserved_before_test_access(self):
        source = Path(__file__).parents[1] / "finetune.py"
        frozen = ({"training_source_sha256": sha(source)}, {"recipe": "original"}, {}, {})
        observed = self.args.study / "screening/original/seed-89/evaluation-test.json"
        write(observed, {"previous": "observation"})
        with (
            patch("lora_evaluate.check_study", return_value=frozen),
            patch("lora_evaluate.verify_evidence") as evidence,
            patch("lora_evaluate.subprocess.run") as inference,
            self.assertRaisesRegex(FileExistsError, "observations are preserved"),
        ):
            run(self.args)
        self.assertEqual(json.loads(observed.read_text()), {"previous": "observation"})
        evidence.assert_not_called()
        inference.assert_not_called()

    def test_original_winner_is_evaluated_once_per_seed(self):
        runs = evaluation_runs(self.args.study, "original")
        self.assertEqual(len(runs), 6)
        self.assertEqual(len(set(runs)), 6)
        self.assertEqual(sum("screening" in path.parts for path in runs), 3)

    def test_other_winner_adds_exactly_three_runs(self):
        runs = evaluation_runs(self.args.study, "slow")
        self.assertEqual(len(runs), 9)
        self.assertEqual(len(set(runs)), 9)
        self.assertEqual(sum("original" in path.parts for path in runs), 3)
        self.assertEqual(sum("slow" in path.parts for path in runs), 3)

    def test_nonpositive_deadline_rejected_before_any_access(self):
        self.args.timeout_seconds = 0
        with (
            patch("lora_evaluate.check_study") as study,
            self.assertRaisesRegex(ValueError, "timeout must be positive"),
        ):
            run(self.args)
        study.assert_not_called()

    def prepared_study(self):
        source = Path(__file__).parents[1] / "finetune.py"
        write(self.args.study / "study-plan.json", {})
        write(self.args.study / "all-frozen.json", {})
        records = {directory: parity_fixture(directory) for directory in evaluation_runs(self.args.study, "original")}
        declared = ({"training_source_sha256": sha(source), "data_sha256": {"512": {}}}, {"recipe": "original"}, {}, {})

        def manifest(directory):
            return {
                "frozen_sha256": sha(directory / "frozen.json"),
                "plan_sha256": sha(directory / "plan.json"),
                "checkpoint_files_sha256": {},
            }

        return declared, records, manifest

    def test_failed_parity_preserves_receipt_and_never_opens_test(self):
        declared, records, manifest = self.prepared_study()
        calls = []

        def child(command, **kwargs):
            self.assertTrue(all(os.environ.get(key) == value for key, value in EAGER_ENV.items()))
            self.assertGreater(kwargs["timeout"], 0)
            directory = Path(command[command.index("--runtime-parity-run") + 1])
            calls.append(directory)
            if len(calls) == 4:
                records[directory]["splits"]["calibration"]["scores"][1] = 0.51
            write(directory / "runtime-parity.json", records[directory])

        with (
            patch("lora_evaluate.check_study", return_value=declared),
            patch("lora_evaluate.checkpoint_manifest", side_effect=manifest),
            patch("lora_evaluate.subprocess.run", side_effect=child),
            patch("lora_evaluate.verify_evidence") as evidence,
            self.assertRaisesRegex(ValueError, "parity failed for calibration"),
        ):
            run(self.args)
        self.assertEqual(len(calls), 4)
        evidence.assert_not_called()
        self.assertTrue((self.args.study / "runtime-parity-incomplete.json").exists())
        self.assertFalse((self.args.study / "evaluation-plan.json").exists())

    def test_all_six_parity_receipts_are_complete_before_test_access(self):
        declared, records, manifest = self.prepared_study()
        calls = []

        def child(command, **kwargs):
            directory = Path(command[command.index("--runtime-parity-run") + 1])
            calls.append(directory)
            write(directory / "runtime-parity.json", records[directory])

        def evidence(_):
            self.assertEqual(len(calls), 6)
            complete = json.loads((self.args.study / "runtime-parity-complete.json").read_text())
            self.assertEqual(len(complete["parity_receipts_sha256"]), 6)
            plan = json.loads((self.args.study / "runtime-parity-plan.json").read_text())
            self.assertEqual(plan["runtime_environment"], EAGER_ENV)
            self.assertEqual(plan["max_margin_error"], 0.03)
            self.assertTrue(plan["require_unchanged_frozen_cutoff_decisions"])
            raise RuntimeError("Stop before opening real test inputs")

        with (
            patch("lora_evaluate.check_study", return_value=declared),
            patch("lora_evaluate.checkpoint_manifest", side_effect=manifest),
            patch("lora_evaluate.subprocess.run", side_effect=child),
            patch("lora_evaluate.verify_evidence", side_effect=evidence),
            self.assertRaisesRegex(RuntimeError, "Stop before opening real test inputs"),
        ):
            run(self.args)

    def test_parity_child_timeout_never_opens_test(self):
        declared, _, manifest = self.prepared_study()
        with (
            patch("lora_evaluate.check_study", return_value=declared),
            patch("lora_evaluate.checkpoint_manifest", side_effect=manifest),
            patch("lora_evaluate.subprocess.run", side_effect=subprocess.TimeoutExpired("parity", 1)),
            patch("lora_evaluate.verify_evidence") as evidence,
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            run(self.args)
        evidence.assert_not_called()
        self.assertTrue((self.args.study / "runtime-parity-incomplete.json").exists())

    def test_existing_parity_attempt_is_not_overwritten(self):
        declared, _, _ = self.prepared_study()
        write(self.args.study / "runtime-parity-plan.json", {"previous": True})
        with (
            patch("lora_evaluate.check_study", return_value=declared),
            patch("lora_evaluate.verify_evidence") as evidence,
            patch("lora_evaluate.subprocess.run") as child,
            self.assertRaisesRegex(FileExistsError, "observations are preserved"),
        ):
            run(self.args)
        evidence.assert_not_called()
        child.assert_not_called()
        self.assertEqual(json.loads((self.args.study / "runtime-parity-plan.json").read_text()), {"previous": True})


class RuntimeParityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.record = parity_fixture(self.directory)

    def test_matching_development_scores_pass(self):
        check_runtime_parity(self.directory, self.record)

    def test_cutoff_crossing_fails_even_below_margin_tolerance(self):
        measured = self.record["splits"]["validation"]
        measured.update(margin_parity([-1.0, 0.51, 0.8, 1.2], measured["scores"], 0.5))
        self.assertLess(measured["max_margin_error"], 0.03)
        with self.assertRaisesRegex(ValueError, "parity failed for validation"):
            check_runtime_parity(self.directory, self.record)

    def test_margin_drift_fails_even_without_cutoff_changes(self):
        measured = self.record["splits"]["calibration"]
        measured.update(margin_parity([-0.9, 0.49, 0.8, 1.2], measured["scores"], 0.5))
        self.assertEqual(measured["decision_changes"], 0)
        with self.assertRaisesRegex(ValueError, "parity failed for calibration"):
            check_runtime_parity(self.directory, self.record)

    def test_changed_labels_and_tampered_summary_are_rejected(self):
        self.record["splits"]["calibration"]["labels"] = [1, 0, 1, 1]
        with self.assertRaisesRegex(ValueError, "parity failed for calibration"):
            check_runtime_parity(self.directory, self.record)
        self.record["splits"]["calibration"]["labels"] = [0, 0, 1, 1]
        self.record["splits"]["validation"]["max_margin_error"] = 0.01
        with self.assertRaisesRegex(ValueError, "parity failed for validation"):
            check_runtime_parity(self.directory, self.record)

    def test_nonfinite_or_misaligned_scores_are_rejected(self):
        for actual in ([float("nan")], [0.0, 1.0]):
            with self.subTest(actual=actual), self.assertRaisesRegex(ValueError, "finite aligned"):
                margin_parity(actual, [0.0], 0.5)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def tensor_file(self, dtype):
        path = self.directory / "lora/model.safetensors"
        path.parent.mkdir(exist_ok=True)
        size = 4 if dtype == "F32" else 2
        header = json.dumps({"weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, size]}}).encode()
        header += b" " * (-len(header) % 8)
        path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(size))
        return path

    def test_actual_tensor_precision_must_match_declaration(self):
        path = self.tensor_file("F32")
        check_float32(path)
        self.tensor_file("F16")
        with self.assertRaisesRegex(ValueError, "float32 export"):
            check_float32(path)

    def test_changed_checkpoint_is_rejected(self):
        path = self.tensor_file("F32")
        write(self.directory / "frozen.json", {"selected": {"lora": {"checkpoint_sha256": sha(path)}}})
        path.write_bytes(path.read_bytes()[:-1] + b"\x01")
        with self.assertRaisesRegex(ValueError, "weights changed after freeze"):
            checkpoint_manifest(self.directory)

    def test_manifest_includes_the_tokenizer_and_model_configuration(self):
        path = self.tensor_file("F32")
        write(self.directory / "plan.json", {"export_dtype": "float32"})
        write(self.directory / "frozen.json", {"selected": {"lora": {"checkpoint_sha256": sha(path)}}})
        names = (
            "rl_agent_config.json",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        )
        for name in names:
            write(self.directory / "lora" / name, {})
        result = checkpoint_manifest(self.directory)
        self.assertEqual(set(result["checkpoint_files_sha256"]), {*names, "model.safetensors"})
        for name in names:
            self.assertEqual(result["checkpoint_files_sha256"][name], sha(self.directory / "lora" / name))


if __name__ == "__main__":
    unittest.main()
