"""Timing boundaries and artifact checks without Torch or model downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import benchmark_arithmetic_native as benchmark
import benefit
from data import file_hash, write


def fixture(root):
    directory, inputs = root / "study/native-2000", root / "inputs.json"
    rows = [{"id": f"training-{index}", "request": f"Question {index}?"} for index in range(21)]
    study = {
        "source_plan_sha256": "a" * 64,
        "warm_probe_ids": [row["id"] for row in rows],
        "training_ids": [row["id"] for row in rows],
    }
    write(directory.parent / "plan.json", study)
    write(inputs, {"salt": "arithmetic-warm-v1", "warmup": 1, "timed": 20, "rows": rows,
                   "source_plan_sha256": study["source_plan_sha256"]})
    plan = {
        "protocol": benchmark.native.PROTOCOL,
        "head_plan_sha256": file_hash(directory.parent / "plan.json"),
        "training_ids": study["training_ids"],
    }
    write(directory / "plan.json", plan)
    checkpoint = directory / "lora"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"test weights")
    (checkpoint / "rl_agent_config.json").write_text("{}")
    write(directory / "trained.json", benefit.seal({
        "format": benchmark.native.TRAINED_FORMAT,
        "plan_sha256": file_hash(directory / "plan.json"),
        "head_plan_sha256": plan["head_plan_sha256"],
        "conditional_cost_statistics": {},
        "checkpoints": {"lora": {
            "parameters": 12,
            "files_sha256": {path.name: file_hash(path) for path in checkpoint.iterdir()},
        }},
    }))
    return inputs, directory, rows


class ArithmeticNativeTimingTests(unittest.TestCase):
    def test_only_selected_export_is_needed_and_warmup_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, directory, rows = fixture(root)
            router = Mock()
            router.score_one.return_value = {"score": 0.1, "margins": [1.0, 2.0], "route": "strong"}
            with (
                patch.object(benchmark, "_runtime", return_value=(Mock(), Mock(), "cpu", {})),
                patch.object(benchmark, "WarmRouter", return_value=router) as factory,
                patch.object(benchmark.finetune, "empty_cache"),
                patch.object(benchmark.finetune, "synchronize"),
            ):
                result = benchmark.run(inputs, directory, root / "output.json", device="cpu")
            factory.assert_called_once()
            self.assertEqual(factory.call_args.args[1], directory / "lora")
            self.assertEqual(factory.call_args.args[-1], {"mode": "threshold", "cutoff": 0.0})
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["models"]["lora"]["n"], 20)
            self.assertEqual([call.args[0] for call in router.score_one.call_args_list], [rows[20], *rows[:20]])
            self.assertEqual([row["id"] for row in result["models"]["lora"]["requests"]], result["timed_ids"])
            self.assertNotIn(result["warmup_id"], result["timed_ids"])

    def test_changed_checkpoint_is_rejected_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, directory, _ = fixture(root)
            (directory / "lora/model.safetensors").write_bytes(b"different weights")
            with patch.object(benchmark, "_runtime") as runtime, self.assertRaisesRegex(ValueError, "bytes changed"):
                benchmark.run(inputs, directory, root / "output.json", device="cpu")
            runtime.assert_not_called()
            self.assertFalse((root / "output.json").exists())

    def test_changed_sample_order_or_answer_fields_are_rejected(self):
        for change in ("order", "answers", "duplicate"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs, directory, _ = fixture(root)
                sample = json.loads(inputs.read_text())
                if change == "order":
                    sample["rows"].reverse()
                elif change == "answers":
                    sample["rows"][0]["correct"] = True
                else:
                    sample["rows"][-1] = sample["rows"][0]
                inputs.write_text(json.dumps(sample))
                with patch.object(benchmark, "_runtime") as runtime, self.assertRaisesRegex(ValueError, "question-only"):
                    benchmark.run(inputs, directory, root / "output.json", device="cpu")
                runtime.assert_not_called()

    def test_existing_output_is_preserved_before_any_input_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.json"
            output.write_text("preserved")
            with patch.object(benchmark, "inputs_and_checkpoint") as check, self.assertRaises(FileExistsError):
                benchmark.run("missing", "missing", output, device="cpu")
            check.assert_not_called()
            self.assertEqual(output.read_text(), "preserved")

    def test_runtime_failure_retains_incomplete_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, directory, _ = fixture(root)
            with patch.object(benchmark, "_runtime", side_effect=RuntimeError("backend failed")), self.assertRaises(RuntimeError):
                benchmark.run(inputs, directory, root / "output.json", device="cpu")
            result = json.loads((root / "output.json").read_text())
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["error_type"], "RuntimeError")
            self.assertEqual(result["models"], {})


if __name__ == "__main__":
    unittest.main()
