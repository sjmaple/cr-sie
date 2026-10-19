"""Exercise benchmark receipts and early rejection without a server or model runtime."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_serving
from data import QUESTION, write
from decision import checksum


class ServingBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "receipt.json"
        self.evaluation = self.root / "test.json"
        self.run = self.root / "run"
        self.rows = [
            {"id": "example-1", "request": "Competition: AMC8\nProblem: What is 2 + 2?"},
            {"id": "example-2", "request": "Competition: AMC8\nProblem: Find the smallest prime above 100."},
        ]
        write(self.evaluation, self.rows)
        write(self.run / "questions.json", QUESTION)
        write(self.run / "lora/rl_agent_config.json", {"temperature": [1.0, 1.0, 1.0]})
        checkpoint = self.run / "lora/model.safetensors"
        checkpoint.write_bytes(b"Public test fixture; never loaded as a model")
        write(self.run / "plan.json", {"data_sha256": {"questions.json": checksum(self.run / "questions.json")}})
        write(
            self.run / "frozen.json",
            {
                "plan_sha256": checksum(self.run / "plan.json"),
                "selected": {"lora": {"threshold": 0.0, "checkpoint_sha256": checksum(checkpoint)}},
            },
        )
        write(
            self.run / "evaluation-test.json",
            {
                "frozen_sha256": checksum(self.run / "frozen.json"),
                "data_sha256": checksum(self.evaluation),
                "ids": [row["id"] for row in self.rows],
                "models": {"lora": {"predictions": [0, 1]}},
            },
        )
        client_patch = patch.object(benchmark_serving, "SingleSendClient")
        self.client_factory = client_patch.start()
        self.addCleanup(client_patch.stop)
        self.client = self.client_factory.return_value.__enter__.return_value
        self.client.get_model.return_value = {}

    def invoke(self, url="http://127.0.0.1:8099"):
        arguments = [
            "benchmark_serving.py",
            "--approach",
            "laya",
            "--url",
            url,
            "--evaluation",
            str(self.evaluation),
            "--run",
            str(self.run),
            "--limit",
            "2",
            "--warmup",
            "1",
            "--output",
            str(self.output),
        ]
        with patch("sys.argv", arguments), patch("sys.stdout", new_callable=io.StringIO):
            benchmark_serving.main()

    @staticmethod
    def reply(score):
        return {"data": {"difficulty": {"probabilities": {"advanced": score}}}}

    def test_progress_is_persisted_and_final_receipt_completes(self):
        observations = []
        replies = iter([self.reply(0.2), self.reply(0.2), self.reply(0.8)])

        def extract(*args, **kwargs):
            receipt = json.loads(self.output.read_text())
            observations.append((receipt["status"], len(receipt["samples"])))
            return next(replies)

        self.client.extract.side_effect = extract
        clock = [0.0, 0.010, 0.012, 1.0, 1.020, 1.023, 2.0, 2.030, 2.034]
        with patch.object(benchmark_serving.time, "perf_counter", side_effect=clock):
            self.invoke()
        receipt = json.loads(self.output.read_text())
        self.assertEqual(observations, [("running", 0), ("running", 0), ("running", 1)])
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual([row["id"] for row in receipt["samples"]], ["example-1", "example-2"])
        self.assertEqual(receipt["reference_matches"], 2)
        self.assertEqual(receipt["reference_predictions"], [0, 1])
        self.assertAlmostEqual(receipt["samples"][0]["total_ms"], 23.0)
        self.assertAlmostEqual(receipt["samples"][1]["total_ms"], 34.0)
        self.assertAlmostEqual(receipt["latency_ms"]["total_ms"]["median"], 28.5)
        self.assertEqual(
            [call.args[1]["text"] for call in self.client.extract.call_args_list],
            [self.rows[0]["request"], self.rows[0]["request"], self.rows[1]["request"]],
        )
        self.client_factory.assert_called_once()

    def test_rpc_failure_preserves_completed_samples_in_incomplete_receipt(self):
        self.client.extract.side_effect = [self.reply(0.2), self.reply(0.2), TimeoutError("RPC failed")]
        with self.assertRaisesRegex(TimeoutError, "RPC failed"):
            self.invoke()
        receipt = json.loads(self.output.read_text())
        self.assertEqual(receipt["status"], "incomplete")
        self.assertEqual([row["id"] for row in receipt["samples"]], ["example-1"])
        self.assertEqual(receipt["error"]["error_type"], "TimeoutError")
        self.assertEqual(self.client.extract.call_count, 3)
        self.assertNotIn("latency_ms", receipt)

    def test_non_loopback_url_is_rejected_before_client_creation(self):
        for url in ("https://example.invalid", "http://127.0.0.1.example.invalid:8099"):
            with self.subTest(url=url), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(url)
                self.assertEqual(error.exception.code, 2)
                self.client_factory.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_existing_receipt_is_preserved_before_client_creation(self):
        original = b'{"status":"previous observation"}\n'
        self.output.write_bytes(original)
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.client_factory.assert_not_called()
        self.assertEqual(self.output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
