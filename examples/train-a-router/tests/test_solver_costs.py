"""Check singleton billing receipts and bounded collections without network calls."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import solver_costs as costs
from data import digest, file_hash
from serving import REVISION


class ChargedHTTPError(Exception):
    status_code = 429

    def __init__(self, message, credits=7):
        super().__init__(message)
        self.request = {"id": "public-failed-request", "credits_debited": credits}


class SolverCostTests(unittest.TestCase):
    def client(self):
        manager = MagicMock()
        client = manager.__enter__.return_value
        # The SIE bundle SHA and HF source revision have different meanings.
        client.last_model_revision = "b" * 64
        client.get_model.return_value = {"revision": REVISION}
        return manager, client

    def reply(self, credits=2):
        return {
            "id": "problem",
            "dense": [1] + [0] * 2559,
            "request": {"id": "public-request", "credits_debited": credits},
        }

    def test_single_request_records_rounded_credits_identity_and_both_revision_namespaces(self):
        manager, client = self.client()
        client.encode.return_value = self.reply(2)
        row = {"id": "problem", "request": "A public math problem"}
        with patch.object(costs, "SingleSendClient", return_value=manager):
            receipt = costs.measure(row)
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["credits"], 2)
        self.assertEqual(receipt["input_sha256"], digest(row["request"]))
        self.assertEqual(receipt["execution_revision"], "b" * 64)
        self.assertNotEqual(receipt["execution_revision"], REVISION)
        self.assertEqual(len(receipt["vector_sha256"]), 64)
        self.assertEqual(client.encode.call_count, 1)
        self.assertEqual(client.encode.call_args.kwargs["max_oom_retries"], 0)

    def test_invalid_billing_is_unknown_and_receipt_remains_serializable(self):
        for credit in (None, True, False, -1, "2", float("nan"), float("inf")):
            with self.subTest(credit=credit):
                manager, client = self.client()
                client.encode.return_value = self.reply(credit)
                with patch.object(costs, "SingleSendClient", return_value=manager):
                    receipt = costs.measure({"id": "problem", "request": "A public math problem"})
                self.assertEqual(receipt["status"], "incomplete")
                self.assertIsNone(receipt["credits"])
                json.dumps(receipt, allow_nan=False)
                self.assertEqual(client.encode.call_count, 1)

    def test_charged_http_failure_retains_billing_without_raw_exception_text(self):
        manager, client = self.client()
        client.encode.side_effect = ChargedHTTPError("sensitive upstream diagnostic")
        with patch.object(costs, "SingleSendClient", return_value=manager):
            receipt = costs.measure({"id": "problem", "request": "A public math problem"})
        self.assertEqual(receipt["status"], "incomplete")
        self.assertEqual(receipt["credits"], 7)
        self.assertEqual(receipt["error"]["http_status"], 429)
        self.assertNotIn("sensitive upstream diagnostic", json.dumps(receipt))
        self.assertEqual(client.encode.call_count, 1)

    def test_nonfinite_http_failure_billing_is_unknown_and_serializable(self):
        for credits in (float("nan"), float("inf"), True):
            with self.subTest(credits=credits):
                manager, client = self.client()
                client.encode.side_effect = ChargedHTTPError("upstream diagnostic", credits=credits)
                with patch.object(costs, "SingleSendClient", return_value=manager):
                    receipt = costs.measure({"id": "problem", "request": "Public"})
                self.assertIsNone(receipt["credits"])
                self.assertIsNone(receipt["error"]["credits"])
                json.dumps(receipt, allow_nan=False)

    def test_wrong_input_or_vector_is_failed_but_its_charge_is_retained(self):
        for changes in ({"id": "another-problem"}, {"dense": [1, 0]}):
            with self.subTest(changes=list(changes)):
                manager, client = self.client()
                client.encode.return_value = self.reply(3) | changes
                with patch.object(costs, "SingleSendClient", return_value=manager):
                    receipt = costs.measure({"id": "problem", "request": "A public math problem"})
                self.assertEqual(receipt["status"], "incomplete")
                self.assertEqual(receipt["credits"], 3)

    def collect(self, root, rows, measure, workers=1):
        run, output = root / "run", root / "output"
        run.mkdir(exist_ok=True)
        (run / "plan.json").write_text('{"public_fixture":true}\n')
        manager, _client = self.client()
        with (
            patch.object(costs, "checked_inputs", return_value=({}, {"calibration": rows}, {})),
            patch.object(costs, "SingleSendClient", return_value=manager),
            patch.object(costs, "measure", side_effect=measure) as mock,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            costs.collect(run, "calibration", output, workers=workers)
        return run, output, mock

    def test_complete_aggregate_preserves_input_order_and_actual_per_request_charges(self):
        rows = [{"id": f"problem{i}", "request": f"Public {i}"} for i in (2, 0, 1)]

        def measure(row):
            return {
                "id": row["id"],
                "status": "complete",
                "credits": int(row["id"][-1]) + 1,
                "input_sha256": digest(row["request"]),
                "execution_revision": "b" * 64,
            }

        with tempfile.TemporaryDirectory() as temporary:
            run, output, mock = self.collect(Path(temporary), rows, measure, workers=3)
            summary = json.loads((output / "collection.json").read_text())
            aggregate = json.loads((output / "router-credits.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["observed_credits"], 6)
            self.assertEqual(aggregate["plan_sha256"], file_hash(run / "plan.json"))
            self.assertEqual(aggregate["collection_plan_sha256"], file_hash(output / "plan.json"))
            self.assertEqual(aggregate["batch_size"], 1)
            self.assertEqual(
                aggregate["calibration"],
                [
                    {"id": "problem2", "credits": 3},
                    {"id": "problem0", "credits": 1},
                    {"id": "problem1", "credits": 2},
                ],
            )
            self.assertEqual(mock.call_count, 3)

    def test_failure_stops_new_requests_and_never_publishes_complete_credit_table(self):
        rows = [{"id": f"problem{i}", "request": f"Public {i}"} for i in range(3)]
        called = []

        def measure(row):
            called.append(row["id"])
            return {"id": row["id"], "status": "incomplete", "credits": 7}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.collect(root, rows, measure)
            self.assertEqual(called, ["problem0"])
            summary = json.loads((root / "output/collection.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["observed_credits"], 7)
            self.assertTrue((root / "output/problem0.json").exists())
            self.assertFalse((root / "output/router-credits.json").exists())

    def test_credit_cap_stops_scheduling_and_existing_outputs_are_preserved(self):
        rows = [{"id": f"problem{i}", "request": f"Public {i}"} for i in range(3)]
        called = []

        def measure(row):
            called.append(row["id"])
            return {"id": row["id"], "status": "complete", "credits": 5000}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.collect(root, rows, measure)
            self.assertEqual(called, ["problem0"])
            before = (root / "output/collection.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self.collect(root, rows, measure)
            self.assertEqual(called, ["problem0"])
            self.assertEqual((root / "output/collection.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
