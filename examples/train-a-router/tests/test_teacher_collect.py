"""Budget/scheduling tests with synthetic async responses; no network traffic."""

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import teacher_collect as collector
from data import digest, file_hash, write


class BudgetTests(unittest.TestCase):
    def test_historical_and_inflight_reservations_share_one_cap(self):
        budget = collector.Budget("15", "1.2")
        self.assertTrue(budget.reserve("a", "6"))
        self.assertTrue(budget.reserve("b", "6"))
        self.assertFalse(budget.reserve("c", "2"))
        self.assertEqual(budget.exposure, Decimal("13.2"))
        budget.settle("a", "0.5")
        self.assertTrue(budget.reserve("c", "2"))
        budget.settle("b")  # Unknown execution retains the entire reservation.
        self.assertEqual(budget.exposure, Decimal("9.7"))
        self.assertFalse(budget.reserve("c", "1"))

    def test_invoice_and_conservative_costs_are_separate(self):
        usage = {
            "input_tokens": 1000,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 200},
        }
        invoice, conservative = collector.usage_cost(usage, 2000)
        self.assertEqual(invoice, Decimal(".0106"))
        self.assertEqual(conservative, Decimal(".01235"))
        for invalid in (
            None,
            {"input_tokens": True, "output_tokens": 2},
            {"input_tokens": 2001, "output_tokens": 2},
            {"input_tokens": 1, "output_tokens": 8193},
            {"input_tokens": 1, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 2}},
        ):
            with self.assertRaises((TypeError, ValueError)):
                collector.usage_cost(invalid, 2000)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.training, self.pilot, self.output = root / "training", root / "pilot", root / "extension"
        self.rows = [{"id": f"e2h-amc-train-{i}", "request": f"Question {i}"} for i in range(510)]
        identities = [row["id"] for row in self.rows]
        inputs = {"train": self.rows}
        write(self.training / "inputs.json", inputs)
        write(self.training / "eligibility.json", {"source_only": True, "eligible_train_ids": identities})
        write(
            self.training / "plan.json",
            {
                "population": "original",
                "counts": {"train": 510},
                "source_counts": {"train": 512},
                "inputs_sha256": digest(inputs),
                "eligibility_sha256": file_hash(self.training / "eligibility.json"),
                "eligible_train_ids": identities,
            },
        )
        write(
            self.pilot / "plan.json",
            {
                "n": 64,
                "inputs": self.rows[:64],
                "settings": collector.SETTINGS,
                "prices_usd_per_million": collector.PRICES,
                "teacher_receives_gold": False,
                "teacher_receives_solver_answers": False,
            },
        )
        hashes = {}
        for row in self.rows[:64]:
            record = {
                "id": row["id"],
                "request_sha256": digest(collector.body_for(row)),
                "status": "complete",
                "response_status": "completed",
                "returned_model": collector.MODEL,
                "answer": "1",
                "output_text": '{"answer":"1"}',
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
            path = self.pilot / "records" / f"{row['id']}.json"
            write(path, record)
            hashes[row["id"]] = file_hash(path)
        write(
            self.pilot / "collection.json",
            {
                "status": "complete",
                "n": 64,
                "complete": 64,
                "returned": 64,
                "plan_sha256": file_hash(self.pilot / "plan.json"),
                "record_hashes": hashes,
            },
        )

    def test_preparation_reuses_raw_bytes_and_binds_every_question(self):
        plan = collector.prepare(self.training, self.pilot, self.output)
        checked, rows = collector.checked_plan(self.output)
        self.assertEqual(plan, checked)
        self.assertEqual(len(rows), 510)
        self.assertEqual(len(plan["new_ids"]), 446)
        self.assertFalse((self.output / "started.json").exists())
        for row in self.rows[:64]:
            name = f"{row['id']}.json"
            self.assertEqual(
                (self.output / "records" / name).read_bytes(), (self.pilot / "records" / name).read_bytes()
            )
        with self.assertRaises(FileExistsError):
            collector.prepare(self.training, self.pilot, self.output)

    def test_changed_planned_question_is_rejected_before_collection(self):
        collector.prepare(self.training, self.pilot, self.output)
        path = self.output / "plan.json"
        plan = json.loads(path.read_text())
        plan["inputs"][-1]["request"] = "Changed question"
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "identities changed"):
            collector.checked_plan(self.output)

    def test_historical_budget_cannot_be_zeroed(self):
        collector.prepare(self.training, self.pilot, self.output)
        path = self.output / "plan.json"
        plan = json.loads(path.read_text())
        plan["reuse"]["conservative_budget_usd"] = "0"
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "Historical teacher budget"):
            collector.checked_plan(self.output)


class CollectorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.rows = {f"row{i}": {"id": f"row{i}", "request": f"question{i}"} for i in range(6)}
        reused = {"id": "reused", "status": "complete"}
        write(self.run / "records/reused.json", reused)
        self.plan = {
            "n": 7,
            "new_ids": list(self.rows),
            "reused_ids": ["reused"],
            "max_usd": "15",
            "max_seconds": 1800,
            "workers": 4,
            "reuse": {
                "record_hashes": {"reused": file_hash(self.run / "records/reused.json")},
                "conservative_budget_usd": "1.2",
                "list_price_estimate_usd": "1.1",
            },
            "request_reservations_usd": {
                identity: str(collector.reservation(row)) for identity, row in self.rows.items()
            },
        }
        write(self.run / "plan.json", self.plan)
        self.checked = patch.object(collector, "checked_plan", return_value=(self.plan, self.rows))
        self.checked.start()
        self.addCleanup(self.checked.stop)

    @staticmethod
    def payload():
        return {
            "payload": {
                "id": "response",
                "model": collector.MODEL,
                "status": "completed",
                "service_tier": "default",
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"answer":"1"}'}]}],
            }
        }

    async def test_success_records_all_calls_and_reuses_existing_bytes(self):
        calls, active, maximum = [], 0, 0
        previous = (self.run / "records/reused.json").read_bytes()

        async def request(client, body, secret, seconds):
            nonlocal active, maximum
            calls.append(body["input"])
            self.assertNotIn("answer", body)
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.001)
            active -= 1
            return self.payload()

        with contextlib.redirect_stdout(io.StringIO()):
            result = await collector.collect(self.run, secret="synthetic", request_fn=request)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(calls), 6)
        self.assertEqual(maximum, 4)
        self.assertEqual(result["inference_retries"], 0)
        self.assertEqual((self.run / "records/reused.json").read_bytes(), previous)
        self.assertEqual(len(result["intent_hashes"]), 6)
        for path in (self.run / "intents").glob("*.json"):
            self.assertLessEqual(
                Decimal(json.loads(path.read_text())["budget_after_reservation"]["total_exposure_usd"]), 15
            )

    async def test_429_stops_all_new_dispatch_and_retains_unknown_reservation(self):
        calls = []

        async def request(client, body, secret, seconds):
            calls.append(body["input"])
            if body["input"] == "question0":
                return {"error": {"kind": "http_error", "status": 429, "execution_unknown": True}}
            await asyncio.sleep(0.01)
            return self.payload()

        with contextlib.redirect_stdout(io.StringIO()):
            result = await collector.collect(self.run, secret="synthetic", request_fn=request)
        self.assertEqual(len(calls), 4)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["stop_reason"], "request_error_or_unknown_execution")
        self.assertFalse(result["invoice_estimate_complete"])
        reservation = Decimal(self.plan["request_reservations_usd"]["row0"])
        self.assertGreaterEqual(Decimal(result["budget"]["settled_or_unknown_usd"]), Decimal("1.2") + reservation)
        with self.assertRaises(FileExistsError):
            await collector.collect(self.run, secret="synthetic", request_fn=request)
        self.assertEqual(len(calls), 4)

    async def test_budget_stop_occurs_before_any_network_request(self):
        self.plan["max_usd"] = "1.21"
        calls = []

        async def request(*args):
            calls.append(True)
            return self.payload()

        with contextlib.redirect_stdout(io.StringIO()):
            result = await collector.collect(self.run, secret="synthetic", request_fn=request)
        self.assertEqual(calls, [])
        self.assertEqual(result["stop_reason"], "budget")
        self.assertEqual(result["complete"], 1)

    async def test_malformed_answer_preserves_raw_usage_and_stops(self):
        self.plan["workers"] = 1

        async def request(*args):
            payload = self.payload()
            payload["payload"]["output"][0]["content"][0]["text"] = "not JSON"
            return payload

        with contextlib.redirect_stdout(io.StringIO()):
            result = await collector.collect(self.run, secret="synthetic", request_fn=request)
        self.assertEqual(result["dispatched_new"], 1)
        record = json.loads((self.run / "records/row0.json").read_text())
        self.assertTrue(record["usage_verified"])
        self.assertEqual(record["status"], "incomplete")
        self.assertEqual(record["raw_response_sha256"], file_hash(self.run / "responses/row0.json"))

    async def test_http_transport_never_follows_redirect_or_retries(self):
        calls = []

        class Response:
            status_code = 302

        class Client:
            async def post(self, url, **kwargs):
                calls.append((url, kwargs))
                return Response()

        result = await collector._http_request(Client(), {"input": "question"}, "synthetic", 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["error"]["status"], 302)


if __name__ == "__main__":
    unittest.main()
