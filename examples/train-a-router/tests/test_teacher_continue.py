"""Known output exhaustion permits exactly one declared fallback, without APIs."""

import contextlib
import io
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import teacher_continue as collector
from data import file_hash, write


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.row = {"id": "new", "request": "What is 2+2?"}
        write(self.run / "records/old.json", {"preserved": True})
        self.plan = {
            "new_ids": ["new"],
            "reused_ids": ["old"],
            "prior_exhausted_ids": [],
            "max_usd": "40",
            "max_seconds": 3600,
            "workers": 4,
            "historical_conservative_usd": "4.7230875",
            "historical_list_price_estimate_usd": "2.85332",
            "prior_unknown_exposure_usd": "1.8009625",
        }
        write(self.run / "plan.json", self.plan)
        self.checked = patch.object(
            collector, "checked_plan", return_value=(self.plan, {"old": {"id": "old"}, "new": self.row})
        )
        self.checked.start()
        self.addCleanup(self.checked.stop)

    @staticmethod
    def payload(limit, complete):
        value = {
            "model": collector.original.MODEL,
            "status": "completed" if complete else "incomplete",
            "usage": {"input_tokens": 10, "output_tokens": 2 if complete else limit},
            "incomplete_details": None if complete else {"reason": "max_output_tokens"},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"answer":"4"}'}]}]
            if complete
            else [],
        }
        return {"payload": value}

    async def run_collector(self, request):
        with contextlib.redirect_stdout(io.StringIO()):
            return await collector.collect(self.run, secret="synthetic", request_fn=request)

    async def test_known_exhaustion_gets_one_fresh_question_only_fallback(self):
        calls = []
        original_hash = file_hash(self.run / "records/old.json")

        async def request(client, body, secret, seconds):
            calls.append(body)
            self.assertEqual(body["input"], self.row["request"])
            self.assertNotIn("previous_response_id", body)
            return self.payload(body["max_output_tokens"], len(calls) == 2)

        summary = await self.run_collector(request)
        self.assertEqual([body["max_output_tokens"] for body in calls], [8192, 32768])
        self.assertEqual(summary["fallback_attempts"], 1)
        self.assertEqual(summary["complete"], 2)
        self.assertEqual(file_hash(self.run / "records/old.json"), original_hash)
        self.assertEqual(len(summary["attempt_hashes"]), 2)
        self.assertEqual(json.loads((self.run / "records/new.json").read_text())["output_limit"], 32768)
        self.assertGreater(
            Decimal(summary["conservative_list_price_usd"]), Decimal(self.plan["historical_conservative_usd"])
        )

    async def test_fallback_exhaustion_is_missing_without_a_third_attempt(self):
        limits = []

        async def request(client, body, secret, seconds):
            limits.append(body["max_output_tokens"])
            return self.payload(body["max_output_tokens"], False)

        summary = await self.run_collector(request)
        self.assertEqual(limits, [8192, 32768])
        self.assertEqual(summary["missing_ids"], ["new"])
        self.assertFalse((self.run / "records/new.json").exists())

    async def test_unknown_or_429_stops_without_fallback(self):
        calls = []

        async def request(*args):
            calls.append(True)
            return {"error": {"kind": "http_error", "status": 429, "execution_unknown": True}}

        summary = await self.run_collector(request)
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary["stop_reason"], "request_error_or_unknown_execution")
        self.assertEqual(summary["fallback_attempts"], 0)
        self.assertEqual(
            Decimal(summary["conservative_list_price_usd"]),
            Decimal(self.plan["historical_conservative_usd"]) + collector.reservation(self.row, 8192),
        )
        with self.assertRaises(FileExistsError):
            await self.run_collector(request)

    async def test_historical_known_failure_starts_at_fallback_budget(self):
        self.plan["prior_exhausted_ids"] = ["new"]
        limits = []

        async def request(client, body, secret, seconds):
            limits.append(body["max_output_tokens"])
            return self.payload(body["max_output_tokens"], True)

        await self.run_collector(request)
        self.assertEqual(limits, [32768])


if __name__ == "__main__":
    unittest.main()
