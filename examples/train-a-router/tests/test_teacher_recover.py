"""Bounded provider recovery chooses by delivery order, never answer quality."""

import contextlib
import io
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import teacher_recover as recovery
from data import file_hash, write


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.row = {"id": "new", "request": "What is2+2?"}
        write(self.run / "records/old.json", {"immutable": "historical"})
        self.plan = {
            "pending": [{"id": "new", "output_limit": 8192, "transient_failures": 0}],
            "reused_ids": ["old"],
            "previously_exhausted_ids": [],
            "max_usd": "40",
            "max_seconds": 3600,
            "workers": 4,
            "historical_conservative_usd": "5",
            "historical_list_price_estimate_usd": "3",
            "prior_unknown_exposure_usd": "1.8",
        }
        write(self.run / "plan.json", self.plan)
        self.patches = patch.object(
            recovery, "checked_plan", return_value=(self.plan, {"old": {"id": "old"}, "new": self.row})
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)

    @staticmethod
    def completed():
        return {
            "payload": {
                "model": recovery.teacher_collect.MODEL,
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"answer":"5"}'}]}],
            }
        }

    async def test_transient520_retries_then_keeps_first_valid_answer(self):
        calls = []
        original = file_hash(self.run / "records/old.json")

        async def request(client, body, secret, seconds):
            calls.append(body)
            if len(calls) == 1:
                return {"error": {"kind": "http_error", "status": 520, "execution_unknown": True}}
            return self.completed()

        with contextlib.redirect_stdout(io.StringIO()):
            result = await recovery.collect(self.run, secret="synthetic", request_fn=request)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["complete"], 2)
        self.assertEqual(result["selected_attempts"]["new"], "calls/new-02/attempts/new-8192.json")
        self.assertEqual(file_hash(self.run / "records/old.json"), original)
        self.assertGreater(Decimal(result["unknown_exposure_usd"]), Decimal("1.8"))
        self.assertFalse(result["invoice_estimate_complete"])

    async def test_exhausted_retries_leave_explicit_missing_id(self):
        self.plan["pending"][0]["transient_failures"] = 2
        calls = []

        async def request(*args):
            calls.append(True)
            return {"error": {"kind": "http_error", "status": 429, "execution_unknown": True}}

        with contextlib.redirect_stdout(io.StringIO()):
            result = await recovery.collect(self.run, secret="synthetic", request_fn=request)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["missing_ids"], ["new"])
        self.assertEqual(result["stop_reason"], None)

    def test_output_fallback_and_transient_allowance_are_separate(self):
        state = {"id": "new", "output_limit": 8192, "transient_failures": 1}
        record = {
            "status": "incomplete",
            "usage_verified": True,
            "response_status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }
        action, state = recovery.next_state(state, record)
        self.assertEqual((action, state["output_limit"], state["transient_failures"]), ("fallback", 32768, 1))
        action, state = recovery.next_state(state, record)
        self.assertEqual(action, "missing")

    def test_auth_failure_is_terminal_and_correctness_never_appears(self):
        state = {"id": "new", "output_limit": 8192, "transient_failures": 0}
        action, _ = recovery.next_state(state, {"status": "incomplete", "error": {"kind": "http_error", "status": 401}})
        self.assertEqual(action, "terminal")
        action, _ = recovery.next_state(state, {"status": "complete", "answer": "mathematically wrong"})
        self.assertEqual(action, "complete")


if __name__ == "__main__":
    unittest.main()
