import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import arithmetic_teacher as acquisition
import arithmetic_teacher_protocol as numeric
import teacher_collect as teacher
from data import write


class TeacherAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs.json"
        self.rows = [{"id": "training-1", "request": "What is 3 plus 4?"}]
        write(self.inputs, self.rows)

    def prepare(self, **kwargs):
        run = self.root / "run"
        acquisition.prepare(self.inputs, run, workers=1, **kwargs)
        return run

    @staticmethod
    async def success(client, body, secret, seconds):
        return {
            "payload": {
                "id": "fake-response",
                "model": teacher.MODEL,
                "status": "completed",
                "service_tier": "default",
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"answer":"7"}'}]}],
            }
        }

    def test_question_only_boundary_rejects_gold_and_duplicate_ids(self):
        self.inputs.write_text(json.dumps([self.rows[0] | {"gold": "7"}]))
        with self.assertRaisesRegex(ValueError, "question-only"):
            acquisition.questions(self.inputs)
        self.inputs.write_text(json.dumps(self.rows * 2))
        with self.assertRaises(ValueError):
            acquisition.questions(self.inputs)

    def test_reuses_completed_answers_without_resending_or_reading_gold(self):
        run = self.prepare()
        called = []

        async def request(client, body, secret, seconds):
            called.append(body)
            self.assertEqual(body["input"], self.rows[0]["request"])
            self.assertNotIn("gold", body)
            return await self.success(client, body, secret, seconds)

        first = asyncio.run(acquisition.collect(run, secret="test-only", request_fn=request))
        second = asyncio.run(acquisition.collect(run, secret="test-only", request_fn=request))
        self.assertEqual(len(called), 1)
        self.assertEqual(first["complete"], 1)
        self.assertEqual(first["known_list_price_usd"], "0.0015")
        self.assertEqual(first["budget"]["settled_or_unknown_usd"], "0.00175")
        self.assertEqual(second["new_attempts"], 0)

    def test_budget_prevents_dispatch(self):
        run = self.prepare(max_usd="0.001")

        async def forbidden(*args):
            raise AssertionError("No budget means no request")

        result = asyncio.run(acquisition.collect(run, secret="test-only", request_fn=forbidden))
        self.assertEqual(result["new_attempts"], 0)
        self.assertEqual(result["pending"], ["training-1"])

    def test_interrupted_attempt_is_not_silently_repeated(self):
        run = self.prepare()
        write(
            run / "attempts/training-1/attempt-01/intent.json",
            {
                "request_sha256": acquisition.digest(numeric.body_for(self.rows[0])),
                "reservation_usd": str(numeric.reservation(self.rows[0])),
            },
        )
        result = asyncio.run(acquisition.collect(run, secret="test-only", request_fn=self.success))
        self.assertEqual(result["new_attempts"], 0)
        self.assertEqual(result["unavailable"], {"training-1": "uncertain_interrupted_attempt"})
        self.assertGreater(float(result["budget"]["settled_or_unknown_usd"]), 0)

    def test_unparseable_answer_is_retained_without_retry(self):
        run = self.prepare()

        async def malformed(client, body, secret, seconds):
            result = await self.success(client, body, secret, seconds)
            result["payload"]["output"][0]["content"][0]["text"] = "bad JSON"
            return result

        result = asyncio.run(acquisition.collect(run, secret="test-only", request_fn=malformed))
        self.assertEqual(result["physical_attempts"], 1)
        self.assertEqual(result["complete"], 0)
        self.assertEqual(result["known_list_price_usd"], "0.0015")
        raw = json.loads((run / "attempts/training-1/attempt-01/responses/training-1.json").read_text())
        self.assertEqual(raw["output"][0]["content"][0]["text"], "bad JSON")

    def test_retry_policy_never_retries_answer_content(self):
        self.assertTrue(acquisition.retryable({"error": {"kind": "http_error", "status": 429}}))
        self.assertTrue(acquisition.retryable({"error": {"kind": "ReadTimeout"}}))
        self.assertFalse(acquisition.retryable({"error": {"kind": "http_error", "status": 401}}))
        self.assertFalse(acquisition.retryable({"error": {"kind": "JSONDecodeError"}}))
        self.assertFalse(acquisition.retryable({"error": {"kind": "incomplete_or_changed_model"}}))

    def test_checkpoint_finishes_before_dispatch(self):
        run = self.prepare()
        persisted = []

        async def checkpoint():
            await asyncio.sleep(0)
            persisted.append((run / "attempts/training-1/attempt-01/intent.json").is_file())

        async def request(client, body, secret, seconds):
            self.assertEqual(persisted, [True])
            return await self.success(client, body, secret, seconds)

        result = asyncio.run(acquisition.collect(run, secret="test-only", request_fn=request, checkpoint=checkpoint))
        self.assertEqual(result["complete"], 1)

    def test_corrupt_raw_receipt_stops_resume_without_resending(self):
        run = self.prepare()
        asyncio.run(acquisition.collect(run, secret="test-only", request_fn=self.success))
        raw = run / "attempts/training-1/attempt-01/responses/training-1.json"
        raw.write_text("{}")
        with self.assertRaisesRegex(ValueError, "raw teacher response"):
            asyncio.run(acquisition.collect(run, secret="test-only", request_fn=self.success))

    def test_numeric_mode_binds_actual_request_and_preserves_non_numeric_answers(self):
        run = self.prepare()
        bodies = []

        async def units(client, body, secret, seconds):
            bodies.append(body)
            result = await self.success(client, body, secret, seconds)
            result["payload"]["output"][0]["content"][0]["text"] = '{"answer":"7 pens"}'
            return result

        result = asyncio.run(acquisition.collect(run, secret="test-only", request_fn=units))
        self.assertEqual(bodies, [numeric.body_for(self.rows[0])])
        self.assertEqual(result["physical_attempts"], 1)
        record = acquisition.read(run / "attempts/training-1/attempt-01/records/training-1.json")
        self.assertEqual(record["answer"], "7 pens")
        self.assertFalse(record["numeric_answer"])
        self.assertEqual(record["request_sha256"], acquisition.digest(bodies[0]))
        self.assertEqual(acquisition.read(run / "plan.json")["settings"], numeric.SETTINGS)

    def test_numeric_contract_rejects_units_and_zero_denominators(self):
        for value in ("42", "-3.5", "7/2", "+.25", "0"):
            self.assertTrue(numeric.numeric_answer(value))
        for value in ("7 pens", "$42", "1,234", "1/0", "NaN", "2e3", ""):
            self.assertFalse(numeric.numeric_answer(value))


if __name__ == "__main__":
    unittest.main()
