import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import teacher_reference_collect as audit
from data import file_hash, render_request


class ReferenceAuditTests(unittest.TestCase):
    def test_public_join_rejects_added_content_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            pq.write_table(pa.Table.from_pylist([{"tag": "AMC", "problem": "What is 1+1?"}]), source / "train.parquet")
            hashes = {"train.parquet": file_hash(source / "train.parquet")}
            row = {"id": "e2h-amc-train-0", "request": render_request("AMC", "What is 1+1?")}
            with patch.object(audit, "COUNTS", {"train": 1}), patch.object(audit, "SOURCE_HASHES", hashes):
                self.assertEqual(audit.verify_public_inputs({"train": [row]}, source), [row])
                with self.assertRaises(ValueError):
                    audit.verify_public_inputs({"train": [row | {"request": row["request"] + "\nAnswer:2"}]}, source)
                with self.assertRaises(ValueError):
                    audit.verify_public_inputs({"train": [row | {"gold": "2"}]}, source)
            with (
                patch.object(audit, "COUNTS", {"train": 2}),
                patch.object(audit, "SOURCE_HASHES", hashes),
                self.assertRaises(ValueError),
            ):
                audit.verify_public_inputs({"train": [row, row]}, source)

    def test_collect_preserves_reuse_and_records_retry_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            rows = {"old": {"id": "old", "request": "1+1?"}, "new": {"id": "new", "request": "2+2?"}}
            plan = {
                "n": 2,
                "reused_ids": ["old"],
                "pending": [{"id": "new", "output_limit": 8192, "transient_failures": 0}],
                "previously_exhausted_ids": [],
                "max_usd": "5",
                "historical_conservative_usd": "0.1",
                "historical_list_price_estimate_usd": "0.09",
                "prior_unknown_exposure_usd": "0",
                "max_seconds": 900,
                "workers": 16,
            }
            (run / "records").mkdir()
            original = b'{"answer":"2","preserved":true}\n'
            (run / "records/old.json").write_bytes(original)
            (run / "plan.json").write_text(json.dumps(plan))
            calls = []

            async def request(_client, body, _secret, _seconds):
                calls.append(body)
                if len(calls) == 1:
                    return {"error": {"kind": "http_error", "status": 520, "execution_unknown": True}}
                return {
                    "payload": {
                        "id": "r",
                        "model": audit.teacher_collect.MODEL,
                        "status": "completed",
                        "service_tier": "default",
                        "usage": {"input_tokens": 5, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 0}},
                        "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"answer":"4"}'}]}],
                    }
                }

            with patch.object(audit, "checked_plan", return_value=(plan, rows)):
                result = asyncio.run(audit.collect(run, secret="mock-not-a-secret", request_fn=request))
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["attempts"], 2)
                self.assertGreater(float(result["unknown_exposure_usd"]), 0)
                self.assertFalse(result["invoice_estimate_complete"])
                self.assertEqual((run / "records/old.json").read_bytes(), original)
                self.assertEqual(calls[0], calls[1])
                self.assertEqual(calls[0]["input"], "2+2?")
                with self.assertRaises(FileExistsError):
                    asyncio.run(audit.collect(run, secret="mock-not-a-secret", request_fn=request))
                self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
