"""Small mocked lifecycle; no provider requests, real scheduler/attempt verification."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect_labels as api


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prepared, self.source, self.output = self.root / "prepared", self.root / "source", self.root / "run"
        self.prepared.mkdir()
        self.source.mkdir()
        contract = {"type": "choice", "allowed": ["A", "B"]}
        self.inputs = [
            {"id": "train-one", "role": "train", "request": "Which label? A or B", "response_contract": contract}
        ]
        self.rows = [self.inputs[0] | {"source": "mmlupro", "family": "family-one", "task": "toy", "fold": 0}]
        gold = {"train-one": {"response_contract": contract, "answer": "A"}}
        api.acquisition.dump(self.prepared / "gold.json", gold)
        api.acquisition.dump(self.prepared / "rows.json", self.rows)
        api.acquisition.dump(self.prepared / "question-only/inputs.json", self.inputs)
        self.source_plan = {
            "files_sha256": {
                name: api.sha(self.prepared / name) for name in ("gold.json", "rows.json", "question-only/inputs.json")
            }
        }
        api.acquisition.dump(self.prepared / "source-plan.json", self.source_plan)
        self.pair = api.read(api.SOURCES / "pair-spec.json")
        self.pair["source_plan_sha256"] = api.sha(self.prepared / "source-plan.json")
        self.pair["source_all_inputs_sha256"] = api.sha(self.prepared / "question-only/inputs.json")
        api.acquisition.dump(self.source / "pair-spec.json", self.pair)
        self.patches = [
            patch.object(api, "SOURCES", self.source),
            patch.dict(api.EXPECTED, {"pair-spec.json": api.sha(self.source / "pair-spec.json")}),
            patch.dict(api.COUNTS, {"train": 1}),
            patch.object(api.os, "sync"),
            patch.object(api.acquisition, "retry_delay", return_value=0),
        ]
        for item in self.patches:
            item.start()
        self.calls = Counter()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def response(self, item, arm):
        self.calls[arm] += 1
        body = api.collect.body_for(item, arm)
        base = {
            "id": item["id"],
            "arm": arm,
            "model": body["model"],
            "body_sha256": api.acquisition.digest(json.dumps(body, sort_keys=True).encode()),
        }
        if arm == "luna" and self.calls[arm] == 1:
            return base | {"status": "failed_or_unknown", "terminal_response": False, "error_type": "ReadTimeout"}
        usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        # The strong arm returns a paid invalid final. It must never be retried.
        content = '{"answer":"A"}' if arm == "luna" else '{"answer":"invalid"}'
        answer = api.collect.parse_answer(content, item["response_contract"])
        return base | {
            "status": "complete",
            "terminal_response": True,
            "resolved_model": body["model"],
            "service_tier": "default",
            "provider_status": "completed",
            "http_status": 200,
            "messages": [{"type": "output_text", "text": content}],
            "usage": usage,
            "cost": api.collect.openai_cost(usage, body),
            "answer": answer,
            "valid_answer": answer is not None,
        }

    def test_unknown_retry_invalid_final_no_retry_resume_and_grade(self):
        result = api.collect_run(self.prepared, self.output, ["train"], "2", call=self.response)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.calls, {"luna": 2, "sol": 1})
        self.assertGreater(float(result["unknown_reservation_usd"]), 0)
        before = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob("*-response.json")}
        api.collect_run(self.prepared, self.output, ["train"], "2", resume=True, call=self.response)
        self.assertEqual(self.calls, {"luna": 2, "sol": 1})
        self.assertTrue(all((self.output / p).read_bytes() == raw for p, raw in before.items()))
        output = self.root / "grades.json"
        api.grade_run(self.prepared, self.output, output, "train")
        grade = api.read(output)
        self.assertTrue(grade["reproduction"])
        self.assertEqual((grade["rows"][0]["cheap_correct"], grade["rows"][0]["strong_correct"]), (1, 0))
        self.assertGreater(grade["rows"][0]["strong_cost_usd"], 0)
        self.assertEqual(grade["provenance"]["physical_acquisition"]["physical_attempts"], 3)
        self.assertGreater(float(grade["provenance"]["physical_acquisition"]["unknown_reservation_usd"]), 0)

    def test_final_requires_separate_frozen_gate(self):
        with self.assertRaisesRegex(ValueError, "separate gated"):
            api.plan_run(self.prepared, ["train", "final"], "2")
        with patch.dict(api.COUNTS, {"final": 0}):
            with self.assertRaisesRegex(ValueError, "frozen route"):
                api.plan_run(self.prepared, ["final"], "2")
        self.assertFalse(self.calls)

    def test_changed_inputs_rejected_before_dispatch(self):
        api.acquisition.dump(self.prepared / "question-only/inputs.json", [])
        with self.assertRaisesRegex(ValueError, "source bytes"):
            api.collect_run(self.prepared, self.output, ["train"], "2", call=self.response)
        self.assertFalse(self.calls)

    def test_undeclared_attempt_cannot_disappear_from_accounting(self):
        api.collect_run(self.prepared, self.output, ["train"], "2", call=self.response)
        extra = self.output / "train/sol/attempts/undeclared/001-intent.json"
        api.acquisition.dump(extra, {"physical": "undeclared"})
        with self.assertRaisesRegex(ValueError, "Undeclared/missing physical"):
            api.grade_run(self.prepared, self.output, self.root / "grades.json", "train")

    def test_resume_cannot_change_budget_or_saved_questions(self):
        api.collect_run(self.prepared, self.output, ["train"], "2", call=self.response)
        with self.assertRaisesRegex(ValueError, "Resume cannot change"):
            api.collect_run(self.prepared, self.output, ["train"], "3", resume=True, call=self.response)
        api.acquisition.dump(self.output / "train/inputs.json", [])
        with self.assertRaisesRegex(ValueError, "Saved input"):
            api.collect_run(self.prepared, self.output, ["train"], "2", resume=True, call=self.response)
        self.assertEqual(self.calls, {"luna": 2, "sol": 1})


if __name__ == "__main__":
    unittest.main()
