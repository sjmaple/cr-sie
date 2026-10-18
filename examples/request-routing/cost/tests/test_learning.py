"""Budget, feedback-isolation and real GEPA integration contracts (no network)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import cloud
import learning
import optimize
import replay_learning
import study

CANDIDATE = {"description": "An easy request.", "instruction": "Judge relevance."}


class FakeClient:
    def __init__(self):
        self.calls = 0

    def score(self, model, query, items, **kwargs):
        self.calls += 1
        return {
            "scores": [
                {"item_id": item["id"], "score": 0.2 + (i % 5) * 0.1} for i, item in reversed(list(enumerate(items)))
            ],
            "usage": {"credits_charged": 2},
        }

    def chat_completions(self, **kwargs):
        self.calls += 1
        return {
            "choices": [{"message": {"content": "```\nA short, routine request.\n```"}}],
            "usage": {"credits_charged": 3},
        }


def fixture():
    n = 40
    o = study.Outcomes(np.ones(n), np.ones(n), np.full(n, 0.001), np.full(n, 0.01), np.zeros(n), np.zeros(n), [])
    return SimpleNamespace(
        ids=[f"id-{i}" for i in range(n)],
        prompts=[f"public request {i}" for i in range(n)],
        strata=np.array(["math"] * 20 + ["chat"] * 20),
        calibration=np.array([True, False] * 20),
        pairs={"openai": o},
        vectors=np.eye(n),
        credits=np.ones(n),
    )


class CloudTests(unittest.TestCase):
    def test_alignment_cache_receipts_and_item_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            c = cloud.Cloud(client, Path(temp), max_score_items=2)
            try:
                items = [{"id": "a", "request": "a"}, {"id": "b", "request": "b"}]
                values, usd = c.score(CANDIDATE, items)
                self.assertEqual(values, [0.2, 0.30000000000000004])
                self.assertEqual(usd, [0.00001, 0.00001])
                self.assertEqual(c.score(CANDIDATE, items[::-1])[0], values[::-1])
                with self.assertRaises(cloud.BudgetExceeded):
                    c.score(CANDIDATE, [{"id": "c", "request": "c"}])
                self.assertEqual(client.calls, 1)
                self.assertEqual(c.summary()["observed_credits"], 2)
            finally:
                c.close()
            ledger = [json.loads(line) for line in (Path(temp) / "calls.jsonl").read_text().splitlines()]
            self.assertEqual(ledger[0]["ids"], ["a", "b"])

    def test_credit_cap_and_failure_do_not_resend_or_log_exception_content(self):
        with tempfile.TemporaryDirectory() as temp:
            c = cloud.Cloud(FakeClient(), Path(temp), max_credits=2)
            try:
                c.reflect("public")
                with self.assertRaises(cloud.BudgetExceeded):
                    c.reflect("public")
                self.assertEqual(c.client.calls, 1)
            finally:
                c.close()
        with tempfile.TemporaryDirectory() as temp:
            c = cloud.Cloud(FakeClient(), Path(temp))
            try:
                with self.assertRaises(cloud.CloudFailure):
                    c.call("test", mock.Mock(side_effect=RuntimeError("private-detail")), {})
                with self.assertRaises(cloud.CloudFailure):
                    c.reflect("GEPA retries its callback")
                self.assertEqual(c.client.calls, 0)
            finally:
                c.close()
            self.assertNotIn("private-detail", (Path(temp) / "calls.jsonl").read_text())

    def test_receipt_requires_billed_credits_and_prefers_request_total(self):
        self.assertEqual(
            cloud.usage_of({"usage": {"credits_charged": 2}, "request": {"credits_debited": 5}})["credits"], 5
        )
        for value in (None, True, -1, float("nan")):
            with self.assertRaises(cloud.CloudFailure):
                cloud.usage_of({"usage": {"credits_charged": value}})

    def test_malformed_success_latches_failure_and_preserves_billing(self):
        for kind, reply in (
            ("reflection", {"choices": [], "usage": {"credits_charged": 3}}),
            ("score", {"scores": [None], "usage": {"credits_charged": 2}}),
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                client = FakeClient()
                method = "score" if kind == "score" else "chat_completions"
                with mock.patch.object(client, method, return_value=reply) as send:
                    c = cloud.Cloud(client, Path(temp))
                    try:
                        with self.assertRaises(cloud.CloudFailure):
                            if kind == "score":
                                c.score(CANDIDATE, [{"id": "a", "request": "a"}])
                            else:
                                c.reflect("public")
                        with self.assertRaises(cloud.CloudFailure):
                            c.reflect("callback retry")
                        self.assertEqual(send.call_count, 1)
                        self.assertEqual(c.credits, reply["usage"]["credits_charged"])
                        self.assertEqual(c.unknown_calls, 0)
                    finally:
                        c.close()


class LearningTests(unittest.TestCase):
    def test_promotion_needs_both_cost_and_quality_and_partition_isolation(self):
        baseline = {"cost_per_1k": 10, "quality_vs_strong": 0.99}
        self.assertTrue(learning.improvement({"cost_per_1k": 9, "quality_vs_strong": 0.99}, baseline))
        self.assertFalse(learning.improvement({"cost_per_1k": 9, "quality_vs_strong": 0.98}, baseline))
        self.assertFalse(learning.improvement({"cost_per_1k": 11, "quality_vs_strong": 1}, baseline))
        s = fixture()
        groups = learning.split(s, 4)
        self.assertTrue(set(groups["initial"]) < set(groups["train"]))
        self.assertFalse(set(groups["train"]) & set(groups["test"]))
        with self.assertRaises(ValueError):
            learning.split(s, -1)
        with self.assertRaises(ValueError):
            learning.evaluate(s.pairs["openai"], np.zeros(40), np.zeros(40), [0, 1], [1, 2])

    @unittest.skipUnless(importlib.util.find_spec("gepa"), "install --group gepa for the optimizer integration")
    def test_real_gepa_runs_with_sie_adapter_and_keeps_test_out_of_reflection(self):
        s = fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            c = cloud.Cloud(FakeClient(), path, max_reflections=2)
            spec = {"query": {"text": CANDIDATE["description"]}, "instruction": CANDIDATE["instruction"]}
            try:
                with mock.patch.object(optimize.reranker, "load", return_value=SimpleNamespace(spec=spec)):
                    result = optimize.experiment(s, "openai", c, initial_size=4, metric_calls=100)
            finally:
                c.close()
            self.assertEqual(c.reflections, 2)
            self.assertEqual(result["reranker"]["heldout"]["static"]["n"], 20)
            self.assertTrue((path / "selection.json").exists())
            replay_learning.replay(path, s)
            recorded = json.loads((path / "results.json").read_text())
            recorded["reranker"]["heldout"]["static"]["cost_per_1k"] += 1
            (path / "results.json").write_text(json.dumps(recorded))
            with self.assertRaises(ValueError):
                replay_learning.replay(path, s)
            for file in path.glob("reflection-*.json"):
                prompt = json.loads(file.read_text())["messages"][0]["content"]
                for i in learning.split(s, 4)["test"]:
                    self.assertNotIn(f"public request {i}\n", prompt)


if __name__ == "__main__":
    unittest.main()
