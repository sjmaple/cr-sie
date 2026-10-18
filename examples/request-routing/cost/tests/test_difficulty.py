"""Real GEPA integration, frozen selection, and reproducible difficulty routing."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from test_learning import FakeClient, fixture

import cloud
import difficulty
import learning
import replay_difficulty


@unittest.skipUnless(importlib.util.find_spec("gepa"), "install --group gepa")
class DifficultyTests(unittest.TestCase):
    def test_self_hosted_usage_is_explicit_and_replays_without_fictitious_credits(self):
        class LocalClient(FakeClient):
            def score(self, *args, **kwargs):
                reply = super().score(*args, **kwargs)
                reply["usage"] = {"input_tokens_details": {"content_tokens": 120}}
                return reply

            def chat_completions(self, **kwargs):
                reply = super().chat_completions(**kwargs)
                reply["usage"] = {"prompt_tokens": 300, "completion_tokens": 20, "total_tokens": 320}
                return reply

        s = fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            score_client, reflection_client = LocalClient(), LocalClient()
            c = cloud.Cloud(
                score_client,
                path,
                reflection_client=reflection_client,
                self_hosted_usd_per_token=1e-7,
                max_reflections=2,
            )
            try:
                result = difficulty.experiment(s, c, initial_size=4)
            finally:
                c.close()
            self.assertEqual(reflection_client.calls, 2)
            self.assertIsNone(result["cloud"]["observed_credits"])
            self.assertIsNone(result["cloud"]["calls_with_unknown_billing"])
            replay_difficulty.replay(path, s)
            calls = [json.loads(line) for line in (path / "calls.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["credits"] is None for r in calls))
            score_calls = [r for r in calls if r["kind"] == "score"]
            records = [json.loads(line) for line in (path / "scores.jsonl").read_text().splitlines()]
            self.assertAlmostEqual(sum(r["router_usd"] for r in records), len(score_calls) * 120e-7)
            calls[0]["content_tokens"] += 1
            (path / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in calls))
            with self.assertRaisesRegex(ValueError, "receipt cost mismatch"):
                replay_difficulty.replay(path, s)
        with tempfile.TemporaryDirectory() as temp:
            c = cloud.Cloud(LocalClient(), Path(temp))
            try:
                with self.assertRaises(cloud.CloudFailure):
                    c.reflect("Cloud mode still requires an actual credit receipt")
            finally:
                c.close()

    def test_gepa_freezes_before_test_replays_and_excludes_test_from_feedback(self):
        s = fixture()
        groups = learning.split(s, 4)
        # Distinct markers avoid confusing id-1 with id-10 in reflective text.
        s.prompts = [f"<request-{i}>" for i in range(len(s.ids))]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)

            class GuardedCloud(cloud.Cloud):
                def score(self, candidate, items):
                    test_ids = {s.ids[i] for i in groups["test"]}
                    if any(item["id"] in test_ids for item in items):
                        self_test.assertTrue((path / "selection.json").exists())
                    return super().score(candidate, items)

            self_test = self
            c = GuardedCloud(FakeClient(), path, max_reflections=2)
            try:
                result = difficulty.experiment(s, c, initial_size=4)
            finally:
                c.close()
            self.assertEqual(c.reflections, 2)
            self.assertEqual(result["test_n"], len(groups["test"]))
            self.assertEqual(set(result["test"]), {"static64", "calibration_only437", "gepa437"})
            replay_difficulty.replay(path, s)
            for file in path.glob("reflection-*.json"):
                prompt = json.loads(file.read_text())["messages"][0]["content"]
                for i in groups["test"] + groups["validation"]:
                    self.assertNotIn(s.prompts[i], prompt)
            original = json.loads((path / "selection.json").read_text())
            original_grades = s.pairs["openai"].q_cheap.copy()
            s.pairs["openai"].q_cheap[groups["test"]] = 0
            with tempfile.TemporaryDirectory() as other:
                a = difficulty.DifficultyAdapter(s, replay_difficulty.Recording(path, Path(other)), groups)
                self.assertEqual(difficulty.freeze(a, original["candidates"]), original)
            s.pairs["openai"].q_cheap[:] = original_grades
            expected = json.loads((path / "results.json").read_text())
            expected["test"]["static64"]["cost_per_1k"] += 1
            (path / "results.json").write_text(json.dumps(expected))
            with self.assertRaises(ValueError):
                replay_difficulty.replay(path, s)


if __name__ == "__main__":
    unittest.main()
