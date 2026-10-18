"""GEPA handler descriptions: score alignment, training-only feedback and audit timing."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from test_learning import FakeClient

import cloud
import intent_gepa
import replay_intents


@unittest.skipUnless(importlib.util.find_spec("gepa"), "install --group gepa")
class IntentTests(unittest.TestCase):
    def test_full_optimizer_keeps_test_out_of_reflection_and_selects_before_test(self):
        groups = {
            name: [
                {"id": f"{name}-{label}", "request": f"{name} request for {label}", "label": label}
                for label in intent_gepa.HANDLERS
            ]
            for name in ("train", "validation", "test")
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)

            class Guarded(cloud.Cloud):
                def score(self, candidate, items):
                    if any(item["id"].startswith("test-") for item in items) and not (path / "selection.json").exists():
                        raise AssertionError("test before selection")
                    return super().score(candidate, items)

            c = Guarded(FakeClient(), path, max_reflections=2)
            try:
                result = intent_gepa.experiment({"splits": groups}, c)
            finally:
                c.close()
            self.assertEqual(result["test_n"], 8)
            self.assertEqual(result["chosen"], 0)
            self.assertEqual(result["test_accuracy"]["seed"], 0)  # tied scores never count as correct
            replay_intents.replay(path)
            result["test_accuracy"]["seed"] = 1
            (path / "results.json").write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                replay_intents.replay(path)
            for file in path.glob("reflection-*.json"):
                text = json.loads(file.read_text())["messages"][0]["content"]
                self.assertNotIn("test request for", text)
                self.assertNotIn("validation request for", text)

    def test_cross_split_request_leak_is_rejected(self):
        groups = {
            name: [
                {"id": f"{name}-{label}", "request": f"{name} {label}", "label": label}
                for label in intent_gepa.HANDLERS
            ]
            for name in ("train", "validation", "test")
        }
        intent_gepa.validate_data({"splits": groups})
        groups["test"][0]["request"] = "  " + groups["train"][0]["request"].upper()
        with self.assertRaisesRegex(ValueError, "overlaps"):
            intent_gepa.validate_data({"splits": groups})


if __name__ == "__main__":
    unittest.main()
