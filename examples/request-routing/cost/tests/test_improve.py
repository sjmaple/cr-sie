"""Reserved-audit isolation and freezing the complete router before evaluation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from test_learning import CANDIDATE, FakeClient, fixture

import cloud
import improve
import replay_improvement


class ImproveTests(unittest.TestCase):
    def test_audit_excludes_all_prior_scored_requests_and_partitions_are_disjoint(self):
        s = SimpleNamespace(
            ids=[f"id-{i}" for i in range(1000)],
            calibration=np.array([True] * 729 + [False] * 271),
            strata=np.array(["a", "b", "c", "d"] * 250),
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "scores.jsonl").write_text("".join(json.dumps({"id": s.ids[i]}) + "\n" for i in range(200)))
            groups = improve.partitions(s, path)
            self.assertEqual(groups, improve.partitions(s, path))
            self.assertFalse(set(groups["audit"]) & set(range(200)))
            rows = [i for indices in groups.values() for i in indices]
            self.assertEqual(len(rows), len(set(rows)))
            self.assertEqual(set(rows), set(range(729)))
            self.assertEqual(
                {k: len(v) for k, v in groups.items()}, {"fit": 128, "train": 249, "validation": 160, "audit": 192}
            )

    @unittest.skipUnless(importlib.util.find_spec("gepa"), "install --group gepa")
    def test_gepa_feedback_is_training_only_and_audit_waits_for_frozen_selection(self):
        s = fixture()
        groups = {
            "fit": list(range(4)),
            "train": list(range(4, 28)),
            "validation": list(range(28, 34)),
            "audit": list(range(34, 40)),
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)

            class GuardedCloud(cloud.Cloud):
                def score(self, candidate, items):
                    if any(item["id"] in {s.ids[i] for i in groups["audit"]} for item in items):
                        self.assert_frozen()
                    return super().score(candidate, items)

                def assert_frozen(self):
                    if not (path / "selection.json").exists():
                        raise AssertionError("audit scored before selection")

            c = GuardedCloud(FakeClient(), path, max_reflections=2)
            spec = {"query": {"text": CANDIDATE["description"]}, "instruction": CANDIDATE["instruction"]}
            try:
                with mock.patch.object(improve.reranker, "load", return_value=SimpleNamespace(spec=spec)):
                    result = improve.experiment(s, c, groups)
            finally:
                c.close()
            self.assertEqual(result["cloud"]["reflections"], 2)
            for file in path.glob("reflection-*.json"):
                content = json.loads(file.read_text())["messages"][0]["content"]
                for i in groups["validation"] + groups["audit"]:
                    self.assertNotIn(f"public request {i}\n", content)
            chosen = json.loads((path / "selection.json").read_text())
            router = json.loads((path / "router.json").read_text())
            self.assertEqual(router["tau"], chosen["thresholds"][chosen["chosen"]])
            report = replay_improvement.replay(path, s, groups)
            self.assertEqual(report["uncertainty"]["resamples"], 4000)
            original = json.loads((path / "selection.json").read_text())
            original["thresholds"][0] = 0.999
            (path / "selection.json").write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                replay_improvement.replay(path, s, groups)


if __name__ == "__main__":
    unittest.main()
