"""Independent-label difficulty routing: split, threshold and error accounting."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import difficulty_policy as policy
import gepa_teacher


class DifficultyPolicyTests(unittest.TestCase):
    def test_teacher_batch_order_is_independent_of_labels_and_input_order(self):
        class Client:
            def __init__(self):
                self.payloads = []

            def chat_completions(self, **kwargs):
                problems = json.loads(kwargs["messages"][1]["content"])
                self.payloads.append(problems)
                scores = [int(row["problem"].split()[-1]) for row in problems]
                return {"choices": [{"message": {"content": json.dumps({"scores": scores})}}]}

        class Cloud:
            def __init__(self, directory):
                self.directory, self.client = directory, Client()
                self.requests, self.credits = 0, 0

            def call(self, kind, send, metadata):
                self.requests += 1
                return send()

        rows = [{"id": str(i), "request": f"problem {i}", "label": i % 2} for i in range(20)]
        reversed_flipped = [{**row, "label": 1 - row["label"]} for row in reversed(rows)]
        payloads = []
        for variant in (rows, reversed_flipped):
            with tempfile.TemporaryDirectory() as temp:
                cloud = Cloud(Path(temp))
                teacher = gepa_teacher.Teacher(cloud)
                try:
                    values = teacher.values(variant, {"rubric": "test"})
                    np.testing.assert_array_equal(values, [int(row["id"]) for row in variant])
                    payloads.append(cloud.client.payloads)
                finally:
                    teacher.close()
        self.assertEqual(payloads[0], payloads[1])

    def test_teacher_scores_require_complete_integer_batch(self):
        self.assertEqual(gepa_teacher.parse('{"scores": [0, 50, 100]}', 3), [0, 50, 100])
        self.assertEqual(gepa_teacher.parse("```json\n[0, 50, 100]\n```", 3), [0, 50, 100])
        for answer in ("[0, 50]", "[0, true, 100]", "[0, 50, 101]", '{"scores": [0, 50, 100], "extra": 1}'):
            with self.assertRaises(ValueError):
                gepa_teacher.parse(answer, 3)

    def test_equal_scores_cannot_be_split_to_inflate_accuracy(self):
        y, values = np.array([0, 0, 1, 1]), np.full(4, 0.5)
        tau = policy.threshold(values, y)
        self.assertEqual(tau, 0.5)
        self.assertEqual(policy.metrics(values, y, tau)["balanced_accuracy"], 0.5)

    def test_loo_nearest_neighbor_excludes_own_label(self):
        x = np.eye(3)
        y = np.array([0, 1, 1])
        np.testing.assert_allclose(policy.knn(x, y, x, loo=True, k=2), [1, 0.5, 0.5])

    def test_cross_split_normalized_duplicates_rejected(self):
        data = {
            "splits": {
                split: [{"id": f"{split}-{i}", "request": f"{split} problem {i}", "label": i} for i in (0, 1)]
                for split in ("train", "validation", "test")
            }
        }
        policy.validate(data)
        data["splits"]["test"][0]["request"] = " TRAIN  problem 0 "
        with self.assertRaises(ValueError):
            policy.validate(data)

    def test_paired_errors_are_counted_on_same_requests(self):
        y = np.array([0, 0, 1, 1])
        result = policy.paired(y, np.array([1, 1, 1, 1]), np.array([0, 0, 0, 1]))
        self.assertEqual(result["corrected"], 2)
        self.assertEqual(result["regressed"], 1)
        self.assertEqual(result["relative_error_reduction_pct"], 50)
        self.assertEqual(result["gain_pp"], 25)

    def test_frozen_end_to_end_replay_and_perfect_baseline(self):
        class Client:
            sends = 0

            def get_model(self, model):
                return {"dims": {"dense": 2}, "revision": "test"}

            def encode(self, model, items, **kwargs):
                self.sends += 1
                receipt = {"id": str(self.sends), "credits_debited": 1}
                return [
                    {"id": r["id"], "dense": [0, 1] if "advanced" in r["text"] else [1, 0], "request": receipt}
                    for r in items
                ]

            def score(self, model, query, items, **kwargs):
                self.sends += 1
                advanced = "challenging" in query["text"].lower()
                return {
                    "scores": [
                        {"item_id": r["id"], "score": 0.9 if advanced == ("advanced" in r["text"]) else 0.1}
                        for r in items
                    ],
                    "request": {"credits_debited": 1},
                }

        data = {
            "splits": {
                split: [
                    {
                        "id": f"{split}-{i}",
                        "request": f"{split} {i} {'advanced' if i % 2 else 'routine'}",
                        "label": i % 2,
                    }
                    for i in range(8)
                ]
                for split in ("train", "validation", "test")
            }
        }
        client = Client()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            policy.collect_vectors(
                data["splits"]["train"] + data["splits"]["validation"], directory / "dev-embeddings", client
            )
            policy.fit_embeddings(data, directory)
            policy.learn.write(
                directory / "reranker-selection.json",
                {
                    "policies": {
                        name: {"candidate": policy.SEED, "tau": 0} for name in ("static64", "calibrated512", "gepa512")
                    }
                },
            )
            policy.freeze(data, directory)
            result = policy.test_stage(data, directory, client)
            self.assertFalse(result["success_gate_passed"])
            self.assertIsNone(result["comparisons"]["static64_to_gepa512"]["relative_error_reduction_pct"])
            sends = client.sends
            self.assertEqual(result, policy.test_stage(data, directory))
            self.assertEqual(sends, client.sends)
            predictions = json.loads((directory / "predictions.json").read_text())
            predictions["gepa512"][0] = 1
            policy.learn.write(directory / "predictions.json", predictions)
            with self.assertRaisesRegex(ValueError, "saved routes"):
                policy.test_stage(data, directory)


if __name__ == "__main__":
    unittest.main()
