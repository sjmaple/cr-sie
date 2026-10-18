"""Embedding collection must preserve split isolation and count each batch receipt once."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_learning import fixture

import encoder_comparison
import study


class Client:
    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    def get_model(self, model):
        return {"name": model, "dims": {"dense": 3}, "revision": "fixture"}

    def encode(self, model, items, **kwargs):
        self.calls.append((items, kwargs))
        request = {"id": str(len(self.calls)), "credits_debited": 2, "usage": {"input_tokens": len(items)}}
        return [
            {"id": item["id"], "dense": [1.0, 2.0, float("nan") if self.invalid else 3.0], "request": request}
            for item in items
        ]


class EncoderComparisonTests(unittest.TestCase):
    def test_shared_receipt_counted_once_and_only_development_ids_collected(self):
        fake = fixture()
        s = study.Study(**vars(fake), raw_vectors=fake.vectors, results={})
        groups = {"train": list(range(16)), "validation": list(range(16, 20))}
        client = Client()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(encoder_comparison.learning, "split", return_value=groups),
        ):
            path = Path(temp)
            encoder_comparison.collect(s, path, client, instruction="Represent routing difficulty.")
            self.assertEqual(json.loads((path / "collection.json").read_text())["observed_credits"], 4)
            self.assertEqual([r["id"] for r in study.read_jsonl(path / "vectors.jsonl.gz")], s.ids[:20])
            self.assertTrue(
                all(
                    kwargs["is_query"] and kwargs["instruction"] == "Represent routing difficulty."
                    for _, kwargs in client.calls
                )
            )
            revised = encoder_comparison.load_vectors(s, path)
            np.testing.assert_allclose(np.linalg.norm(revised.vectors[:20], axis=1), 1)
            np.testing.assert_array_equal(revised.credits, s.credits)
            self.assertEqual(revised.vectors.shape, (len(s.ids), 3))

    def test_invalid_reply_is_recorded_and_replay_refuses_incomplete_data(self):
        s = fixture()
        groups = {"train": list(range(4)), "validation": [4, 5]}
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(encoder_comparison.learning, "split", return_value=groups),
        ):
            path = Path(temp)
            with self.assertRaises(ValueError):
                encoder_comparison.collect(s, path, Client(invalid=True))
            self.assertEqual(study.read_jsonl(path / "calls.jsonl")[0]["status"], "failed_or_unknown")
            with self.assertRaises(ValueError):
                encoder_comparison.load_vectors(s, path)


if __name__ == "__main__":
    unittest.main()
