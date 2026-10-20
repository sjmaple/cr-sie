"""Additional embeddings keep the original input/vector pairs unchanged."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benefit_embeddings as embeddings
import serving
from data import digest, file_hash, write


def recording(directory, rows):
    write(
        directory / "plan.json",
        {
            "model": serving.MODEL,
            "revision": serving.REVISION,
            "instruction": serving.INSTRUCTION,
            "ids": [row["id"] for row in rows],
            "input_sha256": {row["id"]: digest(row["request"]) for row in rows},
        },
    )
    write(directory / "collection.json", {"credits": len(rows)})
    with gzip.open(directory / "vectors.jsonl.gz", "wt") as handle:
        for row in rows:
            handle.write(json.dumps({"id": row["id"], "vector": [1] + [0] * 2559}) + "\n")


class EmbeddingReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.training = self.root / "training"
        write(self.training / "plan.json", {"eligible": ["old", "new"]})
        self.rows = [{"id": "old", "request": "First"}, {"id": "new", "request": "Second"}]
        self.old = self.root / "old"
        recording(self.old, [self.rows[0], {"id": "unused", "request": "Not part of training"}])
        self.fresh = self.root / "fresh"
        self.plan = {"new_train_ids": ["new"]}
        mock = patch.object(embeddings, "checked_training", return_value=(self.plan, {"train": self.rows}, {}))
        mock.start()
        self.addCleanup(mock.stop)

    def acquire(self):
        with patch.object(serving, "collect", side_effect=lambda rows, output, **_: recording(output, rows)) as call:
            embeddings.new_vectors(self.training, self.fresh)
            self.assertEqual(call.call_args.args[0], [self.rows[1]])

    def test_combined_vector_lines_are_exactly_reused(self):
        self.acquire()
        output = self.root / "combined"
        embeddings.combine(self.training, self.old, output, self.fresh)
        with gzip.open(self.old / "vectors.jsonl.gz", "rb") as handle:
            original_line = next(handle)
        with gzip.open(output / "vectors.jsonl.gz", "rb") as handle:
            lines = list(handle)
        self.assertEqual(lines[0], original_line)
        self.assertEqual([json.loads(line)["id"] for line in lines], ["old", "new"])
        receipt = json.loads((output / "collection.json").read_text())
        self.assertEqual(receipt["additional_observed_credits"], 1)

    def test_reentry_cannot_repeat_new_calls(self):
        self.acquire()
        with patch.object(serving, "collect") as call, self.assertRaises(FileExistsError):
            embeddings.new_vectors(self.training, self.fresh)
        call.assert_not_called()

    def test_different_training_plan_cannot_be_combined(self):
        self.acquire()
        (self.training / "plan.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "different source"):
            embeddings.combine(self.training, self.old, self.root / "combined", self.fresh)

    def test_control_needs_no_new_calls(self):
        self.plan["new_train_ids"] = []
        self.rows.pop()
        output = self.root / "control"
        embeddings.combine(self.training, self.old, output)
        self.assertEqual(serving.vectors(self.rows, output).shape, (1, 2560))
        self.assertEqual(
            json.loads((output / "plan.json").read_text())["source_collections_sha256"]["reused"]["plan.json"],
            file_hash(self.old / "plan.json"),
        )


if __name__ == "__main__":
    unittest.main()
