import gzip
import json
import tempfile
import unittest
from pathlib import Path

from data import digest
from serving import INSTRUCTION, MODEL, REVISION, error_receipt, vectors


class RecordingTests(unittest.TestCase):
    def test_vectors_reject_changed_input_with_same_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            row = {"id": "one", "request": "original problem"}
            (path / "plan.json").write_text(
                json.dumps(
                    {
                        "model": MODEL,
                        "revision": REVISION,
                        "instruction": INSTRUCTION,
                        "ids": ["one"],
                        "input_sha256": {"one": digest(row["request"])},
                    }
                )
            )
            with gzip.open(path / "vectors.jsonl.gz", "wt") as handle:
                handle.write(json.dumps({"id": "one", "vector": [1.0, 0.0]}) + "\n")
            self.assertEqual(vectors([row], path).tolist(), [[1.0, 0.0]])
            with self.assertRaisesRegex(ValueError, "different input"):
                vectors([dict(row, request="changed problem")], path)

    def test_error_receipt_does_not_publish_exception_text(self):
        error = RuntimeError("private request text")
        self.assertNotIn("private", json.dumps(error_receipt(error)))
        self.assertIsNone(error_receipt(error)["credits"])


if __name__ == "__main__":
    unittest.main()
