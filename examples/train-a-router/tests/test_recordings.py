import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import decision
from data import digest, file_hash
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

    def test_native_route_rejects_changed_frozen_inputs_before_inference(self):
        for changed in ("plan.json", "questions.json", "full/model.safetensors"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                (run / "full").mkdir()
                (run / "full/model.safetensors").write_bytes(b"test checkpoint")
                (run / "questions.json").write_text('{"difficulty": "test question"}')
                (run / "plan.json").write_text(
                    json.dumps({"data_sha256": {"questions.json": file_hash(run / "questions.json")}})
                )
                (run / "frozen.json").write_text(
                    json.dumps(
                        {
                            "plan_sha256": file_hash(run / "plan.json"),
                            "selected": {"full": {"checkpoint_sha256": file_hash(run / "full/model.safetensors")}},
                        }
                    )
                )
                with (run / changed).open("ab") as handle:
                    handle.write(b" ")
                with (
                    patch("sys.argv", ["decision.py", "route", "--run", str(run)]),
                    patch.object(decision, "SingleSendClient") as client,
                    self.assertRaisesRegex(ValueError, "Frozen plan or questions changed|Checkpoint does not match"),
                ):
                    decision.main()
                client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
