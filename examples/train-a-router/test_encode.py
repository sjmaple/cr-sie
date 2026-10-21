import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import encode


class FakeClient:
    def __init__(self, *, fail=False, self_hosted=False):
        self.calls = []
        self.catalog_calls = 0
        self.fail = fail
        self.self_hosted = self_hosted
        self.last_retry_count = 0
        self.last_model_revision = "a" * 64

    def get_model(self, model):
        self.catalog_calls += 1
        return {"name": model, "revision": encode.core.REVISION, "dims": {"dense": 2560}}

    def encode(self, **request):
        self.calls.append(request)
        if self.fail:
            raise TimeoutError("No response was received")
        metadata = (
            {}
            if self.self_hosted
            else {
                "id": "request-" + str(len(self.calls)),
                "credits_debited": 7,
                "usage": {"input_tokens": 100, "credits_charged": 7},
            }
        )
        return [
            {
                "id": item["id"],
                "model": request["model"],
                "request": copy.deepcopy(metadata),
                "dense": [3.0, 4.0] + [0.0] * 2558,
            }
            for item in reversed(request["items"])
        ]


class EncodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name) / "new-features"
        self.rows = [{"id": f"q-{i}", "request": f"Question {i}?"} for i in range(3)]
        self.plan = {"batch_size": 2, "self_hosted": False, "inputs_sha256": "b" * 64, "source_plan_sha256": "c" * 64}

    def tearDown(self):
        self.temporary.cleanup()

    def run_collect(self, client, **kwargs):
        return encode.collect_rows(self.rows, self.output, self.plan, client, **kwargs)

    def test_alignment_normalization_and_each_batch_charge_counted_once(self):
        client = FakeClient()
        result = self.run_collect(client)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["known_cloud_credits"], "14")
        self.assertEqual(result["physical_attempts"], 2)
        matrix, _ = encode.core.feature_matrix({"source_inputs_sha256": "b" * 64}, self.rows, self.output)
        self.assertEqual(matrix.dtype, np.float32)
        np.testing.assert_allclose(matrix[:, :2], [[0.6, 0.8]] * 3, rtol=0, atol=3e-8)
        self.assertEqual(encode.core.read(self.output / "ids.json"), [row["id"] for row in self.rows])
        for request in client.calls:
            self.assertEqual(request["instruction"], encode.core.INSTRUCTION)
            self.assertEqual(request["model"], encode.core.MODEL)
            self.assertTrue(request["is_query"])
            self.assertEqual(request["output_dtype"], "float32")
            self.assertFalse(request["wait_for_capacity"])
            self.assertEqual(request["max_oom_retries"], 0)

    def test_resume_skips_completed_batches_and_complete_run_makes_no_calls(self):
        first = FakeClient()
        self.assertEqual(self.run_collect(first, max_batches=1)["complete_rows"], 2)
        self.assertFalse((self.output / "features.json").exists())
        digest = encode.core.sha(self.output / "batches/000000/attempt-000/receipt.json")
        second = FakeClient()
        result = self.run_collect(second)
        self.assertEqual([item["id"] for item in second.calls[0]["items"]], ["q-2"])
        self.assertEqual(result["known_cloud_credits"], "14")
        self.assertEqual(encode.core.sha(self.output / "batches/000000/attempt-000/receipt.json"), digest)
        third = FakeClient()
        self.assertEqual(self.run_collect(third), result)
        self.assertEqual(third.calls, [])
        self.assertEqual(third.catalog_calls, 0)

    def test_unknown_requires_explicit_retry_and_retains_prior_cost_uncertainty(self):
        first = FakeClient(fail=True)
        with self.assertRaises(TimeoutError):
            self.run_collect(first)
        intent = self.output / "batches/000000/attempt-000/intent.json"
        digest = encode.core.sha(intent)
        second = FakeClient()
        with self.assertRaisesRegex(ValueError, "retry-unknown"):
            self.run_collect(second)
        self.assertEqual(second.calls, [])
        result = self.run_collect(second, retry_unknown=True)
        self.assertEqual(result["complete_rows"], 3)
        self.assertEqual(result["physical_attempts"], 3)
        self.assertEqual(result["unknown_charge_attempts"], ["batches/000000/attempt-000"])
        self.assertEqual(result["known_cloud_credits"], "14")
        self.assertEqual(encode.core.sha(intent), digest)
        retry = encode.core.read(self.output / "batches/000000/attempt-001/intent.json")
        self.assertEqual(retry["retry_of_unknown_attempts"], ["batches/000000/attempt-000"])

    def test_intent_without_receipt_is_not_reissued_implicitly(self):
        self.output.mkdir()
        encode.write_json(self.output / "encoding-plan.json", self.plan)
        encode.write_json(
            self.output / "batches/000000/attempt-000/intent.json", {"request": encode.request_for(self.rows[:2])}
        )
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "retry-unknown"):
            self.run_collect(client)
        self.assertEqual(client.calls, [])
        self.assertEqual(
            encode.core.read(self.output / "collection.json")["unknown_charge_attempts"], ["batches/000000/attempt-000"]
        )

    def test_completed_vector_tampering_and_wrong_catalog_stop_before_dispatch(self):
        self.run_collect(FakeClient(), max_batches=1)
        (self.output / "batches/000000/attempt-000/vectors.npy").write_bytes(b"changed")
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "vector bytes changed"):
            self.run_collect(client)
        self.assertEqual(client.calls, [])
        with tempfile.TemporaryDirectory() as temporary:
            client.get_model = lambda _: {"revision": "wrong", "dims": {"dense": 2560}}
            with self.assertRaisesRegex(ValueError, "serving encoder"):
                encode.collect_rows(self.rows, Path(temporary) / "new", self.plan, client)
        self.assertEqual(client.calls, [])

    def test_self_hosted_cost_is_unpriced(self):
        self.plan["self_hosted"] = True
        result = self.run_collect(FakeClient(self_hosted=True))
        self.assertIsNone(result["known_cloud_credits"])
        self.assertIsNone(result["known_cloud_usd"])
        receipt = encode.core.read(self.output / "batches/000000/attempt-000/receipt.json")
        self.assertIsNone(receipt["cloud_credits"])
        self.assertIsNone(receipt["request_id"])

    def test_bad_identity_billing_and_vector_data_are_rejected(self):
        rows = self.rows[:2]
        reply = FakeClient().encode(**encode.request_for(rows))
        for change, message in (
            (lambda data: data[0].update(model="other"), "encoder model"),
            (lambda data: data[0].update(id="unknown"), "embedding IDs"),
            (lambda data: data[0]["request"].update(id="different"), "request IDs"),
            (lambda data: data[0]["request"].update(credits_debited=8), "batch charge"),
        ):
            with self.subTest(message=message):
                changed = copy.deepcopy(reply)
                change(changed)
                with self.assertRaisesRegex(ValueError, message):
                    encode.batch_metadata(changed, rows, "a" * 64, self_hosted=False)
        for vector in ([0.0] * 2560, [float("nan")] * 2560, [1.0]):
            changed = copy.deepcopy(reply)
            changed[0]["dense"] = vector
            with self.assertRaises(ValueError):
                encode.normalized_vectors(changed, rows)

    def test_refuses_existing_bundled_directory_and_changed_resume_plan(self):
        self.output.mkdir()
        (self.output / "features.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "new output directory"):
            self.run_collect(FakeClient())
        (self.output / "features.json").unlink()
        self.run_collect(FakeClient(), max_batches=1)
        self.plan["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "same source"):
            self.run_collect(FakeClient())

    def test_sdk_retry_hook_stops_repeated_dispatch(self):
        with self.assertRaisesRegex(RuntimeError, "not automatically repeated"):
            encode.SingleSendClient._record_retry(None)


if __name__ == "__main__":
    unittest.main()
