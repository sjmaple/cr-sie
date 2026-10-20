"""Check that training collection never repeats a completed paid call."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benefit_collect as collector
import solver
from data import digest, write


class MetadataClient:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_model(self, model):
        return {"name": model, "revision": next(a["revision"] for a in solver.ARMS.values() if a["model"] == model)}


class BenefitCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.rows = [{"id": "row0", "request": "Compute 1+1."}, {"id": "row1", "request": "Compute 2+2."}]
        self.settings = copy.deepcopy(solver.SETTINGS)
        write(self.directory / "plan.json", {"source": "synthetic"})
        self.sent = []
        checked = patch.object(collector, "checked_training", return_value=({}, {"train": self.rows}, self.settings))
        metadata = patch.object(collector, "SingleSendClient", MetadataClient)
        stage = patch.object(collector, "stage_call", side_effect=self.success)
        for item in (checked, metadata, stage):
            item.start()
            self.addCleanup(item.stop)

    def success(self, row, arm, settings, phase, messages, self_hosted):
        self.sent.append((row["id"], arm, phase, messages))
        answer = '{"answer":"4"}' if phase == "final" else ("\\boxed{2}" if row["id"] == "row0" else "Some work")
        call = {
            "messages_sha256": digest(messages),
            "answer": answer,
            "finish_reason": "stop",
            "response_valid": True,
            "billing_known": True,
            "billing": "self_hosted" if self_hosted else "cloud",
            "receipt": {"credits": None if self_hosted else 5},
            "elapsed_seconds": 0.1,
            "execution_revision": "fixture",
            "tokens": {"completion_tokens": 10},
        }
        return {
            "id": row["id"],
            "arm": arm,
            "phase": phase,
            "request_sha256": digest(row["request"]),
            "settings_sha256": digest(settings),
            "messages_sha256": digest(messages),
            "status": "complete",
            "call": call,
            "observed_credits": 0 if self_hosted else 5,
            "execution_unknown": False,
            "safe_to_resume": False,
            "http_failures": [],
        }

    def collect(self, arm="cheap", phase="initial", **kwargs):
        return collector.collect(self.directory, arm, phase, workers=1, max_seconds=60, self_hosted=True, **kwargs)

    def test_pilot_resume_and_finalization_preserve_initial_calls(self):
        self.collect(limit=1)
        self.collect(resume=True)
        self.collect(phase="final", resume=True)
        self.collect(phase="final", resume=True)
        self.assertEqual(
            [(i, a, p) for i, a, p, _ in self.sent],
            [
                ("row0", "cheap", "initial"),
                ("row1", "cheap", "initial"),
                ("row1", "cheap", "final"),
            ],
        )
        record = solver.read(self.directory / "benefit-arms/cheap/records/row1-cheap.json")
        self.assertEqual(record["answer"], "\\boxed{4}")
        self.assertEqual(len(record["calls"]), 2)
        self.assertEqual(
            self.sent[-1][-1], collector.messages_for(self.rows[1], "cheap", self.settings, record["calls"][0])
        )

    def test_interrupted_intent_requires_explicit_bounded_unknown_retry(self):
        self.collect(limit=1)
        output = self.directory / "benefit-arms/cheap"
        write(output / "attempts/row1/initial-0001.intent.json", {"id": "row1"})
        with self.assertRaisesRegex(ValueError, "Attempt ceiling"):
            self.collect(resume=True)
        with self.assertRaisesRegex(ValueError, "execution is unknown"):
            self.collect(resume=True, max_attempts=2)
        self.collect(resume=True, max_attempts=2, retry_unknown=True)
        self.assertTrue((output / "attempts/row1/initial-0001.intent.json").exists())
        self.assertTrue((output / "attempts/row1/initial-0002.json").exists())
        self.assertEqual(len(self.sent), 2)

    def test_failure_stops_new_calls_and_retains_receipt(self):
        def failure(*args):
            result = self.success(*args)
            result.update(status="incomplete", execution_unknown=True, error={"type": "RuntimeError"})
            return result

        with (
            patch.object(collector, "stage_call", side_effect=failure),
            self.assertRaisesRegex(RuntimeError, "Collection phase incomplete"),
        ):
            self.collect()
        self.assertEqual(len(self.sent), 1)
        receipt = self.directory / "benefit-arms/cheap/attempts/row0/initial-0001.json"
        self.assertEqual(solver.read(receipt)["call"]["answer"], "\\boxed{2}")
        with self.assertRaisesRegex(ValueError, "Attempt ceiling"):
            self.collect(resume=True)
        self.assertEqual(len(self.sent), 1)

    def test_gather_requires_both_arms_and_retains_all_attempts(self):
        for arm in solver.ARMS:
            self.collect(arm=arm)
            self.collect(arm=arm, phase="final", resume=True)
        summary = collector.gather(self.directory)
        self.assertEqual(summary["completed"], 4)
        self.assertEqual(summary["status"], "complete")
        self.assertFalse(summary["acquisition_cost_complete"])
        self.assertEqual(len(summary["receipt_sha256"]), 12)
        collector.gather(self.directory)
        self.assertEqual(len(self.sent), 6)
        plan = solver.read(self.directory / "answers-train/collection-plan.json")
        solver.collection_binding(plan, self.directory, "train", self.rows, self.settings, None, True)

    def test_changed_endpoint_cannot_resume_existing_arm(self):
        self.collect(limit=1)
        with (
            patch.dict("os.environ", {"SIE_BASE_URL": "http://different.example"}),
            self.assertRaisesRegex(ValueError, "identical inputs, endpoint and runtime"),
        ):
            self.collect(resume=True)
        self.assertEqual(len(self.sent), 1)

    def test_truncated_box_still_requires_finalizer(self):
        self.assertTrue(collector.needs_final({"answer": "\\boxed{2}", "finish_reason": "length"}))
        self.assertFalse(collector.needs_final({"answer": "\\boxed{2}", "finish_reason": "stop"}))


if __name__ == "__main__":
    unittest.main()
