"""Disjoint collection jobs preserve old answers and never retry submitted calls."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benefit_collect as stages
import benefit_expanded_collect as collector
import solver
from data import digest, file_hash, write


class MetadataClient:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_model(self, model):
        return {
            "name": model,
            "revision": next(item["revision"] for item in solver.ARMS.values() if item["model"] == model),
        }


class ExpandedCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory, self.prior = self.root / "expanded", self.root / "prior"
        self.rows = [
            {"id": identity, "request": f"Question {identity}"} for identity in ("old0", "new0", "new1", "new2")
        ]
        self.settings = copy.deepcopy(solver.SETTINGS)
        self.plan = {"new_train_ids": [row["id"] for row in self.rows[1:]], "reuse_sha256": "pending"}
        write(self.prior / "plan.json", {"prior": True})
        write(self.prior / "answers-train/collection-plan.json", {"historical": True})
        write(self.prior / "answers-train/collection.json", {"status": "complete", "completed": 4})
        hashes = {}
        for identity in ("old0", "old1"):
            row = {"id": identity, "request": f"Question {identity}"}
            for arm in solver.ARMS:
                message = stages.messages_for(row, arm, self.settings)
                attempt = self.success(row, arm, self.settings, "initial", message, True, record_send=False)
                record = solver.initial_record(row, arm, self.settings) | {
                    "status": "complete",
                    "calls": [attempt["call"]],
                    "answer": "\\boxed{2}",
                    "observed_credits": 0,
                    "credits": None,
                    "has_unknown_charges": False,
                }
                path = self.prior / "answers-train" / f"{identity}-{arm}.json"
                write(path, record)
                if identity == "old0":
                    hashes[path.name] = file_hash(path)
        reuse = {
            "ids": ["old0"],
            "records_sha256": hashes,
            "plan_sha256": file_hash(self.prior / "plan.json"),
            "collection_plan_sha256": file_hash(self.prior / "answers-train/collection-plan.json"),
            "collection_sha256": file_hash(self.prior / "answers-train/collection.json"),
        }
        write(self.directory / "reuse.json", reuse)
        self.plan["reuse_sha256"] = file_hash(self.directory / "reuse.json")
        write(self.directory / "plan.json", self.plan)
        self.sent = []
        for mocked in (
            patch.object(
                collector,
                "checked_training",
                side_effect=lambda directory: (self.plan, {"train": self.rows}, self.settings),
            ),
            patch.object(collector, "SingleSendClient", MetadataClient),
            patch.object(stages, "stage_call", side_effect=self.success),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)

    def success(self, row, arm, settings, phase, messages, self_hosted, record_send=True):
        if record_send:
            self.sent.append((row["id"], arm, phase))
        answer = '{"answer":"4"}' if phase == "final" else "Some work" if row["id"] == "new1" else "\\boxed{2}"
        call = {
            "messages_sha256": digest(messages),
            "answer": answer,
            "finish_reason": "stop",
            "response_valid": True,
            "billing_known": True,
            "billing": "self_hosted",
            "receipt": {"credits": None},
            "tokens": {"prompt_tokens": 10, "completion_tokens": 10},
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
            "observed_credits": 0,
            "execution_unknown": False,
            "safe_to_resume": False,
            "http_failures": [],
        }

    def run_job(self, name, phase="initial", **kwargs):
        return collector.collect(self.directory, name, phase, workers=1, max_seconds=60, self_hosted=True, **kwargs)

    def test_shards_and_resume_do_not_repeat_old_or_completed_calls(self):
        jobs = collector.prepare_jobs(self.directory)
        self.assertEqual(jobs["jobs"]["strong-0"]["ids"], ["new0", "new2"])
        self.assertEqual(jobs["jobs"]["strong-1"]["ids"], ["new1"])
        for name in jobs["jobs"]:
            self.run_job(name, limit=1)
            self.run_job(name, resume=True)
            self.run_job(name, "final", resume=True)
            self.run_job(name, "final", resume=True)
        self.assertEqual(len(self.sent), 8)
        self.assertEqual(len(set(self.sent)), 8)
        self.assertFalse(any(identity.startswith("old") for identity, _, _ in self.sent))
        summary = collector.gather(self.directory, self.prior)
        self.assertEqual(summary["completed"], 8)
        self.assertEqual(summary["reused_records"], 2)
        self.assertEqual(summary["new_completed_records"], 6)
        self.assertEqual(summary["unknown_execution_attempts"], 0)
        self.assertFalse((self.directory / "answers-train/old1-cheap.json").exists())
        for arm in solver.ARMS:
            self.assertEqual(
                file_hash(self.directory / f"answers-train/old0-{arm}.json"),
                file_hash(self.prior / f"answers-train/old0-{arm}.json"),
            )
        collector.gather(self.directory, self.prior)
        self.assertEqual(len(self.sent), 8)

    def test_duplicate_shard_rejected_before_any_call(self):
        plan = collector.prepare_jobs(self.directory)
        plan["jobs"]["strong-1"]["ids"] = ["new0", "new1"]
        (self.directory / "collection-jobs.json").unlink()
        write(self.directory / "collection-jobs.json", plan)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.run_job("strong-0")
        self.assertEqual(self.sent, [])

    def test_unknown_intent_blocks_resume_without_retry(self):
        collector.prepare_jobs(self.directory)
        self.run_job("cheap", limit=1)
        path = self.directory / "collection-jobs/cheap/attempts/new1/initial-0001.intent.json"
        write(path, {"id": "new1"})
        with self.assertRaisesRegex(ValueError, "no retries"):
            self.run_job("cheap", resume=True)
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(path.exists())

    def test_original_only_control_needs_no_calls(self):
        self.rows = self.rows[:1]
        self.plan["new_train_ids"] = []
        (self.directory / "plan.json").unlink()
        write(self.directory / "plan.json", self.plan)
        self.assertEqual(collector.prepare_jobs(self.directory)["jobs"], {})
        summary = collector.gather(self.directory, self.prior)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["new_completed_records"], 0)
        self.assertEqual(self.sent, [])

    def test_changed_reused_record_is_rejected(self):
        self.rows = self.rows[:1]
        self.plan["new_train_ids"] = []
        collector.prepare_jobs(self.directory)
        path = self.prior / "answers-train/old0-cheap.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Original completed answer changed"):
            collector.gather(self.directory, self.prior)


if __name__ == "__main__":
    unittest.main()
