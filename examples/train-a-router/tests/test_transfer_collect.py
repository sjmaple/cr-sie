"""Transfer acquisition is separate, complete, and never repeats unknown calls."""

import unittest
from unittest.mock import patch

import test_benefit_expanded_collect as fixtures

import transfer_collect as transfer
from data import write


class TransferCollectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExpandedCollectionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.root / "transfer"
        write(self.directory / "inputs.json", self.fixture.rows[1:])
        write(self.directory / "settings.json", self.fixture.settings)
        write(self.directory / "source-plan.json", {"separate_transfer_source": True})
        self.plan = transfer.prepare(self.directory)
        mocked = patch.object(transfer, "SingleSendClient", fixtures.MetadataClient)
        mocked.start()
        self.addCleanup(mocked.stop)

    def acquire(self, job, phase="initial", **kwargs):
        return transfer.collect(self.directory, job, phase, workers=1, max_seconds=30, self_hosted=True, **kwargs)

    def test_full_population_disjoint_shards_and_resume_without_duplicates(self):
        for job in self.plan["jobs"]:
            self.acquire(job)
            self.acquire(job, "final", resume=True)
            self.acquire(job, "final", resume=True)
        calls = self.fixture.sent
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(sum(phase == "initial" for _, _, phase in calls), 6)
        summary = transfer.gather(
            self.directory, {name: self.directory / "collection-jobs" / name for name in self.plan["jobs"]}
        )
        self.assertEqual(summary["completed"], 6)
        self.assertEqual(summary["unknown_execution_attempts"], 0)
        self.assertFalse(summary["graded"])

    def test_dangling_intent_blocks_resume(self):
        self.acquire("cheap-0", limit=1)
        path = self.directory / "collection-jobs/cheap-0/attempts/new2/initial-0001.intent.json"
        write(path, {"id": "new2"})
        with self.assertRaisesRegex(ValueError, "cannot be repeated"):
            self.acquire("cheap-0", resume=True)
        self.assertEqual(len(self.fixture.sent), 1)

    def test_explicit_safe_rejection_resume_is_bounded_and_preserved(self):
        def rejected(row, arm, settings, phase, messages, self_hosted):
            result = self.fixture.success(row, arm, settings, phase, messages, self_hosted)
            result.pop("call")
            result.update(status="incomplete", safe_to_resume=True, execution_unknown=False)
            return result

        with patch.object(transfer.stages, "stage_call", side_effect=rejected):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.acquire("cheap-0", limit=1)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.acquire("cheap-0", limit=1, resume=True)
            with self.assertRaisesRegex(ValueError, "Two-attempt ceiling"):
                self.acquire("cheap-0", limit=1, resume=True)
        attempts = self.directory / "collection-jobs/cheap-0/attempts/new0"
        self.assertTrue((attempts / "initial-0001.json").exists())
        self.assertTrue((attempts / "initial-0002.json").exists())
        self.assertEqual(len(self.fixture.sent), 2)

    def test_changed_source_blocks_requests(self):
        (self.directory / "inputs.json").write_text("[]\n")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.acquire("cheap-0")
        self.assertEqual(self.fixture.sent, [])


if __name__ == "__main__":
    unittest.main()
