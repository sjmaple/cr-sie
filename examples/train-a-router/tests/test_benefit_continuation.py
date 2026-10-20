"""Replacement runtimes preserve completed calls and disjoint pending work."""

import unittest

from test_benefit_expanded_collect import ExpandedCollectionTests

import benefit_continuation as continuation
from data import file_hash, write


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        fixture = ExpandedCollectionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.directory = fixture.directory
        self.root = fixture.root / "continuation"
        self.prior = fixture.directory / "collection-jobs/cheap"
        from benefit_expanded_collect import prepare_jobs

        prepare_jobs(self.directory)
        runtime = fixture.root / "runtime-old.json"
        write(runtime, {"gpu": "old"})
        fixture.run_job("cheap", limit=2, runtime_receipt=runtime)
        self.runtime = fixture.root / "runtime-new.json"
        write(self.runtime, {"gpu": "new"})

    def test_only_missing_stages_sent_and_runtime_mapping_preserved(self):
        plan = continuation.prepare(self.directory, self.prior, self.root)
        self.assertEqual(plan["pending_stage_at_prepare"], {"new1": "final", "new2": "initial"})
        original_hashes = continuation.inventory(self.prior)
        for shard in plan["shards"]:
            for phase in ("initial", "final", "final"):
                continuation.collect(self.directory, self.prior, self.root, shard, phase, self.runtime)
        self.assertEqual(
            self.fixture.sent,
            [
                ("new0", "cheap", "initial"),
                ("new1", "cheap", "initial"),
                ("new1", "cheap", "final"),
                ("new2", "cheap", "initial"),
            ],
        )
        locations = {name: self.root / "shards" / name for name in plan["shards"]}
        merged = continuation.assemble(self.directory, self.prior, self.root, locations, self.root / "merged")
        mapping = merged["call_runtime_sha256"]
        self.assertEqual(mapping["attempts/new0/initial-0001.json"], file_hash(self.prior / "runtime-receipt.json"))
        self.assertEqual(mapping["attempts/new1/final-0001.json"], file_hash(self.runtime))
        self.assertEqual(continuation.inventory(self.prior), original_hashes)

    def test_dangling_original_intent_is_never_repeated(self):
        write(self.prior / "attempts/new2/initial-0001.intent.json", {"id": "new2"})
        with self.assertRaisesRegex(ValueError, "cannot be repeated"):
            continuation.prepare(self.directory, self.prior, self.root)
        self.assertEqual(len(self.fixture.sent), 2)

    def test_changed_original_receipt_blocks_continuation(self):
        plan = continuation.prepare(self.directory, self.prior, self.root)
        path = self.prior / "attempts/new0/initial-0001.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "original receipts changed"):
            continuation.collect(
                self.directory, self.prior, self.root, next(iter(plan["shards"])), "initial", self.runtime
            )

    def test_dangling_new_intent_blocks_resume(self):
        plan = continuation.prepare(self.directory, self.prior, self.root)
        shard = next(name for name, ids in plan["shards"].items() if "new2" in ids)
        path = self.root / "shards" / shard / "attempts/new2/initial-0001.intent.json"
        write(path, {"id": "new2"})
        with self.assertRaisesRegex(ValueError, "cannot be repeated"):
            continuation.collect(self.directory, self.prior, self.root, shard, "initial", self.runtime)
        self.assertEqual(len(self.fixture.sent), 2)


if __name__ == "__main__":
    unittest.main()
