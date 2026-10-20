"""Catalog checks stop fresh collection before generation on drift or failure."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import benefit_acquire as acquire
from data import write


class CatalogPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        write(self.directory / "plan.json", {"synthetic": True})
        self.settings = {"arms": {"cheap": {"model": "public/model", "revision": "a" * 40}}}
        self.client = MagicMock()
        self.client.__enter__.return_value = self.client
        self.enterContext(patch.object(acquire, "checked_training", return_value=({}, {}, self.settings)))
        self.enterContext(patch.object(acquire, "SingleSendClient", return_value=self.client))
        self.collector = self.enterContext(
            patch.object(acquire.benefit_collect, "collect", return_value={"status": "complete"})
        )

    def receipt(self):
        return json.loads(next((self.directory / "catalog-preflights/cheap").glob("*.json")).read_text())

    def test_matching_catalog_delegates_exact_collection_options(self):
        self.client.get_model.return_value = {
            "name": "public/model",
            "revision": "a" * 40,
            "private_extra": "not archived",
        }
        result = acquire.collect(self.directory, "cheap", "initial", workers=2, max_seconds=600, max_credits=5000)
        self.assertEqual(result, {"status": "complete"})
        self.collector.assert_called_once_with(
            self.directory, "cheap", "initial", workers=2, max_seconds=600, max_credits=5000
        )
        self.assertEqual(self.receipt()["status"], "matched")
        self.assertNotIn("private_extra", self.receipt())

    def test_changed_revision_is_recorded_without_generation(self):
        self.client.get_model.return_value = {"name": "public/model", "revision": "b" * 40}
        with self.assertRaisesRegex(ValueError, "catalog differs"):
            acquire.collect(self.directory, "cheap", "initial")
        self.assertEqual(self.receipt()["status"], "mismatch")
        self.collector.assert_not_called()

    def test_changed_name_is_rejected_even_when_revision_matches(self):
        self.client.get_model.return_value = {"name": "other/model", "revision": "a" * 40}
        with self.assertRaises(ValueError):
            acquire.collect(self.directory, "cheap", "final")
        self.collector.assert_not_called()

    def test_metadata_failure_has_no_retry_or_exception_text_in_receipt(self):
        self.client.get_model.side_effect = RuntimeError("synthetic-secret-must-not-be-published")
        with self.assertRaises(RuntimeError):
            acquire.collect(self.directory, "cheap", "initial")
        self.client.get_model.assert_called_once()
        self.collector.assert_not_called()
        self.assertEqual(self.receipt()["status"], "failed")
        self.assertNotIn("synthetic-secret", json.dumps(self.receipt()))

    def test_final_phase_checks_again_and_preserves_prior_receipts(self):
        self.client.get_model.return_value = {"id": "public/model", "revision": "a" * 40}
        acquire.collect(self.directory, "cheap", "initial")
        acquire.collect(self.directory, "cheap", "final", resume=True, self_hosted=True)
        self.assertEqual(self.client.get_model.call_count, 2)
        self.assertEqual(len(list((self.directory / "catalog-preflights/cheap").glob("*.json"))), 2)
        self.collector.assert_called_with(self.directory, "cheap", "final", resume=True, self_hosted=True)


if __name__ == "__main__":
    unittest.main()
