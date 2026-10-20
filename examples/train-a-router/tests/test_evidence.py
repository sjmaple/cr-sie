"""Verify the arithmetic evidence download without network access."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evidence


def manifest(artifacts):
    body = json.dumps(
        {
            "files": {
                name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "group": group}
                for name, (group, data) in artifacts.items()
            }
        }
    ).encode()
    return body, hashlib.sha256(body).hexdigest()


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.artifacts = {
            "arithmetic/study/report.json": ("core", b"{}\n"),
            "arithmetic/embeddings/train/vectors.jsonl.gz": ("core", b"cached vectors"),
            "arithmetic/study/native-2000/lora/model.safetensors": ("native-weights", b"native weights"),
            "arithmetic/records/fresh.tar.gz": ("answers", b"raw records"),
        }
        self.body, self.digest = manifest(self.artifacts)
        self.base = f"{evidence.BASE_URL}arithmetic-revision/{evidence.ARITHMETIC_FOLDER}/"
        self.requested = []
        for patcher in (
            patch.object(evidence, "ARITHMETIC_REVISION", "arithmetic-revision"),
            patch.object(evidence, "ARITHMETIC_MANIFEST_SHA256", self.digest),
            patch.object(evidence.urllib.request, "urlopen", side_effect=self.open_url),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def open_url(self, url, **kwargs):
        self.assertTrue(url.startswith(self.base))
        name = url.removeprefix(self.base)
        self.requested.append(name)
        return io.BytesIO(self.body if name == "manifest.json" else self.artifacts[name][1])

    def test_default_download_has_cached_data_but_no_weights_or_raw_answers(self):
        evidence.fetch(self.directory, arithmetic_study=True)
        core = {name for name, (group, _) in self.artifacts.items() if group == "core"}
        self.assertEqual(set(self.requested), {"manifest.json", *core})
        evidence.verify(self.directory, manifest_sha256=self.digest)
        self.assertFalse((self.directory / "arithmetic/study/native-2000/lora/model.safetensors").exists())
        self.assertFalse((self.directory / "arithmetic/records/fresh.tar.gz").exists())

    def test_weights_and_raw_records_are_independent_optional_downloads(self):
        evidence.fetch(self.directory, arithmetic_study=True, solver_weights=True)
        self.assertEqual((self.directory / "arithmetic/study/native-2000/lora/model.safetensors").read_bytes(), b"native weights")
        self.assertFalse((self.directory / "arithmetic/records/fresh.tar.gz").exists())
        self.requested.clear()
        evidence.fetch(self.directory, arithmetic_study=True, solver_records=True)
        self.assertEqual(self.requested, ["manifest.json", "arithmetic/records/fresh.tar.gz"])
        self.assertEqual((self.directory / "manifest.json").read_bytes(), self.body)

    def test_different_manifest_is_rejected_before_network_access(self):
        (self.directory / "manifest.json").write_bytes(b"another experiment")
        with self.assertRaisesRegex(ValueError, "different experiment"):
            evidence.fetch(self.directory, arithmetic_study=True)
        self.assertEqual(self.requested, [])
        self.assertEqual((self.directory / "manifest.json").read_bytes(), b"another experiment")

    def test_changed_cached_file_is_rejected_without_overwriting_it(self):
        evidence.fetch(self.directory, arithmetic_study=True)
        path = self.directory / "arithmetic/study/report.json"
        path.write_bytes(b"changed")
        self.requested.clear()
        with self.assertRaisesRegex(ValueError, "Artifact hash/size differs"):
            evidence.fetch(self.directory, arithmetic_study=True)
        self.assertEqual(self.requested, ["manifest.json"])
        self.assertEqual(path.read_bytes(), b"changed")

    def test_corrupt_remote_file_is_not_installed_and_temporary_file_is_removed(self):
        self.artifacts["arithmetic/study/report.json"] = ("core", b"[]\n")
        with self.assertRaisesRegex(ValueError, "Artifact hash/size differs"):
            evidence.fetch(self.directory, arithmetic_study=True)
        self.assertFalse((self.directory / "arithmetic/study/report.json").exists())
        self.assertFalse(list(self.directory.rglob(".download-*")))

    def test_oversized_remote_file_is_not_installed(self):
        self.artifacts["arithmetic/study/report.json"] = ("core", b"too much data")
        with self.assertRaisesRegex(ValueError, "exceeds its declared size"):
            evidence.fetch(self.directory, arithmetic_study=True)
        self.assertFalse((self.directory / "arithmetic/study/report.json").exists())
        self.assertFalse(list(self.directory.rglob(".download-*")))

    def test_unsafe_manifest_paths_are_rejected(self):
        for name in ("../outside", "/absolute", "folder\\outside", "folder/../outside"):
            body, digest = manifest({name: ("core", b"contents")})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
                evidence.entries(body, manifest_sha256=digest)

    def test_manifest_allows_float32_checkpoint_but_keeps_size_bound(self):
        for size in (1_686_000_000, 2_000_000_001):
            body = json.dumps({"files": {"weights": {"bytes": size}}}).encode()
            digest = hashlib.sha256(body).hexdigest()
            if size <= 2_000_000_000:
                evidence.entries(body, manifest_sha256=digest)
            else:
                with self.assertRaisesRegex(ValueError, "Invalid artifact size"):
                    evidence.entries(body, manifest_sha256=digest)


if __name__ == "__main__":
    unittest.main()
