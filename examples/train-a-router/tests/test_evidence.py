"""Check experiment isolation and verified downloads without network access."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evidence
import run


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
    def test_followup_fetch_keeps_its_pin_separate_from_original_verification(self):
        artifacts = {
            "report.json": ("core", b"{}\n"),
            "expanded/seed-89/lora/model.safetensors": ("seed89-lora", b"native weights"),
            "other.bin": ("seed89-full", b"not requested"),
        }
        body, digest = manifest(artifacts)
        base = f"{evidence.BASE_URL}followup-revision/{evidence.LORA_FOLDER}/"
        requested = []

        def open_url(url, **kwargs):
            requested.append(url)
            self.assertTrue(url.startswith(base))
            name = url.removeprefix(base)
            return io.BytesIO(body if name == "manifest.json" else artifacts[name][1])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                patch.object(evidence, "LORA_REVISION", "followup-revision"),
                patch.object(evidence, "LORA_MANIFEST_SHA256", digest),
                patch.object(evidence.urllib.request, "urlopen", side_effect=open_url),
            ):
                evidence.fetch(directory, "lora", lora_study=True)
            self.assertEqual((directory / "expanded/seed-89/lora/model.safetensors").read_bytes(), b"native weights")
            self.assertFalse((directory / "other.bin").exists())
            self.assertEqual(len(requested), 3)
            evidence.verify(directory, manifest_sha256=digest)
            with self.assertRaisesRegex(ValueError, "pinned public experiment"):
                evidence.verify(directory)

    def test_original_fetch_retains_original_pin_and_both_weights(self):
        artifacts = {
            "report.json": ("core", b"{}"),
            "full.bin": ("seed89-full", b"full"),
            "lora.bin": ("seed89-lora", b"lora"),
        }
        body, digest = manifest(artifacts)
        base = f"{evidence.BASE_URL}{evidence.REVISION}/{evidence.FOLDER}/"

        def open_url(url, **kwargs):
            self.assertTrue(url.startswith(base))
            name = url.removeprefix(base)
            return io.BytesIO(body if name == "manifest.json" else artifacts[name][1])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                patch.object(evidence, "MANIFEST_SHA256", digest),
                patch.object(evidence.urllib.request, "urlopen", side_effect=open_url),
            ):
                evidence.fetch(directory, "both")
                evidence.verify(directory)
            self.assertEqual((directory / "full.bin").read_bytes(), b"full")
            self.assertEqual((directory / "lora.bin").read_bytes(), b"lora")

    def test_different_manifest_is_rejected_before_network_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "manifest.json").write_bytes(b"another experiment")
            with (
                patch.object(evidence, "LORA_REVISION", "followup-revision"),
                patch.object(evidence, "LORA_MANIFEST_SHA256", "0" * 64),
                patch.object(evidence.urllib.request, "urlopen") as download,
            ):
                with self.assertRaisesRegex(ValueError, "different experiment"):
                    evidence.fetch(directory, lora_study=True)
                download.assert_not_called()
            self.assertEqual((directory / "manifest.json").read_bytes(), b"another experiment")

    def test_followup_rejects_full_weights_before_network_access(self):
        with patch.object(evidence.urllib.request, "urlopen") as download:
            for weights in ("full", "both"):
                with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, "only --weights lora"):
                    evidence.fetch(Path("unused"), weights, lora_study=True)
            download.assert_not_called()

    def test_manifest_allows_float32_checkpoint_but_keeps_size_bound(self):
        for size in (1_686_000_000, 2_000_000_001):
            body = json.dumps({"files": {"weights": {"bytes": size}}}).encode()
            digest = hashlib.sha256(body).hexdigest()
            if size <= 2_000_000_000:
                evidence.entries(body, manifest_sha256=digest)
            else:
                with self.assertRaisesRegex(ValueError, "Invalid artifact size"):
                    evidence.entries(body, manifest_sha256=digest)

    def test_cli_destinations_remain_separate_and_allow_explicit_output(self):
        cases = (
            (["fetch"], Path("evidence"), False),
            (["fetch", "--lora-study"], Path("run-output/lora-study"), True),
            (["fetch", "--lora-study", "--evidence", "custom-followup"], Path("custom-followup"), True),
        )
        for arguments, directory, followup in cases:
            with self.subTest(arguments=arguments), patch("sys.argv", ["run.py", *arguments]):
                with patch.object(evidence, "fetch") as fetch:
                    run.main()
                fetch.assert_called_once_with(directory, None, lora_study=followup)


if __name__ == "__main__":
    unittest.main()
