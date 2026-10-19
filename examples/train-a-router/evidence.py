"""Download and verify the immutable public experiment; weights are opt-in."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from data import file_hash

REVISION = "4136e68feaa90b5f50dce9f30df212982d393356"
MANIFEST_SHA256 = "6fa7b8e94d5894177cf702f00df9daacabcdc8bbab237eac3f0a15eb28c3e709"
FOLDER = "evaluations/2026-10-18/train-a-router-grouped-v2"
LORA_REVISION = "d884d137386ccc4c89eb6d78449197a4e4be9892"
LORA_MANIFEST_SHA256 = "f9f2f521de0dc6bc1711e7695378702efc03a55f017d29b7e52dc6552ae5872f"
LORA_FOLDER = "evaluations/2026-10-18/train-a-router-lora-followup"
BASE_URL = "https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/"


def entries(body, *, manifest_sha256=None):
    if hashlib.sha256(body).hexdigest() != (manifest_sha256 or MANIFEST_SHA256):
        raise ValueError("Manifest differs from the pinned public experiment")
    manifest = json.loads(body)
    for name, item in manifest["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
            raise ValueError("Unsafe artifact path")
        if not isinstance(item["bytes"], int) or not 0 <= item["bytes"] <= 2_000_000_000:
            raise ValueError("Invalid artifact size")
    return manifest["files"]


def verify_file(path, expected):
    if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
        raise ValueError(f"Artifact hash/size differs: {path.name}")


def verify(directory, *, manifest_sha256=None):
    files = entries((directory / "manifest.json").read_bytes(), manifest_sha256=manifest_sha256)
    for name, expected in files.items():
        if expected["group"] == "core":
            verify_file(directory / name, expected)
    return files


def fetch(directory, weights=None, *, lora_study=False):
    if lora_study and weights not in (None, "lora"):
        raise ValueError("The LoRA follow-up supports only --weights lora")
    revision = LORA_REVISION if lora_study else REVISION
    folder = LORA_FOLDER if lora_study else FOLDER
    manifest_sha256 = LORA_MANIFEST_SHA256 if lora_study else MANIFEST_SHA256
    if "PENDING_PUBLICATION" in (revision, manifest_sha256):
        raise RuntimeError("Public evidence has not been pinned yet")
    destination = directory / "manifest.json"
    if destination.exists() and file_hash(destination) != manifest_sha256:
        raise ValueError("Destination contains a different experiment")
    base = f"{BASE_URL}{revision}/{folder}/"
    with urllib.request.urlopen(base + "manifest.json", timeout=60) as response:
        body = response.read(200_001)
    files = entries(body, manifest_sha256=manifest_sha256)
    directory.mkdir(parents=True, exist_ok=True)
    groups = {"core"}
    if weights:
        groups.update(f"seed89-{v}" for v in (("full", "lora") if weights == "both" else (weights,)))
    for name, expected in files.items():
        if expected["group"] not in groups:
            continue
        path = directory / name
        if path.exists():
            verify_file(path, expected)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                with urllib.request.urlopen(base + name, timeout=300) as response:
                    size = 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        size += len(chunk)
                        if size > expected["bytes"]:
                            raise ValueError("Artifact exceeds its declared size")
                        handle.write(chunk)
                handle.flush()
                verify_file(temporary, expected)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Verified {name}", flush=True)
    if destination.exists() and destination.read_bytes() != body:
        raise ValueError("Destination contains a different experiment")
    destination.write_bytes(body)
    verify(directory, manifest_sha256=manifest_sha256)
