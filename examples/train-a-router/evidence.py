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
SOLVER_REVISION = "7810c9631d8fca236ea8a074bab86b25f2b3fa0e"
SOLVER_MANIFEST_SHA256 = "9df32300204ac4f24ffcd032c824619ddf9d0db100179d8d947397d191db6a59"
SOLVER_FOLDER = "evaluations/2026-10-19/train-a-router-answer-quality"
ARITHMETIC_REVISION = "71e0d5109e46f8b9c698806e7d3e47b2aa0da87d"
ARITHMETIC_MANIFEST_SHA256 = "45e0d829c043cb9aaec2ab77859a93a34fda0a3e8765ca89ca81d6e0461d725d"
ARITHMETIC_FOLDER = "evaluations/2026-10-19/train-a-router-arithmetic"
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


def fetch(
    directory, weights=None, *, lora_study=False, solver_study=False, arithmetic_study=False,
    solver_records=False, solver_weights=False,
):
    if sum((lora_study, solver_study, arithmetic_study)) > 1:
        raise ValueError("Choose one study")
    answer_study = solver_study or arithmetic_study
    if answer_study and weights:
        raise ValueError("Choose one study; solver evidence contains no router weights")
    if solver_records and not answer_study:
        raise ValueError("--solver-records requires --solver-study")
    if solver_weights and not answer_study:
        raise ValueError("--solver-weights requires --solver-study")
    if lora_study and weights not in (None, "lora"):
        raise ValueError("The LoRA follow-up supports only --weights lora")
    revision = LORA_REVISION if lora_study else REVISION
    folder = LORA_FOLDER if lora_study else FOLDER
    manifest_sha256 = LORA_MANIFEST_SHA256 if lora_study else MANIFEST_SHA256
    if solver_study:
        revision, folder, manifest_sha256 = SOLVER_REVISION, SOLVER_FOLDER, SOLVER_MANIFEST_SHA256
    if arithmetic_study:
        revision, folder, manifest_sha256 = ARITHMETIC_REVISION, ARITHMETIC_FOLDER, ARITHMETIC_MANIFEST_SHA256
    if "PENDING_PUBLICATION" in (revision, manifest_sha256):
        raise RuntimeError("Public evidence has not been pinned yet")
    destination = directory / "manifest.json"
    if destination.exists() and file_hash(destination) != manifest_sha256:
        raise ValueError("Destination contains a different experiment")
    base = f"{BASE_URL}{revision}/{folder}/"
    with urllib.request.urlopen(base + "manifest.json", timeout=60) as response:
        body = response.read(2_000_001 if answer_study else 200_001)
    files = entries(body, manifest_sha256=manifest_sha256)
    directory.mkdir(parents=True, exist_ok=True)
    groups = {"core"}
    if solver_records:
        groups.add("answers")
    if solver_weights:
        groups.add("native-weights")
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
