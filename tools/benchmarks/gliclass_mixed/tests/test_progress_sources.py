"""Whole-source progress comparisons reject confounded or shadowed serving inputs."""

from __future__ import annotations

import copy
import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from tools.benchmarks.gliclass_mixed.evidence import digest
from tools.benchmarks.gliclass_mixed.prepare import (
    BASELINE,
    PROFILE,
    PROFILE_REVISION,
    PROGRESS_BASELINE,
    PROGRESS_CANDIDATE,
    PROGRESS_PATHS,
    SOURCE_PATHS,
    executable_ast,
    prepare_progress,
)
from tools.benchmarks.gliclass_mixed.run import archive_sources, validate_ready_sources

ROOT = Path(__file__).resolve().parents[4]


def replace_candidate(inputs: Path, path: str, transform) -> None:
    archive = inputs / "candidate.tar"
    prior = archive.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(prior)) as source, tarfile.open(archive, "w") as target:
        for member in source:
            if member.name == path:
                stream = source.extractfile(member)
                assert stream is not None
                content = transform(stream.read())
                member.size = len(content)
                target.addfile(member, io.BytesIO(content))
            else:
                target.addfile(member, source.extractfile(member) if member.isfile() else None)
    manifest = json.loads((inputs / "manifest.json").read_text())
    manifest["archives"]["candidate"]["sha256"] = digest(archive)
    (inputs / "manifest.json").write_text(json.dumps(manifest))


def test_whole_progress_sources_and_actual_import_hashes(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    manifest = prepare_progress(ROOT, inputs)
    assert manifest["baseline_commit"] == PROGRESS_BASELINE
    assert manifest["candidate_commit"] == PROGRESS_CANDIDATE
    assert manifest["candidate_is_derived_archive"] is False
    assert manifest["changed_paths"] == PROGRESS_PATHS
    assert manifest["profiles"]["baseline"] == manifest["profiles"]["candidate"]
    sources = archive_sources(inputs, tmp_path / "output")
    assert (sources["baseline"] / PROFILE).read_bytes() == (sources["candidate"] / PROFILE).read_bytes()
    assert (sources["baseline"] / "uv.lock").read_bytes() == (sources["candidate"] / "uv.lock").read_bytes()
    for source in sources.values():
        profile = yaml.safe_load((source / PROFILE).read_text())
        ready = {
            "source": {name: {"path": path, "sha256": digest(source / path)} for name, path in SOURCE_PATHS.items()},
            "adaptive": {},
        }
        validate_ready_sources(ready, source, profile)
        for name in ("batcher", "worker"):
            wrong = copy.deepcopy(ready)
            wrong["source"][name]["sha256"] = "0" * 64
            with pytest.raises(AssertionError, match="actual imported source"):
                validate_ready_sources(wrong, source, profile)
        with pytest.raises(AssertionError, match="adaptive state"):
            validate_ready_sources({**ready, "adaptive": None}, source, profile)


@pytest.mark.parametrize("candidate", [PROGRESS_BASELINE, BASELINE, PROFILE_REVISION])
def test_progress_prepare_rejects_other_source_deltas(tmp_path: Path, candidate: str) -> None:
    with pytest.raises(ValueError, match="unexpected progress source delta"):
        prepare_progress(ROOT, tmp_path / "inputs", candidate=candidate)
    assert not (tmp_path / "inputs").exists()


@pytest.mark.parametrize("path", [PROFILE, "uv.lock", SOURCE_PATHS["worker"]])
def test_rehashed_archive_still_rejects_profile_lock_or_executable_worker_change(tmp_path: Path, path: str) -> None:
    inputs = tmp_path / "inputs"
    prepare_progress(ROOT, inputs)
    # This retains a matching archive hash: the independent content checks must
    # reject a confounded source even when transport integrity alone is valid.
    replace_candidate(inputs, path, lambda content: content + b"\nUNEXPECTED_BENCHMARK_CHANGE = True\n")
    expected = "worker executable" if path == SOURCE_PATHS["worker"] else "unexpected source delta"
    with pytest.raises(AssertionError, match=expected):
        archive_sources(inputs, tmp_path / "output")


def test_progress_manifest_cannot_claim_derived_candidate(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    prepare_progress(ROOT, inputs)
    path = inputs / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["candidate_is_derived_archive"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="whole-source progress manifest"):
        archive_sources(inputs, tmp_path / "output")


def test_worker_ast_check_ignores_only_comments_and_docstrings() -> None:
    before = b'"""module"""\ndef function():\n    """before"""\n    return 1\n'
    documented = b'"""new module"""\ndef function():\n    """after"""\n    # explanation\n    return 1\n'
    assert executable_ast(before) == executable_ast(documented)
    assert executable_ast(before) != executable_ast(documented.replace(b"return 1", b"return 2"))
