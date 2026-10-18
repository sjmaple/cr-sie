"""Freeze one source checkout and the exact PR656 profile-only candidate.

The candidate is a derived archive, not a claimed historical Git checkout.
Everything except the one catalog file is byte-identical to the baseline.
"""

# Fixed argv-only Git reads, never a shell command or a moving ref after resolution.
# ruff: noqa: S603, S607

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

import yaml

from tools.benchmarks.gliclass_mixed.evidence import atomic_json, digest
from tools.benchmarks.gliclass_mixed.workload import fingerprint

BASELINE = "2281ba860f3c530d36e6f19d638b624bededef26"
PROFILE_REVISION = "a5e8d7b39be4b8156932d902d75fa46dc3e37e53"
PROFILE = "packages/sie_server/models/knowledgator__gliclass-large-v1.0.yaml"
PATHS = (
    "packages/sie_server/src",
    "packages/sie_server/models",
    "packages/sie_server/bundles",
    "packages/sie_sdk/src",
    "packages/sie_config/src",
    "packages/sie_bundles",
    "bundles",
    "uv.lock",
    "pyproject.toml",
)


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def prepare(root: Path, output: Path, baseline: str = BASELINE, profile_revision: str = PROFILE_REVISION) -> dict:
    base = git(root, "rev-parse", "--verify", "--end-of-options", f"{baseline}^{{commit}}").decode().strip()
    profile_ref = (
        git(root, "rev-parse", "--verify", "--end-of-options", f"{profile_revision}^{{commit}}").decode().strip()
    )
    before = git(root, "show", f"{base}:{PROFILE}")
    after = git(root, "show", f"{profile_ref}:{PROFILE}")
    parsed_before, parsed_after = yaml.safe_load(before), yaml.safe_load(after)
    expected = json.loads(json.dumps(parsed_before))
    if "adaptive_batching" in expected["profiles"]["default"]:
        raise ValueError("baseline already overrides adaptive batching")
    expected["profiles"]["default"]["adaptive_batching"] = {"enabled": False}
    if parsed_after != expected:
        raise ValueError("candidate differs beyond the one profile's adaptive_batching.enabled=false")
    lock = git(root, "show", f"{base}:uv.lock")
    if lock != (root / "uv.lock").read_bytes():
        raise ValueError("execution workspace dependency lock differs from baseline")
    existing = [path for path in PATHS if git(root, "ls-tree", "--name-only", base, path).strip()]
    archive = git(root, "archive", "--format=tar", base, *existing)
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline.tar").write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source, tarfile.open(output / "candidate.tar", "w") as target:
        for member in source:
            if member.name == PROFILE:
                member.size = len(after)
                target.addfile(member, io.BytesIO(after))
            else:
                target.addfile(member, source.extractfile(member) if member.isfile() else None)
    manifest = {
        "schema": 1,
        "baseline_commit": base,
        "candidate_base_commit": base,
        "candidate_profile_commit": profile_ref,
        "candidate_profile_path": PROFILE,
        "candidate_is_derived_archive": True,
        "only_changed_path": PROFILE,
        "profiles": {"baseline": fingerprint(parsed_before), "candidate": fingerprint(parsed_after)},
        "archives": {
            arm: {"file": f"{arm}.tar", "sha256": digest(output / f"{arm}.tar")} for arm in ("baseline", "candidate")
        },
        "dependency_lock_sha256": digest(root / "uv.lock"),
        "paths": existing,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.repo, args.output), indent=2))


if __name__ == "__main__":
    main()
