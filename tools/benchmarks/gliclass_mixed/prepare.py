"""Freeze either the historical profile contrast or the whole-source progress fix.

The historical profile candidate is derived. The progress comparison uses two
exact commits and keeps every model profile and dependency byte-identical.
"""

# Fixed argv-only Git reads, never a shell command or a moving ref after resolution.
# ruff: noqa: S603, S607

from __future__ import annotations

import argparse
import ast
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
PROGRESS_BASELINE = "f3b34e2dd88df74d4cc62c1446d28982c769f6ea"
PROGRESS_CANDIDATE = "45794f69402d1c334c1238cfc13082cc0596e08f"
SOURCE_PATHS = {
    "batcher": "packages/sie_server/src/sie_server/core/batcher.py",
    "worker": "packages/sie_server/src/sie_server/core/worker/model_worker.py",
    "gliclass": "packages/sie_server/src/sie_server/adapters/gliclass/__init__.py",
    "sdk": "packages/sie_sdk/src/sie_sdk/__init__.py",
    "config": "packages/sie_config/src/sie_config/__init__.py",
}
PROGRESS_PATHS = sorted(SOURCE_PATHS[name] for name in ("batcher", "worker"))
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


def executable_ast(source: bytes) -> str:
    """Ignore comments/docstrings only when checking the worker documentation delta."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def prepare_progress(
    root: Path, output: Path, baseline: str = PROGRESS_BASELINE, candidate: str = PROGRESS_CANDIDATE
) -> dict:
    commits = {
        arm: git(root, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()
        for arm, ref in (("baseline", baseline), ("candidate", candidate))
    }
    base, fixed = commits["baseline"], commits["candidate"]
    if git(root, "show", f"{base}:uv.lock") != (root / "uv.lock").read_bytes():
        raise ValueError("execution workspace dependency lock differs from baseline")
    paths = [path for path in PATHS if git(root, "ls-tree", "--name-only", base, path).strip()]
    differences = sorted(git(root, "diff", "--name-only", base, fixed, "--", *PATHS).decode().splitlines())
    if differences != PROGRESS_PATHS:
        raise ValueError(f"unexpected progress source delta: {differences}")
    if executable_ast(git(root, "show", f"{base}:{SOURCE_PATHS['worker']}")) != executable_ast(
        git(root, "show", f"{fixed}:{SOURCE_PATHS['worker']}")
    ):
        raise ValueError("worker executable code differs")
    profile = yaml.safe_load(git(root, "show", f"{base}:{PROFILE}"))
    if profile["profiles"]["default"].get("adaptive_batching", {}).get("enabled", True) is not True:
        raise ValueError("progress comparison requires the unchanged adaptive-enabled profile")
    output.mkdir(parents=True, exist_ok=False)
    for arm, commit in commits.items():
        (output / f"{arm}.tar").write_bytes(git(root, "archive", "--format=tar", commit, *paths))
    manifest = {
        "schema": 2,
        "comparison": "batcher_progress",
        "baseline_commit": base,
        "candidate_commit": fixed,
        "candidate_is_derived_archive": False,
        "changed_paths": differences,
        "worker_executable_ast_unchanged": True,
        "profiles": {arm: fingerprint(profile) for arm in commits},
        "archives": {arm: {"file": f"{arm}.tar", "sha256": digest(output / f"{arm}.tar")} for arm in commits},
        "dependency_lock_sha256": digest(root / "uv.lock"),
        "paths": paths,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", choices=("profile", "progress"), default="profile")
    args = parser.parse_args()
    freeze = prepare_progress if args.comparison == "progress" else prepare
    print(json.dumps(freeze(args.repo, args.output), indent=2))


if __name__ == "__main__":
    main()
