"""Collect only new training pairs, with optional disjoint strong-model jobs.

Prepare one immutable job plan, run each job's initial and final phases through
the same SIE SDK, then gather with the original completed records. Each physical
call has an intent and receipt. Resume never repeats a submitted call.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import benefit_collect as stages
import solver
from benefit_expanded import checked_training
from data import digest, file_hash, write
from serving import SingleSendClient, error_receipt

FORMAT = "sie-expanded-benefit-jobs-v1"


def prepare_jobs(directory, strong_shards=2):
    directory = Path(directory)
    plan, _, settings = checked_training(directory)
    if type(strong_shards) is not int or not 1 <= strong_shards <= 2:
        raise ValueError("Use one or two disjoint strong-model jobs")
    ids = plan["new_train_ids"]
    if ids and strong_shards > len(ids):
        raise ValueError("Need at least one new row per job")
    jobs = {"cheap": {"arm": "cheap", "ids": ids}} if ids else {}
    if ids:
        jobs.update(
            {f"strong-{index}": {"arm": "strong", "ids": ids[index::strong_shards]} for index in range(strong_shards)}
        )
    result = {
        "format": FORMAT,
        "plan_sha256": file_hash(directory / "plan.json"),
        "settings_sha256": digest(settings),
        "reuse_sha256": plan["reuse_sha256"],
        "jobs": jobs,
        "scheduling": "Source-order round-robin shards; initial calls before conditional finalizers",
        "max_generation_attempts": 1,
        "collector_sha256": file_hash(Path(__file__)),
    }
    path = directory / "collection-jobs.json"
    if path.exists():
        raise FileExistsError(path)
    write(path, result)
    return result


def checked_jobs(directory):
    plan, inputs, settings = checked_training(directory)
    jobs = solver.read(directory / "collection-jobs.json")
    if (
        jobs.get("format") != FORMAT
        or jobs.get("plan_sha256") != file_hash(directory / "plan.json")
        or jobs.get("settings_sha256") != digest(settings)
        or jobs.get("reuse_sha256") != plan["reuse_sha256"]
        or jobs.get("collector_sha256") != file_hash(Path(__file__))
        or jobs.get("max_generation_attempts") != 1
    ):
        raise ValueError("Collection jobs differ from the frozen source and protocol")
    selected = jobs["jobs"]
    allowed = ({"cheap", "strong-0"}, {"cheap", "strong-0", "strong-1"}) if plan["new_train_ids"] else (set(),)
    if set(selected) not in allowed:
        raise ValueError("Unexpected collection job names")
    for arm in settings["arms"]:
        ids = [identity for job in selected.values() if job["arm"] == arm for identity in job["ids"]]
        if len(ids) != len(set(ids)) or set(ids) != set(plan["new_train_ids"]):
            raise ValueError("Each arm must cover each new request exactly once")
    by_id = {row["id"]: row for row in inputs["train"]}
    return jobs, {name: [by_id[identity] for identity in job["ids"]] for name, job in selected.items()}, settings


def collect(
    directory,
    job,
    phase,
    *,
    workers=4,
    limit=None,
    max_seconds=600,
    max_credits=None,
    self_hosted=False,
    resume=False,
    runtime_receipt=None,
):
    directory = Path(directory)
    jobs, populations, settings = checked_jobs(directory)
    if job not in populations or phase not in stages.PHASES:
        raise ValueError("Choose a prepared job and initial/final phase")
    if not 1 <= workers <= 16 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Use 1..16 workers and a positive finite time limit")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Pilot limit must be a positive integer")
    if not self_hosted and (max_credits is None or not solver.valid_credit(max_credits) or max_credits <= 0):
        raise ValueError("Cloud collection needs a positive observed-credit ceiling")
    arm, rows = jobs["jobs"][job]["arm"], populations[job]
    output = directory / "collection-jobs" / job
    binding = {
        "job": job,
        "arm": arm,
        "ids": [row["id"] for row in rows],
        "jobs_sha256": file_hash(directory / "collection-jobs.json"),
        "self_hosted": self_hosted,
        "endpoint_sha256": digest(os.environ.get("SIE_BASE_URL", "https://api.superlinked.com")),
        "runtime_receipt_sha256": file_hash(runtime_receipt) if runtime_receipt else None,
    }
    plan_path = output / "job-plan.json"
    if plan_path.exists():
        if not resume or solver.read(plan_path) != binding:
            raise ValueError("Resume needs an identical job, endpoint and runtime receipt")
    else:
        if resume:
            raise ValueError("No existing job to resume")
        write(plan_path, binding)
        if runtime_receipt:
            shutil.copyfile(runtime_receipt, output / "runtime-receipt.json")
    todo = []
    for row in rows[:limit] if limit else rows:
        first = None
        if phase == "final":
            _, _, initial = stages.history(
                output, row, arm, settings, "initial", stages.messages_for(row, arm, settings)
            )
            if initial is None:
                raise ValueError("Final phase requires a successful initial receipt for every selected request")
            first = initial[1]["call"]
            if not stages.needs_final(first):
                stages.materialize(output, row, arm, settings, self_hosted)
                continue
        messages = stages.messages_for(row, arm, settings, first)
        previous, interrupted, success = stages.history(output, row, arm, settings, phase, messages)
        if success:
            stages.materialize(output, row, arm, settings, self_hosted)
            continue
        if previous or interrupted:
            raise ValueError("A submitted call has no successful receipt; no retries are permitted")
        todo.append((row, messages))
    sessions = output / "sessions"
    session = sessions / f"session-{len(list(sessions.glob('session-*'))) + 1:04d}"
    write(
        session / "plan.json",
        {
            "job_plan_sha256": file_hash(plan_path),
            "phase": phase,
            "workers": workers,
            "pending_ids": [row["id"] for row, _ in todo],
            "max_seconds": max_seconds,
            "max_credits": max_credits,
            "limit": limit,
            "max_generation_attempts": 1,
            "budget_scope": "New observed charges; in-flight calls drain; external watchdog bounds infrastructure",
        },
    )
    started, sent, failed, observed, fatal = time.monotonic(), 0, 0, 0.0, None
    try:
        if todo:
            with SingleSendClient(timeout_s=60) as client:
                model = settings["arms"][arm]
                metadata = client.get_model(model["model"])
                if (
                    metadata.get("revision") != model["revision"]
                    or metadata.get("name", metadata.get("id")) != model["model"]
                ):
                    raise ValueError("Endpoint model identity or HF revision differs")
                write(session / "model.json", {key: metadata.get(key) for key in ("name", "revision", "profiles")})
        tasks = iter(todo)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}

            def schedule():
                nonlocal sent
                if (
                    failed
                    or time.monotonic() - started >= max_seconds
                    or (max_credits is not None and observed >= max_credits)
                ):
                    return
                task = next(tasks, None)
                if task is None:
                    return
                row, messages = task
                target = output / "attempts" / row["id"] / f"{phase}-0001.json"
                write(
                    target.with_name(target.stem + ".intent.json"),
                    {
                        "id": row["id"],
                        "arm": arm,
                        "phase": phase,
                        "job": job,
                        "session": session.name,
                        "request_sha256": digest(row["request"]),
                        "settings_sha256": digest(settings),
                        "messages_sha256": digest(messages),
                    },
                )
                future = pool.submit(stages.stage_call, row, arm, settings, phase, messages, self_hosted)
                pending[future] = row, messages, target
                sent += 1

            for _ in range(workers):
                schedule()
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    row, messages, target = pending.pop(future)
                    result = future.result()
                    stages.checked_stage(result, row, arm, settings, phase, messages)
                    write(target, result)
                    observed += result["observed_credits"]
                    failed += result["status"] != "complete"
                    stages.materialize(output, row, arm, settings, self_hosted)
                    print(f"{job}/{phase}: {sent - len(pending)}/{len(todo)} returned, {failed} failed", flush=True)
                for _ in done:
                    schedule()
    except Exception as error:  # noqa: BLE001 -- retain all intent and response evidence
        fatal = error_receipt(error)
    summary = {
        "status": "complete" if sent == len(todo) and not failed and fatal is None else "incomplete",
        "phase": phase,
        "sent": sent,
        "failed": failed,
        "observed_credits": observed,
        "elapsed_seconds": time.monotonic() - started,
        "fatal_error": fatal,
    }
    write(session / "collection.json", summary)
    if summary["status"] != "complete":
        raise RuntimeError("Collection phase incomplete; preserve receipts and inspect before continuing")
    return summary


def copy_exact(source, destination):
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ValueError("Archived evidence bytes differ")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def gather(directory, prior, job_directories=None):
    directory, prior = Path(directory), Path(prior)
    plan, inputs, settings = checked_training(directory)
    jobs, populations, _ = checked_jobs(directory)
    locations = (
        {name: directory / "collection-jobs" / name for name in jobs["jobs"]}
        if job_directories is None
        else {name: Path(path) for name, path in job_directories.items()}
    )
    if set(locations) != set(jobs["jobs"]):
        raise ValueError("Gather needs exactly the prepared jobs")
    reuse = solver.read(directory / "reuse.json")
    old_collection = prior / "answers-train"
    if any(
        file_hash(path) != expected
        for path, expected in (
            (prior / "plan.json", reuse["plan_sha256"]),
            (old_collection / "collection-plan.json", reuse["collection_plan_sha256"]),
            (old_collection / "collection.json", reuse["collection_sha256"]),
        )
    ):
        raise ValueError("Original collection binding changed")
    output = directory / "answers-train"
    reused = set(reuse["ids"])
    for row in inputs["train"]:
        if row["id"] in reused:
            for arm in settings["arms"]:
                name = f"{row['id']}-{arm}.json"
                if file_hash(old_collection / name) != reuse["records_sha256"][name]:
                    raise ValueError("Original completed answer changed")
                record = solver.read(old_collection / name)
                solver.validate_record(record, row, arm, settings)
                copy_exact(old_collection / name, output / name)
    failures, unknown, observed, missing = 0, 0, 0.0, 0
    receipts, bindings = {}, {}
    for name, location in locations.items():
        binding = solver.read(location / "job-plan.json")
        expected = jobs["jobs"][name]
        if (
            binding["job"] != name
            or binding["arm"] != expected["arm"]
            or binding["ids"] != expected["ids"]
            or binding["jobs_sha256"] != file_hash(directory / "collection-jobs.json")
            or not binding["self_hosted"]
        ):
            raise ValueError("Job binding differs from prepared IDs, arm or hosting")
        bindings[name] = binding
        if (
            binding["runtime_receipt_sha256"]
            and file_hash(location / "runtime-receipt.json") != binding["runtime_receipt_sha256"]
        ):
            raise ValueError("Runtime receipt changed")
        for row in populations[name]:
            record = stages.materialize(location, row, expected["arm"], settings, True)
            if record is None:
                missing += 1
            else:
                copy_exact(
                    location / "records" / f"{row['id']}-{expected['arm']}.json",
                    output / f"{row['id']}-{expected['arm']}.json",
                )
        for path in location.rglob("*"):
            if (
                not path.is_file()
                or path.suffix not in (".json", ".yaml")
                or "records" in path.relative_to(location).parts
            ):
                continue
            relative = Path(name) / path.relative_to(location)
            copy_exact(path, output / "acquisition" / relative)
            receipts[relative.as_posix()] = file_hash(path)
            if "attempts" not in path.relative_to(location).parts:
                continue
            if path.name.endswith(".intent.json"):
                unknown += not path.with_name(path.name.replace(".intent", "")).exists()
            else:
                attempt = solver.read(path)
                observed += attempt["observed_credits"]
                failures += attempt["status"] != "complete"
                unknown += attempt.get("execution_unknown", True)
    binding = {
        "ids": [row["id"] for row in inputs["train"]],
        "split": "train",
        "plan_sha256": file_hash(directory / "plan.json"),
        "settings_sha256": digest(settings),
        "policies_frozen_sha256": None,
        "self_hosted": True,
        "reuse_sha256": plan["reuse_sha256"],
        "jobs_sha256": file_hash(directory / "collection-jobs.json"),
        "jobs": bindings,
    }
    target = output / "collection-plan.json"
    if target.exists() and solver.read(target) != binding:
        raise ValueError("Combined collection binding changed")
    if not target.exists():
        write(target, binding)
    for name in ("plan.json", "collection-jobs.json", "reuse.json"):
        copy_exact(directory / name, output / "acquisition" / name)
    for name in ("collection-plan.json", "collection.json"):
        copy_exact(old_collection / name, output / "acquisition" / "reused" / name)
    for path in (old_collection / "acquisition").rglob("*"):
        if path.is_file():
            copy_exact(path, output / "acquisition" / "reused" / path.relative_to(old_collection / "acquisition"))
    summary = {
        "status": "complete" if not missing and not failures and not unknown else "incomplete",
        "completed": 2 * len(inputs["train"]) - missing,
        "failed": failures,
        "missing": missing,
        "reused_records": 2 * len(reused),
        "new_completed_records": 2 * len(plan["new_train_ids"]) - missing,
        "observed_credits": observed,
        "unknown_execution_attempts": unknown,
        "acquisition_cost_complete": False,
        "infrastructure_cost_status": "unpriced",
        "receipt_sha256": receipts,
    }
    solver.save_latest(output / "collection.json", summary)
    if summary["status"] != "complete":
        raise RuntimeError("Expanded collection is incomplete")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-jobs")
    prepare.add_argument("--directory", type=Path, required=True)
    prepare.add_argument("--strong-shards", type=int, default=2)
    acquire = commands.add_parser("collect")
    acquire.add_argument("--directory", type=Path, required=True)
    acquire.add_argument("--job", required=True)
    acquire.add_argument("--phase", choices=stages.PHASES, required=True)
    acquire.add_argument("--workers", type=int, default=4)
    acquire.add_argument("--limit", type=int)
    acquire.add_argument("--max-seconds", type=float, required=True)
    acquire.add_argument("--max-credits", type=float)
    acquire.add_argument("--self-hosted", action="store_true")
    acquire.add_argument("--resume", action="store_true")
    acquire.add_argument("--runtime-receipt", type=Path)
    combine = commands.add_parser("gather")
    combine.add_argument("--directory", type=Path, required=True)
    combine.add_argument("--prior", type=Path, required=True)
    combine.add_argument("--job-directory", action="append", default=[], help="Prepared job name=downloaded directory")
    args = parser.parse_args()
    if args.command == "prepare-jobs":
        prepare_jobs(args.directory, args.strong_shards)
    elif args.command == "collect":
        collect(
            args.directory,
            args.job,
            args.phase,
            workers=args.workers,
            limit=args.limit,
            max_seconds=args.max_seconds,
            max_credits=args.max_credits,
            self_hosted=args.self_hosted,
            resume=args.resume,
            runtime_receipt=args.runtime_receipt,
        )
    else:
        gather(args.directory, args.prior, dict(value.split("=", 1) for value in args.job_directory) or None)


if __name__ == "__main__":
    main()
