"""Collect a separately declared transfer sample without reading reference answers.

The solver protocol is unchanged. All attempts are saved; an explicit resume may
repeat only a verified pre-execution rejection, at most twice per stage.
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
import benefit_expanded_collect as archives
import solver
from data import digest, file_hash, write
from serving import SingleSendClient, error_receipt

FORMAT = "sie-transfer-acquisition-v1"
CODE = (
    "transfer_collect.py",
    "benefit_collect.py",
    "benefit_expanded_collect.py",
    "benefit_data.py",
    "solver.py",
    "solver_answers.py",
    "solver_gold.py",
    "serving.py",
    "data.py",
)


def code_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in CODE}


def inputs(directory):
    rows = solver.read(directory / "inputs.json")
    settings = solver.read(directory / "settings.json")
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            set(row) != {"id", "request"} or not all(isinstance(row[k], str) and row[k] for k in row) for row in rows
        )
    ):
        raise ValueError("Transfer inputs must be nonempty id/request rows")
    if len({row["id"] for row in rows}) != len(rows) or settings != solver.SETTINGS:
        raise ValueError("Transfer IDs must be unique and solver settings unchanged")
    return rows, settings


def prepare(directory, shards=2):
    directory = Path(directory)
    rows, settings = inputs(directory)
    if shards not in (1, 2):
        raise ValueError("Use one or two disjoint shards per arm")
    result = {
        "format": FORMAT,
        "source_plan_sha256": file_hash(directory / "source-plan.json"),
        "inputs_sha256": file_hash(directory / "inputs.json"),
        "settings_file_sha256": file_hash(directory / "settings.json"),
        "settings_sha256": digest(settings),
        "source_code_sha256": code_hashes(),
        "jobs": {
            f"{arm}-{index}": {"arm": arm, "ids": [row["id"] for row in rows[index::shards]]}
            for arm in settings["arms"]
            for index in range(shards)
        },
        "max_generation_attempts": 2,
        "retry_policy": "Explicit resume only for proven pre-execution rejection; never unknown execution or answer quality",
        "scheduling": "Disjoint source-order round-robin shards; initial stage then conditional finalizers",
        "reference_answers_read": False,
    }
    path = directory / "collection-jobs.json"
    if path.exists():
        raise FileExistsError(path)
    write(path, result)
    return result


def checked_jobs(directory):
    directory = Path(directory)
    rows, settings = inputs(directory)
    jobs = solver.read(directory / "collection-jobs.json")
    if (
        jobs.get("format") != FORMAT
        or jobs.get("source_plan_sha256") != file_hash(directory / "source-plan.json")
        or jobs.get("inputs_sha256") != file_hash(directory / "inputs.json")
        or jobs.get("settings_file_sha256") != file_hash(directory / "settings.json")
        or jobs.get("settings_sha256") != digest(settings)
        or jobs.get("source_code_sha256") != code_hashes()
        or jobs.get("max_generation_attempts") != 2
    ):
        raise ValueError("Frozen transfer inputs, protocol or source changed")
    by_id = {row["id"]: row for row in rows}
    for arm in settings["arms"]:
        ids = [identity for job in jobs["jobs"].values() if job["arm"] == arm for identity in job["ids"]]
        if len(ids) != len(set(ids)) or set(ids) != set(by_id):
            raise ValueError("Every transfer input needs exactly one job per arm")
    if any(job["arm"] not in settings["arms"] for job in jobs["jobs"].values()):
        raise ValueError("Unknown transfer solver arm")
    return jobs, {name: [by_id[identity] for identity in job["ids"]] for name, job in jobs["jobs"].items()}, settings


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
        if interrupted or (previous and not previous[-1][1].get("safe_to_resume")):
            raise ValueError("Unknown or dispatched executions cannot be repeated")
        if len(previous) >= 2:
            raise ValueError("Two-attempt ceiling reached")
        todo.append((row, messages, len(previous) + 1))
    sessions = output / "sessions"
    session = sessions / f"session-{len(list(sessions.glob('session-*'))) + 1:04d}"
    write(
        session / "plan.json",
        {
            "job_plan_sha256": file_hash(plan_path),
            "phase": phase,
            "workers": workers,
            "pending_ids": [row["id"] for row, _, _ in todo],
            "max_seconds": max_seconds,
            "max_credits": max_credits,
            "limit": limit,
            "max_generation_attempts": 2,
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
                row, messages, attempt_number = task
                target = output / "attempts" / row["id"] / f"{phase}-{attempt_number:04d}.json"
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


def gather(directory, locations):
    directory = Path(directory)
    jobs, populations, settings = checked_jobs(directory)
    if set(locations) != set(populations):
        raise ValueError("Need exactly the declared transfer jobs")
    output = directory / "answers"
    bindings, receipts, records, missing = {}, {}, {}, 0
    failed_attempts, unknown, observed = 0, 0, 0.0
    for name, location in locations.items():
        location = Path(location)
        binding = solver.read(location / "job-plan.json")
        if (
            binding.get("job") != name
            or binding.get("arm") != jobs["jobs"][name]["arm"]
            or binding.get("ids") != jobs["jobs"][name]["ids"]
            or binding.get("jobs_sha256") != file_hash(directory / "collection-jobs.json")
        ):
            raise ValueError("Transfer job binding differs")
        if (
            binding.get("runtime_receipt_sha256")
            and file_hash(location / "runtime-receipt.json") != binding["runtime_receipt_sha256"]
        ):
            raise ValueError("Transfer runtime receipt changed")
        bindings[name] = binding
        arm = binding["arm"]
        for row in populations[name]:
            record = stages.materialize(location, row, arm, settings, binding["self_hosted"])
            if record is None:
                missing += 1
                continue
            filename = f"{row['id']}-{arm}.json"
            archives.copy_exact(location / "records" / filename, output / filename)
            records[filename] = file_hash(output / filename)
        for path in location.rglob("*"):
            if (
                not path.is_file()
                or path.suffix not in (".json", ".yaml")
                or "records" in path.relative_to(location).parts
            ):
                continue
            relative = Path(name) / path.relative_to(location)
            archives.copy_exact(path, output / "acquisition" / relative)
            receipts[relative.as_posix()] = file_hash(path)
            if "attempts" not in path.relative_to(location).parts:
                continue
            if path.name.endswith(".intent.json"):
                unknown += not path.with_name(path.name.replace(".intent", "")).exists()
            else:
                attempt = solver.read(path)
                failed_attempts += attempt["status"] != "complete"
                unknown += attempt.get("execution_unknown", True)
                observed += attempt["observed_credits"]
    plan = {
        "format": FORMAT,
        "source_plan_sha256": file_hash(directory / "source-plan.json"),
        "jobs_sha256": file_hash(directory / "collection-jobs.json"),
        "settings_sha256": digest(settings),
        "jobs": bindings,
        "records_sha256": records,
    }
    target = output / "collection-plan.json"
    if target.exists() and solver.read(target) != plan:
        raise ValueError("Completed transfer collection cannot be overwritten")
    if not target.exists():
        write(target, plan)
    summary = {
        "status": "complete" if not missing and not unknown else "incomplete",
        "completed": len(records),
        "missing": missing,
        "failed_attempts": failed_attempts,
        "unknown_execution_attempts": unknown,
        "observed_credits": observed,
        "receipt_sha256": receipts,
        "graded": False,
    }
    solver.save_latest(output / "collection.json", summary)
    if summary["status"] != "complete":
        raise RuntimeError("Transfer collection is incomplete; retain all acquisition evidence")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--directory", type=Path, required=True)
    prep.add_argument("--shards", type=int, default=2)
    acquire = sub.add_parser("collect")
    acquire.add_argument("--directory", type=Path, required=True)
    acquire.add_argument("--job", required=True)
    acquire.add_argument("--phase", choices=stages.PHASES, required=True)
    acquire.add_argument("--workers", type=int, default=4)
    acquire.add_argument("--max-seconds", type=float, required=True)
    acquire.add_argument("--max-credits", type=float)
    acquire.add_argument("--self-hosted", action="store_true")
    acquire.add_argument("--resume", action="store_true")
    acquire.add_argument("--runtime-receipt", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, args.shards)
    else:
        collect(
            args.directory,
            args.job,
            args.phase,
            workers=args.workers,
            max_seconds=args.max_seconds,
            max_credits=args.max_credits,
            self_hosted=args.self_hosted,
            resume=args.resume,
            runtime_receipt=args.runtime_receipt,
        )


if __name__ == "__main__":
    main()
