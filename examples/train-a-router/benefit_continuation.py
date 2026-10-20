"""Continue unfinished solver stages on explicitly recorded replacement runtimes.

The original job and receipts remain immutable. Disjoint continuation shards send
only stages without an intent or receipt; merge records the runtime of every call.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import benefit_collect as stages
import benefit_expanded_collect as original
import solver
from data import digest, file_hash, write
from serving import error_receipt

FORMAT = "sie-benefit-runtime-continuation-v1"


def inventory(directory):
    return {
        path.relative_to(directory).as_posix(): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in (".json", ".yaml")
    }


def prepare(directory, prior, destination, *, job="cheap", shards=2):
    directory, prior, destination = Path(directory), Path(prior), Path(destination)
    jobs, populations, settings = original.checked_jobs(directory)
    if job not in populations or shards not in (1, 2):
        raise ValueError("Choose a prepared job and one or two continuation shards")
    binding = solver.read(prior / "job-plan.json")
    arm, rows = jobs["jobs"][job]["arm"], populations[job]
    if (
        binding.get("job") != job
        or binding.get("arm") != arm
        or binding.get("ids") != [row["id"] for row in rows]
        or binding.get("jobs_sha256") != file_hash(directory / "collection-jobs.json")
        or not binding.get("self_hosted")
        or binding.get("runtime_receipt_sha256") != file_hash(prior / "runtime-receipt.json")
    ):
        raise ValueError("Original job or runtime binding differs")
    pending, submitted = [], {}
    for row in rows:
        initial_messages = stages.messages_for(row, arm, settings)
        history, interrupted, initial = stages.history(prior, row, arm, settings, "initial", initial_messages)
        if interrupted or (history and initial is None):
            raise ValueError("Submitted initial execution cannot be repeated")
        if initial is None:
            pending.append(row["id"])
            submitted[row["id"]] = "initial"
            continue
        if stages.needs_final(initial[1]["call"]):
            messages = stages.messages_for(row, arm, settings, initial[1]["call"])
            history, interrupted, final = stages.history(prior, row, arm, settings, "final", messages)
            if interrupted or (history and final is None):
                raise ValueError("Submitted final execution cannot be repeated")
            if final is None:
                pending.append(row["id"])
                submitted[row["id"]] = "final"
    result = {
        "format": FORMAT,
        "job": job,
        "arm": arm,
        "jobs_sha256": file_hash(directory / "collection-jobs.json"),
        "original_job_plan_sha256": file_hash(prior / "job-plan.json"),
        "original_runtime_sha256": file_hash(prior / "runtime-receipt.json"),
        "original_files_sha256": inventory(prior),
        "pending_stage_at_prepare": submitted,
        "shards": {f"{job}-continuation-{i}": pending[i::shards] for i in range(shards)},
        "collector_sha256": file_hash(Path(__file__)),
        "max_generation_attempts": 1,
        "selection": "All unfinished source-order IDs, disjoint round-robin; no grading or outcome selection",
    }
    if (destination / "continuation-plan.json").exists():
        raise FileExistsError(destination / "continuation-plan.json")
    write(destination / "continuation-plan.json", result)
    return result


def checked(directory, prior, continuation):
    directory, prior, continuation = Path(directory), Path(prior), Path(continuation)
    jobs, populations, settings = original.checked_jobs(directory)
    plan = solver.read(continuation / "continuation-plan.json")
    if (
        plan.get("format") != FORMAT
        or plan.get("collector_sha256") != file_hash(Path(__file__))
        or plan.get("jobs_sha256") != file_hash(directory / "collection-jobs.json")
        or plan.get("original_files_sha256") != inventory(prior)
        or plan.get("max_generation_attempts") != 1
    ):
        raise ValueError("Continuation source or original receipts changed")
    ids = [identity for values in plan["shards"].values() for identity in values]
    if len(ids) != len(set(ids)) or set(ids) != set(plan["pending_stage_at_prepare"]):
        raise ValueError("Continuation shards must cover unfinished IDs exactly once")
    if plan["arm"] != jobs["jobs"][plan["job"]]["arm"]:
        raise ValueError("Continuation arm changed")
    by_id = {row["id"]: row for row in populations[plan["job"]]}
    return plan, {name: [by_id[i] for i in values] for name, values in plan["shards"].items()}, settings


def collect(directory, prior, continuation, shard, phase, runtime_receipt, *, workers=16, max_seconds=3600):
    directory, prior, continuation = Path(directory), Path(prior), Path(continuation)
    plan, populations, settings = checked(directory, prior, continuation)
    if shard not in populations or phase not in stages.PHASES:
        raise ValueError("Select a declared continuation shard and phase")
    if not 1 <= workers <= 16 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Use 1..16 workers and a finite positive time limit")
    output = continuation / "shards" / shard
    for relative in plan["original_files_sha256"]:
        original.copy_exact(prior / relative, output / relative)
    runtime_hash = file_hash(runtime_receipt)
    binding = {
        "continuation_plan_sha256": file_hash(continuation / "continuation-plan.json"),
        "shard": shard,
        "ids": plan["shards"][shard],
        "runtime_receipt_sha256": runtime_hash,
    }
    target = output / "continuation-binding.json"
    if target.exists() and solver.read(target) != binding:
        raise ValueError("A continuation shard cannot change runtime or IDs")
    if not target.exists():
        write(target, binding)
    original.copy_exact(Path(runtime_receipt), output / "runtime-continuation.json")
    arm, todo = plan["arm"], []
    for row in populations[shard]:
        first = None
        if phase == "final":
            _, _, initial = stages.history(
                output, row, arm, settings, "initial", stages.messages_for(row, arm, settings)
            )
            if initial is None:
                raise ValueError("Final phase requires every initial receipt")
            first = initial[1]["call"]
            if not stages.needs_final(first):
                stages.materialize(output, row, arm, settings, True)
                continue
        messages = stages.messages_for(row, arm, settings, first)
        previous, interrupted, success = stages.history(output, row, arm, settings, phase, messages)
        if success:
            continue
        if previous or interrupted:
            raise ValueError("Submitted calls cannot be repeated")
        todo.append((row, messages))
    sessions = output / "sessions"
    session = sessions / f"{shard}-{len(list(sessions.glob(shard + '-*'))) + 1:04d}"
    write(
        session / "plan.json",
        binding
        | {
            "phase": phase,
            "pending_ids": [row["id"] for row, _ in todo],
            "workers": workers,
            "max_seconds": max_seconds,
            "max_generation_attempts": 1,
        },
    )
    started, sent, failed, fatal = time.monotonic(), 0, 0, None
    try:
        if todo:
            with original.SingleSendClient(timeout_s=60) as client:
                model = settings["arms"][arm]
                metadata = client.get_model(model["model"])
                if (
                    metadata.get("revision") != model["revision"]
                    or metadata.get("name", metadata.get("id")) != model["model"]
                ):
                    raise ValueError("Endpoint model identity or revision differs")
                write(session / "model.json", {key: metadata.get(key) for key in ("name", "revision", "profiles")})
        tasks = iter(todo)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}

            def schedule():
                nonlocal sent
                if failed or time.monotonic() - started >= max_seconds:
                    return
                task = next(tasks, None)
                if task is None:
                    return
                row, messages = task
                path = output / "attempts" / row["id"] / f"{phase}-0001.json"
                write(
                    path.with_name(path.stem + ".intent.json"),
                    {
                        "id": row["id"],
                        "arm": arm,
                        "phase": phase,
                        "session": session.name,
                        "request_sha256": digest(row["request"]),
                        "settings_sha256": digest(settings),
                        "messages_sha256": digest(messages),
                        **binding,
                    },
                )
                pending[pool.submit(stages.stage_call, row, arm, settings, phase, messages, True)] = row, messages, path
                sent += 1

            for _ in range(workers):
                schedule()
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    row, messages, path = pending.pop(future)
                    result = future.result()
                    stages.checked_stage(result, row, arm, settings, phase, messages)
                    write(path, result)
                    failed += result["status"] != "complete"
                    stages.materialize(output, row, arm, settings, True)
                    print(f"{shard}/{phase}: {sent - len(pending)}/{len(todo)} returned, {failed} failed", flush=True)
                for _ in done:
                    schedule()
    except Exception as error:  # noqa: BLE001 -- keep every submitted intent and returned receipt
        fatal = error_receipt(error)
    summary = {
        "status": "complete" if sent == len(todo) and not failed and fatal is None else "incomplete",
        "phase": phase,
        "sent": sent,
        "failed": failed,
        "fatal_error": fatal,
        "elapsed_seconds": time.monotonic() - started,
    }
    write(session / "collection.json", summary)
    if summary["status"] != "complete":
        raise RuntimeError("Continuation incomplete; submitted calls will not be repeated")
    return summary


def assemble(directory, prior, continuation, locations, destination):
    directory, prior, continuation, destination = map(Path, (directory, prior, continuation, destination))
    plan, populations, settings = checked(directory, prior, continuation)
    if set(locations) != set(populations):
        raise ValueError("Need every declared continuation shard")
    for relative in plan["original_files_sha256"]:
        if relative != "job-plan.json":
            original.copy_exact(prior / relative, destination / relative)
    original.copy_exact(prior / "job-plan.json", destination / "provenance/original-job-plan.json")
    original.copy_exact(continuation / "continuation-plan.json", destination / "provenance/continuation-plan.json")
    call_runtimes = {
        relative: plan["original_runtime_sha256"]
        for relative in plan["original_files_sha256"]
        if relative.startswith("attempts/") and not relative.endswith(".intent.json")
    }
    bindings = {}
    for shard, location in locations.items():
        location = Path(location)
        binding = solver.read(location / "continuation-binding.json")
        if (
            binding.get("continuation_plan_sha256") != file_hash(continuation / "continuation-plan.json")
            or binding.get("shard") != shard
            or binding.get("ids") != plan["shards"][shard]
            or binding.get("runtime_receipt_sha256") != file_hash(location / "runtime-continuation.json")
        ):
            raise ValueError("Continuation shard provenance changed")
        bindings[shard] = binding
        for path in location.rglob("*"):
            if not path.is_file() or path.suffix not in (".json", ".yaml"):
                continue
            relative = path.relative_to(location)
            if relative.as_posix() in plan["original_files_sha256"]:
                if file_hash(path) != plan["original_files_sha256"][relative.as_posix()]:
                    raise ValueError("A reused receipt changed")
                continue
            if relative.parts[0] in ("attempts", "records", "sessions"):
                original.copy_exact(path, destination / relative)
                if relative.parts[0] == "attempts" and not path.name.endswith(".intent.json"):
                    if relative.parts[1] not in plan["shards"][shard] or relative.as_posix() in call_runtimes:
                        raise ValueError("Call was repeated or assigned to the wrong shard")
                    call_runtimes[relative.as_posix()] = binding["runtime_receipt_sha256"]
            else:
                original.copy_exact(path, destination / "provenance" / shard / relative)
        for row in populations[shard]:
            if stages.materialize(destination, row, plan["arm"], settings, True) is None:
                raise ValueError("Continuation did not finish every assigned ID")
    merged = solver.read(prior / "job-plan.json") | {
        "runtime_receipt_scope": "Original acquisition only; call_runtime_sha256 maps every continuation call explicitly",
        "continuation_plan_sha256": file_hash(continuation / "continuation-plan.json"),
        "continuations": bindings,
        "call_runtime_sha256": call_runtimes,
    }
    path = destination / "job-plan.json"
    if path.exists() and solver.read(path) != merged:
        raise ValueError("Merged runtime provenance changed")
    if not path.exists():
        write(path, merged)
    return merged
