"""Collect training solver outcomes through a Cloud or self-hosted SIE endpoint.

Each arm saves its initial answers before collecting conditional finalizers.
This changes scheduling only: messages, sampling, answer budgets and finalizer
conditions come from solver.py. Completed calls are never repeated on resume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import solver
from benefit_data import checked_training
from data import digest, file_hash, write
from serving import SingleSendClient, error_receipt
from solver_answers import extract_final_answer

PHASES = ("initial", "final")


def messages_for(row, arm, settings, first=None):
    if first is None:
        return [
            {"role": "system", "content": settings["arms"][arm]["system"]},
            {"role": "user", "content": row["request"]},
        ]
    return [
        {"role": "system", "content": solver.FINAL_SYSTEM},
        {"role": "user", "content": json.dumps({"problem": row["request"], "partial_work": first["answer"]})},
    ]


def needs_final(first):
    return first["finish_reason"] == "length" or not extract_final_answer(first["answer"])


def stage_call(row, arm, settings, phase, messages, self_hosted):
    result = {
        "id": row["id"],
        "arm": arm,
        "phase": phase,
        "request_sha256": digest(row["request"]),
        "settings_sha256": digest(settings),
        "messages_sha256": digest(messages),
        "status": "incomplete",
    }
    client = None
    try:
        with solver.RecordingClient(timeout_s=180, read_timeout_s=180) as client:
            result["call"] = solver.call(
                client, settings["arms"][arm], messages, final=phase == "final", self_hosted=self_hosted
            )
            if not result["call"]["response_valid"] or not result["call"]["billing_known"]:
                raise ValueError("Successful response has no usable answer or required charge receipt")
            result["status"] = "complete"
    except Exception as error:  # noqa: BLE001 -- retain sanitized failures and any successful receipt
        result["error"] = error_receipt(error)
    failures = list(client.http_failures) if client is not None else []
    result["http_failures"] = failures
    if failures and "error" in result:
        result["error"].update(failures[-1])
    receipt = result.get("call", {})
    credits = receipt.get("receipt", {}).get("credits")
    result["observed_credits"] = (credits if solver.valid_credit(credits) else 0) + sum(
        failure["credits"] for failure in failures if solver.valid_credit(failure["credits"])
    )
    known_rejection = bool(failures) and failures[-1]["known_pre_execution_rejection"]
    result["execution_unknown"] = result["status"] != "complete" and not known_rejection
    result["safe_to_resume"] = known_rejection and not receipt
    return result


def checked_stage(record, row, arm, settings, phase, messages):
    expected = {
        "id": row["id"],
        "arm": arm,
        "phase": phase,
        "request_sha256": digest(row["request"]),
        "settings_sha256": digest(settings),
        "messages_sha256": digest(messages),
    }
    if any(record.get(key) != value for key, value in expected.items()) or record.get("status") not in (
        "complete",
        "incomplete",
    ):
        raise ValueError("Saved stage differs from its request or solver protocol")
    if not solver.valid_credit(record.get("observed_credits")):
        raise ValueError("Saved stage has invalid observed charges")
    if record["status"] == "complete":
        call = record.get("call", {})
        if (
            call.get("messages_sha256") != digest(messages)
            or not call.get("response_valid")
            or not call.get("billing_known")
            or not isinstance(call.get("answer"), str)
        ):
            raise ValueError("Saved successful stage has no valid bound call receipt")


def history(output, row, arm, settings, phase, messages):
    folder = output / "attempts" / row["id"]
    receipts = sorted(folder.glob(f"{phase}-*.json"))
    receipts = [path for path in receipts if not path.name.endswith(".intent.json")]
    records = []
    for path in receipts:
        record = solver.read(path)
        checked_stage(record, row, arm, settings, phase, messages)
        records.append((path, record))
    intents = sorted(folder.glob(f"{phase}-*.intent.json"))
    interrupted = [path for path in intents if not path.with_name(path.name.replace(".intent", "")).exists()]
    successful = [(path, record) for path, record in records if record["status"] == "complete"]
    if len(successful) > 1:
        raise ValueError("An already completed stage was submitted again")
    return records, interrupted, successful[0] if successful else None


def materialize(output, row, arm, settings, self_hosted):
    initial_messages = messages_for(row, arm, settings)
    _, _, initial = history(output, row, arm, settings, "initial", initial_messages)
    if initial is None:
        return None
    first = initial[1]["call"]
    calls = [first]
    answer = first["answer"]
    format_error = False
    if needs_final(first):
        _, _, final = history(output, row, arm, settings, "final", messages_for(row, arm, settings, first))
        if final is None:
            return None
        calls.append(final[1]["call"])
        try:
            parsed = json.loads(calls[-1]["answer"])
            if not isinstance(parsed, dict) or set(parsed) != {"answer"} or not isinstance(parsed["answer"], str):
                raise ValueError("Malformed final answer")
            answer = "\\boxed{" + parsed["answer"] + "}"
        except (ValueError, TypeError):
            answer, format_error = "", True
    observed = sum(call["receipt"]["credits"] for call in calls if solver.valid_credit(call["receipt"]["credits"]))
    record = solver.initial_record(row, arm, settings) | {
        "status": "complete",
        "calls": calls,
        "answer": answer,
        "observed_credits": observed,
        "credits": None if self_hosted else observed,
        "has_unknown_charges": False,
        "safe_to_resume": False,
    }
    if format_error:
        record["final_format_error"] = True
    solver.validate_record(record, row, arm, settings)
    target = output / "records" / f"{row['id']}-{arm}.json"
    if target.exists() and solver.read(target) != record:
        raise ValueError("A completed answer changed after acquisition")
    if not target.exists():
        write(target, record)
    return record


def collect(
    directory,
    arm,
    phase,
    *,
    workers=4,
    limit=None,
    max_seconds=600,
    max_credits=None,
    self_hosted=False,
    resume=False,
    retry_unknown=False,
    max_attempts=1,
    runtime_receipt=None,
):
    directory = Path(directory)
    _, inputs, settings = checked_training(directory)
    rows = inputs["train"]
    if arm not in settings["arms"] or phase not in PHASES:
        raise ValueError("Select a declared arm and collection phase")
    if not 1 <= workers <= 16 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Use 1..16 workers and a positive finite time limit")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("The pilot limit must be a positive integer")
    if type(max_attempts) is not int or max_attempts <= 0 or (retry_unknown and not resume):
        raise ValueError("Use positive bounded attempts; unknown repeats require explicit resume")
    if not self_hosted and (max_credits is None or not solver.valid_credit(max_credits) or max_credits <= 0):
        raise ValueError("Cloud collection requires a positive observed-credit ceiling")
    output = directory / "benefit-arms" / arm
    binding = {
        "ids": [row["id"] for row in rows],
        "arm": arm,
        "plan_sha256": file_hash(directory / "plan.json"),
        "settings_sha256": digest(settings),
        "self_hosted": self_hosted,
        "endpoint_sha256": digest(os.environ.get("SIE_BASE_URL", "https://api.superlinked.com")),
        "runtime_receipt_sha256": file_hash(runtime_receipt) if runtime_receipt else None,
        "scheduling": "Initial calls first, then conditional finalizers; no automatic generation retries",
    }
    plan_path = output / "arm-plan.json"
    if plan_path.exists():
        if not resume or solver.read(plan_path) != binding:
            raise ValueError("Existing arm requires resume with identical inputs, endpoint and runtime")
    else:
        if resume:
            raise ValueError("No existing arm to resume")
        write(plan_path, binding)
        if runtime_receipt:
            shutil.copyfile(runtime_receipt, output / "runtime-receipt.json")
    todo = []
    for row in rows[:limit] if limit else rows:
        first = None
        if phase == "final":
            _, _, initial = history(output, row, arm, settings, "initial", messages_for(row, arm, settings))
            if initial is None:
                raise ValueError("Finalization requires a successful initial receipt for every selected row")
            first = initial[1]["call"]
            if not needs_final(first):
                materialize(output, row, arm, settings, self_hosted)
                continue
        messages = messages_for(row, arm, settings, first)
        records, interrupted, success = history(output, row, arm, settings, phase, messages)
        if success:
            materialize(output, row, arm, settings, self_hosted)
            continue
        if len(records) + len(interrupted) >= max_attempts:
            raise ValueError("Attempt ceiling reached; inspect immutable receipts before raising it")
        if (interrupted or (records and not records[-1][1].get("safe_to_resume"))) and not retry_unknown:
            raise ValueError("Prior execution is unknown; inspect receipts before explicit --retry-unknown")
        todo.append((row, messages, len(records) + len(interrupted) + 1))
    sessions = output / "sessions"
    session = sessions / f"session-{len(list(sessions.glob('session-*'))) + 1:04d}"
    write(
        session / "plan.json",
        {
            "arm_plan_sha256": file_hash(plan_path),
            "phase": phase,
            "workers": workers,
            "pending_ids": [row["id"] for row, _, _ in todo],
            "max_seconds": max_seconds,
            "max_credits": max_credits,
            "max_attempts": max_attempts,
            "retry_unknown": retry_unknown,
            "limit": limit,
            "budget_scope": "New observed charges; already running calls are drained",
        },
    )
    started = time.monotonic()
    sent, failed, observed = 0, 0, 0.0
    fatal = None
    try:
        if todo:
            with SingleSendClient(timeout_s=60) as client:
                model = settings["arms"][arm]
                metadata = client.get_model(model["model"])
                if (
                    metadata.get("revision") != model["revision"]
                    or metadata.get("name", metadata.get("id")) != model["model"]
                ):
                    raise ValueError("Endpoint model revision differs from the frozen solver")
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
                row, messages, attempt = task
                target = output / "attempts" / row["id"] / f"{phase}-{attempt:04d}.json"
                write(
                    target.with_name(target.stem + ".intent.json"),
                    {
                        "id": row["id"],
                        "arm": arm,
                        "phase": phase,
                        "session": session.name,
                        "request_sha256": digest(row["request"]),
                        "settings_sha256": digest(settings),
                        "messages_sha256": digest(messages),
                    },
                )
                future = pool.submit(stage_call, row, arm, settings, phase, messages, self_hosted)
                pending[future] = row, messages, target
                sent += 1

            for _ in range(workers):
                schedule()
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    row, messages, target = pending.pop(future)
                    record = future.result()
                    checked_stage(record, row, arm, settings, phase, messages)
                    write(target, record)
                    observed += record["observed_credits"]
                    failed += record["status"] != "complete"
                    materialize(output, row, arm, settings, self_hosted)
                    print(f"{arm}/{phase}: {sent - len(pending)}/{len(todo)} returned, {failed} failed", flush=True)
                for _ in done:
                    schedule()
    except Exception as error:  # noqa: BLE001 -- keep stage receipts after any orchestration failure
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
        raise RuntimeError("Collection phase incomplete; preserve receipts and inspect failures before manual resume")
    return summary


def gather(directory, arm_directories=None):
    directory = Path(directory)
    _, inputs, settings = checked_training(directory)
    rows = inputs["train"]
    locations = arm_directories or {arm: directory / "benefit-arms" / arm for arm in settings["arms"]}
    if set(locations) != set(settings["arms"]):
        raise ValueError("Gather requires exactly the declared solver arms")
    plans = {arm: solver.read(Path(path) / "arm-plan.json") for arm, path in locations.items()}
    self_hosted = plans[next(iter(plans))]["self_hosted"]
    for arm, plan in plans.items():
        if (
            plan["ids"] != [r["id"] for r in rows]
            or plan["arm"] != arm
            or plan["plan_sha256"] != file_hash(directory / "plan.json")
            or plan["settings_sha256"] != digest(settings)
            or plan["self_hosted"] != self_hosted
        ):
            raise ValueError("Arm evidence differs from the prepared training population or hosting mode")
    output = directory / "answers-train"
    plan = {
        "ids": [r["id"] for r in rows],
        "split": "train",
        "plan_sha256": file_hash(directory / "plan.json"),
        "settings_sha256": digest(settings),
        "policies_frozen_sha256": None,
        "self_hosted": self_hosted,
        "arms": plans,
        "acquisition_scheduling": "Initial calls before conditional finalizers, separately per arm",
    }
    plan_path = output / "collection-plan.json"
    if plan_path.exists() and solver.read(plan_path) != plan:
        raise ValueError("Existing combined collection uses a different acquisition plan")
    if not plan_path.exists():
        write(plan_path, plan)
    completed, observed, failed_attempts, unknown = 0, 0.0, 0, 0
    receipts = {}
    for arm, location in locations.items():
        location = Path(location)
        for row in rows:
            record = materialize(location, row, arm, settings, self_hosted)
            if record is not None:
                target = output / f"{row['id']}-{arm}.json"
                if target.exists() and solver.read(target) != record:
                    raise ValueError("Combined answer changed after collection")
                if not target.exists():
                    write(target, record)
                completed += 1
            for phase in PHASES:
                folder = location / "attempts" / row["id"]
                for path in sorted(folder.glob(f"{phase}-*.json")):
                    relative = Path(arm) / path.relative_to(location)
                    target = output / "acquisition" / relative
                    if target.exists() and file_hash(target) != file_hash(path):
                        raise ValueError("Archived acquisition receipt changed")
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
                    receipts[relative.as_posix()] = file_hash(path)
                    if path.name.endswith(".intent.json"):
                        unknown += not path.with_name(path.name.replace(".intent", "")).exists()
                    else:
                        attempt = solver.read(path)
                        observed += attempt["observed_credits"]
                        failed_attempts += attempt["status"] != "complete"
                        unknown += attempt.get("execution_unknown", True)
    summary = {
        "status": "complete" if completed == 2 * len(rows) else "incomplete",
        "completed": completed,
        "failed": 0,
        "missing": 2 * len(rows) - completed,
        "observed_credits": observed,
        "failed_attempts": failed_attempts,
        "unknown_execution_attempts": unknown,
        "acquisition_cost_complete": not self_hosted and unknown == 0,
        "infrastructure_cost_status": "unpriced" if self_hosted else "not_applicable",
        "receipt_sha256": receipts,
    }
    solver.save_latest(output / "collection.json", summary)
    if summary["status"] != "complete":
        raise RuntimeError("Paired training collection is incomplete")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("collect")
    acquire.add_argument("--directory", type=Path, required=True)
    acquire.add_argument("--arm", choices=tuple(solver.ARMS), required=True)
    acquire.add_argument("--phase", choices=PHASES, required=True)
    acquire.add_argument("--workers", type=int, default=4)
    acquire.add_argument("--limit", type=int)
    acquire.add_argument("--max-seconds", type=float, required=True)
    acquire.add_argument("--max-credits", type=float)
    acquire.add_argument("--self-hosted", action="store_true")
    acquire.add_argument("--resume", action="store_true")
    acquire.add_argument("--retry-unknown", action="store_true")
    acquire.add_argument("--max-attempts", type=int, default=1)
    acquire.add_argument("--runtime-receipt", type=Path)
    combine = commands.add_parser("gather")
    combine.add_argument("--directory", type=Path, required=True)
    combine.add_argument("--cheap", type=Path)
    combine.add_argument("--strong", type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        values = vars(args).copy()
        values.pop("command")
        result = collect(**values)
    else:
        if bool(args.cheap) != bool(args.strong):
            parser.error("Provide both --cheap and --strong, or neither")
        result = gather(args.directory, {"cheap": args.cheap, "strong": args.strong} if args.cheap else None)
    print(json.dumps({key: value for key, value in result.items() if key != "receipt_sha256"}, sort_keys=True))


if __name__ == "__main__":
    main()
