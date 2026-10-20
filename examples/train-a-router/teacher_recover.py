"""Finish a public teacher-label collection with bounded transient retries.

Keep the first schema-valid completed answer by attempt order. Retry only
429/5xx/transport/timeouts, at most twice per question across this lineage, and
keep every unknown charge reserved. One8192-to32768 output fallback remains.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from decimal import Decimal
from pathlib import Path

import httpx

import teacher_collect
import teacher_continue as previous
from data import file_hash, write

FORMAT = "sie-teacher-reference-recovery-v1"
TRANSIENT = {
    "ConnectError",
    "ConnectTimeout",
    "ReadError",
    "ReadTimeout",
    "WriteError",
    "WriteTimeout",
    "RemoteProtocolError",
    "TimeoutError",
    "PoolTimeout",
}
MAX_TRANSIENT_RETRIES = 2


def read(path):
    return json.loads(Path(path).read_text())


def transient(record):
    error = record.get("error") or {}
    return (
        error.get("kind") in TRANSIENT
        or error.get("kind") == "http_error"
        and (error.get("status") == 429 or 500 <= error.get("status", 0) <= 599)
    )


def next_state(state, record):
    """Outcome-blind retry rule: inspect delivery/schema status, never correctness."""
    state = dict(state)
    if record["status"] == "complete":
        return "complete", state
    if previous.known_exhaustion(record):
        if state["output_limit"] == previous.INITIAL_TOKENS:
            state["output_limit"] = previous.FALLBACK_TOKENS
            return "fallback", state
        return "missing", state
    if transient(record):
        state["transient_failures"] += 1
        return ("retry" if state["transient_failures"] <= MAX_TRANSIENT_RETRIES else "missing"), state
    status = (record.get("error") or {}).get("status")
    return ("terminal" if status in (400, 401, 403, 404) else "missing"), state


def prepare(prior, output):
    prior, output = Path(prior), Path(output)
    if output.exists():
        raise FileExistsError(output)
    old, rows = previous.checked_plan(prior)
    summary = read(prior / "collection.json")
    if summary["plan_sha256"] != file_hash(prior / "plan.json") or summary["status"] != "incomplete":
        raise ValueError("Recovery requires the preserved completed incomplete phase")
    reused = list(summary["record_hashes"])
    for identity in reused:
        path = prior / "records" / f"{identity}.json"
        if file_hash(path) != summary["record_hashes"][identity]:
            raise ValueError("Completed teacher answer changed")
        previous.check_record(read(path), rows[identity])
    states, exhausted = [], []
    for row in old["inputs"]:
        identity = row["id"]
        if identity in set(reused):
            continue
        state = {"id": identity, "output_limit": previous.INITIAL_TOKENS, "transient_failures": 0}
        history = []
        first = prior / "previous/records" / f"{identity}.json"
        if first.exists():
            history.append(read(first))
        history.extend(
            read(path)
            for path in sorted(
                (prior / "attempts").glob(f"{identity}-*.json"), key=lambda path: int(path.stem.rsplit("-", 1)[1])
            )
        )
        action = "pending"
        for record in history:
            state["output_limit"] = record.get("output_limit", previous.INITIAL_TOKENS)
            action, state = next_state(state, record)
            if action == "complete":
                raise ValueError("A completed answer must be preserved, not repeated")
        if action in ("terminal", "missing"):
            exhausted.append(identity)
        else:
            states.append(state)
    states.sort(key=lambda state: teacher_collect.digest(["teacher-extension-order-v1", state["id"]]))
    plan = {
        "format": FORMAT,
        "created_unix": time.time(),
        "n": 510,
        "inputs": old["inputs"],
        "settings": old["settings"],
        "fallback_settings": old["fallback_settings"],
        "model": old["model"],
        "origin_plan_sha256": old["origin_plan_sha256"],
        "comparison_design_sha256": old["comparison_design_sha256"],
        "previous_plan_sha256": file_hash(prior / "plan.json"),
        "previous_files_sha256": {
            str(path.relative_to(prior)): file_hash(path) for path in sorted(prior.rglob("*")) if path.is_file()
        },
        "reused_ids": reused,
        "pending": states,
        "previously_exhausted_ids": exhausted,
        "historical_list_price_estimate_usd": str(summary["list_price_estimate_usd"]),
        "historical_conservative_usd": summary["conservative_list_price_usd"],
        "prior_unknown_exposure_usd": str(
            Decimal(summary["prior_unknown_exposure_usd"])
            + sum(
                (
                    Decimal(read(prior / "intents" / name)["reservation_usd"])
                    for name in summary["intent_hashes"]
                    if not read(prior / "attempts" / name)["usage_verified"]
                ),
                Decimal(0),
            )
        ),
        "max_usd": "40",
        "max_seconds": 3600,
        "workers": 4,
        "request_timeout_seconds": previous.REQUEST_SECONDS,
        "maximum_transient_retries_per_question": MAX_TRANSIENT_RETRIES,
        "backoff_seconds": [2, 4],
        "selection": "First schema-valid completed answer by attempt order; no correctness or benchmark values are consulted",
        "protocol_amendment": "Retain all historical answers/attempts/costs. Retry429/5xx/transport/timeouts at most twice per item with backoff while continuing other items. Keep the single8192-to32768 fallback; no third output-budget phase. Declare before teacher-label auditing. Learner, folds and calibration remain fixed.",
        "teacher_receives_gold": False,
        "teacher_receives_solver_answers": False,
        "operator_sha256": file_hash(Path(__file__)),
        "previous_operator_sha256": file_hash(Path(previous.__file__)),
        "teacher_collect_sha256": file_hash(Path(teacher_collect.__file__)),
        "data_source_sha256": file_hash(Path(__file__).with_name("data.py")),
    }
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    shutil.copytree(prior, output / "previous")
    (output / "records").mkdir()
    for identity in reused:
        shutil.copyfile(prior / "records" / f"{identity}.json", output / "records" / f"{identity}.json")
    return plan


def checked_plan(run):
    run = Path(run)
    plan = read(run / "plan.json")
    if (
        plan.get("format") != FORMAT
        or plan["settings"] != teacher_collect.SETTINGS
        or plan["fallback_settings"] != teacher_collect.SETTINGS | {"max_output_tokens": previous.FALLBACK_TOKENS}
        or plan["operator_sha256"] != file_hash(Path(__file__))
        or plan["previous_operator_sha256"] != file_hash(Path(previous.__file__))
        or plan["teacher_collect_sha256"] != file_hash(Path(teacher_collect.__file__))
        or plan["data_source_sha256"] != file_hash(Path(__file__).with_name("data.py"))
        or plan["max_usd"] != "40"
        or plan["max_seconds"] != 3600
        or plan["workers"] != 4
        or plan["maximum_transient_retries_per_question"] != 2
        or plan["backoff_seconds"] != [2, 4]
        or plan["teacher_receives_gold"] is not False
        or plan["teacher_receives_solver_answers"] is not False
    ):
        raise ValueError("Teacher recovery protocol changed")
    for name, expected in plan["previous_files_sha256"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or file_hash(run / "previous" / path) != expected:
            raise ValueError("Historical teacher record changed")
    old, rows = previous.checked_plan(run / "previous")
    summary = read(run / "previous/collection.json")
    if (
        plan["inputs"] != old["inputs"]
        or len(rows) != 510
        or plan["historical_conservative_usd"] != summary["conservative_list_price_usd"]
    ):
        raise ValueError("Teacher population or historical costs changed")
    planned = plan["reused_ids"] + [state["id"] for state in plan["pending"]] + plan["previously_exhausted_ids"]
    if len(planned) != 510 or len(set(planned)) != 510 or set(planned) != set(rows):
        raise ValueError("Teacher recovery must account for every original question")
    for identity in plan["reused_ids"]:
        path = run / "records" / f"{identity}.json"
        if file_hash(path) != summary["record_hashes"][identity]:
            raise ValueError("Reused answer bytes changed")
        previous.check_record(read(path), rows[identity])
    return plan, rows


async def collect(run, secret=None, request_fn=teacher_collect._http_request):
    run = Path(run)
    plan, rows = checked_plan(run)
    if any((run / name).exists() for name in ("started.json", "collection.json", "calls", "intents")):
        raise FileExistsError("Recovery already attempted; every new phase must be explicit")
    secret = secret or os.environ.get("OPENAI_API_KEY")
    if not secret:
        raise ValueError("Set OPENAI_API_KEY")
    budget = teacher_collect.Budget(plan["max_usd"], plan["historical_conservative_usd"])
    pending = [dict(state, ready=0, attempt=0) for state in plan["pending"]]
    active, successes, attempted, selected, missing = {}, [], [], {}, list(plan["previously_exhausted_ids"])
    started, stop, ledger = time.monotonic(), None, 0
    write(
        run / "started.json",
        {"plan_sha256": file_hash(run / "plan.json"), "started_unix": time.time(), "budget": budget.receipt()},
    )
    transport = httpx.AsyncHTTPTransport(retries=0, trust_env=False)
    async with httpx.AsyncClient(
        timeout=previous.REQUEST_SECONDS, transport=transport, follow_redirects=False, trust_env=False
    ) as client:
        while pending or active:
            while pending and stop is None and len(active) < plan["workers"]:
                if time.monotonic() - started > plan["max_seconds"] - previous.REQUEST_SECONDS - 5:
                    stop = "deadline"
                    break
                eligible = next(
                    (index for index, state in enumerate(pending) if state["ready"] <= time.monotonic()), None
                )
                if eligible is None:
                    break
                state = pending[eligible]
                identity, limit = state["id"], state["output_limit"]
                number = state["attempt"] + 1
                key = f"{identity}-{number:02d}"
                amount = previous.reservation(rows[identity], limit)
                if not budget.reserve(key, amount):
                    if not active:
                        stop = "budget"
                    break
                pending.pop(eligible)
                state["attempt"] = number
                write(
                    run / "intents" / f"{key}.json",
                    {
                        "id": identity,
                        "output_limit": limit,
                        "attempt": number,
                        "request_sha256": teacher_collect.digest(previous.body_for(rows[identity], limit)),
                        "plan_sha256": file_hash(run / "plan.json"),
                        "reservation_usd": str(amount),
                        "budget_after_reservation": budget.receipt(),
                        "started_unix": time.time(),
                    },
                )
                task = asyncio.create_task(
                    previous.call(run / "calls" / key, rows[identity], limit, secret, client, request_fn)
                )
                active[task] = (state, key)
            if not active:
                if pending and stop is None:
                    await asyncio.sleep(max(0.01, min(5, min(state["ready"] for state in pending) - time.monotonic())))
                    continue
                break
            done, _ = await asyncio.wait(active, timeout=10, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                state, key = active.pop(task)
                record = await task
                attempted.append((key, record))
                budget.settle(key, record.get("conservative_list_price_usd") if record["usage_verified"] else None)
                action, amended = next_state(state, record)
                identity, limit = state["id"], state["output_limit"]
                source = run / "calls" / key / "attempts" / f"{identity}-{limit}.json"
                if action == "complete":
                    shutil.copyfile(source, run / "records" / f"{identity}.json")
                    successes.append(identity)
                    selected[identity] = str(source.relative_to(run))
                elif action in ("retry", "fallback"):
                    amended["ready"] = time.monotonic() + (
                        2 ** amended["transient_failures"] if action == "retry" else 0
                    )
                    pending.append(amended)
                else:
                    missing.append(identity)
                    if action == "terminal":
                        stop = "terminal_auth_or_configuration_error"
            progress = {
                "completed_new": len(successes),
                "attempts": len(attempted),
                "pending": len(pending),
                "inflight": len(active),
                "missing": len(missing),
                "stop_reason": stop,
                "budget": budget.receipt(),
                "elapsed_seconds": time.monotonic() - started,
            }
            write(run / "ledger" / f"{ledger:05d}.json", progress)
            ledger += 1
            print(json.dumps(progress), flush=True)
    record_hashes = {
        identity: file_hash(run / "records" / f"{identity}.json") for identity in plan["reused_ids"] + successes
    }
    invoice = Decimal(plan["historical_list_price_estimate_usd"]) + sum(
        (Decimal(record["list_price_estimate_usd"]) for _, record in attempted if record["usage_verified"]), Decimal(0)
    )
    unknown = Decimal(plan["prior_unknown_exposure_usd"]) + sum(
        (
            Decimal(read(run / "intents" / f"{key}.json")["reservation_usd"])
            for key, record in attempted
            if not record["usage_verified"]
        ),
        Decimal(0),
    )
    summary = {
        "status": "complete" if len(record_hashes) == 510 else "incomplete",
        "n": 510,
        "returned": len(record_hashes),
        "complete": len(record_hashes),
        "plan_sha256": file_hash(run / "plan.json"),
        "record_hashes": record_hashes,
        "selected_attempts": selected,
        "attempts": len(attempted),
        "fallback_attempts": sum(record["output_limit"] == previous.FALLBACK_TOKENS for _, record in attempted),
        "call_files_sha256": {
            str(path.relative_to(run)): file_hash(path) for path in sorted((run / "calls").rglob("*")) if path.is_file()
        },
        "intent_hashes": {path.name: file_hash(path) for path in sorted((run / "intents").glob("*.json"))},
        "missing_ids": sorted(set(rows) - set(record_hashes)),
        "stop_reason": stop,
        "list_price_estimate_usd": float(invoice),
        "conservative_list_price_usd": str(budget.settled),
        "prior_unknown_exposure_usd": plan["prior_unknown_exposure_usd"],
        "unknown_exposure_usd": str(unknown),
        "invoice_estimate_complete": unknown == 0,
        "budget": budget.receipt(),
        "elapsed_seconds": time.monotonic() - started,
    }
    write(run / "collection.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("prepare")
    stage.add_argument("--previous", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    launch = sub.add_parser("collect")
    launch.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.previous, args.output)
        print(
            json.dumps(
                {
                    "reused": len(value["reused_ids"]),
                    "pending": len(value["pending"]),
                    "exhausted": value["previously_exhausted_ids"],
                }
            )
        )
    else:
        value = asyncio.run(collect(args.run))
        print(json.dumps({key: value[key] for key in ("status", "complete", "list_price_estimate_usd", "budget")}))


if __name__ == "__main__":
    main()
