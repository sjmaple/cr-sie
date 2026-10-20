"""Question-only reference audit across a pinned public benchmark.

Reuse completed teacher answers unchanged; no gold or solver outputs enter any
request. This separate acquisition does not fit routers or open solver test data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import time
from decimal import Decimal
from pathlib import Path

import httpx

import teacher_collect
import teacher_continue as previous
import teacher_recover as recovery
from data import SOURCE_HASHES, SOURCE_REVISION, digest, file_hash, render_request, write

FORMAT = "sie-benchmark-question-only-reference-audit-v1"
COUNTS = {"train": 2560, "calibration": 246, "test": 479}
CODE_FILES = (
    "teacher_reference_collect.py",
    "teacher_recover.py",
    "teacher_continue.py",
    "teacher_collect.py",
    "data.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in CODE_FILES}


def verify_public_inputs(inputs, source):
    import pyarrow.parquet as pq

    if {name: len(rows) for name, rows in inputs.items()} != COUNTS:
        raise ValueError("Reference audit population changed")
    originals = {}
    for filename, expected in SOURCE_HASHES.items():
        path = Path(source) / filename
        if file_hash(path) != expected:
            raise ValueError("Pinned public source hash changed")
        originals[filename.split(".")[0]] = pq.read_table(path, columns=["problem", "tag"]).to_pylist()
    rows, identities = [], set()
    for split in COUNTS:
        for row in inputs[split]:
            match = re.fullmatch(r"e2h-amc-(train|eval)-(0|[1-9]\d*)", row.get("id", ""))
            if set(row) != {"id", "request"} or match is None or row["id"] in identities:
                raise ValueError("Inputs must be unique public IDs and question-only text")
            original = originals[match[1]][int(match[2])]
            if row["request"] != render_request(original["tag"], original["problem"]):
                raise ValueError("Request differs from exact pinned public question")
            rows.append(row)
            identities.add(row["id"])
    return rows


def prepare(training_inputs, held_inputs, source, reuse, output, workers=16):
    output, reuse = Path(output), Path(reuse)
    if output.exists():
        raise FileExistsError(output)
    if not 16 <= workers <= 32:
        raise ValueError("Use the authorized 16–32 workers")
    training, held = read(training_inputs), read(held_inputs)
    inputs = {"train": training["train"], "calibration": held["calibration"], "test": held["test"]}
    rows = verify_public_inputs(inputs, source)
    _, old_rows = recovery.checked_plan(reuse)
    summary = read(reuse / "collection.json")
    if (
        summary["status"] != "complete"
        or summary["complete"] != 510
        or summary["plan_sha256"] != file_hash(reuse / "plan.json")
    ):
        raise ValueError("Reuse requires all510 completed historical teacher answers")
    by_id = {row["id"]: row for row in rows}
    for identity, row in old_rows.items():
        path = reuse / "records" / f"{identity}.json"
        if by_id.get(identity) != row or file_hash(path) != summary["record_hashes"][identity]:
            raise ValueError("Reused teacher question or answer changed")
        previous.check_record(read(path), row)
    new_ids = sorted(set(by_id) - set(old_rows), key=lambda identity: digest(["reference-audit-order-v1", identity]))
    if len(new_ids) != 2775:
        raise ValueError("Expected exactly2775 new question-only calls")
    plan = {
        "format": FORMAT,
        "created_unix": time.time(),
        "n": len(rows),
        "counts": COUNTS,
        "inputs": rows,
        "inputs_by_split_sha256": digest(inputs),
        "settings": teacher_collect.SETTINGS,
        "fallback_settings": teacher_collect.SETTINGS | {"max_output_tokens": previous.FALLBACK_TOKENS},
        "model": teacher_collect.MODEL,
        "source_revision": SOURCE_REVISION,
        "source_hashes": SOURCE_HASHES,
        "public_source_url": "https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/002087323da76e925235e0c6acb12f402bf8fd9e/evaluations/2026-10-17/request-routing-difficulty-policy-v1/source",
        "source_inputs_sha256": {"training": file_hash(training_inputs), "held_questions_only": file_hash(held_inputs)},
        "reuse_plan_sha256": file_hash(reuse / "plan.json"),
        "reuse_collection_sha256": file_hash(reuse / "collection.json"),
        "reuse_files_sha256": {
            str(path.relative_to(reuse)): file_hash(path) for path in sorted(reuse.rglob("*")) if path.is_file()
        },
        "reused_ids": list(old_rows),
        "pending": [
            {"id": identity, "output_limit": previous.INITIAL_TOKENS, "transient_failures": 0} for identity in new_ids
        ],
        "previously_exhausted_ids": [],
        "historical_list_price_estimate_usd": str(summary["list_price_estimate_usd"]),
        "historical_conservative_usd": summary["conservative_list_price_usd"],
        "prior_unknown_exposure_usd": summary["unknown_exposure_usd"],
        "max_additional_usd": "70",
        "max_usd": str(Decimal(summary["conservative_list_price_usd"]) + Decimal(70)),
        "max_seconds": 5400,
        "workers": workers,
        "request_timeout_seconds": previous.REQUEST_SECONDS,
        "maximum_transient_retries_per_question": recovery.MAX_TRANSIENT_RETRIES,
        "backoff_seconds": [2, 4],
        "selection": "First schema-valid completed answer by attempt order, never by correctness",
        "purpose": "Independently audit reference-answer correctness before interpreting one-percentage-point quality gaps. Reference review sees public questions and benchmark solutions, never routed solver test answers.",
        "teacher_receives_gold": False,
        "teacher_receives_solver_answers": False,
        "solver_test_outcomes_accessed": False,
        "code_sha256": code_hashes(),
    }
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    write(output / "inputs.json", inputs)
    shutil.copytree(reuse, output / "previous")
    (output / "records").mkdir()
    for identity in old_rows:
        shutil.copyfile(reuse / "records" / f"{identity}.json", output / "records" / f"{identity}.json")
    return plan


def checked_plan(run):
    run = Path(run)
    plan, inputs = read(run / "plan.json"), read(run / "inputs.json")
    if (
        plan.get("format") != FORMAT
        or plan["code_sha256"] != code_hashes()
        or plan["settings"] != teacher_collect.SETTINGS
        or plan["fallback_settings"] != teacher_collect.SETTINGS | {"max_output_tokens": previous.FALLBACK_TOKENS}
        or plan["counts"] != COUNTS
        or plan["n"] != sum(COUNTS.values())
        or plan["inputs_by_split_sha256"] != digest(inputs)
        or plan["inputs"] != [row for split in COUNTS for row in inputs[split]]
        or plan["source_hashes"] != SOURCE_HASHES
        or plan["source_revision"] != SOURCE_REVISION
        or plan["max_additional_usd"] != "70"
        or plan["max_seconds"] != 5400
        or not 16 <= plan["workers"] <= 32
        or plan["teacher_receives_gold"] is not False
        or plan["teacher_receives_solver_answers"] is not False
        or plan["solver_test_outcomes_accessed"] is not False
    ):
        raise ValueError("Reference audit protocol changed")
    for name, expected in plan["reuse_files_sha256"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or file_hash(run / "previous" / path) != expected:
            raise ValueError("Preserved teacher lineage changed")
    _, old_rows = recovery.checked_plan(run / "previous")
    summary = read(run / "previous/collection.json")
    rows = {row["id"]: row for row in plan["inputs"]}
    if (
        plan["reused_ids"] != list(old_rows)
        or len(rows) != plan["n"]
        or {state["id"] for state in plan["pending"]} != set(rows) - set(old_rows)
        or len(plan["pending"]) != 2775
        or plan["historical_conservative_usd"] != summary["conservative_list_price_usd"]
        or plan["max_usd"] != str(Decimal(summary["conservative_list_price_usd"]) + Decimal(70))
        or plan["prior_unknown_exposure_usd"] != summary["unknown_exposure_usd"]
    ):
        raise ValueError("Reference population, reuse or budget changed")
    for identity, row in old_rows.items():
        path = run / "records" / f"{identity}.json"
        if rows[identity] != row or file_hash(path) != summary["record_hashes"][identity]:
            raise ValueError("Historical answer bytes changed")
        previous.check_record(read(path), row)
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
                action, amended = recovery.next_state(state, record)
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
        "status": "complete" if len(record_hashes) == plan["n"] else "incomplete",
        "n": plan["n"],
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
    stage.add_argument("--training-inputs", type=Path, required=True)
    stage.add_argument("--held-inputs", type=Path, required=True)
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument("--reuse", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--workers", type=int, default=16)
    launch = sub.add_parser("collect")
    launch.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.training_inputs, args.held_inputs, args.source, args.reuse, args.output, args.workers)
        print(
            json.dumps(
                {
                    "n": value["n"],
                    "reused": len(value["reused_ids"]),
                    "new": len(value["pending"]),
                    "plan_sha256": file_hash(args.output / "plan.json"),
                }
            )
        )
    else:
        value = asyncio.run(collect(args.run))
        print(json.dumps({key: value[key] for key in ("status", "complete", "list_price_estimate_usd", "budget")}))


if __name__ == "__main__":
    main()
