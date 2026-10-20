"""Collect independent teacher answers for question-only training inputs.

The teacher never receives benchmark answers or either answer model's output.
Completed answers are reused on resume. Transport failures have a fixed retry
limit; incorrect or unparseable answers are never resampled.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import re
import time
from decimal import Decimal
from pathlib import Path

import httpx

import arithmetic_teacher_protocol as numeric_teacher
import teacher_collect as teacher
from data import digest, file_hash, write

FORMAT = "sie-arithmetic-teacher-v1"
MAX_ATTEMPTS = 3
RETRY_KINDS = {"ConnectError", "ReadError", "WriteError", "ReadTimeout", "WriteTimeout", "ConnectTimeout", "TimeoutError"}


def read(path):
    return json.loads(Path(path).read_text())


def questions(path):
    rows = read(path)
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, dict)
            or set(row) != {"id", "request"}
            or not isinstance(row["id"], str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,120}", row["id"]) is None
            or not isinstance(row["request"], str)
            or not row["request"].strip()
            for row in rows
        )
        or len({row["id"] for row in rows}) != len(rows)
    ):
        raise ValueError("Expected unique question-only {id, request} rows; no labels or other fields")
    return rows


def protocol_for(answer_format):
    if answer_format == "numeric":
        return numeric_teacher
    if answer_format == "general":
        return teacher
    raise ValueError("Unknown teacher answer format")


def source_hashes():
    return {
        "arithmetic_teacher.py": file_hash(Path(__file__)),
        "arithmetic_teacher_protocol.py": file_hash(Path(numeric_teacher.__file__)),
        "teacher_collect.py": file_hash(Path(teacher.__file__)),
    }


def prepare(inputs, output, *, max_usd=30, workers=16, answer_format="numeric"):
    inputs, output = Path(inputs), Path(output)
    rows = questions(inputs)
    budget = Decimal(str(max_usd))
    if not budget.is_finite() or budget <= 0 or not 1 <= workers <= 32:
        raise ValueError("Use a finite positive budget and 1..32 workers")
    if output.exists():
        raise FileExistsError(output)
    protocol = protocol_for(answer_format)
    plan = {
        "format": FORMAT,
        "input_sha256": file_hash(inputs),
        "questions_sha256": digest(rows),
        "ids": [row["id"] for row in rows],
        "answer_format": answer_format,
        "settings": protocol.SETTINGS,
        "prices_usd_per_million": teacher.PRICES,
        "price_source": "https://developers.openai.com/api/docs/pricing",
        "price_checked_date": "2026-10-19",
        "max_usd": str(budget),
        "workers": workers,
        "max_attempts_per_question": MAX_ATTEMPTS,
        "request_timeout_seconds": teacher.REQUEST_SECONDS,
        "retry_policy": "Up to three identical attempts for 429/5xx or transport failure; exponential backoff. Never retry answer content, refusals, or output-token exhaustion.",
        "question_only": True,
        "reference_answers_received": False,
        "solver_answers_received": False,
        "source_sha256": source_hashes(),
    }
    write(output / "plan.json", plan)
    write(output / "inputs.json", rows)
    return plan


def retryable(record):
    error = record.get("error") or {}
    return error.get("kind") in RETRY_KINDS or (
        error.get("kind") == "http_error" and (error.get("status") == 429 or 500 <= error.get("status", 0) <= 599)
    )


def attempt_history(run, row, protocol=None):
    protocol = protocol or protocol_for(read(run / "plan.json").get("answer_format", "general"))
    result = []
    for directory in sorted((run / "attempts" / row["id"]).glob("attempt-*")):
        intent = read(directory / "intent.json")
        if intent.get("request_sha256") != digest(protocol.body_for(row)):
            raise ValueError("A saved teacher attempt belongs to a different question or settings")
        path = directory / "records" / f"{row['id']}.json"
        record = read(path) if path.exists() else None
        if record and record.get("request_sha256") != intent["request_sha256"]:
            raise ValueError("Saved request and response bindings differ")
        if record and record.get("raw_response_sha256"):
            raw = directory / "responses" / f"{row['id']}.json"
            if not raw.is_file() or file_hash(raw) != record["raw_response_sha256"]:
                raise ValueError("Saved raw teacher response is missing or changed")
        if record and record.get("status") == "complete":
            if not record.get("raw_response_sha256"):
                raise ValueError("Completed teacher answer has no raw response receipt")
            protocol._check_teacher_record(record, row)
        result.append((directory, intent, record))
    return result


def save_summary(run, result):
    history = run / "summaries"
    number = len(list(history.glob("summary-*.json"))) + 1
    write(history / f"summary-{number:04d}.json", result)
    temporary = run / ".collection.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(run / "collection.json")


async def collect(run, *, secret=None, request_fn=teacher._http_request, checkpoint=None):
    run = Path(run)
    plan, rows = read(run / "plan.json"), questions(run / "inputs.json")
    protocol = protocol_for(plan.get("answer_format", "general"))
    if (
        plan.get("format") != FORMAT
        or plan.get("questions_sha256") != digest(rows)
        or plan.get("ids") != [row["id"] for row in rows]
        or plan.get("settings") != protocol.SETTINGS
        or plan.get("max_attempts_per_question") != MAX_ATTEMPTS
        or plan.get("source_sha256") != source_hashes()
    ):
        raise ValueError("Prepared teacher inputs, source or settings changed")
    secret = secret if secret is not None else os.environ["OPENAI_API_KEY"]
    budget = teacher.Budget(plan["max_usd"], 0)
    histories, completed, terminal = {}, {}, {}
    for row in rows:
        history = attempt_history(run, row, protocol)
        histories[row["id"]] = history
        for directory, intent, record in history:
            charge = (
                Decimal(record["conservative_list_price_usd"])
                if record and record.get("usage_verified")
                else Decimal(intent["reservation_usd"])
            )
            budget.settled += charge
            if record and record["status"] == "complete":
                completed[row["id"]] = str(directory.relative_to(run))
        if history and row["id"] not in completed:
            last = history[-1][2]
            if last is None or not retryable(last) or len(history) >= MAX_ATTEMPTS:
                terminal[row["id"]] = "uncertain_interrupted_attempt" if last is None else "no_more_retries"
    if budget.exposure > budget.maximum:
        raise ValueError("Historical exposure already exceeds the saved budget")
    started, counters = time.monotonic(), {"new_attempts": 0, "finished_questions": len(completed) + len(terminal)}
    queue = asyncio.Queue()
    for row in rows:
        if row["id"] not in completed and row["id"] not in terminal:
            queue.put_nowait(row)
    lock = asyncio.Lock()

    async def persist():
        if checkpoint is not None:
            result = checkpoint()
            if inspect.isawaitable(result):
                await result

    def summary():
        known = Decimal(0)
        attempts = 0
        for row in rows:
            for _, _, record in attempt_history(run, row, protocol):
                attempts += 1
                if record and record.get("usage_verified"):
                    known += Decimal(str(record["list_price_estimate_usd"]))
        return {
            "format": FORMAT,
            "plan_sha256": file_hash(run / "plan.json"),
            "n": len(rows),
            "complete": len(completed),
            "status": "complete" if len(completed) == len(rows) else "incomplete",
            "completed_attempts": completed,
            "unavailable": terminal,
            "pending": [row["id"] for row in rows if row["id"] not in completed and row["id"] not in terminal],
            "known_list_price_usd": str(known),
            "budget": budget.receipt(),
            "physical_attempts": attempts,
            "new_attempts": counters["new_attempts"],
            "elapsed_seconds_this_run": time.monotonic() - started,
            "cost_scope": "Known token usage at pinned list prices plus separate conservative unknown-attempt exposure; not an invoice.",
            "question_only": True,
        }

    async with httpx.AsyncClient(timeout=teacher.REQUEST_SECONDS, follow_redirects=False, trust_env=False) as client:

        async def worker():
            while not queue.empty():
                row = queue.get_nowait()
                identity = row["id"]
                for number in range(len(histories[identity]) + 1, MAX_ATTEMPTS + 1):
                    reservation = protocol.reservation(row)
                    async with lock:
                        if not budget.reserve(identity, reservation):
                            break
                        destination = run / "attempts" / identity / f"attempt-{number:02d}"
                        write(
                            destination / "intent.json",
                            {
                                "id": identity,
                                "number": number,
                                "request_sha256": digest(protocol.body_for(row)),
                                "reservation_usd": str(reservation),
                            },
                        )
                        counters["new_attempts"] += 1
                    # A remote checkpoint must finish before this request can be
                    # sent, so a restart can identify uncertain executions.
                    await persist()
                    if number > 1:
                        await asyncio.sleep(2**number)
                    record = await protocol._call(destination, row, secret, client, request_fn)
                    async with lock:
                        budget.settle(identity, record.get("conservative_list_price_usd"))
                    if record["status"] == "complete":
                        completed[identity] = str(destination.relative_to(run))
                        break
                    if not retryable(record) or number == MAX_ATTEMPTS:
                        terminal[identity] = "no_more_retries"
                        break
                counters["finished_questions"] += 1
                if counters["finished_questions"] % 100 == 0:
                    async with lock:
                        progress = summary()
                        save_summary(run, progress)
                        print(
                            f"Teacher: {len(completed)}/{len(rows)} complete; "
                            f"known usage ${progress['known_list_price_usd']}; exposure ${budget.exposure}",
                            flush=True,
                        )
                    await persist()

        await asyncio.gather(*(worker() for _ in range(plan["workers"])))
    result = summary()
    save_summary(run, result)
    await persist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect"))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--max-usd", type=Decimal, default=Decimal(30))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--answer-format", choices=("numeric", "general"), default="numeric")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.inputs is None:
            parser.error("prepare requires --inputs")
        result = prepare(
            args.inputs, args.run, max_usd=args.max_usd, workers=args.workers, answer_format=args.answer_format
        )
    else:
        result = asyncio.run(collect(args.run))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
