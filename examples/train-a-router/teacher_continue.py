"""Complete a saved teacher run with one declared output-budget fallback.

Completed answers stay byte-for-byte unchanged. Only a known max-output
incomplete answer may get one fresh question-only attempt at32768 tokens;
unknown failures stop new dispatch. Historical charges and unknown reservations
remain inside the40USD study cap. This is an explicit protocol amendment.
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

import teacher_collect as original
from data import digest, file_hash, write

FORMAT = "sie-teacher-reference-completion-v1"
INITIAL_TOKENS, FALLBACK_TOKENS = 8192, 32768
MAX_USD, MAX_SECONDS, REQUEST_SECONDS = Decimal(40), 3600, 600


def read(path):
    return json.loads(Path(path).read_text())


def body_for(row, output_limit):
    if output_limit not in (INITIAL_TOKENS, FALLBACK_TOKENS):
        raise ValueError("Unsupported teacher output budget")
    return original.body_for(row) | {"max_output_tokens": output_limit}


def allowance(row, output_limit):
    return len(json.dumps(body_for(row, output_limit)).encode()) + 2048


def reservation(row, output_limit):
    return (Decimal(allowance(row, output_limit)) * Decimal("12.5") + Decimal(output_limit * 50)) / 1_000_000


def usage_cost(usage, row, output_limit):
    if not isinstance(usage, dict):
        raise TypeError("Missing usage")
    details = usage.get("input_tokens_details") or {}
    counts = [
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        details.get("cached_tokens", 0),
        details.get("cache_write_tokens", 0),
    ]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Invalid usage")
    incoming, outgoing, cached, written = counts
    if incoming > allowance(row, output_limit) or outgoing > output_limit or cached + written > incoming:
        raise ValueError("Usage exceeds reservation")
    invoice = (
        Decimal(incoming - cached - written) * 10
        + Decimal(cached)
        + Decimal(written) * Decimal("12.5")
        + Decimal(outgoing * 50)
    ) / 1_000_000
    upper = (Decimal(incoming - cached) * Decimal("12.5") + Decimal(cached) + Decimal(outgoing * 50)) / 1_000_000
    return invoice, upper


def check_record(record, row):
    limit = record.get("output_limit", INITIAL_TOKENS)
    if (
        record.get("id") != row["id"]
        or record.get("request_sha256") != digest(body_for(row, limit))
        or record.get("status") != "complete"
        or record.get("response_status") != "completed"
        or record.get("returned_model") != original.MODEL
        or not isinstance(record.get("answer"), str)
        or json.loads(record["output_text"]) != {"answer": record["answer"]}
    ):
        raise ValueError("Teacher completion record changed")
    return usage_cost(record["usage"], row, limit)


def known_exhaustion(record):
    return (
        record.get("usage_verified") is True
        and record.get("response_status") == "incomplete"
        and record.get("incomplete_details") == {"reason": "max_output_tokens"}
    )


def prepare(previous, design_path, output):
    previous, output = Path(previous), Path(output)
    if output.exists():
        raise FileExistsError(output)
    old, rows = original.checked_plan(previous)
    summary = read(previous / "collection.json")
    design = read(design_path)
    if (
        summary["plan_sha256"] != file_hash(previous / "plan.json")
        or summary["status"] != "incomplete"
        or design["teacher_acquisition_plan_sha256"] != old["superseded_attempt"]["plan_sha256"]
    ):
        raise ValueError("Completion must extend the designed, preserved acquisition")
    reused, exhausted = [], []
    for identity, expected in summary["record_hashes"].items():
        path = previous / "records" / f"{identity}.json"
        if file_hash(path) != expected:
            raise ValueError("Previous teacher record changed")
        record = read(path)
        if record["status"] == "complete":
            check_record(record, rows[identity])
            reused.append(identity)
        elif known_exhaustion(record):
            exhausted.append(identity)
        else:
            raise ValueError("Unknown execution cannot be retried by this completion protocol")
    pending = [identity for identity in old["new_ids"] if identity not in set(reused)]
    hashes = {
        str(path.relative_to(previous)): file_hash(path) for path in sorted(previous.rglob("*")) if path.is_file()
    }
    plan = {
        "format": FORMAT,
        "created_unix": time.time(),
        "n": 510,
        "inputs": old["inputs"],
        "settings": original.SETTINGS,
        "model": original.MODEL,
        "fallback_settings": original.SETTINGS | {"max_output_tokens": FALLBACK_TOKENS},
        "origin_plan_sha256": design["teacher_acquisition_plan_sha256"],
        "comparison_design_sha256": file_hash(design_path),
        "previous_plan_sha256": file_hash(previous / "plan.json"),
        "previous_files_sha256": hashes,
        "reused_ids": reused,
        "new_ids": pending,
        "prior_exhausted_ids": exhausted,
        "historical_list_price_estimate_usd": str(summary["list_price_estimate_usd"]),
        "historical_conservative_usd": summary["conservative_list_price_usd"],
        "prior_unknown_exposure_usd": summary.get("prior_unknown_exposure_usd", "0"),
        "max_usd": str(MAX_USD),
        "max_seconds": MAX_SECONDS,
        "workers": 4,
        "request_timeout_seconds": REQUEST_SECONDS,
        "automatic_retries": 0,
        "protocol_amendment": "Preserve all135 complete answers; continue untouched requests at8192. A known max-output incomplete may receive exactly one fresh question-only32768 attempt. No reference answers, solver outputs or prior reasoning enter that request. Unknown/429 errors stop new dispatch; exhausted fallback remains missing. Learner/folds/calibration unchanged; declared before label audit.",
        "teacher_receives_gold": False,
        "teacher_receives_solver_answers": False,
        "operator_sha256": file_hash(Path(__file__)),
        "teacher_collect_sha256": file_hash(Path(original.__file__)),
        "data_source_sha256": file_hash(Path(__file__).with_name("data.py")),
    }
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    shutil.copytree(previous, output / "previous")
    shutil.copyfile(design_path, output / "comparison-design.json")
    (output / "records").mkdir()
    for identity in reused:
        shutil.copyfile(previous / "records" / f"{identity}.json", output / "records" / f"{identity}.json")
    return plan


def checked_plan(run):
    run = Path(run)
    plan = read(run / "plan.json")
    if (
        plan.get("format") != FORMAT
        or plan["settings"] != original.SETTINGS
        or plan["fallback_settings"] != original.SETTINGS | {"max_output_tokens": FALLBACK_TOKENS}
        or plan["operator_sha256"] != file_hash(Path(__file__))
        or plan["teacher_collect_sha256"] != file_hash(Path(original.__file__))
        or plan["data_source_sha256"] != file_hash(Path(__file__).with_name("data.py"))
        or plan["max_usd"] != str(MAX_USD)
        or plan["max_seconds"] != MAX_SECONDS
        or plan["workers"] != 4
        or plan["teacher_receives_gold"] is not False
        or plan["teacher_receives_solver_answers"] is not False
        or file_hash(run / "comparison-design.json") != plan["comparison_design_sha256"]
    ):
        raise ValueError("Teacher completion source or protocol changed")
    for name, expected in plan["previous_files_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or file_hash(run / "previous" / name) != expected:
            raise ValueError("Preserved teacher acquisition changed")
    old, _ = original.checked_plan(run / "previous")
    summary = read(run / "previous/collection.json")
    rows = {row["id"]: row for row in plan["inputs"]}
    if (
        plan["inputs"] != old["inputs"]
        or len(rows) != 510
        or set(plan["reused_ids"]) & set(plan["new_ids"])
        or set(rows) != set(plan["reused_ids"]) | set(plan["new_ids"])
    ):
        raise ValueError("Teacher completion population changed")
    if plan["historical_conservative_usd"] != summary["conservative_list_price_usd"] or plan[
        "historical_list_price_estimate_usd"
    ] != str(summary["list_price_estimate_usd"]):
        raise ValueError("Historical teacher cost changed")
    for identity in plan["reused_ids"]:
        path = run / "records" / f"{identity}.json"
        if file_hash(path) != summary["record_hashes"][identity]:
            raise ValueError("Reused answer bytes changed")
        check_record(read(path), rows[identity])
    return plan, rows


async def call(run, row, limit, secret, client, request_fn):
    record = {
        "id": row["id"],
        "output_limit": limit,
        "request_sha256": digest(body_for(row, limit)),
        "status": "incomplete",
        "answer": None,
        "usage_verified": False,
    }
    started = time.time()
    try:
        async with asyncio.timeout(REQUEST_SECONDS):
            response = await request_fn(client, body_for(row, limit), secret, REQUEST_SECONDS)
        if "error" in response:
            record["error"] = response["error"]
        else:
            payload = response["payload"]
            raw = run / "responses" / f"{row['id']}-{limit}.json"
            write(raw, payload)
            record.update(
                {
                    "raw_response_sha256": file_hash(raw),
                    "response_id": payload.get("id"),
                    "returned_model": payload.get("model"),
                    "response_status": payload.get("status"),
                    "usage": payload.get("usage"),
                    "incomplete_details": payload.get("incomplete_details"),
                }
            )
            if payload.get("model") != original.MODEL or payload.get("service_tier") not in (None, "default"):
                raise ValueError("Teacher model or pricing tier changed")
            invoice, upper = usage_cost(payload.get("usage"), row, limit)
            record.update(
                {
                    "usage_verified": True,
                    "list_price_estimate_usd": str(invoice),
                    "conservative_list_price_usd": str(upper),
                }
            )
            output = "".join(
                content.get("text", "")
                for item in payload.get("output", [])
                if item.get("type") == "message"
                for content in item.get("content", [])
                if content.get("type") == "output_text"
            )
            record["output_text"] = output
            parsed = json.loads(output)
            if not isinstance(parsed, dict) or set(parsed) != {"answer"} or not isinstance(parsed["answer"], str):
                raise ValueError("Teacher output schema changed")
            record["answer"] = parsed["answer"]
            if payload.get("status") == "completed":
                record["status"] = "complete"
    except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001 — retain every charged/unknown attempt.
        record["error"] = {"kind": type(error).__name__, "execution_unknown": not record["usage_verified"]}
    record["elapsed_seconds"] = time.time() - started
    write(run / "attempts" / f"{row['id']}-{limit}.json", record)
    return record


async def collect(run, secret=None, request_fn=original._http_request):
    run = Path(run)
    plan, rows = checked_plan(run)
    if any((run / name).exists() for name in ("started.json", "collection.json", "intents", "attempts")):
        raise FileExistsError("Teacher completion already attempted; no automatic resume")
    secret = secret or os.environ.get("OPENAI_API_KEY")
    if not secret:
        raise ValueError("Set OPENAI_API_KEY")
    budget = original.Budget(plan["max_usd"], plan["historical_conservative_usd"])
    pending = [
        (identity, FALLBACK_TOKENS if identity in plan["prior_exhausted_ids"] else INITIAL_TOKENS)
        for identity in plan["new_ids"]
    ]
    active, successful, attempts, missing, stop = {}, [], [], [], None
    started = time.monotonic()
    write(
        run / "started.json",
        {"plan_sha256": file_hash(run / "plan.json"), "started_unix": time.time(), "budget": budget.receipt()},
    )
    transport = httpx.AsyncHTTPTransport(retries=0, trust_env=False)
    async with httpx.AsyncClient(
        timeout=REQUEST_SECONDS, transport=transport, follow_redirects=False, trust_env=False
    ) as client:
        while pending or active:
            while pending and stop is None and len(active) < plan["workers"]:
                if time.monotonic() - started > plan["max_seconds"] - REQUEST_SECONDS - 5:
                    stop = "deadline"
                    break
                identity, limit = pending[0]
                key = f"{identity}-{limit}"
                if not budget.reserve(key, reservation(rows[identity], limit)):
                    if not active:
                        stop = "budget"
                    break
                pending.pop(0)
                write(
                    run / "intents" / f"{key}.json",
                    {
                        "id": identity,
                        "output_limit": limit,
                        "plan_sha256": file_hash(run / "plan.json"),
                        "request_sha256": digest(body_for(rows[identity], limit)),
                        "reservation_usd": str(reservation(rows[identity], limit)),
                        "budget_after_reservation": budget.receipt(),
                        "started_unix": time.time(),
                    },
                )
                task = asyncio.create_task(call(run, rows[identity], limit, secret, client, request_fn))
                active[task] = (identity, limit, key)
            if not active:
                break
            done, _ = await asyncio.wait(active, timeout=30, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                identity, limit, key = active.pop(task)
                record = await task
                attempts.append(record)
                budget.settle(key, record.get("conservative_list_price_usd") if record["usage_verified"] else None)
                if record["status"] == "complete":
                    shutil.copyfile(run / "attempts" / f"{key}.json", run / "records" / f"{identity}.json")
                    successful.append(identity)
                elif known_exhaustion(record):
                    if limit == INITIAL_TOKENS:
                        pending.insert(0, (identity, FALLBACK_TOKENS))
                    else:
                        missing.append(identity)
                else:
                    missing.append(identity)
                    stop = "request_error_or_unknown_execution"
            progress = {
                "completed_new": len(successful),
                "attempts": len(attempts),
                "pending": len(pending),
                "inflight": len(active),
                "missing": len(missing),
                "stop_reason": stop,
                "budget": budget.receipt(),
                "elapsed_seconds": time.monotonic() - started,
            }
            write(run / "ledger" / f"{len(list((run / 'ledger').glob('*.json'))):05d}.json", progress)
            print(json.dumps(progress), flush=True)
    record_hashes = {
        identity: file_hash(run / "records" / f"{identity}.json") for identity in plan["reused_ids"] + successful
    }
    invoice = Decimal(plan["historical_list_price_estimate_usd"]) + sum(
        (Decimal(record["list_price_estimate_usd"]) for record in attempts if record["usage_verified"]), Decimal(0)
    )
    summary = {
        "status": "complete" if len(record_hashes) == 510 else "incomplete",
        "n": 510,
        "returned": len(record_hashes),
        "complete": len(record_hashes),
        "reused": len(plan["reused_ids"]),
        "plan_sha256": file_hash(run / "plan.json"),
        "record_hashes": record_hashes,
        "attempt_hashes": {path.name: file_hash(path) for path in sorted((run / "attempts").glob("*.json"))},
        "intent_hashes": {path.name: file_hash(path) for path in sorted((run / "intents").glob("*.json"))},
        "fallback_attempts": sum(record["output_limit"] == FALLBACK_TOKENS for record in attempts),
        "missing_ids": sorted(set(rows) - set(record_hashes)),
        "stop_reason": stop,
        "list_price_estimate_usd": float(invoice),
        "conservative_list_price_usd": str(budget.settled),
        "prior_unknown_exposure_usd": plan["prior_unknown_exposure_usd"],
        "budget": budget.receipt(),
        "elapsed_seconds": time.monotonic() - started,
        "automatic_retries": 0,
    }
    write(run / "collection.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("prepare")
    for key in ("previous", "design", "output"):
        stage.add_argument(f"--{key}", type=Path, required=True)
    launch = sub.add_parser("collect")
    launch.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.previous, args.design, args.output)
        print(json.dumps({"n": result["n"], "reused": len(result["reused_ids"]), "pending": len(result["new_ids"])}))
    else:
        result = asyncio.run(collect(args.run))
        print(
            json.dumps({key: result[key] for key in ("status", "n", "complete", "list_price_estimate_usd", "budget")})
        )


if __name__ == "__main__":
    main()
