"""Stage and collect a bounded question-only extension of a saved teacher pilot.

``prepare`` makes no API calls and never opens benchmark gold or solver answers.
``collect`` is a single attempt: it refuses an existing start receipt, preserves
every reservation/response, stops new dispatch on any error, and never retries.
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

from data import digest, file_hash, write

MODEL = "gpt-6-astra"
PRICES = {"input": 10, "cached_input": 1, "input_cache_write": 12.5, "output": 50}
SYSTEM = (
    "Solve the mathematical problem independently. Return its final answer as a concise mathematical expression "
    "or value, with any units or requested rounding, in the answer field. Use LaTeX if necessary; "
    "do not include dollar delimiters or a boxed wrapper. Do not use external tools."
)
SETTINGS = {
    "model": MODEL,
    "instructions": SYSTEM,
    "input": "<question>",
    "reasoning": {"effort": "high"},
    "max_output_tokens": 8192,
    "store": False,
    "service_tier": "default",
    "text": {
        "format": {
            "type": "json_schema",
            "name": "math_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        }
    },
}
MAX_USD = Decimal(25)
MAX_SECONDS = 3600
MAX_WORKERS = 4
REQUEST_SECONDS = 300
ENDPOINT = "https://api.openai.com/v1/responses"


def read(path):
    return json.loads(Path(path).read_text())


def body_for(row):
    return SETTINGS | {"input": row["request"]}


def input_allowance(row):
    # Text-only UTF-8 bytes bound tokenization; the extra allowance covers the
    # request/schema framing. Requests stay below the long-context price tier.
    allowance = len(json.dumps(body_for(row)).encode()) + 2048
    if allowance >= 272_000:
        raise ValueError("Teacher input exceeds the declared short-context budget")
    return allowance


def reservation(row):
    return (Decimal(input_allowance(row)) * Decimal("12.5") + Decimal(8192 * 50)) / 1_000_000


def usage_cost(usage, allowance):
    """Return invoice estimate and conservative charge only for verified usage."""
    if not isinstance(usage, dict):
        raise TypeError("Missing usage")
    details = usage.get("input_tokens_details") or {}
    values = [
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        details.get("cached_tokens", 0),
        details.get("cache_write_tokens", 0),
    ]
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Invalid token usage")
    incoming, outgoing, cached, written = values
    if cached + written > incoming or incoming > allowance or outgoing > 8192:
        raise ValueError("Usage exceeds the reserved request bounds")
    invoice = (
        Decimal(incoming - cached - written) * 10
        + Decimal(cached)
        + Decimal(written) * Decimal("12.5")
        + Decimal(outgoing) * 50
    ) / 1_000_000
    conservative = (Decimal(incoming - cached) * Decimal("12.5") + Decimal(cached) + Decimal(outgoing) * 50) / 1_000_000
    return invoice, conservative


def _check_teacher_record(record, row):
    if (
        record.get("id") != row["id"]
        or record.get("request_sha256") != digest(body_for(row))
        or record.get("status") != "complete"
        or record.get("response_status") != "completed"
        or record.get("returned_model") != MODEL
        or not isinstance(record.get("answer"), str)
        or json.loads(record["output_text"]) != {"answer": record["answer"]}
    ):
        raise ValueError("Reused teacher record differs from the exact question-only request")
    return usage_cost(record.get("usage"), input_allowance(row))


def superseded_transport(path):
    """Bind an explicitly authorized single repeat after a connection-only stop."""
    path = Path(path)
    plan, summary = read(path / "plan.json"), read(path / "collection.json")
    if (
        plan.get("superseded_attempt") is not None
        or plan.get("settings") != SETTINGS
        or summary.get("status") != "incomplete"
        or summary.get("stop_reason") != "request_error_or_unknown_execution"
        or summary.get("plan_sha256") != file_hash(path / "plan.json")
        or summary.get("complete") != 64
        or summary.get("dispatched_new") != 4
        or file_hash(path / "source-executed/teacher_collect.py") != plan["operator_sha256"]
    ):
        raise ValueError("Supersession requires the preserved four-call connection-only stop")
    exposure = Decimal(0)
    repeated = plan["new_ids"][:4]
    for identity in repeated:
        record_path = path / "records" / f"{identity}.json"
        intent_path = path / "intents" / f"{identity}.json"
        record, intent = read(record_path), read(intent_path)
        if (
            file_hash(record_path) != summary["record_hashes"][identity]
            or file_hash(intent_path) != summary["intent_hashes"][identity]
            or record.get("status") != "incomplete"
            or record.get("error") != {"kind": "ConnectError", "execution_unknown": True}
            or record.get("answer") is not None
            or record.get("usage_verified") is not False
            or "raw_response_sha256" in record
            or intent["plan_sha256"] != summary["plan_sha256"]
            or intent["reservation_usd"] != plan["request_reservations_usd"][identity]
        ):
            raise ValueError("Only preserved unknown connection failures may be explicitly repeated once")
        exposure += Decimal(intent["reservation_usd"])
    return {
        "files_sha256": {
            str(item.relative_to(path)): file_hash(item) for item in sorted(path.rglob("*")) if item.is_file()
        },
        "plan_sha256": file_hash(path / "plan.json"),
        "unknown_exposure_usd": str(exposure),
        "explicitly_repeated_ids": repeated,
        "reason": "The user authorized one network-enabled repeat after sandbox ConnectError; previous unknown reservations remain charged against the new total cap. No automatic retry.",
    }


def prepare(training, reuse, output, max_usd=Decimal(15), workers=MAX_WORKERS, max_seconds=1800, supersedes=None):
    training, reuse, output = Path(training), Path(reuse), Path(output)
    if output.exists():
        raise FileExistsError(output)
    max_usd = Decimal(str(max_usd))
    if (
        not 0 < max_usd <= MAX_USD
        or not 1 <= workers <= MAX_WORKERS
        or not REQUEST_SECONDS + 5 < max_seconds <= MAX_SECONDS
    ):
        raise ValueError("Budget, concurrency or duration exceeds the authorized limits")
    source_plan, inputs, eligibility = (
        read(training / name) for name in ("plan.json", "inputs.json", "eligibility.json")
    )
    rows = inputs.get("train", [])
    identities = [row["id"] for row in rows]
    if (
        source_plan.get("population") != "original"
        or source_plan.get("counts") != {"train": 510}
        or source_plan.get("source_counts") != {"train": 512}
        or set(inputs) != {"train"}
        or len(rows) != 510
        or len(set(identities)) != 510
        or any(set(row) != {"id", "request"} for row in rows)
        or any(not isinstance(row["request"], str) or not row["request"].strip() for row in rows)
        or any(re.fullmatch(r"e2h-amc-(?:train|eval)-(?:0|[1-9]\d*)", row["id"]) is None for row in rows)
        or source_plan["inputs_sha256"] != digest(inputs)
        or source_plan["eligibility_sha256"] != file_hash(training / "eligibility.json")
        or eligibility.get("source_only") is not True
        or eligibility["eligible_train_ids"] != identities
        or source_plan["eligible_train_ids"] != identities
    ):
        raise ValueError("Need the declared clean510 question-only training inputs")
    old_plan, old_collection = read(reuse / "plan.json"), read(reuse / "collection.json")
    reused_ids = [row["id"] for row in old_plan["inputs"]]
    if (
        old_plan.get("settings") != SETTINGS
        or old_plan.get("prices_usd_per_million") != PRICES
        or old_plan.get("teacher_receives_gold") is not False
        or old_plan.get("teacher_receives_solver_answers") is not False
        or old_plan.get("n") != 64
        or len(set(reused_ids)) != 64
        or len(reused_ids) != 64
        or old_collection.get("status") != "complete"
        or old_collection.get("complete") != 64
        or old_collection.get("returned") != 64
        or old_collection.get("n") != 64
        or old_collection["plan_sha256"] != file_hash(reuse / "plan.json")
        or set(old_collection["record_hashes"]) != set(reused_ids)
        or not set(reused_ids) <= set(identities)
    ):
        raise ValueError("Reuse requires the complete64-answer pilot with identical settings")
    by_id = {row["id"]: row for row in rows}
    invoice, conservative = Decimal(0), Decimal(0)
    for row in old_plan["inputs"]:
        path = reuse / "records" / f"{row['id']}.json"
        if row != by_id[row["id"]] or file_hash(path) != old_collection["record_hashes"][row["id"]]:
            raise ValueError("Reused teacher question or record changed")
        actual, upper = _check_teacher_record(read(path), row)
        invoice += actual
        conservative += upper
    if conservative >= max_usd:
        raise ValueError("Historical teacher cost already exhausts the total budget")
    new_ids = sorted(
        set(identities) - set(reused_ids), key=lambda identity: digest(["teacher-extension-order-v1", identity])
    )
    previous = superseded_transport(supersedes) if supersedes is not None else None
    if previous and (
        read(Path(supersedes) / "plan.json")["inputs"] != rows
        or read(Path(supersedes) / "plan.json")["reuse"]["record_hashes"] != old_collection["record_hashes"]
        or conservative + Decimal(previous["unknown_exposure_usd"]) >= max_usd
    ):
        raise ValueError("Superseded teacher population, reused answers or budget changed")
    plan = {
        "format": "sie-teacher-reference-extension-v1",
        "created_unix": time.time(),
        "model": MODEL,
        "n": 510,
        "inputs": rows,
        "new_ids": new_ids,
        "reused_ids": reused_ids,
        "settings": SETTINGS,
        "teacher_receives_gold": False,
        "teacher_receives_solver_answers": False,
        "purpose": "Teacher-label reliability over clean510 training rows and a later embedding-control training-equivalence audit; no new native candidate or model-settings search.",
        "order": "All64 pilot answers reused unchanged; remaining446 use fixed identity-hash order, independent of answers/outcomes.",
        "source": {
            "plan_sha256": file_hash(training / "plan.json"),
            "inputs_sha256": file_hash(training / "inputs.json"),
            "eligibility_sha256": file_hash(training / "eligibility.json"),
        },
        "reuse": {
            "plan_sha256": file_hash(reuse / "plan.json"),
            "collection_sha256": file_hash(reuse / "collection.json"),
            "record_hashes": old_collection["record_hashes"],
            "list_price_estimate_usd": str(invoice),
            "conservative_budget_usd": str(conservative),
        },
        "max_usd": str(max_usd),
        "workers": workers,
        "max_seconds": max_seconds,
        "request_timeout_seconds": REQUEST_SECONDS,
        "automatic_retries": 0,
        "unknown_execution_repeat": False,
        "superseded_attempt": previous,
        "prices_usd_per_million": PRICES,
        "price_source": "https://developers.openai.com/api/docs/pricing",
        "price_checked_date": "2026-10-19",
        "input_token_allowances": {identity: input_allowance(by_id[identity]) for identity in new_ids},
        "request_reservations_usd": {identity: str(reservation(by_id[identity])) for identity in new_ids},
        "budget_rule": "Historical conservative charges, explicitly superseded unknown reservations, settled conservative usage and every in-flight/unknown reservation may never exceed max_usd. Release excess only after verified usage. A budget stop can leave the population incomplete.",
        "operator_sha256": file_hash(Path(__file__)),
        "data_source_sha256": file_hash(Path(__file__).with_name("data.py")),
        "limitations": "The model alias has no distinct dated snapshot. Cost is a token-price estimate, not an invoice. Input allowances are conservative text-byte/schema bounds and are checked against every returned usage receipt.",
    }
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    for directory, names, origin in (
        ("source", ("plan.json", "inputs.json", "eligibility.json"), training),
        ("reuse", ("plan.json", "collection.json"), reuse),
    ):
        (output / directory).mkdir()
        for name in names:
            shutil.copyfile(origin / name, output / directory / name)
    (output / "records").mkdir()
    for identity in reused_ids:
        shutil.copyfile(reuse / "records" / f"{identity}.json", output / "records" / f"{identity}.json")
    if previous:
        shutil.copytree(supersedes, output / "superseded")
    return plan


class Budget:
    def __init__(self, maximum, historical):
        self.maximum, self.settled = Decimal(str(maximum)), Decimal(str(historical))
        self.active = {}

    @property
    def exposure(self):
        return self.settled + sum(self.active.values(), Decimal(0))

    def reserve(self, identity, amount):
        amount = Decimal(str(amount))
        if identity in self.active or amount <= 0 or self.exposure + amount > self.maximum:
            return False
        self.active[identity] = amount
        return True

    def settle(self, identity, known_conservative=None):
        reserved = self.active.pop(identity)
        value = reserved if known_conservative is None else Decimal(str(known_conservative))
        if value < 0 or value > reserved:
            raise ValueError("Charge exceeds the reserved bound")
        self.settled += value

    def receipt(self):
        return {
            "maximum_usd": str(self.maximum),
            "settled_or_unknown_usd": str(self.settled),
            "inflight_usd": {key: str(value) for key, value in self.active.items()},
            "total_exposure_usd": str(self.exposure),
        }


async def _http_request(client, body, secret, seconds):
    async with asyncio.timeout(seconds):
        response = await client.post(ENDPOINT, json=body, headers={"Authorization": "Bearer " + secret})
        if response.status_code != 200:
            return {"error": {"kind": "http_error", "status": response.status_code, "execution_unknown": True}}
        if len(response.content) > 2_000_000:
            return {"error": {"kind": "response_too_large", "execution_unknown": True}}
        return {"payload": response.json()}


async def _call(run, row, secret, client, request_fn):
    started = time.time()
    record = {
        "id": row["id"],
        "request_sha256": digest(body_for(row)),
        "status": "incomplete",
        "answer": None,
        "usage_verified": False,
    }
    try:
        async with asyncio.timeout(REQUEST_SECONDS):
            result = await request_fn(client, body_for(row), secret, REQUEST_SECONDS)
        if "error" in result:
            record["error"] = result["error"]
        else:
            payload = result["payload"]
            raw = run / "responses" / f"{row['id']}.json"
            write(raw, payload)
            record["raw_response_sha256"] = file_hash(raw)
            record.update(
                {
                    "response_id": payload.get("id"),
                    "returned_model": payload.get("model"),
                    "response_status": payload.get("status"),
                    "usage": payload.get("usage"),
                    "service_tier": payload.get("service_tier"),
                    "incomplete_details": payload.get("incomplete_details"),
                }
            )
            try:
                if payload.get("model") != MODEL or payload.get("service_tier") not in (None, "default"):
                    raise ValueError("Returned model or service tier differs from reserved pricing")
                invoice, conservative = usage_cost(payload.get("usage"), input_allowance(row))
                record.update(
                    {
                        "usage_verified": True,
                        "list_price_estimate_usd": float(invoice),
                        "conservative_list_price_usd": str(conservative),
                    }
                )
            except (TypeError, ValueError):
                record["error"] = {"kind": "invalid_or_missing_usage", "execution_unknown": True}
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
                raise ValueError("Teacher response differs from its answer schema")
            record["answer"] = parsed["answer"]
            if record["usage_verified"] and payload.get("status") == "completed" and payload.get("model") == MODEL:
                record["status"] = "complete"
            else:
                record.setdefault(
                    "error", {"kind": "incomplete_or_changed_model", "execution_unknown": not record["usage_verified"]}
                )
    except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001 — persist uncertain execution without retry.
        record["error"] = {"kind": type(error).__name__, "execution_unknown": not record["usage_verified"]}
    record["elapsed_seconds"] = time.time() - started
    write(run / "records" / f"{row['id']}.json", record)
    return record


def checked_plan(run):
    plan = read(run / "plan.json")
    if (
        plan.get("format") != "sie-teacher-reference-extension-v1"
        or plan["settings"] != SETTINGS
        or plan["operator_sha256"] != file_hash(Path(__file__))
        or plan["data_source_sha256"] != file_hash(Path(__file__).with_name("data.py"))
        or plan["prices_usd_per_million"] != PRICES
        or plan.get("automatic_retries") != 0
        or plan.get("unknown_execution_repeat") is not False
        or plan.get("teacher_receives_gold") is not False
        or plan.get("teacher_receives_solver_answers") is not False
        or not 0 < Decimal(plan["max_usd"]) <= MAX_USD
        or not 1 <= plan["workers"] <= MAX_WORKERS
        or not REQUEST_SECONDS + 5 < plan["max_seconds"] <= MAX_SECONDS
    ):
        raise ValueError("Prepared teacher code/settings/limits changed")
    for category, mapping in (
        ("source", plan["source"]),
        ("reuse", {key: plan["reuse"][key] for key in ("plan_sha256", "collection_sha256")}),
    ):
        for key, expected in mapping.items():
            if file_hash(run / category / (key.removesuffix("_sha256") + ".json")) != expected:
                raise ValueError("Copied source/reuse provenance changed")
    rows = {row["id"]: row for row in plan["inputs"]}
    copied_inputs = read(run / "source/inputs.json")
    copied_source = read(run / "source/plan.json")
    copied_eligibility = read(run / "source/eligibility.json")
    copied_pilot = read(run / "reuse/plan.json")
    copied_collection = read(run / "reuse/collection.json")
    if (
        len(rows) != 510
        or len(plan["inputs"]) != 510
        or len(set(plan["new_ids"])) != 446
        or len(plan["new_ids"]) != 446
        or len(set(plan["reused_ids"])) != 64
        or len(plan["reused_ids"]) != 64
        or copied_inputs != {"train": plan["inputs"]}
        or digest(copied_inputs) != copied_source["inputs_sha256"]
        or copied_eligibility["eligible_train_ids"] != [row["id"] for row in plan["inputs"]]
        or plan["reused_ids"] != [row["id"] for row in copied_pilot["inputs"]]
        or plan["reuse"]["record_hashes"] != copied_collection["record_hashes"]
        or set(plan["new_ids"]) & set(plan["reused_ids"])
        or set(rows) != set(plan["new_ids"]) | set(plan["reused_ids"])
        or plan["new_ids"]
        != sorted(plan["new_ids"], key=lambda identity: digest(["teacher-extension-order-v1", identity]))
    ):
        raise ValueError("Prepared teacher identities changed")
    for identity in plan["new_ids"]:
        if Decimal(plan["request_reservations_usd"][identity]) != reservation(rows[identity]) or plan[
            "input_token_allowances"
        ][identity] != input_allowance(rows[identity]):
            raise ValueError("Teacher reservation changed")
    invoice, conservative = Decimal(0), Decimal(0)
    for identity, expected in plan["reuse"]["record_hashes"].items():
        path = run / "records" / f"{identity}.json"
        if file_hash(path) != expected:
            raise ValueError("Reused raw teacher answer changed")
        actual, upper = _check_teacher_record(read(path), rows[identity])
        invoice += actual
        conservative += upper
    if invoice != Decimal(plan["reuse"]["list_price_estimate_usd"]) or conservative != Decimal(
        plan["reuse"]["conservative_budget_usd"]
    ):
        raise ValueError("Historical teacher budget differs from raw usage")
    if (
        plan.get("superseded_attempt") is not None
        and superseded_transport(run / "superseded") != plan["superseded_attempt"]
    ):
        raise ValueError("Superseded transport provenance or unknown exposure changed")
    return plan, rows


async def collect(run, secret=None, request_fn=_http_request):
    run = Path(run)
    plan, rows = checked_plan(run)
    if (
        (run / "started.json").exists()
        or (run / "collection.json").exists()
        or (run / "intents").exists()
        or any((run / "records" / f"{identity}.json").exists() for identity in plan["new_ids"])
    ):
        raise FileExistsError("This phase has already started; no automatic resume or repeated calls")
    secret = secret or os.environ.get("OPENAI_API_KEY")
    if not secret:
        raise ValueError("Set OPENAI_API_KEY before explicitly running collect")
    previous = plan.get("superseded_attempt") or {}
    budget = Budget(
        plan["max_usd"],
        Decimal(plan["reuse"]["conservative_budget_usd"]) + Decimal(previous.get("unknown_exposure_usd", "0")),
    )
    start, monotonic_start = time.time(), time.monotonic()
    write(
        run / "started.json",
        {
            "plan_sha256": file_hash(run / "plan.json"),
            "started_unix": start,
            "deadline_unix": start + plan["max_seconds"],
            "budget": budget.receipt(),
        },
    )
    stop, index, active, completed = None, 0, {}, {}
    ledger_index = 0
    transport = httpx.AsyncHTTPTransport(retries=0, trust_env=False)
    async with httpx.AsyncClient(
        timeout=REQUEST_SECONDS, follow_redirects=False, trust_env=False, transport=transport
    ) as client:
        while index < len(plan["new_ids"]) or active:
            while stop is None and index < len(plan["new_ids"]) and len(active) < plan["workers"]:
                if time.monotonic() - monotonic_start > plan["max_seconds"] - REQUEST_SECONDS - 5:
                    stop = "deadline"
                    break
                identity = plan["new_ids"][index]
                reserve = plan["request_reservations_usd"][identity]
                if not budget.reserve(identity, reserve):
                    if not active:
                        stop = "budget"
                    break
                write(
                    run / "intents" / f"{identity}.json",
                    {
                        "id": identity,
                        "plan_sha256": file_hash(run / "plan.json"),
                        "request_sha256": digest(body_for(rows[identity])),
                        "reservation_usd": reserve,
                        "budget_after_reservation": budget.receipt(),
                        "started_unix": time.time(),
                    },
                )
                task = asyncio.create_task(_call(run, rows[identity], secret, client, request_fn))
                active[task] = identity
                index += 1
            if not active:
                break
            done, _ = await asyncio.wait(active, timeout=30, return_when=asyncio.FIRST_COMPLETED)
            # Process the entire completion batch before any replacement launch.
            for task in done:
                identity = active.pop(task)
                record = await task
                completed[identity] = record
                budget.settle(identity, record.get("conservative_list_price_usd") if record["usage_verified"] else None)
                if record["status"] != "complete":
                    stop = stop or "request_error_or_unknown_execution"
            write(
                run / "ledger" / f"{ledger_index:05d}.json",
                {
                    "completed": len(completed),
                    "dispatched": index,
                    "stop_reason": stop,
                    "budget": budget.receipt(),
                    "elapsed_seconds": time.monotonic() - monotonic_start,
                },
            )
            ledger_index += 1
            print(
                json.dumps(
                    {
                        "event": "teacher_progress",
                        "completed_new": len(completed),
                        "planned_new": len(plan["new_ids"]),
                        "stop_reason": stop,
                        "budget": budget.receipt(),
                    }
                ),
                flush=True,
            )
    hashes = dict(plan["reuse"]["record_hashes"])
    hashes.update({identity: file_hash(run / "records" / f"{identity}.json") for identity in completed})
    known_invoice = Decimal(plan["reuse"]["list_price_estimate_usd"]) + sum(
        (Decimal(str(record["list_price_estimate_usd"])) for record in completed.values() if record["usage_verified"]),
        Decimal(0),
    )
    successful = len(plan["reused_ids"]) + sum(record["status"] == "complete" for record in completed.values())
    summary = {
        "status": "complete" if successful == plan["n"] else "incomplete",
        "n": plan["n"],
        "reused": len(plan["reused_ids"]),
        "dispatched_new": index,
        "returned": len(hashes),
        "complete": successful,
        "stop_reason": stop,
        "plan_sha256": file_hash(run / "plan.json"),
        "elapsed_seconds": time.monotonic() - monotonic_start,
        "list_price_estimate_usd": float(known_invoice),
        "invoice_estimate_complete": not previous and all(record["usage_verified"] for record in completed.values()),
        "prior_unknown_exposure_usd": previous.get("unknown_exposure_usd", "0"),
        "explicit_prior_transport_repeats": len(previous.get("explicitly_repeated_ids", [])),
        "conservative_list_price_usd": str(budget.settled),
        "budget": budget.receipt(),
        "inference_retries": 0,
        "record_hashes": hashes,
        "intent_hashes": {
            identity: file_hash(run / "intents" / f"{identity}.json") for identity in plan["new_ids"][:index]
        },
    }
    write(run / "collection.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--training", type=Path, required=True)
    prepare_parser.add_argument("--reuse", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--max-usd", type=Decimal, default=Decimal(15))
    prepare_parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    prepare_parser.add_argument("--max-seconds", type=int, default=1800)
    prepare_parser.add_argument("--supersedes", type=Path)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare(
            args.training, args.reuse, args.output, args.max_usd, args.workers, args.max_seconds, args.supersedes
        )
        print(
            json.dumps(
                {
                    "prepared": str(args.output),
                    "n": plan["n"],
                    "reuse": len(plan["reused_ids"]),
                    "new": len(plan["new_ids"]),
                    "max_total_usd": plan["max_usd"],
                    "historical_conservative_usd": plan["reuse"]["conservative_budget_usd"],
                    "plan_sha256": file_hash(args.output / "plan.json"),
                },
                indent=2,
            )
        )
    else:
        result = asyncio.run(collect(args.run))
        print(
            json.dumps(
                {key: value for key, value in result.items() if key not in ("record_hashes", "intent_hashes")}, indent=2
            )
        )


if __name__ == "__main__":
    main()
