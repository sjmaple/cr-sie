"""Question-only arithmetic teacher with a numeric final-answer contract."""

from __future__ import annotations

import asyncio
import json
import re
import time
from decimal import Decimal

import teacher_collect as general
from data import digest, file_hash, write

MODEL = general.MODEL
REQUEST_SECONDS = general.REQUEST_SECONDS
PRICES = general.PRICES
SETTINGS = general.SETTINGS | {
    "instructions": (
        "Solve the arithmetic word problem independently. Return only its final numeric answer in the answer "
        "field: one signed integer, decimal, or fraction such as 42, -3.5, or 7/2. Express the number in the "
        "units requested by the question, with any requested rounding. Do not include units, currency symbols, "
        "thousands separators, percent signs, LaTeX, or explanatory words. For a clock-time question, give the "
        "requested numeric clock hour without AM/PM. Do not use external tools."
    )
}
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?(?:\d+(?:\.\d*)?|\.\d+))?")


def numeric_answer(value):
    if not isinstance(value, str) or NUMBER.fullmatch(value.strip()) is None:
        return False
    pieces = value.strip().split("/")
    return len(pieces) == 1 or Decimal(pieces[1]) != 0


def body_for(row):
    return SETTINGS | {"input": row["request"]}


def input_allowance(row):
    allowance = len(json.dumps(body_for(row)).encode()) + 2048
    if allowance >= 272_000:
        raise ValueError("Teacher input exceeds the declared short-context budget")
    return allowance


def reservation(row):
    return (Decimal(input_allowance(row)) * Decimal("12.5") + Decimal(8192 * 50)) / 1_000_000


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
        raise ValueError("Reused teacher record differs from the exact numeric question-only request")
    return general.usage_cost(record.get("usage"), input_allowance(row))


async def _call(run, row, secret, client, request_fn):
    """Keep every returned string; numeric eligibility is assessed without gold."""
    started = time.monotonic()
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
                invoice, conservative = general.usage_cost(payload.get("usage"), input_allowance(row))
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
            record["numeric_answer"] = numeric_answer(parsed["answer"])
            if record["usage_verified"] and payload.get("status") == "completed" and payload.get("model") == MODEL:
                record["status"] = "complete"
            else:
                record.setdefault(
                    "error", {"kind": "incomplete_or_changed_model", "execution_unknown": not record["usage_verified"]}
                )
    except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001 — retain uncertain execution.
        record["error"] = {"kind": type(error).__name__, "execution_unknown": not record["usage_verified"]}
    record["elapsed_seconds"] = time.monotonic() - started
    write(run / "records" / f"{row['id']}.json", record)
    return record
