"""Question-only, resumable current-model screen. Run acquisition on Modal CPU.

The collector never reads gold answers. Provider failures stop new dispatches;
in-flight requests are allowed to settle and unknown requests are not retried.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import time

import httpx
from sie_sdk import SIEClient


ARMS = {
    "qwen4b": {"provider": "sie", "model": "Qwen/Qwen3.5-4B", "workers": 4,
               "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"},
    "qwen27b": {"provider": "sie", "model": "Qwen/Qwen3.8-27B-FP8", "workers": 4,
                "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"},
    "luna": {"provider": "openai", "model": "gpt-6-luna", "workers": 8},
    "sol": {"provider": "openai", "model": "gpt-6.1-sol", "workers": 8},
}
PRICES = {
    "gpt-6.1-sol": {"input": "2", "cached_input": "0.10", "cache_write": "2.50", "output": "10"},
    "gpt-6-luna": {"input": "0.10", "cached_input": "0.01", "cache_write": "0.125", "output": "0.50"},
}
SYSTEM = (
    "Solve the question independently, using the full rules and options supplied. "
    "Do not use tools or external sources. Return the requested JSON object. "
    "The answer field must contain only the final option label or requested value. "
    "For multiple choice, give the option label, not its text. For integers, give "
    "a signed base-10 integer without punctuation or units. For exact strings, "
    "preserve case, internal spaces and punctuation. This JSON transport replaces "
    "any source instruction to wrap the final answer in brackets or other markup. "
)
SIE_SYSTEM = SYSTEM + "Write concise reasoning in the work field before the final answer."
SIE_SETTINGS = {"max_tokens": 4096, "temperature": 0, "top_p": 1,
                "presence_penalty": 0, "frequency_penalty": 0}
OPENAI_SETTINGS = {"reasoning": {"effort": "medium"}, "max_output_tokens": 16384,
                   "store": False, "service_tier": "default"}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def schema_for(contract, work):
    kind = contract.get("type")
    if kind not in {"choice", "label", "integer", "exact_string"}:
        raise ValueError("Unknown public answer contract")
    answer = {"type": "string"}
    if kind in {"choice", "label"}:
        labels = contract.get("allowed")
        if not labels or not all(isinstance(value, str) and value for value in labels):
            raise ValueError("Missing input-declared labels")
        answer["enum"] = labels
    properties = ({"work": {"type": "string"}} if work else {}) | {"answer": answer}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def body_for(item, arm):
    if set(item) != {"id", "request", "response_contract"}:
        raise ValueError("Only question ID, request and public answer contract may reach collector")
    if not isinstance(item["request"], str) or not item["request"].strip():
        raise ValueError("Empty question")
    config = ARMS[arm]
    contract = item["response_contract"]
    request = item["request"] + "\n\nFinal-answer type: " + contract["type"] + "."
    if config["provider"] == "sie":
        return {"model": config["model"], "messages": [{"role": "system", "content": SIE_SYSTEM},
                {"role": "user", "content": request}], **SIE_SETTINGS,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "worked_answer", "strict": True, "schema": schema_for(contract, True)}}}
    return {"model": config["model"], "instructions": SYSTEM, "input": request,
            **OPENAI_SETTINGS, "text": {"format": {"type": "json_schema", "name": "answer",
                "strict": True, "schema": schema_for(contract, False)}}}


def parse_answer(content, contract):
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
        return None
    answer = value["answer"].strip()
    kind = contract["type"]
    if kind in {"choice", "label"} and answer not in contract["allowed"]:
        return None
    if kind == "integer":
        if not re.fullmatch(r"[+-]?\d+", answer):
            return None
        return str(int(answer))
    if kind == "exact_string" and (not answer or len(answer) > contract.get("max_length", 2048)):
        return None
    return answer


def openai_cost(usage, body):
    incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
    details = usage.get("input_tokens_details") or {}
    cached, written = details.get("cached_tokens", 0), details.get("cache_write_tokens", 0)
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    values = (incoming, outgoing, cached, written, reasoning)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Invalid token usage")
    if cached + written > incoming or reasoning > outgoing or outgoing > body["max_output_tokens"]:
        raise ValueError("Inconsistent usage")
    p = {key: Decimal(value) for key, value in PRICES[body["model"]].items()}
    cost = ((incoming - cached - written) * p["input"] + cached * p["cached_input"]
            + written * p["cache_write"] + outgoing * p["output"]) / 1_000_000
    budget = ((incoming - cached) * p["cache_write"] + cached * p["cached_input"]
              + outgoing * p["output"]) / 1_000_000
    return {"usd_estimate": str(cost), "budget_usd": str(budget),
            "basis": "token-accounted standard list prices; output includes reasoning",
            "cache_write_detail_present": "cache_write_tokens" in details}


def reservation(body, arm):
    if ARMS[arm]["provider"] == "sie":
        return Decimal("0.06")  # 6,000 credits, checked against every receipt.
    p = PRICES[body["model"]]
    input_bound = len(json.dumps(body, ensure_ascii=False).encode()) + 2048
    return (Decimal(input_bound) * Decimal(p["cache_write"])
            + Decimal(body["max_output_tokens"]) * Decimal(p["output"])) / 1_000_000


def acquire(item, arm):
    body = body_for(item, arm)
    row = {"id": item["id"], "arm": arm, "model": body["model"],
           "body_sha256": sha(json.dumps(body, sort_keys=True).encode()), "created_unix": time.time()}
    start = time.monotonic()
    try:
        if ARMS[arm]["provider"] == "sie":
            with SIEClient(base_url="https://api.superlinked.com", api_key=os.environ["SIE_API_KEY"],
                           timeout_s=600) as client:
                reply = client.chat_completions(**body, wait_for_capacity=True,
                    provision_timeout_s=300, read_timeout_s=600, max_oom_retries=0)
            choices = [{"message": {k: v for k, v in c.get("message", {}).items()
                        if k in {"role", "content", "reasoning_content"}},
                        "finish_reason": c.get("finish_reason")} for c in reply.get("choices", [])]
            row.update(choices=choices, usage=reply.get("usage"), request=reply.get("request"))
            usage = reply.get("usage") or {}
            credits = usage.get("credits_charged_decimal", usage.get("credits_charged"))
            if credits is None or Decimal(str(credits)) < 0:
                raise ValueError("Missing measured SIE charge")
            cost = Decimal(str(credits)) * Decimal("0.00001")
            row["cost"] = {"usd_estimate": str(cost), "budget_usd": str(cost),
                           "basis": "measured credits at USD0.00001/credit"}
            finish = choices[0].get("finish_reason") if len(choices) == 1 else None
            content = choices[0]["message"].get("content") if len(choices) == 1 else None
            row["provider_status"] = finish
            row["answer"] = parse_answer(content, item["response_contract"]) if finish == "stop" else None
        else:
            with httpx.Client(timeout=httpx.Timeout(600, connect=30), follow_redirects=False,
                              transport=httpx.HTTPTransport(retries=0), trust_env=False) as client:
                response = client.post("https://api.openai.com/v1/responses", json=body,
                    headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                             "Content-Type": "application/json"})
            row["http_status"] = response.status_code
            payload = response.json()
            if response.status_code != 200:
                error = payload.get("error") or {}
                row.update(status="provider_error", error_code=error.get("code"), error_type=error.get("type"))
                return row
            # Do not retain encrypted reasoning or an opaque raw provider payload.
            messages = [{"type": part.get("type"), "text": part.get("text"),
                         "refusal": part.get("refusal")}
                        for out in payload.get("output", []) if out.get("type") == "message"
                        for part in out.get("content", []) if part.get("type") in {"output_text", "refusal"}]
            row.update(provider_status=payload.get("status"), usage=payload.get("usage"),
                       response_id=payload.get("id"), resolved_model=payload.get("model"),
                       messages=messages, incomplete_details=payload.get("incomplete_details"))
            row["cost"] = openai_cost(payload.get("usage") or {}, body)
            text = "".join(part.get("text") or "" for part in messages if part["type"] == "output_text")
            row["answer"] = parse_answer(text, item["response_contract"]) if payload.get("status") == "completed" else None
        row["status"] = "complete"
        row["valid_answer"] = row["answer"] is not None
        if Decimal(row["cost"]["budget_usd"]) > reservation(body, arm):
            row["status"] = "charge_exceeds_reservation"
    except Exception as error:
        metadata = getattr(error, "request", None)
        metadata = ({key: value for key, value in metadata.items() if key in
            {"id", "credits_debited", "rate_book_version", "usage", "execution_identity_sha256",
             "execution_binding_sha256", "served_by"}} if isinstance(metadata, dict) else None)
        row.update(status="failed_or_unknown", error_type=type(error).__name__,
                   http_status=getattr(error, "status_code", None),
                   error_code=getattr(error, "code", None), request=metadata)
    finally:
        row["elapsed_seconds"] = time.monotonic() - start
    return row


def collect(inputs, destination, arm, budget, commit=lambda: None, limit=None):
    items = json.loads(Path(inputs).read_text())
    if limit is not None:
        items = items[:limit]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Repeated question ID")
    for item in items:
        body_for(item, arm)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plan = {"input_sha256": sha(Path(inputs).read_bytes()), "arm": arm, "config": ARMS[arm],
            "collector_sha256": sha(Path(__file__).read_bytes()), "limit": limit,
            "settings": SIE_SETTINGS if ARMS[arm]["provider"] == "sie" else OPENAI_SETTINGS,
            "budget_usd": str(budget), "count": len(items), "gold_available": False}
    plan_path = destination / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Cannot change an existing collection contract")
    dump(plan_path, plan)
    if ARMS[arm]["provider"] == "sie":
        with SIEClient(base_url="https://api.superlinked.com", api_key=os.environ["SIE_API_KEY"],
                       timeout_s=60) as client:
            metadata = client.get_model(ARMS[arm]["model"])
        if metadata.get("revision") != ARMS[arm]["revision"]:
            raise ValueError("Current catalog revision differs from frozen model revision")
        dump(destination / "catalog.json", metadata)
    records, pending = [], []
    spent = Decimal(0)
    for item in items:
        key = sha(item["id"].encode())
        receipt = destination / "records" / (key + ".json")
        intent = destination / "intents" / (key + ".json")
        if receipt.exists():
            result = json.loads(receipt.read_text())
            body = body_for(item, arm)
            expected = {"id": item["id"], "arm": arm, "model": ARMS[arm]["model"],
                        "body_sha256": sha(json.dumps(body, sort_keys=True).encode())}
            if any(result.get(key) != value for key, value in expected.items()) or not intent.exists():
                raise ValueError("Receipt identity does not match current request")
            prior_intent = json.loads(intent.read_text())
            if any(prior_intent.get(key) != value for key, value in expected.items() if key != "model"):
                raise ValueError("Prior intent identity does not match current request")
            records.append(result)
            if result["status"] != "complete":
                raise ValueError("Prior failure needs explicit resolution before resume")
            spent += Decimal(result["cost"]["budget_usd"])
        elif intent.exists():
            raise ValueError("Unsettled prior request; never retry an unknown dispatch")
        else:
            pending.append((item, intent, receipt))
    stop = None
    workers = ARMS[arm]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(pending), workers):
            wave = pending[offset:offset + workers]
            reserved = sum((reservation(body_for(item, arm), arm) for item, _, _ in wave), Decimal(0))
            if spent + reserved > budget:
                stop = "budget_bound_reached"
                break
            for item, intent, _ in wave:
                body = body_for(item, arm)
                dump(intent, {"id": item["id"], "arm": arm, "created_unix": time.time(),
                    "body_sha256": sha(json.dumps(body, sort_keys=True).encode()),
                    "reserved_usd": str(reservation(body, arm))})
            commit()
            futures = {pool.submit(acquire, item, arm): receipt for item, _, receipt in wave}
            for future in as_completed(futures):
                result = future.result()
                dump(futures[future], result)
                records.append(result)
                spent += Decimal(result.get("cost", {}).get("budget_usd", "0"))
                if result["status"] != "complete":
                    stop = result["status"]
            commit()
            print(json.dumps({"progress": arm, "completed": len(records), "total": len(items),
                              "budget_usd": str(spent), "stop": stop}), flush=True)
            if stop:
                break
    summary = {"status": "complete" if not stop and len(records) == len(items) else "incomplete",
               "stop": stop, "arm": arm, "count": len(records), "expected": len(items),
               "valid_answers": sum(bool(row.get("valid_answer")) for row in records),
               "spent_budget_usd": str(spent), "gold_available": False,
               "failed_ids": [row["id"] for row in records if row["status"] != "complete"],
               "unknown_reserved_usd": str(sum((reservation(body_for(item, arm), arm)
                    for item in items if any(row["id"] == item["id"] and "cost" not in row
                                             for row in records)), Decimal(0)))}
    dump(destination / "summary.json", summary)
    commit()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--budget-usd", type=Decimal, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(collect(args.inputs, args.output, args.arm, args.budget_usd, limit=args.limit), indent=2))
