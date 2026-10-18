"""Bounded SIE SDK collection and deployment for the embedding policy."""

from __future__ import annotations

import gzip
import json
import math
import time

import numpy as np
from sie_sdk import SIEClient
from sie_sdk.types import Item

from data import digest, write

MODEL = "Qwen/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
INSTRUCTION = "Represent the relative difficulty of the mathematics problem within its named competition for classification as routine or advanced."


class SingleSendClient(SIEClient):
    def _record_retry(self):
        raise RuntimeError("Automatic retries are disabled in this example")


def usage(reply):
    request = reply.get("request") or {}
    native = reply.get("usage") or {}
    billed = request.get("usage") or {}
    credits = request.get("credits_debited", billed.get("credits_charged", native.get("credits_charged")))
    return {"request_id": request.get("id"), "credits": credits, "rate_book_version": billed.get("rate_book_version")}


def error_receipt(error):
    result = {"error_type": type(error).__name__, "status": "failed_or_unknown", "credits": None}
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        result["http_status"] = status
    request = getattr(error, "request", None)
    if isinstance(request, dict):
        result.update(usage({"request": request}))
    return result


def normalize(values):
    x = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms == 0):
        raise ValueError("Invalid vector")
    return x / norms


def collect(rows, output, *, self_hosted=False):
    output.mkdir(parents=True, exist_ok=False)
    started, credits, calls = time.monotonic(), 0.0, []
    with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
        metadata = client.get_model(MODEL)
        if metadata["revision"] != REVISION or metadata["dims"]["dense"] != 2560:
            raise ValueError("Serving model differs from the pinned encoder")
        write(
            output / "plan.json",
            {
                "model": MODEL,
                "revision": REVISION,
                "instruction": INSTRUCTION,
                "rows_sha256": digest(rows),
                "ids": [r["id"] for r in rows],
                "input_sha256": {r["id"]: digest(r["request"]) for r in rows},
                "is_query": True,
                "dimensions": 2560,
                "batch_size": 16,
                "max_seconds": 900,
                "max_observed_credits": 5000,
            },
        )
        try:
            with gzip.open(output / "vectors.jsonl.gz", "xt") as handle:
                for offset in range(0, len(rows), 16):
                    if time.monotonic() - started > 900 or credits >= 5000:
                        raise RuntimeError("Collection budget reached")
                    batch = rows[offset : offset + 16]
                    reply = client.encode(
                        MODEL,
                        [Item(id=r["id"], text=r["request"]) for r in batch],
                        output_types=["dense"],
                        instruction=INSTRUCTION,
                        is_query=True,
                        wait_for_capacity=False,
                        max_oom_retries=0,
                        provision_timeout_s=60,
                    )
                    if len(reply) != len(batch) or {r["id"] for r in reply} != {r["id"] for r in batch}:
                        raise ValueError("Response ID mismatch")
                    receipt = usage(reply[0])
                    if any(usage(r) != receipt for r in reply):
                        raise ValueError("Batch receipts differ")
                    charge = receipt["credits"]
                    if not self_hosted and (
                        not isinstance(charge, (int, float)) or not math.isfinite(charge) or charge < 0
                    ):
                        raise ValueError("Cloud response has no valid credit receipt")
                    credits += charge or 0
                    calls.append(receipt | {"ids": [r["id"] for r in batch]})
                    for row in reply:
                        vector = normalize(row["dense"])
                        if vector.shape != (2560,):
                            raise ValueError("Wrong vector shape")
                        handle.write(json.dumps({"id": row["id"], "vector": vector.tolist()}) + "\n")
                    handle.flush()
                    print(f"Encoded {offset + len(batch)}/{len(rows)}; observed credits {credits:g}", flush=True)
        except Exception as error:
            write(output / "incomplete.json", error_receipt(error) | {"completed_calls": calls})
            raise
    write(
        output / "collection.json",
        {"calls": calls, "credits": None if self_hosted else credits, "elapsed_seconds": time.monotonic() - started},
    )


def vectors(rows, directory):
    if (directory / "incomplete.json").exists():
        raise ValueError("Incomplete collection")
    plan = json.loads((directory / "plan.json").read_text())
    if (plan["model"], plan["revision"], plan["instruction"]) != (MODEL, REVISION, INSTRUCTION):
        raise ValueError("Vector configuration differs")
    if any(plan["input_sha256"].get(r["id"]) != digest(r["request"]) for r in rows):
        raise ValueError("Recorded vector belongs to a different input")
    with gzip.open(directory / "vectors.jsonl.gz", "rt") as handle:
        records = [json.loads(line) for line in handle]
    by_id = {r["id"]: r["vector"] for r in records}
    if len(records) != len(by_id) or set(by_id) != set(plan["ids"]):
        raise ValueError("Duplicate or missing vector IDs")
    return normalize([by_id[r["id"]] for r in rows])


def embed(request):
    with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
        metadata = client.get_model(MODEL)
        if metadata["revision"] != REVISION:
            raise ValueError("Live encoder revision differs")
        reply = client.encode(
            MODEL,
            Item(text=request),
            output_types=["dense"],
            instruction=INSTRUCTION,
            is_query=True,
            wait_for_capacity=False,
            max_oom_retries=0,
            provision_timeout_s=60,
        )
    vector = normalize(reply["dense"])
    if vector.shape != (2560,):
        raise ValueError("Wrong vector shape")
    return vector, usage(reply)
