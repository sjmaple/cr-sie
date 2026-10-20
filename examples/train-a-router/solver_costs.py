"""Measure one Cloud embedding request per problem, including billing rounding."""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from sie_sdk.types import Item

from data import digest, file_hash, write
from serving import INSTRUCTION, MODEL, REVISION, SingleSendClient, error_receipt, normalize, usage
from solver import checked_inputs


def measure(row):
    record = {"id": row["id"], "input_sha256": digest(row["request"]), "credits": None}
    started = time.monotonic()
    try:
        with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
            reply = client.encode(
                MODEL,
                Item(id=row["id"], text=row["request"]),
                output_types=["dense"],
                instruction=INSTRUCTION,
                is_query=True,
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=120,
            )
            receipt = usage(reply)
            record.update(receipt)
            record["execution_revision"] = client.last_model_revision
            vector = normalize(reply["dense"])
            if vector.shape != (2560,) or reply.get("id") != row["id"]:
                raise ValueError("Embedding identity or dimensions differ")
            record["vector_sha256"] = digest(vector.tolist())
            credits = receipt["credits"]
            if type(credits) not in (int, float) or not math.isfinite(credits) or credits < 0:
                raise ValueError("Missing Cloud credit receipt")
            record["status"] = "complete"
    except Exception as error:  # noqa: BLE001 -- retain failures without exposing exception contents
        record["status"] = "incomplete"
        record["error"] = error_receipt(error)
        failure_charge = record["error"].get("credits")
        if type(failure_charge) in (int, float) and math.isfinite(failure_charge) and failure_charge >= 0:
            record["credits"] = failure_charge
        else:
            record["error"]["credits"] = None
    charge = record["credits"]
    if type(charge) not in (int, float) or not math.isfinite(charge) or charge < 0:
        record["credits"] = None
    record["elapsed_seconds"] = time.monotonic() - started
    return record


def collect(directory, split, output, workers=4):
    _plan, inputs, _settings = checked_inputs(directory)
    if not 1 <= workers <= 16:
        raise ValueError("Use 1..16 workers")
    rows = inputs[split]
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "plan_sha256": file_hash(directory / "plan.json"),
        "model": MODEL,
        "revision": REVISION,
        "instruction": INSTRUCTION,
        "is_query": True,
        "batch_size": 1,
        "ids": [r["id"] for r in rows],
        "max_observed_credits": 5000,
        "max_seconds": 1800,
        "workers": workers,
        "limits": "Stop new requests at the limit; drain requests already in flight",
    }
    write(output / "plan.json", plan)
    records, tasks = {}, iter(rows)
    started, credits, stopped, fatal = time.monotonic(), 0.0, False, None
    try:
        with SingleSendClient(timeout_s=60) as client:
            if client.get_model(MODEL)["revision"] != REVISION:
                raise ValueError("Cloud embedding revision changed")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = set()

            def schedule():
                nonlocal stopped
                if stopped or credits >= 5000 or time.monotonic() - started >= 1800:
                    stopped = True
                    return
                row = next(tasks, None)
                if row is not None:
                    pending.add(pool.submit(measure, row))

            for _ in range(workers):
                schedule()
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    record = future.result()
                    write(output / (record["id"] + ".json"), record)
                    records[record["id"]] = record
                    charge = record["credits"]
                    if type(charge) in (int, float) and math.isfinite(charge) and charge >= 0:
                        credits += charge
                    stopped |= record["status"] != "complete"
                if done:
                    print(f"Embedding {split}: {len(records)}/{len(rows)}, {credits:g} credits", flush=True)
                for _ in done:
                    schedule()
    except Exception as error:  # noqa: BLE001 -- record unknown failures without sensitive error text
        fatal = error_receipt(error)
    complete = len(records) == len(rows) and not fatal and all(r["status"] == "complete" for r in records.values())
    write(
        output / "collection.json",
        {
            "complete": complete,
            "observed_credits": credits,
            "fatal_error": fatal,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    if not complete:
        raise RuntimeError("Embedding collection incomplete; preserve its receipts before investigating")
    write(
        output / "router-credits.json",
        {
            "plan_sha256": plan["plan_sha256"],
            "model": MODEL,
            "revision": REVISION,
            "batch_size": 1,
            "collection_plan_sha256": file_hash(output / "plan.json"),
            split: [{"id": r["id"], "credits": records[r["id"]]["credits"]} for r in rows],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("calibration", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    collect(args.run, args.split, args.output, args.workers)


if __name__ == "__main__":
    main()
