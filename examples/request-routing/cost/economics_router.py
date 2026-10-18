"""Measure singleton Cloud embedding costs and retain the vectors actually served."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sie_sdk.types import Item

import difficulty_policy as embeddings
import economics
import learn
from cloud import digest, usage_of
from run import SingleSendClient


def encode(row):
    result = {"id": row["id"], "input_sha256": digest(row["request"])}
    started = time.monotonic()
    try:
        with SingleSendClient(timeout_s=90, read_timeout_s=90) as client:
            replies = client.encode(
                embeddings.MODEL,
                [Item(id=row["id"], text=row["request"])],
                output_types=["dense"],
                instruction=embeddings.EMBED_INSTRUCTION,
                is_query=True,
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=60,
            )
            if len(replies) != 1 or replies[0]["id"] != row["id"]:
                raise ValueError("Embedding identity mismatch")
            reply = replies[0]
            result.update(usage_of(reply))
            result["model_revision"] = client.last_model_revision
            vector = np.asarray(reply["dense"])
            if vector.shape != (2560,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                raise ValueError("Invalid Qwen4B embedding")
            result["vector"] = vector.tolist()
            result["status"] = "success"
    except Exception as error:  # noqa: BLE001 -- retain no headers or credential-bearing error bodies
        result.update(economics.failure_metadata(error))
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    return result


def collect(directory, reference):
    inputs = economics.read(directory / "inputs.json")
    economics.frozen_policies(directory)
    original = economics.read(reference / "main/dev-embeddings/plan.json")["metadata"]
    with SingleSendClient(timeout_s=60) as client:
        metadata = client.get_model(embeddings.MODEL)
    if metadata["revision"] != original["revision"] or metadata["dims"] != original["dims"]:
        raise ValueError("Live embedding model differs from the training vectors")
    target = directory / "router-embeddings"
    target.mkdir(exist_ok=False)
    rows = inputs["test"]
    learn.write(
        target / "plan.json",
        {
            "ids": [r["id"] for r in rows],
            "model": embeddings.MODEL,
            "instruction": embeddings.EMBED_INSTRUCTION,
            "batch_size": 1,
            "max_requests_per_second": 4,
            "purpose": "Actual per-request routing cost and the vectors used in financial evaluation",
            "metadata": metadata,
        },
    )
    credits, failed = 0, False
    started = time.monotonic()
    with (
        (directory / "router-receipts.jsonl").open("x") as receipts,
        gzip.open(target / "vectors.jsonl.gz", "xt") as vectors,
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        next_wave = started
        for offset in range(0, len(rows), 4):
            # Keep singleton receipts within the standard key's request rate,
            # even when embeddings finish much faster than generation.
            if next_wave > time.monotonic():
                time.sleep(next_wave - time.monotonic())
            next_wave = time.monotonic() + 1
            if credits >= 5000 or time.monotonic() - started >= 1200:
                failed = True
                break
            for result in pool.map(encode, rows[offset : offset + 4]):
                vector = result.pop("vector", None)
                receipts.write(json.dumps(result) + "\n")
                receipts.flush()
                credits += result.get("credits", 0)
                failed |= result["status"] != "success"
                if vector is not None:
                    vectors.write(json.dumps({"id": result["id"], "vector": vector}) + "\n")
            vectors.flush()
            if (offset + 4) % 32 == 0 or failed:
                print(
                    json.dumps({"embeddings": min(offset + 4, len(rows)), "credits": credits, "failed": failed}),
                    flush=True,
                )
            if failed:
                break
    result = {"credits": credits, "elapsed_s": time.monotonic() - started, "failed": failed}
    learn.write(target / ("incomplete.json" if failed else "collection.json"), result)
    if failed:
        raise RuntimeError("Incomplete embedding receipts retained; no automatic retry")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("evidence/difficulty-policy"))
    args = parser.parse_args()
    print(json.dumps(collect(args.run, args.reference), indent=2))
