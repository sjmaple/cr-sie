"""Compare Qwen embeddings with the archived BGE encoder on development data only.

Collect once through SIE, then replay without credentials. Both arms use the
same historical routing-cost allowance to isolate the encoder change. Actual
batched collection credits are reported separately, not as a serving tariff.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sie_sdk.types import Item

import headroom
import learn
import learning
import study
from cloud import usage_of
from run import SingleSendClient

MODEL = "Qwen/Qwen3-Embedding-4B"


def collect(s, directory, client, model=MODEL, instruction=None, *, indices=None, scope=None):
    groups = learning.split(s)
    indices = groups["train"] + groups["validation"] if indices is None else indices
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("embedding collection needs nonempty unique indices")
    metadata = client.get_model(model)
    plan = {
        "model": model,
        "metadata": metadata,
        "study_revision": study.REVISION,
        "ids": [s.ids[i] for i in indices],
        "options": {
            "output_types": ["dense"],
            "is_query": bool(instruction),
            "instruction": instruction,
            "profile": "default",
        },
        "batch_size": 16,
        "max_observed_credits": 2000,
        "max_seconds": 1200,
        "retries": 0,
        "cost_comparison": "Both encoders use the same archived BGE routing-cost allowance; collection credits are separate.",
        "scope": scope or "437 training and 146 validation requests only; no test or unused calibration requests.",
    }
    learn.write(directory / "encoding-plan.json", plan)
    credits, started, calls = 0.0, time.monotonic(), 0
    with gzip.open(directory / "vectors.jsonl.gz", "xt") as vectors, (directory / "calls.jsonl").open("x") as ledger:
        for offset in range(0, len(indices), 16):
            if credits >= 2000 or time.monotonic() - started > 1200:
                raise RuntimeError("collection budget reached; no automatic restart")
            batch = indices[offset : offset + 16]
            ids = [s.ids[i] for i in batch]
            call = {"ids": ids, "model": model, "status": "attempted"}
            tick = time.monotonic()
            try:
                reply = client.encode(
                    model,
                    [Item(id=s.ids[i], text=s.prompts[i]) for i in batch],
                    output_types=["dense"],
                    is_query=bool(instruction),
                    instruction=instruction,
                    options={"profile": "default"},
                    wait_for_capacity=False,
                    max_oom_retries=0,
                    provision_timeout_s=60,
                )
                receipt = usage_of(reply[0])
                # Each item carries the SAME batch receipt; charge it only once.
                request_id = reply[0]["request"]["id"]
                if len(reply) != len(batch) or {r.get("id") for r in reply} != set(ids):
                    raise ValueError("embedding response IDs do not match the requested batch")
                if any(r.get("request", {}).get("id") != request_id or usage_of(r) != receipt for r in reply):
                    raise ValueError("inconsistent batch billing receipts")
                expected_dim = metadata["dims"]["dense"]
                for row in reply:
                    value = np.asarray(row["dense"], dtype=float)
                    if value.shape != (expected_dim,) or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
                        raise ValueError("invalid embedding vector")
                    vectors.write(json.dumps({"id": row["id"], "vector": value.tolist()}) + "\n")
                vectors.flush()
                credits += receipt["credits"]
                calls += 1
                call.update(receipt, status="success", request_id=request_id)
            except Exception as error:
                call.update(status="failed_or_unknown", error_type=type(error).__name__)
                learn.write(
                    directory / "incomplete.json", {"error_type": type(error).__name__, "observed_credits": credits}
                )
                raise
            finally:
                call["elapsed_s"] = round(time.monotonic() - tick, 3)
                ledger.write(json.dumps(call) + "\n")
                ledger.flush()
            print(f"Embedded {offset + len(batch)}/{len(indices)}; {credits:g} billed credits", flush=True)
    learn.write(
        directory / "collection.json",
        {"calls": calls, "observed_credits": credits, "elapsed_s": round(time.monotonic() - started, 3)},
    )


def load_vectors(s, directory):
    plan = json.loads((directory / "encoding-plan.json").read_text())
    groups = learning.split(s)
    ids = [s.ids[i] for i in groups["train"] + groups["validation"]]
    if plan["ids"] != ids or plan["study_revision"] != study.REVISION or (directory / "incomplete.json").exists():
        raise ValueError("recording is incomplete or uses another study/split")
    rows = study.read_jsonl(directory / "vectors.jsonl.gz")
    by_id = {r["id"]: r["vector"] for r in rows}
    if len(rows) != len(ids) or set(by_id) != set(ids):
        raise ValueError("duplicate or missing embedding records")
    values = np.asarray([by_id[i] for i in ids], dtype=float)
    if values.shape != (len(ids), plan["metadata"]["dims"]["dense"]):
        raise ValueError("embedding dimensions do not match the model metadata")
    normalized = study.normalise(values)
    raw = np.zeros((len(s.ids), values.shape[1]))
    vectors = np.zeros_like(raw)
    indices = groups["train"] + groups["validation"]
    raw[indices], vectors[indices] = values, normalized
    return replace(s, raw_vectors=raw, vectors=vectors)


def analyze(s, directory, difficulty=None, output=None):
    revised = load_vectors(s, directory)
    output = directory if output is None else output
    if output != directory:
        output.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for label, data in (("bge", s), ("qwen", revised)):
        target = output / label
        target.mkdir(exist_ok=False)
        result = headroom.experiment(data, target, difficulty)
        plan = json.loads((target / "plan.json").read_text())
        plan["encoder"] = (
            study.EMBEDDER if label == "bge" else json.loads((directory / "encoding-plan.json").read_text())["model"]
        )
        plan["routing_cost_basis"] = (
            "Common archived BGE allowance for both encoders; isolate encoder change, not a Qwen serving-price claim."
        )
        learn.write(target / "plan.json", plan)
        (target / "REPORT.md").write_text(headroom.report(result) + "\n")
        outputs[label] = result
    summary = {}
    for pair in study.PAIRS:
        summary[pair] = {}
        for label, result in outputs.items():
            rows = result["pairs"][pair]["runs"]
            summary[pair][label] = {
                method: {
                    str(size): {
                        "matched_validation_cost_per_1k": float(
                            np.mean(
                                [
                                    r["matched_validation"]["cost_per_1k"]
                                    for r in rows
                                    if r["method"] == method and r["size"] == size and r["target"] == 0.98
                                ]
                            )
                        ),
                        "training_fitted_validation_cost_per_1k": float(
                            np.mean(
                                [
                                    r["validation"]["cost_per_1k"]
                                    for r in rows
                                    if r["method"] == method and r["size"] == size and r["target"] == 0.98
                                ]
                            )
                        ),
                        "training_fitted_validation_quality": float(
                            np.mean(
                                [
                                    r["validation"]["quality_vs_strong"]
                                    for r in rows
                                    if r["method"] == method and r["size"] == size and r["target"] == 0.98
                                ]
                            )
                        ),
                    }
                    for size in headroom.SIZES
                }
                for method in ("knn", "ridge", "cost_aware_ridge")
            }
    learn.write(output / "comparison.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--collect", action="store_true", help="spend SIE credits to collect a new development recording"
    )
    parser.add_argument("--difficulty", type=Path)
    parser.add_argument("--output", type=Path, help="new analysis directory; defaults to the recording directory")
    parser.add_argument(
        "--instruction", help="optional fixed task instruction, applied equally to bank and evaluation requests"
    )
    args = parser.parse_args()
    s = study.load_study()
    if args.collect:
        args.run.mkdir(parents=True, exist_ok=False)
        with SingleSendClient(timeout_s=90, read_timeout_s=90) as client:
            collect(s, args.run, client, instruction=args.instruction)
    analyze(s, args.run, args.difficulty, args.output)


if __name__ == "__main__":
    main()
