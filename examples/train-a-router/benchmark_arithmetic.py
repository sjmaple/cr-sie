"""Time 20 warm Cloud embedding decisions on the native probe's training inputs.

This makes 21 embedding calls: a separate warmup and 20 measured requests.
It makes no answer-model calls. Network time is included, and model selection
does not depend on this timing probe.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
from sie_sdk.types import Item

import benefit_outcomes as outcomes
import solver
from data import file_hash, write
from serving import INSTRUCTION, MODEL, REVISION, SingleSendClient, error_receipt, normalize, usage


def run(inputs, model, output, *, self_hosted=False):
    inputs, model, output = Path(inputs), Path(model), Path(output)
    sample, head = json.loads(inputs.read_text()), json.loads(model.read_text())["head"]
    rows = sample["rows"]
    if (
        sample.get("salt") != "arithmetic-warm-v1" or sample.get("warmup") != 1 or sample.get("timed") != 20
        or len(rows) != 21 or len({row["id"] for row in rows}) != 21
        or any(set(row) != {"id", "request"} for row in rows)
        or not {row["id"] for row in rows} <= set(head["training_ids"])
    ):
        raise ValueError("Need the saved 20 training requests and separate 21st warmup request")
    timed, warmup = rows[:20], rows[20]
    result = {
        "format": "sie-arithmetic-warm-embedding-v1", "status": "incomplete",
        "inputs_sha256": file_hash(inputs), "model_sha256": file_hash(model),
        "source_sha256": file_hash(Path(__file__)), "model": MODEL, "revision": REVISION,
        "instruction": INSTRUCTION, "is_query": True, "dimensions": 2560,
        "self_hosted": self_hosted, "timed_ids": [row["id"] for row in timed], "warmup_id": warmup["id"],
        "client_runtime": {
            "machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
            "sdk": importlib.metadata.version("sie-sdk"),
        },
        "protocol": {
            "concurrency": 1, "batch_size": 1, "automatic_retries": 0,
            "scope": "Resident SDK client, HTTP, normalization, both classifiers and a timing-only comparison with zero.",
            "excludes": "Artifact verification, metadata, warmup and answer generation.",
            "limitation": "Small sequential probe, not throughput or an equal-hardware comparison with direct Laya. Static policies would skip this work.",
        },
        "calls": [],
    }
    write(output, result)

    def save():
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    deadline = time.monotonic() + 600
    try:
        with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
            metadata = client.get_model(MODEL)
            if metadata.get("revision") != REVISION or metadata.get("dims", {}).get("dense") != 2560:
                raise ValueError("Live encoder differs from the pinned model")
            for index, row in enumerate([warmup, *timed]):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Embedding timing deadline reached")
                started = time.perf_counter()
                reply = client.encode(
                    MODEL, Item(text=row["request"]), output_types=["dense"], instruction=INSTRUCTION,
                    is_query=True, wait_for_capacity=False, max_oom_retries=0, provision_timeout_s=60,
                )
                arrived = time.perf_counter()
                record = {"id": row["id"], "warmup": index == 0, "receipt": usage(reply)}
                result["calls"].append(record)
                vector = normalize(reply["dense"])
                if vector.shape != (2560,):
                    raise ValueError("Wrong vector dimensions")
                score = float(outcomes.score(vector[None, :], head)[0])
                if not np.isfinite(score):
                    raise ValueError("Invalid router score")
                comparison = score >= 0
                finished = time.perf_counter()
                record.update(
                    seconds=finished - started, network_and_embedding_seconds=arrived - started,
                    local_scoring_seconds=finished - arrived, score=score, timing_only_comparison=comparison,
                )
                save()
                if not self_hosted and not solver.valid_credit(record["receipt"]["credits"]):
                    raise ValueError("Cloud call has no valid charge receipt")
        measured = result["calls"][1:]
        seconds = [row["seconds"] for row in measured]
        result.update(
            status="complete", n=len(measured), warmup_seconds_excluded=result["calls"][0]["seconds"],
            p50_ms=float(np.percentile(seconds, 50) * 1000), p95_ms=float(np.percentile(seconds, 95) * 1000),
            local_scoring_p50_ms=float(np.median([row["local_scoring_seconds"] for row in measured]) * 1000),
            total_observed_credits=None if self_hosted else sum(row["receipt"]["credits"] for row in result["calls"]),
            timed_observed_credits=None if self_hosted else sum(row["receipt"]["credits"] for row in measured),
        )
    except Exception as error:
        result["error"] = error_receipt(error)
        raise
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-hosted", action="store_true")
    args = parser.parse_args()
    result = run(args.inputs, args.model, args.output, self_hosted=args.self_hosted)
    print(json.dumps({key: result[key] for key in ("n", "p50_ms", "p95_ms", "local_scoring_p50_ms", "total_observed_credits")}, indent=2))


if __name__ == "__main__":
    main()
