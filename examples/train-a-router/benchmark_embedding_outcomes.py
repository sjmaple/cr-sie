"""Measure a resident embedding router on the native benchmark's training inputs.

The timed path includes the SIE request, vector normalization, both classifiers,
and the saved routing cutoff. It makes 21 embedding calls and no answer-model
calls. Cloud network time is included; this is not a matched hardware comparison
with direct Laya inference or a concurrent-throughput benchmark.
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
from benchmark_outcomes_native import SAMPLE_SALT, N, select_inputs
from data import digest, file_hash, write
from serving import INSTRUCTION, MODEL, REVISION, SingleSendClient, error_receipt, normalize, usage


def run(run_directory, baseline, output, *, self_hosted=False):
    run_directory, baseline, output = map(Path, (run_directory, baseline, output))
    if output.exists():
        raise FileExistsError(output)
    frozen, selected = outcomes.checked_frozen(run_directory)
    plan, head = selected["plan"], selected["head"]
    dataset = outcomes._baseline(plan, baseline)
    rows, warmup = select_inputs(dataset["splits"]["train"], [row["id"] for row in plan["populations"]["train"]])
    policy = frozen["selections"]["1"]
    result = {
        "format": "sie-embedding-outcome-warm-inference-v1",
        "status": "incomplete",
        "source_sha256": file_hash(Path(__file__)),
        "learner_source_sha256": outcomes.code_hashes(),
        "frozen_sha256": file_hash(run_directory / "frozen.json"),
        "selected_sha256": frozen["selected_model_sha256"],
        "model": MODEL,
        "revision": REVISION,
        "instruction": INSTRUCTION,
        "is_query": True,
        "dimensions": 2560,
        "primary_policy_mode": policy["mode"],
        "self_hosted": self_hosted,
        "client_runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "sdk": importlib.metadata.version("sie-sdk"),
        },
        "sampling": {
            "ordering": f"sha256({SAMPLE_SALT!r} + NUL + id), then id; first20 timed,21st warmup",
            "timed_ids": [row["id"] for row in rows],
            "warmup_id": warmup["id"],
            "requests_sha256": digest([{key: row[key] for key in ("id", "request")} for row in [*rows, warmup]]),
        },
        "protocol": {
            "request_batch_size": 1,
            "concurrency": 1,
            "timed_requests": N,
            "warmup_requests": 1,
            "automatic_retries": 0,
            "deadline_seconds": 600,
            "scope": "Persistent SDK client; HTTP request, response decoding, vector normalization, two classifiers and saved cutoff",
            "excludes": "Artifact checks, model metadata, the separate first warmup call, and answer generation",
            "limitation": "A small sequential training-input probe. Cloud hardware is not controlled and network time is included. A static policy skips this work in deployment. This is not throughput or a hardware-matched Laya comparison.",
        },
        "calls": [],
    }
    write(output, result)
    deadline = time.monotonic() + 600

    def save():
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    try:
        with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
            metadata = client.get_model(MODEL)
            if (
                metadata.get("revision") != REVISION
                or metadata.get("name", metadata.get("id")) != MODEL
                or metadata.get("dims", {}).get("dense") != 2560
            ):
                raise ValueError("Live encoder differs from the pinned model")
            for index, row in enumerate([warmup, *rows]):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Sequential embedding benchmark deadline")
                started = time.perf_counter()
                reply = client.encode(
                    MODEL,
                    Item(text=row["request"]),
                    output_types=["dense"],
                    instruction=INSTRUCTION,
                    is_query=True,
                    wait_for_capacity=False,
                    max_oom_retries=0,
                    provision_timeout_s=min(60, remaining),
                )
                arrived = time.perf_counter()
                record = {"id": row["id"], "warmup": index == 0, "receipt": usage(reply)}
                result["calls"].append(record)
                vector = normalize(reply["dense"])
                if vector.shape != (2560,):
                    raise ValueError("Wrong embedding dimensions")
                score = float(outcomes.score(vector[None, :], head)[0])
                if not np.isfinite(score):
                    raise ValueError("Nonfinite router score")
                if policy["mode"] == "threshold":
                    route = "strong" if score >= policy["cutoff"] else "cheap"
                elif policy["mode"] in ("always_strong", "always_cheap"):
                    route = policy["mode"].removeprefix("always_")
                else:
                    raise ValueError("Unknown routing mode")
                finished = time.perf_counter()
                record.update(
                    seconds=finished - started,
                    network_and_embedding_seconds=arrived - started,
                    local_scoring_seconds=finished - arrived,
                    score=score,
                    route=route,
                )
                save()
                if not self_hosted and not solver.valid_credit(record["receipt"]["credits"]):
                    raise ValueError("Cloud response has no usable charge receipt")
        timed = result["calls"][1:]
        seconds = np.asarray([row["seconds"] for row in timed])
        result.update(
            status="complete",
            n=len(timed),
            warmup_seconds_excluded=result["calls"][0]["seconds"],
            p50_ms=float(np.percentile(seconds, 50, method="linear") * 1000),
            p95_ms=float(np.percentile(seconds, 95, method="linear") * 1000),
            mean_ms=float(seconds.mean() * 1000),
            local_scoring_p50_ms=float(np.median([row["local_scoring_seconds"] for row in timed]) * 1000),
            total_timed_seconds=float(seconds.sum()),
            total_observed_credits=None if self_hosted else sum(row["receipt"]["credits"] for row in result["calls"]),
            timed_observed_credits=None if self_hosted else sum(row["receipt"]["credits"] for row in timed),
        )
    except Exception as error:
        result["error"] = error_receipt(error)
        raise
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-hosted", action="store_true")
    args = parser.parse_args()
    result = run(args.run, args.baseline, args.output, self_hosted=args.self_hosted)
    print(
        json.dumps(
            {key: result[key] for key in ("n", "p50_ms", "p95_ms", "local_scoring_p50_ms", "total_observed_credits")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
