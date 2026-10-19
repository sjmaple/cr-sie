"""Measure warm, sequential request latency through your local SIE server.

Run each approach separately on the same machine and evaluation inputs. This
measures request latency, not concurrent throughput or model startup time.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sie_sdk.types import Item

from data import digest, write
from decision import checksum, probability_cutoff
from policies import scores
from serving import INSTRUCTION, MODEL, REVISION, SingleSendClient, error_receipt, normalize, vectors


def native_settings(run, evaluation, variant, rows):
    frozen = json.loads((run / "frozen.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    reference = json.loads((run / "evaluation-test.json").read_text())
    selected = frozen["selected"][variant]
    checkpoint = run / variant
    if (
        frozen["plan_sha256"] != checksum(run / "plan.json")
        or plan["data_sha256"]["questions.json"] != checksum(run / "questions.json")
        or reference["frozen_sha256"] != checksum(run / "frozen.json")
        or reference["data_sha256"] != checksum(evaluation)
        or selected["checkpoint_sha256"] != checksum(checkpoint / "model.safetensors")
        or reference["ids"][: len(rows)] != [row["id"] for row in rows]
    ):
        raise ValueError("Native checkpoint or evaluation inputs differ from the frozen run")
    config = json.loads((checkpoint / "rl_agent_config.json").read_text())
    return (
        json.loads((run / "questions.json").read_text()),
        probability_cutoff(config, selected["threshold"]),
        reference["models"][variant]["predictions"][: len(rows)],
        selected["checkpoint_sha256"],
    )


def summarize(values):
    return {
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "min": min(values),
        "max": max(values),
    }


def save_progress(path, result):
    # The initial exclusive create in main owns this path before any updates.
    path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=("laya", "embedding"), required=True)
    parser.add_argument("--url", required=True, help="Loopback URL of your local SIE server")
    parser.add_argument("--evaluation", type=Path, default=Path("evidence/test.json"))
    parser.add_argument("--run", type=Path, default=Path("run-output/my-lora2581"))
    parser.add_argument("--variant", choices=("lora", "full"), default="lora")
    parser.add_argument("--policy", type=Path, default=Path("run-output/policies/ridge512.json"))
    parser.add_argument("--reference-embeddings", type=Path, default=Path("evidence/test-embeddings"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    address = urlparse(args.url)
    if (
        address.scheme not in ("http", "https")
        or address.hostname not in ("localhost", "127.0.0.1", "::1")
        or address.username is not None
        or address.password is not None
        or address.query
        or address.fragment
        or address.path not in ("", "/")
    ):
        parser.error("Use a loopback server URL without credentials, path, query or fragment")
    if not 1 <= args.limit <= 479 or not 1 <= args.warmup <= args.limit:
        parser.error("Use 1 <= warmup <= limit <= 479")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = json.loads(args.evaluation.read_text())[: args.limit]
    if len(rows) != args.limit or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Need enough distinct evaluation requests")
    policy, questions, expected = None, None, None
    if args.approach == "laya":
        questions, threshold, expected, identity = native_settings(args.run, args.evaluation, args.variant, rows)
        model = f"example/laya-difficulty-{args.variant}"
    else:
        policy = json.loads(args.policy.read_text())
        if policy["method"] != "ridge":
            raise ValueError("This comparison requires a ridge classifier")
        scores(policy, np.zeros(2560))  # Verify the complete policy before any network call.
        threshold, identity, model = policy["threshold"], policy["sha256"], MODEL
        expected = (scores(policy, vectors(rows, args.reference_embeddings)) >= threshold).astype(int).tolist()
    result = {
        "status": "running",
        "approach": args.approach,
        "model": model,
        "checkpoint_or_policy_sha256": identity,
        "evaluation_sha256": checksum(args.evaluation),
        "requests_sha256": digest([{k: row[k] for k in ("id", "request")} for row in rows]),
        "benchmark_source_sha256": checksum(Path(__file__)),
        "sdk_version": importlib.metadata.version("sie-sdk"),
        "protocol": {
            "transport": "local loopback HTTP; persistent SDK client; one request at a time",
            "warmup_requests": args.warmup,
            "timed_requests": args.limit,
            "batch_size": 1,
            "includes": "SDK request, server processing, response decoding, local score and cutoff",
            "local_scoring": "Includes vector normalization and per-request policy verification for embeddings",
            "excludes": "Model loading/download, process startup and initial artifact/model-metadata checks",
            "deadline_seconds": 600,
            "deadline_scope": "Checked between requests; each HTTP request has a separate 120-second timeout",
            "retries": 0,
            "quality": "Latency sample only; not an accuracy or throughput benchmark",
        },
        "samples": [],
    }
    if args.approach == "embedding":
        result["reference_vectors_sha256"] = checksum(args.reference_embeddings / "vectors.jsonl.gz")
    write(args.output, result)
    deadline = time.monotonic() + 600
    try:
        # This local benchmark never sends an inherited Cloud credential.
        with SingleSendClient(base_url=args.url, api_key="", timeout_s=120, read_timeout_s=120) as client:
            metadata = client.get_model(model)
            if args.approach == "embedding" and (metadata["revision"] != REVISION or metadata["dims"]["dense"] != 2560):
                raise ValueError("Serving encoder differs from the frozen embedding policy")
            for index, row in enumerate(rows[: args.warmup] + rows):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Serving benchmark deadline reached")
                item = Item(text=row["request"])
                options = {"wait_for_capacity": False, "max_oom_retries": 0, "provision_timeout_s": min(60, remaining)}
                start = time.perf_counter()
                if args.approach == "laya":
                    reply = client.extract(model, item, output_schema=questions, **options)
                    arrived = time.perf_counter()
                    value = float(reply["data"]["difficulty"]["probabilities"]["advanced"])
                else:
                    reply = client.encode(
                        model, item, output_types=["dense"], instruction=INSTRUCTION, is_query=True, **options
                    )
                    arrived = time.perf_counter()
                    vector = normalize(reply["dense"])
                    if vector.shape != (2560,):
                        raise ValueError("Wrong embedding dimensions")
                    value = float(scores(policy, vector))
                if not math.isfinite(value):
                    raise ValueError("Nonfinite routing score")
                prediction = int(value >= threshold)
                finished = time.perf_counter()
                if index >= args.warmup:
                    result["samples"].append(
                        {
                            "id": row["id"],
                            "http_ms": 1000 * (arrived - start),
                            "local_scoring_ms": 1000 * (finished - arrived),
                            "total_ms": 1000 * (finished - start),
                            "score": value,
                            "prediction": prediction,
                        }
                    )
                    save_progress(args.output, result)
        result["latency_ms"] = {
            field: summarize([row[field] for row in result["samples"]])
            for field in ("http_ms", "local_scoring_ms", "total_ms")
        }
        if expected is not None:
            result["reference_matches"] = sum(
                row["prediction"] == label for row, label in zip(result["samples"], expected, strict=True)
            )
            result["reference_predictions"] = expected
            if args.approach == "laya" and result["reference_matches"] != len(rows):
                raise ValueError("Served decisions differ from the frozen native reference")
        result["status"] = "complete"
        save_progress(args.output, result)
        print(json.dumps({"approach": args.approach, "latency_ms": result["latency_ms"]}, indent=2))
    except Exception as error:
        result.update(status="incomplete", error=error_receipt(error))
        save_progress(args.output, result)
        raise


if __name__ == "__main__":
    main()
