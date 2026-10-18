"""ODSC lab: download, learn from feedback, evaluate, and route a new request."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sie_sdk.types import Item

import difficulty_policy as experiment
import fetch
import learn
import policy_evidence
import study
from cloud import digest, usage_of
from run import SingleSendClient


def evidence(directory):
    policy_evidence.verify_evidence(directory)
    data = json.loads((directory / "main/data.json").read_text())
    experiment.validate(data)
    return data, directory / "main"


def load_policy(path, data, directory):
    policy = json.loads(path.read_text())
    integrity = policy.pop("sha256")
    if digest(policy) != integrity:
        raise ValueError("Policy changed after training")
    if policy["data_sha256"] != digest(data):
        raise ValueError("Policy belongs to different evidence")
    plan = json.loads((directory / "dev-embeddings/plan.json").read_text())
    if (
        policy["encoder"] != plan["model"]
        or policy["revision"] != plan["metadata"]["revision"]
        or policy["instruction"] != plan["instruction"]
    ):
        raise ValueError("Policy embedding configuration differs from evidence")
    return policy


def train(data, directory, labels, output):
    if output.exists():
        raise FileExistsError(f"Choose a new policy path: {output}")
    rows = data["splits"]["train"][:labels]
    x = experiment.vectors(rows, directory / "dev-embeddings")
    y = experiment.target(rows)
    started = time.perf_counter()
    fitted = [learn.fit(x, y, penalty) for penalty in learn.PENALTIES]
    index = int(np.argmin([np.mean((candidate[2] - y) ** 2) for candidate in fitted]))
    weights, bias, loo = fitted[index]
    threshold = experiment.threshold(loo, y)
    seconds = time.perf_counter() - started
    validation = data["splits"]["validation"]
    vx = experiment.vectors(validation, directory / "dev-embeddings")
    quality = experiment.metrics(vx @ weights + bias, experiment.target(validation), threshold)
    plan = json.loads((directory / "dev-embeddings/plan.json").read_text())
    policy = {
        "format": "sie-difficulty-ridge-v1",
        "data_sha256": digest(data),
        "encoder": plan["model"],
        "revision": plan["metadata"]["revision"],
        "instruction": plan["instruction"],
        "is_query": True,
        "labels": labels,
        "train_ids": [row["id"] for row in rows],
        "penalty": learn.PENALTIES[index],
        "weights": weights.tolist(),
        "bias": bias,
        "threshold": threshold,
        "selection": "Regularization and cutoff selected using training leave-one-out predictions only",
        "validation": quality,
    }
    policy["sha256"] = digest(policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(policy, handle, indent=2, allow_nan=False)
    print(f"Trained {len(weights):,} weights + one bias on {labels} feedback labels in {seconds:.3f}s.")
    print(f"Validation accuracy: {quality['accuracy']:.2%}. Frozen policy: {output}")
    print("New inference calls: 0. Run evaluate to reveal held-out performance.")


def evaluate(data, directory, policy):
    rows = data["splits"]["test"]
    x = experiment.vectors(rows, directory / "test-embeddings")
    y = experiment.target(rows)
    values = x @ np.asarray(policy["weights"]) + policy["bias"]
    quality = experiment.metrics(values, y, policy["threshold"])
    predictions = values >= policy["threshold"]
    baseline = np.array(json.loads((directory / "predictions.json").read_text())["knn64"])
    comparison = experiment.paired(y, baseline, predictions)
    print(f"Held-out accuracy: {quality['accuracy']:.2%}; errors: {np.sum(predictions != y)}/{len(y)}.")
    print(
        f"Advanced sent to routine: {quality['false_routine']}; routine sent to advanced: {quality['false_advanced']}."
    )
    lo, hi = comparison["gain_pp_95ci"]
    print(f"Against frozen kNN64: {comparison['gain_pp']:+.2f} points (paired 95% interval {lo:+.2f} to {hi:+.2f}).")
    print(f"Error reduction: {comparison['relative_error_reduction_pct']:.1f}%. New inference calls: 0.")
    return {"test": quality, "comparison": comparison}


def route(data, directory, policy, args):
    if args.recorded_id:
        found = [
            (split, row) for split, rows in data["splits"].items() for row in rows if row["id"] == args.recorded_id
        ]
        if len(found) != 1:
            raise ValueError("Recorded ID not found")
        split, row = found[0]
        folder = "test-embeddings" if split == "test" else "dev-embeddings"
        vector = experiment.vectors([row], directory / folder)[0]
        request, receipt = row["request"], {"mode": "recorded", "new_inference_calls": 0}
    else:
        if args.self_hosted and not os.environ.get("SIE_BASE_URL"):
            raise ValueError("Self-hosted mode requires SIE_BASE_URL")
        request = args.request
        with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
            metadata = client.get_model(policy["encoder"])
            if metadata["revision"] != policy["revision"]:
                raise ValueError("Live encoder revision differs from trained policy; collect vectors and retrain first")
            reply = client.encode(
                policy["encoder"],
                Item(text=request),
                output_types=["dense"],
                instruction=policy["instruction"],
                is_query=True,
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=60,
            )
        vector = np.asarray(reply["dense"], dtype=float)
        if args.self_hosted:
            receipt = {"mode": "self-hosted", "new_inference_calls": 1, "credits": None}
        else:
            receipt = {"mode": "cloud", "new_inference_calls": 1, **usage_of(reply)}
    weights = np.asarray(policy["weights"])
    if vector.shape != weights.shape or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("Invalid embedding vector")
    vector = study.normalise(vector)
    score = float(vector @ weights + policy["bias"])
    result = {
        "request": request,
        "route": "advanced" if score >= policy["threshold"] else "routine",
        "score": score,
        "threshold": policy["threshold"],
        "score_note": "Uncalibrated score, not a probability",
        "receipt": receipt,
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/difficulty-policy"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    learning = sub.add_parser("learn")
    learning.add_argument("--labels", type=int, choices=(64, 128, 256, 512), default=512)
    learning.add_argument("--output", type=Path, required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--policy", type=Path, required=True)
    routing = sub.add_parser("route")
    routing.add_argument("--policy", type=Path, required=True)
    routing.add_argument("--self-hosted", action="store_true")
    query = routing.add_mutually_exclusive_group(required=True)
    query.add_argument("--request")
    query.add_argument("--recorded-id")
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.command == "download":
        fetch.fetch(args.evidence, source=policy_evidence)
        print(f"Verified public evidence at {policy_evidence.REVISION}; no API key needed.")
        return
    data, directory = evidence(args.evidence)
    if args.command == "learn":
        train(data, directory, args.labels, args.output)
    elif args.command == "compare":
        result = experiment.test_stage(data, directory)
        for name, values in result["test"].items():
            print(
                f"{name:18s} {values['accuracy']:7.2%}  {values['n'] - round(values['n'] * values['accuracy']):3d} errors"
            )
        print("All results reconstructed from pinned evidence. GEPA did not improve held-out accuracy.")
    else:
        policy = load_policy(args.policy, data, directory)
        if args.command == "evaluate":
            evaluate(data, directory, policy)
        else:
            route(data, directory, policy, args)


if __name__ == "__main__":
    main()
