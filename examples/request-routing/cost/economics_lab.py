"""Laptop lab: learn a cost-aware router from teacher feedback, then score answers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from sie_sdk.types import Item

import difficulty_policy as embeddings
import economics
import economics_evidence
import economics_policy
import economics_report
import economics_review
import economics_scoring
import fetch
import policy_evidence
import posttrain_evidence
import study
from cloud import digest, usage_of
from run import SingleSendClient


def report(result, directory):
    print("Policy                                  $/1,000   Saving   Answer accuracy   Gap (pp)")
    _, selected = economics_report.comparisons(directory)
    for label, name in selected.items():
        if name not in result["metrics"]:
            continue
        row = result["metrics"][name]
        print(
            f"{label:39s} ${row['usd_per_1000']:6.2f}  {row['saving_vs_strong_pct']:6.1f}%"
            f"     {row['accuracy']:8.2%}     {row['quality_loss_pp']:+6.2f}"
        )
    row = result["metrics"][result["primary"]]
    print(f"Frozen primary: {result['primary']}")
    print(f"Lost correct answers: {row['lost_correct_answers']}; gained: {row['gained_correct_answers']}.")
    print(f"Paired 95% quality-gap interval: {row['quality_loss_95ci_pp']} percentage points.")
    print("The 1/2/5pp budgets were validation settings; the table shows actual held-out quality gaps.")
    print("Serving costs include actual per-request embedding credits. Training costs are separate.")


def learn_from_teacher(source, reference, output, review_paths=None):
    """Stage development evidence only; the fitter receives no answer-key file."""
    economics_evidence.verify_evidence(source)
    policy_evidence.verify_evidence(reference)
    output.mkdir(parents=True, exist_ok=False)
    for name in (
        "plan.json",
        "inputs.json",
        "answer-settings.json",
        "answers-development.jsonl",
        "teacher-input.json",
        "teacher-reviews.jsonl",
    ):
        shutil.copy2(source / name, output / name)
    if review_paths:
        # Validate every supplied record before fitting. Duplicate or missing
        # IDs and different candidate answers are rejected by labels().
        rows = [row for path in review_paths for row in economics.rows(path)]
        (output / "teacher-reviews.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = economics_review.train(
        output, reference, [output / "teacher-input.json"], [output / "teacher-reviews.jsonl"]
    )
    economics_policy.freeze_evaluation(output, ["teacher_review"], "teacher_review")
    print(f"Trained from 512 teacher labels and calibrated on 128 separate teacher reviews: {result['primary']}.")
    print("No gold-answer file was copied into the training directory. No inference calls were made.")
    print(f"Frozen policies: {output / 'review-routing-frozen.json'}")
    return result


def evaluate(source, reference, posttrain, output):
    economics_evidence.verify_evidence(source)
    policy_evidence.verify_evidence(reference)
    posttrain_evidence.verify_evidence(posttrain)
    economics.frozen_policies(output)
    # Test outcomes and independent answer keys enter only after the fit/freeze.
    for name in ("answers-test.jsonl", "gold-for-independent-scoring-only.json", "router-receipts.jsonl"):
        target = output / name
        if target.exists() and target.read_bytes() != (source / name).read_bytes():
            raise ValueError(f"Existing evaluation evidence differs: {name}")
        shutil.copy2(source / name, target)
    shutil.copytree(source / "router-embeddings", output / "router-embeddings", dirs_exist_ok=True)
    economics_policy.evaluate(output, reference, posttrain)
    result = economics_scoring.audit(output)
    report(result, output)
    return result


def replay(source, reference, posttrain, output):
    economics_evidence.verify_evidence(source)
    output.mkdir(parents=True, exist_ok=False)
    for name in economics_evidence.FILES:
        if name.startswith("router-embeddings/"):
            continue
        shutil.copy2(source / name, output / name)
    result = evaluate(source, reference, posttrain, output)
    original = economics.read(source / "economic-results.json")
    for key in ("metrics", "routes", "arrays", "acquisition", "primary_gate"):
        if digest(result[key]) != digest(original[key]):
            raise ValueError(f"Replay differs from the published {key}")
    print("Verified all published metrics, per-question routes and acquisition costs. New inference calls: 0.")


def route(directory, reference, args):
    evaluation, groups = economics.frozen_policies(directory)
    group = "teacher_review"
    name = evaluation["groups"][group]["primary_policy"]
    quality_budget = getattr(args, "quality_budget", 1)
    if quality_budget != 1:
        _, selected = economics_report.comparisons(directory)
        name = selected[f"Teacher review · {quality_budget}pp budget"].removeprefix(group + "_")
    policy = groups[group]["policies"][name]
    metadata = economics.read(reference / "main/dev-embeddings/plan.json")
    if args.self_hosted and not os.environ.get("SIE_BASE_URL"):
        raise ValueError("Self-hosted mode requires SIE_BASE_URL")
    with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
        current = client.get_model(embeddings.MODEL)
        if current["revision"] != metadata["metadata"]["revision"]:
            raise ValueError("Encoder revision differs from training; re-embed and refit first")
        reply = client.encode(
            embeddings.MODEL,
            Item(text=args.request),
            output_types=["dense"],
            instruction=embeddings.EMBED_INSTRUCTION,
            is_query=True,
            wait_for_capacity=False,
            max_oom_retries=0,
            provision_timeout_s=60,
        )
    x = np.asarray(reply["dense"], dtype=float)
    if x.shape != (2560,) or not np.isfinite(x).all() or np.linalg.norm(x) == 0:
        raise ValueError("Invalid embedding")
    x = study.normalise(x)[None, :]
    if policy["method"] == "ridge":
        score = x @ np.asarray(policy["weights"]) + policy["bias"]
    else:
        rows = economics.read(directory / "inputs.json")["train"]
        lookup = {r["id"]: r for r in rows}
        bank = embeddings.vectors([lookup[key] for key in policy["train_ids"]], reference / "main/dev-embeddings")
        score = embeddings.knn(bank, np.asarray(policy["train_targets"]), x)
    ranking = economics_policy.allocate_scores(policy, score, x)[0]
    use_strong = bool(ranking >= economics_policy.threshold(policy["tau"]))
    settings = economics.read(directory / "answer-settings.json")
    arm_name = settings["strong_arm" if use_strong else "cheap_arm"]
    result = {
        "model": settings["arms"][arm_name]["model"],
        "policy": name,
        "validation_quality_budget_pp": quality_budget,
        "budget_note": "A validation selection setting, not a guarantee of the live answer-quality gap",
        "score": float(ranking),
        "score_note": "Uncalibrated routing score, not a probability",
        "embedding_receipt": economics.native_usage(reply) if args.self_hosted else usage_of(reply),
    }
    if args.answer:
        result["generation"] = economics.send(
            {"id": "workshop-live", "request": args.request},
            arm_name,
            settings["arms"][arm_name],
            self_hosted=args.self_hosted,
        )
        if result["generation"]["status"] != "success":
            print(json.dumps(result, indent=2))
            raise RuntimeError("Generation failed; the response was retained above and was not retried")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/economics"))
    parser.add_argument("--reference", type=Path, default=Path("evidence/difficulty-policy"))
    parser.add_argument("--posttrain", type=Path, default=Path("evidence/posttrain"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    for command in ("learn", "evaluate", "replay"):
        item = sub.add_parser(command)
        item.add_argument("--run", type=Path, required=True)
        if command == "learn":
            item.add_argument(
                "--teacher-reviews", type=Path, action="append", help="Use your own complete review shards"
            )
    routing = sub.add_parser("route")
    routing.add_argument("--run", type=Path, required=True)
    routing.add_argument("--request", required=True)
    routing.add_argument("--answer", action="store_true", help="Also call the selected answering model")
    routing.add_argument("--self-hosted", action="store_true")
    routing.add_argument(
        "--quality-budget", type=int, choices=(1, 2, 5), default=1, help="Predeclared validation quality budget (pp)"
    )
    args = parser.parse_args()
    if args.command == "download":
        for path, module in (
            (args.evidence, economics_evidence),
            (args.reference, policy_evidence),
            (args.posttrain, posttrain_evidence),
        ):
            fetch.fetch(path, source=module)
        print("Downloaded and verified all evidence. No model weights or API key required.")
    elif args.command == "learn":
        learn_from_teacher(args.evidence, args.reference, args.run, args.teacher_reviews)
    elif args.command == "evaluate":
        evaluate(args.evidence, args.reference, args.posttrain, args.run)
    elif args.command == "route":
        route(args.run, args.reference, args)
    else:
        replay(args.evidence, args.reference, args.posttrain, args.run)


if __name__ == "__main__":
    main()
