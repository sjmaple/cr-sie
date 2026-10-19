"""Download, learn, evaluate and use a difficulty router in one environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evidence
from data import LABELS, audit, render_request, write
from policies import evaluate_suite, scores, train_suite
from report import display, summarize
from serving import collect, embed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fetch", "replay", "train", "evaluate", "collect", "route"):
        child = sub.add_parser(command)
        child.add_argument("--evidence", type=Path, default=None if command == "fetch" else Path("evidence"))
        if command == "fetch":
            child.add_argument("--weights", choices=("full", "lora", "both"))
            child.add_argument(
                "--lora-study",
                action="store_true",
                help="Fetch the LoRA training study; defaults to --evidence run-output/lora-study",
            )
        elif command == "collect":
            child.add_argument("--split", choices=("development", "test"), required=True)
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--self-hosted", action="store_true")
        elif command == "route":
            child.add_argument("--policy", type=Path, default=Path("run-output/policies/ridge512.json"))
            child.add_argument(
                "--contest", choices=("AMC8", "AMC10", "AMC12", "AIME", "HMMT-Feb", "HMMT-Nov"), required=True
            )
            child.add_argument("--request", required=True)
        else:
            child.add_argument("--policies", type=Path, default=Path("run-output/policies"))
            if command == "replay":
                child.set_defaults(embeddings=None)
                child.add_argument("--output", type=Path, default=Path("run-output/replay.json"))
            else:
                child.add_argument("--embeddings", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        if args.lora_study and args.weights not in (None, "lora"):
            parser.error("--lora-study supports only --weights lora")
        directory = args.evidence or Path("run-output/lora-study" if args.lora_study else "evidence")
        evidence.fetch(directory, args.weights, lora_study=args.lora_study)
        return
    if args.command == "route":
        policy = json.loads(args.policy.read_text())
        # Check the policy before spending a request.
        scores(policy, np.zeros((1, len(policy.get("weights", policy.get("bank", [[]])[0])))))
        vector, receipt = embed(render_request(args.contest, args.request))
        value = float(scores(policy, vector.reshape(1, -1))[0])
        print(
            json.dumps(
                {
                    "route": LABELS[int(value >= policy["threshold"])],
                    "score": value,
                    "cutoff": policy["threshold"],
                    "usage": receipt,
                    "score_note": "Decision score, not a probability of correctness",
                },
                indent=2,
            )
        )
        return
    evidence.verify(args.evidence)
    dataset = json.loads((args.evidence / "data.json").read_text())
    audit(dataset)
    if args.command == "collect":
        splits = ("train", "validation", "calibration") if args.split == "development" else ("test",)
        collect([r for split in splits for r in dataset["splits"][split]], args.output, self_hosted=args.self_hosted)
        return
    if args.command in ("train", "replay"):
        train_suite(dataset, args.embeddings or args.evidence / "development-embeddings", args.policies)
    if args.command in ("evaluate", "replay"):
        tested = evaluate_suite(dataset, args.embeddings or args.evidence / "test-embeddings", args.policies)
        if args.command == "evaluate":
            print(json.dumps(tested["metrics"], indent=2))
        else:
            recorded = json.loads((args.evidence / "embedding-evaluation.json").read_text())
            if tested != recorded:
                raise ValueError("Recomputed embedding evaluation differs from the published record")
            result = summarize(dataset, tested, args.evidence / "native")
            if result != json.loads((args.evidence / "report.json").read_text()):
                raise ValueError("Recomputed report differs from the published record")
            write(args.output, result)
            display(result)


if __name__ == "__main__":
    main()
