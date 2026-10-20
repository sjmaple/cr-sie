"""Build and use a router that trades answer-model cost against answer quality.

fetch: download the public data and recorded results
replay: reproduce the recorded cost and quality tables on CPU
prepare: create a separate folder for your training runs
fit-native: adapt the Laya decision model with LoRA on a CUDA GPU
fit-embedding: fit two small classifiers on saved Qwen embeddings on CPU
fit-teacher: compare teacher and benchmark labels using the same saved inputs
evaluate: measure your trained router on the public test recordings
answer: route a new question and request its answer through SIE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import arithmetic_replay
import arithmetic_route
import evidence

COMMANDS = ("fetch", "replay", "prepare", "fit-native", "fit-embedding", "fit-teacher", "evaluate", "answer")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    arguments = sys.argv[1:] if argv is None else argv
    args, remaining = parser.parse_args(arguments[:1]), arguments[1:]
    if args.command not in ("fetch", "answer"):
        arithmetic_replay.main([args.command, *remaining])
        return
    command = argparse.ArgumentParser(prog=f"router.py {args.command}")
    command.add_argument("--study", type=Path, default=Path("evidence"))
    if args.command == "fetch":
        command.add_argument("--records", action="store_true", help="Include raw answer recordings for auditing")
        command.add_argument(
            "--weights", action="store_true", help="Include recorded native weights for local inference"
        )
        selected = command.parse_args(remaining)
        evidence.fetch(
            selected.study, arithmetic_study=True,
            solver_records=selected.records, solver_weights=selected.weights,
        )
        return
    command.add_argument("--run", type=Path, help="Your prepared/trained folder; omit to use the recorded router")
    command.add_argument("--population", choices=arithmetic_replay.POPULATIONS, default="expanded")
    command.add_argument("--method", choices=("embedding", "lora", "laya", "teacher"), default="embedding")
    command.add_argument("--question", required=True)
    command.add_argument("--target", type=int, choices=(0, 1, 2, 5), default=1)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--self-hosted", action="store_true")
    command.add_argument(
        "--device", choices=("cpu", "cuda"),
        default="cpu", help="Local native inference device"
    )
    selected = command.parse_args(remaining)
    result = arithmetic_route.answer(
        selected.study, selected.output, selected.question, run=selected.run,
        population=selected.population, method=selected.method, target=selected.target,
        self_hosted=selected.self_hosted, device=selected.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
