"""Compare a fixed prefix of native test decisions with your local SIE server."""

import argparse
import json
import os
from pathlib import Path

from sie_sdk.types import Item

from data import write
from decision import checksum, probability_cutoff
from serving import SingleSendClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("evidence/native/seed-89"))
    parser.add_argument("--evaluation", type=Path, default=Path("evidence/test.json"))
    parser.add_argument("--variant", choices=("full", "lora"), default="full")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SIE_BASE_URL") or not 1 <= args.limit <= 479:
        parser.error("Set SIE_BASE_URL for your checkpoint's server; --limit must be 1..479")
    frozen = json.loads((args.run / "frozen.json").read_text())
    reference = json.loads((args.run / "evaluation-test.json").read_text())
    plan = json.loads((args.run / "plan.json").read_text())
    if (
        reference["data_sha256"] != checksum(args.evaluation)
        or frozen["plan_sha256"] != checksum(args.run / "plan.json")
        or plan["data_sha256"]["questions.json"] != checksum(args.run / "questions.json")
    ):
        raise ValueError("Evaluation inputs, plan or questions differ from frozen evidence")
    rows = json.loads(args.evaluation.read_text())[: args.limit]
    if reference["ids"][: len(rows)] != [r["id"] for r in rows]:
        raise ValueError("Evaluation population differs")
    if reference["frozen_sha256"] != checksum(args.run / "frozen.json"):
        raise ValueError("Frozen selection differs")
    selected = frozen["selected"][args.variant]
    checkpoint = args.run / args.variant
    if checksum(checkpoint / "model.safetensors") != selected["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs")
    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())
    cutoff = probability_cutoff(cfg, selected["threshold"])
    questions = json.loads((args.run / "questions.json").read_text())
    predictions, values = [], []
    with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
        for row in rows:
            reply = client.extract(
                f"example/laya-difficulty-{args.variant}",
                Item(text=row["request"]),
                output_schema=questions,
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=60,
            )
            value = float(reply["data"]["difficulty"]["probabilities"]["advanced"])
            values.append(value)
            predictions.append(int(value >= cutoff))
    expected = reference["models"][args.variant]["predictions"][: len(rows)]
    matches = sum(a == b for a, b in zip(predictions, expected, strict=True))
    result = {
        "variant": args.variant,
        "ids": [r["id"] for r in rows],
        "matched": matches,
        "total": len(rows),
        "native_scores": values,
        "predictions": predictions,
        "reference_predictions": expected,
        "cutoff": cutoff,
    }
    write(args.output, result)
    print(f"{args.variant}: {matches}/{len(rows)} SIE decisions match the frozen GPU reference")
    if matches != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
