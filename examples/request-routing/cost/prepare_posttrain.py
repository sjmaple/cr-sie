"""Prepare decision-model training data and reserve a new confirmation set.

    uv run --frozen --group data python prepare_posttrain.py --output run-output/posttrain-v1

This makes no model calls and performs no training. The existing 512-case test
is retained as a reported reference; an unused official-eval sample is reserved
for confirming the next experiment after its policies have been frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import difficulty_policy as policy
import learn
import policy_evidence
import study
from cloud import digest

QUESTION = {
    "difficulty": {
        "type": "choice",
        "instructions": (
            "Classify the problem's difficulty for a well-prepared mathematics competition student. "
            "Routine means the easier half of competition problems; advanced means the harder half. "
            "Judge the reasoning needed, not just the topic or wording."
        ),
        "criteria": {
            "routine": policy.SEED["routine"],
            "advanced": policy.SEED["advanced"],
        },
    }
}
LABELS = ("routine", "advanced")


def select_confirmation(previous, raw_eval, count=512):
    """Exclude every previously used ID and normalized text before sampling."""
    if count <= 0 or count % 2:
        raise ValueError("confirmation size must be positive and even")
    seen_ids = {r["id"] for rows in previous["splits"].values() for r in rows}
    seen_text = {policy.normalize(r["request"]) for rows in previous["splits"].values() for r in rows}
    groups = [[], []]
    for index, raw in enumerate(raw_eval):
        identity = f"e2h-amc-eval-{index}"
        text = raw["problem"].strip()
        normalized = policy.normalize(text)
        if not normalized or identity in seen_ids or normalized in seen_text:
            continue
        seen_text.add(normalized)
        label = int(raw["rating_quantile"] >= 0.5)
        groups[label].append({"id": identity, "request": text, "label": label})
    for group in groups:
        group.sort(key=lambda r: digest(["posttrain-confirmation-v1", r["id"]]))
        if len(group) < count // 2:
            raise ValueError("not enough unused examples for balanced confirmation")
    result = groups[0][: count // 2] + groups[1][: count // 2]
    return sorted(result, key=lambda r: digest(["posttrain-input-order-v1", r["id"]]))


def training_case(row):
    # Whitelist fields: ratings, source tags and solutions never reach the model.
    return {
        "state": row["request"],
        "questions": QUESTION,
        "expected": {"difficulty": LABELS[row["label"]]},
    }


def write_rows(path, rows):
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(training_case(row)) + "\n")


def prepare(evidence, output):
    import pyarrow.parquet as pq

    policy_evidence.verify_evidence(evidence)
    data = json.loads((evidence / "main/data.json").read_text())
    policy.validate(data)
    frozen = json.loads((evidence / "main/frozen.json").read_text())
    if digest(data) != frozen["data_sha256"]:
        raise ValueError("reference data differs from frozen experiment")
    raw = pq.read_table(evidence / "source/eval.parquet").to_pylist()
    confirmation = select_confirmation(data, raw)
    output.mkdir(parents=True, exist_ok=False)
    # Train/validation exports are separate from all test inputs. Upload only
    # these two files and questions.json to a training worker.
    for split in ("train", "validation"):
        write_rows(output / f"{split}.jsonl", data["splits"][split])
    learn.write(output / "questions.json", QUESTION)
    reserved = output / "reserved-evaluation"
    reserved.mkdir()
    learn.write(reserved / "reference.json", data["splits"]["test"])
    learn.write(reserved / "confirmation.json", confirmation)
    plan = {
        "status": "data prepared; no decision-model training or evaluation performed",
        "source_revision": data["source_revision"],
        "reference_data_sha256": digest(data),
        "reference_frozen_sha256": digest(frozen),
        "confirmation_sha256": digest(confirmation),
        "sampling": "Balanced unused official-eval problems; hash selected; excludes every previous ID and normalized text",
        "counts": {"train": 512, "validation": 128, "reference": 512, "confirmation": 512},
        "ids": {
            "train": [r["id"] for r in data["splits"]["train"]],
            "validation": [r["id"] for r in data["splits"]["validation"]],
            "reference": [r["id"] for r in data["splits"]["test"]],
            "confirmation": [r["id"] for r in confirmation],
        },
        "label_source": "Existing independent human-performance-derived difficulty labels, not teacher outputs",
        "candidates": [
            "Frozen Qwen4B plus ridge512 reference",
            "Laya base, with validation-calibrated decision threshold",
            "Laya full supervised fine-tune, same question and 512 training examples",
            "Laya encoder LoRA plus trained decision head, only after training/export parity is verified",
            "ModernBERT-base fixed-class classifier as a small-encoder control",
        ],
        "model_selection": "Development only: choose training settings/checkpoints on validation; freeze before either test; confirm selected comparisons on previously unused confirmation set",
        "training_recipe_status": "Candidate comparison only. Exact optimizer budgets, seeds, model revisions and acceptance criteria must be frozen before training.",
        "serving": "Laya full checkpoint through existing SIE LayaAdapter; any LoRA must merge into a standard checkpoint and pass output parity before serving",
        "cloud_boundary": "Training runs on separate GPU compute. An inference API key alone does not provide a verified custom-model training/deployment workflow.",
        "success_to_measure": "Improvement over the same decision model before training; accuracy and routing errors versus Qwen ridge; measured batch-1 latency, throughput, training time and memory on matched hardware",
        "files": {
            name: {"sha256": study.sha256((output / name).read_bytes()), "bytes": (output / name).stat().st_size}
            for name in (
                "train.jsonl",
                "validation.jsonl",
                "questions.json",
                "reserved-evaluation/reference.json",
                "reserved-evaluation/confirmation.json",
            )
        },
    }
    learn.write(output / "preparation.json", plan)
    return {"output": str(output), "counts": plan["counts"], "confirmation_sha256": plan["confirmation_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/difficulty-policy"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.evidence, args.output), indent=2))


if __name__ == "__main__":
    main()
