"""Recompute GEPA's handler-routing accuracy and paired uncertainty without inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import intent_evidence
import intent_gepa
import replay_learning
import study
from cloud import digest


def replay(directory):
    data = json.loads((directory / "data.json").read_text())
    intent_gepa.validate_data(data)
    selection = json.loads((directory / "selection.json").read_text())
    expected = json.loads((directory / "results.json").read_text())
    records = study.read_jsonl(directory / "scores.jsonl")
    scores = {(r["candidate"], r["id"]): r["score"] for r in records}
    if len(scores) != len(records) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise ValueError("duplicate or invalid scores")
    candidates = selection["candidates"]
    if candidates[0] != intent_gepa.HANDLERS:
        raise ValueError("seed differs from the existing handler labels")

    def predict(candidate, split):
        rows = data["splits"][split]
        keys = [
            digest({"description": candidate[label], "instruction": intent_gepa.INSTRUCTION})
            for label in intent_gepa.HANDLERS
        ]
        matrix = np.array([[scores[key, row["id"]] for key in keys] for row in rows])
        prediction = matrix.argmax(axis=1)
        tied = (matrix == matrix.max(axis=1, keepdims=True)).sum(axis=1) > 1
        labels = list(intent_gepa.HANDLERS)
        correct = np.array([labels[i] == row["label"] for i, row in zip(prediction, rows, strict=True)]) & ~tied
        return correct

    validation = [float(predict(candidate, "validation").mean()) for candidate in candidates]
    replay_learning.verify(validation, selection["validation"], "validation")
    chosen = min(range(len(candidates)), key=lambda i: (-validation[i], sum(map(len, candidates[i].values())), i))
    if validation[chosen] <= validation[0]:
        chosen = 0
    replay_learning.verify(chosen, selection["chosen"], "chosen")
    replay_learning.verify(chosen, expected["chosen"], "results.chosen")
    before, after = predict(candidates[0], "test"), predict(candidates[chosen], "test")
    replay_learning.verify(
        {"seed": float(before.mean()), "selected": float(after.mean())}, expected["test_accuracy"], "test"
    )
    corrected, regressed = int(sum(~before & after)), int(sum(before & ~after))
    replay_learning.verify(corrected, expected["corrected"], "corrected")
    replay_learning.verify(regressed, expected["regressed"], "regressed")
    calls = study.read_jsonl(directory / "calls.jsonl")
    replay_learning.verify(
        sum(row.get("credits", 0) for row in calls), expected["cloud"]["observed_credits"], "credits"
    )
    rows = data["splits"]["test"]
    rng = np.random.default_rng(53)
    groups = [[i for i, row in enumerate(rows) if row["label"] == label] for label in intent_gepa.HANDLERS]
    samples = np.concatenate([rng.choice(group, (4000, len(group))) for group in groups], axis=1)
    gain = 100 * (after[samples].mean(axis=1) - before[samples].mean(axis=1))
    n = corrected + regressed
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(corrected, regressed) + 1)) / 2**n) if n else 1.0
    result = {
        "test_n": len(rows),
        "seed_accuracy": float(before.mean()),
        "selected_accuracy": float(after.mean()),
        "gain_pp": float(100 * (after.mean() - before.mean())),
        "paired_stratified_95ci_pp": np.quantile(gain, [0.025, 0.975]).tolist(),
        "corrected": corrected,
        "regressed": regressed,
        "mcnemar_exact_two_sided_p": p,
        "bootstrap_seed": 53,
        "bootstrap_resamples": 4000,
        "cloud": expected["cloud"],
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=study.EVIDENCE / "intents")
    args = parser.parse_args()
    if args.run.resolve() == (study.EVIDENCE / "intents").resolve():
        intent_evidence.verify_evidence(args.run)
    replay(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
