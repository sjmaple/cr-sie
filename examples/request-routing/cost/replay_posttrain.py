"""Reconstruct the completed decision-model and fresh confirmation comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import difficulty_policy as policy
import posttrain
import workshop
from cloud import digest


def replay(directory, reference):
    data, original = workshop.evidence(reference)
    frozen = json.loads((directory / "frozen.json").read_text())
    if frozen["plan_sha256"] != posttrain.sha(directory / "plan.json"):
        raise ValueError("Training plan changed after freeze")
    plan = json.loads((directory / "plan.json").read_text())
    for name, expected in plan["data_sha256"].items():
        if posttrain.sha(directory / name) != expected:
            raise ValueError("Training inputs changed")
    result = {"splits": {}, "selected": frozen["selected"]}
    for split in ("reference", "confirmation"):
        rows = json.loads((directory / f"{split}.json").read_text())
        if split == "reference" and rows != data["splits"]["test"]:
            raise ValueError("Original test split changed")
        y = policy.target(rows)
        evidence = json.loads((directory / f"evaluation-{split}.json").read_text())
        if (
            evidence["frozen_sha256"] != posttrain.sha(directory / "frozen.json")
            or evidence["data_sha256"] != posttrain.sha(directory / f"{split}.json")
            or evidence["ids"] != [row["id"] for row in rows]
        ):
            raise ValueError("Evaluation does not match the frozen policy or dataset")
        metrics, predictions = {}, {}
        for name in ("base", "full", "lora"):
            saved = evidence["models"][name]
            computed = posttrain.metrics(np.array(saved["scores"]), y, frozen["selected"][name]["threshold"])
            if any(saved[key] != value for key, value in computed.items()):
                raise ValueError(f"{split}/{name} metrics or labels changed")
            metrics[name] = {k: v for k, v in computed.items() if k not in ("scores", "labels", "predictions")}
            predictions[name] = np.array(computed["predictions"])
        if split == "confirmation":
            ridge = json.loads((directory / "confirmation-ridge.json").read_text())
            if ridge["data_sha256"] != digest(rows):
                raise ValueError("Ridge confirmation dataset changed")
            original_frozen = json.loads((original / "frozen.json").read_text())
            if ridge["original_frozen_sha256"] != digest(original_frozen):
                raise ValueError("Ridge baseline changed after confirmation")
            tx = policy.vectors(rows, directory / "confirmation-embeddings")
            training = data["splits"]["train"]
            x = policy.vectors(training, original / "dev-embeddings")
            train_y = policy.target(training)
            for name in ("knn64", "ridge64", "ridge512"):
                selected = original_frozen["selection"]["embedding"]["policies"][name]
                if selected["method"] == "knn":
                    values = policy.knn(x[:64], train_y[:64], tx)
                else:
                    values = tx @ np.array(selected["weights"]) + selected["bias"]
                predictions[name] = (values >= selected["tau"]).astype(int)
                if predictions[name].tolist() != ridge["predictions"][name]:
                    raise ValueError("Recomputed ridge confirmation routes differ")
                metrics[name] = {
                    "accuracy": float(np.mean(predictions[name] == y)),
                    "errors": int(np.sum(predictions[name] != y)),
                    "n": len(y),
                }
        else:
            original_predictions = json.loads((original / "predictions.json").read_text())
            for name in ("knn64", "ridge64", "ridge512"):
                predictions[name] = np.array(original_predictions[name])
                metrics[name] = {
                    "accuracy": float(np.mean(predictions[name] == y)),
                    "errors": int(np.sum(predictions[name] != y)),
                    "n": len(y),
                }
        comparisons = {
            f"{a}_to_{b}": policy.paired(y, predictions[a], predictions[b])
            for a, b in (
                ("base", "full"),
                ("base", "lora"),
                ("knn64", "ridge512"),
                ("ridge64", "ridge512"),
                ("ridge512", "full"),
                ("ridge512", "lora"),
            )
        }
        result["splits"][split] = {"metrics": metrics, "comparisons": comparisons}
    return result


def main():
    import posttrain_evidence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/posttrain"))
    parser.add_argument("--reference", type=Path, default=Path("evidence/difficulty-policy"))
    args = parser.parse_args()
    posttrain_evidence.verify_evidence(args.evidence)
    result = replay(args.evidence, args.reference)
    for split, measured in result["splits"].items():
        print(f"\n{split}: 512 held-out problems")
        for name, metrics in measured["metrics"].items():
            print(f"{name:12s} {metrics['accuracy']:7.2%}  {metrics['errors']:3d} errors")
        for name in ("base_to_full", "base_to_lora", "knn64_to_ridge512"):
            delta = measured["comparisons"][name]
            print(f"{name}: {delta['gain_pp']:+.2f} points; {delta['relative_error_reduction_pct']:.1f}% fewer errors")
    print(
        "\nVerified frozen choices, input hashes, predictions, metrics and embedding confirmation. No inference calls."
    )


if __name__ == "__main__":
    main()
