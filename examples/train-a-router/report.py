"""Recompute metrics from recorded predictions, retaining every declared seed."""

from __future__ import annotations

import json

import numpy as np

from data import digest, file_hash
from policies import metrics, paired


def paper_interval(rows, before, after):
    """Descriptive sensitivity check for dependence within a competition-year."""
    y = np.array([r["label"] for r in rows])
    delta = (np.asarray(after) == y).astype(int) - (np.asarray(before) == y).astype(int)
    groups = sorted({r["paper_group"] for r in rows})
    masks = [np.array([r["paper_group"] == g for r in rows]) for g in groups]
    totals = np.array([delta[m].sum() for m in masks])
    sizes = np.array([m.sum() for m in masks])
    draws = np.random.default_rng(187).integers(0, len(groups), (10000, len(groups)))
    samples = 100 * totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {
        "groups": len(groups),
        "gain_pp_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "method": "paired competition-year cluster bootstrap; descriptive, few groups",
    }


def summarize(dataset, embeddings, native_root):
    rows = dataset["splits"]["test"]
    y = np.array([r["label"] for r in rows])
    ids = [r["id"] for r in rows]
    if embeddings["ids"] != ids or embeddings["dataset_sha256"] != digest(dataset):
        raise ValueError("Embedding evaluation uses another population")
    result = {
        "n": len(rows),
        "embedding": embeddings["metrics"],
        "primary": embeddings["primary_comparison"],
        "native": {},
        "native_paired": {},
    }
    native_predictions = {}
    for seed in (89, 137, 211):
        run = native_root / f"seed-{seed}"
        frozen = json.loads((run / "frozen.json").read_text())
        plan = json.loads((run / "plan.json").read_text())
        record = json.loads((run / "evaluation-test.json").read_text())
        if (
            record["ids"] != ids
            or record["data_sha256"] != file_hash(native_root.parent / "test.json")
            or record["frozen_sha256"] != file_hash(run / "frozen.json")
            or frozen["plan_sha256"] != file_hash(run / "plan.json")
            or plan["config"]["seed"] != seed
        ):
            raise ValueError("Native evaluation selection or population differs")
        native_predictions[seed] = {}
        for variant in ("base", "full", "lora"):
            measured = record["models"][variant]
            if measured["labels"] != y.tolist():
                raise ValueError("Native labels differ")
            if not np.isfinite(measured["scores"]).all():
                raise ValueError("Non-finite native scores")
            prediction = (np.array(measured["scores"]) >= frozen["selected"][variant]["threshold"]).astype(int)
            if prediction.tolist() != measured["predictions"]:
                raise ValueError("Native decisions differ from frozen cutoff")
            result["native"][f"{variant}-{seed}"] = metrics(y, prediction)
            native_predictions[seed][variant] = prediction
        result["native_paired"][str(seed)] = paired(
            y, native_predictions[seed]["full"], native_predictions[seed]["lora"]
        )
        result["native_paired"][str(seed)]["paper_sensitivity"] = paper_interval(
            rows, native_predictions[seed]["full"], native_predictions[seed]["lora"]
        )
    result["native_seed_summary"] = {
        variant: {
            "accuracy_mean": float(np.mean(values)),
            "accuracy_min": float(min(values)),
            "accuracy_max": float(max(values)),
            "seeds": [89, 137, 211],
        }
        for variant in ("base", "full", "lora")
        for values in [[result["native"][f"{variant}-{seed}"]["accuracy"] for seed in (89, 137, 211)]]
    }
    # A source-family control must face the same training-only rule as learned models.
    training = dataset["splits"]["train"]
    majority = {
        tag: int(np.mean([r["label"] for r in training if r["contest"] == tag]) >= 0.5)
        for tag in {r["contest"] for r in training}
    }
    result["contest_only_control"] = metrics(y, [majority[r["contest"]] for r in rows])
    result["primary_paper_sensitivity"] = paper_interval(
        rows, embeddings["predictions"]["knn64"], embeddings["predictions"]["ridge512"]
    )
    result["controls"] = {
        f"{before}_to_{after}": paired(y, embeddings["predictions"][before], embeddings["predictions"][after])
        for before, after in (("knn64", "ridge64"), ("ridge64", "ridge512"), ("knn512", "ridge512"))
    }
    result["per_contest"] = {
        tag: {name: metrics(y[mask], np.asarray(p)[mask]) for name, p in embeddings["predictions"].items()}
        for tag in sorted({r["contest"] for r in rows})
        for mask in [np.array([r["contest"] == tag for r in rows])]
    }
    return result


def display(result):
    print(f"Frozen chronological evaluation: {result['n']} problems")
    for name, measured in (result["embedding"] | result["native"]).items():
        print(f"{name:14s} accuracy {measured['accuracy']:7.2%} balanced {measured['balanced_accuracy']:7.2%}")
    print("Primary comparison:", json.dumps(result["primary"]))
