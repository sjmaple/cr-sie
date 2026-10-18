"""Frozen-encoder classifiers with separate selection and calibration data."""

from __future__ import annotations

import json

import numpy as np

from data import PROTOCOL, audit, digest, file_hash, write
from serving import INSTRUCTION, MODEL, REVISION, vectors

PENALTIES = (0.01, 0.1, 1.0, 10.0)


def ridge(x, y, penalty):
    if penalty <= 0 or x.ndim != 2 or len(x) != len(y) or len(y) < 2:
        raise ValueError("Need positive regularization and aligned training data")
    xmean, ymean = x.mean(axis=0), y.mean()
    centered = x - xmean
    weights = centered.T @ np.linalg.solve(centered @ centered.T + penalty * np.eye(len(x)), y - ymean)
    return weights, float(ymean - xmean @ weights)


def knn(bank, labels, queries, k=5):
    if not 1 <= k <= len(bank):
        raise ValueError("Invalid neighbor count")
    # Stable sorting makes ties follow the declared training order.
    indices = np.argsort(-(queries @ bank.T), axis=1, kind="stable")[:, :k]
    return labels[indices].mean(axis=1)


def cutoff(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if len(scores) != len(labels) or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration observations")
    values = np.unique(scores)
    candidates = np.r_[
        np.nextafter(values[0], -np.inf), (values[:-1] + values[1:]) / 2, np.nextafter(values[-1], np.inf)
    ]

    # Balanced accuracy; ties prefer the default 0.5 score boundary.
    def quality(t):
        prediction = scores >= t
        return np.mean([np.mean(prediction[labels == label] == label) for label in (0, 1)])

    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Calibration needs both classes")
    return float(max(candidates, key=lambda t: (quality(t), -abs(t - 0.5), -t)))


def metrics(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    if y.shape != p.shape or not len(y):
        raise ValueError("Unaligned predictions")
    return {
        "n": len(y),
        "accuracy": float(np.mean(p == y)),
        "errors": int(np.sum(p != y)),
        "balanced_accuracy": float(np.mean([np.mean(p[y == v] == v) for v in (0, 1)])),
        "false_routine": int(np.sum((p == 0) & (y == 1))),
        "false_advanced": int(np.sum((p == 1) & (y == 0))),
    }


def paired(y, before, after):
    delta = (np.asarray(after) == y).astype(int) - (np.asarray(before) == y).astype(int)
    counts = np.random.default_rng(187).multinomial(
        len(y), [(delta == -1).mean(), (delta == 0).mean(), (delta == 1).mean()], 10000
    )
    samples = 100 * (counts[:, 2] - counts[:, 0]) / len(y)
    return {
        "gain_pp": float(100 * delta.mean()),
        "gain_pp_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "method": "paired case bootstrap, conditional on trained models",
    }


def train(dataset, recordings, count, method):
    audit(dataset)
    if count not in PROTOCOL["embedding_candidates"]["sizes"] or method not in ("ridge", "knn"):
        raise ValueError("Undeclared training budget or method")
    train_rows = dataset["splits"]["train"][:count]
    validation, calibration = (dataset["splits"][s] for s in ("validation", "calibration"))
    x, vx, cx = (vectors(rows, recordings) for rows in (train_rows, validation, calibration))
    y, vy, cy = (np.array([r["label"] for r in rows]) for rows in (train_rows, validation, calibration))
    policy = {
        "format": "sie-contest-relative-router-v2",
        "dataset_sha256": digest(dataset),
        "method": method,
        "train_ids": [r["id"] for r in train_rows],
        "labels": count,
        "encoder": MODEL,
        "revision": REVISION,
        "instruction": INSTRUCTION,
        "development_vectors_sha256": file_hash(recordings / "vectors.jsonl.gz"),
        "selection": "validation chooses ridge penalty; separate calibration chooses cutoff",
    }
    if method == "ridge":
        candidates = [ridge(x, y, p) for p in PENALTIES]
        losses = [float(np.mean((vx @ w + b - vy) ** 2)) for w, b in candidates]
        selected = int(np.argmin(losses))
        weights, bias = candidates[selected]
        policy.update(weights=weights.tolist(), bias=bias, penalty=PENALTIES[selected], validation_mse=losses)
        calibration_scores = cx @ weights + bias
    else:
        policy.update(k=5, bank=x.tolist(), targets=y.tolist())
        calibration_scores = knn(x, y, cx)
    policy["threshold"] = cutoff(calibration_scores, cy)
    policy["sha256"] = digest(policy)
    return policy


def scores(policy, x):
    if (
        policy["format"] != "sie-contest-relative-router-v2"
        or (policy["encoder"], policy["revision"], policy["instruction"]) != (MODEL, REVISION, INSTRUCTION)
        or policy["method"] not in ("ridge", "knn")
    ):
        raise ValueError("Unsupported policy configuration")
    expected = policy["sha256"]
    if digest({k: v for k, v in policy.items() if k != "sha256"}) != expected:
        raise ValueError("Frozen policy changed")
    if policy["method"] == "ridge":
        return x @ np.array(policy["weights"]) + policy["bias"]
    return knn(np.array(policy["bank"]), np.array(policy["targets"]), x, policy["k"])


def train_suite(dataset, recordings, output):
    output.mkdir(parents=True, exist_ok=False)
    chosen = {}
    for count in (64, 128, 256, 512):
        for method in ("knn", "ridge"):
            name = f"{method}{count}"
            policy = train(dataset, recordings, count, method)
            write(output / f"{name}.json", policy)
            chosen[name] = policy["sha256"]
            print(f"Frozen {name}", flush=True)
    write(
        output / "frozen.json", {"dataset_sha256": digest(dataset), "policies": chosen, "primary": PROTOCOL["primary"]}
    )


def evaluate_suite(dataset, recordings, policies):
    frozen = json.loads((policies / "frozen.json").read_text())
    if frozen["dataset_sha256"] != digest(dataset):
        raise ValueError("Dataset changed after freeze")
    rows = dataset["splits"]["test"]
    x, y = vectors(rows, recordings), np.array([r["label"] for r in rows])
    result, predictions = {}, {}
    for name, expected in frozen["policies"].items():
        policy = json.loads((policies / f"{name}.json").read_text())
        if policy["sha256"] != expected:
            raise ValueError("Selected policy changed")
        predictions[name] = (scores(policy, x) >= policy["threshold"]).astype(int)
        result[name] = metrics(y, predictions[name])
    return {
        "dataset_sha256": digest(dataset),
        "metrics": result,
        "primary_comparison": paired(y, predictions["knn64"], predictions["ridge512"]),
        "predictions": {k: p.tolist() for k, p in predictions.items()},
        "ids": [r["id"] for r in rows],
    }
