"""Learn a cost router from graded feedback on existing embeddings; no inference calls.

Thresholds are fitted using leave-one-out training predictions. Validation
selects among four ridge penalties. We freeze the exact model AND threshold
before evaluating test requests. Unlike the earlier pilot, there is no
post-selection recalibration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import learning
import reranker
import score
import study

PENALTIES = (0.01, 0.1, 1.0, 10.0)
POLICY_TYPE = "embedding-linear-v1"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit(vectors, targets, penalty):
    """Ridge with an unpenalized intercept and exact leave-one-out predictions."""
    if penalty <= 0 or len(vectors) < 2 or len(vectors) != len(targets):
        raise ValueError("need positive penalty and at least two aligned training rows")
    xmean, ymean = vectors.mean(axis=0), targets.mean()
    centered = vectors - xmean
    gram = centered @ centered.T
    inverse = np.linalg.inv(gram + penalty * np.eye(len(vectors)))
    weights = centered.T @ inverse @ (targets - ymean)
    bias = float(ymean - xmean @ weights)
    fitted = vectors @ weights + bias
    leverage = 1 / len(vectors) + np.diag(gram @ inverse)
    loo = targets - (targets - fitted) / (1 - leverage)
    return weights, bias, loo


def policy_score(policy, vector):
    vector = study.normalise(np.asarray(vector, dtype=np.float64))
    weights = np.asarray(policy["weights"], dtype=np.float64)
    if vector.ndim != 1 or vector.shape != weights.shape:
        raise ValueError("embedding dimensions do not match the learned router")
    return float(vector @ weights + policy["bias"])


def validate_policy(policy, pair):
    weights = np.asarray(policy.get("weights", []), dtype=np.float64)
    if (
        policy.get("type") != POLICY_TYPE
        or "tau" not in policy
        or policy.get("model") != study.EMBEDDER
        or policy.get("pair") != pair
        or weights.ndim != 1
        or not len(weights)
        or not np.isfinite(weights).all()
        or type(policy.get("bias")) not in (float, int)
        or not math.isfinite(policy["bias"])
        or (
            policy.get("tau") is not None
            and (type(policy["tau"]) not in (float, int) or not math.isfinite(policy["tau"]))
        )
    ):
        raise ValueError("invalid learned embedding policy")


def metrics(o, values, usd, tau, indices):
    mask = learning.mask(len(values), indices)
    quality, total, cheap = score.routed(o, values, usd, tau)
    return score.summarize(o, quality, total, cheap, mask) | {
        "embedding_per_1k": float(1000 * usd[mask].mean()),
        "generation_per_1k": float(1000 * (total[mask] - usd[mask]).mean()),
        "strong_requests": int(sum(~cheap[mask])),
    }


def uncertainty(s, o, indices, before, after, usd):
    rng = np.random.default_rng(61)
    groups = [[i for i in indices if s.strata[i] == k] for k in sorted(set(s.strata[indices]))]
    samples = np.concatenate([rng.choice(g, (4000, len(g))) for g in groups], axis=1)
    qa, ca, _ = score.routed(o, before[0], usd, before[1])
    qb, cb, _ = score.routed(o, after[0], usd, after[1])
    cost_gain = 100 * (1 - cb[samples].mean(axis=1) / ca[samples].mean(axis=1))
    quality_gain = 100 * (qb[samples].mean(axis=1) - qa[samples].mean(axis=1)) / o.q_strong[samples].mean(axis=1)
    return {
        "cost_reduction_pct_95ci": np.quantile(cost_gain, [0.025, 0.975]).tolist(),
        "quality_change_pp_95ci": np.quantile(quality_gain, [0.025, 0.975]).tolist(),
        "method": "paired stratified percentile bootstrap",
        "resamples": 4000,
        "seed": 61,
    }


def experiment(s, directory, initial_size=64, recording=None):
    groups = learning.split(s, initial_size)
    usd = s.credits * study.SIE_USD_PER_CREDIT
    train = np.array(groups["train"])
    write(
        directory / "plan.json",
        {
            "study_revision": study.REVISION,
            "splits": {k: [s.ids[i] for i in groups[k]] for k in ("initial", "train", "validation", "test")},
            "unused_partition": "The 146 former threshold-calibration rows are not used.",
            "target": "1 + cheap quality - strong quality (signed difference, not clipped)",
            "penalties": PENALTIES,
            "threshold": "98% training quality target, exact leave-one-out predictions; never refit after selection",
            "selection": "Cheapest validation candidate cheaper than static kNN64, retaining at least 98% of strong "
            "quality with no quality regression versus static. Otherwise keep static.",
            "limitations": "Development used the published benchmark. This is not a new independent test population.",
        },
    )
    selections, variants, policies = {}, {}, {}
    for pair, o in s.pairs.items():
        static = learning.bank_scores(s, pair, groups["initial"])
        tau = score.choose_tau(o, static, usd, learning.mask(len(s.ids), groups["initial"]))
        candidates = [
            {
                "name": "static knn64",
                "tau": None if math.isinf(tau) else tau,
                "validation": metrics(o, static, usd, tau, groups["validation"]),
            }
        ]
        variants[pair] = [(static, tau)]
        policies[pair] = [None]
        target = 1 + o.q_cheap - o.q_strong
        for penalty in PENALTIES:
            weights, bias, loo = fit(s.vectors[train], target[train], penalty)
            values = s.vectors @ weights + bias
            values[train] = loo
            tau = score.choose_tau(o, values, usd, learning.mask(len(s.ids), groups["train"]))
            candidates.append(
                {
                    "name": f"learned linear, penalty={penalty:g}",
                    "tau": None if math.isinf(tau) else tau,
                    "validation": metrics(o, values, usd, tau, groups["validation"]),
                }
            )
            variants[pair].append((values, tau))
            policies[pair].append(
                {
                    "type": POLICY_TYPE,
                    "model": study.EMBEDDER,
                    "pair": pair,
                    "weights": weights.tolist(),
                    "bias": bias,
                    "tau": None if math.isinf(tau) else tau,
                    "penalty": penalty,
                    "training_ids": [s.ids[i] for i in train],
                    "study_revision": study.REVISION,
                }
            )
        eligible = [
            i
            for i in range(1, len(candidates))
            if learning.improvement(candidates[i]["validation"], candidates[0]["validation"])
        ]
        chosen = min(eligible, key=lambda i: candidates[i]["validation"]["cost_per_1k"]) if eligible else 0
        selections[pair] = {"candidates": candidates, "chosen": chosen}
        if recording is not None:
            fixed_tau = score.choose_tau(
                o, recording.scores, recording.usd, learning.mask(len(s.ids), groups["initial"])
            )
            selections[pair]["fixed_reranker_tau"] = None if math.isinf(fixed_tau) else fixed_tau
    # Every pair's model and cutoff are frozen before inspecting any test outcomes.
    write(directory / "selection.json", selections)
    results = {}
    for pair, selection in selections.items():
        o, chosen = s.pairs[pair], selection["chosen"]
        static, updated = variants[pair][0], variants[pair][chosen]
        result = {
            "promoted": chosen != 0,
            "test_n": len(groups["test"]),
            "static": metrics(o, static[0], usd, static[1], groups["test"]),
            "updated": metrics(o, updated[0], usd, updated[1], groups["test"]),
            "uncertainty": uncertainty(s, o, groups["test"], static, updated, usd),
        }
        result["cost_reduction_pct"] = 100 * (1 - result["updated"]["cost_per_1k"] / result["static"]["cost_per_1k"])
        results[pair] = result
        if recording is not None:
            fixed_tau = selection["fixed_reranker_tau"]
            row = metrics(
                o, recording.scores, recording.usd, math.inf if fixed_tau is None else fixed_tau, groups["test"]
            )
            row["reranker_per_1k"] = row.pop("embedding_per_1k")
            result["fixed_reranker"] = row
            write(
                directory / f"policy-reranker-{pair}.json",
                {
                    "model": reranker.MODEL,
                    "profile": "default",
                    "pair": pair,
                    "tau": fixed_tau,
                    "description": recording.spec["query"]["text"],
                    "instruction": recording.spec["instruction"],
                },
            )
        if chosen:
            write(directory / f"policy-{pair}.json", policies[pair][chosen])
    write(directory / "results.json", results)
    return results


def report(results):
    print("Held-out results. Costs are dollars per 1,000: embedding + downstream generation.")
    print(f"{'Pair / router':<26} {'Routing':>10} {'Generation':>11} {'Total':>9} {'Quality':>9}")
    for pair, result in results.items():
        for name in ("static", "updated"):
            row = result[name]
            print(
                f"{pair + ' / ' + name:<26} {row['embedding_per_1k']:10.4f} "
                f"{row['generation_per_1k']:11.4f} {row['cost_per_1k']:9.4f} "
                f"{100 * row['quality_vs_strong']:8.2f}%"
            )
        interval = result["uncertainty"]["cost_reduction_pct_95ci"]
        print(
            f"  Cost reduction: {result['cost_reduction_pct']:.2f}% (paired 95% interval {interval[0]:.2f} to {interval[1]:.2f}%)."
        )
        if "fixed_reranker" in result:
            row = result["fixed_reranker"]
            print(
                f"{pair + ' / fixed reranker':<26} {row['reranker_per_1k']:10.4f} "
                f"{row['generation_per_1k']:11.4f} {row['cost_per_1k']:9.4f} "
                f"{100 * row['quality_vs_strong']:8.2f}%"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=study.HERE / "run-output/learned-router")
    args = parser.parse_args()
    s = study.load_study()
    args.output.mkdir(parents=True, exist_ok=False)
    recording = reranker.load(study.EVIDENCE / "reranker", s.ids)
    report(experiment(s, args.output, recording=recording))
    print(f"Frozen policies and reproducible results saved in {args.output}; no model calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
