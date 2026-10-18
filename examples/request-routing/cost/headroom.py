"""Development-only routing headroom, cost/quality frontiers and feedback curves.

No inference calls. No test metrics or policy promotion. Threshold sweeps on
validation are optimistic ranking diagnostics, explicitly separate from the
thresholds fitted on training data and then applied unchanged to validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import difficulty_evidence
import learn
import learning
import reranker
import study
from cloud import digest

SIZES = (32, 64, 128, 256, 437)
TARGETS = (0.90, 0.95, 0.98, 0.99, 1.0)
SEEDS = tuple(range(10))


def subset(o, indices):
    return {name: getattr(o, name)[indices] for name in ("q_cheap", "q_strong", "usd_cheap", "usd_strong")}


def sweep(o, values, usd):
    """All attainable deterministic cutoffs; equal scores must route together."""
    values, usd = np.asarray(values), np.broadcast_to(usd, len(values))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("need nonempty finite scores")
    order = np.argsort(-values, kind="stable")
    ends = np.r_[np.where(np.diff(values[order]) != 0)[0] + 1, len(order)]
    dq = (o["q_cheap"] - o["q_strong"])[order]
    dc = (o["usd_cheap"] - o["usd_strong"])[order]
    quality = np.r_[o["q_strong"].sum(), o["q_strong"].sum() + np.cumsum(dq)[ends - 1]]
    cost = np.r_[o["usd_strong"].sum(), o["usd_strong"].sum() + np.cumsum(dc)[ends - 1]] + usd.sum()
    return [
        {"tau": tau, "cost_per_1k": float(c * 1000 / len(values)), "quality_vs_strong": float(q / o["q_strong"].sum())}
        for tau, c, q in zip([None, *values[order][ends - 1].tolist()], cost, quality, strict=True)
    ]


def at_quality(points, target):
    eligible = [p for p in points if p["quality_vs_strong"] >= target - 1e-12]
    return min(eligible, key=lambda p: p["cost_per_1k"]) if eligible else None


def apply_threshold(o, values, usd, tau):
    cheap = values >= (math.inf if tau is None else tau)
    return {
        "cost_per_1k": float(1000 * (np.where(cheap, o["usd_cheap"], o["usd_strong"]) + usd).mean()),
        "quality_vs_strong": float(np.where(cheap, o["q_cheap"], o["q_strong"]).sum() / o["q_strong"].sum()),
        "cheap_share": float(cheap.mean()),
    }


def oracle(o, target):
    """Fractional perfect-information cost lower bound; routing overhead excluded.

    Start with the cheaper answer per request (ties prefer higher quality), then
    buy positive quality gains in ascending dollars per quality unit. At most
    one request is fractional. This solves the two-arm fractional knapsack,
    including cases where the nominal strong model is cheaper or worse.
    """
    qc, qs, cc, cs = (o[k] for k in ("q_cheap", "q_strong", "usd_cheap", "usd_strong"))
    cheap = (cc < cs) | ((cc == cs) & (qc >= qs))
    quality = float(np.where(cheap, qc, qs).sum())
    cost = float(np.minimum(cc, cs).sum())
    gains = np.where(cheap, qs - qc, qc - qs)
    extra = np.abs(cs - cc)
    needed = max(0.0, target * qs.sum() - quality)
    fractional = False
    for i in sorted(np.where(gains > 0)[0], key=lambda i: extra[i] / gains[i]):
        take = min(1.0, needed / gains[i])
        cost += take * extra[i]
        quality += take * gains[i]
        needed -= take * gains[i]
        fractional |= 1e-12 < take < 1 - 1e-12
        if needed < 1e-12:
            break
    if needed > 1e-9:
        return None
    return {
        "cost_per_1k": 1000 * cost / len(qc),
        "quality_vs_strong": quality / qs.sum(),
        "fractional": bool(fractional),
    }


def random_baseline(o, target):
    """Expected cost of a request-independent mixture, with no routing charge."""
    cq = o["q_cheap"].sum() / o["q_strong"].sum()
    p = 1.0 if cq >= target else min(1.0, max(0.0, (1 - target) / (1 - cq)))
    # Always strong can be cheaper than the mixture for unusual model pairs.
    cheap_cost, strong_cost = o["usd_cheap"].mean(), o["usd_strong"].mean()
    if cheap_cost > strong_cost:
        p = 0.0
    return {
        "cost_per_1k": float(1000 * (p * cheap_cost + (1 - p) * strong_cost)),
        "quality_vs_strong": float(1 - p + p * cq),
    }


def order_training(s, groups, seed):
    train = groups["train"].copy()
    if seed:
        train.sort(key=lambda i: hashlib.sha256(f"routing-curve:{seed}:{s.ids[i]}".encode()).hexdigest())
    return learning.balanced(train, s.strata, len(train))


def knn(s, o, bank, indices):
    labels = study.labels(o.q_cheap[bank], o.q_strong[bank])
    similarities = s.vectors[indices] @ s.vectors[bank].T
    return np.array(
        [
            study.knn_score(row[np.asarray(bank) != i], labels[np.asarray(bank) != i])
            for i, row in zip(indices, similarities, strict=True)
        ]
    )


def ridge(s, o, bank, indices, cost=False):
    """Select regularization by training LOO error, without validation labels."""
    target = o.usd_strong[bank] - o.usd_cheap[bank] if cost else 1 + o.q_cheap[bank] - o.q_strong[bank]
    fits = [learn.fit(s.vectors[bank], target, penalty) for penalty in learn.PENALTIES]
    losses = [float(np.mean((fit[2] - target) ** 2)) for fit in fits]
    chosen = int(np.argmin(losses))
    weights, bias, loo = fits[chosen]
    values = s.vectors[indices] @ weights + bias
    positions = {i: j for j, i in enumerate(indices)}
    for i, value in zip(bank, loo, strict=True):
        values[positions[i]] = value
    return values, {"penalty": learn.PENALTIES[chosen], "loo_mse": losses}


def gepa_recordings(directory, s, indices):
    difficulty_evidence.verify_evidence(directory)
    selection = json.loads((directory / "selection.json").read_text())
    rows = study.read_jsonl(directory / "scores.jsonl")
    by_key = {(r["candidate"], r["id"]): r for r in rows}
    result = {}
    for name, policy in (("fixed_reranker", "static64"), ("gepa437", "gepa437")):
        candidate = {k: v for k, v in selection["policies"][policy].items() if k != "tau"}
        key = digest(candidate)
        selected = [by_key[(key, s.ids[i])] for i in indices]
        result[name] = (np.array([r["score"] for r in selected]), np.array([r["router_usd"] for r in selected]))
    return result


def experiment(s, directory, difficulty_dir=None, seeds=SEEDS, sizes=SIZES):
    groups = learning.split(s)
    indices = groups["train"] + groups["validation"]
    ntrain = len(groups["train"])
    positions = {i: j for j, i in enumerate(indices)}
    validation = np.arange(ntrain, len(indices))
    embedding_usd = s.credits[indices] * study.SIE_USD_PER_CREDIT
    recordings = {}
    if difficulty_dir:
        recordings = gepa_recordings(difficulty_dir, s, indices)
    else:
        recorded = reranker.load(study.EVIDENCE / "reranker", s.ids)
        recordings["fixed_reranker"] = (recorded.scores[indices], recorded.usd[indices])
    plan = {
        "study_revision": study.REVISION,
        "splits": {k: [s.ids[i] for i in groups[k]] for k in ("train", "validation")},
        "sizes": list(sizes),
        "seeds": list(seeds),
        "targets": TARGETS,
        "method": "Nested stratum-balanced feedback orders; seed 0 reproduces the original initial64. Ridge penalty selected by training leave-one-out MSE. Cutoffs fitted on training LOO scores, never validation.",
        "frontiers": "Separate ex-post validation cutoff sweeps are optimistic ranking diagnostics, not deployable-policy results. No interpolation between thresholds.",
        "oracle": "Perfect-information fractional lower bound, excluding routing overhead. Knows counterfactual grades and costs; not an achievable policy.",
        "gepa": "Optional published candidate trained on all 437 and selected on this validation set. Endpoint only, optimistic selection bias; no invented GEPA learning curve.",
        "cost_aware": "Development follow-up after the first curves: independent ridge predictors for quality gap and incremental answer cost, each selected by training LOO MSE. Rank expected quality gain / predicted extra cost; floor predicted cost at 10% of mean training extra cost. Never use actual validation costs as a feature.",
        "limits": "No test outcomes or metrics used. Development benchmark reused; 10 feedback orders are not independent validation datasets. Historical cost excludes feedback collection and optimization.",
    }
    learn.write(directory / "plan.json", plan)
    result = {"pairs": {}}
    for pair, o in s.pairs.items():
        dev = subset(o, indices)
        val = {k: v[validation] for k, v in dev.items()}
        row = {
            "always_strong": {"cost_per_1k": float(1000 * val["usd_strong"].mean()), "quality_vs_strong": 1.0},
            "always_cheap": {
                "cost_per_1k": float(1000 * val["usd_cheap"].mean()),
                "quality_vs_strong": float(val["q_cheap"].sum() / val["q_strong"].sum()),
            },
            "oracle": {str(t): oracle(val, t) for t in TARGETS},
            "random": {str(t): random_baseline(val, t) for t in TARGETS},
            "outcomes": {
                "strong_better": int(sum(val["q_strong"] > val["q_cheap"])),
                "equal": int(sum(val["q_strong"] == val["q_cheap"])),
                "cheap_better": int(sum(val["q_strong"] < val["q_cheap"])),
            },
            "runs": [],
            "frontiers": {},
        }
        for seed in seeds:
            order = order_training(s, groups, seed)
            frozen = knn(s, o, order[:64], indices)
            for size in sizes:
                bank = order[:size]
                fit = np.array([positions[i] for i in bank])
                values, metadata = ridge(s, o, bank, indices)
                spend, cost_metadata = ridge(s, o, bank, indices, cost=True)
                floor = max(1e-9, float(np.mean(o.usd_strong[bank] - o.usd_cheap[bank])) * 0.1)
                cost_aware = (values - 1) / np.maximum(spend, floor)
                variants = {
                    "knn": (knn(s, o, bank, indices), embedding_usd),
                    "ridge": (values, embedding_usd),
                    "cost_aware_ridge": (cost_aware, embedding_usd),
                    "fixed_reranker": recordings["fixed_reranker"],
                }
                if size >= 64:
                    variants["frozen_knn64_recalibrated"] = (frozen, embedding_usd)
                if size == len(groups["train"]) and pair == "openai" and "gepa437" in recordings:
                    variants["gepa"] = recordings["gepa437"]
                for name, (values, usd) in variants.items():
                    training_points = sweep({k: v[fit] for k, v in dev.items()}, values[fit], usd[fit])
                    validation_points = sweep(val, values[validation], usd[validation])
                    if seed == 0 and (size in (64, len(groups["train"]))):
                        row["frontiers"][f"{name}{size}"] = validation_points
                    for target in TARGETS:
                        chosen = at_quality(training_points, target)
                        row["runs"].append(
                            {
                                "seed": seed,
                                "size": size,
                                "method": name,
                                "target": target,
                                "fit_tau": chosen["tau"],
                                "validation": apply_threshold(val, values[validation], usd[validation], chosen["tau"]),
                                "matched_validation": at_quality(validation_points, target),
                                "ridge_fit": metadata if name in ("ridge", "cost_aware_ridge") else None,
                                "cost_fit": cost_metadata | {"floor": floor} if name == "cost_aware_ridge" else None,
                            }
                        )
        result["pairs"][pair] = row
        print(f"Finished {pair}: {len(row['runs'])} development comparisons", flush=True)
    learn.write(directory / "results.json", result)
    return result


def report(results):
    lines = [
        "# Development headroom and feedback curves",
        "",
        "Validation only: 146 requests. Costs are historical USD / 1,000. No new test result.",
        "",
    ]
    for pair, data in results["pairs"].items():
        lines.extend(
            [
                f"## {pair}",
                "",
                "At least 98% quality: ex-post cutoff sweeps; optimistic diagnostics, not frozen policies.",
                "",
                "| Scorer | Cost / 1,000 | Actual quality |",
                "| --- | ---: | ---: |",
            ]
        )
        comparisons = {
            "Always strong": data["always_strong"],
            "Random mixture": data["random"]["0.98"],
            "Perfect-information fractional bound": data["oracle"]["0.98"],
        }
        comparisons.update({name: at_quality(points, 0.98) for name, points in data["frontiers"].items()})
        for name, row in comparisons.items():
            lines.append(f"| {name} | ${row['cost_per_1k']:.4f} | {100 * row['quality_vs_strong']:.2f}% |")
        lines.extend(
            [
                "",
                "Learning curves at at least 98% validation quality. Each cutoff is swept on validation; these measure ranking potential, not a deployable policy.",
                "",
                "| Method | Feedback | Mean cost / 1,000 | Order range |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in ("knn", "ridge", "cost_aware_ridge", "fixed_reranker"):
            for size in sorted({r["size"] for r in data["runs"]}):
                costs = [
                    r["matched_validation"]["cost_per_1k"]
                    for r in data["runs"]
                    if r["method"] == method and r["size"] == size and r["target"] == 0.98
                ]
                if costs:
                    lines.append(f"| {method} | {size} | ${np.mean(costs):.4f} | ${min(costs):.4f}–${max(costs):.4f} |")
        lines.extend(
            [
                "",
                "Training-fitted 98% cutoffs, applied unchanged to validation. Means across nested feedback orders; target-hit counts are descriptive, not independent trials. At 437 examples the training set is identical across orders. GEPA is one previously validation-selected endpoint, not ten trials.",
                "",
                "| Method | Feedback | Cost / 1,000 | Quality | Orders reaching 98% |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        methods = sorted({r["method"] for r in data["runs"]})
        for method in methods:
            for size in sorted({r["size"] for r in data["runs"]}):
                rows = [
                    r["validation"]
                    for r in data["runs"]
                    if r["method"] == method and r["size"] == size and r["target"] == 0.98
                ]
                if rows:
                    lines.append(
                        f"| {method} | {size} | ${np.mean([r['cost_per_1k'] for r in rows]):.4f} | {100 * np.mean([r['quality_vs_strong'] for r in rows]):.2f}% | {sum(r['quality_vs_strong'] >= 0.98 - 1e-12 for r in rows)}/{len(rows)} |"
                    )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=study.HERE / "run-output/headroom")
    parser.add_argument(
        "--difficulty",
        type=Path,
        help="verified published difficulty evidence; use matched native seed and GEPA scores",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = experiment(study.load_study(), args.output, args.difficulty)
    summary = report(result)
    (args.output / "REPORT.md").write_text(summary + "\n")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
