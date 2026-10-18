"""Freeze the Qwen kNN development lead, then measure held-out answer outcomes.

This is a research confirmation, not the attendee workflow. No fitting on test.
Replay uses saved vectors. Collection requires SIE and explicitly spends credits.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import encoder_comparison
import headroom
import learn
import learning
import study
from cloud import digest
from run import SingleSendClient


def frozen_policies(s, development):
    qwen = encoder_comparison.load_vectors(s, development)
    groups = learning.split(s)
    policies = []
    for encoder, data in (("qwen", qwen), ("bge", s)):
        for pair, outcomes in s.pairs.items():
            for seed in headroom.SEEDS:
                order = headroom.order_training(s, groups, seed)
                for variant, bank, fit in (
                    ("static64", order[:64], order[:64]),
                    ("recalibrated64", order[:64], order),
                    ("feedback437", order, order),
                ):
                    # Full-bank policy is independent of feedback order. Keep it once.
                    if seed and variant == "feedback437":
                        continue
                    values = headroom.knn(data, outcomes, bank, fit)
                    points = headroom.sweep(
                        headroom.subset(outcomes, fit), values, s.credits[fit] * study.SIE_USD_PER_CREDIT
                    )
                    tau = headroom.at_quality(points, 0.98)["tau"]
                    policies.append(
                        {"encoder": encoder, "pair": pair, "seed": seed, "variant": variant, "bank": bank, "tau": tau}
                    )
    return policies


def freeze(s, development, directory):
    directory.mkdir(parents=True, exist_ok=False)
    source_plan = json.loads((development / "encoding-plan.json").read_text())
    plan = {
        "study_revision": study.REVISION,
        "development_plan_sha256": study.sha256((development / "encoding-plan.json").read_bytes()),
        "development_vectors_sha256": study.sha256((development / "vectors.jsonl.gz").read_bytes()),
        "encoding": source_plan,
        "policies": frozen_policies(s, development),
        "primary_pair": "anthropic",
        "primary_comparison": "Qwen feedback437 versus mean of ten predeclared static64 feedback orders",
        "controls": "Fixed64 bank recalibrated on 437; BGE same procedure; OpenAI secondary pair",
        "cost_basis": "Historical answer costs plus identical archived BGE routing allowance; live embedding receipts separate",
        "success_gate": {
            "cost_reduction_pct_at_least": 25,
            "quality_vs_strong_at_least": 0.98,
            "paired_quality_delta_lower_95ci_pp_at_least": -1,
            "paired_cost_reduction_lower_95ci_above": 0,
        },
        "limitations": "728 requests held out of this development exercise; their outcomes were reported for earlier BGE/reranker experiments. This is not a new independently collected test set. Ten banks share the same requests, not ten independent tests.",
        "test_ids": [s.ids[i] for i in learning.split(s)["test"]],
    }
    learn.write(directory / "frozen.json", plan)
    return plan


def load_test(s, development, directory, plan):
    for name in ("encoding-plan.json", "vectors.jsonl.gz"):
        key = "development_plan_sha256" if name.startswith("encoding") else "development_vectors_sha256"
        if study.sha256((development / name).read_bytes()) != plan[key]:
            raise ValueError("development recording changed after policy freeze")
    qwen = encoder_comparison.load_vectors(s, development)
    source = directory / "test-encoding"
    recording = json.loads((source / "encoding-plan.json").read_text())
    if (source / "incomplete.json").exists():
        raise ValueError("test collection incomplete")
    if recording["ids"] != plan["test_ids"] or recording["options"] != plan["encoding"]["options"]:
        raise ValueError("test encoding differs from the frozen plan")
    if recording["metadata"]["revision"] != plan["encoding"]["metadata"]["revision"]:
        raise ValueError("model revision changed between development and test")
    rows = study.read_jsonl(source / "vectors.jsonl.gz")
    if [r["id"] for r in rows] != plan["test_ids"]:
        raise ValueError("test vector IDs are missing, reordered or duplicated")
    indices = learning.split(s)["test"]
    values = np.asarray([r["vector"] for r in rows])
    raw, vectors = qwen.raw_vectors.copy(), qwen.vectors.copy()
    raw[indices], vectors[indices] = values, study.normalise(values)
    return replace(qwen, raw_vectors=raw, vectors=vectors)


def bootstrap(s, indices, outcomes, before, after):
    rng = np.random.default_rng(67)
    groups = [[j for j, i in enumerate(indices) if s.strata[i] == k] for k in sorted(set(s.strata[indices]))]
    sampled = np.concatenate([rng.choice(g, (4000, len(g))) for g in groups], axis=1)
    qa, ca = before
    qb, cb = after
    strong = outcomes.q_strong[indices]
    cost = 100 * (1 - cb[sampled].mean(axis=1) / ca[sampled].mean(axis=1))
    quality = 100 * (qb[sampled].mean(axis=1) - qa[sampled].mean(axis=1)) / strong[sampled].mean(axis=1)
    return {
        "cost_reduction_pct": float(100 * (1 - cb.mean() / ca.mean())),
        "cost_reduction_pct_95ci": np.quantile(cost, [0.025, 0.975]).tolist(),
        "quality_change_pp": float(100 * (qb.mean() - qa.mean()) / strong.mean()),
        "quality_change_pp_95ci": np.quantile(quality, [0.025, 0.975]).tolist(),
        "method": "4000 paired stratum-preserving bootstrap resamples, seed 67; average banks before resampling",
    }


def analyze(s, development, directory, plan):
    qwen = load_test(s, development, directory, plan)
    indices = learning.split(s)["test"]
    usd = s.credits[indices] * study.SIE_USD_PER_CREDIT
    outputs, records = {}, []
    for policy in plan["policies"]:
        pair, variant, encoder = (policy[k] for k in ("pair", "variant", "encoder"))
        o = s.pairs[pair]
        values = headroom.knn(qwen if encoder == "qwen" else s, o, policy["bank"], indices)
        cheap = values >= (math.inf if policy["tau"] is None else policy["tau"])
        quality = np.where(cheap, o.q_cheap[indices], o.q_strong[indices])
        costs = np.where(cheap, o.usd_cheap[indices], o.usd_strong[indices]) + usd
        outputs.setdefault((pair, encoder, variant), []).append((quality, costs, cheap))
        records.append({**policy, "scores": values.tolist(), "cheap": cheap.tolist()})
    result = {"frozen_sha256": digest(plan), "test_n": len(indices), "pairs": {}}
    for pair, o in s.pairs.items():
        averages, rows = {}, {}
        for (p, encoder, variant), replicates in outputs.items():
            if p != pair:
                continue
            quality, costs, cheap = np.mean(np.asarray(replicates), axis=0)
            name = f"{encoder}_{variant}"
            averages[name] = (quality, costs)
            rows[name] = {
                "cost_per_1k": float(1000 * costs.mean()),
                "quality_vs_strong": float(quality.sum() / o.q_strong[indices].sum()),
                "cheap_share": float(cheap.mean()),
                "feedback_orders": len(replicates),
                "embedding_allowance_per_1k": float(1000 * usd.mean()),
            }
        comparisons = {
            before: bootstrap(s, indices, o, averages[before], averages["qwen_feedback437"])
            for before in ("qwen_static64", "qwen_recalibrated64", "bge_feedback437")
        }
        primary = comparisons["qwen_static64"]
        passed = (
            primary["cost_reduction_pct"] >= 25
            and rows["qwen_feedback437"]["quality_vs_strong"] >= 0.98
            and primary["quality_change_pp_95ci"][0] >= -1
            and primary["cost_reduction_pct_95ci"][0] > 0
        )
        result["pairs"][pair] = {"rows": rows, "comparisons": comparisons, "success_gate_passed": passed}
    learn.write(directory / "test-predictions.json", records)
    learn.write(directory / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    s = study.load_study()
    if args.freeze:
        freeze(s, args.development, args.run)
        print("Frozen policies and success gate; no test metrics read.")
        return
    plan = json.loads((args.run / "frozen.json").read_text())
    if args.collect:
        destination = args.run / "test-encoding"
        destination.mkdir(exist_ok=False)
        with SingleSendClient(timeout_s=90, read_timeout_s=90) as client:
            encoder_comparison.collect(
                s,
                destination,
                client,
                instruction=plan["encoding"]["options"]["instruction"],
                indices=learning.split(s)["test"],
                scope="728 held-out requests after frozen.json; no fitting",
            )
    print(json.dumps(analyze(s, args.development, args.run, plan), indent=2))


if __name__ == "__main__":
    main()
