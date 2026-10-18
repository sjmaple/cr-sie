"""Audit the diagnostic GEPA run offline, including paired bootstrap uncertainty."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import improve
import intent_evidence
import learning
import learning_evidence
import replay_learning
import score
import study
from cloud import digest


def bootstrap(s, indices, seed_quality, seed_cost, new_quality, new_cost):
    rng = np.random.default_rng(51)
    strata = [[i for i in indices if s.strata[i] == name] for name in sorted(set(s.strata[indices]))]
    samples = np.concatenate([rng.choice(group, (4000, len(group))) for group in strata], axis=1)
    saving = 100 * (1 - new_cost[samples].mean(axis=1) / seed_cost[samples].mean(axis=1))
    quality = (
        100
        * (new_quality[samples].mean(axis=1) - seed_quality[samples].mean(axis=1))
        / s.pairs["openai"].q_strong[samples].mean(axis=1)
    )
    return {
        "cost_reduction_pct_95ci": np.quantile(saving, [0.025, 0.975]).tolist(),
        "retained_quality_change_pp_95ci": np.quantile(quality, [0.025, 0.975]).tolist(),
        "resamples": 4000,
        "seed": 51,
        "method": "paired stratified percentile bootstrap",
    }


def replay(directory, s, groups):
    plan = json.loads((directory / "plan.json").read_text())
    selection = json.loads((directory / "selection.json").read_text())
    expected = json.loads((directory / "results.json").read_text())
    replay_learning.verify({name: [s.ids[i] for i in rows] for name, rows in groups.items()}, plan["splits"], "splits")
    records = study.read_jsonl(directory / "scores.jsonl")
    by_key = {(r["candidate"], r["id"]): r for r in records}
    if len(by_key) != len(records):
        raise ValueError("duplicate score records")
    candidates = selection["candidates"]
    o = s.pairs["openai"]

    def arrays(candidate, indices):
        values, usd = np.zeros(len(s.ids)), np.zeros(len(s.ids))
        for i in indices:
            row = by_key[digest(candidate), s.ids[i]]
            value, cost = row["score"], row["router_usd"]
            if not math.isfinite(value) or not 0 <= value <= 1 or not math.isfinite(cost) or cost < 0:
                raise ValueError("invalid score or cost")
            values[i], usd[i] = value, cost
        return values, usd

    validation, thresholds = [], []
    for candidate in candidates:
        values, usd = arrays(candidate, groups["fit"] + groups["validation"])
        tau = score.choose_tau(o, values, usd, learning.mask(len(s.ids), groups["fit"]))
        thresholds.append(None if math.isinf(tau) else tau)
        validation.append(
            score.summarize(o, *score.routed(o, values, usd, tau), learning.mask(len(s.ids), groups["validation"]))
        )
    replay_learning.verify(validation, selection["validation"], "validation")
    replay_learning.verify(thresholds, selection["thresholds"], "thresholds")
    eligible = [i for i in range(1, len(candidates)) if improve.pareto_better(validation[i], validation[0])]
    chosen = min(eligible, key=lambda i: validation[i]["cost_per_1k"]) if eligible else 0
    replay_learning.verify(chosen, selection["chosen"], "chosen")
    routed = {}
    rows = {}
    for name, i in (("seed", 0), ("selected", chosen)):
        values, usd = arrays(candidates[i], groups["audit"])
        routed[name] = score.routed(o, values, usd, math.inf if thresholds[i] is None else thresholds[i])
        rows[name] = score.summarize(o, *routed[name], learning.mask(len(s.ids), groups["audit"]))
    replay_learning.verify(rows, expected["audit"], "audit")
    calls = study.read_jsonl(directory / "calls.jsonl")
    replay_learning.verify(sum(c.get("credits", 0) for c in calls), expected["cloud"]["observed_credits"], "credits")
    uncertainty = bootstrap(s, groups["audit"], *routed["seed"][:2], *routed["selected"][:2])
    print("Audit: fixed prompt AND threshold; no recalibration on audit data.")
    print(f"{'Router':<28} {'$/1,000':>8} {'Saving':>8} {'Quality':>8} {'Cheap':>8}")
    for name, row in rows.items():
        learning.report_row(name, row)
    result = {
        "n": len(groups["audit"]),
        "cost_reduction_pct": 100 * (1 - rows["selected"]["cost_per_1k"] / rows["seed"]["cost_per_1k"]),
        "retained_quality_change_pp": 100 * (rows["selected"]["quality_vs_strong"] - rows["seed"]["quality_vs_strong"]),
        "changed_routes": int(sum(routed["seed"][2][groups["audit"]] != routed["selected"][2][groups["audit"]])),
        "uncertainty": uncertainty,
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=study.EVIDENCE / "intents/cost-diagnostic")
    args = parser.parse_args()
    learning_evidence.verify_evidence(study.EVIDENCE / "learning")
    if args.run.resolve() == (study.EVIDENCE / "intents/cost-diagnostic").resolve():
        intent_evidence.verify_evidence(args.run.parent)
    s = study.load_study(study.EVIDENCE)
    replay(args.run, s, improve.partitions(s, study.EVIDENCE / "learning"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
