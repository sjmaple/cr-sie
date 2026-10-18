"""Static versus feedback-updated routers, with validation and test kept separate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import prepare_gepa
import score
import study


def balanced(indices: list[int], strata: np.ndarray, count: int) -> list[int]:
    """Round-robin through the already hash-ordered strata; never select by outcomes."""
    groups = [[i for i in indices if strata[i] == name] for name in sorted(set(strata[indices]))]
    ordered = [group[j] for j in range(max(map(len, groups), default=0)) for group in groups if j < len(group)]
    return ordered[:count]


def split(s: study.Study, initial_size: int = 64) -> dict[str, list[int]]:
    if initial_size <= 0:
        raise ValueError("initial bank size must be positive")
    groups = prepare_gepa.partitions(s)
    groups["initial"] = balanced(groups["train"], s.strata, initial_size)
    groups["test"] = np.where(~s.calibration)[0].tolist()
    if not groups["initial"] or len(groups["initial"]) >= len(groups["train"]):
        raise ValueError("initial bank must be smaller than the training pool")
    return groups


def mask(n: int, indices: list[int]) -> np.ndarray:
    result = np.zeros(n, dtype=bool)
    result[indices] = True
    return result


def evaluate(o: study.Outcomes, values: np.ndarray, usd: np.ndarray, fit: list[int], test: list[int]) -> dict:
    """Fit a threshold on fit only; summarize on a separate evaluation partition."""
    if set(fit) & set(test):
        raise ValueError("threshold fitting and evaluation must be disjoint")
    tau = score.choose_tau(o, values, usd, mask(len(values), fit))
    metrics = score.summarize(o, *score.routed(o, values, usd, tau), mask(len(values), test))
    return metrics | {"tau": None if math.isinf(tau) else float(tau), "n": len(test)}


def bank_scores(s: study.Study, pair: str, indices: list[int]) -> np.ndarray:
    bank_mask = mask(len(s.ids), indices)
    y = study.labels(s.pairs[pair].q_cheap, s.pairs[pair].q_strong)
    return score.leave_one_out_scores(s.vectors, y, bank_mask)


def improvement(candidate: dict, baseline: dict) -> bool:
    """Promote only a cheaper validation result meeting the quality target, with no quality regression."""
    return (
        candidate["quality_vs_strong"] >= max(score.TARGET, baseline["quality_vs_strong"]) - 1e-12
        and candidate["cost_per_1k"] < baseline["cost_per_1k"] - 1e-12
    )


def embedding_experiment(s: study.Study, pair: str, groups: dict) -> dict:
    """A growing bank is a challenger until validation accepts it; test never controls promotion."""
    o = s.pairs[pair]
    usd = s.credits * study.SIE_USD_PER_CREDIT
    frozen = bank_scores(s, pair, groups["initial"])
    expanded = bank_scores(s, pair, groups["train"])
    # Training-bank leave-one-out scores fit provisional thresholds for validation.
    validation = {
        "static": evaluate(o, frozen, usd, groups["initial"], groups["validation"]),
        "challenger": evaluate(o, expanded, usd, groups["train"], groups["validation"]),
    }
    promoted = improvement(validation["challenger"], validation["static"])
    heldout = {
        "static": evaluate(o, frozen, usd, groups["threshold"], groups["test"]),
        "challenger": evaluate(o, expanded, usd, groups["threshold"], groups["test"]),
    }
    return {
        "bank_sizes": {"static": len(groups["initial"]), "challenger": len(groups["train"])},
        "validation": validation,
        "promoted": promoted,
        "heldout": heldout,
        "deployed": "challenger" if promoted else "static",
    }


def report_row(name: str, row: dict) -> None:
    print(
        f"{name:<28} {score.money(row['cost_per_1k']):>8} {score.percent(row['saving_vs_strong']):>8} "
        f"{score.percent(row['quality_vs_strong']):>8} {score.percent(row['cheap_share']):>8}"
    )


def embedding_cost_breakdown(s: study.Study, pair: str, groups: dict, result: dict) -> dict:
    """Separate the embedding receipt from the generation bill caused by routing."""
    o = s.pairs[pair]
    held = mask(len(s.ids), groups["test"])
    router_usd = s.credits * study.SIE_USD_PER_CREDIT
    routes, rows = {}, {}
    for name, bank in (("static", groups["initial"]), ("challenger", groups["train"])):
        tau = result["heldout"][name]["tau"]
        values = bank_scores(s, pair, bank)
        quality, total, cheap = score.routed(o, values, router_usd, math.inf if tau is None else tau)
        routes[name] = cheap
        rows[name] = {
            "embedding_per_1k": float(1000 * router_usd[held].mean()),
            "generation_per_1k": float(1000 * (total[held] - router_usd[held]).mean()),
            "total_per_1k": float(1000 * total[held].mean()),
            "strong_requests": int(sum(~cheap[held])),
            "quality_vs_strong": float(quality[held].mean() / o.q_strong[held].mean()),
        }
    return {
        "rows": rows,
        "cheap_to_strong": int(sum(held & routes["static"] & ~routes["challenger"])),
        "strong_to_cheap": int(sum(held & ~routes["static"] & routes["challenger"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=study.EVIDENCE)
    parser.add_argument("--pair", choices=sorted(study.PAIRS), default="openai")
    parser.add_argument("--initial-size", type=int, default=64)
    args = parser.parse_args()
    s = study.load_study(args.evidence)
    groups = split(s, args.initial_size)
    result = embedding_experiment(s, args.pair, groups)
    print(json.dumps(result, indent=2, allow_nan=False))
    costs = embedding_cost_breakdown(s, args.pair, groups, result)
    print("\nHeld-out costs per 1,000 requests (embedding inference + downstream generation):")
    for name, row in costs["rows"].items():
        print(
            f"{name:12} embedding ${row['embedding_per_1k']:.4f} + generation ${row['generation_per_1k']:.4f} "
            f"= total ${row['total_per_1k']:.4f}; {row['strong_requests']} strong-model requests"
        )
    print(f"Changed routes: {costs['cheap_to_strong']} cheap→strong; {costs['strong_to_cheap']} strong→cheap.")
    print("The embedding charge is identical; generation cost changes with the selected model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
