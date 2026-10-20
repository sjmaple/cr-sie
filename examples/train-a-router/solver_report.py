"""Paired answer quality and measured solver costs for a fixed routing policy.

These functions do no inference. Each row must contain both solver outcomes for
the same problem, collected with frozen solver settings. A positive quality gap
means lost accuracy relative to always using the strong solver; rescues can make
the gap negative. Calibration constraints are empirical, not risk guarantees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 89
BASELINES = ("always_strong", "always_cheap")


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _aligned(values, ids, name):
    if isinstance(values, Mapping):
        if set(values) != set(ids):
            raise ValueError(f"{name} IDs do not match the rows")
        return [values[key] for key in ids]
    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, np.ndarray)):
        raise TypeError(f"{name} must be an aligned sequence or an ID mapping")
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(values) != len(ids):
        raise ValueError(f"{name} length does not match the rows")
    return list(values)


def _inputs(rows, router_credits, credit_usd):
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("rows must be a nonempty sequence")
    required = {"id", "cheap_correct", "strong_correct", "cheap_credits", "strong_credits"}
    ids, cheap, strong, cheap_cost, strong_cost = [], [], [], [], []
    for row in rows:
        if not isinstance(row, Mapping) or not required <= row.keys():
            raise ValueError("Each row requires an ID, both correctness labels, and both measured credit costs")
        if not isinstance(row["id"], str) or not row["id"].strip():
            raise ValueError("Each row requires a nonempty string ID")
        ids.append(row["id"])
        for key in ("cheap_correct", "strong_correct"):
            if not isinstance(row[key], (bool, np.bool_)):
                raise TypeError(f"{key} must be boolean")
        cheap.append(row["cheap_correct"])
        strong.append(row["strong_correct"])
        cheap_cost.append(_number(row["cheap_credits"], "cheap_credits"))
        strong_cost.append(_number(row["strong_credits"], "strong_credits"))
    if len(set(ids)) != len(ids):
        raise ValueError("Row IDs must be unique")
    rate = _number(credit_usd, "credit_usd")
    overhead = None
    if router_credits is not None:
        if isinstance(router_credits, Real):
            overhead = np.full(len(ids), _number(router_credits, "router_credits"))
        else:
            overhead = np.array(
                [_number(value, "router_credits") for value in _aligned(router_credits, ids, "router_credits")]
            )
    return (
        ids,
        np.array(cheap, dtype=bool),
        np.array(strong, dtype=bool),
        np.array(cheap_cost),
        np.array(strong_cost),
        overhead,
        rate,
    )


def _routes(routes, ids):
    if isinstance(routes, str):
        if routes not in BASELINES:
            raise ValueError("Unknown baseline; use always_strong or always_cheap")
        return np.full(len(ids), routes == "always_strong"), routes
    values = _aligned(routes, ids, "routes")
    if any(not isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("routes must be boolean: true selects the strong solver")
    return np.array(values, dtype=bool), "router"


def _means(correct, strong, cost, strong_cost, cheap_share, overhead, rate):
    """Accept scalars or vectors of bootstrap means, keeping pairs together."""
    unit = 1000 * rate
    gross = (strong_cost - cost) * unit
    with np.errstate(divide="ignore", invalid="ignore"):
        gross_pct = np.where(strong_cost > 0, 100 * (strong_cost - cost) / strong_cost, np.nan)
    result = {
        "answer_accuracy": correct,
        "always_strong_answer_accuracy": strong,
        "gap_vs_strong_pp": 100 * (strong - correct),
        "total_solver_cost_usd_per1000": cost * unit,
        "always_strong_solver_cost_usd_per1000": strong_cost * unit,
        "gross_solver_savings_usd_per1000": gross,
        "gross_solver_savings_pct": gross_pct,
        "cheap_share": cheap_share,
        "break_even_router_usd_per1000": gross,
    }
    if overhead is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            net_pct = np.where(strong_cost > 0, 100 * (strong_cost - cost - overhead) / strong_cost, np.nan)
        result.update(
            router_cost_usd_per1000=overhead * unit,
            net_total_cost_usd_per1000=(cost + overhead) * unit,
            net_savings_usd_per1000=(strong_cost - cost - overhead) * unit,
            net_savings_pct=net_pct,
        )
    return result


def _summary(inputs, decisions, mode):
    ids, cheap, strong, cheap_cost, strong_cost, overhead, rate = inputs
    if overhead is not None and mode in BASELINES:
        overhead = np.zeros(len(ids))
    correct = np.where(decisions, strong, cheap)
    losses = strong.astype(int) - correct.astype(int)
    cost = np.where(decisions, strong_cost, cheap_cost)
    means = _means(
        correct.mean(),
        strong.mean(),
        cost.mean(),
        strong_cost.mean(),
        (~decisions).mean(),
        None if overhead is None else overhead.mean(),
        rate,
    )
    means["gap_vs_strong_pp"] = 100 * int(losses.sum()) / len(ids)
    result = {key: float(value) if np.isfinite(value) else None for key, value in means.items()}
    result.update(
        n=len(ids),
        routing_mode=mode,
        regressions=int((strong & ~correct).sum()),
        rescues=int((~strong & correct).sum()),
        router_overhead_known=overhead is not None,
        router_overhead_applied=overhead is not None and mode not in BASELINES,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {key: [] for key in means}
    for start in range(0, BOOTSTRAP_SAMPLES, 128):
        indices = rng.integers(0, len(ids), (min(128, BOOTSTRAP_SAMPLES - start), len(ids)))
        values = _means(
            correct[indices].mean(axis=1),
            strong[indices].mean(axis=1),
            cost[indices].mean(axis=1),
            strong_cost[indices].mean(axis=1),
            (~decisions)[indices].mean(axis=1),
            None if overhead is None else overhead[indices].mean(axis=1),
            rate,
        )
        values["gap_vs_strong_pp"] = 100 * losses[indices].mean(axis=1)
        for key, value in values.items():
            samples[key].append(value)
    result["paired_bootstrap_95ci"] = {
        key: np.quantile(values, [0.025, 0.975]).tolist() if np.isfinite(values).all() else None
        for key, chunks in samples.items()
        for values in [np.concatenate(chunks)]
    }
    result["bootstrap"] = {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "method": "paired problem-level percentile bootstrap, conditional on the fixed policy and recorded outcomes",
        "limitations": (
            "Descriptive intervals assume independent problems; they do not account for shared competition papers, "
            "policy selection, solver randomness, or future price changes. Zero-event intervals are not risk bounds. "
            "A percentage interval is null if any resampled baseline has zero cost."
        ),
    }
    return result


def summarize(rows, routes, router_credits=None, credit_usd=1e-5):
    """Return paired quality/cost metrics and fixed-seed descriptive 95% intervals.

    ``routes`` is an aligned boolean sequence (true means strong), an exact-ID
    mapping, or the explicit baseline ``always_strong`` / ``always_cheap``.
    Only explicit baselines bypass routing overhead; uniform learned decisions
    still incur it. Credit costs include all solver/finalizer calls. Router
    credits may be a nonnegative scalar or one value per row. Unknown overhead
    omits all net-cost fields. Accuracy and cheap share are fractions; the gap
    is percentage points; savings percentages use the always-strong cost.
    """
    inputs = _inputs(rows, router_credits, credit_usd)
    decisions, mode = _routes(routes, inputs[0])
    return _summary(inputs, decisions, mode)


def summarize_quality(rows, routes):
    """Report paired answer quality when solver prices are unknown.

    Correctness and IDs are required; credit fields are neither required nor
    used. This explicit mode reports no costs, savings, or break-even price.
    ``summarize`` and ``select_cutoff`` retain strict measured-cost validation.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("rows must be a nonempty sequence")
    ids, cheap, strong = [], [], []
    for row in rows:
        if not isinstance(row, Mapping) or not {"id", "cheap_correct", "strong_correct"} <= row.keys():
            raise ValueError("Each row requires an ID and both correctness labels")
        if not isinstance(row["id"], str) or not row["id"].strip():
            raise ValueError("Each row requires a nonempty string ID")
        for key in ("cheap_correct", "strong_correct"):
            if not isinstance(row[key], (bool, np.bool_)):
                raise TypeError(f"{key} must be boolean")
        ids.append(row["id"])
        cheap.append(row["cheap_correct"])
        strong.append(row["strong_correct"])
    if len(set(ids)) != len(ids):
        raise ValueError("Row IDs must be unique")
    decisions, mode = _routes(routes, ids)
    strong = np.asarray(strong, dtype=bool)
    correct = np.where(decisions, strong, np.asarray(cheap, dtype=bool))
    losses = strong.astype(int) - correct.astype(int)
    observations = {
        "answer_accuracy": correct,
        "always_strong_answer_accuracy": strong,
        "gap_vs_strong_pp": 100 * losses,
        "cheap_share": ~decisions,
    }
    result = {key: float(values.mean()) for key, values in observations.items()}
    result.update(
        n=len(ids),
        routing_mode=mode,
        regressions=int((strong & ~correct).sum()),
        rescues=int((~strong & correct).sum()),
        cost_status="unknown",
        cost_note="Quality-only analysis: solver and infrastructure costs are unpriced; no savings estimate is made",
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {key: [] for key in observations}
    for start in range(0, BOOTSTRAP_SAMPLES, 128):
        indices = rng.integers(0, len(ids), (min(128, BOOTSTRAP_SAMPLES - start), len(ids)))
        for key, values in observations.items():
            samples[key].append(values[indices].mean(axis=1))
    result["paired_bootstrap_95ci"] = {
        key: np.quantile(np.concatenate(chunks), [0.025, 0.975]).tolist() for key, chunks in samples.items()
    }
    result["bootstrap"] = {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "method": "paired problem-level percentile bootstrap, conditional on the fixed policy and recorded outcomes",
        "limitations": (
            "Descriptive intervals assume independent problems; they do not account for shared competition papers, "
            "policy selection or solver randomness. Zero-event intervals are not risk bounds."
        ),
    }
    return result


def select_cutoff(calibration_rows, scores, max_quality_loss_pp, router_credits=None, credit_usd=1e-5):
    """Select the cheapest empirically eligible policy using calibration only.

    A threshold chooses strong when ``score >= cutoff``; tied scores stay
    together. Always-strong and always-cheap are explicit router-free options.
    A policy must strictly beat always-strong's cost to replace that fallback.
    Remaining ties prefer less quality loss, then bypassing the router, then a
    lower cutoff. This target is the observed *net* accuracy gap: rescues offset
    regressions. It is neither a bound on harmful switches nor a confidence
    bound on future quality. Freeze this selection before reading test answers.
    """
    target = _number(max_quality_loss_pp, "max_quality_loss_pp")
    if target > 100:
        raise ValueError("max_quality_loss_pp cannot exceed 100")
    inputs = _inputs(calibration_rows, router_credits, credit_usd)
    ids, cheap, strong, cheap_cost, strong_cost, overhead, _ = inputs
    values = _aligned(scores, ids, "scores")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values):
        raise ValueError("scores must be finite numbers")
    values = np.array(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite numbers")
    baseline_cost = float(strong_cost.mean())
    best = {
        "mode": "always_strong",
        "cutoff": None,
        "cost": baseline_cost,
        "gap": 0.0,
        "decisions": np.ones(len(ids), dtype=bool),
    }
    candidates = [("always_cheap", None, np.zeros(len(ids), dtype=bool))]
    candidates.extend(("threshold", float(value), values >= value) for value in np.unique(values))
    eligible = 1  # The always-strong bypass has zero loss and zero router overhead.
    best_key = None
    for mode, cutoff, decisions in candidates:
        correct = np.where(decisions, strong, cheap)
        gap = 100 * int((strong.astype(int) - correct.astype(int)).sum()) / len(ids)
        if gap > target:
            continue
        eligible += 1
        cost = float(np.where(decisions, strong_cost, cheap_cost).mean())
        if mode == "threshold" and overhead is not None:
            cost += float(overhead.mean())
        if cost >= baseline_cost:
            continue
        key = (cost, gap, mode == "threshold", float("-inf") if cutoff is None else cutoff)
        if best_key is None or key < best_key:
            best_key = key
            best = {"mode": mode, "cutoff": cutoff, "cost": cost, "gap": gap, "decisions": decisions}
    mode = best["mode"]
    return {
        "mode": mode,
        "cutoff": best["cutoff"],
        "score_rule": "score >= cutoff selects strong" if mode == "threshold" else None,
        "max_quality_loss_pp": target,
        "selection_cost_basis": "solver plus router credits" if overhead is not None else "solver credits only",
        "selection_rule": "minimum empirical cost below always-strong; then loss, router bypass, lower cutoff",
        "candidates": len(candidates) + 1,
        "empirically_eligible_candidates": eligible,
        "calibration_ids": ids,
        "calibration_routes": best["decisions"].tolist(),
        "calibration": _summary(inputs, best["decisions"], "router" if mode == "threshold" else mode),
        "interpretation": "Calibration-only empirical constraint, not a selection-adjusted or prospective quality guarantee",
    }
