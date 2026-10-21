"""Evaluate frozen routing decisions with paired family uncertainty; never fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_lineage(bundle, grades, frozen_routing_sha256):
    require(
        grades["format"] == "routerbench-paired-grades-v1"
        and grades["split"] == "final",
        "Need final paired grades",
    )
    require(
        bundle["source_plan_sha256"] == grades["source_plan_sha256"]
        and bundle["pair_spec_sha256"] == grades["pair_spec_sha256"]
        and bundle["cost_scope"] == grades["cost_scope"],
        "Frozen routes and grades belong to different study/cost basis",
    )
    require(
        grades["parent_acquisition_protocol_sha256"]
        == bundle["acquisition_protocol_sha256"]
        and grades["parent_grader_sha256"] == bundle["grader_sha256"]
        and grades["frozen_final_routing_sha256"] == frozen_routing_sha256,
        "Final-only acquisition/grading does not preserve its frozen parent lineage",
    )
    require(
        all(
            isinstance(grades[key], str) and re.fullmatch(r"[0-9a-f]{64}", grades[key])
            for key in ("acquisition_protocol_sha256", "grader_sha256")
        ),
        "Missing final operator provenance",
    )
    metadata = [
        {
            "id": row["id"],
            "source": row["dataset"],
            "family": row["family"],
            "task": row["task"],
        }
        for row in grades["rows"]
    ]
    require(
        metadata == bundle["source_rows"],
        "Final metadata/family grouping changed since freezing",
    )


def interval(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "lower": float(np.quantile(finite, 0.025)) if len(finite) else None,
        "upper": float(np.quantile(finite, 0.975)) if len(finite) else None,
        "defined_draws": len(finite),
        "total_draws": len(values),
    }


def family_bootstrap(matrix, rows, *, draws=2000, seed=89):
    """Resample whole families within sources, retaining each source's weight."""
    matrix = np.asarray(matrix, dtype=float)
    require(
        matrix.ndim == 2
        and len(matrix) == len(rows)
        and len(rows) > 0
        and np.isfinite(matrix).all(),
        "Invalid paired metric matrix",
    )
    require(
        type(draws) is int and draws >= 100 and type(seed) is int,
        "Invalid bootstrap plan",
    )
    rng = np.random.default_rng(seed)
    result = np.zeros((draws, matrix.shape[1]))
    family_source, sources = {}, {}
    for index, row in enumerate(rows):
        source, family = row["dataset"], row["family"]
        require(
            isinstance(source, str) and source and isinstance(family, str) and family,
            "Missing source/family",
        )
        require(
            family_source.setdefault(family, source) == source,
            "Family spans sources; requires joint cluster resampling",
        )
        sources.setdefault(source, {}).setdefault(family, []).append(index)
    info = {}
    for source, families in sorted(sources.items()):
        indices = [families[name] for name in sorted(families)]
        sizes = np.array([len(values) for values in indices], dtype=float)
        sums = np.array([matrix[values].sum(axis=0) for values in indices])
        count = len(indices)
        weights = rng.multinomial(count, np.full(count, 1 / count), size=draws)
        denominators = weights @ sizes
        source_weight = sizes.sum() / len(rows)
        result += source_weight * ((weights @ sums) / denominators[:, None])
        info[source] = {
            "questions": int(sizes.sum()),
            "families": count,
            "question_weight": float(source_weight),
            "largest_family": int(sizes.max()),
        }
    return result, info


def safe_ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )


def metrics(means, offset, priced):
    cheap_q, strong_q, cheap_usd, strong_usd = (means[..., i] for i in range(4))
    quality, answer_usd, router_usd, strong_fraction = (
        means[..., offset + i] for i in range(4)
    )
    total_usd = answer_usd + router_usd
    matched_spend = total_usd if priced else answer_usd
    p = safe_ratio(matched_spend - cheap_usd, strong_usd - cheap_usd)
    feasible = (p >= -1e-12) & (p <= 1 + 1e-12) & np.isfinite(p)
    p = np.where(feasible, np.clip(p, 0, 1), np.nan)
    expected_random_quality = cheap_q + p * (strong_q - cheap_q)
    values = {
        "answer_accuracy_percent": 100 * quality,
        "quality_gap_from_always_strong_pp": 100 * (strong_q - quality),
        "answer_model_usd_per_10000": 10000 * answer_usd,
        "answer_model_savings_percent": 100 * (1 - safe_ratio(answer_usd, strong_usd)),
        "strong_fraction": strong_fraction,
        "router_break_even_usd_per_10000": 10000 * (strong_usd - answer_usd),
        "matched_spend_random_strong_fraction": p,
        "matched_spend_expected_random_accuracy_percent": 100 * expected_random_quality,
        "accuracy_advantage_over_matched_spend_random_pp": 100
        * (quality - expected_random_quality),
    }
    if priced:
        values.update(
            router_usd_per_10000=10000 * router_usd,
            total_usd_per_10000=10000 * total_usd,
            net_savings_percent=100 * (1 - safe_ratio(total_usd, strong_usd)),
        )
    return values


def scalar(value):
    result = float(value)
    return result if np.isfinite(result) else None


def evaluate(rows, bundle, *, draws=2000, seed=89):
    require(
        bundle["format"] == "routerbench-frozen-final-routing-v1",
        "Unknown frozen policy format",
    )
    require(
        bundle["frozen_before_final_model_outcomes"] is True,
        "Policy freezing boundary is missing",
    )
    ids = [row["id"] for row in rows]
    require(
        ids == bundle["ids"] and len(ids) == len(set(ids)) > 0,
        "Final grade/policy population or order changed",
    )
    require(
        all(row["role"] == "final" for row in rows), "Reporter accepts final rows only"
    )
    base = np.asarray(
        [
            [
                row["cheap_correct"],
                row["strong_correct"],
                row["cheap_cost_usd"],
                row["strong_cost_usd"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    require(
        np.isfinite(base).all()
        and np.isin(base[:, :2], (0, 1)).all()
        and (base[:, 2:] >= 0).all()
        and (base[:, 3] > 0).all(),
        "Invalid answer outcomes/costs",
    )
    columns, descriptors = [base], {}
    policies = bundle["policies"]
    require(isinstance(policies, dict) and policies, "No frozen policies")
    for index, (name, policy) in enumerate(policies.items()):
        require(
            isinstance(name, str)
            and name
            and policy["kind"] in {"always", "learned", "random"},
            "Invalid policy kind/name",
        )
        p = np.asarray(policy["strong_probabilities"], dtype=float)
        require(
            p.shape == (len(rows),)
            and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all(),
            "Invalid routing probabilities",
        )
        if policy["kind"] in {"always", "learned"}:
            require(
                np.isin(p, (0, 1)).all(),
                "Deterministic policy contains fractional decisions",
            )
        if policy["kind"] in {"always", "random"}:
            require(
                (p == p[0]).all(),
                "Always/random policy must have a single frozen probability",
            )
        priced = policy["router_usd"] is not None
        fee = (
            np.asarray(policy["router_usd"], dtype=float)
            if priced
            else np.zeros(len(rows))
        )
        require(
            fee.shape == (len(rows),) and np.isfinite(fee).all() and (fee >= 0).all(),
            "Invalid router fees",
        )
        if policy["kind"] in {"always", "random"}:
            require(
                priced and (fee == 0).all(),
                "Bypass/random control must explicitly avoid learned-router inference",
            )
        target = policy.get("target_gap_pp")
        require(
            target is None
            or type(target) in (int, float)
            and np.isfinite(target)
            and target >= 0,
            "Invalid target",
        )
        quality = (1 - p) * base[:, 0] + p * base[:, 1]
        answer_usd = (1 - p) * base[:, 2] + p * base[:, 3]
        values = np.column_stack((quality, answer_usd, fee, p))
        columns.append(values)
        descriptors[name] = {
            "offset": 4 + 4 * index,
            "priced": priced,
            "method": policy["method"],
            "kind": policy["kind"],
            "target_gap_pp": target,
            "metadata_assisted": bool(policy.get("metadata_assisted", False)),
        }
    matrix = np.column_stack(columns)
    bootstrap, source_info = family_bootstrap(matrix, rows, draws=draws, seed=seed)
    point, estimates, samples = matrix.mean(axis=0), {}, {}
    for name, descriptor in descriptors.items():
        current = metrics(point, descriptor["offset"], descriptor["priced"])
        sampled = metrics(bootstrap, descriptor["offset"], descriptor["priced"])
        samples[name] = sampled
        estimates[name] = {
            key: value for key, value in descriptor.items() if key != "offset"
        }
        estimates[name].update(
            metrics={key: scalar(value) for key, value in current.items()},
            confidence95={key: interval(value) for key, value in sampled.items()},
            cost_scope="answer models plus supplied router fees"
            if descriptor["priced"]
            else "answer-model gross savings; native hosting remains unpriced",
            random_comparison_scope="Descriptive expected random routing at matched test spending; its probability is not a deployable fitted policy. Frozen calibration-selected random policies are listed separately.",
            random_matching_cost="total deployment cost"
            if descriptor["priced"]
            else "answer-model cost only",
            matched_spend_random_available=bool(
                np.isfinite(current["matched_spend_random_strong_fraction"])
            ),
            eligible_for_primary_comparison=not descriptor["metadata_assisted"],
        )
        if not descriptor["priced"]:
            estimates[name]["metrics"].update(
                router_usd_per_10000=None,
                total_usd_per_10000=None,
                net_savings_percent=None,
            )
        target = descriptor["target_gap_pp"]
        if target is not None:
            gap = current["quality_gap_from_always_strong_pp"]
            upper = estimates[name]["confidence95"][
                "quality_gap_from_always_strong_pp"
            ]["upper"]
            estimates[name]["heldout_point_meets_calibration_target"] = bool(
                gap <= target + 1e-12
            )
            estimates[name]["heldout_upper95_meets_calibration_target"] = (
                upper is not None and upper <= target + 1e-12
            )
    per_source = {}
    for source in sorted(source_info):
        indices = [index for index, row in enumerate(rows) if row["dataset"] == source]
        means = matrix[indices].mean(axis=0)
        per_source[source] = {
            "questions": len(indices),
            "families": source_info[source]["families"],
            "always_cheap_accuracy_percent": float(100 * means[0]),
            "always_strong_accuracy_percent": float(100 * means[1]),
            "always_cheap_usd_per_10000": float(10000 * means[2]),
            "always_strong_usd_per_10000": float(10000 * means[3]),
            "policies": {
                name: {
                    key: scalar(value)
                    for key, value in metrics(
                        means, desc["offset"], desc["priced"]
                    ).items()
                }
                for name, desc in descriptors.items()
            },
        }
    comparisons = {}
    for name, desc in descriptors.items():
        if desc["kind"] != "learned" or desc["metadata_assisted"]:
            continue
        for other, reference in descriptors.items():
            if (
                name == other
                or reference["metadata_assisted"]
                or desc["target_gap_pp"] != reference["target_gap_pp"]
            ):
                continue
            if reference["kind"] != "random" and reference["method"] not in {
                "topic",
                "base",
                "lora",
            }:
                continue
            compare = {}
            for metric in (
                "answer_accuracy_percent",
                "answer_model_savings_percent",
                "net_savings_percent",
            ):
                if metric not in samples[name] or metric not in samples[other]:
                    continue
                compare[metric + "_difference_pp"] = {
                    "point": estimates[name]["metrics"][metric]
                    - estimates[other]["metrics"][metric],
                    "confidence95": interval(
                        samples[name][metric] - samples[other][metric]
                    ),
                }
            comparisons[name + " versus " + other] = compare
    return {
        "format": "routerbench-paired-final-report-v1",
        "questions": len(rows),
        "sources": source_info,
        "fitting_or_policy_selection": False,
        "primary_population": "Every question in the fixed final split has equal weight; this differs from the source-balanced900 development screen.",
        "uncertainty": {
            "draws": draws,
            "seed": seed,
            "interval": "Pointwise95% percentile, no multiple-comparison adjustment",
            "unit": "Whole family, paired across models/policies, resampled within each source; observed final source weights fixed",
            "limitations": "Small family counts limit generalization; this does not establish freedom from benchmark pretraining contamination.",
        },
        "always_cheap": {
            "answer_accuracy_percent": float(100 * point[0]),
            "usd_per_10000": float(10000 * point[2]),
        },
        "always_strong": {
            "answer_accuracy_percent": float(100 * point[1]),
            "usd_per_10000": float(10000 * point[3]),
        },
        "policies": estimates,
        "paired_comparisons": comparisons,
        "by_source": per_source,
        "interpretation": "Percentage points describe absolute answer-accuracy differences. Savings percentages use always-strong answer cost as the denominator. Exact-match benchmark quality is not a guarantee for arbitrary production requests.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-routing", type=Path, required=True)
    parser.add_argument("--final-grades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous analysis")
    bundle = json.loads(args.frozen_routing.read_text())
    require(
        bundle["frozen_before_final_model_outcomes"] is True, "Missing frozen boundary"
    )
    grades = json.loads(args.final_grades.read_text())
    check_lineage(bundle, grades, sha(args.frozen_routing))
    report = evaluate(grades["rows"], bundle)
    report["provenance"] = {
        "reporter_sha256": sha(__file__),
        "frozen_routing_sha256": sha(args.frozen_routing),
        "final_grades_sha256": sha(args.final_grades),
        "cost_scope": grades["cost_scope"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"report_sha256": sha(args.output), "questions": report["questions"]}
        )
    )


if __name__ == "__main__":
    main()
