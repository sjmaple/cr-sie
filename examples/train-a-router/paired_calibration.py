"""Freeze utility policies and expected-random controls from calibration only.

Final inputs may be scored, but this module has no final-grade argument. Every
primary calibration row has equal weight; the development-screen mixture is
never imported into selection.
"""

from __future__ import annotations

import argparse
import re
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import paired_training as core
import utility_router

TARGETS = (0, 1, 2, 5)
PROTOCOL = {
    "targets_gap_pp": list(TARGETS),
    "primary_gap_pp": 1,
    "secondary_gap_pp": 2,
    "weighting": "Equal weight per question in the fixed full calibration/final split; no development-mixture weights",
    "random": "Calibrate a single Bernoulli strong-call probability per target; replay expected per-row quality/cost on final",
    "random_selection": "Minimum expected calibration answer USD among feasible fractions; ties prefer smaller signed gap then more strong calls",
    "fees": "Projected singleton embedding fees; native and text-topic hosting unpriced; always-arm bypasses pay no router fee",
    "boundary": "All final routes fixed from input-only predictions before final outcomes; retain every0/1/2/5 policy",
}


def checked_frozen_bundle(bundle, source_rows):
    """Source-only dispatch gate; no answers or reference labels are accepted."""
    expected = [row for row in source_rows if row["role"] == "final"]
    ids = [row["id"] for row in expected]
    metadata = [{key: row[key] for key in ("id", "source", "family", "task")} for row in expected]
    core.require(
        bundle["format"] == "routerbench-frozen-final-routing-v1"
        and bundle["frozen_before_final_model_outcomes"] is True
        and bundle["final_outcomes_accessed"] is False
        and bundle["ids"] == ids and len(ids) == len(set(ids)) > 0
        and bundle["source_rows"] == metadata and bundle["protocol"] == PROTOCOL,
        "Frozen final source population, metadata or timing boundary differs",
    )
    for key in ("plan_sha256", "source_plan_sha256", "pair_spec_sha256", "acquisition_protocol_sha256",
                "grader_sha256", "calibration_grades_sha256", "calibration_sha256", "training_grades_sha256",
                "fee_projection_sha256", "calibration_source_sha256", "utility_source_sha256"):
        core.require(isinstance(bundle.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", bundle[key]),
                     "Missing frozen provenance hash: " + key)
    core.require(set(bundle["scores_sha256"]) == {"embedding", "native", "topic"}
                 and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in bundle["scores_sha256"].values()),
                 "Missing frozen score bindings")
    core.require(bundle["cost_scope"] and isinstance(bundle["cost_scope"], str), "Missing frozen cost scope")
    required = {"always_cheap", "always_strong"} | {
        f"{method}@{target}pp" for method in ("embedding", "base", "lora", "topic", "topic_metadata_diagnostic", "random")
        for target in TARGETS}
    core.require(set(bundle["policies"]) == required, "Missing or undeclared frozen methods/targets")
    for name, policy in bundle["policies"].items():
        kind = policy["kind"]
        p = np.asarray(policy["strong_probabilities"], dtype=float)
        core.require(kind in ("always", "learned", "random") and p.shape == (len(ids),)
                     and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid frozen route mask")
        if kind != "random":
            core.require(np.isin(p, (0, 1)).all(), "Nonbinary deterministic route")
        if kind in ("always", "random"):
            core.require((p == p[0]).all(), "Bypass/random probability must be constant")
        if policy["router_usd"] is not None:
            fee = np.asarray(policy["router_usd"], dtype=float)
            core.require(fee.shape == p.shape and np.isfinite(fee).all() and (fee >= 0).all(), "Invalid frozen router fee")
        else:
            fee = None
        if kind in ("always", "random"):
            core.require(fee is not None and (fee == 0).all(), "Bypass/random policy cannot charge learned-router overhead")
        if name.startswith("always_"):
            core.require(kind == "always" and (p == int(name == "always_strong")).all()
                         and policy["target_gap_pp"] is None, "Always-arm baseline changed")
        else:
            core.require(policy["target_gap_pp"] == int(name.split("@", 1)[1][:-2])
                         and policy["method"] == name.split("@", 1)[0], "Frozen method/target identity changed")
            core.require((kind == "random") == name.startswith("random@"), "Random control identity changed")
        if name.startswith("topic_metadata_diagnostic@"):
            core.require(policy["metadata_assisted"] is True, "Metadata diagnostic must stay explicit")
    return {"n_final": len(ids), "policies": len(required), "ordered_ids_sha256": core.digest(ids),
            "source_metadata_sha256": core.digest(metadata), "targets_gap_pp": list(TARGETS)}


def checked_arrays(probabilities, expected_usd, count):
    p, c = np.asarray(probabilities, dtype=float), np.asarray(expected_usd, dtype=float)
    core.require(
        p.shape == c.shape == (count, 2)
        and np.isfinite(p).all()
        and np.isfinite(c).all()
        and ((0 <= p) & (p <= 1)).all()
        and (c >= 0).all(),
        "Finite, aligned cheap/strong probabilities and USD required",
    )
    return p, c


def random_policy(correct, observed_usd, target_gap_pp):
    """A calibration-chosen expected mixture, with exact count/cost comparisons."""
    y, costs = np.asarray(correct), np.asarray(observed_usd, dtype=float)
    core.require(
        y.ndim == 2 and y.shape[1] == 2 and len(y) > 0 and np.isin(y, (0, 1)).all()
        and costs.shape == y.shape and np.isfinite(costs).all() and (costs >= 0).all(),
        "Invalid paired random-control calibration rows",
    )
    target = Fraction(str(target_gap_pp))
    core.require(0 <= target <= 100, "Invalid random target")
    n = len(y)
    gap_count = sum(int(row[1]) - int(row[0]) for row in y)
    totals = [sum((Fraction.from_float(float(value)) for value in costs[:, arm]), Fraction()) for arm in (0, 1)]
    candidates = {Fraction(0), Fraction(1)}
    if gap_count:
        crossing = 1 - target * n / (100 * gap_count)
        if 0 <= crossing <= 1:
            candidates.add(crossing)
    eligible = [p for p in candidates if (1 - p) * gap_count * 100 <= target * n]
    probability = min(eligible, key=lambda p: ((1-p)*totals[0]+p*totals[1], (1-p)*gap_count, -p))
    return {
        "format": "routerbench-expected-random-policy-v1",
        "target_gap_pp": float(target),
        "strong_probability": float(probability),
        "strong_probability_exact": {"numerator": str(probability.numerator), "denominator": str(probability.denominator)},
        "calibration": {
            "n": n,
            "gap_vs_strong_pp": float(100 * (1-probability) * gap_count / n),
            "answer_cost_usd_per10000": float(10000 * ((1-probability)*totals[0]+probability*totals[1]) / n),
        },
        "interpretation": "Expected random routing; probability chosen on calibration only, no realized random model calls",
    }


def freeze_arrays(methods, correct, observed_usd, final_ids, *, final_rows=None):
    """Pure selection/replay: methods supply only paired input-derived predictions."""
    y, costs = np.asarray(correct), np.asarray(observed_usd, dtype=float)
    n, nf = len(y), len(final_ids)
    core.require(n > 0 and nf > 0 and len(set(final_ids)) == nf, "Empty/duplicate policy population")
    policies, calibration = {}, {}
    for arm, value in (("cheap", 0), ("strong", 1)):
        policies[f"always_{arm}"] = {
            "method": f"always_{arm}", "kind": "always", "target_gap_pp": None,
            "strong_probabilities": [value] * nf, "router_usd": [0.0] * nf,
            "rule": {"format": utility_router.FORMAT, "mode": f"always_{arm}"},
        }
    for name, method in methods.items():
        cp, cc = checked_arrays(method["calibration"]["probabilities"], method["calibration"]["expected_usd"], n)
        fp, fc = checked_arrays(method["final"]["probabilities"], method["final"]["expected_usd"], nf)
        fees = method.get("router_usd")
        if fees is not None:
            cal_fees, final_fees = np.asarray(fees["calibration"], dtype=float), np.asarray(fees["final"], dtype=float)
            core.require(
                cal_fees.shape == (n,) and final_fees.shape == (nf,)
                and np.isfinite(cal_fees).all() and np.isfinite(final_fees).all()
                and (cal_fees >= 0).all() and (final_fees >= 0).all(), "Invalid router overhead rows",
            )
        else:
            cal_fees = final_fees = None
        result = utility_router.calibrate(cp, cc, y, costs, targets_pp=TARGETS, router_usd=cal_fees)
        calibration[name] = result
        for target, selected in result["selected"].items():
            rule = selected["policy"]
            bypass = rule["mode"].startswith("always_")
            decisions = utility_router.replay(rule, fp, fc).astype(int).tolist()
            policies[f"{name}@{target}pp"] = {
                "method": name, "kind": "always" if bypass else "learned",
                "target_gap_pp": selected["target_gap_pp"], "strong_probabilities": decisions,
                "router_usd": [0.0] * nf if bypass else None if final_fees is None else final_fees.tolist(),
                "rule": rule, "calibration_candidate_id": selected["candidate_id"],
                "metadata_assisted": bool(method.get("metadata_assisted", False)),
            }
    random = {}
    for target in TARGETS:
        selected = random_policy(y, costs, target)
        random[str(target)] = selected
        policies[f"random@{target}pp"] = {
            "method": "random", "kind": "random", "target_gap_pp": target,
            "strong_probabilities": [selected["strong_probability"]] * nf, "router_usd": [0.0] * nf,
            "rule": selected,
        }
    bundle = {
        "format": "routerbench-frozen-final-routing-v1", "protocol": PROTOCOL,
        "ids": list(final_ids), "policies": policies, "final_outcomes_accessed": False,
        "frozen_before_final_model_outcomes": True,
    }
    if final_rows is not None:
        core.require([row["id"] for row in final_rows] == list(final_ids), "Final metadata order changed")
        bundle["source_rows"] = [{key: row[key] for key in ("id", "source", "family", "task")} for row in final_rows]
    return {"methods": calibration, "random": random}, bundle


def checked_fees(projection, rows):
    core.require(
        projection["format"] == "routerbench-singleton-embedding-cost-projection-v1"
        and projection["model"] == core.MODEL and projection["hf_revision"] == core.REVISION
        and projection["instruction"] == core.INSTRUCTION
        and projection["batch_validation"]["rows"] == len(rows)
        and projection["batch_validation"]["every_observed_batch_reproduced"]
        and projection["singleton_validation"]["rows"] == 90
        and projection["singleton_validation"]["batches"] == 90
        and projection["singleton_validation"]["every_observed_batch_reproduced"],
        "Complete, physically validated singleton fee projection required",
    )
    items = projection["rows"]
    core.require(
        [(r["id"], r["role"]) for r in items] == [(r["id"], r["role"]) for r in rows],
        "Singleton fee population/order differs",
    )
    values = []
    for row in items:
        value = Decimal(row["projected_singleton_usd"])
        core.require(value.is_finite() and value >= 0, "Invalid singleton USD projection")
        values.append(float(value))
    return {role: [v for v, r in zip(values, rows, strict=True) if r["role"] == role] for role in ("calibration", "final")}


def checked_score_roles(score, plan):
    roles = {}
    for role in ("calibration", "final"):
        value = score["roles"][role]
        core.require(value["ids"] == plan["populations"][role], "Score/source ID order differs")
        checked_arrays(value["probabilities"], value["expected_usd"], len(value["ids"]))
        roles[role] = value
    return roles


def freeze(run, calibration_grades, embedding_scores, native_scores, topic_scores, fee_projection, output):
    run, output = Path(run), Path(output)
    core.require(not output.exists() and not (run / "final-access.json").exists(), "Preserve existing freezes/final holdout")
    plan, rows = core.checked_plan(run)
    plan_hash = core.sha(run / "plan.json")
    grades = core.read(calibration_grades)
    cal_rows, y, observed = core.grade_arrays(plan, rows, grades, "calibration")
    core.require([r["id"] for r in cal_rows] == plan["populations"]["calibration"], "Wrong calibration order")
    embedding, trained, topic = (core.read(run / name) for name in ("embedding.json", "native/trained.json", "topic-model.json"))
    core.require(
        len({m["training_grades_sha256"] for m in (embedding, trained, topic)}) == 1
        and embedding["grader_sha256"] == trained["grader_sha256"] == grades["grader_sha256"]
        and all(m["plan_sha256"] == plan_hash for m in (embedding, trained, topic))
        and embedding["fit"]["training_ids"] == trained["training_ids"] == topic["training_ids"] == plan["populations"]["train"]
        and embedding["fit"]["conditional_cost_usd"] == trained["conditional_cost_usd"],
        "Training population, grader, plan or cost-estimator mismatch",
    )
    files = {"embedding": Path(embedding_scores), "native": Path(native_scores), "topic": Path(topic_scores)}
    scores = {name: core.read(path) for name, path in files.items()}
    core.require(all(v["plan_sha256"] == plan_hash and v["outcomes_accessed"] is False for v in scores.values()),
                 "Scores must be input-only and bound to the prepared study")
    core.require(
        scores["embedding"]["model_sha256"] == core.sha(run / "embedding.json")
        and scores["native"]["trained_sha256"] == core.sha(run / "native/trained.json")
        and scores["topic"]["model_sha256"] == core.sha(run / "topic-model.json"), "Score/model binding changed",
    )
    fees = checked_fees(core.read(fee_projection), rows)
    methods = {
        "embedding": {**checked_score_roles(scores["embedding"], plan), "router_usd": fees},
        "topic": checked_score_roles(scores["topic"], plan),
    }
    for variant in ("base", "lora"):
        item = scores["native"]["variants"][variant]
        core.require(item["conditional_cost_usd"] == trained["conditional_cost_usd"], "Native cost model changed")
        methods[variant] = checked_score_roles(item, plan)
    methods["topic_metadata_diagnostic"] = {
        **{role: methods["topic"][role]["source_topic_diagnostic"] for role in ("calibration", "final")},
        "metadata_assisted": True,
    }
    calibration, bundle = freeze_arrays(methods, y, observed, plan["populations"]["final"],
                                        final_rows=[r for r in rows if r["role"] == "final"])
    provenance = {
        "plan_sha256": plan_hash, "source_plan_sha256": plan["source_plan_sha256"],
        "pair_spec_sha256": plan["pair_spec_sha256"], "acquisition_protocol_sha256": plan["acquisition_protocol_sha256"],
        "calibration_grades_sha256": core.sha(calibration_grades), "grader_sha256": grades["grader_sha256"],
        "training_grades_sha256": embedding["training_grades_sha256"],
        "scores_sha256": {name: core.sha(path) for name, path in files.items()},
        "fee_projection_sha256": core.sha(fee_projection), "cost_scope": plan["pair"]["cost_scope"],
        "calibration_source_sha256": core.sha(__file__), "utility_source_sha256": core.sha(utility_router.__file__),
    }
    output.mkdir(parents=True)
    core.write(output / "calibration.json", {"format": "routerbench-paired-calibration-v1", **provenance,
                                            "protocol": PROTOCOL, "ids": plan["populations"]["calibration"], **calibration})
    bundle.update(provenance, calibration_sha256=core.sha(output / "calibration.json"))
    checked_frozen_bundle(bundle, rows)
    core.write(output / "frozen-final-routes.json", bundle)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "calibration-grades", "embedding-scores", "native-scores", "topic-scores", "fee-projection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = freeze(**vars(args))
    print(f"Frozen {len(result['policies'])} policies for {len(result['ids'])} untouched final inputs")


if __name__ == "__main__":
    main()
