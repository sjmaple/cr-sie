"""Fit cost-aware routing only from teacher answers, then score against answer keys."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import numpy as np

import difficulty_policy as embeddings
import economics
import learn
from cloud import digest


def teacher_agreement(cheap, strong, problem=""):
    answer = economics.final_answer(strong)
    if not answer:
        return None
    measured = economics.grade(cheap, answer, problem)
    return int(measured["correct"]) if measured["reference_parsed"] else None


def select_threshold(scores, teacher_cheap, cheap_cost, strong_cost, router_credits, loss):
    choices = np.r_[-np.inf, np.unique(scores), np.inf]
    candidates = []
    for tau in choices:
        use_strong = scores >= tau
        quality = np.where(use_strong, 1.0, teacher_cheap).mean()
        cost = np.where(use_strong, strong_cost, cheap_cost).mean() + router_credits
        if quality >= 1 - loss:
            candidates.append((cost, -quality, float(tau), float(use_strong.mean())))
    cost, minus_quality, tau, strong_share = min(candidates)
    return {
        "tau": None if np.isneginf(tau) else ("infinity" if np.isposinf(tau) else tau),
        "validation_relative_quality": -minus_quality,
        "validation_mean_credits": cost,
        "validation_strong_share": strong_share,
    }


def threshold(value):
    return -np.inf if value is None else (np.inf if value == "infinity" else float(value))


@cache
def harmful_switch_upper_bound(errors, count, alpha=0.05):
    """One-sided exact binomial upper bound, including zero observed harms."""
    if not 0 <= errors <= count or count <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid binomial counts or confidence level")
    if errors == count:
        return 1.0
    if errors == 0:
        return 1 - alpha ** (1 / count)
    indices = np.arange(errors + 1)
    coefficients = np.array([math.lgamma(count + 1) - math.lgamma(i + 1) - math.lgamma(count - i + 1) for i in indices])
    low, high = errors / count, 1.0
    for _ in range(60):
        p = (low + high) / 2
        terms = coefficients + indices * math.log(p) + (count - indices) * math.log1p(-p)
        log_cdf = float(np.logaddexp.reduce(terms))
        if log_cdf > math.log(alpha):
            low = p
        else:
            high = p
    return high


def train(
    directory,
    reference,
    *,
    development_limit=None,
    development_tag="",
    supervised_reference=None,
    label_source="teacher_agreement",
    label_provenance=None,
):
    # The default teacher path never reads gold-for-independent-scoring-only.json.
    # The explicitly separate gold control supplies graded development labels.
    if label_source not in ("teacher_agreement", "teacher_review", "gold_benefit"):
        raise ValueError("Unknown label source")
    if (supervised_reference is None) != (label_source == "teacher_agreement"):
        raise ValueError("Explicit benefit labels require an explicit label source")
    inputs = economics.read(directory / "inputs.json")
    settings = economics.read(directory / "answer-settings.json")
    train_rows, val_rows = inputs["train"], inputs["validation"]
    if development_limit is not None:
        train_rows, val_rows = train_rows[:development_limit], val_rows[:development_limit]
    selected = train_rows + val_rows
    records = economics.answer_records(directory, {r["id"]: r["request"] for r in selected}, settings["arms"])
    cheap_name, strong_name = settings["cheap_arm"], settings["strong_arm"]
    labels, valid = {}, {}
    for row in selected:
        key = row["id"]
        cheap, strong = records[(key, cheap_name)], records[(key, strong_name)]
        label = (
            supervised_reference[key]
            if supervised_reference is not None
            else teacher_agreement(cheap["answer"], strong["answer"], row["request"])
        )
        labels[key] = label
        valid[key] = label is not None and (supervised_reference is not None or strong["finish_reason"] != "length")
    train_mask = np.array([valid[r["id"]] for r in train_rows])
    val_mask = np.array([valid[r["id"]] for r in val_rows])
    if train_mask.sum() < 32 or val_mask.sum() < 32:
        raise ValueError("Insufficient usable teacher references")
    x = embeddings.vectors(train_rows, reference / "main/dev-embeddings")
    vx = embeddings.vectors(val_rows, reference / "main/dev-embeddings")[val_mask]
    y = np.array([1 - labels[r["id"]] if valid[r["id"]] else 0 for r in train_rows])
    val_kept = [r for r in val_rows if valid[r["id"]]]
    vy = np.array([labels[r["id"]] for r in val_kept])
    costs = {
        name: np.array([records[(r["id"], name)]["credits"] for r in val_kept]) for name in (cheap_name, strong_name)
    }
    result = {
        "answer_settings_sha256": digest(settings),
        "inputs_sha256": digest(inputs),
        "reference_targets": labels,
        "supervision": label_source,
        "label_provenance": label_provenance,
        "teacher_coverage": {"train": int(train_mask.sum()), "validation": int(val_mask.sum())},
        "router_credits_per_request": settings["router_credits_per_request"],
        "policies": {},
        "training_reads_gold": label_source == "gold_benefit",
        "primary": f"ridge{len(train_rows)}_loss1",
        "selection": "Within the largest training prefix, select the lowest validation cost at the 1pp reference-quality budget; all smaller-prefix and 2/5pp policies are retained as predeclared comparisons.",
        "development_only": development_limit is not None,
        "development_ids": {"train": [r["id"] for r in train_rows], "validation": [r["id"] for r in val_rows]},
    }
    for count in (64, 128, 256, 512):
        if count > len(train_rows):
            continue
        mask = train_mask[:count]
        tx, ty = x[:count][mask], y[:count][mask]
        if len(ty) < 32:
            continue
        cost_targets = np.array(
            [
                np.log1p(max(0, records[(r["id"], strong_name)]["credits"] - records[(r["id"], cheap_name)]["credits"]))
                for r in train_rows[:count]
                if valid[r["id"]]
            ]
        )
        cost_fits = [learn.fit(tx, cost_targets, penalty) for penalty in learn.PENALTIES]
        cost_index = int(np.argmin([np.mean((fit[2] - cost_targets) ** 2) for fit in cost_fits]))
        cost_weights, cost_bias, _ = cost_fits[cost_index]
        saved_credits = predicted_savings(vx, cost_weights, cost_bias)
        for method in ("knn", "ridge"):
            policy = {
                "method": method,
                "requested_labels": count,
                "usable_labels": len(ty),
                "supervision": result["supervision"],
            }
            if method == "ridge":
                fits = [learn.fit(tx, ty, penalty) for penalty in learn.PENALTIES]
                index = int(np.argmin([np.mean((fit[2] - ty) ** 2) for fit in fits]))
                weights, bias, _ = fits[index]
                scores = vx @ weights + bias
                policy.update(weights=weights.tolist(), bias=bias, penalty=learn.PENALTIES[index])
            else:
                scores = embeddings.knn(tx, ty, vx)
                policy.update(
                    train_ids=[r["id"] for r in train_rows[:count] if valid[r["id"]]], train_targets=ty.tolist()
                )
            for allocation in ("risk", "risk_per_saved_credit"):
                candidate = policy | {"allocation": allocation}
                ranking = scores
                prefix = method
                if allocation == "risk_per_saved_credit":
                    ranking = (
                        scores / saved_credits
                        if supervised_reference is not None
                        else risk_per_credit(scores, saved_credits, len(ty))
                    )
                    prefix += "_value"
                    candidate.update(
                        cost_weights=cost_weights.tolist(),
                        cost_bias=cost_bias,
                        cost_penalty=learn.PENALTIES[cost_index],
                    )
                for loss_pp in (1, 2, 5):
                    selected_threshold = select_threshold(
                        ranking,
                        vy,
                        costs[cheap_name],
                        costs[strong_name],
                        settings["router_credits_per_request"],
                        loss_pp / 100,
                    )
                    result["policies"][f"{prefix}{count}_loss{loss_pp}"] = (
                        candidate | selected_threshold | {"quality_loss_target_pp": loss_pp}
                    )
    candidates = [
        (p["validation_mean_credits"], -p["validation_relative_quality"], name)
        for name, p in result["policies"].items()
        if p["requested_labels"] == len(train_rows) and p["quality_loss_target_pp"] == 1
    ]
    result["primary"] = min(candidates)[2]
    # Persist the exact development answers that determine labels and thresholds.
    result["development_answers_sha256"] = digest([records[k] for k in sorted(records)])
    suffix = f"-{development_tag}" if development_tag else ""
    name = "routing-frozen.json" if development_limit is None else f"development-fit-{development_limit}{suffix}.json"
    name = source_prefix(label_source) + name
    if (directory / name).exists():
        raise FileExistsError(directory / name)
    learn.write(directory / name, result)
    return {
        "primary": result["primary"],
        "teacher_coverage": result["teacher_coverage"],
        "policies": {
            k: {
                key: p[key]
                for key in (
                    "requested_labels",
                    "usable_labels",
                    "tau",
                    "validation_relative_quality",
                    "validation_mean_credits",
                    "validation_strong_share",
                )
            }
            for k, p in result["policies"].items()
        },
    }


def predicted_savings(x, weights, bias):
    return np.maximum(1.0, np.expm1(np.clip(x @ np.asarray(weights) + bias, 0, 20)))


def source_prefix(label_source):
    return {"teacher_agreement": "", "teacher_review": "review-", "gold_benefit": "gold-"}[label_source]


def risk_per_credit(scores, savings, count):
    # A small smoothed floor avoids pretending a negative ridge prediction
    # means a literally zero chance of disagreement.
    probability = (np.clip(scores, 0, 1) * count + 0.5) / (count + 1)
    return probability / savings


def allocate_scores(policy, scores, vectors):
    if policy.get("allocation") == "risk_per_saved_credit":
        savings = predicted_savings(vectors, policy["cost_weights"], policy["cost_bias"])
        if policy.get("supervision") in ("gold_benefit", "teacher_review"):
            return scores / savings
        return risk_per_credit(scores, savings, policy["usable_labels"])
    return scores


def evaluation_metrics(use_strong, qc, qs, cc, cs, overhead, *, seed=137):
    chosen_q = np.where(use_strong, qs, qc)
    chosen_cost = np.where(use_strong, cs, cc) + overhead
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(qs), (4000, len(qs)))
    gap = (qs - chosen_q)[index].mean(axis=1) * 100
    savings = (1 - chosen_cost[index].mean(axis=1) / cs[index].mean(axis=1)) * 100
    harms = int(np.sum((~use_strong) & (qs == 1) & (qc == 0)))
    return {
        "n": len(qs),
        "accuracy": float(chosen_q.mean()),
        "quality_loss_pp": float((qs - chosen_q).mean() * 100),
        "quality_loss_95ci_pp": np.quantile(gap, [0.025, 0.975]).tolist(),
        "credits_per_request": float(chosen_cost.mean()),
        "usd_per_1000": float(chosen_cost.mean() * 0.01),
        "saving_vs_strong_pct": float((1 - chosen_cost.mean() / cs.mean()) * 100),
        "saving_vs_strong_95ci_pct": np.quantile(savings, [0.025, 0.975]).tolist(),
        "cheap_share": float((~use_strong).mean()),
        "lost_correct_answers": harms,
        "gained_correct_answers": int(np.sum((~use_strong) & (qs == 0) & (qc == 1))),
        "harmful_switch_rate_upper95_pp": 100 * harmful_switch_upper_bound(harms, len(qs)),
    }


def freeze_evaluation(directory, sources, primary_source):
    """Declare every comparison and the primary result before generating test answers."""
    if (directory / "evaluation-plan.json").exists():
        raise FileExistsError(directory / "evaluation-plan.json")
    if list(directory.glob("answers-test*.jsonl")) or (directory / "router-receipts.jsonl").exists():
        raise ValueError("Cannot change comparisons after held-out collection starts")
    if primary_source not in sources or primary_source == "gold_benefit":
        raise ValueError("Primary no-gold recipe must be a declared teacher source")
    settings, inputs = economics.read(directory / "answer-settings.json"), economics.read(directory / "inputs.json")
    groups = {}
    for source in sources:
        name = source_prefix(source) + "routing-frozen.json"
        frozen = economics.read(directory / name)
        if frozen["answer_settings_sha256"] != digest(settings) or frozen["inputs_sha256"] != digest(inputs):
            raise ValueError("Policy does not match the frozen answers")
        if frozen.get("development_only") or frozen["supervision"] != source:
            raise ValueError("Wrong or development-only policy source")
        groups[source] = {"file": name, "sha256": digest(frozen), "primary_policy": frozen["primary"]}
    result = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "answer_settings_sha256": digest(settings),
        "inputs_sha256": digest(inputs),
        "groups": groups,
        "primary": primary_source + "_" + groups[primary_source]["primary_policy"],
        "comparisons": "Always strong, always cheap, previous difficulty routers, and all declared 64/128/256/512-label policies at 1/2/5pp reference-quality budgets. Only the frozen primary receives the primary success test.",
        "source_sha256": {
            name: digest((Path(__file__).parent / name).read_text())
            for name in (
                "economics.py",
                "economics_policy.py",
                "economics_router.py",
                "economics_review.py",
                "economics_supervised.py",
                "learn.py",
                "difficulty_policy.py",
            )
        },
    }
    learn.write(directory / "evaluation-plan.json", result)
    return result


def acquisition_costs(directory, reference, frozen, measured, prefix):
    data = economics.read(directory / "inputs.json")
    settings = economics.read(directory / "answer-settings.json")
    dev = data["train"] + data["validation"]
    records = economics.answer_records(directory, {r["id"]: r["request"] for r in dev}, settings["arms"])
    embedding_calls = economics.rows(reference / "main/dev-embeddings/calls.jsonl")
    if any(r["status"] != "success" for r in embedding_calls):
        raise ValueError("Training embedding acquisition has unsuccessful calls")
    result = {}
    for name, policy in frozen["policies"].items():
        ids = {r["id"] for r in data["train"][: policy["requested_labels"]] + data["validation"]}
        answer_credits = sum(row["credits"] for (key, _), row in records.items() if key in ids)
        # A batched embedding receipt is atomic: count the whole intersecting
        # batch, never pretend it was a separately billed per-item receipt.
        embedding_credits = sum(r["credits"] for r in embedding_calls if ids.intersection(r["ids"]))
        review_costs = (frozen.get("label_provenance") or {}).get("review_costs", {})
        review_credits = sum(r["cloud_credits"] for key, r in review_costs.items() if key in ids)
        native_calls = sum(r["self_hosted_calls"] for key, r in review_costs.items() if key in ids)
        total = answer_credits + embedding_credits + review_credits
        saving = measured["always_strong"]["credits_per_request"] - measured[f"{prefix}_{name}"]["credits_per_request"]
        result[name] = {
            "paired_answer_acquisition_usd": answer_credits * 0.00001,
            "embedding_acquisition_usd": embedding_credits * 0.00001,
            "teacher_review_cloud_usd": review_credits * 0.00001,
            "teacher_review_self_hosted_calls": native_calls,
            "recorded_cloud_acquisition_usd": total * 0.00001,
            "requests_to_recover_recorded_cloud_acquisition": math.ceil(total / saving) if saving > 0 else None,
            "breakeven_excludes_self_hosted_teacher_compute": native_calls > 0,
        }
    return result


def evaluate(directory, reference, posttrain):
    settings = economics.read(directory / "answer-settings.json")
    evaluation_plan, groups = economics.frozen_policies(directory)
    data = economics.read(directory / "inputs.json")
    gold = economics.read(directory / "gold-for-independent-scoring-only.json")
    if economics.read(directory / "plan.json")["gold_sha256"] != digest(gold):
        raise ValueError("Independent answer keys changed")
    test = data["test"]
    records = economics.answer_records(directory, {r["id"]: r["request"] for r in test}, settings["arms"])
    arrays, parse_counts = {}, {}
    for alias in ("cheap", "strong"):
        name = settings[f"{alias}_arm"]
        calls = [records[(r["id"], name)] for r in test]
        grades = [
            economics.grade(c["answer"], gold[c["id"]], row["request"]) for c, row in zip(calls, test, strict=True)
        ]
        arrays[f"q_{alias}"] = np.array([g["correct"] for g in grades], dtype=int)
        arrays[f"c_{alias}"] = np.array([c["credits"] for c in calls])
        parse_counts[alias] = {
            "unparsed_gold": sum(not g["reference_parsed"] for g in grades),
            "unparsed_answers": sum(not g["answer_parsed"] for g in grades),
            "length_stops": sum(c["finish_reason"] == "length" for c in calls),
        }
    qc, qs, cc, cs = (arrays[k] for k in ("q_cheap", "q_strong", "c_cheap", "c_strong"))
    receipt_path = directory / "router-receipts.jsonl"
    if not receipt_path.exists():
        raise ValueError("Collect actual per-request router receipts before financial evaluation")
    receipts = economics.rows(receipt_path)
    by_id = {r["id"]: r for r in receipts}
    if (
        len(by_id) != len(test)
        or len(receipts) != len(test)
        or any(by_id[r["id"]]["status"] != "success" for r in test)
    ):
        raise ValueError("Missing, duplicate or failed router receipts")
    if any(by_id[r["id"]]["input_sha256"] != digest(r["request"]) for r in test):
        raise ValueError("Router receipt belongs to a different request")
    overhead = np.array([by_id[r["id"]]["credits"] for r in test])
    tx = embeddings.vectors(test, directory / "router-embeddings")
    train_x = embeddings.vectors(data["train"], reference / "main/dev-embeddings")
    train_index = {r["id"]: i for i, r in enumerate(data["train"])}
    decisions = {"always_strong": np.ones(len(test), dtype=bool), "always_cheap": np.zeros(len(test), dtype=bool)}
    old = economics.read(posttrain / "confirmation-ridge.json")
    if old["ids"] != [r["id"] for r in test]:
        raise ValueError("Difficulty baseline inputs changed")
    original = economics.read(reference / "main/frozen.json")
    if digest(original) != old["original_frozen_sha256"]:
        raise ValueError("Previously frozen difficulty policies changed")
    difficulty_route_changes = {}
    for name, values in old["predictions"].items():
        policy = original["selection"]["embedding"]["policies"][name]
        if policy["method"] == "ridge":
            scores = tx @ np.asarray(policy["weights"]) + policy["bias"]
        else:
            count = policy["count"]
            scores = embeddings.knn(train_x[:count], embeddings.target(data["train"][:count]), tx)
        decisions[f"difficulty_{name}"] = scores >= policy["tau"]
        difficulty_route_changes[name] = int(np.sum(decisions[f"difficulty_{name}"] != np.asarray(values, dtype=bool)))
    for group, frozen in groups.items():
        for name, policy in frozen["policies"].items():
            if policy["method"] == "ridge":
                scores = tx @ np.array(policy["weights"]) + policy["bias"]
            else:
                bank = train_x[[train_index[k] for k in policy["train_ids"]]]
                scores = embeddings.knn(bank, np.asarray(policy["train_targets"]), tx)
            decisions[f"{group}_{name}"] = allocate_scores(policy, scores, tx) >= threshold(policy["tau"])
    measured = {
        name: evaluation_metrics(route, qc, qs, cc, cs, 0 if name.startswith("always_") else overhead)
        for name, route in decisions.items()
    }
    difficulty = np.asarray([r["label"] for r in test])
    by_difficulty = {
        str(label): {
            "n": int(np.sum(difficulty == label)),
            "cheap_accuracy": float(qc[difficulty == label].mean()),
            "strong_accuracy": float(qs[difficulty == label].mean()),
            "strong_advantage_pp": float((qs - qc)[difficulty == label].mean() * 100),
        }
        for label in (0, 1)
    }
    result = {
        "answer_settings_sha256": digest(settings),
        "evaluation_plan_sha256": digest(evaluation_plan),
        "routing_frozen_sha256": {group: digest(frozen) for group, frozen in groups.items()},
        "test_answer_records_sha256": digest([records[k] for k in sorted(records)]),
        "metrics": measured,
        "accuracy_by_human_difficulty": by_difficulty,
        "format_audit": parse_counts,
        "difficulty_routes_changed_with_fresh_embeddings": difficulty_route_changes,
        "arrays": {k: v.tolist() for k, v in arrays.items()},
        "router_credits": overhead.tolist(),
        "routes": {k: v.astype(int).tolist() for k, v in decisions.items()},
        "primary": evaluation_plan["primary"],
        "acquisition": {
            group: acquisition_costs(directory, reference, frozen, measured, group) for group, frozen in groups.items()
        },
        "credit_usd": 0.00001,
        "meaning": "Actual paired model answers scored independently against source answer keys; dollars convert recorded credits. One generated answer per model/request. No teacher agreement reported as answer accuracy.",
    }
    primary = measured[result["primary"]]
    result["primary_gate"] = {
        "point_estimate_pass": primary["saving_vs_strong_pct"] >= 25 and primary["quality_loss_pp"] <= 1,
        "noninferiority_supported_at_1pp": primary["harmful_switch_rate_upper95_pp"] <= 1,
        "noninferiority_method": "Conservative one-sided exact 95% upper bound on harmful switches; ignores gains and avoids zero-event bootstrap overconfidence.",
        "cost_saving_supported": primary["saving_vs_strong_95ci_pct"][0] > 0,
    }
    learn.write(directory / "economic-results.json", result)
    return {"metrics": measured, "accuracy_by_human_difficulty": by_difficulty, "primary_gate": result["primary_gate"]}


def audit_development(directory, reference, count=64, tag="", gold_control=False, label_source="teacher_agreement"):
    """Independent diagnostic only; gold never changes labels or thresholds."""
    data = economics.read(directory / "inputs.json")
    settings = economics.read(directory / "answer-settings.json")
    suffix = f"-{tag}" if tag else ""
    prefix = source_prefix("gold_benefit" if gold_control else label_source)
    fitted_name = "routing-frozen.json" if count is None else f"development-fit-{count}{suffix}.json"
    frozen = economics.read(directory / f"{prefix}{fitted_name}")
    gold = economics.read(directory / "gold-for-independent-scoring-only.json")
    if digest(gold) != economics.read(directory / "plan.json")["gold_sha256"]:
        raise ValueError("Independent answer keys changed")
    rows = data["validation"][:count]
    records = economics.answer_records(directory, {r["id"]: r["request"] for r in rows}, settings["arms"])
    calls = {alias: [records[(r["id"], settings[f"{alias}_arm"])] for r in rows] for alias in ("cheap", "strong")}
    qc, qs = [
        np.array(
            [
                economics.grade(c["answer"], gold[c["id"]], row["request"])["correct"]
                for c, row in zip(calls[a], rows, strict=True)
            ],
            dtype=int,
        )
        for a in ("cheap", "strong")
    ]
    cc, cs = [np.array([c["credits"] for c in calls[a]]) for a in ("cheap", "strong")]
    x = embeddings.vectors(rows, reference / "main/dev-embeddings")
    train_rows = data["train"][:count]
    tx = embeddings.vectors(train_rows, reference / "main/dev-embeddings")
    train_index = {r["id"]: i for i, r in enumerate(train_rows)}
    routes = {
        "always_strong": np.ones(len(rows), dtype=bool),
        "always_cheap": np.zeros(len(rows), dtype=bool),
        "oracle_safe_not_deployable": (qc < qs) | (cc >= cs),
    }
    for name, policy in frozen["policies"].items():
        if policy["method"] == "ridge":
            scores = x @ np.array(policy["weights"]) + policy["bias"]
        else:
            bank = tx[[train_index[k] for k in policy["train_ids"]]]
            scores = embeddings.knn(bank, np.array(policy["train_targets"]), x)
        routes[name] = allocate_scores(policy, scores, x) >= threshold(policy["tau"])
    metrics = {
        name: evaluation_metrics(
            route, qc, qs, cc, cs, 0 if name.startswith("always_") else settings["router_credits_per_request"]
        )
        for name, route in routes.items()
    }
    agreement = [
        teacher_agreement(c["answer"], s["answer"], row["request"])
        for c, s, row in zip(calls["cheap"], calls["strong"], rows, strict=True)
    ]
    result = {
        "development_only": True,
        "validation_n": len(rows),
        "router_cost_is_validation_allowance": True,
        "metrics": metrics,
        "teacher_audit": {
            "teacher_correct": int(qs.sum()),
            "agreed": sum(a == 1 for a in agreement),
            "disagreed": sum(a == 0 for a in agreement),
            "unusable_reference": sum(a is None for a in agreement),
            "disagreement_but_no_teacher_quality_gain": int(
                sum(a == 0 and s <= c for a, c, s in zip(agreement, qc, qs, strict=True))
            ),
            "agreement_but_teacher_correct_cheap_wrong": int(
                sum(a == 1 and s > c for a, c, s in zip(agreement, qc, qs, strict=True))
            ),
        },
        "by_difficulty": {
            str(label): {
                "n": sum(r["label"] == label for r in rows),
                "cheap_accuracy": float(qc[[r["label"] == label for r in rows]].mean()),
                "strong_accuracy": float(qs[[r["label"] == label for r in rows]].mean()),
            }
            for label in (0, 1)
        },
    }
    result["policy_supervision"] = frozen.get("supervision", "teacher_agreement")
    learn.write(directory / f"{prefix}development-audit-{count if count is not None else 'full'}{suffix}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("evidence/difficulty-policy"))
    parser.add_argument("--posttrain", type=Path, default=Path("evidence/posttrain"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--evaluate", action="store_true")
    action.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("teacher_agreement", "teacher_review", "gold_benefit"),
        default=["teacher_agreement"],
    )
    parser.add_argument(
        "--primary-source", choices=("teacher_agreement", "teacher_review"), default="teacher_agreement"
    )
    parser.add_argument("--development-limit", type=int, choices=(64, 128, 256, 512))
    parser.add_argument("--development-tag", default="")
    args = parser.parse_args()
    if args.freeze:
        result = freeze_evaluation(args.run, args.sources, args.primary_source)
    elif args.evaluate:
        result = evaluate(args.run, args.reference, args.posttrain)
    else:
        result = train(
            args.run, args.reference, development_limit=args.development_limit, development_tag=args.development_tag
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
