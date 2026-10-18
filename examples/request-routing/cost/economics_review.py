"""Train from blinded solution reviews; audit teacher errors separately with gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import economics
import economics_policy
import learn
from cloud import digest


def labels(directory, input_paths, review_paths, development_limit=None, *, invalid_as_strong=False):
    data = economics.read(directory / "inputs.json")
    settings = economics.read(directory / "answer-settings.json")
    selected = data["train"][:development_limit] + data["validation"][:development_limit]
    selected_by_id = {r["id"]: r for r in selected}
    records = economics.answer_records(directory, {r["id"]: r["request"] for r in selected}, settings["arms"])
    plans = [economics.read(path) for path in input_paths]
    if any(p["gold_provided"] or p["test_provided"] or p["answer_settings_sha256"] != digest(settings) for p in plans):
        raise ValueError("Reviews must use matching, blinded development inputs")
    inputs = {}
    for plan in plans:
        for row in plan["rows"]:
            if row["id"] in inputs and inputs[row["id"]] != row:
                raise ValueError("Conflicting review inputs")
            inputs[row["id"]] = row
    reviews = {}
    for path in review_paths:
        for row in economics.rows(path):
            if row["id"] not in selected_by_id:
                continue
            if row["id"] in reviews:
                raise ValueError("Duplicate teacher reviews")
            expected = inputs[row["id"]]
            if row["input_sha256"] != digest(expected["messages"]) or row["arm_order"] != expected["arm_order"]:
                raise ValueError("Review identity or candidate ordering changed")
            expected_sources = [digest(records[(row["id"], arm)]) for arm in row["arm_order"]]
            if (
                row["source_records_sha256"] != expected_sources
                or expected["source_records_sha256"] != expected_sources
            ):
                raise ValueError("Review refers to different candidate answers")
            if row["status"] != "success" or (not row["valid_label"] and not invalid_as_strong):
                raise ValueError("Invalid teacher review; do not silently omit it from validation")
            if not row["valid_label"]:
                reviews[row["id"]] = row
                continue
            values = row["correct_by_arm"]
            if set(values) != {settings["cheap_arm"], settings["strong_arm"]} or any(
                type(v) is not bool for v in values.values()
            ):
                raise ValueError("Teacher correctness judgments must be booleans for both arms")
            reviews[row["id"]] = row
    if set(reviews) != set(selected_by_id):
        raise ValueError("Missing development reviews")
    targets = {
        key: (
            1 + int(row["correct_by_arm"][settings["cheap_arm"]]) - int(row["correct_by_arm"][settings["strong_arm"]])
            if row["valid_label"]
            else 0
        )
        for key, row in reviews.items()
    }
    provenance = {
        "input_sha256": [digest(p) for p in plans],
        "reviews_sha256": digest([reviews[key] for key in sorted(reviews)]),
        "models": sorted({r["model"] for r in reviews.values()}),
        "model_revisions": sorted({r["model_revision"] for r in reviews.values()}),
        "gold_provided": False,
        "test_provided": False,
        "invalid_review_policy": "assume_strong_only_win" if invalid_as_strong else "reject",
        "invalid_reviews": sum(not r["valid_label"] for r in reviews.values()),
        "review_costs": {
            key: {
                "cloud_credits": row.get("credits", 0) if row.get("billing") == "cloud" else 0,
                "self_hosted_calls": (1 + int("continuation" in row)) if row.get("billing") == "self_hosted" else 0,
            }
            for key, row in reviews.items()
        },
    }
    return targets, provenance, reviews


def train(directory, reference, input_paths, review_paths, development_limit=None, *, invalid_as_strong=False, tag=""):
    targets, provenance, _ = labels(
        directory, input_paths, review_paths, development_limit, invalid_as_strong=invalid_as_strong
    )
    return economics_policy.train(
        directory,
        reference,
        development_limit=development_limit,
        development_tag=tag,
        supervised_reference=targets,
        label_source="teacher_review",
        label_provenance=provenance,
    )


def audit(directory, input_paths, review_paths, development_limit=64, *, invalid_as_strong=False, tag=""):
    """Gold-based diagnostics never alter the teacher labels used by train()."""
    targets, provenance, reviews = labels(
        directory, input_paths, review_paths, development_limit, invalid_as_strong=invalid_as_strong
    )
    data = economics.read(directory / "inputs.json")
    gold = economics.read(directory / "gold-for-independent-scoring-only.json")
    settings = economics.read(directory / "answer-settings.json")
    if digest(gold) != economics.read(directory / "plan.json")["gold_sha256"]:
        raise ValueError("Independent answer keys changed")
    results = {}
    for split in ("train", "validation"):
        selected = data[split][:development_limit]
        records = economics.answer_records(directory, {r["id"]: r["request"] for r in selected}, settings["arms"])
        rows = []
        for row in selected:
            calls = {alias: records[(row["id"], settings[f"{alias}_arm"])] for alias in ("cheap", "strong")}
            actual = {
                alias: economics.grade(call["answer"], gold[row["id"]], row["request"])["correct"]
                for alias, call in calls.items()
            }
            valid = reviews[row["id"]]["valid_label"]
            estimated = (
                {alias: reviews[row["id"]]["correct_by_arm"][settings[f"{alias}_arm"]] for alias in calls}
                if valid
                else {"cheap": False, "strong": True}
            )
            rows.append(
                {
                    "id": row["id"],
                    "actual": actual,
                    "teacher": estimated,
                    "valid_review": valid,
                    "actual_benefit": int(actual["strong"]) - int(actual["cheap"]),
                    "teacher_benefit": 1 - targets[row["id"]],
                    "agreement": economics_policy.teacher_agreement(
                        calls["cheap"]["answer"], calls["strong"]["answer"], row["request"]
                    ),
                }
            )
        n = len(rows)
        results[split] = {
            "n": n,
            "valid_reviews": sum(r["valid_review"] for r in rows),
            "correct_benefit_labels": sum(r["actual_benefit"] == r["teacher_benefit"] for r in rows),
            "agreement_correct_benefit_labels": sum(
                r["agreement"] is not None and r["actual_benefit"] == 1 - r["agreement"] for r in rows
            ),
            "missed_strong_only_wins": sum(r["actual_benefit"] == 1 and r["teacher_benefit"] <= 0 for r in rows),
            "false_strong_advantages": sum(r["actual_benefit"] <= 0 and r["teacher_benefit"] == 1 for r in rows),
            "correctness_accuracy": {
                alias: sum(r["actual"][alias] == r["teacher"][alias] for r in rows) / n for alias in ("cheap", "strong")
            },
            "rows": rows,
        }
    result = {"development_only": True, "provenance": provenance, "splits": results}
    suffix = f"-{tag}" if tag else ""
    learn.write(directory / f"review-label-audit-{development_limit}{suffix}.json", result)
    return {split: {k: v for k, v in value.items() if k != "rows"} for split, value in results.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "audit"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("evidence/difficulty-policy"))
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--reviews", type=Path, action="append", required=True)
    parser.add_argument("--development-limit", type=int, choices=(64, 128, 256, 512))
    parser.add_argument("--tag", default="", help="Preserve separate development variants")
    parser.add_argument(
        "--invalid-as-strong",
        action="store_true",
        help="Explicit conservative fallback: count an incomplete review as a strong-only win; never omit it",
    )
    args = parser.parse_args()
    result = (
        audit(
            args.run,
            args.input,
            args.reviews,
            args.development_limit or 512,
            invalid_as_strong=args.invalid_as_strong,
            tag=args.tag,
        )
        if args.command == "audit"
        else train(
            args.run,
            args.reference,
            args.input,
            args.reviews,
            args.development_limit,
            invalid_as_strong=args.invalid_as_strong,
            tag=args.tag,
        )
    )
    print(json.dumps(result, indent=2))
