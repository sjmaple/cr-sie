"""Estimate Cloud token charges for paired answers collected on your own server.

These estimates train the router's cost model. They are not Modal bills or
observed Cloud charges: every input token is priced without a cache discount.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import solver
from data import digest, file_hash, write

MODELS = {"cheap": "Qwen/Qwen3.5-4B", "strong": "Qwen/Qwen3.8-27B-FP8"}


def read(path):
    return json.loads(Path(path).read_text())


def _fraction(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def rates_from_snapshot(snapshot):
    if snapshot.get("currency") != "usd" or not snapshot.get("pricing_version"):
        raise ValueError("Need a versioned USD price snapshot")
    rates, selected = {}, []
    for arm, model in MODELS.items():
        rates[arm] = {}
        for unit in ("input_tokens", "output_tokens"):
            matches = [
                row
                for row in snapshot.get("items", [])
                if row.get("operation") == "generate"
                and row.get("model") == model
                and row.get("profile") == "default"
                and row.get("unit") == unit
                and row.get("region") == "us"
                and row.get("currency") == "usd"
            ]
            if len(matches) != 1:
                raise ValueError(f"Need exactly one default/us price for {model}/{unit}")
            value = _fraction(matches[0]["usd_per_unit"]) * 100000
            if value < 0:
                raise ValueError("Token prices cannot be negative")
            rates[arm][unit] = {"numerator": value.numerator, "denominator": value.denominator}
            selected.append(matches[0])
    return rates, selected


def physical_costs(record, rates):
    if record.get("status") != "complete" or record.get("has_unknown_charges"):
        raise ValueError("Cost estimates require a complete, unambiguous answer record")
    calls = record.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("Every answer must retain every physical call")
    priced = []
    for call in calls:
        tokens = call.get("tokens", {})
        if any(type(tokens.get(key)) is not int or tokens[key] < 0 for key in ("prompt_tokens", "completion_tokens")):
            raise ValueError("Physical calls need nonnegative integer token counts")
        price = math.ceil(
            tokens["prompt_tokens"] * _fraction(rates["input_tokens"])
            + tokens["completion_tokens"] * _fraction(rates["output_tokens"])
        )
        priced.append(
            {
                "prompt_tokens": tokens["prompt_tokens"],
                "completion_tokens": tokens["completion_tokens"],
                "credits_proxy": price,
                "finish_reason": call.get("finish_reason"),
            }
        )
    return priced


def build(training, prices_path, *, grades_path=None):
    training, prices_path = Path(training), Path(prices_path)
    grades_path = training / "grades-train.json" if grades_path is None else Path(grades_path)
    plan_path = training / "plan.json"
    collection_path = training / "answers-train" / "collection.json"
    binding_path = training / "answers-train" / "collection-plan.json"
    grades, collection = read(grades_path), read(collection_path)
    plan, inputs, settings, binding = map(
        read, (plan_path, training / "inputs.json", training / "settings.json", binding_path)
    )
    if plan.get("inputs_sha256") != digest(inputs) or plan.get("settings_sha256") != digest(settings):
        raise ValueError("Training inputs or solver settings changed")
    solver.collection_binding(binding, training, "train", inputs["train"], settings, None, binding["self_hosted"])
    collected_ids = [row["id"] for row in inputs["train"]]
    graded_ids = [row["id"] for row in grades["rows"]]
    filtered = "collected_ids" in grades or "excluded_ids" in grades
    excluded_ids = grades.get("excluded_ids", [])
    if (
        not isinstance(excluded_ids, list)
        or any(not isinstance(identity, str) for identity in excluded_ids)
        or len(set(excluded_ids)) != len(excluded_ids)
        or (filtered and (grades.get("collected_ids") != collected_ids or "excluded_ids" not in grades))
        or not set(excluded_ids).issubset(collected_ids)
        or excluded_ids != [identity for identity in collected_ids if identity in set(excluded_ids)]
        or graded_ids != [identity for identity in collected_ids if identity not in set(excluded_ids)]
    ):
        raise ValueError("Retained and excluded IDs must partition the original collected training population in order")
    if (
        grades.get("split") != "train"
        or grades.get("plan_sha256") != file_hash(plan_path)
        or grades.get("collection_plan_sha256") != file_hash(binding_path)
        or grades.get("collection_summary_sha256") != file_hash(collection_path)
        or collection.get("status") != "complete"
        or collection.get("completed") != 2 * len(collected_ids)
        or collection.get("failed") != 0
        or collection.get("missing", 0) != 0
        or collection.get("unknown_execution_attempts", 0) != 0
    ):
        raise ValueError("Training grades must bind the complete paired collection and plan")
    prices = read(prices_path)
    rates, public_rows = rates_from_snapshot(prices)
    input_by_id = {row["id"]: row for row in inputs["train"]}
    rows, groups, seen = [], defaultdict(list), set()
    for grade in grades["rows"]:
        identity = grade["id"]
        if identity in seen or not identity or Path(identity).name != identity:
            raise ValueError("Training IDs must be unique filenames")
        seen.add(identity)
        row = {"id": identity}
        for arm, model in MODELS.items():
            path = training / "answers-train" / f"{identity}-{arm}.json"
            if file_hash(path) != grade[f"{arm}_record_sha256"]:
                raise ValueError("Raw solver answer changed after grading")
            record = read(path)
            solver.validate_record(record, input_by_id[identity], arm, settings)
            if record.get("id") != identity or record.get("arm") != arm or record.get("model") != model:
                raise ValueError("Raw solver answer uses a different question, arm or model")
            correct = grade[f"{arm}_correct"]
            if type(correct) is not bool:
                raise ValueError("Correctness labels must be booleans")
            calls = physical_costs(record, rates[arm])
            total = sum(call["credits_proxy"] for call in calls)
            row.update(
                {
                    f"{arm}_credits_proxy": total,
                    f"{arm}_correct": correct,
                    f"{arm}_record_sha256": file_hash(path),
                    f"{arm}_calls": calls,
                }
            )
            groups[f"{arm}_{'correct' if correct else 'wrong'}"].append(total)
        rows.append(row)
    if set(groups) != {f"{arm}_{state}" for arm in MODELS for state in ("correct", "wrong")}:
        raise ValueError("Need correct and incorrect examples for both solvers")
    return {
        "format": "sie-training-uncached-cloud-cost-proxy-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "split": "train",
        "n": len(rows),
        "credit_usd": 1e-5,
        "training_plan_sha256": file_hash(plan_path),
        "training_grades_sha256": file_hash(grades_path),
        "training_collection_sha256": file_hash(collection_path),
        "training_collection_plan_sha256": file_hash(binding_path),
        "collected_ids": collected_ids,
        "excluded_ids": excluded_ids,
        "builder_source_sha256": file_hash(Path(__file__)),
        "price_source": {
            "sha256": file_hash(prices_path),
            "pricing_version": prices["pricing_version"],
            "rate_book_sha256": prices["rate_book_sha256"],
            "public_rows": public_rows,
        },
        "rates_credits_per_token": rates,
        "formula": "Sum every physical call, including finalizers: ceil(prompt_tokens*input_rate + completion_tokens*output_rate), using exact rational rates.",
        "cache_assumption": "All input tokens use the uncached rate; saved token summaries do not retain cached-token counts.",
        "cost_interpretation": "Uncached Cloud list-price estimates for the observed output lengths, not measured Cloud charges or self-hosting costs.",
        "conditional_means": {
            name: {"n": len(values), "sum_credits_proxy": sum(values), "mean_credits_proxy": sum(values) / len(values)}
            for name, values in groups.items()
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument(
        "--grades",
        type=Path,
        help="Separate source-reviewed training grades; defaults to the collection's grades-train.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build(args.training, args.prices, grades_path=args.grades)
    write(args.output, result)
    print(json.dumps({"n": result["n"], "conditional_means": result["conditional_means"]}, indent=2))


if __name__ == "__main__":
    main()
