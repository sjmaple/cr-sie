"""Score the separate arithmetic test after every router prediction is saved.

Generation costs are estimates from recorded token counts and pinned public
prices. Embedding charges are observed, amortized over the recorded batches.
This transfer study never chooses new thresholds from arithmetic outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import benefit
import benefit_costs
import generalization_score
import outcome_summary
import solver
import solver_report
import teacher_format
from benefit_expanded_gold_v2 import outcome_grader_metadata
from data import digest, file_hash, write

REQUIRED = {"embedding-original", "embedding-expanded", "embedding-teacher", "native-original", "native-expanded"}
OPTIONAL = {"embedding-teacher-control"}


def read(path):
    return json.loads(Path(path).read_text())


def validate_scores(study, bindings):
    """Bind saved input-only predictions and unchanged policies before grading."""
    study = Path(study)
    source, inputs = generalization_score.inputs(study)
    if not REQUIRED <= bindings.keys() or set(bindings) - REQUIRED - OPTIONAL:
        raise ValueError("Need every declared router, with only the optional matched teacher control")
    result = {}
    for name, (score_path, frozen_path) in bindings.items():
        score_path, frozen_path = Path(score_path), Path(frozen_path)
        score, frozen = read(score_path), read(frozen_path)
        if (
            score.get("format") != "sie-arithmetic-router-scores-v1"
            or score.get("n") != source["n"]
            or score.get("ids") != [row["id"] for row in inputs]
            or score.get("inputs_sha256") != file_hash(study / "inputs.json")
            or score.get("source_plan_sha256") != file_hash(study / "source-plan.json")
            or score.get("frozen_sha256") != file_hash(frozen_path)
            or score.get("scoring_source_sha256") != file_hash(Path(generalization_score.__file__))
        ):
            raise ValueError(f"Saved transfer predictions differ from the frozen inputs or router: {name}")
        native = name.startswith("native-")
        controls = frozen["matched_benchmark_control"] if name == "embedding-teacher-control" else frozen
        expected = (
            {variant: row["policies"] for variant, row in frozen["models"].items()}
            if native
            else {score["design"]: controls["selections"]}
        )
        if (
            name == "embedding-teacher"
            and "matched_benchmark_control" in frozen
            and "embedding-teacher-control" not in bindings
        ):
            raise ValueError("Save the matched benchmark control before opening arithmetic outcomes")
        if set(score["models"]) != set(expected):
            raise ValueError("Transfer model variants changed")
        for variant, model in score["models"].items():
            values = np.asarray(model["scores"], dtype=float)
            probabilities = np.asarray(model["probabilities"], dtype=float)
            if (
                values.shape != (len(inputs),)
                or not np.isfinite(values).all()
                or probabilities.shape != (len(inputs), 2)
                or not np.isfinite(probabilities).all()
                or not ((probabilities >= 0) & (probabilities <= 1)).all()
                or model["policies"] != expected[variant]
                or set(model["policies"]) != {"0", "1", "2", "5"}
            ):
                raise ValueError("Transfer scores or frozen policy set changed")
        result[name] = {
            "score_file": score_path.name,
            "score_sha256": file_hash(score_path),
            "frozen_sha256": file_hash(frozen_path),
            "models": score["models"],
            "design": score["design"],
        }
    return source, inputs, result


def grade(study, prices_path, bindings):
    study, prices_path = Path(study), Path(prices_path)
    if (study / "grades.json").exists():
        raise FileExistsError(study / "grades.json")
    source, inputs, predictions = validate_scores(study, bindings)
    access = {
        "format": "sie-arithmetic-transfer-outcome-access-v1",
        "authorized_at": benefit.now(),
        "source_plan_sha256": file_hash(study / "source-plan.json"),
        "predictions": predictions,
        "primary_calibration_target_pp": 1,
        "grading": outcome_grader_metadata() | {"format_normalizer_sha256": file_hash(Path(teacher_format.__file__))},
        "statement": "Every saved router prediction and competition-calibrated cutoff is bound before arithmetic solver outcomes are opened. No arithmetic tuning is permitted.",
    }
    access_path = study / "outcome-access.json"
    if access_path.exists():
        previous = read(access_path)
        access["authorized_at"] = previous["authorized_at"]
        if access != previous:
            raise ValueError("Cannot change frozen predictions after outcome access")
    else:
        write(access_path, access)
    gold_path, rows_path = study / "gold-for-scoring-only.json", study / "rows.json"
    if any(file_hash(study / name) != expected for name, expected in source["files_sha256"].items()):
        raise ValueError("Arithmetic source artifacts changed")
    gold, metadata = read(gold_path), read(rows_path)
    if [row["id"] for row in metadata] != [row["id"] for row in inputs]:
        raise ValueError("Arithmetic metadata order differs")
    collection = study / "answers"
    plan, summary = read(collection / "collection-plan.json"), read(collection / "collection.json")
    expected_files = {f"{row['id']}-{arm}.json" for row in inputs for arm in solver.SETTINGS["arms"]}
    if (
        summary.get("status") != "complete"
        or summary.get("completed") != 2 * len(inputs)
        or summary.get("missing") != 0
        or summary.get("unknown_execution_attempts") != 0
        or plan.get("source_plan_sha256") != file_hash(study / "source-plan.json")
        or plan.get("settings_sha256") != digest(solver.SETTINGS)
        or set(plan.get("records_sha256", {})) != expected_files
    ):
        raise ValueError("Need the complete paired arithmetic collection")
    rates, public_rows = benefit_costs.rates_from_snapshot(read(prices_path))
    rows = []
    for question, meta in zip(inputs, metadata, strict=True):
        row = {"id": question["id"], "original_split": meta["original_split"], "family": meta["family"]}
        for arm in solver.SETTINGS["arms"]:
            path = collection / f"{question['id']}-{arm}.json"
            if file_hash(path) != plan["records_sha256"][path.name]:
                raise ValueError("Collected answer changed after gathering")
            record = read(path)
            solver.validate_record(record, question, arm, solver.SETTINGS)
            measured = teacher_format.grade_reference(question["request"], gold[question["id"]], record["answer"])
            if measured["status"] in ("gold_unparseable", "grading_error", "unit_contract_unresolved"):
                raise ValueError(f"Arithmetic grading requires review: {question['id']}/{arm}")
            calls = benefit_costs.physical_costs(record, rates[arm])
            row.update(
                {
                    f"{arm}_correct": measured["correct"],
                    f"{arm}_grade": measured,
                    f"{arm}_record_sha256": file_hash(path),
                    f"{arm}_credits": sum(call["credits_proxy"] for call in calls),
                    f"{arm}_cost_calls": calls,
                }
            )
        rows.append(row)
    result = {
        "format": "sie-arithmetic-transfer-grades-v1",
        "outcome_access_sha256": file_hash(study / "outcome-access.json"),
        "source_plan_sha256": file_hash(study / "source-plan.json"),
        "collection_plan_sha256": file_hash(collection / "collection-plan.json"),
        "collection_summary_sha256": file_hash(collection / "collection.json"),
        "gold_sha256": file_hash(gold_path),
        "grading": access["grading"],
        "prices_sha256": file_hash(prices_path),
        "public_price_rows": public_rows,
        "credit_usd": 1e-5,
        "cost_scope": "The credits fields are uncached Cloud list-price estimates computed from every physical generation call. They are neither observed Cloud charges nor Modal GPU bills.",
        "rows": rows,
    }
    write(study / "grades.json", result)
    return result


def embedding_overhead(study, ids):
    """Allocate each observed embedding batch charge to its member inputs."""
    receipt = read(Path(study) / "embeddings" / "collection.json")
    costs = {}
    for call in receipt["calls"]:
        if not call["ids"] or not solver.valid_credit(call.get("credits")):
            raise ValueError("Embedding batch has no valid charge receipt")
        for identity in call["ids"]:
            if identity in costs:
                raise ValueError("Embedding input charged in multiple batches")
            costs[identity] = call["credits"] / len(call["ids"])
    if set(costs) != set(ids):
        raise ValueError("Embedding charges do not cover the exact transfer population")
    return costs


def summarize(rows, predictions, overhead):
    ids = [row["id"] for row in rows]
    output = {}
    for name, record in predictions.items():
        for variant, model in record["models"].items():
            key = name + "/" + variant if name.startswith("native-") else name
            values = np.asarray(model["scores"])
            policies = {}
            for target, policy in model["policies"].items():
                mode = policy["mode"]
                decisions = (values >= policy["cutoff"]).tolist() if mode == "threshold" else mode
                metrics = solver_report.summarize(
                    rows,
                    decisions,
                    router_credits=(None if name.startswith("native-") else [overhead[identity] for identity in ids])
                    if mode == "threshold"
                    else 0,
                )
                policies[target] = {"mode": mode, "cutoff": policy["cutoff"], "metrics": metrics}
            output[key] = policies
    return output


def report(study):
    study = Path(study)
    access, grades = read(study / "outcome-access.json"), read(study / "grades.json")
    if grades["outcome_access_sha256"] != file_hash(study / "outcome-access.json") or access[
        "source_plan_sha256"
    ] != file_hash(study / "source-plan.json"):
        raise ValueError("Transfer grade/access binding changed")
    rows, predictions = grades["rows"], access["predictions"]
    ids = [row["id"] for row in rows]
    overhead = embedding_overhead(study, ids)
    baselines = {mode: solver_report.summarize(rows, mode, router_credits=0) for mode in solver_report.BASELINES}
    subsets = {}
    for split in ("test", "train"):
        positions = [i for i, row in enumerate(rows) if row["original_split"] == split]
        selected = [rows[i] for i in positions]
        subset_predictions = {
            name: record
            | {
                "models": {
                    variant: model | {"scores": [model["scores"][i] for i in positions]}
                    for variant, model in record["models"].items()
                }
            }
            for name, record in predictions.items()
        }
        if selected:
            subsets[split] = {
                "n": len(selected),
                "baselines": {
                    mode: solver_report.summarize(selected, mode, router_credits=0) for mode in solver_report.BASELINES
                },
                "methods": summarize(selected, subset_predictions, overhead),
            }
    return {
        "format": "sie-arithmetic-transfer-report-v1",
        "n": len(rows),
        "ids": ids,
        "grades_sha256": file_hash(study / "grades.json"),
        "source_plan_sha256": file_hash(study / "source-plan.json"),
        "embedding_receipt_sha256": file_hash(study / "embeddings" / "collection.json"),
        "primary_calibration_target_pp": 1,
        "population": "Custom held-out arithmetic transfer sample: all1319 official GSM8K test rows plus681 fixed source-train rows; no arithmetic rows trained or calibrated any router. This is not a standard GSM8K leaderboard score or a representative production traffic mix.",
        "cost_scope": grades["cost_scope"]
        + " Embedding overhead is observed batch16 Cloud charges divided equally among batch members. Local classifier compute is unpriced. Native router hosting is unpriced, so only gross solver savings and break-even router cost are reported for native threshold policies.",
        "selection": "All0/1/2/5pp competition-calibrated policies transfer unchanged;1pp is the declared primary target. Realized arithmetic gaps can exceed those calibration targets.",
        "intervals": "Paired question-level95% bootstrap intervals are descriptive. Selected duplicate families are distinct, but semantic dependence, pretraining contamination, future traffic and sampling-mixture uncertainty remain unmeasured.",
        "baselines": baselines,
        "methods": summarize(rows, predictions, overhead),
        "by_original_source_split": subsets,
    }


def markdown(result):
    labels = {
        "embedding-original": "Qwen embeddings + classifiers; original training set",
        "embedding-expanded": "Qwen embeddings + classifiers; expanded training set",
        "embedding-teacher": "Qwen embeddings + classifiers; teacher labels",
        "embedding-teacher-control": "Qwen embeddings + classifiers; matched benchmark labels",
        **{
            f"native-{population}/{variant}": f"Laya {'unchanged' if variant == 'base' else '+ LoRA'}; {population} training set"
            for population in ("original", "expanded")
            for variant in ("base", "lora")
        },
    }
    entries = [
        ("Always Qwen 27B", result["baselines"]["always_strong"]),
        ("Always Qwen 4B", result["baselines"]["always_cheap"]),
        *[
            (
                outcome_summary.policy_label(labels.get(name, name), policies["1"]["mode"]),
                policies["1"]["metrics"],
            )
            for name, policies in result["methods"].items()
        ],
    ]
    lines = [
        "# Arithmetic transfer: unchanged routers on a different workload",
        "",
        result["population"],
        "",
        "Generation costs below are estimates from observed tokens and pinned Cloud prices.",
        "",
        "The primary policies used a 1-point competition-calibration target. Positive gaps lose accuracy; negative gaps gain accuracy.",
        "",
        *outcome_summary.quality_table(entries, estimated=True),
        "",
        *outcome_summary.cost_table(entries, estimated=True),
    ]
    lines.extend(
        [
            "",
            result["cost_scope"],
            "",
            outcome_summary.COST_NOTE,
            "",
            result["selection"],
            "",
            result["intervals"],
            "",
            outcome_summary.INTERVAL_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("grade", "report"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--binding", nargs=3, action="append", metavar=("NAME", "SCORES", "FROZEN"))
    args = parser.parse_args()
    if args.command == "grade":
        if args.prices is None or not args.binding:
            parser.error("Grading requires public prices and every saved router binding")
        grade(args.study, args.prices, {name: (scores, frozen) for name, scores, frozen in args.binding})
    else:
        result = report(args.study)
        if (args.study / "report.json").exists():
            raise FileExistsError(args.study / "report.json")
        write(args.study / "report.json", result)
        (args.study / "report.md").write_text(markdown(result))
        print(json.dumps({"n": result["n"], "methods": list(result["methods"])}))


if __name__ == "__main__":
    main()
