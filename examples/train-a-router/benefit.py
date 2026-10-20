"""Learn the benefit of the strong solver from paired training outcomes.

This separate CPU experiment audits the original 512-input training pool and
uses its 511 source-eligible questions with cached Qwen vectors. Human difficulty
labels are not training targets. Calibration
answers select cost cutoffs; solver test answers may be opened only after the
new model, cutoffs and test scores are frozen and access is recorded.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import evidence
import policies
import solver_policies
import solver_report
from data import SOURCE_HASHES, SOURCE_REVISION, audit, digest, file_hash, write
from serving import INSTRUCTION, MODEL, REVISION, vectors

PROTOCOL = {
    "target": "strong_correct - cheap_correct; -1 = cheap-only success, 0 = equal outcomes, +1 = strong-only success",
    "training_pool_rows": 512,
    "training_rows": 511,
    "excluded_training_ids": ["e2h-amc-eval-661"],
    "eligibility": "Exclude the predeclared missing-choice interval question by source review before training outcomes; retain every other original input in order",
    "encoder": MODEL,
    "revision": REVISION,
    "instruction": INSTRUCTION,
    "dimensions": 2560,
    "penalties": [0.01, 0.1, 1.0, 10.0],
    "folds": 5,
    "seed": 89,
    "fold_assignment": "Sort paper groups, permute with NumPy seed89, assign whole groups round-robin to five folds",
    "selection": "Minimum pooled training out-of-fold mean squared error; exact ties prefer larger regularization",
    "fit": "Unpenalized intercept and L2-regularized linear ridge head on normalized cached vectors; refit all training rows",
    "primary_max_quality_loss_pp": 1,
    "secondary_max_quality_loss_pp": [0, 2, 5],
    "motivation": "New experiment prompted by calibration-only mismatch between human difficulty and paired solver benefit",
    "test_boundary": "Freeze before solver test OUTCOME ACCESS, not before test collection; no test outcomes used in selection",
}
CODE_FILES = ("benefit.py", "policies.py", "serving.py", "solver_policies.py", "solver_report.py", "data.py")
PLAN_FORMAT = "sie-solver-benefit-plan-v1"
FROZEN_FORMAT = "sie-solver-benefit-frozen-v1"
ACCESS_FORMAT = "sie-solver-benefit-test-access-v1"
TARGETS = (1, 0, 2, 5)


def read(path):
    return json.loads(Path(path).read_text())


def seal(value):
    return value | {"sha256": digest(value)}


def checked(value, format_name):
    if value.get("format") != format_name or value.get("sha256") != digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError("Benefit artifact format or content hash differs")
    return value


def code_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in CODE_FILES}


def now():
    return datetime.now(UTC).isoformat()


def group_folds(groups):
    """Assign whole papers to deterministic folds without inspecting outcomes."""
    if not groups or any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("Need a paper group for every training input")
    unique = sorted(set(groups))
    if len(unique) < PROTOCOL["folds"]:
        raise ValueError("Need at least five distinct training paper groups")
    order = np.random.default_rng(PROTOCOL["seed"]).permutation(unique).tolist()
    assignments = {group: index % PROTOCOL["folds"] for index, group in enumerate(order)}
    return [assignments[group] for group in groups]


def signed_targets(rows):
    """Validate paired correctness without requiring training inference prices."""
    solver_report.summarize_quality(rows, "always_strong")
    return np.array([int(row["strong_correct"]) - int(row["cheap_correct"]) for row in rows], dtype=float)


def fit_head(matrix, rows, groups):
    """Select regularization from training papers only, then refit every row."""
    matrix = np.asarray(matrix, dtype=float)
    targets = signed_targets(rows)
    if matrix.ndim != 2 or len(matrix) != len(rows) or len(groups) != len(rows) or not np.isfinite(matrix).all():
        raise ValueError("Training vectors, outcomes and groups must be aligned and finite")
    folds = np.asarray(group_folds(groups))
    candidates = []
    for penalty in PROTOCOL["penalties"]:
        predictions = np.empty(len(rows))
        fold_mse = []
        for fold in range(PROTOCOL["folds"]):
            held = folds == fold
            if (~held).sum() < 2:
                raise ValueError("Each fold needs at least two fitting rows")
            weights, bias = policies.ridge(matrix[~held], targets[~held], penalty)
            predictions[held] = matrix[held] @ weights + bias
            fold_mse.append(float(np.mean((predictions[held] - targets[held]) ** 2)))
        candidates.append(
            {"penalty": penalty, "pooled_mse": float(np.mean((predictions - targets) ** 2)), "fold_mse": fold_mse}
        )
    winner = min(candidates, key=lambda item: (item["pooled_mse"], -item["penalty"]))
    weights, bias = policies.ridge(matrix, targets, winner["penalty"])
    return {
        "weights": weights.tolist(),
        "bias": bias,
        "penalty": winner["penalty"],
        "training_ids": [row["id"] for row in rows],
        "targets": targets.astype(int).tolist(),
        "folds": folds.tolist(),
        "cv": candidates,
    }


def _inputs(baseline, original_path):
    baseline, original_path = Path(baseline), Path(original_path)
    evidence.verify(baseline)
    dataset = read(baseline / "data.json")
    audit(dataset)
    original = read(original_path)
    solver_policies.check_seal(original, solver_policies.FROZEN_FORMAT)
    solver_policies._grading_provenance(original)
    solver_policies._checked_scores(original["source_scores"])
    if (
        original["analysis_mode"] != "cost"
        or original["embedding_calibration_credits"] is None
        or original["source_scores"]["dataset_sha256"] != digest(dataset)
        or len(dataset["splits"]["train"]) != PROTOCOL["training_pool_rows"]
        or original["source_code_sha256"]
        != {name: file_hash(Path(__file__).with_name(name)) for name in ("solver_policies.py", "solver_report.py")}
    ):
        raise ValueError("Need the same frozen priced study, original 512 training inputs and reporting code")
    for split in ("calibration", "test"):
        if original["source_scores"]["ids"][split] != [row["id"] for row in dataset["splits"][split]]:
            raise ValueError("Benefit and original policy populations differ")
    solver_plan_path = original_path.with_name("plan.json")
    if file_hash(solver_plan_path) != original["plan_sha256"]:
        raise ValueError("Original solver plan differs from its freeze")
    solver_plan = read(solver_plan_path)
    artifacts = {
        name: file_hash(baseline / name)
        for name in (
            "manifest.json",
            "data.json",
            "development-embeddings/plan.json",
            "development-embeddings/vectors.jsonl.gz",
            "test-embeddings/plan.json",
            "test-embeddings/vectors.jsonl.gz",
        )
    }
    return dataset, original, solver_plan, artifacts


def _eligibility(receipt, dataset_sha, pool_ids):
    excluded = PROTOCOL["excluded_training_ids"]
    expected = [identity for identity in pool_ids if identity not in excluded]
    exclusions = receipt.get("exclusions", [])
    source_sha = receipt.get("training_gold_review_source_sha256")
    if (
        receipt.get("format") != "sie-benefit-training-eligibility-v1"
        or receipt.get("source_only") is not True
        or receipt.get("decided_before_training_outcomes") is not True
        or receipt.get("source_revision") != SOURCE_REVISION
        or receipt.get("source_hashes") != SOURCE_HASHES
        or receipt.get("dataset_sha256") != dataset_sha
        or receipt.get("original_pool_ids") != pool_ids
        or len(pool_ids) != PROTOCOL["training_pool_rows"]
        or len(set(pool_ids)) != len(pool_ids)
        or receipt.get("eligible_train_ids") != expected
        or len(expected) != PROTOCOL["training_rows"]
        or [row.get("id") for row in exclusions] != excluded
        or any(not isinstance(row.get("reason"), str) or not row["reason"].strip() for row in exclusions)
        or not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(char not in "0123456789abcdef" for char in source_sha)
    ):
        raise ValueError(
            "Eligibility must bind the original pool and exact source-only exclusion before training outcomes"
        )
    return expected


def prepare(baseline, original_path, eligibility_path, *, attest_test_outcomes_unopened):
    if attest_test_outcomes_unopened is not True:
        raise ValueError("An explicit assertion that solver test outcomes remain unopened is required")
    dataset, original, solver_plan, artifacts = _inputs(baseline, original_path)
    eligibility = read(eligibility_path)
    eligible_ids = _eligibility(eligibility, digest(dataset), [row["id"] for row in dataset["splits"]["train"]])
    populations = {
        split: [
            {"id": row["id"], "request_sha256": digest(row["request"]), "paper_group": row["paper_group"]}
            for row in dataset["splits"][split]
        ]
        for split in ("train", "calibration", "test")
    }
    populations["train_pool"] = populations["train"]
    populations["train"] = [row for row in populations["train_pool"] if row["id"] in eligible_ids]
    return seal(
        {
            "format": PLAN_FORMAT,
            "created_at": now(),
            "protocol": PROTOCOL,
            "dataset_sha256": digest(dataset),
            "original_frozen_sha256": file_hash(original_path),
            "original_solver_plan_sha256": original["plan_sha256"],
            "solver_settings_sha256": solver_plan["settings_sha256"],
            "eligibility": eligibility,
            "eligibility_sha256": file_hash(eligibility_path),
            "populations": populations,
            "training_folds": group_folds([row["paper_group"] for row in populations["train"]]),
            "baseline_artifacts_sha256": artifacts,
            "source_code_sha256": code_hashes(),
            "attest_test_outcomes_unopened": True,
            "access_limitation": "Local receipts record the declared access boundary; they cannot prove nobody read an external copy",
        }
    )


def _checked_plan(plan):
    checked(plan, PLAN_FORMAT)
    eligible_ids = _eligibility(
        plan["eligibility"], plan["dataset_sha256"], [row["id"] for row in plan["populations"]["train_pool"]]
    )
    if (
        plan["protocol"] != PROTOCOL
        or plan["source_code_sha256"] != code_hashes()
        or plan["attest_test_outcomes_unopened"] is not True
        or plan["populations"]["train"]
        != [row for row in plan["populations"]["train_pool"] if row["id"] in eligible_ids]
        or plan["training_folds"] != group_folds([row["paper_group"] for row in plan["populations"]["train"]])
    ):
        raise ValueError("Benefit protocol, training folds or source code changed after preparation")
    return plan


def _training(plan, training_plan_path, training_inputs_path, training_grades_path, original):
    training_plan, inputs, grades = map(read, (training_plan_path, training_inputs_path, training_grades_path))
    expected = plan["populations"]["train"]
    if (
        training_plan["dataset_sha256"] != plan["dataset_sha256"]
        or training_plan["settings_sha256"] != plan["solver_settings_sha256"]
        or training_plan["inputs_sha256"] != digest(inputs)
        or training_plan["source_counts"]["train"] != PROTOCOL["training_pool_rows"]
        or training_plan["counts"]["train"] != PROTOCOL["training_rows"]
        or training_plan["eligibility_sha256"] != plan["eligibility_sha256"]
        or training_plan["eligible_train_ids"] != [row["id"] for row in expected]
        or training_plan["excluded_train_ids"] != PROTOCOL["excluded_training_ids"]
        or [{"id": row["id"], "request_sha256": digest(row["request"])} for row in inputs["train"]]
        != [{key: row[key] for key in ("id", "request_sha256")} for row in expected]
        or grades["split"] != "train"
        or grades["plan_sha256"] != file_hash(training_plan_path)
        or [row["id"] for row in grades["rows"]] != [row["id"] for row in expected]
    ):
        raise ValueError("Paired training outcomes differ from the original training population or solver settings")
    provenance = solver_policies._grading_provenance(grades)
    if (
        any(provenance[name] != original[name] for name in ("grader_source_sha256", "gold_review_source_sha256"))
        or grades.get("training_gold_review_source_sha256") != plan["eligibility"]["training_gold_review_source_sha256"]
        or any(
            grades.get(name) != training_plan.get(name)
            for name in (
                "grader_source_sha256",
                "gold_review_source_sha256",
                "gold_review_sha256",
                "training_gold_rules_sha256",
                "training_gold_review_source_sha256",
            )
        )
    ):
        raise ValueError("Training grades must match their reviewed source plan and the unchanged base grader")
    review_hashes = ("training_gold_rules_sha256", "training_gold_review_source_sha256")
    if any(name in grades for name in review_hashes) and any(
        not isinstance(grades.get(name), str)
        or len(grades[name]) != 64
        or any(char not in "0123456789abcdef" for char in grades[name])
        for name in review_hashes
    ):
        raise ValueError("Training-specific gold review requires both source and rule hashes")
    signed_targets(grades["rows"])
    return grades


def _selections(rows, scores, credits, rate):
    return {
        str(target): solver_report.select_cutoff(rows, scores, target, router_credits=credits, credit_usd=rate)
        for target in TARGETS
    }


def freeze(run, baseline, original_path, training_plan_path, training_inputs_path, training_grades_path):
    run, baseline = Path(run), Path(baseline)
    for name in ("frozen.json", "test-access.json", "report.json"):
        if (run / name).exists():
            raise FileExistsError(run / name)
    plan = _checked_plan(read(run / "plan.json"))
    dataset, original, solver_plan, artifacts = _inputs(baseline, original_path)
    if (
        digest(dataset) != plan["dataset_sha256"]
        or file_hash(original_path) != plan["original_frozen_sha256"]
        or artifacts != plan["baseline_artifacts_sha256"]
        or solver_plan["settings_sha256"] != plan["solver_settings_sha256"]
    ):
        raise ValueError("Original experiment or vectors changed after benefit preparation")
    grades = _training(plan, training_plan_path, training_inputs_path, training_grades_path, original)
    eligible_ids = {row["id"] for row in plan["populations"]["train"]}
    populations = dataset["splits"] | {
        "train": [row for row in dataset["splits"]["train"] if row["id"] in eligible_ids]
    }
    matrices = {
        split: vectors(
            populations[split], baseline / ("test-embeddings" if split == "test" else "development-embeddings")
        )
        for split in ("train", "calibration", "test")
    }
    if any(
        matrix.shape != (len(plan["populations"][split]), PROTOCOL["dimensions"]) for split, matrix in matrices.items()
    ):
        raise ValueError("Cached vector dimensions differ from the pinned encoder")
    head = fit_head(matrices["train"], grades["rows"], [row["paper_group"] for row in plan["populations"]["train"]])
    scores = {
        split: (matrix @ np.array(head["weights"]) + head["bias"]).tolist()
        for split, matrix in matrices.items()
        if split != "train"
    }
    selections = _selections(
        original["calibration_rows"],
        scores["calibration"],
        original["embedding_calibration_credits"],
        original["credit_usd"],
    )
    return seal(
        {
            "format": FROZEN_FORMAT,
            "frozen_at": now(),
            "plan": plan,
            "plan_file_sha256": file_hash(run / "plan.json"),
            "training_pool_rows": PROTOCOL["training_pool_rows"],
            "training_rows": len(head["training_ids"]),
            "head": head,
            "scores": scores,
            "training_artifacts_sha256": {
                "plan": file_hash(training_plan_path),
                "inputs": file_hash(training_inputs_path),
                "grades": file_hash(training_grades_path),
            },
            "training_grade_provenance": {key: value for key, value in grades.items() if key != "rows"},
            "grading_provenance": solver_policies._grading_provenance(original),
            "calibration_grades_sha256": original["calibration_grades_sha256"],
            "calibration_rows": original["calibration_rows"],
            "embedding_calibration_credits": original["embedding_calibration_credits"],
            "embedding_calibration_credits_sha256": original["embedding_calibration_credits_sha256"],
            "credit_usd": original["credit_usd"],
            "selections": selections,
            "test_access": PROTOCOL["test_boundary"],
        }
    )


def _checked_frozen(run):
    run = Path(run)
    frozen = checked(read(run / "frozen.json"), FROZEN_FORMAT)
    plan = _checked_plan(frozen["plan"])
    if frozen["plan_file_sha256"] != file_hash(run / "plan.json") or read(run / "plan.json") != plan:
        raise ValueError("Prepared benefit plan changed after freezing")
    head = frozen["head"]
    expected_ids = [row["id"] for row in plan["populations"]["train"]]
    if (
        head["training_ids"] != expected_ids
        or frozen["training_pool_rows"] != PROTOCOL["training_pool_rows"]
        or frozen["training_rows"] != PROTOCOL["training_rows"]
        or head["folds"] != plan["training_folds"]
        or len(head["targets"]) != len(expected_ids)
        or any(target not in (-1, 0, 1) for target in head["targets"])
        or np.asarray(head["weights"]).shape != (PROTOCOL["dimensions"],)
        or not np.isfinite(head["weights"]).all()
        or not np.isfinite(head["bias"])
        or [item["penalty"] for item in head["cv"]] != PROTOCOL["penalties"]
        or any(
            np.asarray(item["fold_mse"]).shape != (PROTOCOL["folds"],)
            or not np.isfinite([item["pooled_mse"], *item["fold_mse"]]).all()
            or min(item["pooled_mse"], *item["fold_mse"]) < 0
            for item in head["cv"]
        )
        or head["penalty"] != min(head["cv"], key=lambda item: (item["pooled_mse"], -item["penalty"]))["penalty"]
    ):
        raise ValueError("Frozen benefit model differs from the declared training/CV protocol")
    for split in ("calibration", "test"):
        solver_policies.finite_scores(frozen["scores"][split], len(plan["populations"][split]))
    if [row["id"] for row in frozen["calibration_rows"]] != [
        row["id"] for row in plan["populations"]["calibration"]
    ] or frozen["selections"] != _selections(
        frozen["calibration_rows"],
        frozen["scores"]["calibration"],
        frozen["embedding_calibration_credits"],
        frozen["credit_usd"],
    ):
        raise ValueError("Frozen benefit cutoffs differ from calibration-only selection")
    solver_policies._grading_provenance(frozen["grading_provenance"])
    return frozen


def authorize_test(run):
    """Record the new freeze before callers remove any held-output guard."""
    run = Path(run)
    if (run / "test-access.json").exists() or (run / "report.json").exists():
        raise FileExistsError(run / "test-access.json")
    frozen = _checked_frozen(run)
    return seal(
        {
            "format": ACCESS_FORMAT,
            "authorized_at": now(),
            "frozen_sha256": file_hash(run / "frozen.json"),
            "plan_sha256": frozen["plan_file_sha256"],
            "source_code_sha256": code_hashes(),
            "statement": "This receipt was created after the benefit model and all cutoffs froze, before opening solver test outcomes",
        }
    )


def report(run, test_grades_path, router_credits_path=None):
    """Read test outcomes only after validating the frozen model/access chain."""
    run = Path(run)
    frozen = _checked_frozen(run)
    access = checked(read(run / "test-access.json"), ACCESS_FORMAT)
    if (
        access["frozen_sha256"] != file_hash(run / "frozen.json")
        or access["plan_sha256"] != frozen["plan_file_sha256"]
        or access["source_code_sha256"] != code_hashes()
    ):
        raise ValueError("Test-access receipt does not bind this frozen benefit model")
    plan = frozen["plan"]
    ids = [row["id"] for row in plan["populations"]["test"]]
    grades = solver_policies._grade_rows(test_grades_path, "test", plan["original_solver_plan_sha256"], ids)
    if (
        grades["policies_frozen_sha256"] != plan["original_frozen_sha256"]
        or solver_policies._grading_provenance(grades) != frozen["grading_provenance"]
    ):
        raise ValueError("Test answers differ from the original frozen solver/grading experiment")
    credits, credit_sha = solver_policies._credits(
        router_credits_path,
        "test",
        plan["original_solver_plan_sha256"],
        ids,
        {row["id"]: row["request_sha256"] for row in plan["populations"]["test"]},
    )
    rows, rate = grades["rows"], frozen["credit_usd"]
    measured = {}
    for target in TARGETS:
        selection = frozen["selections"][str(target)]
        mode = selection["mode"]
        routes = (np.array(frozen["scores"]["test"]) >= selection["cutoff"]).tolist() if mode == "threshold" else mode
        measured[str(target)] = {
            "mode": mode,
            "cutoff": selection["cutoff"],
            "metrics": solver_report.summarize(
                rows, routes, router_credits=credits if mode == "threshold" else 0, credit_usd=rate
            ),
        }
    return {
        "format": "sie-solver-benefit-report-v1",
        "n": len(ids),
        "training_pool_rows": frozen["training_pool_rows"],
        "training_rows": frozen["training_rows"],
        "eligibility_sha256": plan["eligibility_sha256"],
        "ids": ids,
        "frozen_sha256": file_hash(run / "frozen.json"),
        "test_access_sha256": file_hash(run / "test-access.json"),
        "test_grades_sha256": file_hash(test_grades_path),
        "embedding_test_credits_sha256": credit_sha,
        "primary_max_quality_loss_pp": 1,
        "baselines": {
            name: solver_report.summarize(rows, name, router_credits=0, credit_usd=rate)
            for name in ("always_strong", "always_cheap")
        },
        "policies": measured,
        "interpretation": "Outcome-targeted head prompted by calibration analysis and frozen before solver test outcome access. Difficulty benchmark was already exposed. Calibration loss targets are empirical, not test guarantees. Acquisition/training costs are separate from per-request serving costs.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "freeze", "authorize-test", "report"):
        child = commands.add_parser(command)
        child.add_argument("--run", type=Path, required=True)
        if command in ("prepare", "freeze"):
            child.add_argument("--baseline", type=Path, required=True)
            child.add_argument("--original-frozen", type=Path, required=True)
        if command == "prepare":
            child.add_argument("--eligibility", type=Path, required=True)
            child.add_argument("--attest-test-outcomes-unopened", action="store_true", required=True)
        if command == "freeze":
            for name in ("training-plan", "training-inputs", "training-grades"):
                child.add_argument(f"--{name}", type=Path, required=True)
        if command == "report":
            child.add_argument("--test-grades", type=Path, required=True)
            child.add_argument("--router-credits", type=Path)
    args = parser.parse_args()
    destination = (
        args.run
        / {
            "prepare": "plan.json",
            "freeze": "frozen.json",
            "authorize-test": "test-access.json",
            "report": "report.json",
        }[args.command]
    )
    if destination.exists():
        raise FileExistsError(destination)
    if args.command == "prepare":
        if args.run.exists():
            raise FileExistsError(args.run)
        result = prepare(
            args.baseline,
            args.original_frozen,
            args.eligibility,
            attest_test_outcomes_unopened=args.attest_test_outcomes_unopened,
        )
    elif args.command == "freeze":
        result = freeze(
            args.run,
            args.baseline,
            args.original_frozen,
            args.training_plan,
            args.training_inputs,
            args.training_grades,
        )
    elif args.command == "authorize-test":
        result = authorize_test(args.run)
    else:
        result = report(args.run, args.test_grades, args.router_credits)
    write(destination, result)
    if args.command == "report":
        metrics = result["policies"]["1"]["metrics"]
        gross = "n/a" if metrics["gross_solver_savings_pct"] is None else f"{metrics['gross_solver_savings_pct']:.2f}%"
        print(
            f"Benefit router: answer accuracy {100 * metrics['answer_accuracy']:.2f}%; gap {metrics['gap_vs_strong_pp']:+.2f}pp; gross savings {gross}"
        )
        if "net_savings_pct" in metrics:
            net = "n/a" if metrics["net_savings_pct"] is None else f"{metrics['net_savings_pct']:.2f}%"
            print(
                f"Including encoder: {metrics['net_total_cost_usd_per1000']:.4f} USD/1,000 requests; net savings {net}"
            )
        else:
            print("Router overhead unpriced; net savings unknown")
    print(json.dumps({"output": str(destination), "sha256": file_hash(destination)}))


if __name__ == "__main__":
    main()
