"""A fixed competition-name baseline for the paired-benefit routing study.

This control asks whether problem embeddings add value beyond supplied exam
metadata. Its smoothing strength is fixed before training outcomes are read.
It uses no embedding/model API call for routing. Application CPU costs are not
priced, just as the embedding router's small linear-head CPU cost is unpriced.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import benefit
import evidence
import solver_policies
import solver_report
from data import audit, digest, file_hash, write

PROTOCOL = {
    "name": "competition_only",
    "target": "strong_correct - cheap_correct",
    "smoothing": 20,
    "fit": "(sum of training benefits for the competition + 20 * global training mean) / (competition training count + 20)",
    "unknown_competition": "Use the global training mean",
    "selection": "No smoothing search; same calibration-only cutoff selection and quality targets as the embedding head",
    "router_api_credits": 0,
    "cost_scope": "Competition name is already supplied; no model API call is needed. Application CPU cost is unpriced.",
    "targets": [1, 0, 2, 5],
    "test_boundary": "Freeze the control before the embedding head's test-access receipt exists",
}
PLAN_FORMAT = "sie-benefit-competition-control-plan-v1"
FROZEN_FORMAT = "sie-benefit-competition-control-frozen-v1"
# Downloader pins change when the final bundle is published. Bind its verified
# manifest/data bytes below; the routing and reporting implementation stays fixed.
CODE_FILES = ("benefit_control.py", "benefit.py", "solver_policies.py", "solver_report.py", "data.py")


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(UTC).isoformat()


def code_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in CODE_FILES}


def fit_metadata(contests, targets):
    """Shrink each competition's signed training mean toward the global mean."""
    if not contests or len(contests) != len(targets):
        raise ValueError("Need aligned nonempty training competitions and benefits")
    if any(not isinstance(value, str) or not value.strip() for value in contests):
        raise ValueError("Training competition names must be nonempty strings")
    if any(type(value) is not int or value not in (-1, 0, 1) for value in targets):
        raise ValueError("Training benefits must be signed integer outcomes")
    mean = sum(targets) / len(targets)
    grouped = {}
    for contest, target in zip(contests, targets, strict=True):
        record = grouped.setdefault(contest, {"n": 0, "sum": 0})
        record["n"] += 1
        record["sum"] += target
    return {
        "n": len(targets),
        "global_mean": mean,
        "smoothing": PROTOCOL["smoothing"],
        "competitions": {
            contest: record
            | {"score": (record["sum"] + PROTOCOL["smoothing"] * mean) / (record["n"] + PROTOCOL["smoothing"])}
            for contest, record in sorted(grouped.items())
        },
    }


def predict(model, contests):
    if any(not isinstance(contest, str) or not contest.strip() for contest in contests):
        raise ValueError("Competition names must be nonempty strings")
    return [model["competitions"].get(contest, {}).get("score", model["global_mean"]) for contest in contests]


def metadata(baseline, head_plan):
    baseline = Path(baseline)
    evidence.verify(baseline)
    dataset = read(baseline / "data.json")
    audit(dataset)
    if digest(dataset) != head_plan["dataset_sha256"]:
        raise ValueError("Competition metadata differs from the embedding head's dataset")
    result = {}
    for split in ("train", "calibration", "test"):
        source = {row["id"]: row for row in dataset["splits"][split]}
        result[split] = []
        for selected in head_plan["populations"][split]:
            row = source[selected["id"]]
            if digest(row["request"]) != selected["request_sha256"]:
                raise ValueError("Competition metadata belongs to a different request")
            result[split].append({"id": row["id"], "contest": row["contest"]})
    return result, {
        "manifest.json": file_hash(baseline / "manifest.json"),
        "data.json": file_hash(baseline / "data.json"),
    }


def prepare(head, baseline):
    head = Path(head)
    if (head / "frozen.json").exists() or (head / "test-access.json").exists():
        raise ValueError("Prepare this fixed control before the embedding head is fitted or test access is authorized")
    head_plan = benefit._checked_plan(read(head / "plan.json"))
    populations, artifacts = metadata(baseline, head_plan)
    return benefit.seal(
        {
            "format": PLAN_FORMAT,
            "prepared_at": now(),
            "protocol": PROTOCOL,
            "head_plan_sha256": file_hash(head / "plan.json"),
            "dataset_sha256": head_plan["dataset_sha256"],
            "eligibility_sha256": head_plan["eligibility_sha256"],
            "populations": populations,
            "baseline_artifacts_sha256": artifacts,
            "source_code_sha256": code_hashes(),
            "preparation_boundary": "Prepared from source metadata before head fitting; no training outcomes or test answers are read by this command",
        }
    )


def checked_plan(plan, head, baseline):
    benefit.checked(plan, PLAN_FORMAT)
    head_plan = benefit._checked_plan(read(Path(head) / "plan.json"))
    populations, artifacts = metadata(baseline, head_plan)
    if (
        plan["protocol"] != PROTOCOL
        or plan["source_code_sha256"] != code_hashes()
        or plan["head_plan_sha256"] != file_hash(Path(head) / "plan.json")
        or plan["dataset_sha256"] != head_plan["dataset_sha256"]
        or plan["eligibility_sha256"] != head_plan["eligibility_sha256"]
        or plan["populations"] != populations
        or plan["baseline_artifacts_sha256"] != artifacts
    ):
        raise ValueError("Competition control protocol, source metadata or head binding changed after preparation")
    return plan


def selections(rows, scores, rate):
    return {
        str(target): solver_report.select_cutoff(rows, scores, target, router_credits=0, credit_usd=rate)
        for target in PROTOCOL["targets"]
    }


def fit_from_head(plan, head_frozen):
    ids = [row["id"] for row in plan["populations"]["train"]]
    if head_frozen["head"]["training_ids"] != ids:
        raise ValueError("Competition control and embedding head use different training populations")
    model = fit_metadata([row["contest"] for row in plan["populations"]["train"]], head_frozen["head"]["targets"])
    scores = {
        split: predict(model, [row["contest"] for row in plan["populations"][split]])
        for split in ("calibration", "test")
    }
    rows = head_frozen["calibration_rows"]
    if [row["id"] for row in rows] != [row["id"] for row in plan["populations"]["calibration"]]:
        raise ValueError("Competition cutoff selection has different calibration IDs")
    return model, scores, selections(rows, scores["calibration"], head_frozen["credit_usd"])


def freeze(run, head, baseline):
    run, head = Path(run), Path(head)
    for path in (run / "frozen.json", run / "report.json", head / "test-access.json"):
        if path.exists():
            raise FileExistsError(path)
    plan = checked_plan(read(run / "plan.json"), head, baseline)
    head_frozen = benefit._checked_frozen(head)
    if head_frozen["plan_file_sha256"] != plan["head_plan_sha256"]:
        raise ValueError("Competition control binds a different fitted head")
    model, scores, chosen = fit_from_head(plan, head_frozen)
    result = benefit.seal(
        {
            "format": FROZEN_FORMAT,
            "frozen_at": now(),
            "plan": plan,
            "plan_file_sha256": file_hash(run / "plan.json"),
            "head_frozen_sha256": file_hash(head / "frozen.json"),
            "training_targets_sha256": digest(head_frozen["head"]["targets"]),
            "model": model,
            "scores": scores,
            "selections": chosen,
            "calibration_grades_sha256": head_frozen["calibration_grades_sha256"],
            "grading_provenance": head_frozen["grading_provenance"],
            "credit_usd": head_frozen["credit_usd"],
        }
    )
    if (head / "test-access.json").exists():
        raise ValueError("Test access was authorized while the control was freezing")
    return result


def checked_frozen(run, head, baseline):
    run, head = Path(run), Path(head)
    frozen = benefit.checked(read(run / "frozen.json"), FROZEN_FORMAT)
    plan = checked_plan(frozen["plan"], head, baseline)
    head_frozen = benefit._checked_frozen(head)
    if frozen["plan_file_sha256"] != file_hash(run / "plan.json") or read(run / "plan.json") != plan:
        raise ValueError("Competition preparation changed after freezing")
    model, scores, chosen = fit_from_head(plan, head_frozen)
    if (
        frozen["head_frozen_sha256"] != file_hash(head / "frozen.json")
        or frozen["training_targets_sha256"] != digest(head_frozen["head"]["targets"])
        or frozen["model"] != model
        or frozen["scores"] != scores
        or frozen["selections"] != chosen
        or frozen["calibration_grades_sha256"] != head_frozen["calibration_grades_sha256"]
        or frozen["grading_provenance"] != head_frozen["grading_provenance"]
        or frozen["credit_usd"] != head_frozen["credit_usd"]
    ):
        raise ValueError("Competition control differs from its fixed training/calibration recipe")
    return frozen, head_frozen


def report(run, head, baseline, test_grades):
    run, head = Path(run), Path(head)
    frozen, head_frozen = checked_frozen(run, head, baseline)
    access = benefit.checked(read(head / "test-access.json"), benefit.ACCESS_FORMAT)
    if (
        access["frozen_sha256"] != file_hash(head / "frozen.json")
        or access["plan_sha256"] != head_frozen["plan_file_sha256"]
        or access["source_code_sha256"] != benefit.code_hashes()
        or datetime.fromisoformat(access["authorized_at"]) < datetime.fromisoformat(frozen["frozen_at"])
    ):
        raise ValueError("Competition control was not frozen before matching head test access")
    plan = head_frozen["plan"]
    ids = [row["id"] for row in frozen["plan"]["populations"]["test"]]
    grades = solver_policies._grade_rows(test_grades, "test", plan["original_solver_plan_sha256"], ids)
    if (
        grades["policies_frozen_sha256"] != plan["original_frozen_sha256"]
        or solver_policies._grading_provenance(grades) != frozen["grading_provenance"]
    ):
        raise ValueError("Competition control test grades differ from the paired solver experiment")
    measured = {}
    for target in PROTOCOL["targets"]:
        chosen = frozen["selections"][str(target)]
        routes = (
            [score >= chosen["cutoff"] for score in frozen["scores"]["test"]]
            if chosen["mode"] == "threshold"
            else chosen["mode"]
        )
        measured[str(target)] = {
            "mode": chosen["mode"],
            "cutoff": chosen["cutoff"],
            "metrics": solver_report.summarize(
                grades["rows"], routes, router_credits=0, credit_usd=frozen["credit_usd"]
            ),
        }
    return {
        "format": "sie-benefit-competition-control-report-v1",
        "policy_name": "competition_only",
        "n": len(ids),
        "ids": ids,
        "training_pool_rows": head_frozen["training_pool_rows"],
        "training_rows": frozen["model"]["n"],
        "frozen_sha256": file_hash(run / "frozen.json"),
        "head_frozen_sha256": file_hash(head / "frozen.json"),
        "head_test_access_sha256": file_hash(head / "test-access.json"),
        "test_grades_sha256": file_hash(test_grades),
        "primary_max_quality_loss_pp": 1,
        "policies": measured,
        "baselines": {
            name: solver_report.summarize(grades["rows"], name, router_credits=0, credit_usd=frozen["credit_usd"])
            for name in ("always_strong", "always_cheap")
        },
        "interpretation": "A fixed competition-name control, not a problem-text model. Scores use paired training outcomes; cutoffs use calibration only. Quality targets are empirical, not test guarantees. No additional model API charge; application CPU and acquisition/training costs are outside serving API savings.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "freeze", "report"):
        child = sub.add_parser(name)
        child.add_argument("--run", type=Path, required=True)
        child.add_argument("--head", type=Path, required=True)
        child.add_argument("--baseline", type=Path, required=True)
        if name == "report":
            child.add_argument("--test-grades", type=Path, required=True)
    args = parser.parse_args()
    output = args.run / {"prepare": "plan.json", "freeze": "frozen.json", "report": "report.json"}[args.command]
    if output.exists() or (args.command == "prepare" and args.run.exists()):
        raise FileExistsError(output)
    if args.command == "prepare":
        result = prepare(args.head, args.baseline)
    elif args.command == "freeze":
        result = freeze(args.run, args.head, args.baseline)
    else:
        result = report(args.run, args.head, args.baseline, args.test_grades)
    write(output, result)
    print(json.dumps({"output": str(output), "sha256": file_hash(output)}))


if __name__ == "__main__":
    main()
