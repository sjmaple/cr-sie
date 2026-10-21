"""Advanced training/evaluation for newly collected public reproduction labels.

Use the existing paired_training and paired_native CLIs for the actual fitting.
This adapter supplies explicit grade paths for the remaining lifecycle steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import paired_calibration
import paired_report
import paired_topic
import paired_training as core
import reproduction


KIND = "new_label_acquisition"


def fit_topic(run, grades):
    return paired_topic.fit_and_score(run, grades, Path(run) / "topic-scores.json")


def calibrate(run, calibration_grades, fee_projection):
    """Read only calibration labels; preserve the public retrospective envelope."""
    run = Path(run)
    output = run / "frozen"
    core.require(not output.exists(), "Preserve previous calibration; use a new run")
    result = paired_calibration.freeze(
        run,
        calibration_grades,
        run / "embedding-scores.json",
        run / "native-scores.json",
        run / "topic-scores.json",
        fee_projection,
        output / "internal",
    )
    envelope = {
        "format": reproduction.FORMAT,
        "evaluation_mode": reproduction.MODE,
        "reproduction_kind": KIND,
        "frozen_before_final_model_outcomes": False,
        "final_outcomes_accessed_during_calibration": False,
        "generated_routes_path": reproduction.INTERNAL_ROUTES,
        "generated_routes_sha256": core.sha(output / reproduction.INTERNAL_ROUTES),
        "plan_sha256": result["plan_sha256"],
        "acquisition_protocol_sha256": result["acquisition_protocol_sha256"],
        "grader_sha256": result["grader_sha256"],
        "adapter_sha256": core.sha(__file__),
        "interpretation": (
            "New paid responses on published questions. Calibration uses only the calibration split; "
            "published final outcomes may already be known, so this is not a new blind evaluation."
        ),
    }
    core.write(output / "frozen-final-routes.json", envelope)
    return envelope


def evaluate(run, final_grades, output):
    """Apply the new routes to their own final acquisition, without refitting."""
    run, final_grades, output = Path(run), Path(final_grades), Path(output)
    core.require(not output.exists(), "Preserve previous reports")
    plan, rows = core.checked_plan(run)
    routing = run / "frozen/frozen-final-routes.json"
    bundle, envelope = reproduction.checked_routes(routing, rows)
    core.require(
        envelope is not None
        and envelope.get("reproduction_kind") == KIND
        and envelope["adapter_sha256"] == core.sha(__file__),
        "Need this adapter's explicit new-label reproduction freeze",
    )
    core.require(
        bundle["plan_sha256"] == core.sha(run / "plan.json")
        and bundle["source_plan_sha256"] == plan["source_plan_sha256"]
        and bundle["pair_spec_sha256"] == plan["pair_spec_sha256"]
        and bundle["acquisition_protocol_sha256"]
        == envelope["acquisition_protocol_sha256"]
        == plan["acquisition_protocol_sha256"]
        and bundle["grader_sha256"] == envelope["grader_sha256"]
        and bundle["cost_scope"] == plan["pair"]["cost_scope"],
        "New-label freeze/plan provenance changed",
    )
    for method, expected in bundle["scores_sha256"].items():
        core.require(core.sha(run / f"{method}-scores.json") == expected, "Frozen input scores changed")
    grades = core.read(final_grades)
    raw_routing = routing.parent / reproduction.INTERNAL_ROUTES
    paired_report.check_lineage(bundle, grades, core.sha(raw_routing))
    result = paired_report.evaluate(grades["rows"], bundle)
    result.update(
        format="train-a-router-reproduction-report-v1",
        evaluation_mode=reproduction.MODE,
        reproduction_kind=KIND,
        new_blind_evaluation=False,
        original_evidence_replaced=False,
        public_outcomes_previously_available=True,
    )
    result["provenance"] = {
        "adapter_sha256": core.sha(__file__),
        "reporter_sha256": core.sha(paired_report.__file__),
        "reproduction_routing_sha256": core.sha(routing),
        "generated_routes_sha256": core.sha(raw_routing),
        "final_grades_sha256": core.sha(final_grades),
        "parent_acquisition_protocol_sha256": grades["parent_acquisition_protocol_sha256"],
        "parent_grader_sha256": grades["parent_grader_sha256"],
        "acquisition_protocol_sha256": grades["acquisition_protocol_sha256"],
        "grader_sha256": grades["grader_sha256"],
        "cost_scope": grades["cost_scope"],
    }
    core.write(output, result)
    return {"output": str(output), "sha256": core.sha(output), "questions": result["questions"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    topic = sub.add_parser("fit-topic", help="Fit the fixed topic control from explicit training grades")
    topic.add_argument("--run", type=Path, required=True)
    topic.add_argument("--grades", type=Path, required=True)
    cal = sub.add_parser("calibrate", help="Freeze all policies from explicit calibration grades")
    cal.add_argument("--run", type=Path, required=True)
    cal.add_argument("--calibration-grades", type=Path, required=True)
    cal.add_argument("--fee-projection", type=Path, required=True)
    report = sub.add_parser("evaluate", help="Evaluate the same freeze against its new final grades")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--final-grades", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"fit-topic": fit_topic, "calibrate": calibrate, "evaluate": evaluate}[command](**args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
