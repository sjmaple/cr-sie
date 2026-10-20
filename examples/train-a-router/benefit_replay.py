"""Reproduce published benefit routers without claiming a new unseen test.

These commands reuse historical plan bytes and recorded solver labels. Fitting
never calls a solver API. Native fitting requires a GPU and may download the
pinned public Laya checkpoint. Evaluation is explicitly a public regression.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import benefit
import benefit_control
import benefit_native
import benefit_summary
import evidence
import solver_policies
import solver_report
from data import file_hash, write

MODE = "public_regression_reproduction"
FORMAT = "sie-benefit-public-reproduction-v1"
PLAN_NAMES = ("head", "control", "native")


def read(path):
    return json.loads(Path(path).read_text())


def fresh(*paths):
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path.name}; use a new reproduction directory")


def reference(study):
    study = Path(study)
    if evidence.SOLVER_MANIFEST_SHA256 == "PENDING_PUBLICATION":
        raise RuntimeError("The completed public solver study has not been pinned yet")
    evidence.verify(study, manifest_sha256=evidence.SOLVER_MANIFEST_SHA256)
    return {
        "root": study,
        "baseline": study / "benefit/baseline",
        "training": study / "benefit/training",
        **{name: study / "benefit" / name for name in PLAN_NAMES},
    }


def checked_run(run, study):
    run = Path(run)
    sources = reference(study)
    receipt = read(run / "reproduction.json")
    if (
        receipt.get("format") != FORMAT
        or receipt.get("evaluation_mode") != MODE
        or receipt.get("reference_manifest_sha256") != file_hash(Path(study) / "manifest.json")
        or receipt.get("wrapper_source_sha256") != file_hash(Path(__file__))
    ):
        raise ValueError("Reproduction source or pinned reference changed")
    for name in PLAN_NAMES:
        expected = receipt["historical_plan_files_sha256"][name]
        if file_hash(run / name / "plan.json") != expected or file_hash(sources[name] / "plan.json") != expected:
            raise ValueError("Historical plan copy changed")
        if (run / name / "test-access.json").exists():
            raise ValueError("Public reproduction must not invent a new unopened-test receipt")
    return sources


def prepare(study, output):
    output, study = Path(output), Path(study)
    fresh(output)
    if output.resolve() == study.resolve() or study.resolve() in output.resolve().parents:
        raise ValueError("Keep the reproduction outside the immutable study directory")
    sources = reference(study)
    benefit._checked_plan(read(sources["head"] / "plan.json"))
    benefit_control.checked_plan(read(sources["control"] / "plan.json"), sources["head"], sources["baseline"])
    benefit_native.checked_plan(sources["native"], sources["head"])
    hashes = {name: file_hash(sources[name] / "plan.json") for name in PLAN_NAMES}
    output.mkdir(parents=True, exist_ok=False)
    for name in PLAN_NAMES:
        destination = output / name / "plan.json"
        destination.parent.mkdir()
        shutil.copyfile(sources[name] / "plan.json", destination)
    receipt = {
        "format": FORMAT,
        "evaluation_mode": MODE,
        "reference_manifest_sha256": file_hash(study / "manifest.json"),
        "wrapper_source_sha256": file_hash(Path(__file__)),
        "historical_plan_files_sha256": hashes,
        "interpretation": "Plan files are unchanged historical records, not new assertions that this student has not seen published test outcomes. No new authorize-test receipt is created.",
    }
    write(output / "reproduction.json", receipt)
    return receipt


def stage_start(run, stage, *destinations):
    fresh(run / f"{stage}-started.json", run / f"{stage}.json", *destinations)
    write(run / f"{stage}-started.json", {"evaluation_mode": MODE, "stage": stage})


def stage_done(run, stage, files):
    receipt = {
        "evaluation_mode": MODE,
        "stage": stage,
        "files_sha256": {name: file_hash(run / name) for name in files},
    }
    write(run / f"{stage}.json", receipt)
    return receipt


def completed_stage(run, stage):
    receipt = read(run / f"{stage}.json")
    if receipt.get("evaluation_mode") != MODE or receipt.get("stage") != stage:
        raise ValueError("Missing reproduction stage receipt")
    for name, expected in receipt["files_sha256"].items():
        if file_hash(run / name) != expected:
            raise ValueError("Completed reproduction artifact changed")


def fit_embedding(run, study):
    run = Path(run)
    sources = checked_run(run, study)
    stage_start(run, "fit-embedding", run / "head/frozen.json", run / "control/frozen.json")
    training = sources["training"]
    fitted = benefit.freeze(
        run / "head",
        sources["baseline"],
        sources["root"] / "policies-frozen.json",
        training / "plan.json",
        training / "inputs.json",
        training / "grades-train.json",
    )
    write(run / "head/frozen.json", fitted)
    control = benefit_control.freeze(run / "control", run / "head", sources["baseline"])
    write(run / "control/frozen.json", control)
    return stage_done(run, "fit-embedding", ("head/frozen.json", "control/frozen.json"))


def fit_native(run, study, device, *, local_files_only=False):
    run = Path(run)
    sources = checked_run(run, study)
    stage_start(
        run,
        "fit-native",
        run / "native/frozen.json",
        run / "native/training-start.json",
        run / "native/base",
        run / "native/lora",
    )
    training = sources["training"]
    benefit_native.train(
        run / "native",
        run / "head",
        sources["baseline"],
        sources["root"] / "policies-frozen.json",
        training / "plan.json",
        training / "inputs.json",
        training / "grades-train.json",
        device=device,
        local_files_only=local_files_only,
    )
    benefit_native.checked_frozen(run / "native", run / "head")
    return stage_done(run, "fit-native", ("native/frozen.json",))


def measured(rows, scores, selections, overhead, rate):
    result = {}
    for target in benefit.TARGETS:
        policy = selections[str(target)]
        mode = policy["mode"]
        routes = (np.asarray(scores) >= policy["cutoff"]).tolist() if mode == "threshold" else mode
        result[str(target)] = {
            "mode": mode,
            "cutoff": policy["cutoff"],
            "metrics": solver_report.summarize(
                rows,
                routes,
                router_credits=overhead if mode == "threshold" else 0,
                credit_usd=rate,
            ),
        }
    return result


def evaluate(run, study):
    run = Path(run)
    fresh(run / "evaluation.json", run / "evaluation.md")
    sources = checked_run(run, study)
    completed_stage(run, "fit-embedding")
    completed_stage(run, "fit-native")
    head = benefit._checked_frozen(run / "head")
    control, _ = benefit_control.checked_frozen(run / "control", run / "head", sources["baseline"])
    _, native = benefit_native.checked_frozen(run / "native", run / "head")
    # Read published outcomes only after validating every reproduced model/cutoff.
    plan = head["plan"]
    ids = [row["id"] for row in plan["populations"]["test"]]
    grades_path = sources["root"] / "grades-test.json"
    grades = solver_policies._grade_rows(grades_path, "test", plan["original_solver_plan_sha256"], ids)
    if (
        grades["policies_frozen_sha256"] != plan["original_frozen_sha256"]
        or solver_policies._grading_provenance(grades) != head["grading_provenance"]
        or native["grading_provenance"] != head["grading_provenance"]
    ):
        raise ValueError("Published solver outcomes or grading protocol differ")
    credits, credits_sha = solver_policies._credits(
        sources["root"] / "embedding-test/router-credits.json",
        "test",
        plan["original_solver_plan_sha256"],
        ids,
        {row["id"]: row["request_sha256"] for row in plan["populations"]["test"]},
    )
    rows, rate = grades["rows"], head["credit_usd"]
    policies = {
        variant: measured(rows, model["scores"]["test"], model["policies"], None, rate)
        for variant, model in native["models"].items()
    }
    policies["embedding"] = measured(rows, head["scores"]["test"], head["selections"], credits, rate)
    policies["competition"] = measured(rows, control["scores"]["test"], control["selections"], 0, rate)
    result = {
        "format": "sie-benefit-public-regression-report-v1",
        "evaluation_mode": MODE,
        "n": len(ids),
        "ids": ids,
        "primary_target_pp": 1,
        "training_rows": head["training_rows"],
        "native_selected_epoch": native["selected_epoch"],
        "test_grades_sha256": file_hash(grades_path),
        "embedding_test_credits_sha256": credits_sha,
        "reproduction_sha256": file_hash(run / "reproduction.json"),
        "frozen_files_sha256": {name: file_hash(run / name / "frozen.json") for name in PLAN_NAMES},
        "baselines": {
            name: solver_report.summarize(rows, name, router_credits=0, credit_usd=rate)
            for name in ("always_strong", "always_cheap")
        },
        "policies": policies,
        "cost_scope": "Recorded selected-answer charges and embedding API overhead. Native hosting and common application CPU remain unpriced. Acquisition and training costs are separate.",
        "interpretation": "Public regression reproduction on already-published problems and solver outcomes. No new unseen-test claim or unopened-test receipt. All four calibration targets are applied unchanged; none guarantees the same loss on test or new requests.",
    }
    write(run / "evaluation.json", result)
    rendered = "Evaluation mode: `public_regression_reproduction`.\n\n" + benefit_summary.markdown(result)
    (run / "evaluation.md").write_text(rendered)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "fit-embedding", "fit-native", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--study", type=Path, default=Path("run-output/solver-study"))
        child.add_argument("--output" if command == "prepare" else "--run", type=Path, required=True)
        if command == "fit-native":
            child.add_argument("--device", choices=("mps", "cuda"), required=True)
            child.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.study, args.output)
    elif args.command == "fit-embedding":
        result = fit_embedding(args.run, args.study)
    elif args.command == "fit-native":
        result = fit_native(args.run, args.study, args.device, local_files_only=args.local_files_only)
    else:
        result = evaluate(args.run, args.study)
        print("Evaluation mode: public_regression_reproduction\n")
        print(benefit_summary.markdown(result))
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
