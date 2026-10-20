"""Train from the public recordings and reproduce the cost/quality comparison.

Public test answers are already available to students. These commands call the
same training functions as the original study, but never create a receipt that
claims the published test is newly unseen.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import benefit_outcomes as embedding
import benefit_outcomes_native as native
import evidence
import generalization_report
import outcome_summary
import solver_policies
import solver_report
import teacher_outcomes
from benefit_replay import measured
from data import file_hash, write

MODE = "public_regression_reproduction"
POPULATIONS = ("original", "expanded")


def read(path):
    return json.loads(Path(path).read_text())


def reference(study):
    study = Path(study)
    evidence.verify(study, manifest_sha256=evidence.SOLVER_MANIFEST_SHA256)
    return study / "outcomes"


def prepare(study, output):
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if output.resolve() == study.resolve() or study.resolve() in output.resolve().parents:
        raise ValueError("Keep your training outputs outside the downloaded reference directory")
    source = reference(study)
    plans = {}
    for population in POPULATIONS:
        head_path = source / "heads" / population
        native_path = source / "native" / population
        embedding.checked_frozen(head_path)
        native.checked_frozen(native_path, head_path)
        for design in ("heads", "native"):
            relative = Path(design) / population / "plan.json"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
            plans[str(relative)] = file_hash(destination)
        # Native training uses the published head only to share the exact
        # training targets/cost statistics. Students can fit their own head
        # later, independently of the native training stage.
        for name in ("plan.json", "selected-model.json", "frozen.json"):
            destination = output / "reference-heads" / population / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(head_path / name, destination)
    if (source / "teacher/frozen.json").exists():
        teacher_outcomes.checked_frozen(source / "teacher")
        for name in ("plan.json", "teacher-labels.json", "teacher-audit.json"):
            relative = Path("teacher") / name
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
            plans[str(relative)] = file_hash(destination)
    receipt = {
        "format": "sie-outcome-public-reproduction-v1",
        "evaluation_mode": MODE,
        "reference_manifest_sha256": file_hash(study / "manifest.json"),
        "historical_plans_sha256": plans,
        "wrapper_source_sha256": file_hash(Path(__file__)),
        "interpretation": "Unchanged historical plans describe the original study. This reproduction creates no new unseen-test authorization. Native training and embedding training can run independently using the same saved paired labels.",
    }
    write(output / "reproduction.json", receipt)
    return receipt


def checked_run(run, study):
    run, study = Path(run), Path(study)
    source = reference(study)
    receipt = read(run / "reproduction.json")
    if (
        receipt.get("format") != "sie-outcome-public-reproduction-v1"
        or receipt.get("evaluation_mode") != MODE
        or receipt.get("reference_manifest_sha256") != file_hash(study / "manifest.json")
        or receipt.get("wrapper_source_sha256") != file_hash(Path(__file__))
    ):
        raise ValueError("The public reproduction or its reference changed")
    for relative, expected in receipt["historical_plans_sha256"].items():
        if file_hash(run / relative) != expected or file_hash(source / relative) != expected:
            raise ValueError("A copied historical training plan changed")
    for root in ("heads", "native", "reference-heads"):
        for population in POPULATIONS:
            if (run / root / population / "test-access.json").exists():
                raise ValueError("Public reproduction must not create an unseen-test receipt")
    if (run / "teacher/test-access.json").exists():
        raise ValueError("Public reproduction must not create an unseen-test receipt")
    return source


def fit_embedding(run, study, population):
    run = Path(run)
    source = checked_run(run, study)
    if population not in POPULATIONS:
        raise ValueError("Choose the original or expanded training population")
    training, target = source / "training" / population, run / "heads" / population
    result = embedding.fit(
        target,
        source / "baseline",
        training / "plan.json",
        training / "inputs.json",
        training / "training-rows.json",
        training / "grades-train.json",
        source / "embeddings" / population,
        training / "cost-proxy.json",
        source / "cloud-prices.json",
        progress=print,
    )
    write(target / "selected-model.json", result)
    frozen = embedding.freeze(target, source / "baseline", source / "calibration/policies-frozen.json")
    write(target / "frozen.json", frozen)
    return {
        "method": "embedding",
        "training_examples": len(result["head"]["training_ids"]),
        "selected_penalty": result["head"]["penalty"],
    }


def fit_native(run, study, population, *, local_files_only=False):
    run = Path(run)
    source = checked_run(run, study)
    if population not in POPULATIONS:
        raise ValueError("Choose the original or expanded training population")
    training, target = source / "training" / population, run / "native" / population
    reference_head = run / "reference-heads" / population
    for name in ("plan.json", "selected-model.json", "frozen.json"):
        if file_hash(reference_head / name) != file_hash(source / "heads" / population / name):
            raise ValueError("Native training's shared reference head changed")
    result = native.train(
        target,
        reference_head,
        source / "baseline",
        training / "plan.json",
        training / "inputs.json",
        training / "training-rows.json",
        training / "grades-train.json",
        training / "cost-proxy.json",
        source / "cloud-prices.json",
        device="cuda",
        local_files_only=local_files_only,
    )
    frozen = native.freeze(
        target, reference_head, source / "baseline", source / "calibration/policies-frozen.json", device="cuda"
    )
    write(target / "frozen.json", frozen)
    return {
        "method": "native",
        "training_examples": len(result["training_targets"]),
        "selected_epoch": result["selected_epoch"],
    }


def fit_teacher(run, study):
    """Refit both label sources on their recorded, identical training subset."""
    run = Path(run)
    source = checked_run(run, study)
    target, training = run / "teacher", source / "training/expanded"
    if not (target / "teacher-labels.json").exists():
        raise ValueError("This reference bundle does not include the teacher-label comparison")
    result = teacher_outcomes.fit(
        target,
        training,
        source / "embeddings/expanded",
        source / "cloud-prices.json",
        progress=print,
    )
    write(target / "selected-model.json", result)
    frozen = teacher_outcomes.freeze(target, source / "baseline", source / "calibration/policies-frozen.json")
    write(target / "frozen.json", frozen)
    return {
        "method": "teacher",
        "training_examples": result["training_rows"],
        "teacher_penalty": result["head"]["penalty"],
        "benchmark_penalty": result["matched_benchmark_control"]["penalty"],
        "evaluation_mode": MODE,
    }


def _evaluation_inputs(source, plan, provenance):
    grades_path = source / "grades-test.json"
    ids = [row["id"] for row in plan["populations"]["test"]]
    grades = solver_policies._grade_rows(grades_path, "test", plan["original_solver_plan_sha256"], ids)
    if (
        grades["policies_frozen_sha256"] != plan["original_frozen_sha256"]
        or embedding.grading_provenance(grades) != provenance
    ):
        raise ValueError("Public evaluation answers or grading protocol differ from training")
    return grades, grades_path, ids


def evaluate(run, study, population, method, output):
    run, output = Path(run), Path(output)
    if method not in ("native", "embedding", "teacher"):
        raise ValueError("Choose native, embedding or teacher")
    if output.exists():
        raise FileExistsError(output)
    source = checked_run(run, study)
    if population not in POPULATIONS:
        raise ValueError("Choose the original or expanded training population")
    target = run / ("heads" if method == "embedding" else "native") / population
    if method == "teacher":
        if population != "expanded":
            raise ValueError("The teacher comparison uses the expanded training population")
        target = run / "teacher"
        frozen, selected = teacher_outcomes.checked_frozen(target)
        plan = selected["plan"]["reference_plan"]
        grades, grades_path, ids = _evaluation_inputs(source, plan, frozen["grading_provenance"])
        credits, _ = embedding.test_credits(plan, source / "router-credits.json")
        control = frozen["matched_benchmark_control"]
        variants = {
            "teacher": measured(
                grades["rows"], frozen["scores"]["test"], frozen["selections"], credits, frozen["credit_usd"]
            ),
            "matched_benchmark_control": measured(
                grades["rows"], control["scores"]["test"], control["selections"], credits, frozen["credit_usd"]
            ),
        }
    elif method == "embedding":
        frozen, selected = embedding.checked_frozen(target)
        plan = selected["plan"]
        grades, grades_path, ids = _evaluation_inputs(source, plan, frozen["grading_provenance"])
        credits, _ = embedding.test_credits(plan, source / "router-credits.json")
        policies = measured(
            grades["rows"], frozen["scores"]["test"], frozen["selections"], credits, frozen["credit_usd"]
        )
        variants = {"embedding": policies}
    else:
        native_plan, _trained, frozen = native.checked_frozen(target, run / "reference-heads" / population)
        plan = native_plan["head_plan"]
        grades, grades_path, ids = _evaluation_inputs(source, plan, frozen["grading_provenance"])
        variants = {
            variant: measured(grades["rows"], model["scores"]["test"], model["policies"], None, frozen["credit_usd"])
            for variant, model in frozen["models"].items()
        }
    result = {
        "format": "sie-outcome-reproduced-evaluation-v1",
        "evaluation_mode": MODE,
        "method": method,
        "n": len(ids),
        "ids": ids,
        "training_examples": selected["training_rows"] if method == "teacher" else len(plan["populations"]["train"]),
        "frozen_sha256": file_hash(target / "frozen.json"),
        "test_grades_sha256": file_hash(grades_path),
        "baselines": {
            name: solver_report.summarize(grades["rows"], name, router_credits=0, credit_usd=frozen["credit_usd"])
            for name in ("always_strong", "always_cheap")
        },
        "variants": variants,
        "primary_target_pp": 1,
        "interpretation": "A reproduction on public test answers. Cutoffs still use calibration only; these results are not a new unseen test. Native threshold hosting remains unpriced.",
    }
    write(output, result)
    return result


def replay(study, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = reference(study)
    heads, natives = [], []
    for population in POPULATIONS:
        head_path = source / "heads" / population
        head = embedding.report(head_path, source / "grades-test.json", source / "router-credits.json")
        adapted = native.report(source / "native" / population, head_path, source / "grades-test.json")
        if head != read(head_path / "report.json") or adapted != read(source / "native" / population / "report.json"):
            raise ValueError("Recomputed public report differs from the recorded experiment")
        heads.append(head)
        natives.append(adapted)
    teacher = None
    if (source / "teacher/report.json").exists():
        # Optional label-source study is kept separate from the four main fits.
        teacher = teacher_outcomes.report(
            source / "teacher", source / "grades-test.json", source / "router-credits.json"
        )
        if teacher != read(source / "teacher/report.json"):
            raise ValueError("Recomputed teacher-label report differs")
    comparison = outcome_summary.combine(heads, natives, teacher)
    recorded = read(source / "comparison/comparison.json")
    if comparison != {key: value for key, value in recorded.items() if key != "source_reports_sha256"}:
        raise ValueError("Recomputed comparison differs from the public record")
    write(output / "comparison.json", comparison)
    rendered = outcome_summary.markdown(comparison)
    (output / "comparison.md").write_text(rendered)
    transfer = Path(study) / "transfer"
    if (transfer / "report.json").exists():
        transferred = generalization_report.report(transfer)
        if transferred != read(transfer / "report.json"):
            raise ValueError("Recomputed arithmetic transfer report differs from the public record")
        write(output / "arithmetic-transfer.json", transferred)
        transfer_text = generalization_report.markdown(transferred)
        (output / "arithmetic-transfer.md").write_text(transfer_text)
        rendered += "\n" + transfer_text
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "fit-embedding", "fit-native", "fit-teacher", "evaluate", "replay"):
        child = sub.add_parser(command)
        child.add_argument("--study", type=Path, default=Path("evidence"))
        if command in ("prepare", "replay"):
            child.add_argument("--output", type=Path, required=True)
        else:
            child.add_argument("--run", type=Path, default=Path("run-output/my-router"))
            child.add_argument("--population", choices=POPULATIONS, default="expanded")
            if command == "fit-native":
                child.add_argument("--local-files-only", action="store_true")
            if command == "evaluate":
                child.add_argument("--method", choices=("native", "embedding", "teacher"), required=True)
                child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.study, args.output)
    elif args.command == "replay":
        print(replay(args.study, args.output))
        return
    elif args.command == "fit-native":
        result = fit_native(args.run, args.study, args.population, local_files_only=args.local_files_only)
    elif args.command == "fit-embedding":
        result = fit_embedding(args.run, args.study, args.population)
    elif args.command == "fit-teacher":
        if args.population != "expanded":
            parser.error("The teacher comparison uses --population expanded")
        result = fit_teacher(args.run, args.study)
    else:
        result = evaluate(args.run, args.study, args.population, args.method, args.output)
        result = {
            "method": args.method,
            "training_examples": result["training_examples"],
            "primary": {key: value["1"]["metrics"] for key, value in result["variants"].items()},
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
