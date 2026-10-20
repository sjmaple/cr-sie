"""Reproduce the public arithmetic study and train a router from its saved labels.

This is an exposed reproduction: the published test results are already known.
The original acquisition, model freeze and test-access records stay unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import arithmetic_controls
import arithmetic_optimizer
import arithmetic_study as study_code
import benefit_outcomes as outcomes
import benefit_outcomes_native as native
import evidence
import solver_report
from data import digest, file_hash, write
from plot_arithmetic import LABELS, ORDER, label_for
from serving import vectors

POPULATIONS = {"original": 500, "expanded": 2000}
MODE = "Public reproduction using previously exposed outcomes; this is not a new blind test."


def read(path):
    return json.loads(Path(path).read_text())


def reference(study):
    study = Path(study)
    evidence.verify(study, manifest_sha256=evidence.ARITHMETIC_MANIFEST_SHA256)
    root = study / "arithmetic"
    study_code.checked_source(root / "source")
    return root


def prepare(study, output):
    root, output = reference(study), Path(output)
    plan = study_code.prepare(
        root / "source", output, root / "reference/embedding", root / "reference/native"
    )
    records = {}
    for split in ("train", "calibration"):
        path = root / "study" / f"grades-{split}.json"
        record = read(path)
        original = record["study_plan_sha256"]
        record.update(
            study_plan_sha256=file_hash(output / "plan.json"),
            reproduction={
                "mode": MODE,
                "source_grades_sha256": file_hash(path),
                "source_study_plan_sha256": original,
                "grades_recomputed": False,
            },
        )
        write(output / path.name, record)
        records[path.name] = file_hash(output / path.name)
    provenance = {
        "format": "sie-arithmetic-public-reproduction-v1",
        "mode": MODE,
        "evidence_manifest_sha256": file_hash(Path(study) / "manifest.json"),
        "study_plan_sha256": file_hash(output / "plan.json"),
        "copied_grades_sha256": records,
        "solver_settings_sha256": digest(study_code.solver.SETTINGS),
        "training_rows": len(plan["training_ids"]),
        "policy_files_sha256": {},
    }
    write(output / "reproduction.json", provenance)
    return provenance


def checked_run(run, study):
    root, run = reference(study), Path(run)
    saved = read(run / "reproduction.json")
    if (
        saved.get("format") != "sie-arithmetic-public-reproduction-v1"
        or saved.get("evidence_manifest_sha256") != file_hash(Path(study) / "manifest.json")
        or saved.get("study_plan_sha256") != file_hash(run / "plan.json")
        or saved.get("solver_settings_sha256") != digest(study_code.solver.SETTINGS)
        or set(saved.get("copied_grades_sha256", {})) != {"grades-train.json", "grades-calibration.json"}
        or any(file_hash(run / name) != sha for name, sha in saved["copied_grades_sha256"].items())
    ):
        raise ValueError("Prepared reproduction inputs or model protocol changed")
    study_code.checked_plan(root / "source", run)
    return root


def score_head(matrix, head):
    probabilities = np.column_stack([outcomes.probability(matrix, head["models"][arm]) for arm in outcomes.ARMS])
    return outcomes.normalized_scores(probabilities, head["conditional_cost_statistics"])[0].tolist()


def save_policy(root, run, name, scores, *, design, model_path, training_rows):
    """Choose cutoffs from calibration, without opening published test outcomes."""
    _, inputs, _ = study_code.checked_plan(root / "source", run)
    calibration = read(run / "grades-calibration.json")["rows"]
    ids = {split: [row["id"] for row in inputs[split]] for split in ("calibration", "test")}
    if [row["id"] for row in calibration] != ids["calibration"]:
        raise ValueError("Calibration outcomes and question scores have different order")
    if any(len(scores[split]) != len(ids[split]) or not np.isfinite(scores[split]).all() for split in ids):
        raise ValueError("Model scores do not cover the saved question order")
    charges = {
        split: study_code.overhead(root / "embeddings/fresh", ids[split]) if design == "embedding" else None
        for split in ids
    }
    result = {
        "format": "sie-arithmetic-public-policy-v1",
        "mode": MODE,
        "method": name,
        "design": design,
        "training_rows": training_rows,
        "study_plan_sha256": file_hash(run / "plan.json"),
        "model_path": str(model_path.relative_to(run)),
        "model_sha256": file_hash(model_path),
        "calibration_grades_sha256": file_hash(run / "grades-calibration.json"),
        "ids": ids,
        "scores": scores,
        "router_credits": charges,
        "policies": {
            str(target): solver_report.select_cutoff(
                calibration, scores["calibration"], target, router_credits=charges["calibration"], credit_usd=1e-5
            )
            for target in study_code.TARGETS
        },
    }
    policy_path = run / f"policy-{name}.json"
    write(policy_path, result)
    provenance = read(run / "reproduction.json")
    provenance["policy_files_sha256"][policy_path.name] = file_hash(policy_path)
    temporary = run / ".reproduction.tmp"
    temporary.write_text(json.dumps(provenance, indent=2) + "\n")
    temporary.replace(run / "reproduction.json")
    return result


def fit_embedding(run, study, population):
    run = Path(run)
    root = checked_run(run, study)
    size = POPULATIONS[population]
    model = study_code.fit_embedding(root / "source", run, root / "embeddings/train", size)
    _, inputs, _ = study_code.checked_plan(root / "source", run)
    scores = {
        split: score_head(vectors(inputs[split], root / "embeddings/fresh"), model["head"])
        for split in ("calibration", "test")
    }
    save_policy(
        root, run, f"embedding_{size}", scores,
        design="embedding", model_path=run / f"embedding-{size}.json", training_rows=size,
    )
    return {"method": "embedding", "training_rows": size, "selected_penalty": model["head"]["penalty"]}


def fit_native(run, study, population, *, local_files_only=False):
    run = Path(run)
    root = checked_run(run, study)
    size = POPULATIONS[population]
    trained = study_code.train_native(root / "source", run, size, local_files_only=local_files_only)
    study_code.checked_native(root / "source", run, size)
    _, inputs, _ = study_code.checked_plan(root / "source", run)
    checkpoint = run / f"native-{size}/lora"
    manifest = trained["checkpoints"]["lora"]["files_sha256"]
    if any(file_hash(checkpoint / name) != expected for name, expected in manifest.items()):
        raise ValueError("Native checkpoint differs from its saved export")
    scores = {
        split: native.score_checkpoint(checkpoint, inputs[split], "cuda", trained["conditional_cost_statistics"])
        for split in ("calibration", "test")
    }
    save_policy(
        root, run, f"lora_{size}", scores,
        design="native", model_path=run / f"native-{size}/trained.json", training_rows=size,
    )
    return {"method": "native", "training_rows": size, "selected_epoch": trained["selected_epoch"]}


def fit_teacher(run, study):
    """Refit the same two heads from the published, matched training targets."""
    run = Path(run)
    root = checked_run(run, study)
    teacher = read(root / "study/teacher-models.json")
    labels_path = root / "study/teacher-labels.json"
    labels = read(labels_path)
    if teacher["teacher_labels_sha256"] != file_hash(labels_path):
        raise ValueError("Saved teacher labels differ from their published provenance")
    plan, inputs, metadata = study_code.checked_plan(root / "source", run)
    source_heads = teacher["heads"]
    names = study_code.OPTIONAL_METHODS
    teacher_rows = {row["id"]: row for row in labels["rows"] if row["usable"]}
    selected = set(teacher_rows)
    positions = [i for i, row in enumerate(inputs["train"]) if row["id"] in selected]
    ids = [inputs["train"][i]["id"] for i in positions]
    if (
        set(source_heads) != set(names)
        or set(ids) != selected
        or any(head["training_ids"] != ids for head in source_heads.values())
    ):
        raise ValueError("Teacher and reference targets must cover the same ordered training questions")
    training_inputs = [inputs["train"][i] for i in positions]
    matrix = vectors(training_inputs, root / "embeddings/train")
    groups = [metadata["train"][i]["family"] for i in positions]
    folds = [plan["training_folds"][i] for i in positions]
    reference_rows = {row["id"]: row for row in read(run / "grades-train.json")["rows"]}
    costs = [[reference_rows[identity][f"{arm}_credits"] for arm in outcomes.ARMS] for identity in ids]
    heads = {}
    for name in names:
        original = source_heads[name]
        targets = [
            [
                int(teacher_rows[identity]["grades"][arm]["correct"])
                if name == names[0] else int(reference_rows[identity][f"{arm}_correct"])
                for arm in outcomes.ARMS
            ]
            for identity in ids
        ]
        if original["folds"] != folds or original["targets"] != targets or original["training_cost_proxies"] != costs:
            raise ValueError("Saved teacher targets or folds differ from the prepared source")
        rows = [
            {"id": identity, **{f"{arm}_correct": bool(value) for arm, value in zip(outcomes.ARMS, target, strict=True)}}
            for identity, target in zip(ids, targets, strict=True)
        ]
        heads[name] = arithmetic_optimizer.fit_head(
            matrix, rows, groups, costs, fold_assignments=folds, progress=print
        )
    result = {
        "heads": heads,
        "summary": teacher["summary"],
        "study_plan_sha256": file_hash(run / "plan.json"),
        "training_grades_sha256": file_hash(run / "grades-train.json"),
        "source_teacher_models_sha256": file_hash(root / "study/teacher-models.json"),
        "source_teacher_labels_sha256": file_hash(labels_path),
        "mode": MODE,
    }
    write(run / "teacher-models.json", result)
    matrices = {split: vectors(inputs[split], root / "embeddings/fresh") for split in ("calibration", "test")}
    for name, head in heads.items():
        scores = {split: score_head(features, head) for split, features in matrices.items()}
        save_policy(
            root, run, name, scores,
            design="embedding", model_path=run / "teacher-models.json", training_rows=len(ids),
        )
    return {"method": "teacher", "training_rows": len(ids), "matched_reference_rows": len(ids)}


def method_report(model, rows):
    if [row["id"] for row in rows] != model["ids"]["test"]:
        raise ValueError("Published outcomes and frozen decisions have different question order")
    scores = np.asarray(model["scores"]["test"])
    policies = {}
    for target, policy in model["policies"].items():
        mode = policy["mode"]
        decisions = (scores >= policy["cutoff"]).tolist() if mode == "threshold" else mode
        charges = model["router_credits"]["test"] if mode == "threshold" else 0
        policies[target] = {
            "mode": mode, "cutoff": policy["cutoff"],
            "metrics": solver_report.summarize(rows, decisions, router_credits=charges),
        }
    return {
        "label": LABELS.get(model.get("method"), model.get("method", "")),
        "design": model["design"], "training_rows": model["training_rows"], "policies": policies,
    }


def evaluate(run, study, population, method, output):
    run, output = Path(run), Path(output)
    root = checked_run(run, study)
    size = POPULATIONS[population]
    names = study_code.OPTIONAL_METHODS if method == "teacher" else [f"{'lora' if method == 'native' else 'embedding'}_{size}"]
    models = {}
    provenance = read(run / "reproduction.json")
    for name in names:
        policy_path = run / f"policy-{name}.json"
        model = read(policy_path)
        model_path = Path(model["model_path"])
        if (
            model_path.is_absolute() or ".." in model_path.parts
            or provenance["policy_files_sha256"].get(policy_path.name) != file_hash(policy_path)
            or model.get("method") != name
            or model.get("design") != ("native" if method == "native" else "embedding")
            or (method != "teacher" and model.get("training_rows") != size)
            or model["study_plan_sha256"] != file_hash(run / "plan.json")
            or model["model_sha256"] != file_hash(run / model_path)
            or model["calibration_grades_sha256"] != file_hash(run / "grades-calibration.json")
        ):
            raise ValueError("Trained model or calibration policy changed after selection")
        models[name] = model
    # All cutoffs above are already saved. These are public, previously exposed outcomes.
    grades = read(root / "study/grades-test.json")
    result = {
        "format": "sie-arithmetic-public-evaluation-v1", "mode": MODE, "n": len(grades["rows"]),
        "test_grades_sha256": file_hash(root / "study/grades-test.json"),
        "methods": {name: method_report(model, grades["rows"]) for name, model in models.items()},
    }
    write(output, result)
    return result


def markdown(report, target="1"):
    lines = [
        "# Routing cost and answer quality", "", MODE, "",
        f"{report['n']:,} test questions; cutoffs selected for a {target}-point calibration gap.", "",
        "| Method | Answer accuracy | Quality gap | Cost per 10,000 | Savings |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    rows = [(name, metric) for name, metric in report["baselines"].items()]
    rows += [(name, report["methods"][name]["policies"][target]["metrics"]) for name in ORDER if name in report["methods"]]
    for name, metric in rows:
        known = metric.get("net_total_cost_usd_per1000") is not None
        cost = 10 * metric["net_total_cost_usd_per1000" if known else "total_solver_cost_usd_per1000"]
        saving = metric["net_savings_pct" if known else "gross_solver_savings_pct"]
        lines.append(
            f"| {label_for(report, name)} | {100 * metric['answer_accuracy']:.2f}% | "
            f"{metric['gap_vs_strong_pp']:.2f} points | ${cost:.2f}{'' if known else ' + hosting'} | "
            f"{saving:.2f}%{'' if known else ' before hosting'} |"
        )
    lines += ["", report["cost_scope"], "", "The JSON retains every declared cutoff target and paired 95% uncertainty interval."]
    return "\n".join(lines) + "\n"


def replay_cost_controls(root, grades):
    directory = root / "study-cost-controls"
    frozen = arithmetic_controls.checked_frozen(root / "source", root / "study", directory)
    recorded, access = (read(directory / name) for name in ("report.json", "test-access.json"))
    if (
        recorded["frozen_sha256"] != file_hash(directory / "frozen.json")
        or recorded["core_frozen_sha256"] != frozen["core_frozen_sha256"]
        or recorded["test_grades_sha256"] != file_hash(root / "study/grades-test.json")
        or access["frozen_sha256"] != recorded["frozen_sha256"]
        or access["core_test_access_sha256"] != file_hash(root / "study/test-access.json")
        or grades["test_access_sha256"] != access["core_test_access_sha256"]
        or set(recorded["methods"]) != set(frozen["methods"])
    ):
        raise ValueError("Cost controls differ from their saved decisions or shared test outcomes")
    recomputed = copy.deepcopy(recorded)
    for name, model in frozen["methods"].items():
        selected = method_report(
            model | {"ids": frozen["ids"], "method": name, "router_credits": {"test": None}},
            grades["rows"],
        )
        recomputed["methods"][name]["policies"] = selected["policies"]
    if digest(recomputed) != digest(recorded):
        raise ValueError("Recomputed cost-control metrics differ from the published result")
    return recomputed


def replay(study, output):
    root, output = reference(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    recorded, frozen, grades = (read(root / "study" / name) for name in ("report.json", "frozen.json", "grades-test.json"))
    if (
        recorded["frozen_sha256"] != file_hash(root / "study/frozen.json")
        or recorded["test_grades_sha256"] != file_hash(root / "study/grades-test.json")
    ):
        raise ValueError("Recorded comparison and source decisions differ")
    recomputed = copy.deepcopy(recorded)
    recomputed["baselines"] = {
        mode: solver_report.summarize(grades["rows"], mode, router_credits=0) for mode in solver_report.BASELINES
    }
    for name, model in frozen["methods"].items():
        selected = method_report(model | {"ids": frozen["ids"], "method": name}, grades["rows"])
        recomputed["methods"][name]["policies"] = selected["policies"]
    if digest(recomputed) != digest(recorded):
        raise ValueError("Recomputed cost or quality metrics differ from the published result")
    controls = replay_cost_controls(root, grades)
    write(output / "comparison.json", recomputed)
    write(output / "cost-controls.json", controls)
    presentation = copy.deepcopy(recomputed)
    presentation["methods"].update(controls["methods"])
    (output / "comparison.md").write_text(
        markdown(presentation)
        + "\nUnchanged-Laya controls use the same cost estimates as each LoRA run, isolating changes to model weights. "
        "Their separate pre-test freeze and metrics are retained in `cost-controls.json`.\n"
    )
    write(output / "replay.json", {
        "mode": MODE, "metrics_match": True,
        "source_report_sha256": file_hash(root / "study/report.json"),
        "source_cost_controls_sha256": file_hash(root / "study-cost-controls/report.json"),
    })
    return {"n": recomputed["n"], "metrics_match": True, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("replay", "prepare", "fit-native", "fit-embedding", "fit-teacher", "evaluate"):
        child = commands.add_parser(name)
        child.add_argument("--study", type=Path, default=Path("evidence"))
        if name not in ("prepare", "replay"):
            child.add_argument("--run", type=Path, required=True)
        if name in ("fit-native", "fit-embedding", "evaluate"):
            child.add_argument("--population", choices=POPULATIONS, default="expanded")
        if name in ("prepare", "replay", "evaluate"):
            child.add_argument("--output", type=Path, required=True)
        if name == "fit-native":
            child.add_argument("--local-files-only", action="store_true")
        if name == "evaluate":
            child.add_argument("--method", choices=("embedding", "native", "teacher"), default="embedding")
    args = parser.parse_args(argv)
    if args.command in ("replay", "prepare"):
        result = (replay if args.command == "replay" else prepare)(args.study, args.output)
    elif args.command == "fit-embedding":
        result = fit_embedding(args.run, args.study, args.population)
    elif args.command == "fit-native":
        result = fit_native(args.run, args.study, args.population, local_files_only=args.local_files_only)
    elif args.command == "fit-teacher":
        result = fit_teacher(args.run, args.study)
    else:
        result = evaluate(args.run, args.study, args.population, args.method, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
