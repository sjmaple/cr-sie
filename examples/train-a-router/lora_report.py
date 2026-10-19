"""Recompute the LoRA follow-up without treating repeated seeds as extra test cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import LABELS
from evidence import verify as verify_evidence
from finetune import BASE, LORA_RECIPES, REVISION, auc, cutoff, lora_targets, metrics, sha, write
from lora_study import FILES, SEEDS


def read(path):
    return json.loads(path.read_text())


def recipe_means(values):
    # Match the driver's Python 3.12 compensated sum and declaration-order ties.
    return {name: sum(scores) / len(SEEDS) for name, scores in values.items()}


def development_labels(path):
    return [LABELS.index(json.loads(line)["expected"]["difficulty"]) for line in path.read_text().splitlines()]


def check_study_run(directory, receipt, study_plan, recipe, seed, size, validation_labels, calibration_labels):
    frozen, plan = read(directory / "frozen.json"), read(directory / "plan.json")
    if sha(directory / "frozen.json") != receipt["frozen_sha256"] or frozen["plan_sha256"] != sha(
        directory / "plan.json"
    ):
        raise ValueError("Study run changed after its stage was frozen")
    if set(frozen["selected"]) != {"lora"}:
        raise ValueError("Expected one declared LoRA checkpoint")
    saved = frozen["selected"]["lora"]
    if saved != {key: value for key, value in receipt.items() if key != "frozen_sha256"}:
        raise ValueError("Stage receipt differs from its selected checkpoint")
    expected_recipe = study_plan["recipes"][recipe]
    if (
        plan["base"] != study_plan["base"]
        or plan["revision"] != study_plan["revision"]
        or plan["recipe"] != recipe
        or plan["validation_metric"] != "auc"
        or plan["export_dtype"] != study_plan["export_dtype"]
        or plan["variants"] != ["lora"]
        or plan["config"]["seed"] != seed
        or saved["seed"] != seed
        or plan["config"]["epochs"] != study_plan["epochs"]
        or plan["data_sha256"] != study_plan["data_sha256"][size]
        or any(plan["lora"].get(key) != value for key, value in expected_recipe.items())
        or plan["lora"]["alpha"] != 2 * expected_recipe["rank"]
        or plan["lora"]["targets"] != lora_targets(expected_recipe)
    ):
        raise ValueError("Run recipe, seed or development inputs differ from the frozen study")
    history = read(directory / "lora-history.json")
    if [row["epoch"] for row in history] != list(range(study_plan["epochs"] + 1)):
        raise ValueError("Expected every declared epoch in chronological order, including epoch zero")
    for row in history:
        if row["labels"] != validation_labels:
            raise ValueError("Checkpoint selection uses different validation labels")
        if row["auc"] != auc(row["scores"], validation_labels):
            raise ValueError("Recorded validation AUC differs")
    selected = max(history, key=lambda row: row["auc"])
    if (
        selected["epoch"] != saved["epoch"]
        or selected["auc"] != saved["validation_auc"]
        or selected["cross_entropy"] != saved["validation_cross_entropy"]
        or selected["accuracy"] != saved["validation_accuracy_at_zero"]
    ):
        raise ValueError("Checkpoint differs from validation selection")
    calibration = read(directory / "lora-calibration.json")
    if calibration["labels"] != calibration_labels or saved["threshold"] != cutoff(
        calibration["scores"], calibration_labels
    ):
        raise ValueError("Frozen cutoff differs from separate calibration")
    calculated = metrics(np.asarray(calibration["scores"]), np.asarray(calibration_labels), saved["threshold"])
    if calculated["predictions"] != calibration["predictions"]:
        raise ValueError("Calibration decisions differ from frozen cutoff")
    for name, limit in (("lora-merge-parity.json", 1e-3), ("lora-export-parity.json", 0.03)):
        error = read(directory / name)["max_margin_error"]
        if not np.isfinite(error) or not 0 <= error <= limit:
            raise ValueError("Export or merge failed the declared validation parity check")
    return saved["validation_auc"]


def predictions(directory, rows, test_sha):
    frozen = read(directory / "frozen.json")
    plan = read(directory / "plan.json")
    record = read(directory / "evaluation-test.json")
    if (
        frozen["plan_sha256"] != sha(directory / "plan.json")
        or record["frozen_sha256"] != sha(directory / "frozen.json")
        or record["data_sha256"] != test_sha
        or record["ids"] != [r["id"] for r in rows]
    ):
        raise ValueError(f"Evaluation is not bound to its frozen model and population: {directory}")
    values = record["models"]["lora"]
    labels = np.array([r["label"] for r in rows])
    if values["labels"] != labels.tolist():
        raise ValueError("Evaluation labels differ")
    calculated = metrics(np.array(values["scores"]), labels, frozen["selected"]["lora"]["threshold"])
    if calculated["predictions"] != values["predictions"]:
        raise ValueError("Decisions differ from frozen cutoff")
    return np.asarray(values["predictions"]), {
        key: value for key, value in calculated.items() if key not in ("scores", "labels", "predictions")
    } | {
        "auc": auc(values["scores"], labels),
        "epoch": frozen["selected"]["lora"]["epoch"],
        "recipe": plan.get("recipe", "original"),
        "export_dtype": plan.get("export_dtype", "float16"),
    }


def comparison(rows, before, after):
    labels = np.array([r["label"] for r in rows])
    for values in (before, after):
        values = np.asarray(values)
        if values.ndim not in (1, 2) or values.shape[-1] != len(rows) or not np.isin(values, (0, 1)).all():
            raise ValueError("Expected one binary prediction per problem and seed")
    # Average over seeds first; the resampling unit remains a problem or a paper.
    old = (np.atleast_2d(before) == labels).mean(axis=0)
    new = (np.atleast_2d(after) == labels).mean(axis=0)
    difference = new - old
    draws = np.random.default_rng(89).integers(0, len(rows), (10000, len(rows)))
    case_interval = np.quantile(100 * difference[draws].mean(axis=1), [0.025, 0.975])
    groups = sorted({r["paper_group"] for r in rows})
    masks = [np.array([r["paper_group"] == group for r in rows]) for group in groups]
    totals, sizes = np.array([difference[m].sum() for m in masks]), np.array([m.sum() for m in masks])
    draws = np.random.default_rng(187).integers(0, len(groups), (10000, len(groups)))
    paper_interval = np.quantile(100 * totals[draws].sum(axis=1) / sizes[draws].sum(axis=1), [0.025, 0.975])
    return {
        "accuracy_gain_pp": float(100 * difference.mean()),
        "case_gain_pp_95ci": case_interval.tolist(),
        "paper_gain_pp_95ci": paper_interval.tolist(),
        "error_reduction_fraction": float((new.mean() - old.mean()) / (1 - old.mean())) if old.mean() < 1 else None,
        "method": (
            f"paired bootstrap over {len(rows)} problems / {len(groups)} competition-year groups; "
            "seed-mean correctness first; conditional on these trained seeds, not uncertainty over training runs"
        ),
    }


def check_study(study, development):
    """Validate the complete frozen study without reading test data or predictions."""
    complete, choice, plan = (
        read(study / name) for name in ("all-frozen.json", "recipe-frozen.json", "study-plan.json")
    )
    for key, file in (
        ("study_plan_sha256", "study-plan.json"),
        ("recipe_frozen_sha256", "recipe-frozen.json"),
        ("screening_sha256", "screening.json"),
        ("expanded_sha256", "expanded.json"),
    ):
        if complete[key] != sha(study / file):
            raise ValueError(f"Study changed after freeze: {file}")
    if (
        choice["study_plan_sha256"] != complete["study_plan_sha256"]
        or choice["screening_sha256"] != complete["screening_sha256"]
    ):
        raise ValueError("Recipe choice is not bound to the screening runs")
    if (
        plan["base"] != BASE
        or plan["revision"] != REVISION
        or plan["seeds"] != list(SEEDS)
        or plan["epochs"] != 8
        or plan["export_dtype"] != "float32"
        or list(plan["recipes"]) != list(LORA_RECIPES)
        or plan["recipes"] != LORA_RECIPES
        or set(plan["data_sha256"]) != {"512", "expanded"}
        or any(set(hashes) != set(FILES) for hashes in plan["data_sha256"].values())
        or plan["data_sha256"]["512"] != {name: sha(development / name) for name in FILES}
        or any(plan["data_sha256"]["512"][name] != plan["data_sha256"]["expanded"][name] for name in FILES[1:])
    ):
        raise ValueError("Study declarations or development inputs differ from the reference protocol")
    validation_labels = development_labels(development / "validation.jsonl")
    calibration_labels = development_labels(development / "calibration.jsonl")
    screening, expanded = read(study / "screening.json"), read(study / "expanded.json")
    seed_keys = {str(seed) for seed in SEEDS}
    if (
        set(screening) != set(plan["recipes"])
        or any(set(runs) != seed_keys for runs in screening.values())
        or set(expanded) != seed_keys
    ):
        raise ValueError("Study must retain every declared recipe and seed")
    values = {}
    for recipe in plan["recipes"]:
        values[recipe] = []
        for seed in SEEDS:
            run = study / "screening" / recipe / f"seed-{seed}"
            values[recipe].append(
                check_study_run(
                    run, screening[recipe][str(seed)], plan, recipe, seed, "512", validation_labels, calibration_labels
                )
            )
    means = recipe_means(values)
    if choice["recipe"] != max(means, key=means.get) or means != choice["mean_validation_auc"]:
        raise ValueError("Recipe differs from declared validation selection")
    for seed in SEEDS:
        check_study_run(
            study / "expanded" / f"seed-{seed}",
            expanded[str(seed)],
            plan,
            choice["recipe"],
            seed,
            "expanded",
            validation_labels,
            calibration_labels,
        )
    by_seed = {recipe: dict(zip(map(str, SEEDS), scores)) for recipe, scores in values.items()}
    return plan, choice, means, by_seed


def check_evaluation_lifecycle(study, evidence, study_plan, choice):
    """Verify eager parity and completed evaluations using metadata only."""
    environment = {"TORCH_COMPILE_DISABLE": "1", "TORCHDYNAMO_DISABLE": "1"}
    names = list(
        dict.fromkeys(
            f"{root}/seed-{seed}"
            for root in ("screening/original", f"screening/{choice['recipe']}", "expanded")
            for seed in SEEDS
        )
    )
    expected_names = set(names)
    if any((study / f"{stage}-incomplete.json").exists() for stage in ("runtime-parity", "evaluation")):
        raise ValueError("Incomplete attempts cannot be reported as completed evaluations")
    parity_plan, parity_complete, evaluation_plan, evaluation_complete = (
        read(study / name)
        for name in (
            "runtime-parity-plan.json",
            "runtime-parity-complete.json",
            "evaluation-plan.json",
            "evaluation-complete.json",
        )
    )
    parity_plan_sha = sha(study / "runtime-parity-plan.json")
    if (
        parity_complete["runtime_parity_plan_sha256"] != parity_plan_sha
        or evaluation_plan["runtime_parity_plan_sha256"] != parity_plan_sha
        or evaluation_plan["runtime_parity_complete_sha256"] != sha(study / "runtime-parity-complete.json")
        or evaluation_complete["evaluation_plan_sha256"] != sha(study / "evaluation-plan.json")
    ):
        raise ValueError("Evaluation lifecycle hash chain differs")
    for plan in (parity_plan, evaluation_plan):
        if (
            plan["runtime_environment"] != environment
            or plan["all_frozen_sha256"] != sha(study / "all-frozen.json")
            or plan["study_plan_sha256"] != sha(study / "study-plan.json")
            or plan["training_source_sha256"] != study_plan["training_source_sha256"]
            or set(plan["runs"]) != expected_names
        ):
            raise ValueError("Evaluation runtime, frozen source or declared run set differs")
    if (
        parity_plan["max_margin_error"] != 0.03
        or parity_plan["require_unchanged_frozen_cutoff_decisions"] is not True
        or parity_plan["development_data_sha256"] != study_plan["data_sha256"]["512"]
        or parity_plan["runs"] != evaluation_plan["runs"]
        or parity_plan["evaluation_source_sha256"] != evaluation_plan["evaluation_source_sha256"]
        or parity_plan["device"] != evaluation_plan["device"]
        or parity_plan["timeout_seconds"] != evaluation_plan["timeout_seconds"]
        or set(parity_plan["references"]) != expected_names
        or set(parity_complete["parity_receipts_sha256"]) != expected_names
        or set(evaluation_complete["evaluations_sha256"]) != expected_names
        or evaluation_plan["selected_recipe"] != choice["recipe"]
        or evaluation_plan["test_sha256"] != sha(evidence / "test.json")
        or evaluation_plan["questions_sha256"] != sha(evidence / "questions.json")
        or evaluation_plan["evidence_manifest_sha256"] != sha(evidence / "manifest.json")
    ):
        raise ValueError("Evaluation parity protocol or evidence binding differs")
    maximum_error = 0.0
    required_files = {
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    }
    for name in names:
        directory = study / name
        frozen = read(directory / "frozen.json")
        selected = frozen["selected"]["lora"]
        checkpoint = parity_plan["runs"][name]
        if (
            checkpoint["frozen_sha256"] != sha(directory / "frozen.json")
            or checkpoint["plan_sha256"] != sha(directory / "plan.json")
            or not required_files <= checkpoint["checkpoint_files_sha256"].keys()
            or checkpoint["checkpoint_files_sha256"]["model.safetensors"] != selected["checkpoint_sha256"]
            or parity_plan["references"][name]
            != {file: sha(directory / file) for file in ("lora-history.json", "lora-calibration.json")}
            or parity_complete["parity_receipts_sha256"][name] != sha(directory / "runtime-parity.json")
            or evaluation_complete["evaluations_sha256"][name] != sha(directory / "evaluation-test.json")
        ):
            raise ValueError("Evaluation checkpoint, reference or prediction receipt differs")
        record = read(directory / "runtime-parity.json")
        history = read(directory / "lora-history.json")
        references = {
            "validation": next(row for row in history if row["epoch"] == selected["epoch"]),
            "calibration": read(directory / "lora-calibration.json"),
        }
        if (
            record["runtime_environment"] != environment
            or record["frozen_sha256"] != checkpoint["frozen_sha256"]
            or set(record["splits"]) != set(references)
        ):
            raise ValueError("Runtime parity receipt has different frozen inputs")
        for split, reference in references.items():
            measured = record["splits"][split]
            actual, previous = np.asarray(measured["scores"]), np.asarray(reference["scores"])
            if (
                actual.ndim != 1
                or actual.shape != previous.shape
                or not len(actual)
                or not np.isfinite(actual).all()
                or not np.isfinite(previous).all()
                or measured["labels"] != reference["labels"]
            ):
                raise ValueError("Runtime parity scores or labels differ from the development population")
            error = float(np.max(np.abs(actual - previous)))
            changes = int(np.sum((actual >= selected["threshold"]) != (previous >= selected["threshold"])))
            if (
                error > 0.03
                or changes
                or measured["max_margin_error"] != error
                or measured["decision_changes"] != changes
            ):
                raise ValueError("Runtime parity did not preserve development margins and cutoff decisions")
            maximum_error = max(maximum_error, error)
    return {
        "backend": "eager",
        "runtime_environment": environment,
        "unique_evaluated_runs": len(names),
        "lifecycle_verified": True,
        "maximum_development_margin_error": maximum_error,
        "development_decision_changes": 0,
        "evaluation_source_sha256": evaluation_plan["evaluation_source_sha256"],
    }


def summarize(study, evidence):
    plan, choice, means, by_seed = check_study(study, evidence)
    verify_evidence(evidence)
    runtime = check_evaluation_lifecycle(study, evidence, plan, choice)
    rows = read(evidence / "test.json")
    paths = {
        "original_ce_512": evidence / "native",
        "original_auc_512": study / "screening/original",
        "selected_auc_512": study / "screening" / choice["recipe"],
        "selected_auc_2581": study / "expanded",
    }
    predictions_by_stage, result = {}, {}
    for stage, root in paths.items():
        measured, predictions_by_stage[stage] = {}, []
        for seed in SEEDS:
            prediction, record = predictions(root / f"seed-{seed}", rows, sha(evidence / "test.json"))
            measured[str(seed)] = record
            predictions_by_stage[stage].append(prediction)
        result[stage] = {
            "seeds": measured,
            "accuracy_mean": float(np.mean([r["accuracy"] for r in measured.values()])),
            "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in measured.values()])),
        }
    embedding = read(evidence / "embedding-evaluation.json")
    if embedding["ids"] != [r["id"] for r in rows]:
        raise ValueError("Embedding reference population differs")
    predictions_by_stage["ridge512"] = embedding["predictions"]["ridge512"]
    pairs = (
        ("original_ce_512", "original_auc_512"),
        ("original_auc_512", "selected_auc_512"),
        ("selected_auc_512", "selected_auc_2581"),
        ("original_ce_512", "selected_auc_2581"),
        ("ridge512", "selected_auc_2581"),
    )
    return {
        "population": "Exposed regression benchmark; not a new prospective test",
        "n": len(rows),
        "recipe": choice["recipe"],
        "export_dtype": plan["export_dtype"],
        "export_amendment": plan["export_amendment"],
        "control_note": "Historical CE control uses float16; the new AUC control also changes export to float32 and uses eager evaluation after development parity checks. Runtime latency comparisons require the same execution backend.",
        "evaluation_runtime": runtime,
        "validation_auc": means,
        "validation_auc_by_seed": by_seed,
        "ridge512_accuracy": float(
            np.mean(np.asarray(predictions_by_stage["ridge512"]) == np.asarray([row["label"] for row in rows]))
        ),
        "stages": result,
        "comparisons": {
            f"{before}_to_{after}": comparison(rows, predictions_by_stage[before], predictions_by_stage[after])
            for before, after in pairs
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path, default=Path("run-output/lora-report.json"))
    args = parser.parse_args()
    result = summarize(args.study, args.evidence)
    write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
