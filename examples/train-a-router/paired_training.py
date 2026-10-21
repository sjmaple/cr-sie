"""One fixed, currency-explicit paired-correctness training path.

Preparation and fitting read source metadata plus a TRAIN grade artifact only.
Calibration and final outcomes are intentionally separate inputs/commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

ARMS = ("cheap", "strong")
SOURCE_PLAN_SHA256 = "e290eb7963829ad31dc2aaaba9bb4ffbf189623023805b563ab83f5b3b2e3d1e"
MODEL = "Qwen/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
INSTRUCTION = "Represent the knowledge and reasoning required to answer this question correctly."
PROTOCOL = {
    "seed": 89,
    "folds": 5,
    "penalties": [0.01, 0.1, 1.0, 10.0],
    "objective": "Two independent correctness logits; minimum summed whole-family OOF Bernoulli NLL",
    "tie": "Equal OOF NLL selects larger L2 penalty",
    "optimizer": {"scipy": "1.17.0", "maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7, "maxls": 40},
    "cost_model": "Four training-only USD means, conditional on arm and correctness; no positive cost-difference floor",
    "native": {
        "base": "convaiinnovations/laya",
        "revision": "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8",
        "max_len": 4096,
        "head_max_len": 256,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "encoder_lr": 0.0002,
        "head_lr": 0.0001,
        "weight_decay": 0.01,
        "micro_batch": 8,
        "grad_accum": 2,
        "evaluation_batch": 16,
        "gradient_checkpointing": True,
        "selection_fold": 0,
        "selection_epochs": 8,
        "selection": "Minimum held-family summed NLL, epoch0..8, ties earlier",
        "refit": "Fresh base and seed89; all training rows for selected epoch count; cosine schedule spans that refit duration, as in the original recipe; epoch0 skips training",
        "base_export": "Copy original checkpoint bytes; amend only max_len/head_max_len in rl_agent_config.json; transient compatibility copies prevent loader mutation",
        "versions": {"laya": "0.3.29", "torch": "2.9.1", "transformers": "4.57.6", "peft": "0.18.1"},
    },
    "policy": "p_strong-p_cheap - lambda*(expected_USD_strong-expected_USD_cheap) >= 0; lambda>=0",
    "primary_gap_pp": 1,
    "secondary_gap_pp": 2,
    "other_targets_pp": [0, 5],
    "calibration": "Cheapest empirically feasible utility policy; all crossings/intervals, exact ties and always-arm bypasses",
    "fitting_boundary": "Training outcomes only; no calibration/final outcomes or development rows in fitting",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sources():
    root = Path(__file__).parent
    return {
        name: sha(root / name)
        for name in ("paired_training.py", "paired_native.py", "paired_topic.py", "utility_router.py")
    }


def prepare(source, pair_path, output, acquisition_protocol_sha256):
    source, output = Path(source), Path(output)
    require(not output.exists(), "Preserve existing prepared studies")
    source_plan, pair = read(source / "source-plan.json"), read(pair_path)
    require(sha(source / "source-plan.json") == SOURCE_PLAN_SHA256, "Different fixed source plan")
    for name in ("question-only/inputs.json", "rows.json"):
        require(sha(source / name) == source_plan["files_sha256"][name], "Source metadata changed")
    inputs, metadata = read(source / "question-only/inputs.json"), read(source / "rows.json")
    require(len(inputs) == len(metadata) == 16110, "Wrong fixed source population")
    require([row["id"] for row in inputs] == [row["id"] for row in metadata], "Source metadata order changed")
    rows = [
        {key: row[key] for key in ("id", "request", "role", "family", "source", "task", "fold") if key in row}
        for row in metadata
    ]
    require(
        all(
            row["request"] == item["request"] and row["role"] == item["role"]
            for row, item in zip(rows, inputs, strict=True)
        ),
        "Actual question text/role differs",
    )
    check_source_rows(rows)
    require(
        pair["pair"] == "sol_luna"
        and pair["arms"]["cheap"]["model"] == "gpt-6-luna"
        and pair["arms"]["strong"]["model"] == "gpt-6.1-sol",
        "Only the selected Sol/Luna pair is prepared",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", acquisition_protocol_sha256)
        and isinstance(pair["cost_scope"], str)
        and pair["cost_scope"],
        "Missing acquisition/cost protocol",
    )
    require(
        pair["source_plan_sha256"] == SOURCE_PLAN_SHA256
        and pair["source_all_inputs_sha256"] == sha(source / "question-only/inputs.json"),
        "Pair specification uses different source inputs",
    )
    output.mkdir(parents=True)
    write(output / "source-rows.json", rows)
    (output / "pair.json").write_bytes(Path(pair_path).read_bytes())
    plan = {
        "format": "routerbench-paired-training-plan-v1",
        "protocol": PROTOCOL,
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "source_inputs_sha256": sha(source / "question-only/inputs.json"),
        "source_rows_sha256": sha(output / "source-rows.json"),
        "pair_sha256": sha(output / "pair.json"),
        "pair": pair,
        "pair_spec_sha256": sha(pair_path),
        "acquisition_protocol_sha256": acquisition_protocol_sha256,
        "source_code_sha256": sources(),
        "populations": {
            role: [row["id"] for row in rows if row["role"] == role]
            for role in ("train", "calibration", "final", "development")
        },
        "training_folds": [row["fold"] for row in rows if row["role"] == "train"],
        "questions": questions(pair),
        "reference_answers_in_packet": False,
    }
    write(output / "plan.json", plan)
    return plan


def check_source_rows(rows):
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)), "Duplicate source ID")
    groups = {}
    for row in rows:
        require(isinstance(row["request"], str) and row["request"].strip(), "Empty source request")
        groups.setdefault(row["family"], set()).add((row["role"], row.get("fold")))
        require(row["role"] in {"train", "calibration", "final", "development"}, "Unknown source role")
        if row["role"] == "train":
            require(type(row.get("fold")) is int and 0 <= row["fold"] < 5, "Missing fixed training fold")
    require(all(len(values) == 1 for values in groups.values()), "Source family crossed roles or training folds")
    require({row["fold"] for row in rows if row["role"] == "train"} == set(range(5)), "Need all five training folds")


def checked_plan(run):
    run = Path(run)
    plan, rows = read(run / "plan.json"), read(run / "source-rows.json")
    require(plan["protocol"] == PROTOCOL and plan["source_code_sha256"] == sources(), "Prepared method/source changed")
    require(
        plan["source_rows_sha256"] == sha(run / "source-rows.json") and plan["pair_sha256"] == sha(run / "pair.json"),
        "Prepared inputs changed",
    )
    require(
        plan["pair"] == read(run / "pair.json") and plan["pair_spec_sha256"] == plan["pair_sha256"],
        "Pair metadata changed",
    )
    check_source_rows(rows)
    for role, identities in plan["populations"].items():
        require(identities == [row["id"] for row in rows if row["role"] == role], "Population order changed")
    return plan, rows


def questions(pair):
    return {
        f"{arm}_correct": {
            "type": "choice",
            "instructions": f"Predict whether {pair['arms'][arm]['model']} will answer this request correctly. Judge its likely answer quality; do not answer the request.",
            "criteria": {
                "incorrect": "The solver will give an incorrect answer.",
                "correct": "The solver will give the correct answer.",
            },
        }
        for arm in ARMS
    }


def grade_arrays(plan, source_rows, grades, split):
    require(grades["format"] == "routerbench-paired-grades-v1" and grades["split"] == split, "Wrong grade role/format")
    pair = plan["pair"]
    require(
        grades["source_plan_sha256"] == plan["source_plan_sha256"]
        and grades["pair"] == pair["pair"]
        and grades["arms"] == pair["arms"]
        and grades["pair_spec_sha256"] == plan["pair_spec_sha256"]
        and grades["acquisition_protocol_sha256"] == plan["acquisition_protocol_sha256"]
        and grades["cost_scope"] == pair["cost_scope"],
        "Grade source/model/request/cost contract changed",
    )
    require(re.fullmatch(r"[0-9a-f]{64}", grades["grader_sha256"]), "Missing grader provenance")
    rows = [row for row in source_rows if row["role"] == split]
    by_id = {row["id"]: row for row in grades["rows"]}
    require(
        len(by_id) == len(grades["rows"]) == len(rows) and set(by_id) == {row["id"] for row in rows},
        "Grades must cover exactly the fixed role, without dropped rows",
    )
    labels, costs = [], []
    for row in rows:
        grade = by_id[row["id"]]
        require(
            all(type(grade[f"{arm}_correct"]) in (bool, int) and grade[f"{arm}_correct"] in (0, 1) for arm in ARMS),
            "Paired labels must be binary, including invalid finals as incorrect",
        )
        money = [grade[f"{arm}_cost_usd"] for arm in ARMS]
        require(
            all(type(value) in (float, int) and math.isfinite(value) and value >= 0 for value in money),
            "Missing/nonfinite USD cannot be imputed free",
        )
        require(
            all(re.fullmatch(r"[0-9a-f]{64}", grade[f"{arm}_record_sha256"]) for arm in ARMS),
            "Missing physical response provenance",
        )
        labels.append([int(grade[f"{arm}_correct"]) for arm in ARMS])
        costs.append(money)
    return rows, np.asarray(labels, dtype=float), np.asarray(costs, dtype=float)


def conditional_usd(targets, costs):
    targets, costs = np.asarray(targets), np.asarray(costs, dtype=float)
    require(
        targets.ndim == 2
        and targets.shape[1] == 2
        and costs.shape == targets.shape
        and np.isin(targets, (0, 1)).all()
        and np.isfinite(costs).all()
        and (costs >= 0).all(),
        "Bad paired targets/USD",
    )
    means, counts = np.empty((2, 2)), np.empty((2, 2), dtype=int)
    for arm in range(2):
        for label in range(2):
            selected = targets[:, arm] == label
            require(selected.any(), "Every arm/correctness conditional cost mean must be defined")
            means[arm, label], counts[arm, label] = costs[selected, arm].mean(), selected.sum()
    return {
        "unit": "USD",
        "arm_order": list(ARMS),
        "class_order": [0, 1],
        "means": means.tolist(),
        "counts": counts.tolist(),
    }


def expected_usd(probabilities, statistics):
    values, means = np.asarray(probabilities, dtype=float), np.asarray(statistics["means"], dtype=float)
    require(
        statistics["unit"] == "USD"
        and statistics["arm_order"] == list(ARMS)
        and statistics["class_order"] == [0, 1]
        and means.shape == (2, 2),
        "Wrong cost units/order",
    )
    require(
        values.ndim == 2
        and values.shape[1] == 2
        and np.isfinite(values).all()
        and ((values >= 0) & (values <= 1)).all()
        and np.isfinite(means).all()
        and (means >= 0).all(),
        "Invalid cost prediction inputs",
    )
    return (1 - values) * means[:, 0] + values * means[:, 1]


def probability(matrix, model):
    return np.exp(-np.logaddexp(0, -(np.asarray(matrix, dtype=float) @ np.asarray(model["weights"]) + model["bias"])))


def nll(targets, probabilities):
    p = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return np.mean(-targets * np.log(p) - (1 - targets) * np.log1p(-p), axis=0)


def fit_logistic(matrix, targets, penalty):
    """Same stable softplus objective/tolerances as the tested arithmetic optimizer."""
    import scipy
    from scipy.optimize import minimize
    from scipy.special import expit

    require(scipy.__version__ == PROTOCOL["optimizer"]["scipy"], "Use pinned SciPy")
    require(
        type(penalty) in (float, int) and math.isfinite(penalty) and penalty > 0,
        "L2 penalty must be positive and finite",
    )
    matrix, targets = np.asarray(matrix, dtype=float), np.asarray(targets, dtype=float)
    require(
        matrix.ndim == 2
        and targets.shape == (len(matrix),)
        and np.isfinite(matrix).all()
        and np.isin(targets, (0, 1)).all()
        and 0 < targets.mean() < 1,
        "Each training fold needs finite vectors and both classes",
    )
    mean, frequency = matrix.mean(axis=0), targets.mean()
    centered = np.ascontiguousarray(matrix - mean)
    initial = np.zeros(matrix.shape[1] + 1)
    initial[-1] = np.log(frequency / (1 - frequency))

    def loss(theta):
        weights, bias = theta[:-1], theta[-1]
        logits = centered @ weights + bias
        residual = expit(logits) - targets
        value = np.logaddexp(0, (1 - 2 * targets) * logits).sum() + 0.5 * penalty * (weights @ weights)
        return float(value), np.r_[centered.T @ residual + penalty * weights, residual.sum()]

    result = minimize(
        loss,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={key: PROTOCOL["optimizer"][key] for key in ("maxiter", "ftol", "gtol", "maxls")},
    )
    require(result.success and np.isfinite(result.x).all(), f"Logistic optimization failed: {result.message}")
    return {
        "weights": result.x[:-1].tolist(),
        "bias": float(result.x[-1] - mean @ result.x[:-1]),
        "optimizer": {"success": bool(result.success), "iterations": int(result.nit), "objective": float(result.fun)},
    }


def fit_embedding_arrays(matrix, rows, targets, costs, *, fitter=fit_logistic, progress=print):
    matrix, targets = np.asarray(matrix, dtype=float), np.asarray(targets, dtype=float)
    check_source_rows(rows)
    require(all(row["role"] == "train" for row in rows), "Only training rows may enter fitting")
    require(
        matrix.ndim == 2
        and len(matrix) == len(rows)
        and targets.shape == (len(rows), 2)
        and np.isfinite(matrix).all()
        and np.isin(targets, (0, 1)).all(),
        "Misaligned training features/labels",
    )
    cost_statistics = conditional_usd(targets, costs)
    folds, candidates = np.asarray([row["fold"] for row in rows]), []
    for penalty in PROTOCOL["penalties"]:
        predictions, optimizer_runs = np.empty_like(targets), []
        for fold in range(5):
            held = folds == fold
            for arm, name in enumerate(ARMS):
                model = fitter(matrix[~held], targets[~held, arm], penalty)
                predictions[held, arm] = probability(matrix[held], model)
                optimizer_runs.append({"fold": fold, "arm": name, **model["optimizer"]})
        losses = nll(targets, predictions)
        candidates.append(
            {
                "penalty": penalty,
                "summed_binary_nll": float(losses.sum()),
                "arm_nll": losses.tolist(),
                "oof_probabilities": predictions.tolist(),
                "optimizer_runs": optimizer_runs,
            }
        )
        progress(f"Training-family CV lambda={penalty:g}, summed NLL={losses.sum():.6f}")
    chosen = min(candidates, key=lambda row: (row["summed_binary_nll"], -row["penalty"]))
    models = {name: fitter(matrix, targets[:, arm], chosen["penalty"]) for arm, name in enumerate(ARMS)}
    return {
        "training_ids": [row["id"] for row in rows],
        "folds": folds.tolist(),
        "penalty": chosen["penalty"],
        "models": models,
        "cv": candidates,
        "conditional_cost_usd": cost_statistics,
    }


def feature_matrix(plan, all_rows, features):
    features = Path(features)
    info, ids = read(features / "features.json"), read(features / "ids.json")
    require(
        info["inputs_sha256"] == plan["source_inputs_sha256"]
        and info["model"] == MODEL
        and info["hf_revision"] == REVISION
        and info["instruction"] == INSTRUCTION
        and info["dimensions"] == 2560,
        "Incompatible cached embeddings",
    )
    require(
        sha(features / "ids.json") == info["ids_sha256"] and sha(features / "vectors.npy") == info["vectors_sha256"],
        "Cached vector bytes changed",
    )
    require(ids == [row["id"] for row in all_rows], "Vector order differs from all fixed questions")
    matrix = np.load(features / "vectors.npy", mmap_mode="r", allow_pickle=False)
    require(matrix.shape == (len(ids), 2560), "Feature dimension mismatch")
    return matrix, info


def train_embedding(run, grades_path, features):
    run, features = Path(run), Path(features)
    require(not (run / "final-access.json").exists(), "Final outcomes already accessed")
    require(not (run / "embedding.json").exists(), "Preserve previous selected model")
    plan, all_rows = checked_plan(run)
    rows, targets, costs = grade_arrays(plan, all_rows, read(grades_path), "train")
    matrix, _ = feature_matrix(plan, all_rows, features)
    indices = [index for index, row in enumerate(all_rows) if row["role"] == "train"]
    fit = fit_embedding_arrays(matrix[indices], rows, targets, costs)
    artifact = {
        "format": "routerbench-paired-embedding-v1",
        "plan_sha256": sha(run / "plan.json"),
        "training_grades_sha256": sha(grades_path),
        "grader_sha256": read(grades_path)["grader_sha256"],
        "features_sha256": sha(features / "features.json"),
        "fit": fit,
        "cost_scope": plan["pair"]["cost_scope"],
        "boundary": "Selected from training-only family CV; calibration/final outcomes unopened.",
    }
    write(run / "embedding.json", artifact)
    return artifact


def score_embedding(run, features, output):
    run, features = Path(run), Path(features)
    plan, rows = checked_plan(run)
    model = read(run / "embedding.json")
    require(
        model["plan_sha256"] == sha(run / "plan.json") and model["fit"]["training_ids"] == plan["populations"]["train"],
        "Different fitted population/plan",
    )
    matrix, _ = feature_matrix(plan, rows, features)
    result = {
        "format": "routerbench-input-only-probabilities-v1",
        "method": "embedding",
        "plan_sha256": sha(run / "plan.json"),
        "model_sha256": sha(run / "embedding.json"),
        "features_sha256": sha(features / "features.json"),
        "outcomes_accessed": False,
        "conditional_cost_usd": model["fit"]["conditional_cost_usd"],
        "roles": {},
    }
    for role in ("calibration", "final", "development"):
        indices = [i for i, row in enumerate(rows) if row["role"] == role]
        probabilities = np.column_stack([probability(matrix[indices], model["fit"]["models"][arm]) for arm in ARMS])
        result["roles"][role] = {
            "ids": [rows[i]["id"] for i in indices],
            "probabilities": probabilities.tolist(),
            "expected_usd": expected_usd(probabilities, result["conditional_cost_usd"]).tolist(),
        }
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--pair", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--acquisition-protocol-sha256", required=True)
    p = commands.add_parser("train-embedding")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--grades", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p = commands.add_parser("score-embedding")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.pair, args.run, args.acquisition_protocol_sha256)
        print(json.dumps({"plan_sha256": sha(args.run / "plan.json"), "n_train": len(result["populations"]["train"])}))
    elif args.command == "train-embedding":
        result = train_embedding(args.run, args.grades, args.features)
        print(
            json.dumps(
                {"selected_penalty": result["fit"]["penalty"], "artifact_sha256": sha(args.run / "embedding.json")}
            )
        )
    else:
        score_embedding(args.run, args.features, args.output)
        print(json.dumps({"scores_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
