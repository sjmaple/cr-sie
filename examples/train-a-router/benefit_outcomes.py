"""Predict each solver's correctness and route by benefit per expected credit.

Prepare the source-only population before acquiring new outcomes. Fit the two
heads using paper-group training folds, then freeze calibration cutoffs in a
separate step. Replay and live scoring need NumPy; fitting also needs SciPy.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

import benefit
import solver_policies
import solver_report
import source_review
from data import SOURCE_HASHES, SOURCE_REVISION, audit, digest, file_hash, write
from serving import INSTRUCTION, MODEL, REVISION, vectors

PROTOCOL = {
    "target": "Two separate Bernoulli targets: cheap_correct and strong_correct",
    "training_pool_sizes": [512, 2581],
    "required_original_exclusions": ["e2h-amc-eval-661"],
    "encoder": MODEL,
    "revision": REVISION,
    "instruction": INSTRUCTION,
    "dimensions": 2560,
    "penalties": [0.01, 0.1, 1.0, 10.0],
    "folds": 5,
    "seed": 89,
    "fold_assignment": "Sort paper groups, permute with NumPy seed89, assign whole groups round-robin to five folds",
    "fit": "Independent logistic heads: summed Bernoulli NLL plus 0.5*lambda*L2(weights); unpenalized intercept; no class weighting",
    "selection": "One shared lambda minimizes pooled training-fold cheap NLL plus strong NLL; exact ties prefer larger lambda",
    "optimizer": {"scipy": "1.17.0", "method": "L-BFGS-B", "maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7, "maxls": 40},
    "cost_model": "Four training-only means of uncached Cloud token-cost proxies, conditioned on each solver's correctness; expected cost is the probability-weighted mixture",
    "score": "(p_strong_correct-p_cheap_correct)/max(expected_strong_credits-expected_cheap_credits,1.0)",
    "incremental_cost_floor_credits": 1.0,
    "cost_interpretation": "Training token-cost proxies are not observed Cloud charges or self-hosting costs. Calibration and test accounting use measured credits.",
    "primary_max_quality_loss_pp": 1,
    "secondary_max_quality_loss_pp": [0, 2, 5],
    "expansion": "The larger pool adds examples within the same32 training paper groups, not new papers or years",
    "eligibility": "Original collection eligibility is preserved; a separately bound source-reference audit may exclude malformed questions after training outcomes exist, before corrected fitting and held-out solver outcome access",
    "test_boundary": "Freeze every model and cutoff before solver test OUTCOME ACCESS, not before test collection",
    "motivation": "Calibration diagnostics and training-only learning curves motivated this bounded original-versus-expanded comparison; no additional architecture search",
}
CODE_FILES = (
    "benefit_outcomes.py",
    "benefit.py",
    "policies.py",
    "serving.py",
    "solver_policies.py",
    "solver_report.py",
    "data.py",
    "source_review.py",
    "teacher_format.py",
)
PLAN_FORMAT = "sie-solver-outcome-plan-v2"
MODEL_FORMAT = "sie-solver-outcome-selected-v1"
FROZEN_FORMAT = "sie-solver-outcome-frozen-v1"
ACCESS_FORMAT = "sie-solver-outcome-test-access-v1"
TARGETS = (1, 0, 2, 5)
ARMS = ("cheap", "strong")
GRADE_HASHES = (
    "grader_source_sha256",
    "gold_review_source_sha256",
    "gold_review_sha256",
    "training_gold_rules_sha256",
    "training_gold_review_source_sha256",
    "outcome_grader_source_sha256",
    "outcome_grader_protocol_sha256",
)
SHARED_GRADE_HASHES = (
    "grader_source_sha256",
    "gold_review_source_sha256",
    "outcome_grader_source_sha256",
    "outcome_grader_protocol_sha256",
)


def read(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in CODE_FILES}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def grading_provenance(record):
    """Bind base semantics plus the same explicit source audit on every split."""
    legacy = solver_policies._grading_provenance(record)
    extra = {name: record.get(name) for name in SHARED_GRADE_HASHES[2:]}
    _require(
        all(_sha(value) for value in extra.values()), "Outcome grades require the shared v2 source and protocol hashes"
    )
    reviewed = {name: record[name] for name in source_review.METADATA_KEYS if name in record}
    _require(
        not reviewed or (len(reviewed) == len(source_review.METADATA_KEYS) and all(_sha(v) for v in reviewed.values())),
        "Source-review provenance must be complete",
    )
    return legacy | extra | reviewed


def _unopened(run):
    for name in ("test-access.json", "report.json"):
        if (Path(run) / name).exists():
            raise FileExistsError(Path(run) / name)


def _population(rows):
    return [
        {
            "id": row["id"],
            "request_sha256": digest(row["request"]),
            "paper_group": row["paper_group"],
            "family": row["family"],
            "year": row["year"],
        }
        for row in rows
    ]


def _eligibility(receipt, pool, dataset_sha):
    ids = [row["id"] for row in pool]
    exclusions = receipt.get("exclusions", [])
    excluded = [row.get("id") for row in exclusions]
    eligible = [identity for identity in ids if identity not in excluded]
    _require(
        receipt.get("format") == "sie-benefit-training-eligibility-v1"
        and receipt.get("source_only") is True
        and (
            receipt.get("decided_before_training_outcomes") is True
            or (
                receipt.get("decided_before_new_training_outcomes") is True
                and receipt.get("reused_training_outcomes_already_available") is True
            )
        )
        and receipt.get("source_revision") == SOURCE_REVISION
        and receipt.get("source_hashes") == SOURCE_HASHES
        and receipt.get("dataset_sha256") == dataset_sha
        and receipt.get("original_pool_ids") == ids
        and len(ids) in PROTOCOL["training_pool_sizes"]
        and len(set(ids)) == len(ids)
        and len(set(excluded)) == len(excluded)
        and set(excluded).issubset(ids)
        and set(PROTOCOL["required_original_exclusions"]).issubset(excluded)
        and all(isinstance(row.get("reason"), str) and row["reason"].strip() for row in exclusions)
        and receipt.get("eligible_train_ids") == eligible
        and _sha(receipt.get("training_gold_review_source_sha256")),
        "Eligibility must preserve every nonexcluded source input in order, before new outcomes",
    )
    return eligible


def _training_contract(plan, inputs, eligibility, eligibility_sha, population, pool_size, dataset_sha, settings_sha):
    expected = [{key: row[key] for key in ("id", "request_sha256")} for row in population]
    _require(
        plan.get("dataset_sha256") == dataset_sha
        and plan.get("settings_sha256") == settings_sha
        and plan.get("inputs_sha256") == digest(inputs)
        and plan.get("source_counts", {}).get("train") == pool_size
        and plan.get("counts", {}).get("train") == len(population)
        and plan.get("eligible_train_ids") == [row["id"] for row in population]
        and plan.get("excluded_train_ids") == [row["id"] for row in eligibility["exclusions"]]
        and plan.get("eligibility_sha256") == eligibility_sha
        and [{"id": row["id"], "request_sha256": digest(row["request"])} for row in inputs["train"]] == expected
        and all(_sha(plan.get(name)) for name in GRADE_HASHES)
        and plan["training_gold_review_source_sha256"] == eligibility["training_gold_review_source_sha256"],
        "Training collection plan, inputs, source review and eligible population differ",
    )


def prepare(
    baseline,
    original_path,
    training_plan_path,
    training_inputs_path,
    training_rows_path,
    eligibility_path,
    *,
    expansion_audit_path=None,
    source_review_path=None,
    attest_test_outcomes_unopened=False,
):
    if attest_test_outcomes_unopened is not True:
        raise ValueError("An explicit assertion that solver test outcomes remain unopened is required")
    dataset, original, solver_plan, artifacts = benefit._inputs(baseline, original_path)
    pool, eligibility, training_plan, inputs = map(
        read, (training_rows_path, eligibility_path, training_plan_path, training_inputs_path)
    )
    original_rows = dataset["splits"]["train"]
    _require(pool[: len(original_rows)] == original_rows, "Training pool must preserve the original512 prefix exactly")
    # The existing audit checks IDs, canonical requests, families, papers and
    # temporal partitions. Human difficulty labels are never fitting targets.
    audit(dataset | {"splits": dataset["splits"] | {"train": pool}})
    _require(
        {row["paper_group"] for row in pool} == {row["paper_group"] for row in original_rows},
        "This bounded expansion must retain exactly the original training paper groups",
    )
    expansion = None
    if len(pool) > len(original_rows):
        if expansion_audit_path is None:
            raise ValueError("Expanded training requires the pinned source expansion audit")
        expansion = read(expansion_audit_path)
        _require(
            expansion.get("source_revision") == SOURCE_REVISION
            and expansion.get("source_hashes") == SOURCE_HASHES
            and expansion.get("original_dataset_sha256") == digest(dataset)
            and expansion.get("expanded_rows_sha256") == digest(pool)
            and expansion.get("original_prefix_rows") == len(original_rows)
            and expansion.get("training_rows") == len(pool)
            and expansion.get("ids") == [row["id"] for row in pool]
            and expansion.get("families") == [row["family"] for row in pool],
            "Expanded source rows differ from the original pinned expansion",
        )
    ids = _eligibility(eligibility, pool, digest(dataset))
    population = _population([row for row in pool if row["id"] in set(ids)])
    _training_contract(
        training_plan,
        inputs,
        eligibility,
        file_hash(eligibility_path),
        population,
        len(pool),
        digest(dataset),
        solver_plan["settings_sha256"],
    )
    _require(
        all(training_plan.get(name) == original.get(name) for name in SHARED_GRADE_HASHES),
        "Training and calibration must use the same v2 grader and base dependencies",
    )
    source_populations = {
        "train": population,
        **{split: _population(dataset["splits"][split]) for split in ("calibration", "test")},
    }
    populations, review_binding = source_populations, None
    if source_review_path is not None:
        review = source_review.checked_review(read(source_review_path))
        populations = {
            split: source_review.eligible_rows(rows, split, review) for split, rows in source_populations.items()
        }
        review_binding = source_review.metadata(source_review_path) | {
            "exclusions": {
                split: [row["id"] for row in rows if row["id"] not in {item["id"] for item in populations[split]}]
                for split, rows in source_populations.items()
            },
        }
        _require(
            all(original.get(key) == review_binding[key] for key in source_review.METADATA_KEYS),
            "Calibration and planned source review differ",
        )
    else:
        _require(
            not any(key in original for key in source_review.METADATA_KEYS),
            "Reviewed calibration needs its source review",
        )
    return benefit.seal(
        {
            "format": PLAN_FORMAT,
            "created_at": benefit.now(),
            "protocol": PROTOCOL,
            "source_code_sha256": code_hashes(),
            "dataset_sha256": digest(dataset),
            "original_frozen_sha256": file_hash(original_path),
            "original_solver_plan_sha256": original["plan_sha256"],
            "solver_settings_sha256": solver_plan["settings_sha256"],
            "grading_provenance": grading_provenance(original),
            "training_grade_provenance": {name: training_plan[name] for name in GRADE_HASHES},
            "training_artifacts_sha256": {
                "plan": file_hash(training_plan_path),
                "inputs": file_hash(training_inputs_path),
                "rows": file_hash(training_rows_path),
                "eligibility": file_hash(eligibility_path),
                "expansion_audit": None if expansion_audit_path is None else file_hash(expansion_audit_path),
            },
            "eligibility": eligibility,
            "expansion_audit": expansion,
            "training_pool": _population(pool),
            "source_populations": source_populations,
            "source_review": review_binding,
            "populations": populations,
            "training_folds": benefit.group_folds([row["paper_group"] for row in populations["train"]]),
            "baseline_artifacts_sha256": artifacts,
            "attest_test_outcomes_unopened": True,
            "access_limitation": "Receipts record the declared boundary; they cannot prove nobody read another copy",
        }
    )


def checked_plan(plan):
    benefit.checked(plan, PLAN_FORMAT)
    grading_provenance(plan["grading_provenance"])
    eligible = _eligibility(plan["eligibility"], plan["training_pool"], plan["dataset_sha256"])
    _require(
        plan["protocol"] == PROTOCOL
        and plan["source_code_sha256"] == code_hashes()
        and plan["attest_test_outcomes_unopened"] is True
        and plan["source_populations"]["train"] == [row for row in plan["training_pool"] if row["id"] in set(eligible)]
        and plan["training_folds"] == benefit.group_folds([row["paper_group"] for row in plan["populations"]["train"]])
        and all(_sha(plan["training_grade_provenance"].get(name)) for name in GRADE_HASHES),
        "Prepared outcome protocol, source, eligible rows or paper folds changed",
    )
    review = plan["source_review"]
    if review is not None:
        _require(
            all(
                _sha(review.get(key)) and plan["grading_provenance"].get(key) == review[key]
                for key in source_review.METADATA_KEYS
            )
            and review["source_review_grader_sha256"] == file_hash(Path(source_review.__file__))
            and review["format_normalizer_sha256"] == file_hash(Path(__file__).with_name("teacher_format.py")),
            "Source review or formatting implementation changed after preparation",
        )
    for split, rows in plan["source_populations"].items():
        excluded = [] if review is None else review["exclusions"][split]
        _require(
            len(excluded) == len(set(excluded))
            and set(excluded).issubset(row["id"] for row in rows)
            and plan["populations"][split] == [row for row in rows if row["id"] not in set(excluded)],
            "Reviewed population must preserve every nonexcluded source input in order",
        )
    _require(
        all(
            plan["training_grade_provenance"][name] == plan["grading_provenance"][name] for name in SHARED_GRADE_HASHES
        ),
        "Prepared training and held-out v2 grading contracts differ",
    )
    for split in ("calibration", "test"):
        for key in ("id", "request_sha256", "paper_group", "family"):
            _require(
                not {row[key] for row in plan["training_pool"]} & {row[key] for row in plan["populations"][split]},
                "Training overlaps a held population",
            )
    return plan


def _baseline(plan, baseline):
    baseline = Path(baseline)
    _require(
        all(file_hash(baseline / name) == sha for name, sha in plan["baseline_artifacts_sha256"].items()),
        "Pinned baseline inputs or vectors changed",
    )
    dataset = read(baseline / "data.json")
    _require(digest(dataset) == plan["dataset_sha256"], "Baseline dataset changed")
    return dataset


def held_rows(plan, dataset, split):
    rows = dataset["splits"][split]
    _require(_population(rows) == plan["source_populations"][split], "Held acquisition input order changed")
    ids = {row["id"] for row in plan["populations"][split]}
    selected = [row for row in rows if row["id"] in ids]
    _require(_population(selected) == plan["populations"][split], "Reviewed held input order changed")
    return selected


def test_credits(plan, path):
    """Validate the complete measured receipt, then select reviewed test IDs."""
    full = plan["source_populations"]["test"]
    credits, sha = solver_policies._credits(
        path,
        "test",
        plan["original_solver_plan_sha256"],
        [row["id"] for row in full],
        {row["id"]: row["request_sha256"] for row in full},
    )
    if credits is None:
        return None, sha
    eligible = {row["id"] for row in plan["populations"]["test"]}
    return [value for row, value in zip(full, credits, strict=True) if row["id"] in eligible], sha


def _training(plan, training_plan_path, training_inputs_path, training_rows_path, grades_path):
    for name, path in (("plan", training_plan_path), ("inputs", training_inputs_path), ("rows", training_rows_path)):
        _require(
            file_hash(path) == plan["training_artifacts_sha256"][name], "Training inputs changed after preparation"
        )
    collection, inputs, pool, grades = map(
        read, (training_plan_path, training_inputs_path, training_rows_path, grades_path)
    )
    _require(_population(pool) == plan["training_pool"], "Training source metadata changed")
    _training_contract(
        collection,
        inputs,
        plan["eligibility"],
        plan["training_artifacts_sha256"]["eligibility"],
        plan["source_populations"]["train"],
        len(pool),
        plan["dataset_sha256"],
        plan["solver_settings_sha256"],
    )
    _require(
        grades.get("split") == "train"
        and grades.get("plan_sha256") == file_hash(training_plan_path)
        and [row["id"] for row in grades["rows"]] == [row["id"] for row in plan["populations"]["train"]]
        and all(
            grades.get(name) == collection.get(name) == plan["training_grade_provenance"][name] for name in GRADE_HASHES
        ),
        "Training grades differ from the frozen population or reviewed source provenance",
    )
    grading_provenance(grades)
    if plan["source_review"] is not None:
        _require(
            grades.get("collected_ids") == [row["id"] for row in plan["source_populations"]["train"]]
            and grades.get("excluded_ids") == plan["source_review"]["exclusions"]["train"]
            and all(grades.get(key) == plan["source_review"][key] for key in source_review.METADATA_KEYS),
            "Training grades must retain the complete collection and explicit source-only exclusions",
        )
    solver_report.summarize_quality(grades["rows"], "always_strong")
    eligible = {row["id"] for row in plan["populations"]["train"]}
    return [row for row in inputs["train"] if row["id"] in eligible], grades


def _fraction(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def conditional_costs(targets, costs):
    targets, costs = np.asarray(targets), np.asarray(costs, dtype=float)
    _require(
        targets.shape == costs.shape
        and targets.ndim == 2
        and targets.shape[1] == 2
        and np.isin(targets, (0, 1)).all()
        and np.isfinite(costs).all()
        and (costs >= 0).all(),
        "Cost targets must be finite, nonnegative and aligned with both outcomes",
    )
    result = {}
    for arm, name in enumerate(ARMS):
        for correct, state in ((0, "wrong"), (1, "correct")):
            values = costs[targets[:, arm] == correct, arm]
            _require(len(values) > 0, "Each outcome head needs correct and incorrect training examples")
            result[f"{name}_{state}"] = {
                "n": len(values),
                "sum_credits_proxy": float(values.sum()),
                "mean_credits_proxy": float(values.mean()),
            }
    return result


def cost_proxy(path, price_path, training_plan_path, grades_path, grades):
    record, prices = read(path), read(price_path)
    _require(
        record.get("format") == "sie-training-uncached-cloud-cost-proxy-v1"
        and record.get("split") == "train"
        and record.get("n") == len(grades["rows"])
        and record.get("credit_usd") == 1e-5
        and record.get("training_plan_sha256") == file_hash(training_plan_path)
        and record.get("training_grades_sha256") == file_hash(grades_path)
        and record.get("training_collection_sha256") == grades.get("collection_summary_sha256")
        and record.get("price_source", {}).get("sha256") == file_hash(price_path)
        and [row["id"] for row in record["rows"]] == [row["id"] for row in grades["rows"]],
        "Training cost proxy must bind the same paired rows, collection and public prices",
    )
    rates = record["rates_credits_per_token"]
    public_rows = record["price_source"]["public_rows"]
    _require(
        prices.get("currency") == "usd"
        and record["price_source"].get("pricing_version") == prices.get("pricing_version")
        and record["price_source"].get("rate_book_sha256") == prices.get("rate_book_sha256")
        and all(row in prices.get("items", []) for row in public_rows),
        "Embedded price rows must match the bound public snapshot",
    )
    models = {"cheap": "Qwen/Qwen3.5-4B", "strong": "Qwen/Qwen3.8-27B-FP8"}
    for arm in ARMS:
        for unit in ("input_tokens", "output_tokens"):
            matching = [
                row
                for row in public_rows
                if row["model"] == models[arm]
                and row["unit"] == unit
                and row["profile"] == "default"
                and row["region"] == "us"
            ]
            _require(
                len(matching) == 1
                and _fraction(rates[arm][unit]) >= 0
                and _fraction(rates[arm][unit]) == _fraction(matching[0]["usd_per_unit"]) * 100000,
                "Credit rates disagree with the public USD prices",
            )
    costs = []
    for proxy, grade in zip(record["rows"], grades["rows"], strict=True):
        row_costs = []
        for arm in ARMS:
            _require(
                type(proxy[f"{arm}_correct"]) is bool
                and proxy[f"{arm}_correct"] == grade[f"{arm}_correct"]
                and _sha(proxy[f"{arm}_record_sha256"])
                and proxy[f"{arm}_record_sha256"] == grade[f"{arm}_record_sha256"],
                "Cost proxy correctness or raw-record binding differs",
            )
            calls = proxy[f"{arm}_calls"]
            _require(bool(calls), "Every training answer needs at least one priced physical call")
            total = 0
            for call in calls:
                _require(
                    all(
                        type(call[key]) is int and call[key] >= 0
                        for key in ("prompt_tokens", "completion_tokens", "credits_proxy")
                    ),
                    "Token counts and cost proxies must be nonnegative integers",
                )
                charge = math.ceil(
                    call["prompt_tokens"] * _fraction(rates[arm]["input_tokens"])
                    + call["completion_tokens"] * _fraction(rates[arm]["output_tokens"])
                )
                _require(charge == call["credits_proxy"], "Physical-call cost formula or rounding differs")
                total += charge
            _require(
                type(proxy[f"{arm}_credits_proxy"]) is int and total == proxy[f"{arm}_credits_proxy"],
                "Question cost must sum every initial and finalizer call",
            )
            row_costs.append(total)
        costs.append(row_costs)
    targets = [[int(row[f"{arm}_correct"]) for arm in ARMS] for row in grades["rows"]]
    statistics = conditional_costs(targets, costs)
    _require(statistics == record["conditional_means"], "Training cost means disagree with recomputation")
    return costs, statistics, record


def probability(matrix, model):
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    _require(
        matrix.ndim == 2
        and weights.shape == (matrix.shape[1],)
        and np.isfinite(matrix).all()
        and np.isfinite(weights).all()
        and np.isfinite(model["bias"]),
        "Outcome features or weights are invalid",
    )
    logits = matrix @ weights + model["bias"]
    return np.exp(-np.logaddexp(0, -logits))


def normalized_scores(probabilities, statistics):
    probabilities = np.asarray(probabilities, dtype=float)
    _require(
        probabilities.ndim == 2
        and probabilities.shape[1] == 2
        and np.isfinite(probabilities).all()
        and ((probabilities >= 0) & (probabilities <= 1)).all(),
        "Need two finite probabilities per request",
    )
    expected = np.column_stack(
        [
            probabilities[:, arm] * statistics[f"{name}_correct"]["mean_credits_proxy"]
            + (1 - probabilities[:, arm]) * statistics[f"{name}_wrong"]["mean_credits_proxy"]
            for arm, name in enumerate(ARMS)
        ]
    )
    _require(np.isfinite(expected).all() and (expected >= 0).all(), "Invalid conditional cost means")
    delta = expected[:, 1] - expected[:, 0]
    scores = (probabilities[:, 1] - probabilities[:, 0]) / np.maximum(delta, PROTOCOL["incremental_cost_floor_credits"])
    return scores, expected


def score(matrix, head):
    """Return the fixed benefit-per-credit score; no SciPy or model download."""
    probabilities = np.column_stack([probability(matrix, head["models"][arm]) for arm in ARMS])
    return normalized_scores(probabilities, head["conditional_cost_statistics"])[0]


def fit_logistic(matrix, targets, penalty):
    # Optional training dependency; replay/live inference remain NumPy-only.
    import scipy
    from scipy.optimize import minimize
    from scipy.special import expit

    _require(scipy.__version__ == PROTOCOL["optimizer"]["scipy"], "Fit requires the declared SciPy version")
    mean = matrix.mean(axis=0)
    centered = np.ascontiguousarray(matrix - mean)
    frequency = float(targets.mean())
    _require(0 < frequency < 1, "Each training fold needs both correctness classes for each solver")
    initial = np.zeros(matrix.shape[1] + 1)
    initial[-1] = np.log(frequency / (1 - frequency))

    def loss(theta):
        weights, bias = theta[:-1], theta[-1]
        logits = centered @ weights + bias
        residual = expit(logits) - targets
        value = np.logaddexp(0, logits).sum() - targets @ logits + 0.5 * penalty * (weights @ weights)
        return float(value), np.r_[centered.T @ residual + penalty * weights, residual.sum()]

    options = {key: PROTOCOL["optimizer"][key] for key in ("maxiter", "ftol", "gtol", "maxls")}
    result = minimize(loss, initial, method="L-BFGS-B", jac=True, options=options)
    _require(result.success and np.isfinite(result.x).all(), f"Logistic optimization failed: {result.message}")
    return {
        "weights": result.x[:-1].tolist(),
        "bias": float(result.x[-1] - mean @ result.x[:-1]),
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "objective": float(result.fun),
            "gradient_absolute_max": float(np.abs(result.jac).max()),
        },
    }


def _nll(targets, probabilities):
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return np.mean(-targets * np.log(probabilities) - (1 - targets) * np.log1p(-probabilities), axis=0)


def fit_head(matrix, rows, groups, costs, *, progress=None, fold_assignments=None):
    solver_report.summarize_quality(rows, "always_strong")
    matrix = np.asarray(matrix, dtype=float)
    targets = np.array([[int(row[f"{arm}_correct"]) for arm in ARMS] for row in rows], dtype=float)
    _require(
        matrix.ndim == 2 and len(matrix) == len(rows) == len(groups) and np.isfinite(matrix).all(),
        "Training vectors, paired targets and paper groups must align",
    )
    folds = np.array(benefit.group_folds(groups) if fold_assignments is None else fold_assignments)
    _require(
        folds.shape == (len(rows),)
        and set(folds.tolist()) == set(range(PROTOCOL["folds"]))
        and all(len({int(folds[i]) for i, item in enumerate(groups) if item == group}) == 1 for group in set(groups)),
        "Every training paper must stay in one of the declared folds",
    )
    statistics = conditional_costs(targets, costs)
    candidates = []
    for penalty in PROTOCOL["penalties"]:
        predictions = np.empty_like(targets)
        optimizers = []
        for fold in range(PROTOCOL["folds"]):
            held = folds == fold
            for arm, name in enumerate(ARMS):
                model = fit_logistic(matrix[~held], targets[~held, arm], penalty)
                predictions[held, arm] = probability(matrix[held], model)
                optimizers.append({"fold": fold, "arm": name, **model["optimizer"]})
        losses = _nll(targets, predictions)
        candidates.append(
            {
                "penalty": penalty,
                "cheap_nll": float(losses[0]),
                "strong_nll": float(losses[1]),
                "summed_binary_logloss": float(losses.sum()),
                "oof_probabilities": predictions.tolist(),
                "optimizer_runs": optimizers,
            }
        )
        if progress is not None:
            progress(f"Training paper-fold CV lambda={penalty:g}: summed NLL={losses.sum():.6f}")
    winner = min(candidates, key=lambda item: (item["summed_binary_logloss"], -item["penalty"]))
    models = {name: fit_logistic(matrix, targets[:, arm], winner["penalty"]) for arm, name in enumerate(ARMS)}
    return {
        "training_ids": [row["id"] for row in rows],
        "targets": targets.astype(int).tolist(),
        "folds": folds.tolist(),
        "cv": candidates,
        "penalty": winner["penalty"],
        "models": models,
        "training_cost_proxies": costs,
        "conditional_cost_statistics": statistics,
    }


def fit(
    run,
    baseline,
    training_plan_path,
    training_inputs_path,
    training_rows_path,
    grades_path,
    embeddings,
    cost_path,
    price_path,
    *,
    progress=None,
):
    run = Path(run)
    _unopened(run)
    for name in ("selected-model.json", "frozen.json"):
        if (run / name).exists():
            raise FileExistsError(run / name)
    embeddings = Path(embeddings)
    plan = checked_plan(read(run / "plan.json"))
    _baseline(plan, baseline)
    rows, grades = _training(plan, training_plan_path, training_inputs_path, training_rows_path, grades_path)
    costs, statistics, cost_record = cost_proxy(cost_path, price_path, training_plan_path, grades_path, grades)
    matrix = vectors(rows, embeddings)
    _require(matrix.shape == (len(rows), PROTOCOL["dimensions"]), "Training vectors differ from the pinned dimensions")
    head = fit_head(
        matrix, grades["rows"], [row["paper_group"] for row in plan["populations"]["train"]], costs, progress=progress
    )
    _require(head["conditional_cost_statistics"] == statistics, "Training cost statistics changed while fitting")
    return benefit.seal(
        {
            "format": MODEL_FORMAT,
            "selected_at": benefit.now(),
            "plan": plan,
            "plan_file_sha256": file_hash(run / "plan.json"),
            "head": head,
            "training_grades_sha256": file_hash(grades_path),
            "training_grade_provenance": {key: value for key, value in grades.items() if key != "rows"},
            "training_embeddings_sha256": {
                name: file_hash(embeddings / name) for name in ("plan.json", "vectors.jsonl.gz")
            },
            "cost_proxy_sha256": file_hash(cost_path),
            "price_snapshot_sha256": file_hash(price_path),
            "cost_proxy_provenance": {key: value for key, value in cost_record.items() if key != "rows"},
            "versions": {"numpy": np.__version__, "scipy": PROTOCOL["optimizer"]["scipy"]},
            "boundary": "Only training outcomes used; this selected model must be written before calibration is opened",
        }
    )


def checked_model(run):
    run = Path(run)
    model = benefit.checked(read(run / "selected-model.json"), MODEL_FORMAT)
    plan = checked_plan(model["plan"])
    _require(
        model["plan_file_sha256"] == file_hash(run / "plan.json") and read(run / "plan.json") == plan,
        "Selected outcome model differs from its prepared plan",
    )
    head = model["head"]
    n = len(plan["populations"]["train"])
    targets = np.asarray(head["targets"])
    _require(
        head["training_ids"] == [row["id"] for row in plan["populations"]["train"]]
        and head["folds"] == plan["training_folds"]
        and targets.shape == (n, 2)
        and np.isin(targets, (0, 1)).all()
        and head["conditional_cost_statistics"] == conditional_costs(targets, head["training_cost_proxies"])
        and [item["penalty"] for item in head["cv"]] == PROTOCOL["penalties"],
        "Selected outcome model changes training targets, costs, folds or candidates",
    )
    for candidate in head["cv"]:
        probabilities = np.asarray(candidate["oof_probabilities"], dtype=float)
        _require(
            probabilities.shape == (n, 2)
            and np.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all(),
            "Invalid training-fold probabilities",
        )
        losses = _nll(targets, probabilities)
        _require(
            [candidate["cheap_nll"], candidate["strong_nll"], candidate["summed_binary_logloss"]]
            == [float(losses[0]), float(losses[1]), float(losses.sum())]
            and [(item["fold"], item["arm"]) for item in candidate["optimizer_runs"]]
            == [(fold, arm) for fold in range(PROTOCOL["folds"]) for arm in ARMS]
            and all(item["success"] is True for item in candidate["optimizer_runs"]),
            "Training-only CV losses or optimizer completion differ",
        )
    winner = min(head["cv"], key=lambda item: (item["summed_binary_logloss"], -item["penalty"]))
    _require(
        head["penalty"] == winner["penalty"] and set(head["models"]) == set(ARMS),
        "Selected penalty does not minimize the declared training-only objective",
    )
    for item in head["models"].values():
        probability(np.zeros((1, PROTOCOL["dimensions"])), item)
        _require(item["optimizer"]["success"] is True, "Full training refit did not converge")
    _require(
        all(
            model["training_grade_provenance"].get(name) == plan["training_grade_provenance"][name]
            for name in GRADE_HASHES
        )
        and model["cost_proxy_provenance"]["training_grades_sha256"] == model["training_grades_sha256"]
        and model["cost_proxy_provenance"]["conditional_means"] == head["conditional_cost_statistics"],
        "Selected model grade or cost provenance changed",
    )
    return model


def _selections(rows, scores, credits, rate):
    return {
        str(target): solver_report.select_cutoff(rows, scores, target, router_credits=credits, credit_usd=rate)
        for target in TARGETS
    }


def freeze(run, baseline, original_path):
    run = Path(run)
    _unopened(run)
    if (run / "frozen.json").exists():
        raise FileExistsError(run / "frozen.json")
    model = checked_model(run)
    plan = model["plan"]
    # The selected training-only model already exists on disk before this
    # function opens the original frozen calibration outcomes.
    _require(file_hash(original_path) == plan["original_frozen_sha256"], "Original calibration study changed")
    dataset, original, _, artifacts = benefit._inputs(baseline, original_path)
    _require(
        digest(dataset) == plan["dataset_sha256"] and artifacts == plan["baseline_artifacts_sha256"],
        "Held inputs or vectors changed after preparing the outcome experiment",
    )
    scores, probabilities, expected_costs = {}, {}, {}
    for split in ("calibration", "test"):
        rows = held_rows(plan, dataset, split)
        matrix = vectors(rows, Path(baseline) / ("test-embeddings" if split == "test" else "development-embeddings"))
        probabilities[split] = np.column_stack(
            [probability(matrix, model["head"]["models"][arm]) for arm in ARMS]
        ).tolist()
        values, costs = normalized_scores(probabilities[split], model["head"]["conditional_cost_statistics"])
        scores[split], expected_costs[split] = values.tolist(), costs.tolist()
    selections = _selections(
        original["calibration_rows"],
        scores["calibration"],
        original["embedding_calibration_credits"],
        original["credit_usd"],
    )
    return benefit.seal(
        {
            "format": FROZEN_FORMAT,
            "frozen_at": benefit.now(),
            "plan_file_sha256": file_hash(run / "plan.json"),
            "selected_model_sha256": file_hash(run / "selected-model.json"),
            "scores": scores,
            "probabilities": probabilities,
            "expected_costs_proxy": expected_costs,
            "calibration_rows": original["calibration_rows"],
            "calibration_grades_sha256": original["calibration_grades_sha256"],
            "grading_provenance": grading_provenance(original),
            "embedding_calibration_credits": original["embedding_calibration_credits"],
            "embedding_calibration_credits_sha256": original["embedding_calibration_credits_sha256"],
            "credit_usd": original["credit_usd"],
            "selections": selections,
            "test_access": PROTOCOL["test_boundary"],
        }
    )


def checked_frozen(run):
    run = Path(run)
    model = checked_model(run)
    plan = model["plan"]
    frozen = benefit.checked(read(run / "frozen.json"), FROZEN_FORMAT)
    _require(
        frozen["plan_file_sha256"] == model["plan_file_sha256"]
        and frozen["selected_model_sha256"] == file_hash(run / "selected-model.json")
        and frozen["grading_provenance"] == plan["grading_provenance"],
        "Frozen outcome lifecycle hashes differ",
    )
    for split in ("calibration", "test"):
        _require(
            np.asarray(frozen["probabilities"][split]).shape == (len(plan["populations"][split]), 2),
            "Frozen held probabilities are misaligned",
        )
        scores, costs = normalized_scores(frozen["probabilities"][split], model["head"]["conditional_cost_statistics"])
        _require(
            frozen["scores"][split] == scores.tolist() and frozen["expected_costs_proxy"][split] == costs.tolist(),
            "Frozen cost-normalized scores differ from their probabilities and training costs",
        )
    _require(
        [row["id"] for row in frozen["calibration_rows"]] == [row["id"] for row in plan["populations"]["calibration"]]
        and frozen["selections"]
        == _selections(
            frozen["calibration_rows"],
            frozen["scores"]["calibration"],
            frozen["embedding_calibration_credits"],
            frozen["credit_usd"],
        ),
        "Frozen cutoffs differ from calibration-only cost selection",
    )
    return frozen, model


def authorize_test(run):
    _unopened(run)
    frozen, model = checked_frozen(run)
    return benefit.seal(
        {
            "format": ACCESS_FORMAT,
            "authorized_at": benefit.now(),
            "frozen_sha256": file_hash(Path(run) / "frozen.json"),
            "selected_model_sha256": frozen["selected_model_sha256"],
            "plan_sha256": model["plan_file_sha256"],
            "source_code_sha256": code_hashes(),
            "statement": "All outcome models and cutoffs froze before this receipt authorizes solver test outcome access",
        }
    )


def report(run, grades_path, router_credits_path=None):
    run = Path(run)
    frozen, model = checked_frozen(run)
    access = benefit.checked(read(run / "test-access.json"), ACCESS_FORMAT)
    _require(
        access["frozen_sha256"] == file_hash(run / "frozen.json")
        and access["selected_model_sha256"] == frozen["selected_model_sha256"]
        and access["plan_sha256"] == model["plan_file_sha256"]
        and access["source_code_sha256"] == code_hashes(),
        "Test-access receipt does not bind this frozen outcome model",
    )
    plan = model["plan"]
    ids = [row["id"] for row in plan["populations"]["test"]]
    grades = solver_policies._grade_rows(grades_path, "test", plan["original_solver_plan_sha256"], ids)
    _require(
        grades["policies_frozen_sha256"] == plan["original_frozen_sha256"]
        and grading_provenance(grades) == frozen["grading_provenance"],
        "Test answers differ from the original solver and grading study",
    )
    credits, credit_sha = test_credits(plan, router_credits_path)
    measured = {}
    for target in TARGETS:
        selection = frozen["selections"][str(target)]
        mode = selection["mode"]
        routes = (np.array(frozen["scores"]["test"]) >= selection["cutoff"]).tolist() if mode == "threshold" else mode
        measured[str(target)] = {
            "mode": mode,
            "cutoff": selection["cutoff"],
            "metrics": solver_report.summarize(
                grades["rows"],
                routes,
                router_credits=credits if mode == "threshold" else 0,
                credit_usd=frozen["credit_usd"],
            ),
        }
    return {
        "format": "sie-solver-outcome-report-v1",
        "n": len(ids),
        "ids": ids,
        "training_pool_rows": len(plan["training_pool"]),
        "training_rows": len(model["head"]["training_ids"]),
        "training_papers": len({row["paper_group"] for row in plan["populations"]["train"]}),
        "selected_penalty": model["head"]["penalty"],
        "frozen_sha256": file_hash(run / "frozen.json"),
        "test_access_sha256": file_hash(run / "test-access.json"),
        "test_grades_sha256": file_hash(grades_path),
        "embedding_test_credits_sha256": credit_sha,
        "primary_max_quality_loss_pp": 1,
        "baselines": {
            name: solver_report.summarize(grades["rows"], name, router_credits=0, credit_usd=frozen["credit_usd"])
            for name in ("always_strong", "always_cheap")
        },
        "policies": measured,
        "interpretation": "Two outcome heads and training-only cost proxies; calibration targets are empirical, not test guarantees. Expansion adds within-paper examples. Acquisition/training costs are separate from serving costs; predicted training cost proxies are never reported as measured bills.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "fit", "freeze", "authorize-test", "report"):
        child = commands.add_parser(command)
        child.add_argument("--run", type=Path, required=True)
        if command in ("prepare", "fit", "freeze"):
            child.add_argument("--baseline", type=Path, required=True)
        if command in ("prepare", "freeze"):
            child.add_argument("--original-frozen", type=Path, required=True)
        if command in ("prepare", "fit"):
            for name in ("training-plan", "training-inputs", "training-rows"):
                child.add_argument(f"--{name}", type=Path, required=True)
        if command == "prepare":
            child.add_argument("--eligibility", type=Path, required=True)
            child.add_argument("--expansion-audit", type=Path)
            child.add_argument("--source-review", type=Path)
            child.add_argument("--attest-test-outcomes-unopened", action="store_true", required=True)
        if command == "fit":
            for name in ("training-grades", "training-embeddings", "cost-proxy", "price-snapshot"):
                child.add_argument(f"--{name}", type=Path, required=True)
        if command == "report":
            child.add_argument("--test-grades", type=Path, required=True)
            child.add_argument("--router-credits", type=Path)
    args = parser.parse_args()
    names = {
        "prepare": "plan.json",
        "fit": "selected-model.json",
        "freeze": "frozen.json",
        "authorize-test": "test-access.json",
        "report": "report.json",
    }
    destination = args.run / names[args.command]
    if destination.exists() or (args.command == "prepare" and args.run.exists()):
        raise FileExistsError(destination)
    if args.command == "prepare":
        result = prepare(
            args.baseline,
            args.original_frozen,
            args.training_plan,
            args.training_inputs,
            args.training_rows,
            args.eligibility,
            expansion_audit_path=args.expansion_audit,
            source_review_path=args.source_review,
            attest_test_outcomes_unopened=args.attest_test_outcomes_unopened,
        )
    elif args.command == "fit":
        result = fit(
            args.run,
            args.baseline,
            args.training_plan,
            args.training_inputs,
            args.training_rows,
            args.training_grades,
            args.training_embeddings,
            args.cost_proxy,
            args.price_snapshot,
            progress=lambda message: print(message, flush=True),
        )
    elif args.command == "freeze":
        result = freeze(args.run, args.baseline, args.original_frozen)
    elif args.command == "authorize-test":
        result = authorize_test(args.run)
    else:
        result = report(args.run, args.test_grades, args.router_credits)
    write(destination, result)
    if args.command == "report":
        metrics = result["policies"]["1"]["metrics"]
        print(
            f"Outcome router: accuracy {100 * metrics['answer_accuracy']:.2f}%; gap {metrics['gap_vs_strong_pp']:+.2f}pp"
        )
        if "net_savings_pct" in metrics:
            net = "n/a" if metrics["net_savings_pct"] is None else f"{metrics['net_savings_pct']:.2f}%"
            print(f"Including encoder: {metrics['net_total_cost_usd_per1000']:.4f} USD/1,000; net savings {net}")
        else:
            print("Router overhead unpriced; net savings unknown")
    print(json.dumps({"output": str(destination), "sha256": file_hash(destination)}))


if __name__ == "__main__":
    main()
