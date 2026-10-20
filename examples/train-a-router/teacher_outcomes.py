"""Compare question-only teacher labels with the fixed benchmark-label router.

Training labels have their own provenance: they are never passed off as reviewed
benchmark grades. The learner, vectors, paper folds, token-cost proxies and
calibration procedure are inherited unchanged from the frozen reference study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import benefit
import benefit_outcomes as outcomes
import benefit_training_review
import solver
import solver_policies
import solver_report
import source_review
import teacher_collect
import teacher_continue
import teacher_format
import teacher_recover
import teacher_reference_collect
from benefit_expanded_gold_v2 import outcome_grader_metadata
from data import digest, file_hash, write
from serving import vectors
from teacher_format import grade_reference as grade_teacher_reference

PLAN_FORMAT = "sie-teacher-outcome-plan-v2"
LABEL_FORMAT = "sie-teacher-derived-training-labels-v2"
MODEL_FORMAT = "sie-teacher-outcome-selected-v2"
FROZEN_FORMAT = "sie-teacher-outcome-frozen-v2"
ACCESS_FORMAT = "sie-teacher-outcome-test-access-v2"
DESIGN_FORMAT = "sie-teacher-outcome-comparison-design-v2"
ARMS = outcomes.ARMS
UNUSABLE = {"gold_unparseable", "grading_error", "unit_contract_unresolved"}
COMPARISON = {
    "targets": "Teacher targets use only question + independent teacher final + solver final; no benchmark fallback",
    "eligibility": "Source-reviewed training rows, then teacher-reference self-parse and independently supported requested-unit contract only; solver correctness and benchmark agreement never select rows",
    "matched_control": "Both heads use the same eligible rows, vectors, original paper folds, penalty grid and physical token costs",
    "costs": "Each head recomputes correctness-conditional cost means under its own labels",
    "boundary": "Teacher answers and benchmark references were audited before this amendment; routed solver test outcomes remain unopened",
}
CODE_FILES = sorted(
    set(outcomes.CODE_FILES)
    | {
        "teacher_outcomes.py",
        "teacher_collect.py",
        "teacher_continue.py",
        "teacher_recover.py",
        "teacher_reference_collect.py",
        "benefit_expanded_gold_v2.py",
        "benefit_expanded_gold.py",
        "benefit_data.py",
        "benefit_training_review.py",
        "solver.py",
        "solver_gold.py",
        "solver_answers.py",
    }
)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def code_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in CODE_FILES}


def grading_metadata():
    return outcome_grader_metadata() | {
        "format_normalizer_sha256": file_hash(Path(teacher_format.__file__)),
        "format_protocol": teacher_format.PROTOCOL,
    }


def checked_teacher(directory):
    if read(Path(directory) / "plan.json").get("format") == teacher_reference_collect.FORMAT:
        return teacher_reference_collect.checked_plan(directory)
    if read(Path(directory) / "plan.json").get("format") == teacher_recover.FORMAT:
        return teacher_recover.checked_plan(directory)
    if read(Path(directory) / "plan.json").get("format") == teacher_continue.FORMAT:
        return teacher_continue.checked_plan(directory)
    return teacher_collect.checked_plan(directory)


def unopened(run, *outputs):
    for name in ("test-access.json", "report.json", *outputs):
        if (Path(run) / name).exists():
            raise FileExistsError(Path(run) / name)


def _training_files(plan, training):
    reference = plan["reference_plan"]
    for name, filename in (
        ("plan", "plan.json"),
        ("inputs", "inputs.json"),
        ("rows", "training-rows.json"),
        ("eligibility", "eligibility.json"),
    ):
        require(
            file_hash(Path(training) / filename) == reference["training_artifacts_sha256"][name],
            "Training inputs changed",
        )
    inputs = read(Path(training) / "inputs.json")["train"]
    require(
        [{"id": row["id"], "request_sha256": digest(row["request"])} for row in inputs]
        == [{key: row[key] for key in ("id", "request_sha256")} for row in reference["source_populations"]["train"]],
        "Training inputs differ from the complete original acquisition population",
    )
    ids = {row["id"] for row in reference["populations"]["train"]}
    rows = [row for row in inputs if row["id"] in ids]
    require([row["id"] for row in rows] == plan["design"]["training_ids"], "Reviewed training order changed")
    return rows


def _grade_path(training, path):
    return Path(training) / "grades-train.json" if path is None else Path(path)


def _cost_path(training, path):
    return Path(training) / "cost-proxy.json" if path is None else Path(path)


def prepare(
    reference_run,
    teacher,
    design_path,
    training,
    embeddings,
    prices,
    *,
    attest_test_outcomes_unopened,
    source_review_path=None,
    training_grades=None,
    cost_proxy=None,
):
    require(attest_test_outcomes_unopened is True, "Explicit unopened-test assertion is required")
    reference_run, teacher, training, embeddings = map(Path, (reference_run, teacher, training, embeddings))
    unopened(reference_run)
    reference = outcomes.checked_plan(read(reference_run / "plan.json"))
    acquisition, _ = checked_teacher(teacher)
    require(
        reference["source_review"] is not None and source_review_path is not None,
        "Matched comparison requires the independently reviewed source manifest",
    )
    review = source_review.checked_review(read(source_review_path))
    review_metadata = source_review.metadata(source_review_path)
    require(
        all(reference["source_review"][key] == review_metadata[key] for key in source_review.METADATA_KEYS),
        "Teacher and benchmark source review differ",
    )
    for split, rows in reference["source_populations"].items():
        require(
            source_review.eligible_rows(rows, split, review) == reference["populations"][split],
            "Teacher and benchmark source eligibility differ",
        )
    design = benefit.seal(
        {
            "format": DESIGN_FORMAT,
            "comparison": COMPARISON,
            "benchmark_control_plan_sha256": file_hash(reference_run / "plan.json"),
            "learner_protocol": outcomes.PROTOCOL,
            "learner_source_code_sha256": outcomes.code_hashes(),
            "teacher_grading": grading_metadata(),
            "teacher_protocol": acquisition["settings"],
            "teacher_acquisition_plan_sha256": file_hash(teacher / "plan.json"),
            "teacher_answers_inspected_for_source_audit": True,
            "solver_test_outcomes_accessed": False,
            "training_ids": [row["id"] for row in reference["populations"]["train"]],
            "training_folds": reference["training_folds"],
            "training_paper_groups": [row["paper_group"] for row in reference["populations"]["train"]],
            "original_frozen_sha256": reference["original_frozen_sha256"],
            "calibration_grading_provenance": reference["grading_provenance"],
            "source_review": review_metadata,
        }
    )
    if design_path is not None:
        require(
            benefit.checked(read(design_path), DESIGN_FORMAT) == design,
            "External design differs from the truthful matched-comparison amendment",
        )
    grades_path, costs_path = _grade_path(training, training_grades), _cost_path(training, cost_proxy)
    plan = {
        "format": PLAN_FORMAT,
        "created_at": benefit.now(),
        "design": design,
        "design_file_sha256": None if design_path is None else file_hash(design_path),
        "reference_plan": reference,
        "reference_plan_file_sha256": file_hash(reference_run / "plan.json"),
        "teacher_plan_sha256": file_hash(teacher / "plan.json"),
        "teacher_model": acquisition["model"],
        "teacher_settings": acquisition["settings"],
        "source_review": review,
        "source_review_json": Path(source_review_path).read_bytes().decode("utf-8"),
        "source_review_metadata": review_metadata,
        "training_embeddings_sha256": {
            name: file_hash(embeddings / name) for name in ("plan.json", "vectors.jsonl.gz")
        },
        "benchmark_training_grades_sha256": file_hash(grades_path),
        "cost_proxy_sha256": file_hash(costs_path),
        "price_snapshot_sha256": file_hash(prices),
        "source_code_sha256": code_hashes(),
        "outcome_grading": grading_metadata(),
        "label_source": "independent_question_only_teacher",
        "coverage_rule": COMPARISON["eligibility"],
        "attest_test_outcomes_unopened": True,
        "limitations": "This is an explicit post-source-audit amendment, not a claim that teacher answers were unseen. Receipts cannot prove nobody read another copy of routed test outcomes.",
    }
    inputs = _training_files(plan, training)
    teacher_inputs = {row["id"]: row for row in acquisition["inputs"]}
    require(
        len(teacher_inputs) == len(acquisition["inputs"])
        and all(teacher_inputs.get(row["id"]) == row for row in inputs),
        "Teacher questions differ from eligible training inputs",
    )
    return benefit.seal(plan)


def checked_plan(run):
    plan = benefit.checked(read(Path(run) / "plan.json"), PLAN_FORMAT)
    reference = outcomes.checked_plan(plan["reference_plan"])
    design = benefit.checked(plan["design"], DESIGN_FORMAT)
    review = source_review.checked_review(plan["source_review"])
    require(
        plan["source_code_sha256"] == code_hashes()
        and plan["outcome_grading"] == grading_metadata()
        and plan["label_source"] == "independent_question_only_teacher"
        and plan["attest_test_outcomes_unopened"] is True
        and design["comparison"] == COMPARISON
        and design["learner_protocol"] == outcomes.PROTOCOL
        and design["learner_source_code_sha256"] == outcomes.code_hashes()
        and design["teacher_grading"] == plan["outcome_grading"]
        and design["teacher_protocol"] == plan["teacher_settings"]
        and design["teacher_acquisition_plan_sha256"] == plan["teacher_plan_sha256"]
        and design["teacher_answers_inspected_for_source_audit"] is True
        and design["solver_test_outcomes_accessed"] is False
        and design["training_ids"] == [row["id"] for row in reference["populations"]["train"]]
        and design["training_folds"] == reference["training_folds"]
        and design["training_paper_groups"] == [row["paper_group"] for row in reference["populations"]["train"]]
        and design["calibration_grading_provenance"] == reference["grading_provenance"]
        and design["original_frozen_sha256"] == reference["original_frozen_sha256"]
        and design["benchmark_control_plan_sha256"] == plan["reference_plan_file_sha256"]
        and design["source_review"] == plan["source_review_metadata"]
        and json.loads(plan["source_review_json"]) == review
        and hashlib.sha256(plan["source_review_json"].encode("utf-8")).hexdigest()
        == plan["source_review_metadata"]["source_review_sha256"]
        and all(
            plan["source_review_metadata"][key] == reference["source_review"][key]
            for key in source_review.METADATA_KEYS
        ),
        "Teacher experiment source, grading or population changed",
    )
    for split, rows in reference["source_populations"].items():
        require(
            source_review.eligible_rows(rows, split, review) == reference["populations"][split],
            "Reviewed teacher comparison population changed",
        )
    return plan


def label_pair(question, teacher_final, answers):
    """Targets have no benchmark answers or IDs; eligibility is reference-only."""
    reference_contract = teacher_format.normalize_final(question, teacher_final)
    self_grade = grade_teacher_reference(question, teacher_final, r"\boxed{" + teacher_final + "}")
    self_grade = self_grade | {"reference_unit_contract": reference_contract}
    grades = {arm: grade_teacher_reference(question, teacher_final, answers[arm]) for arm in ARMS}
    return reference_usable(self_grade), self_grade, grades


def reference_usable(self_grade):
    """One reference-only eligibility rule, shared by construction and replay."""
    contract = self_grade.get("reference_unit_contract")
    require(isinstance(contract, dict) and "unit_status" in contract, "Missing independent reference-unit contract")
    return bool(
        self_grade["correct"]
        and self_grade["status"] == "graded"
        and contract["unit_status"] not in {"unknown_requested_unit", "incompatible"}
    )


def audit_teacher_reference(item, teacher_final):
    """Audit accuracy without using benchmark values to construct teacher labels.

    Historic symbolic rules cover specific solver finals, not every possible
    teacher notation. Their independently proved canonical literal is covered;
    other unsupported forms remain explicitly unresolved in this audit.
    """
    if item["semantic_kind"] == "reviewed_symbolic_training":
        rule = benefit_training_review.REVIEW[item["id"]]
        require(
            item["original_reference"] == rule["original"]
            and item["reference"] == rule["canonical"]
            and bool(rule["proof"]),
            "Symbolic source proof changed",
        )
        final = teacher_final.strip()
        if final == rule["canonical"] or final in rule["answers"]:
            return {
                "correct": True if final == rule["canonical"] else rule["answers"][final],
                "status": "graded",
                "source_proof": rule["proof"],
                "audit_only": True,
            }
        return {"correct": False, "status": "grading_error", "reason": "Teacher notation outside symbolic source proof"}
    contract = teacher_format.normalize_final(item["question"], teacher_final)
    if contract["unit_status"] in {"unknown_requested_unit", "incompatible"}:
        return {"correct": False, "status": "unit_contract_unresolved", "reference_unit_contract": contract}
    try:
        result = source_review._grade_reviewed_row(item, r"\boxed{" + teacher_final + "}")
    except (ValueError, TypeError, TimeoutError, ArithmeticError) as error:
        return {"correct": False, "status": "grading_error", "error_type": type(error).__name__, "error": str(error)}
    if result["status"] in {"prediction_unparseable", "no_final_answer"}:
        return result | {"status": "grading_error", "automatic_status": result["status"]}
    return result


def teacher_accuracy_summary(rows):
    gradeable = [row for row in rows if row["teacher_accuracy_gradeable"]]
    confirmed = sum(bool(row["teacher_correct"]) for row in gradeable)
    return {
        "teacher_correct": confirmed,
        "teacher_accuracy_denominator": len(gradeable),
        "teacher_accuracy_population": len(rows),
        "teacher_accuracy_unresolved_ids": [row["id"] for row in rows if not row["teacher_accuracy_gradeable"]],
        "teacher_accuracy_coverage": len(gradeable) / len(rows) if rows else None,
        "teacher_confirmed_correct_fraction_lower_bound": confirmed / len(rows) if rows else None,
    }


def require_comparable_targets(identity, usable, measured):
    require(
        not usable or all(value["status"] not in UNUSABLE for value in measured.values()),
        f"Unresolved teacher comparison for {identity}; retain the row and adjudicate the grading contract, never infer an incorrect label or change eligibility",
    )


def checked_teacher_coverage(acquisition, collection, training_rows, review):
    """An exhausted call may be outside the independently eligible population."""
    all_ids = {row["id"] for row in acquisition["inputs"]}
    completed = set(collection["record_hashes"])
    missing = sorted(all_ids - completed)
    excluded = {row["id"] for row in review["records"] if row["decision"] == "exclude"}
    require(
        len(all_ids) == len(acquisition["inputs"])
        and completed.issubset(all_ids)
        and collection.get("n") == len(all_ids)
        and collection.get("returned") == collection.get("complete") == len(completed)
        and collection.get("status") == ("incomplete" if missing else "complete")
        and collection.get("missing_ids", []) == missing
        and set(missing).issubset(excluded)
        and {row["id"] for row in training_rows}.issubset(completed),
        "Every source-eligible training row requires a completed teacher answer; missing acquisition IDs must be explicitly source-excluded",
    )
    return {"n": len(all_ids), "completed": len(completed), "missing_ids": missing, "status": collection["status"]}


def grade(run, teacher, training, *, training_grades=None):
    run, teacher, training = map(Path, (run, teacher, training))
    unopened(run, "teacher-labels.json", "teacher-audit.json", "selected-model.json", "frozen.json")
    plan = checked_plan(run)
    rows = _training_files(plan, training)
    acquisition, _ = checked_teacher(teacher)
    collection = read(teacher / "collection.json")
    require(file_hash(teacher / "plan.json") == plan["teacher_plan_sha256"], "Teacher acquisition changed")
    require(collection["plan_sha256"] == plan["teacher_plan_sha256"], "Teacher collection belongs to another plan")
    coverage = checked_teacher_coverage(acquisition, collection, rows, plan["source_review"])
    source_plan, _, settings = solver.checked_inputs(training)
    require(
        source_plan["settings_sha256"] == plan["reference_plan"]["solver_settings_sha256"], "Solver settings changed"
    )
    grades_path = _grade_path(training, training_grades)
    require(file_hash(grades_path) == plan["benchmark_training_grades_sha256"], "Reviewed training grades changed")
    _, benchmark = outcomes._training(
        plan["reference_plan"],
        training / "plan.json",
        training / "inputs.json",
        training / "training-rows.json",
        grades_path,
    )
    gold = read(training / "gold-for-scoring-only.json")
    require(digest(gold) == source_plan["gold_sha256"], "Benchmark answers changed")
    review_rows = {row["id"]: row for row in plan["source_review"]["records"]}
    labels, audit_rows, unavailable, matched = [], [], [], []
    for row, previous in zip(rows, benchmark["rows"], strict=True):
        identity = row["id"]
        item = review_rows[identity]
        question = row["request"].partition("\nProblem: ")[2].strip()
        require(
            item["decision"] == "retain"
            and item["question"].strip() == question
            and item["original_reference"] == gold[identity],
            "Reviewed source question or answer changed",
        )
        teacher_path = teacher / "records" / f"{identity}.json"
        require(file_hash(teacher_path) == collection["record_hashes"][identity], "Teacher record changed")
        teacher_record = read(teacher_path)
        teacher_continue.check_record(teacher_record, row)
        records, hashes = {}, {}
        for arm in ARMS:
            path = training / "answers-train" / f"{identity}-{arm}.json"
            hashes[arm] = file_hash(path)
            require(hashes[arm] == previous[f"{arm}_record_sha256"], "Paired training record changed")
            records[arm] = read(path)
            solver.validate_record(records[arm], row, arm, settings)
            require(records[arm]["status"] == "complete", "Incomplete paired solver answer")
        usable, self_grade, measured = label_pair(
            question, teacher_record["answer"], {arm: records[arm]["answer"] for arm in ARMS}
        )
        require_comparable_targets(identity, usable, measured)
        result = {
            "id": identity,
            "teacher_record_sha256": file_hash(teacher_path),
            "usable": usable,
            "teacher_self_grade": self_grade,
        }
        for arm in ARMS:
            result.update(
                {
                    f"{arm}_correct": measured[arm]["correct"],
                    f"{arm}_grade": measured[arm],
                    f"{arm}_record_sha256": hashes[arm],
                }
            )
        labels.append(result)
        if not usable:
            unavailable.append(identity)
        # The source manifest was checked once above. These benchmark values
        # never enter label_pair or its eligibility mask.
        reference = {arm: source_review._grade_reviewed_row(item, records[arm]["answer"]) for arm in ARMS}
        require(
            all(
                reference[arm]["status"] not in UNUSABLE and reference[arm]["correct"] == previous[f"{arm}_correct"]
                for arm in ARMS
            ),
            "Recomputed benchmark outcomes differ from the reviewed training grades",
        )
        matched.append(
            {
                "id": identity,
                **{f"{arm}_correct": reference[arm]["correct"] for arm in ARMS},
                **{f"{arm}_record_sha256": hashes[arm] for arm in ARMS},
            }
        )
        teacher_grade = audit_teacher_reference(item, teacher_record["answer"])
        audit_rows.append(
            {
                "id": identity,
                "teacher_answer": teacher_record["answer"],
                "original_source_gold": gold[identity],
                "teacher_correct": teacher_grade["correct"],
                "teacher_accuracy_gradeable": teacher_grade["status"] not in UNUSABLE,
                "teacher_grade": teacher_grade,
                "teacher_reference_usable": usable,
                "arms": {
                    arm: {
                        "teacher_label": measured[arm]["correct"],
                        "reference_correct": reference[arm]["correct"],
                        "reference_grade": reference[arm],
                        "agreement": usable and measured[arm]["correct"] == reference[arm]["correct"],
                    }
                    for arm in ARMS
                },
                "signed_benefit_agreement": usable
                and int(measured["strong"]["correct"]) - int(measured["cheap"]["correct"])
                == int(reference["strong"]["correct"]) - int(reference["cheap"]["correct"]),
            }
        )
    eligible_ids = [row["id"] for row in labels if row["usable"]]
    label_document = benefit.seal(
        {
            "format": LABEL_FORMAT,
            "status": "complete",
            "n": len(rows),
            "usable_rows": len(eligible_ids),
            "eligible_ids": eligible_ids,
            "unusable_ids": unavailable,
            "eligibility_rule": COMPARISON["eligibility"],
            "plan_sha256": file_hash(run / "plan.json"),
            "teacher_plan_sha256": file_hash(teacher / "plan.json"),
            "teacher_collection_sha256": file_hash(teacher / "collection.json"),
            "teacher_acquisition_coverage": coverage,
            "paired_training_plan_sha256": file_hash(training / "plan.json"),
            "paired_collection_sha256": benchmark["collection_summary_sha256"],
            "benchmark_training_grades_sha256": file_hash(grades_path),
            "reference_source": "independent_question_only_teacher",
            "teacher_model": acquisition["model"],
            "outcome_grading": grading_metadata(),
            "rows": labels,
            "matched_benchmark_rows": matched,
            "benchmark_values_used_for_teacher_targets": False,
        }
    )
    audit = {
        "format": "sie-teacher-reference-extension-audit-v2",
        "n": len(rows),
        **teacher_accuracy_summary(audit_rows),
        "reference_coverage": label_document["usable_rows"],
        "agreement_denominator": len(eligible_ids),
        "matched_training_ids": eligible_ids,
        "unusable_teacher_reference_ids": unavailable,
        "agreement_by_arm": {arm: sum(row["arms"][arm]["agreement"] for row in audit_rows) for arm in ARMS},
        "signed_benefit_agreement": sum(row["signed_benefit_agreement"] for row in audit_rows),
        "teacher_plan_sha256": file_hash(teacher / "plan.json"),
        "teacher_collection_sha256": file_hash(teacher / "collection.json"),
        "training_plan_sha256": file_hash(training / "plan.json"),
        "training_grades_sha256": file_hash(grades_path),
        "original_source_gold_sha256": file_hash(training / "gold-for-scoring-only.json"),
        "teacher_labels_content_sha256": label_document["sha256"],
        "outcome_grading": grading_metadata(),
        "source_review": plan["source_review_metadata"],
        "full_teacher_acquisition_n": collection["n"],
        "teacher_acquisition_coverage": coverage,
        "list_price_estimate_usd": collection["list_price_estimate_usd"],
        "conservative_budget_including_unknown_usd": collection["conservative_list_price_usd"],
        "prior_unknown_exposure_usd": collection.get("prior_unknown_exposure_usd", "0"),
        "rows": audit_rows,
        "interpretation": "Training-sample agreement after source review. Teacher accuracy uses independently gradeable rows and reports coverage; unsupported notation is not demonstrated mathematical error. Both learners use the same teacher-usable subset. Acquisition costs include the full source audit. Benchmark references are never teacher-target fallbacks.",
    }
    return label_document, audit


def checked_labels(run, plan):
    labels = benefit.checked(read(Path(run) / "teacher-labels.json"), LABEL_FORMAT)
    rows, ids = labels["rows"], plan["design"]["training_ids"]
    eligible = [row["id"] for row in rows if row["usable"]]
    unavailable = [row["id"] for row in rows if not row["usable"]]
    require(
        labels["status"] == "complete"
        and labels["n"] == len(ids)
        and labels["usable_rows"] == len(eligible) > 0
        and labels["eligible_ids"] == eligible
        and labels["unusable_ids"] == unavailable
        and labels["eligibility_rule"] == COMPARISON["eligibility"]
        and labels["plan_sha256"] == file_hash(Path(run) / "plan.json")
        and labels["teacher_plan_sha256"] == plan["teacher_plan_sha256"]
        and labels["benchmark_training_grades_sha256"] == plan["benchmark_training_grades_sha256"]
        and labels["reference_source"] == "independent_question_only_teacher"
        and labels["benchmark_values_used_for_teacher_targets"] is False
        and labels["outcome_grading"] == plan["outcome_grading"]
        and [row["id"] for row in rows] == ids
        and [row["id"] for row in labels["matched_benchmark_rows"]] == ids,
        "Teacher labels must retain the full reviewed population and one explicit matched eligibility mask",
    )
    for teacher, benchmark in zip(rows, labels["matched_benchmark_rows"], strict=True):
        self_grade = teacher["teacher_self_grade"]
        measured = {arm: teacher[f"{arm}_grade"] for arm in ARMS}
        require_comparable_targets(teacher["id"], teacher["usable"], measured)
        require(
            type(teacher["usable"]) is bool
            and teacher["usable"] == reference_usable(self_grade)
            and all(type(row[f"{arm}_correct"]) is bool for row in (teacher, benchmark) for arm in ARMS)
            and all(teacher[f"{arm}_correct"] == measured[arm]["correct"] for arm in ARMS)
            and all(teacher[f"{arm}_record_sha256"] == benchmark[f"{arm}_record_sha256"] for arm in ARMS),
            "Eligibility depends only on the teacher reference; both label sources bind the same calls",
        )
    return labels


def _matched(labels, design):
    mask = [row["usable"] for row in labels["rows"]]
    return {
        "mask": mask,
        "teacher": [row for row, keep in zip(labels["rows"], mask, strict=True) if keep],
        "benchmark": [row for row, keep in zip(labels["matched_benchmark_rows"], mask, strict=True) if keep],
        "groups": [value for value, keep in zip(design["training_paper_groups"], mask, strict=True) if keep],
        "folds": [value for value, keep in zip(design["training_folds"], mask, strict=True) if keep],
    }


def fit(run, training, embeddings, prices, *, training_grades=None, cost_proxy=None, progress=None):
    run, training, embeddings = map(Path, (run, training, embeddings))
    unopened(run, "selected-model.json", "frozen.json")
    plan = checked_plan(run)
    labels = checked_labels(run, plan)
    rows = _training_files(plan, training)
    grades_path, costs_path = _grade_path(training, training_grades), _cost_path(training, cost_proxy)
    require(
        {name: file_hash(embeddings / name) for name in ("plan.json", "vectors.jsonl.gz")}
        == plan["training_embeddings_sha256"]
        and file_hash(costs_path) == plan["cost_proxy_sha256"]
        and file_hash(grades_path) == plan["benchmark_training_grades_sha256"]
        and file_hash(prices) == plan["price_snapshot_sha256"],
        "Training vectors, grades, costs or prices changed",
    )
    _, benchmark = outcomes._training(
        plan["reference_plan"],
        training / "plan.json",
        training / "inputs.json",
        training / "training-rows.json",
        grades_path,
    )
    costs, _, _ = outcomes.cost_proxy(costs_path, prices, training / "plan.json", grades_path, benchmark)
    require(
        all(
            row["id"] == previous["id"]
            and all(
                row[f"{arm}_record_sha256"] == previous[f"{arm}_record_sha256"]
                and row[f"{arm}_correct"] == previous[f"{arm}_correct"]
                for arm in ARMS
            )
            for row, previous in zip(labels["matched_benchmark_rows"], benchmark["rows"], strict=True)
        ),
        "Matched control labels or physical solver calls changed",
    )
    matched = _matched(labels, plan["design"])
    matrix = vectors(rows, embeddings)
    require(matrix.shape == (len(rows), outcomes.PROTOCOL["dimensions"]), "Teacher feature dimensions changed")
    matrix = matrix[np.array(matched["mask"], dtype=bool)]
    costs = [value for value, keep in zip(costs, matched["mask"], strict=True) if keep]
    heads = {}
    for name, target in (("head", "teacher"), ("matched_benchmark_control", "benchmark")):
        if progress is not None:
            progress(f"Fitting {target} labels on the same {len(matrix)} training rows")
        heads[name] = outcomes.fit_head(
            matrix, matched[target], matched["groups"], costs, progress=progress, fold_assignments=matched["folds"]
        )
        require(heads[name]["folds"] == matched["folds"], "Original paper folds changed")
    return benefit.seal(
        {
            "format": MODEL_FORMAT,
            "selected_at": benefit.now(),
            "plan": plan,
            "plan_file_sha256": file_hash(run / "plan.json"),
            **heads,
            "teacher_labels_sha256": file_hash(run / "teacher-labels.json"),
            "training_reference_source": "independent_question_only_teacher",
            "teacher_model": plan["teacher_model"],
            "training_rows": len(matrix),
            "training_ids": labels["eligible_ids"],
            "versions": {"numpy": np.__version__, "scipy": outcomes.PROTOCOL["optimizer"]["scipy"]},
            "boundary": "Both heads fit the same teacher-usable subset and original paper folds. Only training labels and label-conditioned cost means differ; calibration follows separately.",
        }
    )


def _check_head(head, rows, folds):
    targets = np.array([[int(row[f"{arm}_correct"]) for arm in ARMS] for row in rows])
    require(
        head["targets"] == targets.tolist()
        and head["training_ids"] == [row["id"] for row in rows]
        and head["folds"] == folds,
        "Selected model labels, eligible rows or original folds changed",
    )
    require(
        head["conditional_cost_statistics"] == outcomes.conditional_costs(targets, head["training_cost_proxies"]),
        "Label-conditioned cost means changed",
    )
    require([row["penalty"] for row in head["cv"]] == outcomes.PROTOCOL["penalties"], "Penalty grid changed")
    for candidate in head["cv"]:
        probabilities = np.array(candidate["oof_probabilities"])
        require(
            probabilities.shape == (len(rows), 2)
            and np.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all(),
            "Invalid paper-fold probabilities",
        )
        losses = outcomes._nll(targets, probabilities)
        require(
            [candidate["cheap_nll"], candidate["strong_nll"], candidate["summed_binary_logloss"]]
            == [float(losses[0]), float(losses[1]), float(losses.sum())]
            and [(item["fold"], item["arm"]) for item in candidate["optimizer_runs"]]
            == [(fold, arm) for fold in range(outcomes.PROTOCOL["folds"]) for arm in ARMS]
            and all(item["success"] is True for item in candidate["optimizer_runs"]),
            "CV scores or optimization changed",
        )
    winner = min(head["cv"], key=lambda row: (row["summed_binary_logloss"], -row["penalty"]))
    require(head["penalty"] == winner["penalty"] and set(head["models"]) == set(ARMS), "Model selection changed")
    for model in head["models"].values():
        outcomes.probability(np.zeros((1, outcomes.PROTOCOL["dimensions"])), model)
        require(model["optimizer"]["success"] is True, "Full refit failed")


def checked_model(run):
    run = Path(run)
    model = benefit.checked(read(run / "selected-model.json"), MODEL_FORMAT)
    plan = checked_plan(run)
    labels = checked_labels(run, plan)
    require(
        model["plan"] == plan
        and model["plan_file_sha256"] == file_hash(run / "plan.json")
        and model["teacher_labels_sha256"] == file_hash(run / "teacher-labels.json")
        and model["training_rows"] == labels["usable_rows"]
        and model["training_ids"] == labels["eligible_ids"],
        "Matched teacher model lifecycle bindings changed",
    )
    matched = _matched(labels, plan["design"])
    for name, target in (("head", "teacher"), ("matched_benchmark_control", "benchmark")):
        _check_head(model[name], matched[target], matched["folds"])
    require(
        model["head"]["training_cost_proxies"] == model["matched_benchmark_control"]["training_cost_proxies"],
        "Matched heads must use exactly the same physical token costs",
    )
    return model


def _held_scores(head, matrices, original):
    scores, probabilities, expected = {}, {}, {}
    for split, matrix in matrices.items():
        probabilities[split] = np.column_stack(
            [outcomes.probability(matrix, head["models"][arm]) for arm in ARMS]
        ).tolist()
        values, costs = outcomes.normalized_scores(probabilities[split], head["conditional_cost_statistics"])
        scores[split], expected[split] = values.tolist(), costs.tolist()
    return {
        "scores": scores,
        "probabilities": probabilities,
        "expected_costs_proxy": expected,
        "selections": outcomes._selections(
            original["calibration_rows"],
            scores["calibration"],
            original["embedding_calibration_credits"],
            original["credit_usd"],
        ),
    }


def freeze(run, baseline, original_path):
    run = Path(run)
    unopened(run, "frozen.json")
    model = checked_model(run)
    plan, reference = model["plan"], model["plan"]["reference_plan"]
    require(file_hash(original_path) == reference["original_frozen_sha256"], "Calibration study changed")
    dataset, original, _, artifacts = benefit._inputs(baseline, original_path)
    require(
        artifacts == reference["baseline_artifacts_sha256"]
        and outcomes.grading_provenance(original) == reference["grading_provenance"],
        "Calibration inputs or independent v2b grading changed",
    )
    matrices = {
        split: vectors(
            outcomes.held_rows(reference, dataset, split),
            Path(baseline) / ("test-embeddings" if split == "test" else "development-embeddings"),
        )
        for split in ("calibration", "test")
    }
    teacher_scores = _held_scores(model["head"], matrices, original)
    control_scores = _held_scores(model["matched_benchmark_control"], matrices, original)
    return benefit.seal(
        {
            "format": FROZEN_FORMAT,
            "frozen_at": benefit.now(),
            "plan_file_sha256": file_hash(run / "plan.json"),
            "selected_model_sha256": file_hash(run / "selected-model.json"),
            **teacher_scores,
            "matched_benchmark_control": control_scores,
            "calibration_rows": original["calibration_rows"],
            "calibration_grades_sha256": original["calibration_grades_sha256"],
            "grading_provenance": outcomes.grading_provenance(original),
            "embedding_calibration_credits": original["embedding_calibration_credits"],
            "embedding_calibration_credits_sha256": original["embedding_calibration_credits_sha256"],
            "credit_usd": original["credit_usd"],
            "test_access": outcomes.PROTOCOL["test_boundary"],
            "teacher_model": plan["teacher_model"],
        }
    )


def checked_frozen(run):
    run = Path(run)
    model = checked_model(run)
    reference = model["plan"]["reference_plan"]
    frozen = benefit.checked(read(run / "frozen.json"), FROZEN_FORMAT)
    require(
        frozen["plan_file_sha256"] == model["plan_file_sha256"]
        and frozen["selected_model_sha256"] == file_hash(run / "selected-model.json")
        and frozen["grading_provenance"] == reference["grading_provenance"]
        and [row["id"] for row in frozen["calibration_rows"]]
        == [row["id"] for row in reference["populations"]["calibration"]],
        "Teacher freeze bindings changed",
    )
    for head, result in (
        (model["head"], frozen),
        (model["matched_benchmark_control"], frozen["matched_benchmark_control"]),
    ):
        for split in ("calibration", "test"):
            require(
                np.asarray(result["probabilities"][split]).shape == (len(reference["populations"][split]), 2),
                "Matched held scores are misaligned",
            )
            scores, costs = outcomes.normalized_scores(
                result["probabilities"][split], head["conditional_cost_statistics"]
            )
            require(
                result["scores"][split] == scores.tolist() and result["expected_costs_proxy"][split] == costs.tolist(),
                "Matched score formula changed",
            )
        require(
            result["selections"]
            == outcomes._selections(
                frozen["calibration_rows"],
                result["scores"]["calibration"],
                frozen["embedding_calibration_credits"],
                frozen["credit_usd"],
            ),
            "Matched calibration cutoffs changed",
        )
    return frozen, model


def authorize_test(run):
    unopened(run)
    frozen, model = checked_frozen(run)
    return benefit.seal(
        {
            "format": ACCESS_FORMAT,
            "authorized_at": benefit.now(),
            "frozen_sha256": file_hash(Path(run) / "frozen.json"),
            "selected_model_sha256": frozen["selected_model_sha256"],
            "plan_sha256": model["plan_file_sha256"],
            "source_code_sha256": code_hashes(),
            "statement": "Both matched models and calibration cutoffs froze before routed solver test outcome access",
        }
    )


def report(run, grades_path, router_credits_path=None):
    run = Path(run)
    frozen, model = checked_frozen(run)
    access = benefit.checked(read(run / "test-access.json"), ACCESS_FORMAT)
    require(
        access["frozen_sha256"] == file_hash(run / "frozen.json")
        and access["selected_model_sha256"] == frozen["selected_model_sha256"]
        and access["plan_sha256"] == model["plan_file_sha256"]
        and access["source_code_sha256"] == code_hashes(),
        "Teacher test-access receipt changed",
    )
    reference = model["plan"]["reference_plan"]
    ids = [row["id"] for row in reference["populations"]["test"]]
    grades = solver_policies._grade_rows(grades_path, "test", reference["original_solver_plan_sha256"], ids)
    require(
        grades["policies_frozen_sha256"] == reference["original_frozen_sha256"]
        and outcomes.grading_provenance(grades) == frozen["grading_provenance"],
        "Independent benchmark test grading changed",
    )
    credits, credits_sha = outcomes.test_credits(reference, router_credits_path)

    def policies_for(result):
        policies = {}
        for target in outcomes.TARGETS:
            selection = result["selections"][str(target)]
            mode = selection["mode"]
            routes = (
                (np.array(result["scores"]["test"]) >= selection["cutoff"]).tolist() if mode == "threshold" else mode
            )
            policies[str(target)] = {
                "mode": mode,
                "cutoff": selection["cutoff"],
                "metrics": solver_report.summarize(
                    grades["rows"],
                    routes,
                    router_credits=credits if mode == "threshold" else 0,
                    credit_usd=frozen["credit_usd"],
                ),
            }
        return policies

    baselines = {
        name: solver_report.summarize(grades["rows"], name, router_credits=0, credit_usd=frozen["credit_usd"])
        for name in ("always_strong", "always_cheap")
    }

    return {
        "format": "sie-solver-outcome-teacher-report-v1",
        "n": len(ids),
        "ids": ids,
        "teacher_model": model["plan"]["teacher_model"],
        "training_rows": model["training_rows"],
        "training_ids": model["training_ids"],
        "source_eligible_training_rows": len(model["plan"]["design"]["training_ids"]),
        "training_reference_source": "independent_question_only_teacher",
        "selected_penalty": model["head"]["penalty"],
        "frozen_sha256": file_hash(run / "frozen.json"),
        "test_access_sha256": file_hash(run / "test-access.json"),
        "test_grades_sha256": file_hash(grades_path),
        "embedding_test_credits_sha256": credits_sha,
        "teacher_labels_sha256": model["teacher_labels_sha256"],
        "primary_max_quality_loss_pp": 1,
        "baselines": baselines,
        "policies": policies_for(frozen),
        "matched_benchmark_control": {
            "n": len(ids),
            "ids": ids,
            "training_rows": model["training_rows"],
            "training_ids": model["training_ids"],
            "training_reference_source": "independently_reviewed_benchmark",
            "selected_penalty": model["matched_benchmark_control"]["penalty"],
            "test_grades_sha256": file_hash(grades_path),
            "primary_max_quality_loss_pp": 1,
            "embedding_test_credits_sha256": credits_sha,
            "baselines": baselines,
            "policies": policies_for(frozen["matched_benchmark_control"]),
        },
        "interpretation": "Both heads train on the same teacher-usable subset; only training labels change. Calibration and test use reviewed benchmark references. Teacher/reference-audit costs are separate from serving bills; calibration quality targets are not test guarantees.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "grade", "fit", "freeze", "authorize-test", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        arguments = {
            "prepare": ("reference", "teacher", "training", "embeddings", "prices", "source-review"),
            "grade": ("teacher", "training"),
            "fit": ("training", "embeddings", "prices"),
            "freeze": ("baseline", "original"),
            "report": ("test-grades",),
        }.get(name, ())
        for argument in arguments:
            command.add_argument(f"--{argument}", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--design", type=Path)
            command.add_argument("--attest-test-outcomes-unopened", action="store_true")
        if name in ("prepare", "grade", "fit"):
            command.add_argument("--training-grades", type=Path)
        if name in ("prepare", "fit"):
            command.add_argument("--cost-proxy", type=Path)
        if name == "report":
            command.add_argument("--router-credits", type=Path)
    args = parser.parse_args()
    outputs = {
        "prepare": "plan.json",
        "grade": "teacher-labels.json",
        "fit": "selected-model.json",
        "freeze": "frozen.json",
        "authorize-test": "test-access.json",
        "report": "report.json",
    }
    path = args.run / outputs[args.command]
    if path.exists() or args.command == "prepare" and args.run.exists():
        raise FileExistsError(path)
    if args.command == "prepare":
        value = prepare(
            args.reference,
            args.teacher,
            args.design,
            args.training,
            args.embeddings,
            args.prices,
            attest_test_outcomes_unopened=args.attest_test_outcomes_unopened,
            source_review_path=args.source_review,
            training_grades=args.training_grades,
            cost_proxy=args.cost_proxy,
        )
    elif args.command == "grade":
        value, audit = grade(args.run, args.teacher, args.training, training_grades=args.training_grades)
        write(args.run / "teacher-audit.json", audit)
    elif args.command == "fit":
        value = fit(
            args.run,
            args.training,
            args.embeddings,
            args.prices,
            progress=print,
            training_grades=args.training_grades,
            cost_proxy=args.cost_proxy,
        )
    elif args.command == "freeze":
        value = freeze(args.run, args.baseline, args.original)
    elif args.command == "authorize-test":
        value = authorize_test(args.run)
    else:
        value = report(args.run, args.test_grades, args.router_credits)
    write(path, value)
    print(json.dumps({"output": str(path), "sha256": file_hash(path), "status": value.get("status", "complete")}))


if __name__ == "__main__":
    main()
