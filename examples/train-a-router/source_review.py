"""Apply a source-reference audit without changing saved model responses.

The review is data, not a list of model-specific exceptions. It binds the
original question, reference, independent proof and any exclusion. Training
answers were already available when this amendment was made; solver test
answers remain closed until all models and cutoffs have been saved.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import benefit
import benefit_data
import benefit_expanded_gold
import benefit_expanded_gold_v2
import benefit_training_review
import solver
import solver_gold
import solver_policies
import teacher_format
from data import SOURCE_HASHES, SOURCE_REVISION, digest, file_hash, write

FORMAT = "sie-router-source-review-v3"
PROTOCOL = {
    "references": "Pinned source answers, prior source corrections, and independently proved amendments; teacher disagreements nominate reviews but are not accepted as proof",
    "eligibility": "Exclude missing inputs, contradictory or ambiguous prompts, and partial-credit estimation or proof-only tasks outside the final-answer comparison contract; apply source-only rules to every split",
    "timing": "After training outcome collection and exploratory v2b fits; before corrected fits and held-out solver outcome access",
    "formatting": "Normalize both finals by question-only rules; preserve original text and transformations",
    "old_results": "Original acquisition plans, responses, graders and exploratory results remain unchanged",
}
METADATA_KEYS = ("source_review_sha256", "source_review_grader_sha256", "format_normalizer_sha256")
SPECIAL_SEMANTICS = {
    "clock_time",
    "progression",
    "rounded_percentage",
    "prime_approximation",
    "division_operator",
    "ordered_labels",
    "weekday",
    "official_alternatives",
    "exact_rational",
}


def read(path):
    return json.loads(Path(path).read_text())


def canonical_reference(identity, original):
    """Preserve previously reviewed source semantics before applying new data."""
    if identity in benefit_training_review.REVIEW:
        rule = benefit_training_review.REVIEW[identity]
        if original != rule["original"]:
            raise ValueError("Symbolic source reference changed")
        return rule["canonical"], "reviewed_symbolic_training"
    for rules in (
        benefit_expanded_gold_v2.RULES,
        benefit_expanded_gold.RULES,
        benefit_data.RULES,
        solver_gold.RULES,
    ):
        if identity in rules:
            expected, canonical, kind = rules[identity]
            if original != expected:
                raise ValueError("Prior source reference changed")
            return canonical, kind
    return original, "scalar"


def checked_review(review):
    if not (
        review.get("format") == FORMAT
        and review.get("finalized") is True
        and review.get("protocol") == PROTOCOL
        and review.get("source_revision") == SOURCE_REVISION
        and review.get("source_hashes") == SOURCE_HASHES
        and review.get("solver_test_outcomes_accessed") is False
        and review.get("training_outcomes_already_available") is True
    ):
        raise ValueError("Source review must declare the pinned data and truthful outcome-access boundary")
    records = review.get("records")
    if not isinstance(records, list) or not records or any(not isinstance(row, dict) for row in records):
        raise ValueError("Source review records must be a nonempty list")
    for row in records:
        if not (
            isinstance(row.get("id"), str)
            and row["id"]
            and Path(row["id"]).name == row["id"]
            and row["id"] not in (".", "..")
            and "\\" not in row["id"]
            and row.get("split") in ("train", "calibration", "test")
            and isinstance(row.get("question"), str)
            and row["question"].strip()
            and row.get("question_sha256") == digest(row["question"])
            and isinstance(row.get("request_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", row["request_sha256"])
            and row.get("decision") in ("retain", "exclude")
            and isinstance(row.get("reason"), str)
            and row["reason"].strip()
            and isinstance(row.get("original_reference"), str)
            and isinstance(row.get("reference"), str)
            and row["reference"]
            and isinstance(row.get("semantic_kind"), str)
            and row["semantic_kind"]
        ):
            raise ValueError("Invalid source-only review row")
        if row["decision"] == "retain" and row["semantic_kind"] in SPECIAL_SEMANTICS | {"reviewed_symbolic_training"}:
            canonical, kind = canonical_reference(row["id"], row["original_reference"])
            if row["reference"] != canonical or row["semantic_kind"] != kind:
                raise ValueError("A changed reference must use source_correction, not prior special semantics")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Source review IDs must be nonempty and unique")
    return review


def metadata(path):
    checked_review(read(path))
    return {
        "source_review_sha256": file_hash(path),
        "source_review_grader_sha256": file_hash(Path(__file__)),
        "format_normalizer_sha256": file_hash(Path(teacher_format.__file__)),
    }


def eligible_rows(rows, split, review):
    """Filter a collected population by source content, preserving its order."""
    mapping = {row["id"]: row for row in checked_review(review)["records"]}
    if split not in ("train", "calibration", "test") or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Invalid split or duplicate collected question")
    selected = []
    for row in rows:
        if row["id"] not in mapping:
            raise ValueError("Collected question is missing from the source review")
        item = mapping[row["id"]]
        request_sha = row.get("request_sha256")
        if "request" in row:
            actual = digest(row["request"])
            if request_sha is not None and request_sha != actual:
                raise ValueError("Collected request text differs from its digest")
            request_sha = actual
            if row["request"].partition("\nProblem: ")[2].strip() != item["question"].strip():
                raise ValueError("Review question differs from the collected request text")
        if item["split"] != split or item["request_sha256"] != request_sha:
            raise ValueError("Review belongs to a different split or question")
        if item["decision"] == "retain":
            selected.append(row)
    if not selected:
        raise ValueError("Source review removed an entire split")
    return selected


def grade_reference_row(identity, question, original_reference, response, review):
    """Compare a solver final with the reviewed source, retaining its provenance."""
    mapping = {row["id"]: row for row in checked_review(review)["records"]}
    item = mapping[identity]
    if item["decision"] != "retain" or item["question"] != question or item["original_reference"] != original_reference:
        raise ValueError("Cannot grade an excluded or mismatched source question")
    return _grade_reviewed_row(item, response)


def _grade_reviewed_row(item, response):
    """Use a row from an already checked review; avoid rechecking all rows per answer."""
    identity, question, original_reference = item["id"], item["question"], item["original_reference"]
    kind = item["semantic_kind"]
    if kind == "reviewed_symbolic_training":
        # These two training-only huge-power answers were proved symbolically
        # before this amendment; do not invoke an unbounded integer expansion.
        result = benefit_training_review.reviewed_grade(identity, original_reference, response)
    elif kind in SPECIAL_SEMANTICS:
        result = benefit_expanded_gold_v2.grade_expanded(identity, original_reference, response)
    else:
        result = teacher_format.grade_reference(question, item["reference"], response)
    return result | {
        "source_reference": {
            "original": original_reference,
            "reviewed": item["reference"],
            "question_sha256": item["question_sha256"],
            "semantic_kind": kind,
            "review_reason": item["reason"],
        },
    }


def grade_collection(directory, split, review_path, *, test_access_path=None):
    """Grade eligible saved answers; test requires a prior all-model access receipt."""
    directory, review_path = Path(directory), Path(review_path)
    if split not in ("train", "calibration", "test"):
        raise ValueError("Unknown collection split")
    if split == "test":
        if test_access_path is None or not Path(test_access_path).is_file():
            raise ValueError("Save the all-model test-access receipt before opening test answers")
        try:
            access = read(test_access_path)
        except (ValueError, OSError) as error:
            raise ValueError("Invalid all-model test-access receipt") from error
        if not (
            isinstance(access, dict)
            and access.get("format") == "sie-reviewed-outcome-all-model-access-v1"
            and access.get("split") == "test"
            and access.get("source_review_sha256") == file_hash(review_path)
            and access.get("original_solver_plan_sha256") == file_hash(directory / "plan.json")
            and isinstance(access.get("calibration_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", access["calibration_sha256"])
            and isinstance(access.get("studies"), list)
            and access["studies"]
            and all(isinstance(study, dict) and study for study in access["studies"])
        ):
            raise ValueError("All-model test-access receipt does not bind this review, plan and calibration")
    review = checked_review(read(review_path))
    review_metadata = metadata(review_path)
    plan, inputs, settings = (read(directory / name) for name in ("plan.json", "inputs.json", "settings.json"))
    if plan["inputs_sha256"] != digest(inputs) or plan["settings_sha256"] != digest(settings):
        raise ValueError("Saved solver inputs or settings changed")
    collection = directory / f"answers-{split}"
    summary, binding = read(collection / "collection.json"), read(collection / "collection-plan.json")
    frozen_hash = file_hash(directory / "policies-frozen.json") if split == "test" else None
    solver.collection_binding(binding, directory, split, inputs[split], settings, frozen_hash, binding["self_hosted"])
    if (
        summary["status"] != "complete"
        or summary["completed"] != 2 * len(inputs[split])
        or summary["failed"]
        or summary.get("missing", 0) != 0
        or summary.get("unknown_execution_attempts", 0) != 0
    ):
        raise ValueError("Need every paired response in the original collected population")
    gold = read(directory / "gold-for-scoring-only.json")
    if digest(gold) != plan["gold_sha256"]:
        raise ValueError("Acquisition reference bytes changed")
    selected = eligible_rows(inputs[split], split, review)
    reference = {row["id"]: row for row in review["records"]}
    rows = []
    for row in selected:
        identity = row["id"]
        item = reference[identity]
        if gold[identity] != item["original_reference"]:
            raise ValueError("Acquisition and audited source references differ")
        outcome = {"id": identity}
        for arm in ("cheap", "strong"):
            path = collection / f"{identity}-{arm}.json"
            record = read(path)
            solver.validate_record(record, row, arm, settings)
            if record["status"] != "complete":
                raise ValueError("Incomplete saved answer")
            measured = _grade_reviewed_row(item, record["answer"])
            if measured["status"] in ("gold_unparseable", "grading_error", "unit_contract_unresolved"):
                raise ValueError(f"Unresolved comparison for {identity}/{arm}: {measured['status']}")
            outcome.update(
                {
                    f"{arm}_correct": measured["correct"],
                    f"{arm}_credits": record["credits"],
                    f"{arm}_grade": measured,
                    f"{arm}_record_sha256": file_hash(path),
                }
            )
        rows.append(outcome)
    parent_keys = (
        "grader_source_sha256",
        "gold_review_source_sha256",
        "gold_review_sha256",
        "training_gold_rules_sha256",
        "training_gold_review_source_sha256",
        "outcome_grader_source_sha256",
        "outcome_grader_protocol_sha256",
    )
    parent = {key: plan[key] for key in parent_keys if key in plan}
    parent.setdefault("gold_review_sha256", file_hash(directory / "gold-review.json"))
    parent |= benefit_expanded_gold_v2.outcome_grader_metadata()
    if metadata(review_path) != review_metadata:
        raise ValueError("Source review or grading code changed during grading")
    return {
        "split": split,
        "plan_sha256": file_hash(directory / "plan.json"),
        "collection_plan_sha256": file_hash(collection / "collection-plan.json"),
        "collection_summary_sha256": file_hash(collection / "collection.json"),
        "source_collection_policies_frozen_sha256": frozen_hash,
        "policies_frozen_sha256": None,
        "collected_ids": [row["id"] for row in inputs[split]],
        "excluded_ids": [row["id"] for row in inputs[split] if reference[row["id"]]["decision"] == "exclude"],
        "test_access_sha256": None if test_access_path is None else file_hash(test_access_path),
        **parent,
        **review_metadata,
        "rows": rows,
    }


def calibration_snapshot(directory, output, review_path):
    """Keep historical input provenance, replace only the eligible scored rows."""
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    old_path = directory / "policies-frozen.json"
    old = read(old_path)
    solver_policies.check_seal(old, solver_policies.FROZEN_FORMAT)
    graded = grade_collection(directory, "calibration", review_path)
    original_ids = [row["id"] for row in old["calibration_rows"]]
    if original_ids != graded["collected_ids"] or len(original_ids) != len(set(original_ids)):
        raise ValueError("Historical calibration cost order differs from the collected questions")
    original_credits = dict(zip(original_ids, old["embedding_calibration_credits"], strict=True))
    credits = [original_credits[row["id"]] for row in graded["rows"]]
    output.mkdir(parents=True)
    shutil.copyfile(directory / "plan.json", output / "plan.json")
    write(output / "grades-calibration.json", graded)
    keys = ("id", "cheap_correct", "strong_correct", "cheap_credits", "strong_credits")
    amended = {key: value for key, value in old.items() if key != "sha256"}
    amended.update(
        {
            "calibration_rows": [{key: row[key] for key in keys} for row in graded["rows"]],
            "calibration_grades_sha256": file_hash(output / "grades-calibration.json"),
            "embedding_calibration_credits": credits,
            "selections": {},
            **benefit_expanded_gold_v2.outcome_grader_metadata(),
            **metadata(review_path),
            "amendment": {
                "created_at": benefit.now(),
                "original_frozen_sha256": file_hash(old_path),
                "boundary": "after_collection_before_corrected_fits_and_test_outcome_access",
                "attest_test_outcomes_unopened": True,
                "source_scores": "Historical full-population input provenance only; no historical cutoff is reused",
                "excluded_calibration_ids": graded["excluded_ids"],
                "source_review_sha256": file_hash(review_path),
            },
            "test_access": "Calibration-only source amendment; does not authorize test outcome access",
        }
    )
    write(output / "policies-frozen.json", solver_policies.seal(amended))
    return {
        "n": len(graded["rows"]),
        "excluded": graded["excluded_ids"],
        "frozen_sha256": file_hash(output / "policies-frozen.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("grade-train", "calibration"))
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "calibration":
        print(json.dumps(calibration_snapshot(args.collection, args.output, args.review), indent=2))
    else:
        result = grade_collection(args.collection, "train", args.review)
        write(args.output, result)
        print(json.dumps({"n": len(result["rows"]), "excluded": result["excluded_ids"]}, indent=2))


if __name__ == "__main__":
    main()
