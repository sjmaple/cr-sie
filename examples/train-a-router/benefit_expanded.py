"""Prepare a larger, source-reviewed training population and grade its answers.

The original 511 completed pairs are reused byte for byte. Additional source
exclusions are settled before collecting more answers; held-out solver answers
are never opened by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import benefit_data
import solver
from data import SOURCE_HASHES, SOURCE_REVISION, audit, canonical, digest, file_hash, render_request, write
from serving import INSTRUCTION, MODEL, REVISION
from solver_answers import grader_metadata

FORMAT = "sie-expanded-benefit-training-source-v2"
CODE_FILES = (
    "benefit_expanded.py",
    "benefit_expanded_collect.py",
    "benefit_expanded_gold.py",
    "benefit_expanded_gold_v2.py",
    "benefit_data.py",
    "benefit_collect.py",
    "solver.py",
    "solver_answers.py",
    "solver_gold.py",
    "data.py",
    "serving.py",
)


def read(path):
    return solver.read(Path(path))


def source_hashes():
    from benefit_expanded_gold_v2 import outcome_grader_metadata

    here = Path(__file__).parent
    return {
        "grader_source_sha256": file_hash(here / "solver_answers.py"),
        "gold_review_source_sha256": file_hash(here / "solver_gold.py"),
        "training_gold_review_source_sha256": file_hash(here / "benefit_expanded_gold_v2.py"),
        "collector_source_sha256": file_hash(here / "solver.py"),
        "expanded_source_sha256": {name: file_hash(here / name) for name in CODE_FILES},
        **outcome_grader_metadata(),
    }


def pool_rows(reference, source, expansion):
    import pyarrow.parquet as pq

    dataset, receipt = read(reference / "data.json"), read(expansion / "training-expansion-audit.json")
    audit(dataset)
    if (
        dataset["source_revision"] != SOURCE_REVISION
        or receipt["original_dataset_sha256"] != digest(dataset)
        or receipt["source_revision"] != SOURCE_REVISION
        or receipt["source_hashes"] != SOURCE_HASHES
        or receipt["training_rows"] != 2581
        or receipt["original_prefix_rows"] != 512
        or file_hash(expansion / "train-expanded.jsonl") != receipt["train_expanded_jsonl_sha256"]
    ):
        raise ValueError("Expansion differs from the pinned source and original training prefix")
    originals = {}
    for name, expected in SOURCE_HASHES.items():
        if file_hash(source / name) != expected:
            raise ValueError("Pinned source hash differs")
        originals[name.removesuffix(".parquet")] = pq.read_table(source / name).to_pylist()
    rows = []
    for identity, family in zip(receipt["ids"], receipt["families"], strict=True):
        split, index = identity.removeprefix("e2h-amc-").rsplit("-", 1)
        original = originals[split][int(index)]
        rows.append(
            {
                "id": identity,
                "request": render_request(original["tag"], original["problem"]),
                "label": int(original["rating"] >= dataset["label_thresholds"][original["tag"]]),
                "contest": original["tag"],
                "year": original["year"],
                "family": family,
                "paper_group": f"{original['contest']}:{original['year']}",
            }
        )
    if digest(rows) != receipt["expanded_rows_sha256"] or rows[:512] != dataset["splits"]["train"]:
        raise ValueError("Reconstructed expansion differs from its source audit")
    audit(dict(dataset, splits=dict(dataset["splits"], train=rows)))
    return dataset, rows, originals, receipt


def reviewed_exclusions(gold, sources, original_ids, leakage, mandatory):
    """Combine frozen source judgments and deduplicate connected components."""
    if (
        leakage.get("format") != "sie-expanded-source-leakage-review-v1"
        or leakage.get("source_only") is not True
        or leakage.get("solver_outcomes_read_for_decisions") is not False
        or leakage.get("decided_before_new_training_outcomes") is not True
        or leakage.get("reused_training_outcomes_already_available") is not True
    ):
        raise ValueError("Leakage review must acknowledge existing outcomes and use source inputs only")
    excluded = {}
    for kind, rules in (
        ("original_source_context", benefit_data.EXCLUSIONS),
        ("expanded_source_context", mandatory),
        ("cross_split_family", leakage["exclusions"]),
    ):
        for identity, rule in rules.items():
            if identity not in gold or identity in excluded or not rule.get("reason"):
                raise ValueError("Invalid or repeated source-only exclusion")
            for key, value in (
                ("original_gold", gold[identity]),
                ("problem_sha256", digest(sources[identity]["problem"])),
                ("solution_sha256", digest(sources[identity]["solution"])),
            ):
                if key in rule and rule[key] != value:
                    raise ValueError("Exclusion differs from its pinned source")
            excluded[identity] = {"id": identity, "kind": kind, **rule}
    for review in leakage["candidate_reviews"]:
        for identity, expected in review["source_problem_sha256"].items():
            if identity in sources and hashlib.sha256(sources[identity]["problem"].encode()).hexdigest() != expected:
                raise ValueError("Leakage review source problem differs")
    parent = {identity: identity for identity in gold}

    def root(identity):
        while parent[identity] != identity:
            identity = parent[identity]
        return identity

    for group in leakage["duplicate_groups"]:
        if len(group) < 2 or len(set(group)) != len(group) or any(identity not in gold for identity in group):
            raise ValueError("Duplicate review must identify distinct source training rows")
        for identity in group[1:]:
            parent[root(identity)] = root(group[0])
    components = {}
    for identity in gold:
        components.setdefault(root(identity), []).append(identity)
    duplicate_decisions = []
    for component in components.values():
        if len(component) < 2:
            continue
        available = [identity for identity in component if identity not in excluded]
        if not available:
            continue
        candidates = [identity for identity in available if identity in original_ids] or available
        retained = min(candidates, key=lambda identity: digest(["expanded-source-duplicate-v1", identity]))
        duplicate_decisions.append({"ids": component, "retained_id": retained})
        for identity in available:
            if identity != retained:
                excluded[identity] = {
                    "id": identity,
                    "kind": "within_training_duplicate",
                    "reason": f"Source-reviewed duplicate of retained training request {retained}",
                    "retained_id": retained,
                }
    return excluded, duplicate_decisions


def prepare(reference, source, expansion, prior, leakage_path, output, population="expanded"):
    from benefit_expanded_gold_v2 import EXCLUSIONS, review_expanded

    reference, source, expansion, prior, output = map(Path, (reference, source, expansion, prior, output))
    if output.exists():
        raise FileExistsError(output)
    dataset, pool, originals, expansion_audit = pool_rows(reference, source, expansion)
    if population not in ("original", "expanded"):
        raise ValueError("Choose the cleaned original or expanded population")
    prior_plan, prior_inputs, settings = benefit_data.checked_training(prior)
    old_ids = [row["id"] for row in prior_inputs["train"]]
    pool_inputs, pool_gold, sources = benefit_data.join_training(dict(dataset, splits={"train": pool}), originals)
    leakage = read(leakage_path)
    excluded, duplicates = reviewed_exclusions(
        pool_gold, sources, {row["id"] for row in pool[:512]}, leakage, EXCLUSIONS
    )
    if population == "original":
        pool = pool[:512]
    pool_ids = {row["id"] for row in pool}
    excluded = {identity: rule for identity, rule in excluded.items() if identity in pool_ids}
    inputs = {"train": [row for row in pool_inputs["train"] if row["id"] in pool_ids and row["id"] not in excluded]}
    eligible = [row["id"] for row in inputs["train"]]
    reuse_ids = [identity for identity in old_ids if identity not in excluded]
    if eligible[: len(reuse_ids)] != reuse_ids or prior_plan["dataset_sha256"] != digest(dataset):
        raise ValueError("Expansion must retain the original eligible prefix")
    gold = {identity: pool_gold[identity] for identity in eligible}
    corrected, preflight, review = review_expanded(gold, sources)
    if not review["ready"]:
        raise ValueError("Expanded source review remains unresolved")
    prior_collection = prior / "answers-train"
    old_summary, old_binding = (
        read(prior_collection / "collection.json"),
        read(prior_collection / "collection-plan.json"),
    )
    solver.collection_binding(old_binding, prior, "train", prior_inputs["train"], settings, None, True)
    if old_summary["status"] != "complete" or old_summary["completed"] != 2 * len(old_ids):
        raise ValueError("Original paired collection is incomplete")
    reused_hashes = {}
    for row in prior_inputs["train"]:
        if row["id"] not in reuse_ids:
            continue
        for arm in settings["arms"]:
            name = f"{row['id']}-{arm}.json"
            record = read(prior_collection / name)
            solver.validate_record(record, row, arm, settings)
            if record["status"] != "complete":
                raise ValueError("Only completed original answers can be reused")
            reused_hashes[name] = file_hash(prior_collection / name)
    eligibility = {
        "format": "sie-benefit-training-eligibility-v1",
        "original_pool_ids": [row["id"] for row in pool],
        "eligible_train_ids": eligible,
        "exclusions": [
            {
                "id": identity,
                "kind": record["kind"],
                "reason": record["reason"],
                "problem_sha256": digest(sources[identity]["problem"]),
            }
            for identity, record in excluded.items()
        ],
        "source_only": True,
        "decided_before_new_training_outcomes": True,
        "reused_training_outcomes_already_available": True,
        "timing": "Historical outcomes already exist. This revised source-only audit precedes the new controlled fit and additional acquisition; no removals use model outcomes.",
        "leakage_review_sha256": file_hash(leakage_path),
        "duplicate_decisions": duplicates,
        "source_revision": SOURCE_REVISION,
        "source_hashes": SOURCE_HASHES,
        "dataset_sha256": digest(dataset),
        "training_gold_review_source_sha256": source_hashes()["training_gold_review_source_sha256"],
    }
    for name, value in (
        ("training-rows.json", pool),
        ("inputs.json", inputs),
        ("eligibility.json", eligibility),
        ("expansion-audit.json", expansion_audit),
        ("gold-for-scoring-only.json", gold),
        ("gold-reviewed.json", corrected),
        ("settings.json", settings),
        ("gold-preflight.json", preflight),
    ):
        write(output / name, value)
    shutil.copyfile(leakage_path, output / "source-leakage-review.json")
    review["preflight_sha256"] = file_hash(output / "gold-preflight.json")
    review["eligibility_sha256"] = file_hash(output / "eligibility.json")
    write(output / "training-gold-adjudications.json", review)
    review_gate = {
        key: review[key]
        for key in ("preflight_sha256", "original_gold_sha256", "canonical_gold_sha256", "approved_ids", "ready")
    }
    review_gate.update(
        unresolved_ids=review.get("unresolved_ids", []),
        training_gold_rules_sha256=file_hash(output / "training-gold-adjudications.json"),
    )
    write(output / "gold-review.json", review_gate)
    reuse = {
        "ids": reuse_ids,
        "records_sha256": reused_hashes,
        "plan_sha256": file_hash(prior / "plan.json"),
        "collection_plan_sha256": file_hash(prior_collection / "collection-plan.json"),
        "collection_sha256": file_hash(prior_collection / "collection.json"),
        "same_completed_record_bytes": True,
        "historical_ids_excluded_by_new_source_audit": [identity for identity in old_ids if identity in excluded],
    }
    write(output / "reuse.json", reuse)
    overlaps = {}
    for split in ("validation", "calibration", "test"):
        held = dataset["splits"][split]
        overlaps[split] = {
            key: len({row[key] for row in pool} & {row[key] for row in held}) for key in ("id", "family", "paper_group")
        }
        overlaps[split]["canonical_request"] = len(
            {canonical(row["request"]) for row in pool} & {canonical(row["request"]) for row in held}
        )
        if any(overlaps[split].values()):
            raise ValueError("Expansion overlaps held-out source inputs")
    plan = {
        "format": FORMAT,
        "population": population,
        "dataset_sha256": digest(dataset),
        "inputs_sha256": digest(inputs),
        "gold_sha256": digest(gold),
        "settings_sha256": digest(settings),
        "counts": {"train": len(eligible)},
        "source_counts": {"train": len(pool)},
        "eligible_train_ids": eligible,
        "excluded_train_ids": list(excluded),
        "new_train_ids": [identity for identity in eligible if identity not in set(reuse_ids)],
        "eligibility_sha256": file_hash(output / "eligibility.json"),
        "leakage_review_sha256": file_hash(output / "source-leakage-review.json"),
        "training_rows_sha256": file_hash(output / "training-rows.json"),
        "reuse_sha256": file_hash(output / "reuse.json"),
        "expansion_audit_sha256": file_hash(output / "expansion-audit.json"),
        "source": {"revision": SOURCE_REVISION, "hashes": SOURCE_HASHES},
        "source_revision": SOURCE_REVISION,
        "source_hashes": SOURCE_HASHES,
        "held_out_source_overlap": overlaps,
        "paper_groups": len({row["paper_group"] for row in pool}),
        "features": {"model": MODEL, "revision": REVISION, "instruction": INSTRUCTION, "dimensions": 2560},
        "inference_reads_gold": False,
        "grader": grader_metadata(),
        **source_hashes(),
        "training_gold_rules_sha256": file_hash(output / "training-gold-adjudications.json"),
        "gold_review_sha256": file_hash(output / "gold-review.json"),
    }
    write(output / "plan.json", plan)
    return plan


def checked_training(directory):
    directory = Path(directory)
    plan, inputs, settings = solver.checked_inputs(directory)
    ids = [row["id"] for row in inputs.get("train", [])]
    if (
        plan.get("format") != FORMAT
        or set(inputs) != {"train"}
        or any(set(row) != {"id", "request"} for row in inputs["train"])
        or len(ids) != len(set(ids))
        or plan.get("counts") != {"train": len(ids)}
        or plan.get("eligible_train_ids") != ids
        or any(plan.get(key) != value for key, value in source_hashes().items())
    ):
        raise ValueError("Expanded inputs, population or frozen source differs")
    for key, name in (
        ("eligibility_sha256", "eligibility.json"),
        ("leakage_review_sha256", "source-leakage-review.json"),
        ("reuse_sha256", "reuse.json"),
        ("training_rows_sha256", "training-rows.json"),
        ("expansion_audit_sha256", "expansion-audit.json"),
        ("training_gold_rules_sha256", "training-gold-adjudications.json"),
        ("gold_review_sha256", "gold-review.json"),
    ):
        if plan.get(key) != file_hash(directory / name):
            raise ValueError("Expanded population or source review evidence changed")
    eligibility, reuse = read(directory / "eligibility.json"), read(directory / "reuse.json")
    excluded = [row["id"] for row in eligibility["exclusions"]]
    if (
        eligibility.get("source_only") is not True
        or eligibility.get("decided_before_new_training_outcomes") is not True
        or eligibility.get("reused_training_outcomes_already_available") is not True
        or eligibility.get("leakage_review_sha256") != plan["leakage_review_sha256"]
        or eligibility["eligible_train_ids"] != ids
        or excluded != plan["excluded_train_ids"]
        or ids != [identity for identity in eligibility["original_pool_ids"] if identity not in set(excluded)]
        or len(eligibility["original_pool_ids"]) != plan["source_counts"]["train"]
        or len(set(eligibility["original_pool_ids"])) != plan["source_counts"]["train"]
        or ids[: len(reuse["ids"])] != reuse["ids"]
        or plan["new_train_ids"] != [identity for identity in ids if identity not in set(reuse["ids"])]
    ):
        raise ValueError("Expanded eligibility or reused prefix differs")
    solver.check_review(directory, ids)
    return plan, inputs, settings


def grade(directory, output=None):
    from benefit_expanded_gold_v2 import grade_expanded

    directory = Path(directory)
    output = Path(output) if output else directory / "grades-train.json"
    if output.exists():
        raise FileExistsError(output)
    plan, inputs, settings = checked_training(directory)
    gold = read(directory / "gold-for-scoring-only.json")
    if digest(gold) != plan["gold_sha256"] or plan["grader"] != grader_metadata():
        raise ValueError("Expanded reference answers or grading dependency changed")
    collection = directory / "answers-train"
    summary, binding = read(collection / "collection.json"), read(collection / "collection-plan.json")
    solver.collection_binding(binding, directory, "train", inputs["train"], settings, None, True)
    if summary["status"] != "complete" or summary["completed"] != 2 * len(inputs["train"]) or summary["failed"]:
        raise ValueError("Expanded paired collection is incomplete")
    results = []
    for row in inputs["train"]:
        outcome = {"id": row["id"]}
        for arm in settings["arms"]:
            path = collection / f"{row['id']}-{arm}.json"
            record = read(path)
            solver.validate_record(record, row, arm, settings)
            if record["status"] != "complete":
                raise ValueError("Incomplete expanded training answer")
            measured = grade_expanded(row["id"], gold[row["id"]], record["answer"])
            if measured["status"] in ("gold_unparseable", "grading_error"):
                raise ValueError("Source-reviewed training answer could not be graded")
            outcome.update(
                {
                    f"{arm}_correct": measured["correct"],
                    f"{arm}_credits": record["credits"],
                    f"{arm}_grade": measured,
                    f"{arm}_record_sha256": file_hash(path),
                }
            )
        results.append(outcome)
    result = {
        "split": "train",
        "plan_sha256": file_hash(directory / "plan.json"),
        "collection_plan_sha256": file_hash(collection / "collection-plan.json"),
        "collection_summary_sha256": file_hash(collection / "collection.json"),
        "policies_frozen_sha256": None,
        **source_hashes(),
        "gold_review_sha256": plan["gold_review_sha256"],
        "training_gold_rules_sha256": plan["training_gold_rules_sha256"],
        "rows": results,
    }
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in ("reference", "source", "expansion", "prior", "leakage-review", "output"):
        prepare_parser.add_argument(f"--{name}", type=Path, required=True)
    prepare_parser.add_argument("--population", choices=("original", "expanded"), default="expanded")
    grade_parser = commands.add_parser("grade")
    grade_parser.add_argument("--directory", type=Path, required=True)
    grade_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.reference, args.source, args.expansion, args.prior, args.leakage_review, args.output, args.population
        )
    else:
        grade(args.directory, args.output)


if __name__ == "__main__":
    main()
