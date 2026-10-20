"""Apply an explicit, hash-bound calibration answer-format review."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import arithmetic_study as study
from data import digest, file_hash, write


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def apply_decision(row, flag, decision, reference):
    """Only an unresolved comparison with the exact reviewed bytes can change."""
    arm = flag["arm"]
    current = row[f"{arm}_grade"]
    study.require(
        flag["status"] in study.UNRESOLVED
        and current["status"] == flag["status"]
        and decision["id"] == flag["id"] == row["id"]
        and decision["arm"] == arm
        and decision["record_sha256"] == flag["record_sha256"] == row[f"{arm}_record_sha256"]
        and decision["question_sha256"] == text_hash(flag["question"])
        and decision["reference_sha256"] == text_hash(reference) == text_hash(flag["reference"])
        and decision["final_answer_sha256"] == text_hash(flag["final_answer"])
        and current["final_answer"] == flag["final_answer"]
        and type(decision["correct"]) is bool
        and bool(decision["reason"]),
        "Calibration adjudication does not match the exact unresolved comparison",
    )
    row[f"{arm}_correct"] = decision["correct"]
    row[f"{arm}_grade"] = {
        "correct": decision["correct"],
        "status": "graded",
        "final_answer": flag["final_answer"],
        "base_grade": current,
        "adjudication_sha256": digest(decision),
        "method": "Explicit calibration answer-format review before cutoff selection",
    }


def verified_scan(source, run, collection, prices, manifest, workers):
    # The original function performs every source, acquisition and record check,
    # and writes a complete scan in a fresh directory, preserving the first scan.
    with tempfile.TemporaryDirectory(prefix="arithmetic-calibration-validation-") as temporary:
        scratch = Path(temporary)
        shutil.copyfile(run / "plan.json", scratch / "plan.json")
        try:
            study.grade(source, scratch, collection, prices, "calibration", workers=workers)
        except ValueError as error:
            if (
                str(error)
                != f"{len(manifest['decisions'])} unresolved comparisons; retain every row and review the complete list"
            ):
                raise
        else:
            raise ValueError("The reviewed unresolved comparison no longer exists")
        fresh_scan = scratch / "unresolved-calibration.json"
        study.require(file_hash(fresh_scan) == manifest["original_scan_sha256"], "Recomputed review scan differs")
        diagnostic_path = run / "unresolved-calibration.json"
        if not diagnostic_path.exists():
            with diagnostic_path.open("xb") as stream:
                stream.write(fresh_scan.read_bytes())
    diagnostic_path = run / "unresolved-calibration.json"
    study.require(
        file_hash(diagnostic_path) == manifest["original_scan_sha256"],
        "The complete calibration-format scan differs from the reviewed scan",
    )
    return diagnostic_path


def grade(source, run, collection, prices, manifest_path, *, workers=8):
    source, run, collection, prices, manifest_path = map(Path, (source, run, collection, prices, manifest_path))
    study.unopened(run)
    if (run / "grades-calibration.json").exists():
        raise FileExistsError(run / "grades-calibration.json")
    manifest = study.read(manifest_path)
    study.require(
        manifest["format"] == "sie-arithmetic-calibration-adjudication-v1"
        and manifest["split"] == "calibration"
        and manifest["base_source_sha256"] == study.code_hashes()
        and manifest["decisions"],
        "Review must bind the unchanged base grader and calibration-only scope",
    )
    diagnostic_path = verified_scan(source, run, collection, prices, manifest, workers)
    decisions = {(item["id"], item["arm"]): item for item in manifest["decisions"]}
    study.require(len(decisions) == len(manifest["decisions"]), "Duplicate review decisions")
    plan, inputs, metadata = study.checked_plan(source, run)
    gold = study.read(source / "gold-for-scoring-only.json")
    rates, public_rows = study.benefit_costs.rates_from_snapshot(study.read(prices))
    acquisition = study.read(collection / "collection.json")
    jobs = []
    for question, meta in zip(inputs["calibration"], metadata["calibration"], strict=True):
        records = {arm: study.read(collection / f"{question['id']}-{arm}.json") for arm in study.outcomes.ARMS}
        hashes = {arm: file_hash(collection / f"{question['id']}-{arm}.json") for arm in study.outcomes.ARMS}
        jobs.append((question, meta, gold[question["id"]], records, hashes, rates))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        computed = list(pool.map(study._grade_pair, jobs, chunksize=8))
    base_rows_sha256 = digest([row for row, _ in computed])
    applied = []
    for row, flags in computed:
        for flag in flags:
            key = flag["id"], flag["arm"]
            study.require(key in decisions, "An unresolved comparison has no explicit adjudication")
            apply_decision(row, flag, decisions[key], gold[flag["id"]])
            applied.append(key)
    study.require(set(applied) == set(decisions) and len(applied) == len(decisions), "Review coverage differs")
    result = {
        "format": "sie-arithmetic-adaptation-grades-v1",
        "split": "calibration",
        "study_plan_sha256": file_hash(run / "plan.json"),
        "source_plan_sha256": plan["source_plan_sha256"],
        "collection_plan_sha256": file_hash(collection / "collection-plan.json"),
        "collection_summary_sha256": file_hash(collection / "collection.json"),
        "acquisition_receipt_mapping_sha256": digest(acquisition["receipt_sha256"]),
        "acquisition_verification": (
            "The unchanged base grade command verifies every selected record and the complete acquisition mapping."
        ),
        "gold_sha256": file_hash(source / "gold-for-scoring-only.json"),
        "prices_sha256": file_hash(prices),
        "public_price_rows": public_rows,
        "grading_source_sha256": {
            name: study.code_hashes()[name]
            for name in ("arithmetic_answers.py", "teacher_format.py", "solver_answers.py")
        },
        "test_access_sha256": None,
        "cost_scope": "Every physical call priced at pinned uncached Cloud rates; not observed Cloud bills or Modal spend.",
        "grading_extension": {
            "source_sha256": file_hash(Path(__file__)),
            "manifest_sha256": file_hash(manifest_path),
            "original_scan_sha256": file_hash(diagnostic_path),
            "base_rows_sha256": base_rows_sha256,
            "applied": len(applied),
            "scope": "One exact calibration format review before cutoff selection; no row exclusion.",
        },
        "rows": [row for row, _ in computed],
    }
    write(run / "grades-calibration.json", result)
    return result
