"""Fit and evaluate the fixed arithmetic routing recipe on disjoint source families."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import arithmetic_answers
import benefit
import benefit_costs
import benefit_outcomes as outcomes
import benefit_outcomes_native as native
import solver
import solver_report
from arithmetic_data import COUNTS
from arithmetic_data import FORMAT as SOURCE_FORMAT
from data import digest, file_hash, write
from serving import vectors

TARGETS = (1, 0, 2, 5)
METHODS = (
    "base",
    "embedding_original_cutoff",
    "lora_original_cutoff",
    "embedding_calibration_only",
    "lora_calibration_only",
    "embedding_500",
    "embedding_2000",
    "lora_500",
    "lora_2000",
)
OPTIONAL_METHODS = ("embedding_teacher_2000", "embedding_teacher_matched_reference")
UNRESOLVED = {"gold_unparseable", "grading_error", "unit_contract_unresolved"}
PROTOCOL = {
    "population": "All exposed2000 arithmetic rows train/develop; fresh1000 calibration and3000 test families.",
    "training_sizes": [500, 2000],
    "embedding": {
        key: outcomes.PROTOCOL[key]
        for key in (
            "encoder",
            "revision",
            "instruction",
            "dimensions",
            "penalties",
            "folds",
            "seed",
            "fit",
            "selection",
            "optimizer",
            "cost_model",
            "score",
            "incremental_cost_floor_credits",
        )
    },
    "native": native.PROTOCOL,
    "folds": "One assignment from all2000 source families; nested500 keeps its original family folds.",
    "primary_target_pp": 1,
    "secondary_targets_pp": [0, 2, 5],
    "test_boundary": "Save every model, input-only test score and calibration cutoff before reading fresh test answers.",
    "desired_result": "At least30% net saving near1pp quality loss is a goal, not a promised or guaranteed result.",
    "optional_teacher_comparison": (
        "Question-only teacher answers may replace training references in a matched pair of embedding heads. "
        "Both use the same reference-usable subset, original family folds, penalty grid and physical costs; "
        "conditional cost means are recomputed under each label source. No benchmark fallback. "
        "Include both only if their heads are saved before test access."
    ),
}


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def code_hashes():
    names = (
        "arithmetic_study.py",
        "arithmetic_answers.py",
        "teacher_format.py",
        "benefit_outcomes.py",
        "benefit_outcomes_native.py",
        "finetune.py",
        "solver.py",
        "solver_answers.py",
        "solver_gold.py",
        "solver_report.py",
        "benefit_costs.py",
        "serving.py",
        "data.py",
    )
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def checked_source(directory, *, include_gold=False):
    directory = Path(directory)
    plan = read(directory / "source-plan.json")
    require(plan.get("format") == SOURCE_FORMAT and plan.get("counts") == COUNTS, "Arithmetic source protocol differs")
    require(
        all(
            file_hash(directory / name) == expected
            for name, expected in plan["files_sha256"].items()
            if include_gold or name != "gold-for-scoring-only.json"
        ),
        "Prepared arithmetic source changed",
    )
    inputs, metadata = read(directory / "inputs.json"), read(directory / "rows.json")
    families = {}
    all_ids = []
    for split in ("train", "calibration", "test"):
        ids = [row["id"] for row in inputs[split]]
        require(
            len(ids) == COUNTS[split]
            and ids == [row["id"] for row in metadata[split]]
            and all(set(row) == {"id", "request"} for row in inputs[split]),
            "Input order, metadata or counts changed",
        )
        all_ids.extend(ids)
        for row in metadata[split]:
            previous = families.setdefault(row["family"], split)
            require(previous == split, "A question family crosses study roles")
    require(len(set(all_ids)) == len(all_ids), "Question identities cross study roles")
    small = read(directory / "train-small-ids.json")
    small_ids = set(small)
    train_ids = {row["id"] for row in inputs["train"]}
    require(
        len(small) == len(small_ids) == COUNTS["train_small"] and small_ids <= train_ids,
        "The nested training subset changed",
    )
    for family in {row["family"] for row in metadata["train"]}:
        members = {row["id"] for row in metadata["train"] if row["family"] == family}
        require(not members & small_ids or members <= small_ids, "Nested training splits a source family")
    return plan, inputs, metadata


def prepare(source, run, reference_embedding, reference_native):
    source, run, reference_embedding, reference_native = map(Path, (source, run, reference_embedding, reference_native))
    if run.exists():
        raise FileExistsError(run)
    _, inputs, metadata = checked_source(source)
    groups = [row["family"] for row in metadata["train"]]
    folds = benefit.group_folds(groups)
    plan = {
        "format": "sie-arithmetic-adaptation-plan-v1",
        "created_at": benefit.now(),
        "protocol": PROTOCOL,
        "methods": list(METHODS),
        "optional_methods": list(OPTIONAL_METHODS),
        "source_plan_sha256": file_hash(source / "source-plan.json"),
        "training_ids": [row["id"] for row in inputs["train"]],
        "training_families": groups,
        "training_folds": folds,
        "small_training_ids": read(source / "train-small-ids.json"),
        "warm_probe_ids": [
            row["id"] for row in sorted(inputs["train"], key=lambda row: digest(["arithmetic-warm-v1", row["id"]]))[:21]
        ],
        "reference_embedding_files_sha256": {
            name: file_hash(reference_embedding / name) for name in ("selected-model.json", "frozen.json")
        },
        "reference_native_files_sha256": {
            name: file_hash(reference_native / name) for name in ("trained.json", "frozen.json")
        },
        "source_code_sha256": code_hashes(),
        "exposure": "The previous2000 arithmetic evaluation outcomes are development data; none are fresh test examples.",
    }
    write(run / "plan.json", plan)
    return plan


def checked_plan(source, run):
    plan = read(Path(run) / "plan.json")
    _, inputs, metadata = checked_source(source)
    require(
        plan.get("format") == "sie-arithmetic-adaptation-plan-v1"
        and plan.get("protocol") == PROTOCOL
        and plan.get("methods") == list(METHODS)
        and plan.get("optional_methods") == list(OPTIONAL_METHODS)
        and plan.get("source_plan_sha256") == file_hash(Path(source) / "source-plan.json")
        and plan.get("training_ids") == [row["id"] for row in inputs["train"]]
        and plan.get("training_families") == [row["family"] for row in metadata["train"]]
        and plan.get("small_training_ids") == read(Path(source) / "train-small-ids.json")
        and plan.get("source_code_sha256") == code_hashes(),
        "Prepared experiment, source or protocol changed",
    )
    require(
        plan["training_folds"] == benefit.group_folds(plan["training_families"]),
        "Training family folds changed",
    )
    return plan, inputs, metadata


def unopened(run):
    for name in ("test-access.json", "grades-test.json", "report.json"):
        if (Path(run) / name).exists():
            raise FileExistsError(Path(run) / name)


def _grade_pair(item):
    question, meta, reference, records, hashes, rates = item
    row, unresolved = {"id": question["id"], "family": meta["family"]}, []
    for arm in outcomes.ARMS:
        record = records[arm]
        try:
            measured = arithmetic_answers.grade_reference(question["request"], reference, record["answer"])
        except Exception as error:  # noqa: BLE001 - collect every unresolved row, then fail the whole grading phase.
            measured = {"correct": False, "status": "grading_error", "error_type": type(error).__name__}
        if measured["status"] in UNRESOLVED:
            unresolved.append(
                {
                    "id": question["id"],
                    "arm": arm,
                    "status": measured["status"],
                    "question": question["request"],
                    "reference": reference,
                    "final_answer": measured.get("final_answer"),
                    "record_sha256": hashes[arm],
                }
            )
        calls = benefit_costs.physical_costs(record, rates[arm])
        row.update(
            {
                f"{arm}_correct": measured["correct"],
                f"{arm}_grade": measured,
                f"{arm}_record_sha256": hashes[arm],
                f"{arm}_credits": sum(call["credits_proxy"] for call in calls),
                f"{arm}_cost_calls": calls,
            }
        )
    return row, unresolved


def grade(source, run, collection, prices, split, *, workers=8):
    source, run, collection, prices = map(Path, (source, run, collection, prices))
    plan, inputs, metadata = checked_plan(source, run)
    if split not in ("train", "calibration", "test"):
        raise ValueError("Choose train, calibration or test")
    require(type(workers) is int and 1 <= workers <= 16, "Use 1..16 grading processes")
    output = run / f"grades-{split}.json"
    if output.exists():
        raise FileExistsError(output)
    if split == "test":
        access = read(run / "test-access.json")
        require(
            access["frozen_sha256"] == file_hash(run / "frozen.json")
            and access["study_plan_sha256"] == file_hash(run / "plan.json"),
            "Freeze every model and cutoff before opening fresh test answers",
        )
    else:
        unopened(run)
    checked_source(source, include_gold=True)
    acquisition_plan = read(collection / "collection-plan.json")
    acquisition = read(collection / "collection.json")
    source_plan = read(source / "source-plan.json")
    expected_source = (
        source_plan["exposed_source_plan_sha256"] if split == "train" else file_hash(source / "fresh/source-plan.json")
    )
    expected_count = COUNTS["train"] if split == "train" else COUNTS["calibration"] + COUNTS["test"]
    collection_inputs = inputs["train"] if split == "train" else inputs["calibration"] + inputs["test"]
    require(
        acquisition.get("status") == "complete"
        and acquisition.get("completed") == 2 * expected_count
        and acquisition.get("missing") == 0
        and acquisition.get("unknown_execution_attempts") == 0
        and isinstance(acquisition.get("receipt_sha256"), dict)
        and acquisition_plan.get("source_plan_sha256") == expected_source
        and acquisition_plan.get("settings_sha256") == digest(solver.SETTINGS)
        and set(acquisition_plan["records_sha256"])
        == {f"{row['id']}-{arm}.json" for row in collection_inputs for arm in outcomes.ARMS},
        "Need the complete pinned paired collection",
    )
    for relative, expected in acquisition["receipt_sha256"].items():
        path = Path(relative)
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and isinstance(expected, str)
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected),
            "Invalid acquisition lineage mapping",
        )
    gold = read(source / "gold-for-scoring-only.json")
    rates, public_rows = benefit_costs.rates_from_snapshot(read(prices))
    jobs = []
    for question, meta in zip(inputs[split], metadata[split], strict=True):
        records, hashes = {}, {}
        for arm in outcomes.ARMS:
            path = collection / f"{question['id']}-{arm}.json"
            hashes[arm] = file_hash(path)
            require(
                hashes[arm] == acquisition_plan["records_sha256"].get(path.name),
                "A collected answer differs from the gathered manifest",
            )
            record = read(path)
            solver.validate_record(record, question, arm, solver.SETTINGS)
            records[arm] = record
        jobs.append((question, meta, gold[question["id"]], records, hashes, rates))
    rows, unresolved = [], []
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = executor.map(_grade_pair, jobs, chunksize=8) if executor else map(_grade_pair, jobs)
        for i, (row, flags) in enumerate(results, 1):
            rows.append(row)
            unresolved.extend(flags)
            if i % 100 == 0:
                print(f"Graded {i}/{len(jobs)} {split} pairs; unresolved={len(unresolved)}", flush=True)
    finally:
        if executor:
            executor.shutdown()
    if unresolved:
        write(run / f"unresolved-{split}.json", {"cases": unresolved, "source_code_sha256": code_hashes()})
        raise ValueError(f"{len(unresolved)} unresolved comparisons; retain every row and review the complete list")
    result = {
        "format": "sie-arithmetic-adaptation-grades-v1",
        "split": split,
        "study_plan_sha256": file_hash(run / "plan.json"),
        "source_plan_sha256": plan["source_plan_sha256"],
        "collection_plan_sha256": file_hash(collection / "collection-plan.json"),
        "collection_summary_sha256": file_hash(collection / "collection.json"),
        "acquisition_receipt_mapping_sha256": digest(acquisition["receipt_sha256"]),
        "acquisition_verification": (
            "Every selected final record is hash-verified and validated against the exact request/model/settings. "
            "The unchanged complete acquisition-receipt mapping is bound; optional raw execution lineage "
            "is not required to recompute answer grades."
        ),
        "gold_sha256": file_hash(source / "gold-for-scoring-only.json"),
        "prices_sha256": file_hash(prices),
        "public_price_rows": public_rows,
        "grading_source_sha256": {
            name: code_hashes()[name] for name in ("arithmetic_answers.py", "teacher_format.py", "solver_answers.py")
        },
        "test_access_sha256": file_hash(run / "test-access.json") if split == "test" else None,
        "cost_scope": "Every physical call priced at pinned uncached Cloud rates; not observed Cloud bills or Modal spend.",
        "rows": rows,
    }
    write(output, result)
    return result


def training(source, run, size):
    plan, inputs, metadata = checked_plan(source, run)
    require(size in PROTOCOL["training_sizes"], "Choose the declared500 or2000 training stage")
    grades = read(Path(run) / "grades-train.json")
    require(
        grades["study_plan_sha256"] == file_hash(Path(run) / "plan.json")
        and [row["id"] for row in grades["rows"]] == plan["training_ids"],
        "Training outcomes changed or do not match source order",
    )
    ids = set(plan["small_training_ids"] if size == 500 else plan["training_ids"])
    positions = [i for i, row in enumerate(inputs["train"]) if row["id"] in ids]
    require(len(positions) == size, "Training subset differs")
    return {
        "inputs": [inputs["train"][i] for i in positions],
        "rows": [grades["rows"][i] for i in positions],
        "groups": [metadata["train"][i]["family"] for i in positions],
        "folds": [plan["training_folds"][i] for i in positions],
        "grades": grades,
        "positions": positions,
    }


def fit_embedding(source, run, embeddings, size):
    run, embeddings = Path(run), Path(embeddings)
    unopened(run)
    output = run / f"embedding-{size}.json"
    if output.exists():
        raise FileExistsError(output)
    context = training(source, run, size)
    matrix = vectors(context["inputs"], embeddings)
    require(matrix.shape == (size, PROTOCOL["embedding"]["dimensions"]), "Pinned embedding dimensions changed")
    costs = [[row[f"{arm}_credits"] for arm in outcomes.ARMS] for row in context["rows"]]
    head = outcomes.fit_head(
        matrix,
        context["rows"],
        context["groups"],
        costs,
        fold_assignments=context["folds"],
        progress=lambda value: print(value, flush=True),
    )
    result = {
        "format": "sie-arithmetic-embedding-model-v1",
        "training_rows": size,
        "study_plan_sha256": file_hash(run / "plan.json"),
        "training_grades_sha256": file_hash(run / "grades-train.json"),
        "embedding_files_sha256": {
            name: file_hash(embeddings / name) for name in ("plan.json", "vectors.jsonl.gz", "collection.json")
        },
        "head": head,
        "source_code_sha256": code_hashes(),
    }
    write(output, result)
    return result


def train_native(source, run, size, *, local_files_only=False):
    run = Path(run)
    unopened(run)
    output = run / f"native-{size}"
    if output.exists():
        raise FileExistsError(output)
    context = training(source, run, size)
    targets = np.array([[int(row[f"{arm}_correct"]) for arm in outcomes.ARMS] for row in context["rows"]])
    costs = [[row[f"{arm}_credits"] for arm in outcomes.ARMS] for row in context["rows"]]
    held = np.asarray(context["folds"]) == 0
    plan = {
        "format": "sie-arithmetic-native-plan-v1",
        "protocol": native.PROTOCOL,
        "head_plan_sha256": file_hash(run / "plan.json"),
        "selection_validation_ids": [row["id"] for row, flag in zip(context["inputs"], held, strict=True) if flag],
        "training_ids": [row["id"] for row in context["inputs"]],
        "training_folds": context["folds"],
        "source_code_sha256": code_hashes(),
    }
    write(output / "plan.json", plan)
    return native._train_cuda(
        output,
        run,
        {
            "plan": plan,
            "rows": context["inputs"],
            "targets": targets,
            "held": held,
            "costs": costs,
            "statistics": outcomes.conditional_costs(targets, costs),
            "cost_proxy_provenance": {
                "prices_sha256": context["grades"]["prices_sha256"],
                "cost_scope": context["grades"]["cost_scope"],
            },
            "training_grade_provenance": {key: value for key, value in context["grades"].items() if key != "rows"},
            "training_artifacts_sha256": {
                "study_plan": file_hash(run / "plan.json"),
                "grades": file_hash(run / "grades-train.json"),
                "source_plan": file_hash(Path(source) / "source-plan.json"),
            },
        },
        local_files_only,
    )


def score_native(source, run, reference_native, *, device="cuda"):
    run, reference_native = Path(run), Path(reference_native)
    unopened(run)
    output = run / "native-scores.json"
    if output.exists():
        raise FileExistsError(output)
    plan, inputs, _ = checked_plan(source, run)
    require(
        all(
            file_hash(reference_native / name) == expected
            for name, expected in plan["reference_native_files_sha256"].items()
        ),
        "Reference native model metadata changed",
    )
    populations = {split: inputs[split] for split in ("calibration", "test")}
    models = {}
    for label, directory, variants in (
        ("reference", reference_native, ("base", "lora")),
        ("500", run / "native-500", ("lora",)),
        ("2000", run / "native-2000", ("lora",)),
    ):
        trained = read(directory / "trained.json") if label == "reference" else checked_native(source, run, int(label))
        statistics = trained["conditional_cost_statistics"]
        for variant in variants:
            manifest = trained["checkpoints"][variant]["files_sha256"]
            require(
                all(file_hash(directory / variant / name) == expected for name, expected in manifest.items()),
                "Native checkpoint differs from its training export",
            )
            key = f"{label}_{variant}"
            models[key] = {"trained_sha256": file_hash(directory / "trained.json"), "splits": {}}
            for split, rows in populations.items():
                margins = native.checkpoint_margins(directory / variant, rows, device)
                probabilities = native.probabilities(margins)
                scores, expected = outcomes.normalized_scores(probabilities, statistics)
                models[key]["splits"][split] = {
                    "ids": [row["id"] for row in rows],
                    "margins": margins.tolist(),
                    "probabilities": probabilities.tolist(),
                    "scores": scores.tolist(),
                    "expected_costs_proxy": expected.tolist(),
                }
    result = {
        "format": "sie-arithmetic-native-scores-v1",
        "study_plan_sha256": file_hash(run / "plan.json"),
        "source_plan_sha256": file_hash(Path(source) / "source-plan.json"),
        "source_code_sha256": code_hashes(),
        "models": models,
        "solver_test_answers_read": False,
    }
    write(output, result)
    return result


def checked_native(source, run, size):
    run = Path(run)
    context = training(source, run, size)
    directory = run / f"native-{size}"
    plan = read(directory / "plan.json")
    trained = benefit.checked(read(directory / "trained.json"), native.TRAINED_FORMAT)
    targets = [[int(row[f"{arm}_correct"]) for arm in outcomes.ARMS] for row in context["rows"]]
    costs = [[row[f"{arm}_credits"] for arm in outcomes.ARMS] for row in context["rows"]]
    held = np.asarray(context["folds"]) == 0
    selected = native.selected_epoch(trained["selection_history"], np.asarray(targets)[held])
    start = read(directory / "training-start.json")
    require(
        plan["protocol"] == native.PROTOCOL
        and plan["source_code_sha256"] == code_hashes()
        and plan["head_plan_sha256"] == file_hash(run / "plan.json")
        and plan["training_ids"] == [row["id"] for row in context["inputs"]]
        and plan["training_folds"] == context["folds"]
        and trained["plan_sha256"] == file_hash(directory / "plan.json")
        and trained["head_plan_sha256"] == file_hash(run / "plan.json")
        and trained["training_artifacts_sha256"]["grades"] == file_hash(run / "grades-train.json")
        and trained["training_start_sha256"] == file_hash(directory / "training-start.json")
        and start["plan_sha256"] == file_hash(directory / "plan.json")
        and start["runtime"] == native.PROTOCOL["runtime"]
        and start["eager_flags"] == native.PROTOCOL["eager_flags"]
        and {name: version.split("+", 1)[0] for name, version in start["versions"].items()}
        == native.PROTOCOL["versions"]
        and plan["selection_validation_ids"]
        == [row["id"] for row, flag in zip(context["inputs"], held, strict=True) if flag]
        and trained["selected_epoch"] == selected
        and len(trained["refit_losses"]) == selected
        and set(trained["checkpoints"]) == {"base", "lora"}
        and trained["training_targets"] == targets
        and trained["training_cost_proxies"] == costs
        and trained["conditional_cost_statistics"] == outcomes.conditional_costs(targets, costs)
        and all(file_hash(directory / name) == sha for name, sha in trained["receipts_sha256"].items()),
        "Native checkpoint training labels, protocol or receipts changed",
    )
    return trained


def checked_embedding(source, run, size):
    run = Path(run)
    context = training(source, run, size)
    model = read(run / f"embedding-{size}.json")
    head = model["head"]
    targets = [[int(row[f"{arm}_correct"]) for arm in outcomes.ARMS] for row in context["rows"]]
    costs = [[row[f"{arm}_credits"] for arm in outcomes.ARMS] for row in context["rows"]]
    require(
        model["training_rows"] == size
        and model["study_plan_sha256"] == file_hash(run / "plan.json")
        and model["training_grades_sha256"] == file_hash(run / "grades-train.json")
        and model["source_code_sha256"] == code_hashes()
        and head["training_ids"] == [row["id"] for row in context["inputs"]]
        and head["folds"] == context["folds"]
        and head["targets"] == targets
        and head["training_cost_proxies"] == costs
        and head["conditional_cost_statistics"] == outcomes.conditional_costs(targets, costs),
        "Embedding model training labels, folds or protocol changed",
    )
    return model


def overhead(directory, ids):
    receipt = read(Path(directory) / "collection.json")
    costs = {}
    for call in receipt["calls"]:
        require(bool(call["ids"]) and solver.valid_credit(call.get("credits")), "Need measured embedding charges")
        for identity in call["ids"]:
            require(identity not in costs, "Embedding input charged in multiple batches")
            costs[identity] = call["credits"] / len(call["ids"])
    require(set(ids) <= set(costs), "Embedding charges do not cover selected inputs")
    return [costs[identity] for identity in ids]


def freeze(source, run, embeddings, reference_embedding, reference_native):
    run, embeddings, reference_embedding, reference_native = map(
        Path, (run, embeddings, reference_embedding, reference_native)
    )
    unopened(run)
    if (run / "frozen.json").exists():
        raise FileExistsError(run / "frozen.json")
    plan, inputs, _ = checked_plan(source, run)
    calibration = read(run / "grades-calibration.json")
    require(
        calibration["study_plan_sha256"] == file_hash(run / "plan.json")
        and [row["id"] for row in calibration["rows"]] == [row["id"] for row in inputs["calibration"]],
        "Calibration outcomes differ from saved source order",
    )
    require(
        all(
            file_hash(reference_embedding / name) == expected
            for name, expected in plan["reference_embedding_files_sha256"].items()
        )
        and all(
            file_hash(reference_native / name) == expected
            for name, expected in plan["reference_native_files_sha256"].items()
        ),
        "Reference model metadata changed",
    )
    matrices = {split: vectors(inputs[split], embeddings) for split in ("calibration", "test")}
    embedding_costs = {split: overhead(embeddings, [row["id"] for row in inputs[split]]) for split in matrices}
    methods = {}
    reference = read(reference_embedding / "selected-model.json")
    old_embedding = read(reference_embedding / "frozen.json")
    embedding_models = [
        ("embedding_calibration_only", reference),
        ("embedding_500", checked_embedding(source, run, 500)),
        ("embedding_2000", checked_embedding(source, run, 2000)),
    ]
    teacher_path = run / "teacher-models.json"
    if teacher_path.exists():
        teacher = read(teacher_path)
        require(
            teacher["study_plan_sha256"] == file_hash(run / "plan.json")
            and teacher["training_grades_sha256"] == file_hash(run / "grades-train.json")
            and set(teacher["heads"]) == set(OPTIONAL_METHODS),
            "Teacher comparison must bind this study and both matched heads",
        )
        ids = teacher["heads"][OPTIONAL_METHODS[0]]["training_ids"]
        require(
            ids
            and set(ids) <= set(plan["training_ids"])
            and ids == [identity for identity in plan["training_ids"] if identity in set(ids)]
            and all(head["training_ids"] == ids for head in teacher["heads"].values())
            and all(
                head["folds"]
                == [
                    fold
                    for identity, fold in zip(plan["training_ids"], plan["training_folds"], strict=True)
                    if identity in set(ids)
                ]
                for head in teacher["heads"].values()
            ),
            "Teacher and reference heads must use the same source-ordered rows and original folds",
        )
        embedding_models.extend((name, {"head": head}) for name, head in teacher["heads"].items())
    for name, model in embedding_models:
        head = model["head"]
        scores = {}
        for split, matrix in matrices.items():
            probabilities = np.column_stack(
                [outcomes.probability(matrix, head["models"][arm]) for arm in outcomes.ARMS]
            )
            scores[split] = outcomes.normalized_scores(probabilities, head["conditional_cost_statistics"])[0].tolist()
        methods[name] = {
            "design": "embedding",
            "training_rows": len(head["training_ids"]),
            "scores": scores,
            "router_credits": embedding_costs,
        }
    methods["embedding_original_cutoff"] = {
        **methods["embedding_calibration_only"],
        "original_policies": old_embedding["selections"],
    }
    native_scores = read(run / "native-scores.json")
    require(
        native_scores["study_plan_sha256"] == file_hash(run / "plan.json")
        and native_scores["source_code_sha256"] == code_hashes(),
        "Native scores belong to a different study",
    )
    old_native = read(reference_native / "frozen.json")
    for name, key, size in (
        ("base", "reference_base", 0),
        ("lora_calibration_only", "reference_lora", 2503),
        ("lora_500", "500_lora", 500),
        ("lora_2000", "2000_lora", 2000),
    ):
        model = native_scores["models"][key]
        trained_path = (
            reference_native / "trained.json" if key.startswith("reference_") else run / f"native-{size}/trained.json"
        )
        if not key.startswith("reference_"):
            checked_native(source, run, size)
        require(
            model["trained_sha256"] == file_hash(trained_path)
            and all(model["splits"][split]["ids"] == [row["id"] for row in inputs[split]] for split in matrices),
            "Native scores differ from calibration/test question order",
        )
        methods[name] = {
            "design": "native",
            "training_rows": size,
            "cost_model_source": "competition training" if key.startswith("reference_") else "arithmetic training",
            "scores": {split: model["splits"][split]["scores"] for split in matrices},
            "router_credits": {split: None for split in matrices},
        }
    methods["lora_original_cutoff"] = {
        **methods["lora_calibration_only"],
        "original_policies": old_native["models"]["lora"]["policies"],
    }
    for model in methods.values():
        model["policies"] = model.pop("original_policies", None) or {
            str(target): solver_report.select_cutoff(
                calibration["rows"],
                model["scores"]["calibration"],
                target,
                router_credits=model["router_credits"]["calibration"],
                credit_usd=1e-5,
            )
            for target in TARGETS
        }
    expected_methods = set(METHODS) | (set(OPTIONAL_METHODS) if teacher_path.exists() else set())
    require(set(methods) == expected_methods, "The declared method set changed")
    result = {
        "format": "sie-arithmetic-adaptation-frozen-v1",
        "study_plan_sha256": file_hash(run / "plan.json"),
        "calibration_grades_sha256": file_hash(run / "grades-calibration.json"),
        "model_artifacts_sha256": {
            name: file_hash(run / name)
            for name in [
                "embedding-500.json",
                "embedding-2000.json",
                "native-500/trained.json",
                "native-2000/trained.json",
                "native-scores.json",
            ]
            + (["teacher-models.json"] if teacher_path.exists() else [])
        },
        "embedding_files_sha256": {
            name: file_hash(embeddings / name) for name in ("plan.json", "vectors.jsonl.gz", "collection.json")
        },
        "source_code_sha256": code_hashes(),
        "ids": {split: [row["id"] for row in inputs[split]] for split in matrices},
        "methods": methods,
        "teacher_comparison": teacher.get("summary") if teacher_path.exists() else None,
    }
    write(run / "frozen.json", result)
    return result


def authorize_test(source, run):
    run = Path(run)
    unopened(run)
    _, inputs, _ = checked_plan(source, run)
    frozen = read(run / "frozen.json")
    require(
        frozen["study_plan_sha256"] == file_hash(run / "plan.json")
        and frozen["ids"]["test"] == [row["id"] for row in inputs["test"]]
        and set(frozen["methods"]) in (set(METHODS), set(METHODS) | set(OPTIONAL_METHODS))
        and all(file_hash(run / name) == expected for name, expected in frozen["model_artifacts_sha256"].items()),
        "Models, predictions or cutoff freeze changed",
    )
    result = {
        "format": "sie-arithmetic-adaptation-test-access-v1",
        "frozen_sha256": file_hash(run / "frozen.json"),
        "study_plan_sha256": file_hash(run / "plan.json"),
        "authorized_at": benefit.now(),
        "statement": "Every declared model and cutoff is saved before fresh test solver answers are read.",
    }
    write(run / "test-access.json", result)
    return result


def report(source, run):
    run = Path(run)
    if (run / "report.json").exists():
        raise FileExistsError(run / "report.json")
    _, inputs, _ = checked_plan(source, run)
    frozen, access, grades = (read(run / name) for name in ("frozen.json", "test-access.json", "grades-test.json"))
    require(
        access["frozen_sha256"] == file_hash(run / "frozen.json")
        and grades["test_access_sha256"] == file_hash(run / "test-access.json")
        and [row["id"] for row in grades["rows"]] == [row["id"] for row in inputs["test"]],
        "Test outcomes, model freeze or access changed",
    )
    rows, methods = grades["rows"], {}
    for name, model in frozen["methods"].items():
        policies = {}
        values = np.asarray(model["scores"]["test"])
        for target, policy in model["policies"].items():
            mode = policy["mode"]
            decisions = (values >= policy["cutoff"]).tolist() if mode == "threshold" else mode
            credits = model["router_credits"]["test"] if mode == "threshold" else 0
            policies[target] = {
                "mode": mode,
                "cutoff": policy["cutoff"],
                "metrics": solver_report.summarize(rows, decisions, router_credits=credits),
            }
        methods[name] = {
            "label": name.replace("_", " "),
            "design": model["design"],
            "training_rows": model["training_rows"],
            "policies": policies,
            "show_by_default": not name.endswith("original_cutoff"),
            "cost_model_source": model.get(
                "cost_model_source",
                "competition training"
                if "calibration_only" in name or "original_cutoff" in name
                else "arithmetic training",
            ),
        }
    result = {
        "format": "sie-arithmetic-adaptation-report-v1",
        "n": len(rows),
        "ids": [row["id"] for row in rows],
        "source_plan_sha256": file_hash(Path(source) / "source-plan.json"),
        "test_grades_sha256": file_hash(run / "grades-test.json"),
        "frozen_sha256": file_hash(run / "frozen.json"),
        "primary_target_pp": 1,
        "baselines": {mode: solver_report.summarize(rows, mode, router_credits=0) for mode in solver_report.BASELINES},
        "methods": methods,
        "teacher_comparison": frozen.get("teacher_comparison"),
        "cost_scope": grades["cost_scope"] + " Embeddings use observed Cloud charges; native hosting remains unpriced.",
        "population": "Fresh3000 source-family-disjoint GSM8K source-train questions; known2000 used only for training/development.",
        "uncertainty": "Targets are calibration choices, not guarantees; retain every declared target and paired interval.",
    }
    write(run / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--reference-embedding", type=Path, required=True)
    preparation.add_argument("--reference-native", type=Path, required=True)
    grading = commands.add_parser("grade")
    grading.add_argument("--split", choices=("train", "calibration", "test"), required=True)
    grading.add_argument("--collection", type=Path, required=True)
    grading.add_argument("--prices", type=Path, required=True)
    grading.add_argument("--workers", type=int, default=8)
    embedding = commands.add_parser("fit-embedding")
    embedding.add_argument("--size", type=int, choices=(500, 2000), required=True)
    embedding.add_argument("--embeddings", type=Path, required=True)
    training = commands.add_parser("train-native")
    training.add_argument("--size", type=int, choices=(500, 2000), required=True)
    training.add_argument("--local-files-only", action="store_true")
    scoring = commands.add_parser("score-native")
    scoring.add_argument("--reference-native", type=Path, required=True)
    scoring.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--embeddings", type=Path, required=True)
    freezing.add_argument("--reference-embedding", type=Path, required=True)
    freezing.add_argument("--reference-native", type=Path, required=True)
    commands.add_parser("authorize-test")
    commands.add_parser("report")
    args = parser.parse_args()
    common = (args.source, args.run)
    if args.command == "prepare":
        result = prepare(*common, args.reference_embedding, args.reference_native)
    elif args.command == "grade":
        result = grade(*common, args.collection, args.prices, args.split, workers=args.workers)
    elif args.command == "fit-embedding":
        result = fit_embedding(*common, args.embeddings, args.size)
    elif args.command == "train-native":
        result = train_native(*common, args.size, local_files_only=args.local_files_only)
    elif args.command == "score-native":
        result = score_native(*common, args.reference_native, device=args.device)
    elif args.command == "freeze":
        result = freeze(*common, args.embeddings, args.reference_embedding, args.reference_native)
    elif args.command == "authorize-test":
        result = authorize_test(*common)
    else:
        result = report(*common)
    print(
        json.dumps({key: value for key, value in result.items() if key in ("format", "n", "training_rows")}, indent=2)
    )


if __name__ == "__main__":
    main()
