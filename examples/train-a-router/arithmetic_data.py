"""Prepare a new arithmetic study with exposed training and fresh grouped splits."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import solver
from data import SOURCE_HASHES, digest, duplicate_components, file_hash, render_request, write
from generalization_data import HASHES, REVISION, SOURCE_URL, final_answer

COUNTS = {"train": 2000, "train_small": 500, "calibration": 1000, "test": 3000}
NEAR_DUPLICATES = {
    "character_ngram": 4,
    "plain_cosine": 0.80,
    "numeric_masked_cosine": 0.85,
    "plain_review_cosine": 0.80,
    "numeric_masked_review_cosine": 0.85,
}
FORMAT = "sie-arithmetic-adaptation-source-v1"


def read(path):
    return json.loads(Path(path).read_text())


def source_rows(source, math_source):
    import pyarrow.parquet as pq

    source, math_source = Path(source), Path(math_source)
    rows = []
    for name, expected in SOURCE_HASHES.items():
        if file_hash(math_source / name) != expected:
            raise ValueError("Pinned competition source changed")
        rows.extend(
            {
                "id": f"e2h-amc-{name.split('.')[0]}-{index}",
                "problem": row["problem"],
                "dataset": "competition",
            }
            for index, row in enumerate(pq.read_table(math_source / name).to_pylist())
        )
    for name, expected in HASHES.items():
        if file_hash(source / name) != expected:
            raise ValueError("Pinned arithmetic source changed")
        rows.extend(
            {
                "id": f"gsm8k-{name.split('.')[0]}-{index}",
                "problem": row["question"],
                "gold": final_answer(row["answer"]),
                "dataset": "arithmetic",
                "original_split": name.split(".")[0],
            }
            for index, row in enumerate(map(json.loads, (source / name).read_text().splitlines()))
        )
    return rows


def _character_matrix(texts, masked):
    from scipy.sparse import csr_matrix

    vocabulary, indices, indptr, values, document_counts = {}, [], [0], [], Counter()
    for text in texts:
        normalized = " ".join(re.findall(r"[a-z]+|\d+", text.casefold()))
        if masked:
            normalized = re.sub(r"\d+", "#", normalized)
        normalized = f" {normalized} "
        grams = Counter(normalized[index : index + 4] for index in range(max(0, len(normalized) - 3)))
        for gram, count in grams.items():
            column = vocabulary.setdefault(gram, len(vocabulary))
            indices.append(column)
            values.append(1 + np.log(count))
            document_counts[column] += 1
        indptr.append(len(indices))
    matrix = csr_matrix((values, indices, indptr), shape=(len(texts), len(vocabulary)), dtype=np.float64)
    idf = np.log((1 + len(texts)) / (1 + np.array([document_counts[i] for i in range(len(vocabulary))]))) + 1
    matrix = matrix.multiply(idf).tocsr()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    if np.any(norms == 0):
        raise ValueError("Every question must contain character features")
    return matrix.multiply(1 / norms[:, None]).tocsr()


def grouped_sources(rows, *, progress=None):
    """Union lexical/template families using only the question text."""
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        parent[max(left, right)] = min(left, right)

    for component in duplicate_components(rows):
        for index in component[1:]:
            union(component[0], index)
    pairs = {}
    for masked, threshold, review_threshold in (
        (False, NEAR_DUPLICATES["plain_cosine"], NEAR_DUPLICATES["plain_review_cosine"]),
        (True, NEAR_DUPLICATES["numeric_masked_cosine"], NEAR_DUPLICATES["numeric_masked_review_cosine"]),
    ):
        matrix = _character_matrix([row["problem"] for row in rows], masked)
        for start in range(0, len(rows), 128):
            similarities = (matrix[start : start + 128] @ matrix.T).toarray()
            for offset, current in enumerate(similarities):
                left = start + offset
                for right in np.flatnonzero(current[:left] >= review_threshold):
                    if rows[left]["dataset"] == rows[int(right)]["dataset"] == "competition":
                        continue
                    pair = tuple(sorted((left, int(right))))
                    entry = pairs.setdefault(pair, {"plain_cosine": None, "numeric_masked_cosine": None})
                    entry["numeric_masked_cosine" if masked else "plain_cosine"] = float(current[right])
                    if current[right] >= threshold:
                        union(*pair)
            if progress is not None and start % 1024 == 0:
                progress(f"Source-only {'masked' if masked else 'plain'} similarity: {start}/{len(rows)}")
    components = {}
    for index in range(len(rows)):
        components.setdefault(find(index), []).append(index)
    review = [
        {
            "ids": [rows[left]["id"], rows[right]["id"]],
            **scores,
            "same_family": find(left) == find(right),
        }
        for (left, right), scores in sorted(pairs.items())
    ]
    return list(components.values()), review


def assign(rows, components, exposed, counts=None):
    counts = COUNTS if counts is None else counts
    known = {row["id"]: row for row in rows}
    exposed_ids = [row["id"] for row in exposed]
    if len(exposed_ids) != counts["train"] or len(set(exposed_ids)) != len(exposed_ids):
        raise ValueError("Exposed training pool count or identities differ")
    for row in exposed:
        if row["id"] not in known or row["request"] != render_request("GSM8K", known[row["id"]]["problem"]):
            raise ValueError("Exposed requests must match the pinned source exactly")
    exposed_set, families, candidates = set(exposed_ids), {}, []
    excluded, grouped = [], []
    for component in components:
        members = [rows[index] for index in component]
        identities = sorted(row["id"] for row in members)
        family = digest(["arithmetic-family-v1", identities])
        for identity in identities:
            families[identity] = family
        arithmetic = [row for row in members if row["dataset"] == "arithmetic"]
        if not arithmetic:
            continue
        touches_exposed = bool(exposed_set.intersection(identities))
        if len(arithmetic) != len(members):
            if touches_exposed:
                raise ValueError("An exposed arithmetic training family overlaps competition source")
            excluded.append({"ids": identities, "reason": "overlaps_competition_source"})
            continue
        if len(members) > 1:
            grouped.append({"ids": identities, "family": family, "touches_exposed": touches_exposed})
        if touches_exposed:
            excluded.extend(
                {"ids": [row["id"]], "reason": "same_family_as_exposed_training"}
                for row in arithmetic
                if row["id"] not in exposed_set
            )
            continue
        if any(row["original_split"] != "train" for row in arithmetic):
            raise ValueError("Every unexposed arithmetic family must come from source train")
        representative = min(arithmetic, key=lambda row: digest(["arithmetic-representative-v1", row["id"]]))
        candidates.append(representative)
    candidates.sort(key=lambda row: digest(["arithmetic-fresh-split-v1", families[row["id"]]]))
    required = counts["calibration"] + counts["test"]
    if len(candidates) < required:
        raise ValueError("Not enough unused disjoint source families for declared splits")
    selected = {
        "train": [known[identity] for identity in exposed_ids],
        "calibration": candidates[: counts["calibration"]],
        "test": candidates[counts["calibration"] : required],
    }
    training_families = {}
    for row in selected["train"]:
        training_families.setdefault(families[row["id"]], []).append(row["id"])
    reachable = {0: ()}
    for family in sorted(training_families, key=lambda key: digest(["arithmetic-small-v1", key])):
        size = len(training_families[family])
        for total, chosen in list(reachable.items()):
            if total + size <= counts["train_small"] and total + size not in reachable:
                reachable[total + size] = chosen + (family,)
        if counts["train_small"] in reachable:
            break
    if counts["train_small"] not in reachable:
        raise ValueError("Whole families cannot satisfy the declared small training count")
    small = [identity for family in reachable[counts["train_small"]] for identity in training_families[family]]
    split_families = {split: {families[row["id"]] for row in population} for split, population in selected.items()}
    if any(
        split_families[left] & split_families[right]
        for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test"))
    ):
        raise ValueError("Source families cross study roles")
    return (
        selected,
        families,
        small,
        {
            "available_unused_families": len(candidates),
            "unused_after_selection": len(candidates) - required,
            "grouped_arithmetic_families": grouped,
            "excluded": excluded,
        },
    )


def prepare(source, math_source, exposed_directory, output, *, progress=None):
    output, exposed_directory = Path(output), Path(exposed_directory)
    if output.exists():
        raise FileExistsError(output)
    exposed_plan = read(exposed_directory / "source-plan.json")
    exposed = read(exposed_directory / "inputs.json")
    if (
        exposed_plan.get("source_revision") != REVISION
        or exposed_plan.get("source_hashes") != HASHES
        or exposed_plan["files_sha256"]["inputs.json"] != file_hash(exposed_directory / "inputs.json")
    ):
        raise ValueError("Exposed development source differs")
    rows = source_rows(source, math_source)
    components, pairs = grouped_sources(rows, progress=progress)
    selected, families, small, audit = assign(rows, components, exposed)
    inputs = {
        split: [{"id": row["id"], "request": render_request("GSM8K", row["problem"])} for row in population]
        for split, population in selected.items()
    }
    metadata = {
        split: [
            {"id": row["id"], "family": families[row["id"]], "original_split": row["original_split"]}
            for row in population
        ]
        for split, population in selected.items()
    }
    audit |= {
        "question_only": True,
        "solver_answers_or_scores_read": False,
        "arithmetic_source_counts": dict(
            Counter(row["original_split"] for row in rows if row["dataset"] == "arithmetic")
        ),
        "grouping": NEAR_DUPLICATES,
        "initial_grouping": "Pinned data.duplicate_components canonical and five-token Jaccard rules",
        "limitation": "Lexical/template grouping does not prove absence of all semantic relatives or model pretraining.",
    }
    write(output / "inputs.json", inputs)
    write(output / "rows.json", metadata)
    write(output / "train-small-ids.json", small)
    write(
        output / "gold-for-scoring-only.json",
        {row["id"]: row["gold"] for population in selected.values() for row in population},
    )
    write(output / "source-audit.json", audit)
    write(output / "near-duplicate-candidates.json", pairs)
    plan = {
        "format": FORMAT,
        "counts": COUNTS,
        "source_revision": REVISION,
        "source_url": SOURCE_URL,
        "source_hashes": HASHES,
        "competition_source_hashes": SOURCE_HASHES,
        "exposed_source_plan_sha256": file_hash(exposed_directory / "source-plan.json"),
        "exposed_inputs_sha256": file_hash(exposed_directory / "inputs.json"),
        "exposed_rows_use": "All prior 2000 arithmetic evaluation rows are now exposed training/development only.",
        "primary_target_pp": 1,
        "secondary_targets_pp": [0, 2, 5],
        "success_target": "Aim for >=30% net savings near 1pp loss; report paired uncertainty, not a guarantee.",
        "comparison": "Unchanged competition models with arithmetic recalibration versus same-recipe 500/2000 arithmetic fits.",
        "sampling": "Question-only connected families; one fresh representative per family; fixed hash order.",
        "settings_sha256": digest(solver.SETTINGS),
        "files_sha256": {
            name: file_hash(output / name)
            for name in (
                "inputs.json",
                "rows.json",
                "train-small-ids.json",
                "gold-for-scoring-only.json",
                "source-audit.json",
                "near-duplicate-candidates.json",
            )
        },
        "source_code_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("arithmetic_data.py", "data.py", "generalization_data.py", "solver.py")
        },
    }
    write(output / "source-plan.json", plan)
    fresh = inputs["calibration"] + inputs["test"]
    write(output / "fresh/inputs.json", fresh)
    write(output / "fresh/settings.json", solver.SETTINGS)
    write(
        output / "fresh/source-plan.json",
        {
            "format": FORMAT,
            "parent_source_plan_sha256": file_hash(output / "source-plan.json"),
            "n": len(fresh),
            "counts": {split: COUNTS[split] for split in ("calibration", "test")},
            "inputs_sha256": digest(fresh),
            "settings_sha256": digest(solver.SETTINGS),
            "reference_answers_staged": False,
        },
    )
    return {
        "counts": COUNTS,
        "available_unused_families": audit["available_unused_families"],
        "source_plan_sha256": file_hash(output / "source-plan.json"),
        "near_duplicate_candidates": len(pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--math-source", type=Path, required=True)
    parser.add_argument("--exposed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.math_source, args.exposed, args.output, progress=print), indent=2))


if __name__ == "__main__":
    main()
