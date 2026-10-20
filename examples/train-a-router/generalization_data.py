"""Prepare an arithmetic transfer test without using any router outcomes.

The competition test contains only 479 clean questions. This separately
declared population measures transfer to arithmetic, not a larger sample of
that same competition distribution. No arithmetic examples train the routers.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import solver
from benefit_expanded_gold_v2 import outcome_grader_metadata
from data import SOURCE_HASHES, digest, duplicate_components, file_hash, render_request, write

REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
HASHES = {
    "train.jsonl": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}
SOURCE_URL = f"https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}/grade_school_math/data/"
COUNT = 2000


def final_answer(solution):
    if solution.count("####") != 1:
        raise ValueError("GSM8K source must have exactly one final-answer marker")
    answer = solution.split("####", 1)[1].strip()
    if re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", answer) is None:
        raise ValueError("Unexpected arithmetic source answer")
    Fraction(answer.replace(",", ""))
    return answer


def prepare(source, math_source, output):
    import pyarrow.parquet as pq

    source, math_source, output = map(Path, (source, math_source, output))
    for name in ("inputs.json", "settings.json", "source-plan.json", "gold-for-scoring-only.json", "source-audit.json"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    rows = []
    for name, expected in SOURCE_HASHES.items():
        if file_hash(math_source / name) != expected:
            raise ValueError("Competition source changed")
        rows.extend(
            {"id": f"e2h-amc-{name.split('.')[0]}-{index}", "problem": row["problem"], "dataset": "competition"}
            for index, row in enumerate(pq.read_table(math_source / name).to_pylist())
        )
    for name, expected in HASHES.items():
        if file_hash(source / name) != expected:
            raise ValueError("Pinned arithmetic source changed")
        for index, line in enumerate((source / name).read_text().splitlines()):
            row = json.loads(line)
            rows.append(
                {
                    "id": f"gsm8k-{name.split('.')[0]}-{index}",
                    "problem": row["question"],
                    "gold": final_answer(row["answer"]),
                    "dataset": "arithmetic",
                    "original_split": name.split(".")[0],
                }
            )
    retained, cross_source, duplicates = [], [], []
    for component in duplicate_components(rows):
        members = [rows[index] for index in component]
        arithmetic = [row for row in members if row["dataset"] == "arithmetic"]
        if not arithmetic:
            continue
        identities = sorted(row["id"] for row in members)
        if len(arithmetic) != len(members):
            cross_source.append(identities)
            continue
        preferred = [row for row in arithmetic if row["original_split"] == "test"] or arithmetic
        representative = min(preferred, key=lambda row: digest(["arithmetic-transfer-representative-v1", row["id"]]))
        retained.append(representative | {"family": digest(identities)})
        if len(members) > 1:
            duplicates.append({"ids": identities, "representative": representative["id"]})
    official = sorted(
        [row for row in retained if row["original_split"] == "test"],
        key=lambda row: digest(["arithmetic-transfer-order-v1", row["id"]]),
    )
    supplement = sorted(
        [row for row in retained if row["original_split"] == "train"],
        key=lambda row: digest(["arithmetic-transfer-order-v1", row["id"]]),
    )
    if not 0 < len(official) <= COUNT <= len(retained):
        raise ValueError("Source cannot satisfy the predeclared transfer population")
    selected = official + supplement[: COUNT - len(official)]
    inputs = [{"id": row["id"], "request": render_request("GSM8K", row["problem"])} for row in selected]
    gold = {row["id"]: row["gold"] for row in selected}
    metadata = [{key: row[key] for key in ("id", "family", "original_split")} for row in selected]
    audit = {
        "source_only": True,
        "solver_outcomes_read": False,
        "router_training_arithmetic_examples": 0,
        "arithmetic_source_rows": sum(row["dataset"] == "arithmetic" for row in rows),
        "eligible_families": len(retained),
        "selected": len(selected),
        "selected_by_original_split": dict(Counter(row["original_split"] for row in selected)),
        "cross_source_families_removed": cross_source,
        "within_arithmetic_duplicate_families": duplicates,
        "duplicate_rule": "Same pinned canonical-text and five-token/numeric-masked Jaccard connected-components rule as data.py. One arithmetic representative per family; exclude any component touching any competition source question.",
        "limitation": "These lexical duplicate checks do not prove the absence of every semantic relationship or pretrained-model contamination.",
    }
    write(output / "inputs.json", inputs)
    write(output / "settings.json", solver.SETTINGS)
    write(output / "gold-for-scoring-only.json", gold)
    write(output / "rows.json", metadata)
    write(output / "source-audit.json", audit)
    plan = {
        "format": "sie-arithmetic-transfer-source-v1",
        "n": COUNT,
        "source_revision": REVISION,
        "source_url": SOURCE_URL,
        "source_hashes": HASHES,
        "competition_source_hashes": SOURCE_HASHES,
        "files_sha256": {
            name: file_hash(output / name)
            for name in ("inputs.json", "settings.json", "gold-for-scoring-only.json", "rows.json", "source-audit.json")
        },
        "inputs_sha256": digest(inputs),
        "settings_sha256": digest(solver.SETTINGS),
        "request_format": "Competition: GSM8K followed by the public question; no answer, solution or original-split marker is supplied",
        "selection": "Keep every eligible official-test family, then identity-hash-selected official-train families to reach2000. None are used in router fitting or cutoff selection.",
        "study": "Separate arithmetic transfer evaluation; not a replacement for the competition test, a standard GSM8K leaderboard score, or an assumed production workload mix.",
        "models": "Apply all already-declared original/expanded Laya and embedding routers plus the teacher-label control unchanged. No transfer training or cutoff retuning.",
        "targets_pp": [1, 0, 2, 5],
        "primary_target_pp": 1,
        "quality": "Paired final-answer correctness relative to the pinned numeric reference; use frozen question-only arithmetic normalization/equivalence rules.",
        "cost": "Self-hosted generation token usage priced with the same explicit uncachedCloud token-price snapshot; not observedCloud bills. Embedding charges, teacher costs and GPU acquisition costs are recorded separately.",
        "precision_planning": "2000 independent questions gives approximately +/-0.98pp two-sided95% precision if routed-vs-strong correctness differs on5%; this is an assumption, not a promised realized interval.",
        "source_code_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("generalization_data.py", "data.py", "solver.py", "benefit_expanded_gold_v2.py")
        },
        **outcome_grader_metadata(),
    }
    write(output / "source-plan.json", plan)
    return {
        "n": COUNT,
        "by_original_split": audit["selected_by_original_split"],
        "cross_source_families_removed": len(cross_source),
        "duplicate_families": len(duplicates),
        "plan_sha256": file_hash(output / "source-plan.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--math-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.math_source, args.output), indent=2))


if __name__ == "__main__":
    main()
