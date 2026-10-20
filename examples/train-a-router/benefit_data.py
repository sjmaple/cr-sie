"""Prepare and grade paired solver outcomes on the original 512 training rows.

This is a separate source-only review protocol. The published validation/test
grader is unchanged. Its labels are automated solver-outcome grades against
reviewed benchmark reference answers, not human difficulty or teacher labels.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import ROUND_HALF_UP, Decimal, localcontext
from functools import lru_cache
from pathlib import Path

import numpy as np

import evidence
import solver
from data import (
    SOURCE_HASHES,
    SOURCE_REVISION,
    audit,
    canonical,
    digest,
    download_source,
    file_hash,
    render_request,
    write,
)
from serving import INSTRUCTION, MODEL, REVISION, vectors
from solver_answers import extract_final_answer, grade_answer, grader_metadata, preflight_gold
from solver_gold import grade_reviewed

PROTOCOL = "source-only-benefit-training-gold-v1"
# ID and original text are checked against the pinned Parquet before use. These
# rules were decided from source questions/solutions before solver collection.
RULES = {
    "e2h-amc-eval-287": ("1,-1, i,-i", r"\{1,-1,i,-i\}", "finite_set"),
    "e2h-amc-train-966": ("251,750,0", "251750000", "source_correction"),
    "e2h-amc-eval-605": (
        r"\left(\frac{1-\sqrt{3}}{2}, \frac{3-\sqrt{3}}{2}\right)",
        r"(\frac{1-\sqrt{3}}{2},\frac{3-\sqrt{3}}{2})",
        "ordered_tuple",
    ),
    "e2h-amc-eval-2451": (
        "614.519...",
        "1129.41806919024220383020154465861873859432518364370131999664",
        "prime_approximation",
    ),
    "e2h-amc-train-574": ("1,2,4", r"\{1,2,4\}", "finite_set"),
    "e2h-amc-train-230": (
        r"(1,2),(4,2),\left(\frac{5}{2}, 2 \pm \frac{3}{2} i\right)",
        r"\{(1,2),(4,2),(\frac52,2+\frac32 i),(\frac52,2-\frac32 i)\}",
        "set_of_ordered_tuples",
    ),
    "e2h-amc-eval-2684": ("(13,-22)", "(13,-22)", "ordered_tuple"),
    "e2h-amc-eval-2430": (r"\sqrt{\frac{2}{7}}", r"\frac27", "source_correction"),
    "e2h-amc-eval-1984": ("80", r"\frac{3400}{41}\%", "rounded_percentage"),
    "e2h-amc-eval-2931": (r"1 \%", r"1\%", "percentage"),
    "e2h-amc-eval-1324": ("20", r"20\%", "percentage"),
    "e2h-amc-eval-759": ("75", r"75\%", "percentage"),
    "e2h-amc-eval-225": ("40", r"40\%", "percentage"),
    "e2h-amc-train-649": ("70", r"70\%", "percentage"),
    "e2h-amc-train-76": ("70", r"70\%", "percentage"),
}
REASONS = {
    "e2h-amc-eval-287": "Subtracting k times the first polynomial from the second gives a*(1-k^4)=0; all four fourth roots are required.",
    "e2h-amc-train-966": "The first five divisors are 1,2,4,5,8; 2014000000/8=251750000. The source prose agrees and its box lost zeros.",
    "e2h-amc-eval-605": "The equilateral-triangle construction returns an ordered point; swapping coordinates changes the answer.",
    "e2h-amc-eval-2451": "Exact integer DP W[0]=1, W[n]=sum(W[n-p] for prime p<=n) counts ordered prime tuples. log2(W[2013])=1129.418069190242..., contradicting the source heuristic 614.519.... The source awards partial credit; binary success here means at least 24 of its 25 points: min(C/A,A/C)>=123/125. This is a declared approximation rubric, not exact-answer grading.",
    "e2h-amc-train-574": "Direct primality checks for all n=1..15 give exactly {1,2,4} for n^n+1 prime.",
    "e2h-amc-train-230": "Subtracting the coupled equations gives (2a-5)*(b-2)=0. The four explicit ordered pairs satisfy both original equations; plus/minus denotes two points.",
    "e2h-amc-eval-2684": "The square spiral reaches (-22,-22) at step1980 and moves35 east to (13,-22) at step2015.",
    "e2h-amc-eval-2430": "The source equation is .7*(1-p)=.5, so p=2/7. Its derivation agrees; the boxed square root is erroneous.",
    "e2h-amc-eval-1984": "The exact decrease from41 to7 is3400/41 percent. Accept the source's coarse80 percent, exact equivalents, and decimal values correctly rounded to their stated precision (for example83,82.9,82.93 percent). Other approximate answers are not silently accepted.",
}
CORRECTIONS = {"e2h-amc-train-966", "e2h-amc-eval-2430", "e2h-amc-eval-2451"}
EXCLUSIONS = {
    "e2h-amc-eval-661": {
        "original_gold": "(3, 4)",
        "reason": "The question asks which interval contains a root, but the original multiple-choice options are missing. Infinitely many intervals qualify; requiring the reference (3,4) would reject mathematically valid answers. Excluded from training before any solver outcomes were collected or inspected.",
    },
}


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {
        "grader_source_sha256": file_hash(Path(__file__).with_name("solver_answers.py")),
        "gold_review_source_sha256": file_hash(Path(__file__).with_name("solver_gold.py")),
        "training_gold_review_source_sha256": file_hash(Path(__file__)),
        "collector_source_sha256": file_hash(Path(__file__).with_name("solver.py")),
    }


@lru_cache(maxsize=1)
def prime_compositions():
    """Independently reconstruct the exceptional source answer with integer DP."""
    primes = []
    for number in range(2, 2014):
        if all(number % prime for prime in primes if prime * prime <= number):
            primes.append(number)
    ways = [1] + [0] * 2013
    for total in range(1, 2014):
        ways[total] = sum(ways[total - prime] for prime in primes if prime <= total)
    with localcontext() as context:
        context.prec = 60
        logarithm = Decimal(ways[2013]).ln() / Decimal(2).ln()
    return str(ways[2013]), str(logarithm)


def scalar_decimal(final):
    """Read a bounded decimal or literal fraction, never an arbitrary expression."""
    text = re.sub(r"\\[,;! ]|\s+", "", final.replace(r"\left", "").replace(r"\right", ""))
    text = re.sub(r"^[a-zA-Z]=", "", text)
    if len(text) > 160:
        return None, None
    number = r"[+-]?(?:\d{1,60}(?:\.\d{1,60})?|\.\d{1,60})"
    if re.fullmatch(number, text):
        return Decimal(text), text
    fraction = re.fullmatch(r"\\(?:dfrac|frac)\{(" + number + r")\}\{(" + number + r")\}", text)
    if fraction is None:
        fraction = re.fullmatch(r"(" + number + r")/(" + number + r")", text)
    if fraction is not None and Decimal(fraction[2]):
        with localcontext() as context:
            context.prec = 70
            return Decimal(fraction[1]) / Decimal(fraction[2]), None
    return None, None


def special_result(correct, final, kind, expected, observed):
    return {
        "correct": bool(correct),
        "status": "graded" if final else "no_final_answer",
        "final_answer": final,
        "gold": {"kind": kind, "value": str(expected)},
        "prediction": {"kind": kind, "value": str(observed) if observed is not None else None},
    }


def grade_training(identity, original, response):
    """Apply only this predeclared training-source review to an explicit answer."""
    if identity not in RULES:
        return grade_reviewed(identity, original, response)
    expected_original, answer, kind = RULES[identity]
    if original != expected_original:
        raise ValueError("Training adjudication differs from the original source answer")
    final = extract_final_answer(response)
    if kind == "prime_approximation":
        observed, _ = scalar_decimal(final)
        target = Decimal(answer)
        with localcontext() as context:
            context.prec = 70
            ratio = min(observed / target, target / observed) if observed is not None and observed > 0 else None
        result = special_result(ratio is not None and ratio >= Decimal(123) / 125, final, kind, answer, observed)
        result["minimum_source_points"] = 24
    elif kind == "rounded_percentage":
        number = final.removesuffix(r"\%").removesuffix("%")
        observed, literal = scalar_decimal(number)
        with localcontext() as context:
            context.prec = 80
            target = Decimal(3400) / 41
            rounded = (
                target.quantize(Decimal(1).scaleb(Decimal(literal).as_tuple().exponent), rounding=ROUND_HALF_UP)
                if literal
                else None
            )
        expected = answer if "%" in final else answer.removesuffix(r"\%")
        exact = grade_answer(expected, response)
        result = special_result(
            exact["correct"] or observed == 80 or (literal is not None and observed == rounded),
            final,
            kind,
            answer,
            observed,
        )
    elif kind == "percentage":
        expected = answer if "%" in final else answer.removesuffix(r"\%")
        result = grade_answer(expected, response)
    else:
        if kind == "set_of_ordered_tuples" and r"\pm" in final:
            # In this source, one plus/minus inside one point denotes two
            # solutions. A repeated point is still checked by the base grader.
            compact = final.replace(r"\left", "").replace(r"\right", "")
            pairs = re.findall(r"\([^()]*\)", compact)
            if len(pairs) == 3 and sum(r"\pm" in pair for pair in pairs) == 1 and compact.count(r"\pm") == 1:
                expanded = [
                    value
                    for pair in pairs
                    for value in ([pair.replace(r"\pm", "+"), pair.replace(r"\pm", "-")] if r"\pm" in pair else [pair])
                ]
                remainder = compact
                for pair in pairs:
                    remainder = remainder.replace(pair, "", 1)
                if re.fullmatch(r"[\s,{}\\]*", remainder):
                    response = "\\boxed{\\{" + ",".join(expanded) + "\\}}"
        result = grade_answer(answer, response)
    return result | {"adjudication": {"protocol": PROTOCOL, "id": identity, "kind": kind}}


def join_training(dataset, originals):
    inputs, gold, sources = {"train": []}, {}, {}
    for row in dataset["splits"]["train"]:
        match = re.fullmatch(r"e2h-amc-(train|eval)-(0|[1-9]\d*)", row["id"])
        if match is None or int(match[2]) >= len(originals[match[1]]):
            raise ValueError("Unknown training source ID")
        source = originals[match[1]][int(match[2])]
        if (
            row["request"] != render_request(source["tag"], source["problem"])
            or row["contest"] != source["tag"]
            or row["year"] != source["year"]
            or row["id"] in gold
        ):
            raise ValueError("Training source/request join differs or is duplicated")
        if not isinstance(source["answer"], str) or not source["answer"].strip():
            raise ValueError("Training source has no answer")
        inputs["train"].append({"id": row["id"], "request": row["request"]})
        gold[row["id"]] = source["answer"]
        sources[row["id"]] = source
    return inputs, gold, sources


def review_training(gold, sources):
    preflight = preflight_gold(gold)
    corrected, records = dict(gold), []
    for identity, (original, answer, kind) in RULES.items():
        if identity not in gold:
            continue
        if gold[identity] != original or not grade_training(identity, original, "\\boxed{" + answer + "}")["correct"]:
            raise ValueError("Training source review is not bound or cannot grade its canonical answer")
        corrected[identity] = answer
        source = sources[identity]
        records.append(
            {
                "id": identity,
                "original": original,
                "canonical": answer,
                "kind": kind,
                "source_answer_correction": identity in CORRECTIONS,
                "reason": REASONS.get(
                    identity,
                    "The source explicitly asks for a percentage; allow an omitted percent sign, but do not treat a count as a percentage.",
                ),
                "problem_sha256": digest(source["problem"]),
                "solution_sha256": digest(source["solution"]),
            }
        )
    unresolved = {identity: value for identity, value in preflight["needs_review"].items() if identity not in RULES}
    review = {
        "protocol": PROTOCOL,
        "source_hashes": SOURCE_HASHES,
        "original_gold_sha256": digest(gold),
        "canonical_gold_sha256": digest(corrected),
        "approved_ids": sorted(set(gold) - set(unresolved)),
        "adjudicated_ids": sorted(set(gold) & set(RULES)),
        "unresolved_ids": sorted(unresolved),
        "needs_review": unresolved,
        "ready": not unresolved,
        "n": len(gold),
        "excluded": [],
        "correction_records": records,
        "label_meaning": "Automated solver-outcome grades against reviewed benchmark reference answers; not human difficulty labels or a no-reference teacher experiment",
        "review_scope": "Parser-flagged and percentage-context training answers reviewed from original source only; this is not an independent solution audit of all512 problems",
        "prime_composition_check": {"ordered_tuple_count": prime_compositions()[0], "log2": prime_compositions()[1]},
    }
    if prime_compositions()[1] != RULES["e2h-amc-eval-2451"][1]:
        raise ValueError("Independent prime-composition reconstruction differs")
    return corrected, preflight, review


def prepare(reference, source, output):
    if output.exists():
        raise FileExistsError(output)
    evidence.verify(reference)
    dataset = read(reference / "data.json")
    audit(dataset)
    if dataset["source_revision"] != SOURCE_REVISION or len(dataset["splits"]["train"]) != 512:
        raise ValueError("Need the pinned original512 training population")
    download_source(source)
    import pyarrow.parquet as pq

    originals = {}
    for name, expected in SOURCE_HASHES.items():
        if file_hash(source / name) != expected:
            raise ValueError("Training source hash differs")
        originals[name.removesuffix(".parquet")] = pq.read_table(
            source / name, columns=["problem", "answer", "tag", "year", "solution"]
        ).to_pylist()
    pool_inputs, pool_gold, sources = join_training(dataset, originals)
    if (set(RULES) | set(EXCLUSIONS)) - set(pool_gold):
        raise ValueError("The pinned training population is missing reviewed rows")
    for identity, rule in EXCLUSIONS.items():
        if pool_gold[identity] != rule["original_gold"]:
            raise ValueError("Training exclusion does not match its original source")
    inputs = {"train": [row for row in pool_inputs["train"] if row["id"] not in EXCLUSIONS]}
    gold = {row["id"]: pool_gold[row["id"]] for row in inputs["train"]}
    eligibility = {
        "format": "sie-benefit-training-eligibility-v1",
        "original_pool_ids": [row["id"] for row in pool_inputs["train"]],
        "eligible_train_ids": [row["id"] for row in inputs["train"]],
        "exclusions": [
            {"id": identity, **rule, "problem_sha256": digest(sources[identity]["problem"])}
            for identity, rule in EXCLUSIONS.items()
        ],
        "source_only": True,
        "decided_before_training_outcomes": True,
        "source_revision": SOURCE_REVISION,
        "source_hashes": SOURCE_HASHES,
        "dataset_sha256": digest(dataset),
        "training_gold_review_source_sha256": file_hash(Path(__file__)),
    }
    corrected, preflight, review = review_training(gold, sources)
    train, held_out = dataset["splits"]["train"], {}
    for split in ("validation", "calibration", "test"):
        rows = dataset["splits"][split]
        overlaps = {
            key: len({row[key] for row in train} & {row[key] for row in rows})
            for key in ("id", "family", "paper_group")
        }
        overlaps["canonical_request"] = len(
            {canonical(row["request"]) for row in train} & {canonical(row["request"]) for row in rows}
        )
        if any(overlaps.values()):
            raise ValueError("Training overlap with a held-out population")
        held_out[split] = {"n": len(rows), "overlaps": overlaps, "ids_sha256": digest([row["id"] for row in rows])}
    feature_directory = reference / "development-embeddings"
    matrix = vectors(train, feature_directory)
    if matrix.shape != (512, 2560) or not np.isfinite(matrix).all():
        raise ValueError("Frozen training embedding shape or values differ")
    write(output / "inputs.json", inputs)
    write(output / "inputs-original-pool.json", pool_inputs)
    write(output / "gold-original-pool.json", pool_gold)
    write(output / "eligibility.json", eligibility)
    write(output / "gold-for-scoring-only.json", gold)
    write(output / "gold-reviewed.json", corrected)
    write(output / "settings.json", solver.SETTINGS)
    write(output / "gold-preflight.json", preflight)
    review["preflight_sha256"] = file_hash(output / "gold-preflight.json")
    review["eligibility_sha256"] = file_hash(output / "eligibility.json")
    review["original_pool_gold_sha256"] = digest(pool_gold)
    review["excluded"] = eligibility["exclusions"]
    write(output / "training-gold-adjudications.json", review)
    rules_hash = file_hash(output / "training-gold-adjudications.json")
    write(
        output / "gold-review.json",
        {
            key: review[key]
            for key in (
                "preflight_sha256",
                "original_gold_sha256",
                "canonical_gold_sha256",
                "approved_ids",
                "adjudicated_ids",
                "unresolved_ids",
                "ready",
            )
        }
        | {"training_gold_rules_sha256": rules_hash},
    )
    write(
        output / "plan.json",
        {
            "format": "sie-benefit-training-source-v1",
            "dataset_sha256": digest(dataset),
            "inputs_sha256": digest(inputs),
            "gold_sha256": digest(gold),
            "settings_sha256": digest(solver.SETTINGS),
            "counts": {"train": len(inputs["train"])},
            "source_counts": {"train": 512},
            "eligible_train_ids": eligibility["eligible_train_ids"],
            "excluded_train_ids": list(EXCLUSIONS),
            "eligibility_sha256": file_hash(output / "eligibility.json"),
            "source": {"revision": SOURCE_REVISION, "hashes": SOURCE_HASHES},
            "baseline_manifest_sha256": file_hash(reference / "manifest.json"),
            "baseline_data_sha256": file_hash(reference / "data.json"),
            "train_ids_sha256": digest([row["id"] for row in train]),
            "held_out": held_out,
            "features": {
                "model": MODEL,
                "revision": REVISION,
                "instruction": INSTRUCTION,
                "shape": list(matrix.shape),
                "plan_sha256": file_hash(feature_directory / "plan.json"),
                "vectors_sha256": file_hash(feature_directory / "vectors.jsonl.gz"),
            },
            "inference_reads_gold": False,
            "selection": "Original training IDs only; no calibration/test solver outcomes for fitting",
            "grader": grader_metadata(),
            **source_hashes(),
            "training_gold_rules_sha256": rules_hash,
            "gold_review_sha256": file_hash(output / "gold-review.json"),
        },
    )
    if not review["ready"]:
        raise ValueError("Training gold review remains unresolved; inspect prepared artifacts before collection")
    print(
        json.dumps(
            {
                "prepared": str(output),
                "original_pool": 512,
                "eligible": len(gold),
                "reviewed": len(review["adjudicated_ids"]),
                "source_corrections": len(CORRECTIONS),
                "plan_sha256": file_hash(output / "plan.json"),
            }
        )
    )


def checked_training(directory):
    """Check input/protocol bindings without opening either answer-key file."""
    plan, inputs, settings = solver.checked_inputs(directory)
    if (
        plan.get("format") != "sie-benefit-training-source-v1"
        or plan.get("source_counts") != {"train": 512}
        or plan.get("counts") != {"train": 511}
        or set(inputs) != {"train"}
        or len(inputs["train"]) != 511
    ):
        raise ValueError("Need the original512 pool with its declared511 eligible rows")
    if (
        any(set(row) != {"id", "request"} for row in inputs["train"])
        or len({row["id"] for row in inputs["train"]}) != 511
    ):
        raise ValueError("Training inputs contain labels or duplicate identities")
    for key, value in source_hashes().items():
        if plan.get(key) != value:
            raise ValueError("Training collection/grading source changed after preparation")
    for key, name in (
        ("training_gold_rules_sha256", "training-gold-adjudications.json"),
        ("gold_review_sha256", "gold-review.json"),
        ("eligibility_sha256", "eligibility.json"),
    ):
        if plan.get(key) != file_hash(directory / name):
            raise ValueError("Training source review changed after preparation")
    eligibility = read(directory / "eligibility.json")
    if (
        eligibility.get("format") != "sie-benefit-training-eligibility-v1"
        or eligibility.get("source_only") is not True
        or eligibility.get("decided_before_training_outcomes") is not True
        or eligibility.get("eligible_train_ids") != [row["id"] for row in inputs["train"]]
        or plan.get("eligible_train_ids") != eligibility["eligible_train_ids"]
        or [row["id"] for row in eligibility["exclusions"]] != list(EXCLUSIONS)
        or plan.get("excluded_train_ids") != list(EXCLUSIONS)
        or eligibility["eligible_train_ids"]
        != [identity for identity in eligibility["original_pool_ids"] if identity not in EXCLUSIONS]
        or len(set(eligibility["original_pool_ids"])) != 512
    ):
        raise ValueError("Training eligibility differs from the predeclared source-only exclusion")
    solver.check_review(directory, [row["id"] for row in inputs["train"]])
    return plan, inputs, settings


def grade(directory, output=None):
    output = output or directory / "grades-train.json"
    if output.exists():
        raise FileExistsError(output)
    plan, inputs, settings = checked_training(directory)
    if plan["grader"] != grader_metadata():
        raise ValueError("Training grading dependency or parser protocol changed")
    gold = read(directory / "gold-for-scoring-only.json")
    if digest(gold) != plan["gold_sha256"] or set(gold) != {row["id"] for row in inputs["train"]}:
        raise ValueError("Training answer key changed after preparation")
    collection = directory / "answers-train"
    summary, collection_plan = read(collection / "collection.json"), read(collection / "collection-plan.json")
    selected = collection_plan["ids"]
    rows = inputs["train"][: len(selected)]
    solver.collection_binding(collection_plan, directory, "train", rows, settings, None, collection_plan["self_hosted"])
    if (
        not selected
        or summary["status"] != "complete"
        or summary["completed"] != 2 * len(rows)
        or summary["failed"] != 0
    ):
        raise ValueError("Training collection is incomplete or its count differs")
    results = []
    for row in rows:
        outcome = {"id": row["id"]}
        for arm in settings["arms"]:
            path = collection / f"{row['id']}-{arm}.json"
            answer = read(path)
            solver.validate_record(answer, row, arm, settings)
            if answer["status"] != "complete":
                raise ValueError("Incomplete training answer")
            measured = grade_training(row["id"], gold[row["id"]], answer["answer"])
            if measured["status"] in ("gold_unparseable", "grading_error"):
                raise ValueError("Training grader could not safely score a reviewed answer")
            outcome.update(
                {
                    f"{arm}_correct": measured["correct"],
                    f"{arm}_credits": answer["credits"],
                    f"{arm}_grade": measured,
                    f"{arm}_record_sha256": file_hash(path),
                }
            )
        results.append(outcome)
    write(
        output,
        {
            "split": "train",
            "plan_sha256": file_hash(directory / "plan.json"),
            "collection_plan_sha256": file_hash(collection / "collection-plan.json"),
            "collection_summary_sha256": file_hash(collection / "collection.json"),
            "policies_frozen_sha256": None,
            **source_hashes(),
            "gold_review_sha256": file_hash(directory / "gold-review.json"),
            "training_gold_rules_sha256": plan["training_gold_rules_sha256"],
            "rows": results,
        },
    )
    print(
        json.dumps(
            {
                "graded": str(output),
                "n": len(results),
                "label_meaning": "Automated solver-outcome grades against reviewed benchmark reference answers",
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    child = sub.add_parser("prepare")
    child.add_argument("--evidence", type=Path, default=Path("evidence"))
    child.add_argument("--source", type=Path, default=Path("run-output/solver-source"))
    child.add_argument("--output", type=Path, required=True)
    child = sub.add_parser("grade")
    child.add_argument("--run", type=Path, required=True)
    child.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.evidence, args.source, args.output)
    else:
        grade(args.run, args.output)


if __name__ == "__main__":
    main()
