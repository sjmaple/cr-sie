"""Freeze downstream routing on calibration answers, then report fixed test policies.

This CPU-only orchestration reads recorded router scores and solver grades. It
does not call models, collect answers, or select a policy using test answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evidence
import policies
import solver_report
from data import LABELS, digest, file_hash, write
from serving import MODEL, REVISION, vectors

STAGES = ("base", "lora512", "lora2581", "ridge64", "ridge512")
TARGETS = (1, 0, 2, 5)
SCORE_FORMAT = "sie-solver-router-scores-v1"
FROZEN_FORMAT = "sie-solver-policies-v1"
GRADING_HASHES = ("grader_source_sha256", "gold_review_source_sha256", "gold_review_sha256")


def read(path):
    return json.loads(Path(path).read_text())


def seal(document):
    return document | {"sha256": digest(document)}


def check_seal(document, expected_format):
    if document.get("format") != expected_format or document.get("sha256") != digest(
        {key: value for key, value in document.items() if key != "sha256"}
    ):
        raise ValueError("Frozen document format or content hash differs")


def finite_scores(values, count):
    result = np.asarray(values, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError("Expected one finite router score per input")
    return result


def _native_stage(directory, variant, dataset, calibration_sha, test_sha):
    plan, frozen, calibration, test = (
        read(directory / name)
        for name in ("plan.json", "frozen.json", f"{variant}-calibration.json", "evaluation-test.json")
    )
    selected = frozen["selected"][variant]
    if (
        frozen["plan_sha256"] != file_hash(directory / "plan.json")
        or plan["data_sha256"]["calibration.jsonl"] != calibration_sha
        or plan["config"]["seed"] != 89
        or selected["seed"] != 89
        or test["frozen_sha256"] != file_hash(directory / "frozen.json")
        or test["data_sha256"] != test_sha
        or test["ids"] != [row["id"] for row in dataset["splits"]["test"]]
    ):
        raise ValueError("Native scores differ from their frozen model, seed or data")
    threshold = float(selected["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Nonfinite original threshold")
    scores = {}
    for split, measured in (("calibration", calibration), ("test", test["models"][variant])):
        rows = dataset["splits"][split]
        values = finite_scores(measured["scores"], len(rows))
        if (
            measured["labels"] != [row["label"] for row in rows]
            or measured["predictions"] != (values >= threshold).astype(int).tolist()
        ):
            raise ValueError("Native scores, labels or original decisions are not aligned")
        scores[split] = values.tolist()
    return {
        "family": "native",
        "seed": 89,
        "variant": variant,
        "original_threshold": threshold,
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "scores": scores,
    }


def load_stage_scores(baseline, study, policy_directory):
    """Load five public frozen stages, verifying metadata only; weights are optional."""
    baseline, study, policy_directory = map(Path, (baseline, study, policy_directory))
    evidence.verify(baseline)
    evidence.verify(study, manifest_sha256=evidence.LORA_MANIFEST_SHA256)
    dataset = read(baseline / "data.json")
    rows = {split: dataset["splits"][split] for split in ("calibration", "test")}
    if read(baseline / "test.json") != rows["test"]:
        raise ValueError("Native test data differs from the router dataset")
    calibration = [json.loads(line) for line in (baseline / "calibration.jsonl").read_text().splitlines()]
    if [row["state"] for row in calibration] != [row["request"] for row in rows["calibration"]] or [
        row["expected"]["difficulty"] for row in calibration
    ] != [LABELS[row["label"]] for row in rows["calibration"]]:
        raise ValueError("Native calibration order differs from the router dataset")
    artifacts = {}

    def bind(root_name, root, names):
        for name in names:
            artifacts[f"{root_name}/{name}"] = file_hash(root / name)

    bind("baseline", baseline, ("manifest.json", "data.json", "calibration.jsonl", "test.json"))
    bind("study", study, ("manifest.json", "all-frozen.json", "evaluation-plan.json", "evaluation-complete.json"))
    stages = {}
    for name, root_name, root, relative, variant in (
        ("base", "baseline", baseline, "native/seed-89", "base"),
        ("lora512", "study", study, "screening/original/seed-89", "lora"),
        ("lora2581", "study", study, "expanded/seed-89", "lora"),
    ):
        stages[name] = _native_stage(
            root / relative,
            variant,
            dataset,
            artifacts["baseline/calibration.jsonl"],
            artifacts["baseline/test.json"],
        )
        names = ("plan.json", "frozen.json", f"{variant}-calibration.json", "evaluation-test.json")
        bind(root_name, root, [f"{relative}/{name}" for name in names])
    frozen = read(policy_directory / "frozen.json")
    if frozen["dataset_sha256"] != digest(dataset):
        raise ValueError("Embedding policy suite uses a different dataset")
    bind("policies", policy_directory, ("frozen.json", "ridge64.json", "ridge512.json"))
    matrices = {}
    for split, directory in (("calibration", "development-embeddings"), ("test", "test-embeddings")):
        matrices[split] = vectors(rows[split], baseline / directory)
        bind("baseline", baseline, [f"{directory}/{name}" for name in ("plan.json", "vectors.jsonl.gz")])
    reference = read(baseline / "embedding-evaluation.json")
    if reference["dataset_sha256"] != digest(dataset) or reference["ids"] != [row["id"] for row in rows["test"]]:
        raise ValueError("Embedding reference uses different test inputs")
    bind("baseline", baseline, ("embedding-evaluation.json",))
    for size in (64, 512):
        name = f"ridge{size}"
        policy = read(policy_directory / f"{name}.json")
        if (
            policy["sha256"] != frozen["policies"][name]
            or policy["dataset_sha256"] != digest(dataset)
            or policy["development_vectors_sha256"] != artifacts["baseline/development-embeddings/vectors.jsonl.gz"]
            or policy["labels"] != size
            or policy["train_ids"] != [row["id"] for row in dataset["splits"]["train"][:size]]
        ):
            raise ValueError("Embedding policy differs from its frozen training inputs")
        scores = {
            split: finite_scores(policies.scores(policy, matrix), len(rows[split]))
            for split, matrix in matrices.items()
        }
        threshold = float(policy["threshold"])
        if (
            not np.isfinite(threshold)
            or (scores["test"] >= threshold).astype(int).tolist() != reference["predictions"][name]
        ):
            raise ValueError("Embedding decisions differ from the frozen reference")
        stages[name] = {
            "family": "embedding",
            "encoder": MODEL,
            "revision": REVISION,
            "original_threshold": threshold,
            "policy_sha256": policy["sha256"],
            "scores": {split: values.tolist() for split, values in scores.items()},
        }
    return seal(
        {
            "format": SCORE_FORMAT,
            "dataset_sha256": digest(dataset),
            "ids": {split: [row["id"] for row in values] for split, values in rows.items()},
            "input_sha256": {
                split: {row["id"]: digest(row["request"]) for row in values} for split, values in rows.items()
            },
            "stages": stages,
            "source_artifacts_sha256": artifacts,
            "source_code_sha256": {
                name: file_hash(Path(__file__).with_name(name)) for name in ("policies.py", "serving.py")
            },
            "interpretation": "Pre-existing difficulty-router scores on an exposed benchmark; no solver test answers used",
        }
    )


def _checked_scores(scores):
    check_seal(scores, SCORE_FORMAT)
    if set(scores["stages"]) != set(STAGES) or set(scores["ids"]) != {"calibration", "test"}:
        raise ValueError("Expected all five declared stages and both populations")
    seen = set()
    for split, ids in scores["ids"].items():
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Router populations must have unique, disjoint IDs")
        if set(scores["input_sha256"][split]) != set(ids):
            raise ValueError("Router input hashes do not match IDs")
        seen.update(ids)
        for name in STAGES:
            stage = scores["stages"][name]
            if stage["family"] != ("embedding" if name.startswith("ridge") else "native"):
                raise ValueError("Router family differs from the declared stage")
            finite_scores(stage["scores"][split], len(ids))
            if not np.isfinite(stage["original_threshold"]):
                raise ValueError("Nonfinite original threshold")


def _grading_provenance(document):
    hashes = {name: document.get(name) for name in GRADING_HASHES}
    if any(
        not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in hashes.values()
    ):
        raise ValueError("Grading provenance requires SHA256 hashes for the grader, gold-review source and gold review")
    return hashes


def _grade_rows(path, split, plan_sha, ids, *, quality_only=False):
    grades = read(path)
    _grading_provenance(grades)
    if grades.get("split") != split or grades.get("plan_sha256") != plan_sha:
        raise ValueError("Grades use a different split or solver plan")
    if [row["id"] for row in grades["rows"]] != ids:
        raise ValueError("Grades must contain every expected ID in the frozen order")
    # The pure reporter validates both boolean outcomes and complete measured costs.
    if quality_only:
        solver_report.summarize_quality(grades["rows"], "always_strong")
    else:
        solver_report.summarize(grades["rows"], "always_strong")
    return grades


def _credits(path, split, plan_sha, ids, input_hashes):
    if path is None:
        return None, None
    record = read(path)
    if (
        record.get("plan_sha256") != plan_sha
        or (record.get("model"), record.get("revision")) != (MODEL, REVISION)
        or record.get("batch_size") != 1
    ):
        raise ValueError("Router credits require the same plan and pinned encoder measured one request at a time")
    rows = record[split]
    if [row["id"] for row in rows] != ids:
        raise ValueError("Router credit IDs do not match the frozen order")
    values = []
    for row in rows:
        value = row["credits"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("Router credits must be measured finite nonnegative numbers")
        if "request_sha256" in row and row["request_sha256"] != input_hashes[row["id"]]:
            raise ValueError("Router credit receipt belongs to another request")
        values.append(value)
    return values, file_hash(path)


def _selections(rows, scores, embedding_credits, credit_usd, *, quality_only=False):
    if [row["id"] for row in rows] != scores["ids"]["calibration"]:
        raise ValueError("Calibration outcomes differ from the frozen score order")
    if quality_only:
        return {}
    return {
        name: {
            str(target): solver_report.select_cutoff(
                rows,
                scores["stages"][name]["scores"]["calibration"],
                target,
                router_credits=embedding_credits if name.startswith("ridge") else None,
                credit_usd=credit_usd,
            )
            for target in TARGETS
        }
        for name in STAGES
    }


def freeze_calibrations(
    plan_path,
    calibration_grades_path,
    scores,
    router_credits_path=None,
    credit_usd=1e-5,
    *,
    quality_only=False,
):
    """Return a sealed policy document without reading any solver test answer."""
    plan_path = Path(plan_path)
    if quality_only and router_credits_path is not None:
        raise ValueError("Quality-only mode does not accept priced router credits")
    if (plan_path.parent / "answers-test").exists() or (plan_path.parent / "grades-test.json").exists():
        raise ValueError("Refuse a new policy freeze after test-answer collection has started")
    _checked_scores(scores)
    plan, inputs = read(plan_path), read(plan_path.with_name("inputs.json"))
    if plan["dataset_sha256"] != scores["dataset_sha256"] or plan["inputs_sha256"] != digest(inputs):
        raise ValueError("Solver plan inputs differ from the frozen router population")
    for split, ids in scores["ids"].items():
        if (
            [row["id"] for row in inputs[split]] != ids
            or plan["counts"][split] != len(ids)
            or {row["id"]: digest(row["request"]) for row in inputs[split]} != scores["input_sha256"][split]
        ):
            raise ValueError("Solver and router input IDs, order or requests differ")
    plan_sha = file_hash(plan_path)
    grades = _grade_rows(
        calibration_grades_path,
        "calibration",
        plan_sha,
        scores["ids"]["calibration"],
        quality_only=quality_only,
    )
    credits, credit_sha = _credits(
        router_credits_path,
        "calibration",
        plan_sha,
        scores["ids"]["calibration"],
        scores["input_sha256"]["calibration"],
    )
    keys = ("id", "cheap_correct", "strong_correct")
    if not quality_only:
        keys += ("cheap_credits", "strong_credits")
    rows = [{key: row[key] for key in keys} for row in grades["rows"]]
    selections = _selections(rows, scores, credits, credit_usd, quality_only=quality_only)
    return seal(
        {
            "format": FROZEN_FORMAT,
            "analysis_mode": "quality_only" if quality_only else "cost",
            "plan_sha256": plan_sha,
            "calibration_grades_sha256": file_hash(calibration_grades_path),
            **_grading_provenance(grades),
            "source_scores": scores,
            "calibration_rows": rows,
            "embedding_calibration_credits": credits,
            "embedding_calibration_credits_sha256": credit_sha,
            "credit_usd": None if quality_only else credit_usd,
            "primary_max_quality_loss_pp": None if quality_only else 1,
            "secondary_max_quality_loss_pp": [] if quality_only else [0, 2, 5],
            "selections": selections,
            "selection_note": (
                "Quality-only: retain original difficulty cutoffs; unknown costs cannot define a cheapest policy"
                if quality_only
                else "Calibrate cost policies using measured solver credits"
            ),
            "source_code_sha256": {
                name: file_hash(Path(__file__).with_name(name)) for name in ("solver_policies.py", "solver_report.py")
            },
            "test_access": "Router scores only; solver test answers are not opened during policy selection",
            "interpretation": "Targets constrain observed calibration gaps; prospective test quality is not guaranteed",
        }
    )


def report(frozen_path, test_grades_path, router_credits_path=None):
    """Apply every predeclared policy; never choose a winner from test results."""
    frozen = read(frozen_path)
    check_seal(frozen, FROZEN_FORMAT)
    grading = _grading_provenance(frozen)
    if frozen.get("analysis_mode") not in ("cost", "quality_only"):
        raise ValueError("Unknown frozen analysis mode")
    quality_only = frozen["analysis_mode"] == "quality_only"
    if quality_only and router_credits_path is not None:
        raise ValueError("Quality-only mode does not accept priced router credits")
    scores = frozen["source_scores"]
    _checked_scores(scores)
    if (
        frozen["primary_max_quality_loss_pp"] != (None if quality_only else 1)
        or frozen["secondary_max_quality_loss_pp"] != ([] if quality_only else [0, 2, 5])
        or (quality_only and (frozen["credit_usd"] is not None or frozen["embedding_calibration_credits"] is not None))
        or frozen["source_code_sha256"]
        != {name: file_hash(Path(__file__).with_name(name)) for name in ("solver_policies.py", "solver_report.py")}
        or frozen["selections"]
        != _selections(
            frozen["calibration_rows"],
            scores,
            frozen["embedding_calibration_credits"],
            frozen["credit_usd"],
            quality_only=quality_only,
        )
    ):
        raise ValueError("Frozen calibration policy, declared targets or reporting code differs")
    ids = scores["ids"]["test"]
    grades = _grade_rows(test_grades_path, "test", frozen["plan_sha256"], ids, quality_only=quality_only)
    if grades.get("policies_frozen_sha256") != file_hash(frozen_path) or _grading_provenance(grades) != grading:
        raise ValueError(
            "Test grades are not bound to the pre-collection policy freeze, same grader and same gold review"
        )
    credits, credits_sha = _credits(
        router_credits_path,
        "test",
        frozen["plan_sha256"],
        ids,
        scores["input_sha256"]["test"],
    )
    rows, rate = grades["rows"], frozen["credit_usd"]

    def summarize(routes, overhead=None):
        if quality_only:
            return solver_report.summarize_quality(rows, routes)
        return solver_report.summarize(rows, routes, router_credits=overhead, credit_usd=rate)

    stages = {}
    for name in STAGES:
        stage = scores["stages"][name]
        values = np.array(stage["scores"]["test"])
        overhead = credits if name.startswith("ridge") else None
        calibrated = {}
        for target in () if quality_only else TARGETS:
            selection = frozen["selections"][name][str(target)]
            mode = selection["mode"]
            routes = (values >= selection["cutoff"]).tolist() if mode == "threshold" else mode
            calibrated[str(target)] = {
                "mode": mode,
                "cutoff": selection["cutoff"],
                "metrics": summarize(routes, overhead if mode == "threshold" else 0),
            }
        stages[name] = {
            "original_threshold": stage["original_threshold"],
            "original_difficulty_policy": summarize((values >= stage["original_threshold"]).tolist(), overhead),
            "calibrated_cost_policies": calibrated,
        }
    return {
        "format": "sie-solver-policy-report-v1",
        "analysis_mode": frozen["analysis_mode"],
        "plan_sha256": frozen["plan_sha256"],
        "policies_frozen_sha256": file_hash(frozen_path),
        "test_grades_sha256": file_hash(test_grades_path),
        "source_scores_sha256": scores["sha256"],
        "embedding_test_credits_sha256": credits_sha,
        "ids": ids,
        "n": len(ids),
        "credit_usd": rate,
        "primary_max_quality_loss_pp": frozen["primary_max_quality_loss_pp"],
        "secondary_max_quality_loss_pp": frozen["secondary_max_quality_loss_pp"],
        "selection_note": frozen["selection_note"],
        "baselines": {name: summarize(name, 0) for name in ("always_strong", "always_cheap")},
        "stages": stages,
        "interpretation": (
            "Quality-only replay of original frozen difficulty cutoffs. Solver and infrastructure prices are unknown; "
            "no costs, savings, break-even prices or cost-calibrated policies are reported. "
            "All policies share the same paired solver outcomes on an exposed benchmark."
        )
        if quality_only
        else (
            "Every frozen policy shares the same paired solver outcomes. Positive gap means answer-accuracy loss; "
            "negative gap preserves cheap-solver rescues. Unknown native overhead is excluded from gross savings. "
            "Calibration targets are not test guarantees; this population is an exposed benchmark."
        ),
    }


def _display_table(headers, rows):
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    for index, row in enumerate([headers, *rows]):
        print(
            " | ".join(
                str(cell).ljust(width) if i == 0 else str(cell).rjust(width)
                for i, (cell, width) in enumerate(zip(row, widths, strict=True))
            )
        )
        if index == 0:
            print("-+-".join("-" * width for width in widths))


def display(result):
    """Print quality and explicitly separated gross/net costs without changing the report."""
    names = {
        "base": "Base Laya",
        "lora512": "LoRA 512",
        "lora2581": "LoRA 2,581",
        "ridge64": "Qwen ridge 64",
        "ridge512": "Qwen ridge 512",
    }
    groups = [
        (
            "Original frozen difficulty cutoffs",
            [
                ("Always strong", result["baselines"]["always_strong"]),
                ("Always cheap", result["baselines"]["always_cheap"]),
            ]
            + [(names[name], result["stages"][name]["original_difficulty_policy"]) for name in STAGES],
        )
    ]
    quality_only = result["analysis_mode"] == "quality_only"
    if not quality_only:
        target = result["primary_max_quality_loss_pp"]
        groups.append(
            (
                f"Cost policies selected for a {target}pp calibration loss target (test results)",
                [
                    (names[name], result["stages"][name]["calibrated_cost_policies"][str(target)]["metrics"])
                    for name in STAGES
                ],
            )
        )
    print(f"Answer quality on {result['n']} paired test requests (exposed benchmark)")
    print("Gap = always-strong accuracy minus policy accuracy; negative gaps mean more correct answers.")
    print("Regressions / rescues count answers lost / gained versus always strong.")
    if quality_only:
        print("Costs unknown: quality-only comparison; no cost-based policy selection.")

    def number(metrics, key, *, missing="unpriced", digits=4):
        value = metrics.get(key)
        return missing if value is None else f"{value:.{digits}f}"

    for title, entries in groups:
        print(f"\n{title}")
        _display_table(
            ["Policy", "Answer %", "Gap pp", "Cheap %", "Regressions", "Rescues"],
            [
                [
                    name,
                    f"{100 * metrics['answer_accuracy']:.2f}",
                    f"{metrics['gap_vs_strong_pp']:+.2f}",
                    f"{100 * metrics['cheap_share']:.2f}",
                    metrics["regressions"],
                    metrics["rescues"],
                ]
                for name, metrics in entries
            ],
        )
        if not quality_only:
            print("\nCosts per 1,000 requests; savings versus always strong")
            _display_table(
                ["Policy", "Solver $", "Gross save %", "Router $", "Net total $", "Net save %"],
                [
                    [
                        name,
                        number(metrics, "total_solver_cost_usd_per1000"),
                        number(metrics, "gross_solver_savings_pct", missing="n/a", digits=2),
                        number(metrics, "router_cost_usd_per1000"),
                        number(metrics, "net_total_cost_usd_per1000"),
                        number(
                            metrics,
                            "net_savings_pct",
                            missing="n/a" if metrics["router_overhead_known"] else "unpriced",
                            digits=2,
                        ),
                    ]
                    for name, metrics in entries
                ],
            )
    if not quality_only:
        print("\nGross savings exclude router costs; net totals and savings include them.")
        print("unpriced = unknown router overhead; n/a = undefined percentage at a zero baseline cost.")
        print("Calibration loss targets are not test guarantees. The 0, 2 and 5pp targets remain in JSON.")
    print("Paired bootstrap intervals and full provenance are retained in JSON.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Select cutoffs from complete calibration answers only")
    for name in ("plan", "calibration-grades", "baseline", "study", "policies", "output"):
        freeze.add_argument(f"--{name}", type=Path, required=True)
    freeze.add_argument("--router-credits", type=Path)
    freeze.add_argument("--credit-usd", type=float, default=1e-5)
    freeze.add_argument(
        "--quality-only",
        action="store_true",
        help="Keep original cutoffs and report quality without costs when self-hosted prices are unknown",
    )
    evaluate = commands.add_parser("report", help="Report all frozen policies on complete test answers")
    for name in ("frozen", "test-grades", "output"):
        evaluate.add_argument(f"--{name}", type=Path, required=True)
    evaluate.add_argument("--router-credits", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "freeze":
        scores = load_stage_scores(args.baseline, args.study, args.policies)
        result = freeze_calibrations(
            args.plan,
            args.calibration_grades,
            scores,
            args.router_credits,
            args.credit_usd,
            quality_only=args.quality_only,
        )
    else:
        result = report(args.frozen, args.test_grades, args.router_credits)
    write(args.output, result)
    if args.command == "report":
        display(result)
    print(json.dumps({"output": str(args.output), "sha256": file_hash(args.output)}))


if __name__ == "__main__":
    main()
