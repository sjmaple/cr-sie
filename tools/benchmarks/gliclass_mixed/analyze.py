"""Offline, fixed-criterion output and request-lifecycle audit; never drops failures."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.benchmarks.gliclass_mixed.client import is_timeout, valid
from tools.benchmarks.gliclass_mixed.evidence import atomic_json, digest
from tools.benchmarks.gliclass_mixed.workload import templates

ATOL = 0.002
PHASES = ("correctness", "C1", "C8", "C32", "mixed")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def canonical(row: dict) -> dict:
    result = copy.deepcopy(row)
    result.pop("id")
    result["classifications"].sort(key=lambda entry: entry["label"])
    return result


def difference(actual: Any, expected: Any, path: str = "") -> tuple[list[str], float]:
    """Exact structure/non-numerics; one prospective bound for scores/confidence."""
    if type(actual) in (int, float) and type(expected) in (int, float):
        delta = abs(actual - expected)
        return ([path] if delta > ATOL else []), delta
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        pairs = [difference(actual[key], expected[key], f"{path}/{key}") for key in actual]
    elif isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        pairs = [difference(a, b, f"{path}/{index}") for index, (a, b) in enumerate(zip(actual, expected, strict=True))]
    else:
        return ([] if actual == expected else [path]), 0.0
    return [item for failures, _ in pairs for item in failures], max((delta for _, delta in pairs), default=0.0)


def compare(record: dict, reference: dict) -> dict:
    if not valid(record) or not valid(reference) or record["kind"] != reference["kind"]:
        return {"passed": False, "reason": "invalid result or fixture", "max_abs_delta": None}
    failures, maximum = difference(
        [canonical(row) for row in record["result"]], [canonical(row) for row in reference["result"]]
    )
    return {"passed": not failures, "paths": failures, "max_abs_delta": maximum}


def request_accounting(client: list[dict], events: list[dict], deadline: float) -> dict:
    starts = {row["request"]: row for row in events if row["event"] == "admitted"}
    completions = {row["request"]: row for row in events if row["event"] == "completed"}
    batches: dict[str, list[int]] = {}
    for row in events:
        if row["event"] == "batch_start":
            for member in row["members"]:
                request = member["item_id"].rsplit("/", 1)[0]
                batches.setdefault(request, []).append(row["batch_id"])
    cheap_ids = {row["request"] for row in client if row["kind"] == "short"}
    ledger = []
    witnesses = []
    for row in client:
        request = row["request"]
        admitted, completed = starts.get(request), completions.get(request)
        cheap_during_tail = (
            [
                event["request"]
                for event in events
                if event["event"] == "completed"
                and event["request"].startswith("mixed/lane")
                and not event.get("cancelled")
                and not event.get("error")
                and row["end_ns"] - 30_000_000_000 <= event["monotonic_ns"] < row["end_ns"]
                and event["request"] in cheap_ids
            ]
            if is_timeout(row)
            else []
        )
        successful_before_timeout = (
            completed is not None
            and not completed.get("cancelled")
            and not completed.get("error")
            and completed["monotonic_ns"] <= row["end_ns"]
        )
        reproduced = (
            row["kind"] != "short"
            and is_timeout(row)
            and len(row["wire"]) == 1
            and row["latency_s"] >= deadline - 0.01
            and admitted is not None
            and admitted["monotonic_ns"] < row["end_ns"]
            and not successful_before_timeout
            and bool(cheap_during_tail)
        )
        entry = {
            "request": request,
            "kind": row["kind"],
            "client_timeout": is_timeout(row),
            "client_end_ns": row["end_ns"],
            "server_admitted_ns": admitted["monotonic_ns"] if admitted else None,
            "server_completion": completed,
            "batch_ids": sorted(set(batches.get(request, []))),
            "late_server_completion": completed is not None and completed["monotonic_ns"] > row["end_ns"],
            "server_unresolved_at_shutdown": admitted is not None and completed is None,
            "starvation_reproduction_witness": reproduced,
            "cheap_completions_in_last_30s": cheap_during_tail,
        }
        ledger.append(entry)
        if reproduced:
            witnesses.append(request)
    return {
        "requests": ledger,
        "reproduction_witnesses": witnesses,
        "per_class_witnesses": {
            kind: sum(row["kind"] == kind and row["starvation_reproduction_witness"] for row in ledger)
            for kind in sorted({row["kind"] for row in client})
        },
        "late_completions": sum(row["late_server_completion"] for row in ledger),
        "cancelled": sum(
            bool(row["server_completion"] and row["server_completion"].get("cancelled")) for row in ledger
        ),
        "unresolved_at_shutdown": sum(row["server_unresolved_at_shutdown"] for row in ledger),
    }


def audit(experiment: Path) -> dict:
    summary = json.loads((experiment / "summary.json").read_text())
    report: dict[str, Any] = {
        "schema": 1,
        "qualification": False,
        "descriptive": True,
        "atol": ATOL,
        "experiment_completed": summary["completed"],
        "arms": {},
        "raw_sha256": {},
    }
    baseline = {}
    fixture_counts = {
        kind: min(2, len(value["texts"])) if summary["smoke"] else len(value["texts"])
        for kind, value in templates().items()
    }
    for arm in ("baseline", "candidate"):
        directory = experiment / arm
        result: dict[str, Any] = {
            "complete": False,
            "output_equivalence": True,
            "checked": 0,
            "max_abs_delta": 0.0,
            "failed_checks": [],
            "request_failures": [],
        }
        report["arms"][arm] = result
        reference = {}
        for phase in PHASES:
            path = directory / f"{phase}.jsonl"
            if not path.exists():
                result["failed_checks"].append({"phase": phase, "reason": "missing phase ledger"})
                result["output_equivalence"] = False
                continue
            report["raw_sha256"][f"{arm}/{path.name}"] = digest(path)
            events = rows(path)
            requests = [row for row in events if row["event"] == "request_end"]
            starts = [row["request"] for row in events if row["event"] == "request_start"]
            if not requests or (
                phase == "correctness" and Counter(row["kind"] for row in requests) != dict.fromkeys(fixture_counts, 2)
            ):
                result["failed_checks"].append(
                    {"phase": phase, "reason": "missing expected requests/fixture repetitions"}
                )
                result["output_equivalence"] = False
            if (
                len(set(starts)) != len(starts)
                or set(starts) != {row["request"] for row in requests}
                or len(starts) != len(requests)
            ):
                result["failed_checks"].append({"phase": phase, "reason": "duplicate or unresolved client start"})
                result["output_equivalence"] = False
            for record in requests:
                if record.get("item_count") != fixture_counts.get(str(record.get("kind"))):
                    result["failed_checks"].append({"request": record["request"], "reason": "wrong fixture item count"})
                    result["output_equivalence"] = False
                if not valid(record):
                    result["request_failures"].append(
                        {
                            "request": record["request"],
                            "kind": record["kind"],
                            "error": record.get("error"),
                            "wire": record["wire"],
                        }
                    )
                    # Expected timeout outcomes stay explicit; malformed 200s fail equivalence.
                    if "result" in record or phase == "correctness":
                        result["output_equivalence"] = False
                    continue
                kind = record["kind"]
                if phase == "correctness":
                    reference.setdefault(kind, record)
                    if arm == "baseline":
                        baseline.setdefault(kind, record)
                expected = reference.get(kind)
                comparisons = (
                    [compare(record, expected)] if expected else [{"passed": False, "reason": "missing reference"}]
                )
                if arm == "candidate":
                    comparisons.append(
                        compare(record, baseline[kind])
                        if kind in baseline
                        else {"passed": False, "reason": "missing baseline"}
                    )
                for comparison in comparisons:
                    result["checked"] += 1
                    result["max_abs_delta"] = max(result["max_abs_delta"], comparison.get("max_abs_delta") or 0)
                    if not comparison["passed"]:
                        result["output_equivalence"] = False
                        result["failed_checks"].append({"request": record["request"], **comparison})
            if phase == "mixed":
                server = directory / "server-events.jsonl"
                if server.exists():
                    report["raw_sha256"][f"{arm}/{server.name}"] = digest(server)
                    server_events = rows(server)
                    result["server_ledger_complete"] = bool(
                        server_events
                        and server_events[-1].get("event") == "shutdown"
                        and server_events[-1].get("server_exited") is True
                    )
                    result["accounting"] = request_accounting(
                        [row for row in requests if "/warm/" not in row["request"]],
                        server_events,
                        3 if summary["smoke"] else 300,
                    )
        arm_summary = directory / "arm.json"
        result["complete"] = arm_summary.exists() and json.loads(arm_summary.read_text())["completed"]
    readiness = [report["arms"][arm] for arm in ("baseline", "candidate")]
    report["output_equivalence"] = all(row["complete"] and row["output_equivalence"] for row in readiness)
    report["baseline_reproduced"] = bool(report["arms"]["baseline"].get("accounting", {}).get("reproduction_witnesses"))
    report["candidate_timeouts"] = sum(
        row["client_timeout"] for row in report["arms"]["candidate"].get("accounting", {}).get("requests", [])
    )
    report["scoped_fix_observed"] = (
        not summary["smoke"]
        and summary["completed"]
        and report["output_equivalence"]
        and report["baseline_reproduced"]
        and all(arm.get("server_ledger_complete") and "accounting" in arm for arm in readiness)
        and report["candidate_timeouts"] == 0
        and not report["arms"]["candidate"]["request_failures"]
    )
    report["scope"] = (
        "One profile and fixed synthetic traffic; no general fairness or statistical performance qualification."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, audit(args.experiment))


if __name__ == "__main__":
    main()
