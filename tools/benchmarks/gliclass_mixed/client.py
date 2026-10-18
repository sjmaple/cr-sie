"""One SDK load generator process, separate from the SIE server interpreter."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import sys
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

import httpx
import sie_sdk
from sie_sdk import Item, SIEClient

from tools.benchmarks.gliclass_mixed.evidence import Journal, atomic_json, digest, error_chain, parent_guard, percentile
from tools.benchmarks.gliclass_mixed.workload import (
    MAX_CALLS_PER_LANE,
    MIXED_SECONDS,
    MODEL,
    items,
    mixed_kinds,
    templates,
)

EXPECTED_SCHEMAS = {
    name: {"labels": value["labels"], "options": value["options"]} for name, value in templates().items()
}


class Client:
    def __init__(self, url: str, timeout: float, journal: Journal, fixtures: dict, model: str = MODEL) -> None:
        self.wire: list[dict] = []
        self.journal, self.fixtures, self.timeout, self.model = journal, fixtures, timeout, model
        self.transport = httpx.Client(
            base_url=url,
            timeout=timeout,
            trust_env=False,
            event_hooks={"request": [self.request], "response": [self.response]},
        )
        self.sdk = SIEClient(url, timeout_s=timeout, http_client=self.transport, remote="forbid")

    def request(self, _: httpx.Request) -> None:
        self.wire.append({"start_ns": time.monotonic_ns()})

    def response(self, response: httpx.Response) -> None:
        self.wire[-1].update(
            {
                "headers_ns": time.monotonic_ns(),
                "status": response.status_code,
                "headers": {
                    name: value
                    for name, value in response.headers.items()
                    if name.startswith(
                        ("x-queue", "x-inference", "x-total", "x-tokenization", "x-postprocessing", "x-sie-error")
                    )
                    or name == "retry-after"
                },
            }
        )

    def call(self, kind: str, request_id: str) -> dict:
        fixture = self.fixtures[kind]
        values = cast("list[Item]", items(kind, request_id, self.fixtures))
        self.wire = []
        self.journal.write("request_start", request=request_id, kind=kind, deadline_s=self.timeout)
        record: dict[str, Any] = {
            "request": request_id,
            "kind": kind,
            "start_ns": time.monotonic_ns(),
            "deadline_s": self.timeout,
            "item_count": len(values),
        }
        try:
            record["result"] = self.sdk.extract(
                self.model,
                values,
                labels=fixture["labels"],
                options=fixture["options"],
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=self.timeout,
            )
        except Exception as error:  # noqa: BLE001 -- retain every failed SDK attempt; errors remain unsuccessful.
            record["error"] = error_chain(error)
        finally:
            record["end_ns"] = time.monotonic_ns()
            record["wire"] = self.wire
            record["latency_s"] = (record["end_ns"] - record["start_ns"]) / 1e9
            self.journal.write("request_end", **record)
            self.wire = []
        return record

    def close(self) -> None:
        try:
            self.sdk.close()
        finally:
            for name, hook in (("request", self.request), ("response", self.response)):
                self.transport.event_hooks[name][:] = [
                    item for item in self.transport.event_hooks[name] if item != hook
                ]
            self.transport.close()
            self.wire.clear()


def probability(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def valid(record: dict[str, Any]) -> bool:
    """Fail closed on the entire expected SDK projection, including partial 200s."""
    if "error" in record or len(record.get("wire", [])) != 1 or record["wire"][0].get("status") != HTTPStatus.OK:
        return False
    kind = record.get("kind")
    fixture = EXPECTED_SCHEMAS.get(kind) if isinstance(kind, str) else None
    count, results = record.get("item_count"), record.get("result")
    if (
        fixture is None
        or type(count) is not int
        or count <= 0
        or not isinstance(results, list)
        or len(results) != count
    ):
        return False
    groups = fixture["options"].get("label_groups", {})
    labels = (
        {f"{group}.{label}" for group, choices in groups.items() for label in choices}
        if groups
        else set(fixture["labels"])
    )
    for index, entry in enumerate(results):
        if not isinstance(entry, dict):
            return False
        row = cast("dict[str, Any]", entry)
        if row.get("error") or row.get("id") != f"{record['request']}/{index}":
            return False
        if row.get("model") != MODEL or row.get("request", {}).get("served_by") != "local":
            return False
        if any(row.get(name) != [] for name in ("entities", "relations", "objects")):
            return False
        classes = row.get("classifications")
        if not isinstance(classes, list) or len(classes) != len(labels):
            return False
        if any(not isinstance(value, dict) or not probability(value.get("score")) for value in classes):
            return False
        scores = {value.get("label"): value["score"] for value in classes}
        if set(scores) != labels:
            return False
        data = row.get("data", {})
        if not isinstance(data, dict) or set(data) != set(groups):
            return False
        for group, choices in groups.items():
            value = data[group]
            if not isinstance(value, dict) or value.get("type") != "choice" or value.get("choice") not in choices:
                return False
            if not probability(value.get("confidence")):
                return False
            probabilities = value.get("probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != set(choices):
                return False
            if any(
                not probability(score) or score != scores[f"{group}.{label}"] for label, score in probabilities.items()
            ):
                return False
    return True


def is_timeout(record: dict) -> bool:
    return any(error["type"] == "ReadTimeout" for error in record.get("error", []))


def summarize(records: list[dict]) -> dict:
    """Outcomes of a submission cohort, including completions after its window."""
    successful = [record for record in records if valid(record)]
    latencies = [record["latency_s"] for record in successful]
    return {
        "submitted": len(records),
        "successful": len(successful),
        "read_timeouts": sum(map(is_timeout, records)),
        "other_errors": sum("error" in record and not is_timeout(record) for record in records),
        "retry_calls": sum(len(record["wire"]) > 1 for record in records),
        "invalid_results": sum("error" not in record and not valid(record) for record in records),
        "latency_mean_s": sum(latencies) / len(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 0.5) if latencies else None,
        "latency_p95_s": percentile(latencies, 0.95) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
    }


def completion_interval(records: list[dict], start_ns: int, duration: float) -> dict:
    """End-timestamp accounting; includes calls submitted in an earlier interval."""
    end_ns = start_ns + round(duration * 1e9)
    ended = [row for row in records if start_ns <= row["end_ns"] < end_ns]
    successful = [row for row in ended if valid(row)]
    return {
        "duration_s": duration,
        "logical_calls_ended": len(ended),
        "successful": len(successful),
        "read_timeouts": sum(map(is_timeout, ended)),
        "other_errors": sum("error" in row and not is_timeout(row) for row in ended),
        "invalid_results": sum("error" not in row and not valid(row) for row in ended),
        "successful_requests_s": len(successful) / duration,
        "successful_items_s": sum(row["item_count"] for row in successful) / duration,
    }


def window(records: list[dict], start_ns: int, width: float) -> dict:
    end_ns = start_ns + round(width * 1e9)
    cohort = [row for row in records if start_ns <= row["start_ns"] < end_ns]
    return {
        "submission_cohort": summarize(cohort),
        "completion_interval": completion_interval(records, start_ns, width),
    }


def finalize(output: Path, name: str, summary: dict, clients: list[Client], journal: Journal) -> None:
    """Try every cleanup and persist failures without masking an active exception."""
    primary = sys.exception()
    failures = []
    for action in [*(owner.close for owner in clients), journal.close]:
        try:
            action()
        except BaseException as error:  # noqa: BLE001 -- preserve primary failure and attempt every cleanup.
            failures.extend(error_chain(error))
    if failures:
        summary["cleanup_errors"] = failures
        summary["completed"] = False
    try:
        summary["raw_sha256"] = digest(output / f"{name}.jsonl")
        atomic_json(output / f"{name}-summary.json", summary)
    except BaseException as error:
        if primary is None:
            raise
        primary.add_note(f"Evidence finalization also failed: {error!r}")
    if failures and primary is None:
        raise RuntimeError(f"client cleanup failed: {failures}")


def phase(
    url: str,
    output: Path,
    name: str,
    kinds: tuple[str, ...],
    duration: float,
    timeout: float,
    fixtures: dict,
    *,
    warmup: int = 0,
    model: str = MODEL,
) -> dict:
    journal = Journal(output / f"{name}.jsonl")
    clients: list[Client] = []
    start = threading.Event()
    stop = threading.Event()
    records: list[list[dict]] = [[] for _ in kinds]
    start_ns = 0
    summary: dict[str, Any] = {
        "phase": name,
        "completed": False,
        "duration_s": duration,
        "deadline_s": timeout,
        "lanes": list(kinds),
        "raw": f"{name}.jsonl",
    }
    try:
        for _ in kinds:
            clients.append(Client(url, timeout, journal, fixtures, model))
        for index in range(warmup):
            result = clients[0].call(kinds[0], f"{name}/warm/{index}")
            if not valid(result):
                raise AssertionError("warmup failed or retried")

        def lane(index: int) -> None:
            start.wait()
            sequence = 0
            while not stop.is_set() and time.monotonic_ns() < start_ns + round(duration * 1e9):
                if sequence >= MAX_CALLS_PER_LANE:
                    raise RuntimeError("fixed per-lane safety cap exceeded")
                record = clients[index].call(kinds[index], f"{name}/lane{index}/{sequence}")
                records[index].append(record)
                sequence += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(kinds)) as executor:
            futures = [executor.submit(lane, index) for index in range(len(kinds))]
            start_ns = time.monotonic_ns()
            journal.write("phase_start", phase=name, start_ns=start_ns, duration_s=duration)
            start.set()
            for future in futures:
                remaining = start_ns / 1e9 + duration + timeout + 10 - time.monotonic()
                future.result(timeout=max(0.001, remaining))
        flat = [record for rows in records for record in rows]
        summary.update(
            {
                "completed": True,
                "start_ns": start_ns,
                "end_ns": time.monotonic_ns(),
                "overall": summarize(flat),
                "arrival_interval": completion_interval(flat, start_ns, duration),
                "drain_interval": completion_interval(flat, start_ns + round(duration * 1e9), timeout),
                "ended_after_drain": sum(row["end_ns"] >= start_ns + round((duration + timeout) * 1e9) for row in flat),
                "classes": {kind: summarize([r for r in flat if r["kind"] == kind]) for kind in sorted(set(kinds))},
                "windows": [
                    {
                        "start_s": offset,
                        **window(flat, start_ns + round(offset * 1e9), width),
                        "classes": {
                            kind: window([r for r in flat if r["kind"] == kind], start_ns + round(offset * 1e9), width)
                            for kind in sorted(set(kinds))
                        },
                    }
                    for offset, width in (
                        [(0, 240), (240, 240)]
                        if name == "mixed" and duration == MIXED_SECONDS
                        else [(offset, min(60, duration - offset)) for offset in range(0, math.ceil(duration), 60)]
                    )
                ],
            }
        )
    except BaseException as error:
        summary["error"] = error_chain(error)
        raise
    finally:
        stop.set()
        start.set()
        finalize(output, name, summary, clients, journal)
    return summary


def correctness(url: str, output: Path, fixtures: dict, timeout: float, model: str) -> dict:
    journal = Journal(output / "correctness.jsonl")
    client = Client(url, timeout, journal, fixtures, model)
    records = []
    summary: dict[str, Any] = {"phase": "correctness", "completed": False}
    try:
        for repeat in range(2):
            for kind in fixtures:
                record = client.call(kind, f"correctness/{repeat}/{kind}")
                records.append(record)
                if not valid(record):
                    raise AssertionError(f"invalid/retried correctness request: {record['request']}")
        summary.update({"completed": True, "requests": len(records)})
        return summary
    except BaseException as error:
        summary["error"] = error_chain(error)
        raise
    finally:
        try:
            atomic_json(output / "correctness.json", {"records": records, "passed": summary["completed"]})
        finally:
            finalize(output, "correctness", summary, [client], journal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("correctness", "C1", "C8", "C32", "mixed"), required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not Path(sie_sdk.__file__).resolve().is_relative_to(args.source.resolve()):
        raise AssertionError("SDK import escaped the immutable source")
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = templates()
    if args.smoke:
        for fixture in fixtures.values():
            fixture["texts"] = ["The subscription was charged twice."] * min(2, len(fixture["texts"]))
    atomic_json(
        args.output / f"{args.phase}-identity.json",
        {
            "pid": os.getpid(),
            "sdk": str(Path(sie_sdk.__file__).resolve()),
            "sdk_sha256": digest(Path(sie_sdk.__file__)),
            "fixtures": fixtures,
            "smoke": args.smoke,
        },
    )
    if args.phase == "correctness":
        correctness(args.url, args.output, fixtures, 30, args.model)
    else:
        mixed = args.phase == "mixed"
        kinds = mixed_kinds() if mixed else ("steady",) * int(args.phase[1:])
        duration = 0.25 if args.smoke else MIXED_SECONDS if mixed else 240 if args.phase == "C8" else 60
        phase(
            args.url,
            args.output,
            args.phase,
            kinds,
            duration,
            3 if args.smoke else 300 if mixed else 30,
            fixtures,
            warmup=1 if args.smoke else 8,
            model=args.model,
        )


if __name__ == "__main__":
    with parent_guard():
        main()
