"""Synthetic ledger controls for the fixed offline interpretation, without inference."""

from __future__ import annotations

import json
from pathlib import Path

from tools.benchmarks.gliclass_mixed.analyze import audit
from tools.benchmarks.gliclass_mixed.evidence import atomic_json
from tools.benchmarks.gliclass_mixed.tests.test_benchmark import valid_record
from tools.benchmarks.gliclass_mixed.workload import templates


def ledger(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(event) + "\n"
            for row in records
            for event in ({"event": "request_start", "request": row["request"]}, {"event": "request_end", **row})
        )
    )


def fixture(root: Path) -> None:
    atomic_json(root / "summary.json", {"completed": True, "smoke": False})
    for arm in ("baseline", "candidate"):
        folder = root / arm
        folder.mkdir()
        atomic_json(folder / "arm.json", {"completed": True})
        records = [
            valid_record(kind, f"correctness/{repeat}/{kind}", len(value["texts"]))
            for repeat in range(2)
            for kind, value in templates().items()
        ]
        ledger(folder / "correctness.jsonl", records)
        for phase in ("C1", "C8", "C32"):
            ledger(folder / f"{phase}.jsonl", [valid_record("steady", f"{phase}/lane0/0", 4)])
        heavy = valid_record("long8", "mixed/lane15/0", 24)
        cheap = valid_record("short", "mixed/lane0/0")
        cheap.update(start_ns=280_000_000_000, end_ns=291_000_000_000, latency_s=11)
        events = [
            {"event": "admitted", "request": heavy["request"], "monotonic_ns": 1},
            {"event": "completed", "request": cheap["request"], "monotonic_ns": 290_000_000_000, "cancelled": False},
        ]
        if arm == "baseline":
            heavy.pop("result")
            heavy.update(end_ns=300_000_000_000, latency_s=300, error=[{"type": "ReadTimeout", "message": "fixture"}])
        else:
            events.append(
                {"event": "completed", "request": heavy["request"], "monotonic_ns": 900_000_000, "cancelled": False}
            )
        ledger(folder / "mixed.jsonl", [heavy, cheap])
        events.append({"event": "shutdown", "server_exited": True})
        (folder / "server-events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))


def test_missing_candidate_server_ledger_never_means_zero_timeouts(tmp_path: Path) -> None:
    fixture(tmp_path)
    assert audit(tmp_path)["scoped_fix_observed"]
    (tmp_path / "candidate/server-events.jsonl").unlink()
    assert not audit(tmp_path)["scoped_fix_observed"]


def test_every_candidate_success_compares_directly_to_baseline(tmp_path: Path) -> None:
    fixture(tmp_path)
    for phase, delta in (("correctness", 0.0015), ("C1", 0.003)):
        path = tmp_path / "candidate" / f"{phase}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["event"] == "request_end" and row["kind"] == "steady":
                for item in row["result"]:
                    item["classifications"][0]["score"] += delta
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit(tmp_path)
    assert not result["output_equivalence"]
    assert not result["scoped_fix_observed"]
    failures = result["arms"]["candidate"]["failed_checks"]
    assert any(row.get("request") == "C1/lane0/0" and row.get("max_abs_delta", 0) > 0.002 for row in failures)


def test_empty_phase_and_wrong_fixture_count_fail_closed(tmp_path: Path) -> None:
    fixture(tmp_path)
    (tmp_path / "candidate/C8.jsonl").write_text("")
    ledger(tmp_path / "candidate/C1.jsonl", [valid_record("steady", "C1/lane0/0", 1)])
    result = audit(tmp_path)
    assert not result["scoped_fix_observed"]
    reasons = {row.get("reason") for row in result["arms"]["candidate"]["failed_checks"]}
    assert "wrong fixture item count" in reasons
    assert "missing expected requests/fixture repetitions" in reasons
