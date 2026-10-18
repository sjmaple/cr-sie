"""Controls for immutable source, real SDK behavior, raw persistence and cleanup."""

# Local child commands and fixed Git reads only.
# ruff: noqa: S603, S106

from __future__ import annotations

import asyncio
import copy
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import msgpack
import pytest
import torch
import yaml
from gliclass import GLiClassModel, GLiClassModelConfig
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import DebertaV2Config, PreTrainedTokenizerFast

from tools.benchmarks.gliclass_mixed import client
from tools.benchmarks.gliclass_mixed.analyze import ATOL, audit, compare, request_accounting
from tools.benchmarks.gliclass_mixed.evidence import Journal, digest, percentile
from tools.benchmarks.gliclass_mixed.prepare import (
    BASELINE,
    PROFILE,
    PROFILE_REVISION,
    SOURCE_PATHS,
    prepare,
    prepare_progress,
)
from tools.benchmarks.gliclass_mixed.run import archive_sources, source_environment, stop, switch, wait_json
from tools.benchmarks.gliclass_mixed.server import GLiClassAdapter, ModelWorker, Observers
from tools.benchmarks.gliclass_mixed.workload import GROUPS, LABELS, MODEL, items, mixed_kinds, templates

ROOT = Path(__file__).resolve().parents[4]


def tiny_checkpoint(path: Path) -> Path:
    path.mkdir()
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    tokens += sorted(
        {word for row in templates().values() for text in row["texts"] for word in text.split()}
        | {word for labels in GROUPS.values() for label in labels for word in label.split()}
    )
    vocabulary = {token: index for index, token in enumerate(dict.fromkeys(tokens))}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=1024,
    )
    tokenizer.add_tokens(["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"], special_tokens=True)
    tokenizer.save_pretrained(path)
    encoder = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=1024,
        relative_attention=True,
        position_buckets=16,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    config = GLiClassModelConfig(
        encoder_config=encoder,
        encoder_model="offline-tiny",
        vocab_size=len(tokenizer),
        class_token_index=tokenizer.convert_tokens_to_ids("<<LABEL>>"),
        text_token_index=tokenizer.convert_tokens_to_ids("<<SEP>>"),
        example_token_index=tokenizer.convert_tokens_to_ids("<<EXAMPLE>>"),
        scorer_type="simple",
        prompt_first=False,
        dropout=0.0,
    )
    torch.manual_seed(29)
    GLiClassModel(config).eval().save_pretrained(path)
    return path


def test_workload_classes_and_ids() -> None:
    fixture = templates()
    assert mixed_kinds() == ("short",) * 12 + ("medium7", "medium8", "long7", "long8")
    for band in ("medium", "long"):
        for groups in (7, 8):
            kind = f"{band}{groups}"
            assert len(fixture[kind]["texts"]) == 24
            assert len(fixture[kind]["options"]["label_groups"]) == groups
            assert fixture[kind]["options"]["group_encoding"] == "separate"
            rows = items(kind, "phase/lane0/request1", fixture)
            assert len({row["id"] for row in rows}) == 24
            assert [row["text"] for row in rows] == fixture[kind]["texts"]


def test_nearest_rank_and_invalid_raw_latencies() -> None:
    assert percentile([float(value) for value in range(1, 21)], 0.95) == 19
    assert percentile([1], 0.95) == 1
    for values in ([], [0], [1, -1, 20], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="latency samples"):
            percentile(values, 0.95)


def test_derived_source_only_profile(tmp_path: Path) -> None:
    manifest = prepare(ROOT, tmp_path / "inputs")
    assert manifest["baseline_commit"] == BASELINE
    assert manifest["candidate_profile_commit"] == PROFILE_REVISION
    output = tmp_path / "output"
    output.mkdir()
    sources = archive_sources(tmp_path / "inputs", output)
    assert (sources["baseline"] / "packages/sie_server/bundles/default.yaml").is_file()
    assert (sources["baseline"] / PROFILE).read_bytes() != (sources["candidate"] / PROFILE).read_bytes()
    assert manifest["candidate_is_derived_archive"]
    (tmp_path / "inputs/baseline.tar").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        archive_sources(tmp_path / "inputs", tmp_path / "other")


def test_prepare_rejects_additional_profile_change(tmp_path: Path) -> None:
    # A baseline file cannot be silently relabeled as the opt-out candidate.
    with pytest.raises(ValueError, match="candidate differs"):
        prepare(ROOT, tmp_path / "inputs", profile_revision=BASELINE)


def sdk_transport(monkeypatch: pytest.MonkeyPatch, handler):
    original = httpx.Client
    monkeypatch.setattr(
        client.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )


def reply(request: httpx.Request) -> httpx.Response:
    body = msgpack.unpackb(request.content, raw=False)
    data = {
        "items": [
            {"id": row["id"], "entities": [], "classifications": [{"label": label, "score": 0.2} for label in LABELS]}
            for row in body["items"]
        ],
        "model": "knowledgator/gliclass-large-v1.0",
        "request": {"served_by": "local"},
    }
    return httpx.Response(
        200, content=msgpack.packb(data), headers={"Content-Type": "application/msgpack", "X-SIE-Served-By": "local"}
    )


def test_actual_sdk_success_and_headers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sdk_transport(monkeypatch, reply)
    journal = Journal(tmp_path / "events.jsonl")
    owner = client.Client("http://fixture", 1, journal, templates())
    try:
        record = owner.call("short", "phase/lane0/request0")
        assert client.valid(record), record
        assert len(record["wire"]) == 1
        assert record["latency_s"] > 0
    finally:
        owner.close()
        journal.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events] == ["request_start", "request_end"]
    assert owner.request not in owner.transport.event_hooks["request"]
    assert owner.response not in owner.transport.event_hooks["response"]
    assert owner.transport.event_hooks["request"]  # SDK origin confinement remains installed.
    assert owner.transport.is_closed


def test_actual_sdk_timeout_persists_cause(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def timeout(request):
        raise httpx.ReadTimeout("retained diagnostic timeout", request=request)

    sdk_transport(monkeypatch, timeout)
    journal = Journal(tmp_path / "events.jsonl")
    owner = client.Client("http://fixture", 0.1, journal, templates())
    try:
        record = owner.call("short", "timeout/0")
        assert not client.valid(record)
        assert any(row["type"] == "ReadTimeout" for row in record["error"])
        assert len(record["wire"]) == 1
    finally:
        owner.close()
        journal.close()
    assert json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])["error"] == record["error"]


def test_actual_sdk_hidden_retry_is_not_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    attempts = []

    def response(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(
                503, json={"error": {"code": "QUEUE_FULL", "message": "busy"}}, headers={"Retry-After": "0.001"}
            )
        return reply(request)

    sdk_transport(monkeypatch, response)
    journal = Journal(tmp_path / "events.jsonl")
    owner = client.Client("http://fixture", 1, journal, templates())
    try:
        record = owner.call("short", "retry/0")
        assert [row.get("status") for row in record["wire"]] == [503, 200], record
        assert not client.valid(record)
    finally:
        owner.close()
        journal.close()


def test_phase_failure_retains_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def response(request):
        return httpx.Response(400, json={"error": {"code": "INVALID_INPUT", "message": "invalid"}})

    sdk_transport(monkeypatch, response)
    with pytest.raises(AssertionError, match="warmup failed"):
        client.phase("http://fixture", tmp_path, "failure", ("short",), 0.01, 0.1, templates(), warmup=1)
    summary = json.loads((tmp_path / "failure-summary.json").read_text())
    assert not summary["completed"]
    assert summary["error"][0]["type"] == "AssertionError"
    assert summary["raw_sha256"] == digest(tmp_path / "failure.jsonl")
    assert any(
        row["event"] == "request_end" for row in map(json.loads, (tmp_path / "failure.jsonl").read_text().splitlines())
    )


def valid_record(kind="short", request="window/lane0/0", count=1):
    fixture = templates()[kind]
    groups = fixture["options"].get("label_groups", {})
    classes = (
        [
            {"label": f"{group}.{label}", "score": 1 / len(labels)}
            for group, labels in groups.items()
            for label in labels
        ]
        if groups
        else [{"label": label, "score": 1 / len(fixture["labels"])} for label in fixture["labels"]]
    )
    data = {
        group: {
            "type": "choice",
            "choice": labels[0],
            "confidence": 0,
            "probabilities": {label: 1 / len(labels) for label in labels},
        }
        for group, labels in groups.items()
    }
    return {
        "kind": kind,
        "request": request,
        "item_count": count,
        "wire": [{"status": 200}],
        "start_ns": 0,
        "end_ns": 1_000_000_000,
        "latency_s": 1.0,
        "result": [
            {
                "id": f"{request}/{index}",
                "model": MODEL,
                "request": {"served_by": "local"},
                "entities": [],
                "relations": [],
                "objects": [],
                "classifications": copy.deepcopy(classes),
                **({"data": copy.deepcopy(data)} if groups else {}),
            }
            for index in range(count)
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "partial",
        "order",
        "duplicate",
        "class",
        "nan",
        "infinite",
        "range",
        "item_error",
        "remote",
        "model",
        "groups",
        "choice",
        "confidence",
        "probability",
    ],
)
def test_invalid_partial_or_malformed_success_is_rejected(mutation) -> None:
    record = valid_record("medium7", count=2)
    assert client.valid(record)
    if mutation == "empty":
        record["result"] = []
    elif mutation == "partial":
        record["result"].pop()
    elif mutation == "order":
        record["result"].reverse()
    elif mutation == "duplicate":
        record["result"][1]["id"] = record["result"][0]["id"]
    else:
        row = record["result"][0]
        if mutation == "class":
            row["classifications"].pop()
        elif mutation in ("nan", "infinite", "range"):
            row["classifications"][0]["score"] = {"nan": float("nan"), "infinite": float("inf"), "range": 1.01}[
                mutation
            ]
        elif mutation == "item_error":
            row["error"] = {"code": "INPUT_TOO_LONG"}
        elif mutation == "remote":
            row["request"]["served_by"] = "remote"
        elif mutation == "model":
            row["model"] = "wrong"
        elif mutation == "groups":
            row["data"].pop("topic")
        elif mutation == "choice":
            row["data"]["topic"]["choice"] = "undeclared"
        elif mutation == "confidence":
            row["data"]["topic"]["confidence"] = -0.1
        elif mutation == "probability":
            row["data"]["topic"]["probabilities"]["billing"] = 0.1
    assert not client.valid(record)


def test_submission_cohort_is_distinct_from_completion_interval() -> None:
    early = valid_record(request="w/early")
    early.update(start_ns=230_000_000_000, end_ns=250_000_000_000, latency_s=20)
    late_timeout = valid_record(request="w/timeout")
    late_timeout.update(
        start_ns=235_000_000_000,
        end_ns=535_000_000_000,
        latency_s=300,
        error=[{"type": "ReadTimeout", "message": "fixed deadline"}],
    )
    rows = [early, late_timeout]
    first = client.window(rows, 0, 240)
    second = client.window(rows, 240_000_000_000, 240)
    drain = client.completion_interval(rows, 480_000_000_000, 300)
    assert first["submission_cohort"]["submitted"] == 2
    assert first["submission_cohort"]["read_timeouts"] == 1
    assert first["completion_interval"]["logical_calls_ended"] == 0
    assert second["submission_cohort"]["submitted"] == 0
    assert second["completion_interval"]["successful"] == 1
    assert second["completion_interval"]["successful_requests_s"] == 1 / 240
    assert drain["read_timeouts"] == 1


def test_local_client_ignores_proxy_environment(monkeypatch, tmp_path) -> None:
    observed = {}
    original = httpx.Client

    def factory(**kwargs):
        observed.update(kwargs)
        return original(transport=httpx.MockTransport(reply), **kwargs)

    monkeypatch.setattr(client.httpx, "Client", factory)
    journal = Journal(tmp_path / "local.jsonl")
    owner = client.Client("http://127.0.0.1:9999", 1, journal, templates())
    try:
        assert observed["trust_env"] is False
        captured = []
        owner.transport._transport = httpx.MockTransport(lambda request: (captured.append(request), reply(request))[1])
        assert client.valid(owner.call("short", "local/0"))
        assert captured[0].headers["x-sie-remote"] == "forbid"
    finally:
        owner.close()
        journal.close()


def test_primary_failure_survives_cleanup_failure(monkeypatch, tmp_path) -> None:
    def response(request):
        return httpx.Response(400, json={"error": {"code": "INVALID_INPUT", "message": "primary"}})

    sdk_transport(monkeypatch, response)
    original = client.Client.close

    def broken_close(owner):
        original(owner)
        raise OSError("secondary close failure")

    monkeypatch.setattr(client.Client, "close", broken_close)
    with pytest.raises(AssertionError, match="warmup failed"):
        client.phase("http://fixture", tmp_path, "cleanup", ("short",), 0.01, 0.1, templates(), warmup=1)
    summary = json.loads((tmp_path / "cleanup-summary.json").read_text())
    assert summary["error"][0]["type"] == "AssertionError"
    assert summary["cleanup_errors"][0]["type"] == "OSError"
    assert summary["raw_sha256"] == digest(tmp_path / "cleanup.jsonl")


def test_fixed_numerical_equivalence_and_selected_labels() -> None:
    expected = valid_record("medium7", count=2)
    actual = copy.deepcopy(expected)
    actual["result"][0]["classifications"][0]["score"] += ATOL / 2
    actual["result"][0]["data"]["topic"]["probabilities"]["billing"] += ATOL / 2
    assert compare(actual, expected)["passed"]
    actual["result"][0]["classifications"][0]["score"] += ATOL
    actual["result"][0]["data"]["topic"]["probabilities"]["billing"] += ATOL
    assert not compare(actual, expected)["passed"]
    actual = copy.deepcopy(expected)
    actual["result"][0]["data"]["topic"]["choice"] = "other"
    assert not compare(actual, expected)["passed"]


def test_timeout_requires_admission_and_continuing_cheap_completions() -> None:
    heavy = valid_record("long8", "mixed/lane15/0")
    heavy.update(end_ns=300_000_000_000, latency_s=300, error=[{"type": "ReadTimeout", "message": "timeout"}])
    cheap = valid_record("short", "mixed/lane1/10")
    cheap.update(start_ns=289_000_000_000, end_ns=291_000_000_000, latency_s=2)
    events = [
        {"event": "admitted", "request": heavy["request"], "monotonic_ns": 1},
        {"event": "completed", "request": cheap["request"], "cancelled": False, "monotonic_ns": 290_000_000_000},
    ]
    result = request_accounting([heavy, cheap], events, 300)
    assert result["reproduction_witnesses"] == [heavy["request"]]
    assert not request_accounting([heavy, cheap], events[1:], 300)["reproduction_witnesses"]
    assert not request_accounting([heavy, cheap], events[:1], 300)["reproduction_witnesses"]
    events.append(
        {"event": "completed", "request": heavy["request"], "cancelled": False, "monotonic_ns": 330_000_000_000}
    )
    assert request_accounting([heavy, cheap], events, 300)["late_completions"] == 1
    events[-1]["monotonic_ns"] = 280_000_000_000
    assert not request_accounting([heavy, cheap], events, 300)["reproduction_witnesses"]


def test_observers_restore_originals_and_refuse_pending_requests(tmp_path) -> None:
    original = (
        ModelWorker.submit_extract,
        ModelWorker._process_batch,
        GLiClassAdapter.extract,
        GLiClassAdapter._forward,
    )
    journal = Journal(tmp_path / "observer.jsonl")
    observer = Observers(journal)
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    try:
        observer.set_enabled(True)
        assert ModelWorker.submit_extract is not original[0]
        observer.active["pending"] = future
        with pytest.raises(RuntimeError, match="unresolved"):
            observer.set_enabled(False)
        future.cancel()
        observer.set_enabled(False)
        assert (
            ModelWorker.submit_extract,
            ModelWorker._process_batch,
            GLiClassAdapter.extract,
            GLiClassAdapter._forward,
        ) == original
    finally:
        future.cancel()
        observer.set_enabled(False)
        loop.close()
        journal.close()


def test_actual_timeout_keeps_late_server_completion(tmp_path) -> None:
    checkpoint = tiny_checkpoint(tmp_path / "model")
    models = tmp_path / "models"
    models.mkdir()
    config = yaml.safe_load((ROOT / PROFILE).read_text())
    config.pop("hf_id", None)
    config.pop("hf_revision", None)
    config["weights_path"] = str(checkpoint)
    (models / "model.yaml").write_text(yaml.safe_dump(config))
    output = tmp_path / "server"
    output.mkdir()
    reservation = socket.socket()
    reservation.bind(("127.0.0.1", 0))
    reservation.listen()
    port = reservation.getsockname()[1]
    program = """import time
from tools.benchmarks.gliclass_mixed import server
original = server.GLiClassAdapter._forward
def delayed(self, *args, **kwargs):
    time.sleep(0.25)
    return original(self, *args, **kwargs)
server.GLiClassAdapter._forward = delayed
server.main()
"""
    process = None
    owner = None
    journal = Journal(tmp_path / "client.jsonl")
    try:
        with (tmp_path / "server.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    program,
                    "--source",
                    str(ROOT),
                    "--models-dir",
                    str(models),
                    "--output",
                    str(output),
                    "--port",
                    str(port),
                    "--device",
                    "cpu",
                    "--fd",
                    str(reservation.fileno()),
                ],
                env=source_environment(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(reservation.fileno(),),
            )
        reservation.close()
        wait_json(output / "ready.json", process, 30)
        switch(process, output, True)
        owner = client.Client(f"http://127.0.0.1:{port}", 0.1, journal, templates())
        record = owner.call("short", "mixed/lane0/0")
        assert client.is_timeout(record)
        deadline = time.monotonic() + 10
        completed = []
        while time.monotonic() < deadline:
            completed = [
                row
                for row in map(json.loads, (output / "server-events.jsonl").read_text().splitlines())
                if row["event"] == "completed"
            ]
            if completed:
                break
            time.sleep(0.01)
        assert len(completed) == 1
        assert not completed[0]["cancelled"]
        assert completed[0]["monotonic_ns"] > record["end_ns"]
    finally:
        reservation.close()
        if owner:
            owner.close()
        journal.close()
        terminal = stop(process)
    assert terminal["reaped"]
    assert not terminal["forced"]
    assert (output / "server-terminal.json").is_file()


@pytest.mark.parametrize("comparison", ["profile", "progress"])
def test_real_tiny_gliclass_server_sdk_and_cleanup(tmp_path: Path, comparison: str) -> None:
    checkpoint = tiny_checkpoint(tmp_path / "model")
    freeze = prepare_progress if comparison == "progress" else prepare
    freeze(ROOT, tmp_path / "inputs")
    output = tmp_path / "experiment"
    command = [
        sys.executable,
        "-m",
        "tools.benchmarks.gliclass_mixed.run",
        "--inputs",
        str(tmp_path / "inputs"),
        "--output",
        str(output),
        "--smoke",
        "--model-path",
        str(checkpoint),
        "--deadline-s",
        "600",
    ]
    with (tmp_path / "run.log").open("w") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=source_environment(ROOT),
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    assert process.returncode == 0, (tmp_path / "run.log").read_text() + str(output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed"]
    assert not summary["qualification"]
    analyzed = audit(output)
    assert analyzed["output_equivalence"]
    assert not analyzed["scoped_fix_observed"]
    for arm in ("baseline", "candidate"):
        record = json.loads((output / arm / "arm.json").read_text())
        assert (record["ready"]["adaptive"] is not None) == (comparison == "progress" or arm == "baseline")
        assert record["ready"]["source"]["batcher"]["sha256"] == digest(
            output / "sources" / arm / SOURCE_PATHS["batcher"]
        )
        assert record["completed"]
        assert record["server_cleanup"]["reaped"]
        assert not record["server_cleanup"]["forced"]
        assert (output / arm / "server-terminal.json").is_file()
        with pytest.raises(ProcessLookupError):
            os.kill(record["server_cleanup"]["pid"], 0)
        events = [json.loads(line) for line in (output / arm / "server-events.jsonl").read_text().splitlines()]
        assert any(row["event"] == "forward" for row in events)
        assert any(row["event"] == "admitted" for row in events)
        assert any(row["event"] == "completed" for row in events)
        assert [row["enabled"] for row in events if row["event"] == "observers"] == [True, False, True]
        assert not any(row.get("request", "").startswith(("C1/", "C8/", "C32/")) for row in events)
        assert events[-1]["event"] == "shutdown"
        assert not events[-1]["unresolved"]
        for phase in ("C1", "C8", "C32", "mixed"):
            measured = json.loads((output / arm / f"{phase}-summary.json").read_text())
            assert measured["completed"]
            assert measured["overall"]["submitted"] > 0
            assert (
                measured["overall"]["read_timeouts"]
                == measured["overall"]["other_errors"]
                == measured["overall"]["retry_calls"]
                == 0
            )
