"""One fresh ordinary server plus one exec SDK child, with exact owned cleanup."""

from __future__ import annotations

# Fixed reviewed protocol counts/statuses are intentionally explicit.
# ruff: noqa: PLR2004
import json
import os
import select
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from tools.benchmarks.merged_runtime_lora.common import (
    Observer,
    Server,
    atomic,
    close_enough,
    digest,
    normalized,
    require,
)


def read(process: Any, pending: Any, deadline: Any) -> Any:
    while b"\n" not in pending:
        wait = deadline - time.monotonic()
        require(wait > 0, "SDK child protocol timeout")
        require(select.select([process.stdout], [], [], wait)[0], "SDK child protocol timeout")
        data = os.read(process.stdout.fileno(), 4096)
        require(data, "SDK child closed protocol")
        pending.extend(data)
        require(len(pending) <= 65536, "SDK child protocol oversized")
    split = pending.index(b"\n") + 1
    value = json.loads(pending[:split])
    del pending[:split]
    require(isinstance(value, dict), "SDK child protocol not an object")
    return value


def run(
    plan_path: Any, family: Any, source: Any, output: Any, *, limit_s: Any = 150, existing_server: Any = None
) -> Any:
    plan_path, source, output = Path(plan_path), Path(source), Path(output)
    plan = json.loads(plan_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + limit_s
    mode = "ordinary" if existing_server is None else "opt-in-resident-base"
    receipt = {
        "passed": False,
        "family": family,
        "source": str(source),
        "mode": mode,
        "sdk_child": {},
        "cleanup": {},
        "ordinary_gate_input": mode == "ordinary",
    }
    process = None
    history = None
    try:
        context = (
            Server(
                plan, family, output / "server", role="ordinary-timing", adapter_path=plan["models"][family]["ordinary"]
            )
            if existing_server is None
            else nullcontext(existing_server)
        )
        with context as server:
            if mode == "ordinary":
                require(not hasattr(server.adapter, "_lora"), "ordinary timing initialized LoRA manager")
            else:
                require(
                    server.adapter._lora._active is None and len(server.adapter._lora._loaded) == 2,
                    "resident-base timing configuration differs",
                )
                server.state.check(server.adapter, family)
            native = server.call(f"native-before-{mode}-timing")
            history = server.observer.records
            server.observer.close()
            server.observer = None
            token = uuid.uuid4().hex
            command = [
                sys.executable,
                "-m",
                "tools.benchmarks.merged_runtime_lora.timing_client",
                "--plan",
                str(plan_path),
                "--plan-sha256",
                digest(plan_path),
                "--source",
                str(source),
                "--family",
                family,
                "--output",
                str(output / "client"),
                "--url",
                server.url,
                "--token",
                token,
                "--mode",
                mode,
            ]
            with (output / "client.log").open("wb") as log:
                process = subprocess.Popen(  # noqa: S603 -- exact owned SDK child argv
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, close_fds=True
                )
                receipt["sdk_child"].update(
                    pid=process.pid, parent_pid=os.getpid(), process_group=os.getpgid(process.pid)
                )
                pending = bytearray()
                hello = read(process, pending, min(deadline, time.monotonic() + 30))
                require(hello.get("event") == "hello" and hello.get("token") == token, "SDK child hello differs")
                ident = hello["identity"]
                require(
                    ident["pid"] == process.pid
                    and ident["ppid"] == os.getpid()
                    and ident["process_group"] == os.getpgrp(),
                    "SDK child ownership differs",
                )
                require(
                    ident["source"] == str(source.resolve()) and ident["forbidden_imports_present"] is False,
                    "SDK child source/imports differ",
                )
                receipt["sdk_child"]["identity"] = ident
                atomic(
                    output / "sdk-ownership.json", {"token": token, "identity": ident, "family": family, "mode": mode}
                )
                assert process.stdin is not None
                process.stdin.write(
                    (json.dumps({"token": token, "command": "phase", "concurrency": 1}) + "\n").encode()
                )
                process.stdin.flush()
                done = read(process, pending, deadline)
                require(done.get("event") == "finished" and done.get("token") == token, "SDK child terminal differs")
                require(process.wait(timeout=max(0.1, deadline - time.monotonic())) == 0, "SDK child exit failed")
                require(digest(output / "client/terminal.json") == done["terminal_sha256"], "SDK terminal hash differs")
                terminal = json.loads((output / "client/terminal.json").read_text())
                require(
                    terminal["passed"] and terminal["identity"] == ident and terminal["mode"] == mode,
                    "SDK terminal identity/verdict differs",
                )
                require(
                    digest(output / "client/performance.json") == terminal["result_sha256"], "SDK result hash differs"
                )
                result = json.loads((output / "client/performance.json").read_text())
                settings = plan["latency"] if mode == "ordinary" else plan["opt_in_latency"]
                require(
                    result["count"] == settings["measured_calls"] and result["identity"] == ident,
                    "SDK sample count/identity differs",
                )
                events = [json.loads(line) for line in (output / "client/events.jsonl").read_text().splitlines()]
                attempts = [event for event in events if event["event"] == "performance_attempt"]
                require(
                    len(attempts) == 1
                    and attempts[0]["primary_error"] is None
                    and not attempts[0]["finalization_errors"],
                    "wire attempt evidence incomplete",
                )
                require(
                    len(attempts[0]["warmups"]) == settings["warmup_calls"]
                    and attempts[0]["completed_samples"] == result["samples"],
                    "warmup/sample evidence differs",
                )
                require(
                    len(attempts[0]["wire_responses"]) == settings["warmup_calls"] + settings["measured_calls"]
                    and all(row["status"] == 200 for row in attempts[0]["wire_responses"]),
                    "wire attempt count/status differs",
                )
                for row in [*attempts[0]["warmups"], *result["samples"]]:
                    ids = [f"{row['label']}:{i}" for i in range(4)]
                    close_enough(
                        normalized(row["result"], ids),
                        normalized(native["result"], native["ids"]),
                        plan["numeric"]["atol"],
                    )
                receipt["metrics"] = {k: result[k] for k in ("count", "mean_ms", "p95_ms", "elapsed_s", "rps")}
                receipt["result_sha256"] = terminal["result_sha256"]
                receipt["passed"] = True
    finally:
        cleanup_deadline = time.monotonic() + 20
        primary = sys.exception()
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=min(5, max(0.01, cleanup_deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=max(0.01, cleanup_deadline - time.monotonic()))
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
                receipt["cleanup"] = {"reaped": process.poll() is not None, "returncode": process.returncode}
        except BaseException as secondary:
            receipt["cleanup"]["error"] = str(secondary)
            if primary is None:
                raise
            primary.add_note(f"SDK child cleanup also failed: {secondary!r}")
        finally:
            if existing_server is not None and history is not None:
                existing_server.observer = Observer(existing_server)
                existing_server.observer.records = history
            atomic(output / "summary.json", receipt)
    return receipt
