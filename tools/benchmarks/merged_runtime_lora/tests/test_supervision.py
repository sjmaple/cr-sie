"""Real owned-process refusal, exited-leader and server/SDK EOF controls."""

import json
import os
import subprocess
import sys
import time
from typing import Any

import pytest

from tools.benchmarks.merged_runtime_lora.runner import cleanup, group_exists
from tools.benchmarks.merged_runtime_lora.tests.test_proof import ROOT
from tools.benchmarks.merged_runtime_lora.tests.test_proof import plan as make_plan


@pytest.fixture
def tiny_plan(tmp_path: Any, tmp_path_factory: Any, monkeypatch: Any, request: Any) -> Any:
    return make_plan.__wrapped__(tmp_path, tmp_path_factory, monkeypatch, request)


def wait_for(path: Any, process: Any, limit: Any = 30) -> Any:
    deadline = time.monotonic() + limit
    while not path.exists():
        assert process.poll() is None, "owned child exited before control milestone"
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.mark.parametrize("leader_exits", [False, True])
def test_shared_cleanup_terminates_exact_group(tmp_path: Any, leader_exits: Any) -> Any:
    ready = tmp_path / "ready"
    if leader_exits:
        code = f"import os,signal,time,pathlib; p=os.fork();\nif p: os._exit(0)\nsignal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)"
    else:
        code = f"import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)"
    process = subprocess.Popen(  # noqa: S603 -- exact owned inert/test-server argv
        [sys.executable, "-c", code], start_new_session=True, close_fds=True
    )
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        if leader_exits:
            assert process.wait(timeout=1) == 0
        started = time.monotonic()
        result = cleanup(process, started + 2)
        assert result["reaped"]
        assert result["group_absent"]
        assert "KILL" in result["signals"]
        assert time.monotonic() - started < 2.25
    finally:
        if group_exists(process.pid):
            os.killpg(process.pid, 9)
        process.wait(timeout=2)


def test_real_server_and_sdk_exit_on_supervisor_pipe_eof(tiny_plan: Any, tmp_path: Any) -> Any:
    plan = tiny_plan
    plan["latency"] = {"warmup_calls": 2, "measured_calls": 10000}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / "timing"
    code = (
        "from pathlib import Path\n"
        "from tools.benchmarks.cross_encoder_lora.loadgen import Control\n"
        "control=Control('owned-eof-control')\n"
        "from tools.benchmarks.merged_runtime_lora.timing import run\n"
        f"run(Path({str(plan_path)!r}), 'bert', Path({str(ROOT)!r}), Path({str(output)!r}), limit_s=60)\n"
        "control.close()\n"
    )
    with (tmp_path / "child.log").open("wb") as log:
        process = subprocess.Popen(  # noqa: S603 -- exact owned inert/test-server argv
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
        try:
            ownership = output / "sdk-ownership.json"
            wait_for(ownership, process)
            sdk = json.loads(ownership.read_text())["identity"]
            assert sdk["ppid"] == process.pid
            assert sdk["process_group"] == process.pid
            assert process.stdin is not None
            process.stdin.close()
            assert process.wait(timeout=5) == -15
            deadline = time.monotonic() + 5
            while group_exists(process.pid) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not group_exists(process.pid), "SDK survived its serving parent"
            assert ownership.exists(), "completed ownership evidence lost"
        finally:
            cleanup(process, time.monotonic() + 2)
            assert process.stdin is not None
            if not process.stdin.closed:
                process.stdin.close()
