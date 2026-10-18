"""Run both immutable GLiClass comparison arms, preserving partial evidence.

This is a descriptive reproduction, not a statistical non-inferiority test.
"""

# All child processes use explicit local argv and task-owned paths.
# ruff: noqa: S603

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import yaml

from tools.benchmarks.gliclass_mixed.evidence import atomic_json, digest, error_chain, parent_guard, process_identity
from tools.benchmarks.gliclass_mixed.prepare import PROFILE, PROGRESS_PATHS, SOURCE_PATHS, executable_ast
from tools.benchmarks.gliclass_mixed.workload import MODEL, RUNNER_CEILING

HARNESS_ROOT = Path(__file__).resolve().parents[3]


def source_environment(source: Path) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("SIE_")}
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                str(source / path)
                for path in (
                    "packages/sie_server/src",
                    "packages/sie_sdk/src",
                    "packages/sie_config/src",
                )
            )
            + os.pathsep
            + str(HARNESS_ROOT),
            "SIE_BENCH_PARENT_PID": str(os.getpid()),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "SIE_TELEMETRY_DISABLED": "1",
            "SIE_OBSERVABILITY_ENABLED": "false",
            "SIE_WORKER_METRICS": "false",
        }
    )
    return env


def stop(process: subprocess.Popen | None) -> dict:
    if process is None:
        return {"started": False}
    forced = False
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            forced = True
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    return {"started": True, "pid": process.pid, "returncode": process.returncode, "forced": forced, "reaped": True}


def wait_json(path: Path, process: subprocess.Popen, seconds: float, predicate: Any = None) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"child exited {process.returncode} while waiting for {path.name}")
        if path.exists():
            record = json.loads(path.read_text())
            if "error" in record:
                raise RuntimeError(f"child control failed: {record['error']}")
            if predicate is None or predicate(record):
                return record
        time.sleep(0.02)
    raise TimeoutError(f"deadline waiting for {path.name}")


def switch(server: subprocess.Popen, output: Path, enabled: bool) -> None:
    path = output / "observers.json"
    path.unlink(missing_ok=True)
    server.send_signal(signal.SIGUSR1 if enabled else signal.SIGUSR2)
    wait_json(path, server, 10, lambda row: row.get("enabled") is enabled)


def archive_sources(inputs: Path, output: Path) -> dict[str, Path]:
    manifest = json.loads((inputs / "manifest.json").read_text())
    sources = {}
    for arm in ("baseline", "candidate"):
        record = manifest["archives"][arm]
        archive = inputs / record["file"]
        if digest(archive) != record["sha256"]:
            raise ValueError(f"{arm} archive hash mismatch")
        destination = output / "sources" / arm
        destination.mkdir(parents=True)
        with tarfile.open(archive) as stream:
            stream.extractall(destination, filter="data")
        sources[arm] = destination
    paths = {str(path.relative_to(sources["baseline"])) for path in sources["baseline"].rglob("*") if path.is_file()}
    candidate_paths = {
        str(path.relative_to(sources["candidate"])) for path in sources["candidate"].rglob("*") if path.is_file()
    }
    if paths != candidate_paths:
        raise AssertionError("source archive paths differ")
    differences = [
        path for path in sorted(paths) if digest(sources["baseline"] / path) != digest(sources["candidate"] / path)
    ]
    progress = manifest.get("comparison") == "batcher_progress"
    expected = PROGRESS_PATHS if progress else [PROFILE]
    if differences != expected:
        raise AssertionError(f"unexpected source delta: {differences}")
    if progress:
        if manifest.get("changed_paths") != expected or manifest.get("candidate_is_derived_archive") is not False:
            raise AssertionError("invalid whole-source progress manifest")
        worker = SOURCE_PATHS["worker"]
        if executable_ast((sources["baseline"] / worker).read_bytes()) != executable_ast(
            (sources["candidate"] / worker).read_bytes()
        ):
            raise AssertionError("worker executable code differs")
        profile = yaml.safe_load((sources["baseline"] / PROFILE).read_text())
        if profile["profiles"]["default"].get("adaptive_batching", {}).get("enabled", True) is not True:
            raise AssertionError("progress comparison requires adaptive batching in both arms")
    atomic_json(output / "source-verification.json", {"manifest": manifest, "changed_paths": differences})
    return sources


def validate_ready_sources(ready: dict, source: Path, profile: dict) -> None:
    expected = {name: {"path": path, "sha256": digest(source / path)} for name, path in SOURCE_PATHS.items()}
    if ready["source"] != expected:
        raise AssertionError("actual imported source differs from the assigned immutable arm")
    adaptive = profile["profiles"]["default"].get("adaptive_batching", {}).get("enabled", True)
    if (ready["adaptive"] is not None) != adaptive:
        raise AssertionError("loaded worker adaptive state differs from its archived profile")


def arm_run(args: argparse.Namespace, arm: str, source: Path, deadline: float) -> dict:
    output = args.output / arm
    output.mkdir()
    models = output / "models"
    models.mkdir()
    profile = yaml.safe_load((source / PROFILE).read_text())
    if args.model_path is not None:
        if not args.smoke:
            raise ValueError("model-path is restricted to explicit CPU smoke")
        profile.pop("hf_id", None)
        profile.pop("hf_revision", None)
        profile["weights_path"] = str(args.model_path.resolve())
    (models / "model.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    env = source_environment(source)
    # Keep the socket reserved until the server binds it: Uvicorn receives the fd.
    server = child = None
    reservation = socket.socket()
    record: dict[str, Any] = {"arm": arm, "completed": False, "smoke": args.smoke, "phases": []}
    try:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        reservation.listen()
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tools.benchmarks.gliclass_mixed.server",
                    "--source",
                    str(source),
                    "--models-dir",
                    str(models),
                    "--output",
                    str(output),
                    "--port",
                    str(port),
                    "--device",
                    "cpu" if args.smoke else "cuda",
                    "--model",
                    MODEL,
                    "--fd",
                    str(reservation.fileno()),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(reservation.fileno(),),
            )
        atomic_json(
            output / "server-process-start.json",
            {**process_identity(server.pid), "role": "server", "output": str(output)},
        )
        reservation.close()
        ready = wait_json(output / "ready.json", server, min(180, deadline - time.monotonic() - 45))
        validate_ready_sources(ready, source, profile)
        if not args.smoke and ready["dtype"] != "torch.float16":
            raise AssertionError("GPU run did not use the fixed FP16 precision")
        record["ready"] = ready
        switch(server, output, True)
        for phase in ("correctness", "C1", "C8", "C32", "mixed"):
            if phase == "C1":
                switch(server, output, False)
            elif phase == "mixed":
                switch(server, output, True)
            phase_limit = (
                120 if args.smoke else {"correctness": 240, "C1": 100, "C8": 280, "C32": 100, "mixed": 790}[phase]
            )
            if deadline - time.monotonic() <= phase_limit + 45:
                raise TimeoutError(f"insufficient full phase and cleanup budget before {arm}/{phase}")
            command = [
                sys.executable,
                "-m",
                "tools.benchmarks.gliclass_mixed.client",
                "--url",
                f"http://127.0.0.1:{port}",
                "--source",
                str(source),
                "--output",
                str(output),
                "--phase",
                phase,
                "--model",
                MODEL,
            ]
            if args.smoke:
                command.append("--smoke")
            with (output / f"{phase}.log").open("w") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(
                output / f"{phase}-process-start.json",
                {**process_identity(child.pid), "role": "client", "output": str(output), "phase": phase},
            )
            try:
                code = child.wait(timeout=phase_limit)
            finally:
                cleanup = stop(child)
                atomic_json(output / f"{phase}-process.json", cleanup)
            if code:
                raise RuntimeError(f"{arm}/{phase} failed with exit {code}")
            record["phases"].append(phase)
            atomic_json(output / "arm.json", record)
        # Keep the original server alive after every client has reached its full
        # deadline; trace late completions rather than pretending timeout cancels it.
        observation = 0.1 if args.smoke else 60
        if deadline - time.monotonic() <= observation + 45:
            raise TimeoutError("insufficient post-drain observation and cleanup budget")
        time.sleep(observation)
        record["completed"] = True
    except BaseException as error:
        record["error"] = error_chain(error)
        raise
    finally:
        reservation.close()
        for name, process in (("client_cleanup", child), ("server_cleanup", server)):
            try:
                record[name] = stop(process)
            except BaseException as error:  # noqa: BLE001 -- retain failure and still clean the other owned child.
                record[name] = {"error": error_chain(error), "reaped": False}
                record["completed"] = False
        atomic_json(output / "arm.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-s", type=float, default=RUNNER_CEILING)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.inputs, args.output = args.inputs.resolve(), args.output.resolve()
    if not 0 < args.deadline_s <= RUNNER_CEILING:
        raise ValueError("deadline must be positive and no greater than3900s")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    summary: dict[str, Any] = {
        "schema": 1,
        "completed": False,
        "smoke": args.smoke,
        "arms": [],
        "start_monotonic": start,
        "deadline_monotonic": start + args.deadline_s,
        "qualification": False,
        "interpretation": "descriptive mixed-load reproduction",
    }

    def terminate(signum: int, _: Any) -> None:
        raise RuntimeError(f"runner received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    try:
        sources = archive_sources(args.inputs, args.output)
        for arm in ("baseline", "candidate"):
            summary["arms"].append(arm_run(args, arm, sources[arm], start + args.deadline_s))
        baseline, candidate = summary["arms"]
        for key in (
            "versions",
            "dtype",
            "device",
            "gpu_name",
            "threads",
            "checkpoint_path",
            "checkpoint_revision",
            "cuda_graphs",
        ):
            if baseline["ready"][key] != candidate["ready"][key]:
                raise AssertionError(f"arm runtime/source mismatch: {key}")
        if not all(arm["completed"] and arm["server_cleanup"]["reaped"] for arm in summary["arms"]):
            raise AssertionError("arm failed or child was not reaped")
        summary["completed"] = True
    except BaseException as error:
        summary["error"] = error_chain(error)
        raise
    finally:
        summary["end_monotonic"] = time.monotonic()
        atomic_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    with parent_guard():
        main()
