"""A single fixed three-model proof and six-pair ordinary C1 experiment."""

from __future__ import annotations

# Fixed reviewed protocol counts/statuses are intentionally explicit.
# ruff: noqa: PLR2004
import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest
from tools.benchmarks.merged_runtime_lora.provenance import require, terminal_result, verify_tree


def group_exists(pgid: Any) -> Any:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # A just-killed reparented process can remain visible briefly on Darwin.
        # EPERM is never evidence of absence; keep the original bounded wait.
        return True


def cleanup(process: Any, deadline: Any) -> Any:
    """One exact child group, one shared deadline, including an exited leader."""
    signals = []
    if group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            signals.append("TERM")
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=max(0.01, min(5, (deadline - time.monotonic()) / 2)))
    except subprocess.TimeoutExpired:
        pass
    if group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
            signals.append("KILL")
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
    # No repeated grace period: descendants share the original cleanup bound.
    while group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    return {
        "pid": process.pid,
        "pgid": process.pid,
        "signals": signals,
        "reaped": process.poll() is not None,
        "returncode": process.returncode,
        "group_absent": not group_exists(process.pid),
        "finished": time.monotonic(),
    }


def gate(ratios: Any, limits: Any) -> Any:
    if len(ratios) != 6 or not all(math.isfinite(r) and r > 0 for r in ratios):
        raise ValueError("six finite positive independent process-pair ratios required")
    logs = [math.log(r) for r in ratios]
    mean = statistics.mean(logs)
    point = math.exp(mean)
    upper = math.exp(mean + limits["t95_df5"] * statistics.stdev(logs) / math.sqrt(6))
    return {
        "ratios": ratios,
        "geometric_ratio": point,
        "one_sided_95_upper": upper,
        "worst_pair": max(ratios),
        "passed": point <= limits["geometric_ratio_max"]
        and upper <= limits["one_sided_95_upper_max"]
        and max(ratios) <= limits["worst_pair_max"],
    }


def run(plan_path: Any, inputs: Any, output: Any, *, deadline_s: Any = 3900) -> Any:
    plan = json.loads(plan_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + deadline_s
    cleanup_s = plan["proposed_bounds_s"]["child_cleanup_shared"]
    summary = {
        "completed_valid": False,
        "functional_passed": False,
        "ordinary_c1_passed": False,
        "historical_ce_performance_qualification": "failed_unchanged",
        "source_commit": plan["source_commit"],
        "plan_sha256": digest(plan_path),
        "started": started,
        "deadline": deadline,
        "jobs": [],
        "gates": {},
    }
    jobs = [(family, "lifecycle", None, "N") for family in plan["latency"]["families"]]
    jobs += [
        (family, "timing", pair, arm)
        for family in plan["latency"]["families"]
        for pair, order in enumerate(plan["latency"]["orders"])
        for arm in order
    ]
    sources = {"B": inputs / "baseline", "N": inputs / "candidate"}
    revisions = {"B": plan["latency"]["whole_source_baseline"], "N": plan["source_commit"]}
    try:
        manifest_path = inputs / "source-inputs.json"
        require(digest(manifest_path) == plan["source_inputs_sha256"], "source input manifest differs")
        manifest = json.loads(manifest_path.read_text())
        for arm, source in sources.items():
            require(manifest["sources"][arm]["revision"] == revisions[arm], "source revision differs")
            verify_tree(source, manifest["sources"][arm]["files"])
        for index, (family, kind, pair, arm) in enumerate(jobs):
            limit = plan["proposed_bounds_s"]["model_child" if kind == "lifecycle" else "performance_child"]
            admission = time.monotonic()
            record = {
                "index": index,
                "family": family,
                "kind": kind,
                "pair": pair,
                "arm": arm,
                "admitted_at": admission,
                "limit_s": limit,
                "cleanup_s": cleanup_s,
                "margin_s": deadline - admission - limit - cleanup_s,
                "admitted": admission + limit + cleanup_s < deadline,
            }
            summary["jobs"].append(record)
            atomic_json(output / "summary.json", summary)
            if not record["admitted"]:
                raise TimeoutError("fixed runner admission reserve exhausted")
            job_output = output / f"{index:02d}-{family}-{kind}-{arm}"
            token = uuid.uuid4().hex
            expected = {
                "token": token,
                "source": str(sources[arm].resolve()),
                "revision": revisions[arm],
                "family": family,
                "kind": kind,
                "limit_s": limit,
                "plan_sha256": digest(plan_path),
                "inputs_manifest_sha256": digest(manifest_path),
            }
            record["expected_terminal"] = expected
            command = [
                sys.executable,
                "-m",
                "tools.benchmarks.merged_runtime_lora.job",
                "--plan",
                str(plan_path),
                "--plan-sha256",
                digest(plan_path),
                "--source",
                str(sources[arm]),
                "--output",
                str(job_output),
                "--inputs-manifest",
                str(manifest_path),
                "--revision",
                revisions[arm],
                "--family",
                family,
                "--kind",
                kind,
                "--token",
                token,
                "--limit-s",
                str(limit),
            ]
            env = os.environ.copy()
            harness_root = Path(__file__).resolve().parents[3]
            env["PYTHONPATH"] = (
                os.pathsep.join(
                    str(sources[arm] / "packages" / p / "src") for p in ("sie_server", "sie_sdk", "sie_config")
                )
                + os.pathsep
                + str(harness_root)
            )
            env.update(
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                SIE_TELEMETRY_DISABLED="1",
                SIE_DISK_CACHE_ENABLED="false",
                TOKENIZERS_PARALLELISM="false",
                PYTHONDONTWRITEBYTECODE="1",
            )
            print(f"Start {index + 1}/{len(jobs)} {family} {kind} {arm}", flush=True)
            with (output / f"job-{index:02d}.log").open("wb") as log:
                process = subprocess.Popen(  # noqa: S603 -- exact owned source-pinned argv
                    command,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    env=env,
                    cwd=output,
                    start_new_session=True,
                    close_fds=True,
                )
                record.update(pid=process.pid, pgid=os.getpgid(process.pid))
                work_deadline = min(deadline - cleanup_s, time.monotonic() + limit)
                cleanup_deadline = min(deadline, work_deadline + cleanup_s)
                record.update(work_deadline=work_deadline, cleanup_deadline=cleanup_deadline)
                atomic_json(output / "summary.json", summary)
                try:
                    record["returncode"] = process.wait(timeout=max(0.01, work_deadline - time.monotonic()))
                finally:
                    primary = sys.exception()
                    try:
                        record["cleanup"] = cleanup(process, min(cleanup_deadline, time.monotonic() + cleanup_s))
                        if not record["cleanup"]["group_absent"] or not record["cleanup"]["reaped"]:
                            raise RuntimeError("owned child group cleanup incomplete")
                    except BaseException as secondary:
                        record["cleanup_error"] = str(secondary)
                        if primary is None:
                            raise
                        primary.add_note(f"Secondary cleanup error: {secondary!r}")
                    finally:
                        assert process.stdin is not None
                        process.stdin.close()
                        atomic_json(output / "summary.json", summary)
            terminal = job_output / "terminal.json"
            record["terminal_path"] = str(terminal.relative_to(output))
            record["terminal_sha256"] = digest(terminal)
            result = terminal_result(
                job_output, expected, pid=process.pid, ppid=os.getpid(), returncode=record["returncode"]
            )
            record["passed"] = result["passed"] and record["returncode"] == 0
            require(
                result.get("setup_complete") is True, "child environment/source preflight failed; no later jobs started"
            )
            if "metrics" in result:
                record["metrics"] = result["metrics"]
            print(f"Finished {index + 1}/{len(jobs)} {family} {kind} {arm}", flush=True)
            atomic_json(output / "summary.json", summary)
            if not record["passed"] and kind == "timing":
                raise RuntimeError("timing child failed; the fixed paired experiment is incomplete")
            # Preserve independent later model jobs after a numerical verdict;
            # setup/ownership failures raise and abort rather than restarting.
        if len(summary["jobs"]) != 39:
            raise AssertionError("fixed job inventory differs")
        summary["functional_passed"] = all(j["passed"] for j in summary["jobs"] if j["kind"] == "lifecycle")
        measured = [j for j in summary["jobs"] if j["kind"] == "timing"]
        if all(j["passed"] for j in measured):
            for family in plan["latency"]["families"]:
                for metric in plan["latency"]["metrics"]:
                    ratios = []
                    for pair in range(6):
                        arms = {j["arm"]: j for j in measured if j["family"] == family and j["pair"] == pair}
                        ratios.append(arms["N"]["metrics"][metric] / arms["B"]["metrics"][metric])
                    summary["gates"][family + ":" + metric] = gate(ratios, plan["latency"]["gates"])
            summary["ordinary_c1_passed"] = all(g["passed"] for g in summary["gates"].values())
        summary["completed_valid"] = all(j["passed"] for j in summary["jobs"])
    except BaseException as error:  # noqa: BLE001 -- preserve incomplete fixed experiment
        summary["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        summary["finished"] = time.monotonic()
        atomic_json(output / "summary.json", summary)
    return summary


def main() -> Any:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "inputs", "output"):
        p.add_argument(f"--{name}", required=True, type=Path)
    args = p.parse_args()
    result = run(args.plan, args.inputs, args.output)
    raise SystemExit(
        0 if result["completed_valid"] and result["functional_passed"] and result["ordinary_c1_passed"] else 1
    )


if __name__ == "__main__":
    main()
