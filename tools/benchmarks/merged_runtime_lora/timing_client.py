"""SDK-only C1 child; reuse the reviewed wire capture, lane loop and EOF guard."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, ClassVar, cast

from tools.benchmarks.cross_encoder_lora import client as retained
from tools.benchmarks.cross_encoder_lora.evidence import Evidence, atomic_json, digest
from tools.benchmarks.cross_encoder_lora.loadgen import Control, identity, require_sdk_only
from tools.benchmarks.cross_encoder_lora.process_metrics import boundary_gc_report, snapshot, validate_boundary_gc
from tools.benchmarks.cross_encoder_lora.protocol import require

ROWS = 4


class ExtractClient(retained.Client):
    plan: ClassVar[dict[str, Any]] = {}

    def score(self, selection: Any, label: Any, *, cold: Any = False, timed: Any = False) -> Any:
        require(selection is None and not cold and timed, "ordinary timing must not select adapters or retry")
        started = time.perf_counter_ns()
        wall_started = time.time_ns()
        first = len(self.responses)
        result = self.sdk.extract(
            "proof/model",
            [{"id": f"{label}:{i}", "text": text} for i, text in enumerate(self.plan["entity_texts"])],
            labels=self.plan["entity_labels"],
            options={},
            wait_for_capacity=False,
            max_oom_retries=0,
            provision_timeout_s=30,
        )
        elapsed = (time.perf_counter_ns() - started) / 1e6
        return {
            "label": label,
            "selection": None,
            "latency_ms": elapsed,
            "start_ns": wall_started,
            "end_ns": time.time_ns(),
            "result": result,
            "responses": self.responses[first:],
        }


def validate_extract(row: Any, *, allow_loading: Any = False) -> Any:
    require(not allow_loading, "cold requests are outside timing")
    retained.serialize_responses(row)
    require([r["status"] for r in row["responses"]] == [200], "warm extraction retried/failed")
    require(len(row["result"]) == ROWS, "extraction row coverage differs")
    require([r["id"] for r in row["result"]] == [f"{row['label']}:{i}" for i in range(ROWS)], "extraction IDs differ")


def run(args: Any) -> Any:
    plan = json.loads(args.plan.read_text())
    require(digest(args.plan) == args.plan_sha256, "timing plan differs")
    require_sdk_only()
    cast("Any", retained).MODEL = "proof/model"
    cast("Any", retained).QUERY = {"text": plan["query"]}
    cast("Any", retained).DOCUMENTS = plan["documents"]
    if args.family == "gliner2":
        ExtractClient.plan = plan
        cast("Any", retained).Client = ExtractClient
        cast("Any", retained).validate_response = validate_extract
    control = Control(args.token)
    evidence = Evidence(args.output)
    terminal = {
        "passed": False,
        "family": args.family,
        "identity": identity(args.source),
        "plan_sha256": args.plan_sha256,
        "mode": args.mode,
    }
    observations = {}
    callbacks = ()
    try:
        control.send("hello", identity=terminal["identity"])
        control.receive("phase", 1)

        def before() -> Any:
            nonlocal callbacks
            callbacks = tuple(gc.callbacks)
            observations["before"] = snapshot()

        def after(clock: Any) -> Any:
            observations["clock"] = clock
            observations["after"] = snapshot()
            observations["gc"] = boundary_gc_report(observations["before"], observations["after"], callbacks)

        settings = plan["latency"] if args.mode == "ordinary" else plan["opt_in_latency"]
        result = retained.performance(
            args.url,
            1,
            evidence,
            smoke=False,
            warmup=settings["warmup_calls"],
            sequential=settings["measured_calls"],
            future_timeout=120,
            before_measure=before,
            after_measure=after,
        )
        validate_boundary_gc(observations)
        require_sdk_only()
        result.update(identity=terminal["identity"], observations=observations, family=args.family)
        path = args.output / "performance.json"
        atomic_json(path, result)
        terminal.update(passed=True, result_sha256=digest(path), result_bytes=path.stat().st_size)
    except BaseException as error:
        terminal["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        control.close()
        atomic_json(args.output / "observations.json", observations)
        atomic_json(args.output / "terminal.json", terminal)
        if terminal["passed"]:
            control.send("finished", terminal_sha256=digest(args.output / "terminal.json"))


def main() -> Any:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output", "source"):
        p.add_argument(f"--{name}", type=Path, required=True)
    for name in ("plan-sha256", "url", "token", "family"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--mode", choices=("ordinary", "opt-in-resident-base"), required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
