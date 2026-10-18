"""One source-pinned proof child. stdin EOF belongs solely to its supervisor."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import signal
import time
import traceback
from pathlib import Path
from typing import Any

from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest
from tools.benchmarks.cross_encoder_lora.loadgen import Control
from tools.benchmarks.merged_runtime_lora.provenance import verify_import


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "source", "output", "inputs-manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("plan-sha256", "family", "kind", "token", "revision"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--limit-s", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("job output already exists")
    args.output.mkdir(parents=True)
    control = Control(args.token)
    terminal = {
        "passed": False,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgrp(),
        "token": args.token,
        "inputs_manifest_sha256": digest(args.inputs_manifest),
        "source": str(args.source.resolve()),
        "revision": args.revision,
        "family": args.family,
        "kind": args.kind,
        "started_ns": time.time_ns(),
        "started_monotonic": time.monotonic(),
        "limit_s": args.limit_s,
        "plan_sha256": args.plan_sha256,
    }
    atomic_json(args.output / "ownership.json", terminal)

    def alarm(_signal: Any, _frame: Any) -> Any:
        raise TimeoutError("fixed proof child deadline")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(args.limit_s)
    try:
        import torch  # noqa: PLC0415 -- owned optional ML bootstrap
        from sie_sdk import SIEClient  # noqa: PLC0415 -- owned optional ML bootstrap
        from sie_server.core.registry import ModelRegistry  # noqa: PLC0415 -- owned optional ML bootstrap

        from tools.benchmarks.merged_runtime_lora import (  # noqa: PLC0415 -- owned optional ML bootstrap
            lifecycle,
            timing,
        )

        if digest(args.plan) != args.plan_sha256:
            raise ValueError("job plan hash differs")
        plan = json.loads(args.plan.read_text())
        if digest(args.inputs_manifest) != plan["source_inputs_sha256"]:
            raise ValueError("source input manifest differs")
        manifest = json.loads(args.inputs_manifest.read_text())
        arm = "N" if args.revision == plan["source_commit"] else "B"
        selected = manifest["sources"][arm]
        if selected["revision"] != args.revision:
            raise ValueError("selected source revision differs")
        source = args.source.resolve()
        terminal["imports"] = {}
        for value in (SIEClient, ModelRegistry):
            path = Path(inspect.getfile(value)).resolve()
            terminal["imports"][value.__name__] = verify_import(path, source, selected["files"])
        terminal["versions"] = {name: importlib.metadata.version(name) for name in plan["required_versions"]}
        if terminal["versions"] != plan["required_versions"]:
            raise ValueError("pinned dependency environment differs")
        if plan["environment"]["device"] != "cuda:0" or not torch.cuda.is_available():
            raise ValueError("pretrained GPU proof requires actual CUDA")
        if torch.cuda.device_count() != 1 or "L4" not in torch.cuda.get_device_name(0):
            raise ValueError("expected exactly one L4")
        torch.set_num_threads(plan["environment"]["torch_threads"])
        torch.set_num_interop_threads(plan["environment"]["torch_interop_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.init()
        terminal["device"] = {
            "name": torch.cuda.get_device_name(0),
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
            "cuda_version": torch.version.cuda,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        terminal["setup_complete"] = True
        if args.kind == "lifecycle":
            if args.revision != plan["source_commit"]:
                raise ValueError("lifecycle must use current explicit source")
            result = lifecycle.run(plan, args.family, args.output / "proof", source=source)
        elif args.kind == "timing":
            result = timing.run(args.plan, args.family, source, args.output / "proof", limit_s=args.limit_s - 15)
        else:
            raise ValueError("unknown fixed job kind")
        terminal["passed"] = result["passed"]
        terminal["result_sha256"] = digest(args.output / "proof/summary.json")
        if "metrics" in result:
            terminal["metrics"] = result["metrics"]
    except BaseException as error:  # noqa: BLE001 -- retain all terminal failures before exiting
        terminal["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        signal.alarm(0)
        try:
            control.close()
        except BaseException as error:  # noqa: BLE001 -- retain all terminal failures before exiting
            terminal["passed"] = False
            terminal["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
            terminal.setdefault("error", terminal["cleanup_error"])
        terminal["finished_ns"] = time.time_ns()
        terminal["finished_monotonic"] = time.monotonic()
        atomic_json(args.output / "terminal.json", terminal)
    raise SystemExit(0 if terminal["passed"] else 1)


if __name__ == "__main__":
    main()
