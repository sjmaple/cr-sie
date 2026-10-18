"""Actual local SIE server; optional admission/batch/forward observations.

SIGUSR1 installs observers between phases; SIGUSR2 restores the exact original
functions before homogeneous timing. No control endpoint or serving code change.
Forward intervals are host dispatch intervals, not synchronized CUDA kernel time.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.metadata
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import sie_config
import sie_sdk
import torch
import uvicorn
from sie_server.adapters import gliclass
from sie_server.adapters.gliclass import GLiClassAdapter
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.worker import ModelWorker, model_worker

from tools.benchmarks.gliclass_mixed.evidence import Journal, atomic_json, digest, error_chain, parent_guard
from tools.benchmarks.gliclass_mixed.workload import EXPECTED_VERSIONS, MODEL


class Observers:
    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.enabled = False
        self.active: dict[str, asyncio.Future] = {}
        self.originals: dict[str, Any] = {
            "submit": ModelWorker.submit_extract,
            "batch": ModelWorker._process_batch,
            "extract": GLiClassAdapter.extract,
            "forward": GLiClassAdapter._forward,
        }
        self.local = threading.local()
        self.sequence = 0

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self.enabled:
            return
        if any(not future.done() for future in self.active.values()):
            raise RuntimeError("cannot switch instrumentation with unresolved requests")
        originals, journal, local = self.originals, self.journal, self.local
        owner = self

        async def submit(worker: ModelWorker, prepared_items: Any, items: Any, **kwargs: Any) -> Any:
            ids = [str(item.id) for item in items]
            request = ids[0].rsplit("/", 1)[0]
            journal.write("submit", request=request, item_ids=ids, costs=[item.cost for item in prepared_items])
            try:
                future = await originals["submit"](worker, prepared_items, items, **kwargs)
            except BaseException as error:
                journal.write("admission_error", request=request, error=error_chain(error))
                raise
            owner.active[request] = future
            journal.write("admitted", request=request, pending_items=worker.pending_count)

            def completed(done: asyncio.Future) -> None:
                fields: dict[str, Any] = {"request": request, "cancelled": done.cancelled()}
                if not done.cancelled():
                    error = done.exception()
                    if error is not None:
                        fields["error"] = error_chain(error)
                    else:
                        fields["timing"] = done.result().timing.to_headers()
                journal.write("completed", **fields)
                owner.active.pop(request, None)

            future.add_done_callback(completed)
            return future

        async def batch(worker: ModelWorker, value: Any, *, engine_queue_owned: bool = False) -> None:
            owner.sequence += 1
            batch_id = owner.sequence
            members = [
                {"item_id": str(metadata.items[item.original_index].id), "cost": item.cost}
                for item, metadata in zip(value.items, value.metadata, strict=True)
            ]
            state = worker.get_adaptive_state()
            journal.write(
                "batch_start",
                batch_id=batch_id,
                members=members,
                total_cost=value.total_cost,
                budget=worker._batch_config.max_batch_cost,
                pending_items=worker.pending_count,
                adaptive=None if state is None else dataclasses.asdict(state),
            )
            try:
                await originals["batch"](worker, value, engine_queue_owned=engine_queue_owned)
            except BaseException as error:
                journal.write("batch_error", batch_id=batch_id, error=error_chain(error))
                raise
            finally:
                journal.write("batch_end", batch_id=batch_id)

        def extract(adapter: GLiClassAdapter, values: Any, **kwargs: Any) -> Any:
            ids = [str(item.id) for item in values]
            local.items = ids
            journal.write(
                "extract_start", item_ids=ids, groups=len((kwargs.get("options") or {}).get("label_groups", {}))
            )
            try:
                return originals["extract"](adapter, values, **kwargs)
            finally:
                journal.write("extract_end", item_ids=ids)
                local.items = None

        def forward(adapter: GLiClassAdapter, pipe: Any, inputs: Any, labels: Any, **kwargs: Any) -> Any:
            ids = getattr(local, "items", None)
            if not ids:
                raise AssertionError("unjoined GLiClass forward observation")
            start = time.monotonic_ns()
            result = originals["forward"](adapter, pipe, inputs, labels, **kwargs)
            journal.write(
                "forward",
                item_ids=ids,
                start_ns=start,
                end_ns=time.monotonic_ns(),
                input_shape=list(inputs["input_ids"].shape),
                graphs=kwargs.get("graphs"),
                score_shape=list(result.shape),
                dtype=str(result.dtype),
            )
            return result

        for target, name, observer, original in (
            (ModelWorker, "submit_extract", submit, originals["submit"]),
            (ModelWorker, "_process_batch", batch, originals["batch"]),
            (GLiClassAdapter, "extract", extract, originals["extract"]),
            (GLiClassAdapter, "_forward", forward, originals["forward"]),
        ):
            setattr(target, name, observer if enabled else original)
        self.enabled = enabled
        journal.write("observers", enabled=enabled)


def imported_sources(root: Path) -> dict:
    result = {}
    for name, module in (("worker", model_worker), ("gliclass", gliclass), ("sdk", sie_sdk), ("config", sie_config)):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise AssertionError(f"{name} import escaped immutable source: {path}")
        result[name] = {"path": str(path.relative_to(root)), "sha256": digest(path)}
    return result


async def serve(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    if args.device == "cuda":
        actual = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
        if actual != EXPECTED_VERSIONS or "L4" not in torch.cuda.get_device_name(0):
            raise AssertionError(f"fixed GPU runtime mismatch: {actual}")
    args.output.mkdir(parents=True, exist_ok=True)
    journal = Journal(args.output / "server-events.jsonl")
    observers = Observers(journal)
    sources = imported_sources(args.source.resolve())
    app = AppFactory.create_app(
        AppStateConfig(
            models_dir=args.models_dir,
            device=args.device,
            model_filter=[args.model],
            preload_models=[args.model],
            pinned_models=[args.model],
            remote_serving=False,
        )
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=args.port,
            log_level="warning",
            access_log=False,
            loop="asyncio",
            timeout_graceful_shutdown=30,
            fd=args.fd,
        )
    )

    def switch(signum: int, _: Any) -> None:
        try:
            observers.set_enabled(signum == signal.SIGUSR1)
            atomic_json(
                args.output / "observers.json", {"enabled": observers.enabled, "monotonic_ns": time.monotonic_ns()}
            )
        except BaseException as error:  # noqa: BLE001 -- fail the explicit control acknowledgement.
            atomic_json(args.output / "observers.json", {"error": error_chain(error)})

    # Uvicorn restores and replays TERM after serving; retain normal finally cleanup.
    signal.signal(signal.SIGTERM, lambda *_: None)
    signal.signal(signal.SIGINT, lambda *_: None)
    signal.signal(signal.SIGUSR1, switch)
    signal.signal(signal.SIGUSR2, switch)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("server exited before startup")
            await asyncio.sleep(0.01)
        worker = app.state.registry.get_worker(args.model)
        if worker is None:
            raise RuntimeError("preloaded model has no worker")
        model = worker._adapter._require_pipe().model
        state = worker.get_adaptive_state()
        atomic_json(
            args.output / "ready.json",
            {
                "pid": os.getpid(),
                "source": sources,
                "time_ns": time.time_ns(),
                "adaptive": None if state is None else dataclasses.asdict(state),
                "batch_budget": worker._batch_config.max_batch_cost,
                "dtype": str(next(model.parameters()).dtype),
                "device": str(next(model.parameters()).device),
                "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                "checkpoint_path": worker._adapter._model_name_or_path,
                "checkpoint_revision": worker._adapter._revision,
                "cuda_graphs": worker._adapter._cuda_graphs,
                "threads": torch.get_num_threads(),
                "versions": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
                "profile_sha256": digest(args.models_dir / "model.yaml"),
            },
        )
        await task
    finally:
        if not task.done():
            server.should_exit = True
            await task
        journal.write("shutdown", unresolved=list(observers.active), server_exited=True)
        journal.close()
        atomic_json(args.output / "server-terminal.json", {"pid": os.getpid(), "server_exited": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--fd", type=int)
    args = parser.parse_args()
    with parent_guard():
        asyncio.run(serve(args))


if __name__ == "__main__":
    main()
