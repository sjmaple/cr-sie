"""Bounded real-server fixtures shared by the fixed pretrained GPU proof."""

from __future__ import annotations

# Fixed reviewed protocol counts/statuses are intentionally explicit.
# ruff: noqa: PLR2004
import asyncio
import gc
import hashlib
import json
import math
import os
import socket
import sys
import threading
import time
import weakref
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import httpx
import msgpack
import numpy as np
import torch
import uvicorn
import yaml
from peft.tuners.tuners_utils import BaseTunerLayer
from sie_sdk import SIEClient
from sie_sdk.client.errors import RequestError, ServerError
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig


def require(condition: Any, message: Any) -> Any:
    if not condition:
        raise AssertionError(message)


def digest(path: Any) -> Any:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path: Any, value: Any) -> Any:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def tensor_hash(value: Any) -> Any:
    array = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def memory(device: Any) -> Any:
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        }
    return {"allocated": None, "reserved": None, "peak_allocated": None, "peak_reserved": None}


def close_enough(actual: Any, expected: Any, atol: Any, where: Any = "value") -> Any:
    """Exact structure/nonfloating values, fixed absolute tolerance for finite floats."""
    require(type(actual) is type(expected), f"{where}: type differs")
    if isinstance(actual, float):
        require(math.isfinite(actual) and math.isfinite(expected), f"{where}: nonfinite")
        require(abs(actual - expected) <= atol, f"{where}: numerical difference {actual - expected}")
    elif isinstance(actual, dict):
        require(actual.keys() == expected.keys(), f"{where}: fields differ")
        for key in actual:
            close_enough(actual[key], expected[key], atol, f"{where}.{key}")
    elif isinstance(actual, list):
        require(len(actual) == len(expected), f"{where}: length differs")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            close_enough(a, b, atol, f"{where}[{index}]")
    else:
        require(actual == expected, f"{where}: value differs")


def native_model(adapter: Any, family: Any) -> Any:
    return adapter._model if family == "gliner2" else adapter._model.model


def normalized(value: Any, ids: Any) -> Any:
    aliases = {key: f"row-{i}" for i, key in enumerate(ids)}
    if isinstance(value, dict):
        return {
            k: aliases.get(v, v) if k in {"id", "item_id"} and isinstance(v, str) else normalized(v, ids)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [normalized(v, ids) for v in value]
    return value


class State:
    def __init__(self, adapter: Any, family: Any) -> None:
        self.model = native_model(adapter, family)
        self.modules = dict(self.model.named_modules())
        self.parameters = dict(self.model.named_parameters())
        self.buffers = dict(self.model.named_buffers())
        self.hashes = {name: tensor_hash(p) for name, p in self.parameters.items()}
        self.buffer_hashes = {name: tensor_hash(p) for name, p in self.buffers.items()}
        self.grad = {name: p.requires_grad for name, p in self.parameters.items()}
        self.training = {name: m.training for name, m in self.modules.items()}
        self.outer = adapter._model
        self.processor = self.model.processor if family == "gliner2" else self.outer.tokenizer
        self.activation = None if family == "gliner2" else self.outer.activation_fn

    def check(self, adapter: Any, family: Any, *, hashes: Any = False, training: Any = False) -> Any:
        raw = native_model(adapter, family)
        require(raw is self.model and adapter._model is self.outer, "root/outer identity differs")
        require(dict(raw.named_modules()) == self.modules, "native module tree differs")
        require(dict(raw.named_parameters()) == self.parameters, "native parameters differ")
        require(dict(raw.named_buffers()) == self.buffers, "native buffers differ")
        require(not any(isinstance(m, BaseTunerLayer) for m in raw.modules()), "native tree retains tuner")
        require(all(p.requires_grad == self.grad[n] for n, p in self.parameters.items()), "grad flags differ")
        require(
            (raw.processor if family == "gliner2" else adapter._model.tokenizer) is self.processor,
            "processor/tokenizer identity differs",
        )
        if family != "gliner2":
            require(adapter._model.activation_fn is self.activation, "activation identity differs")
        if training:
            require(all(m.training == self.training[n] for n, m in self.modules.items()), "training flags differ")
        if hashes:
            require({n: tensor_hash(p) for n, p in self.parameters.items()} == self.hashes, "base bytes changed")
            require({n: tensor_hash(p) for n, p in self.buffers.items()} == self.buffer_hashes, "buffer bytes changed")


class Observer:
    """Observe the actual bound root forward, including raw arrays and tuner executions."""

    server: Any
    adapter: Any
    raw: Any
    root: Any
    method: Any
    forward: Any
    prior_forward: Any

    def __init__(self, server: Any) -> None:
        self.server = server
        self.adapter = server.adapter
        self.raw = native_model(self.adapter, server.family)
        self.root = self.raw.encoder if server.family == "gliner2" else self.raw
        self.method_name = "extract" if server.family == "gliner2" else "score_pairs"
        self.method = getattr(self.adapter, self.method_name)
        self.forward = self.root.forward
        self.prior_method = self.adapter.__dict__.get(self.method_name)
        self.had_method = self.method_name in self.adapter.__dict__
        self.prior_forward = self.root.__dict__.get("forward")
        self.had_forward = "forward" in self.root.__dict__
        self.local = threading.local()
        self.records = []
        setattr(self.adapter, self.method_name, self.call)
        self.root.forward = self.root_forward

    def array(self, tensor: Any, label: Any) -> Any:
        value = tensor.detach().cpu().contiguous().numpy()
        require(value.size and np.isfinite(value).all(), "empty/nonfinite observed tensor")
        filename = f"array-{self.server.array_index:05d}-{label}.npy"
        self.server.array_index += 1
        path = self.server.output / filename
        np.save(path, value, allow_pickle=False)
        return {"file": filename, "sha256": digest(path), "shape": list(value.shape), "dtype": str(value.dtype)}

    def call(self, *args: Any, **kwargs: Any) -> Any:
        items = args[0] if self.server.family == "gliner2" else args[1]
        manager = getattr(self.adapter, "_lora", None)
        selection = manager._active if manager is not None else self.server.static_selection
        row: dict[str, Any] = {
            "ids": [i.id for i in items],
            "texts": [i.text for i in items],
            "selection": selection,
            "forwards": [],
            "tuners": [],
            "started_ns": time.monotonic_ns(),
        }
        self.local.row = row
        self.records.append(row)
        layers = {n: m for n, m in self.raw.named_modules() if isinstance(m, BaseTunerLayer)}
        expected = PEFTLoRAMixin._peft_adapter_name(selection) if manager is not None and selection else selection
        handles = []
        for name, layer in layers.items():

            def observed(module: Any, _args: Any, _output: Any, name: Any = name) -> Any:
                row["tuners"].append(
                    {"name": name, "active": list(module.active_adapters), "disabled": bool(module.disable_adapters)}
                )

            handles.append(layer.register_forward_hook(observed))
        try:
            if manager is not None and selection is None:
                self.server.state.check(self.adapter, self.server.family)
            result = self.method(*args, **kwargs)
            require(row["forwards"], "actual root forward not observed")
            require(sum(r["rows"] for r in row["forwards"]) == len(items), "forward row coverage differs")
            if selection is None:
                require(not layers and not row["tuners"], "native base forward used PEFT")
            else:
                require(layers and row["tuners"], "selected tuner executions missing")
                require(
                    {r["name"] for r in row["tuners"]} == layers.keys(), "selected tuner execution coverage differs"
                )
                require(
                    all(r["active"] == [expected] and not r["disabled"] for r in row["tuners"]),
                    "selected tuner state differs",
                )
            row["finished_ns"] = time.monotonic_ns()
            return result
        finally:
            for handle in handles:
                handle.remove()
            del self.local.row

    def root_forward(self, *args: Any, **kwargs: Any) -> Any:
        row = self.local.row
        require(not args, "unexpected positional root inputs")
        inputs = {k: self.array(v, "input") for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        output = self.forward(*args, **kwargs)
        value = output.last_hidden_state if self.server.family == "gliner2" else output.logits
        row["forwards"].append({"inputs": inputs, "output": self.array(value, "output"), "rows": int(value.shape[0])})
        return output

    def close(self) -> Any:
        if self.had_method:
            setattr(self.adapter, self.method_name, self.prior_method)
        else:
            delattr(self.adapter, self.method_name)
        if self.had_forward:
            self.root.forward = self.prior_forward
        else:
            delattr(self.root, "forward")
        self.server = self.adapter = self.raw = self.root = self.method = self.forward = None
        self.prior_method = self.prior_forward = None


class Server:
    loop: Any
    observer: Any
    state: Any
    adapter: Any
    worker: Any
    registry: Any
    requests: list[dict[str, Any]]

    def __init__(
        self, plan: Any, family: Any, output: Any, *, role: Any, adapter_path: Any, paths: Any = None, capacity: Any = 2
    ) -> None:
        self.plan, self.family, self.output, self.role = plan, family, Path(output), role
        self.output.mkdir(parents=True, exist_ok=False)
        self.model_id = "proof/model"
        self.device = plan["environment"]["device"]
        self.paths = paths or {}
        self.capacity = capacity
        self.adapter_path = adapter_path
        self.requests = []
        self.request_stages = set()
        self.request_guard = threading.Lock()
        self.array_index = 0
        self.observer = self.state = self.adapter = self.worker = self.registry = None
        self.static_selection = None
        self.records = {
            "role": role,
            "family": family,
            "adapter_path": adapter_path,
            "requests": self.requests,
            "capacity": capacity,
            "cleanup": {"passed": False},
        }
        self.submit_restore = None
        self.wave = None

    def on_loop(self, coroutine: Any, timeout: Any = 130) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout=timeout)

    def mutate(self, function: Any) -> Any:
        return self.on_loop(self.worker.run_adapter_mutation(function))

    def __enter__(self) -> Any:
        spec = self.plan["models"][self.family]
        opts: dict[str, Any] = {"compute_precision": "float32"}
        if self.family == "gliner2":
            opts.update(threshold=self.plan["entity_threshold"], max_seq_length=self.plan["sequence_limits"]["gliner2"])
        else:
            opts["max_length"] = self.plan["sequence_limits"]["ce"]
        if self.paths:
            opts["lora_paths"] = self.paths
        # A local weights_path is the model source. hf_id would ask the registry
        # to fetch that remote repository even when the local copy is complete.
        # The exact public repo/revision remains in the independently hashed plan.
        config = {
            "sie_id": self.model_id,
            "weights_path": spec["directory"],
            "inputs": {"text": True},
            "tasks": {"extract" if self.family == "gliner2" else "score": {}},
            "profiles": {
                "default": {
                    "adapter_path": self.adapter_path,
                    "max_batch_tokens": 32768,
                    "adapter_options": {"loadtime": opts},
                }
            },
        }
        models = self.output / "models"
        models.mkdir()
        (models / "model.yaml").write_text(yaml.safe_dump(config))
        self.prior_capacity = os.environ.get("SIE_MAX_LORAS_PER_MODEL")
        os.environ["SIE_MAX_LORAS_PER_MODEL"] = str(self.capacity)
        app = AppFactory.create_app(
            AppStateConfig(models_dir=models, device=self.device, remote_serving=False, bind_host="127.0.0.1")
        )
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(128)
        self.url = f"http://127.0.0.1:{self.socket.getsockname()[1]}"
        self.server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", access_log=False, timeout_graceful_shutdown=5)
        )

        async def serve() -> Any:
            self.loop = asyncio.get_running_loop()
            await self.server.serve(sockets=[self.socket])

        self.thread = threading.Thread(target=lambda: asyncio.run(serve()), name=f"proof-{self.role}")
        self.thread.start()
        try:
            deadline = time.monotonic() + 30
            while not self.server.started:
                require(self.thread.is_alive() and time.monotonic() < deadline, "server startup failed")
                time.sleep(0.01)
            self.registry = app.state.registry
            require(not self.registry.loaded_model_names, "model was not cold")
            self.records["initially_unloaded"] = True
            self.call("model-cold", cold_model=True)
            require(self.observer is not None, "cold load did not install observation before first forward")
            return self
        except BaseException:
            self.close()
            raise

    def loaded(self) -> Any:
        self.adapter = self.registry._loaded[self.model_id].adapter
        self.worker = self.registry._loaded[self.model_id].worker
        require(self.worker is not None, "actual worker missing")
        require(self.registry._loaded[self.model_id].max_loras == self.capacity, "actual registry capacity differs")
        raw = native_model(self.adapter, self.family)
        require(
            all(p.dtype == torch.float32 and p.device.type == self.device.split(":")[0] for p in raw.parameters()),
            "model dtype/device differs",
        )
        self.records["model_class"] = type(raw).__name__
        self.records["adapter_class"] = f"{type(self.adapter).__module__}:{type(self.adapter).__name__}"
        self.state = State(self.adapter, self.family)
        self.weak_model = weakref.ref(raw)
        self.observer = Observer(self)
        self.records["observations"] = self.observer.records

    def ready(self, *, model: Any = False) -> Any:
        async def wait() -> Any:
            if model:
                await asyncio.wait_for(self.registry.load_async(self.model_id, self.device), 120)
            else:
                await asyncio.wait_for(asyncio.gather(*tuple(self.registry._background_tasks)), 30)

        self.on_loop(wait())

    def call(
        self,
        stage: Any,
        selection: Any = None,
        rows: Any = 4,
        *,
        expected: Any = 200,
        code: Any = None,
        ids: Any = None,
        observe: Any = True,
        cold_model: Any = False,
    ) -> Any:
        ids = ids or [f"row-{i}" for i in range(rows)]
        texts = self.plan["entity_texts"] if self.family == "gliner2" else self.plan["documents"]
        items: Any = [{"id": i, "text": text} for i, text in zip(ids, texts[:rows], strict=True)]
        record: dict[str, Any] = {"stage": stage, "selection": selection, "ids": ids, "texts": texts[:rows], "wire": []}
        first_observation = len(self.observer.records) if self.observer is not None else 0
        with self.request_guard:
            require(stage not in self.request_stages, "duplicate immutable request stage")
            self.request_stages.add(stage)
            self.requests.append(record)

        def capture(response: Any) -> Any:
            response.read()
            body = (
                msgpack.unpackb(response.content, raw=False)
                if "msgpack" in response.headers.get("content-type", "")
                else response.json()
            )
            record["wire"].append(
                {
                    "status": response.status_code,
                    "body": body,
                    "body_hex": response.content.hex(),
                    "retry_after": response.headers.get("retry-after"),
                    "timing_headers": {k: v for k, v in response.headers.items() if k.startswith("x-")},
                }
            )
            if cold_model and response.status_code == 503 and self.observer is None:
                error = body.get("detail", body.get("error", body))
                require(error.get("code") == "MODEL_LOADING", "unexpected cold-model response")
                # Hold only this cold SDK retry while its already-started load
                # finishes. Snapshot the native model before its first forward.
                self.ready(model=True)
                self.loaded()

        try:
            with (
                httpx.Client(
                    base_url=self.url, timeout=30, trust_env=False, event_hooks={"response": [capture]}
                ) as http,
                SIEClient(self.url, http_client=http, remote="forbid") as client,
            ):
                sdk = cast("Any", client)
                options = {"lora": selection} if selection is not None else {}
                kwargs: Any = {
                    "options": options,
                    "wait_for_capacity": False,
                    "max_oom_retries": 0,
                    "provision_timeout_s": 120 if cold_model else 30,
                }
                started = time.perf_counter_ns()
                if self.family == "gliner2":
                    result = sdk.extract(self.model_id, items, labels=self.plan["entity_labels"], **kwargs)
                else:
                    result = sdk.score(self.model_id, {"text": self.plan["query"]}, items, **kwargs)
                record["elapsed_ns"] = time.perf_counter_ns() - started
                record["result"] = result
            require(expected == 200, "expected denial succeeded")
        except (RequestError, ServerError) as error:
            record["error"] = {"status": error.status_code, "code": error.code, "type": type(error).__name__}
            if error.status_code != expected or error.code != code:
                raise
            if expected == 503:
                retry_after = getattr(error, "retry_after", None)
                require(retry_after is not None and retry_after > 0, "loading Retry-After missing")
        finally:
            primary = sys.exception()
            if self.observer is not None and observe:
                record["observations"] = [
                    row for row in self.observer.records[first_observation:] if set(row["ids"]) == set(ids)
                ]
            # Request threads only persist their own completed evidence. The
            # aggregate is written after all callers join, during close().
            filename = "request-" + hashlib.sha256(stage.encode()).hexdigest() + ".json"
            try:
                atomic(self.output / filename, record)
            except BaseException as secondary:
                if primary is None:
                    raise
                primary.add_note(f"Request evidence also failed: {secondary!r}")
        if cold_model:
            require(len(record["wire"]) >= 2 and record["wire"][-1]["status"] == 200, "cold loading sequence missing")
            for row in record["wire"][:-1]:
                error = row["body"].get("detail", row["body"].get("error", row["body"]))
                require(
                    row["status"] == 503 and error["code"] == "MODEL_LOADING" and float(row["retry_after"]) > 0,
                    "unexpected cold-model retry",
                )
        else:
            require([r["status"] for r in record["wire"]] == [expected], "unexpected wire retry/count")
        if expected == 200 and self.observer is not None and observe:
            require(len(record["observations"]) == 1, "actual worker batch missing")
        if expected != 200:
            require(not record.get("observations"), "denied request reached inference")
        return record

    def close(self) -> Any:
        errors = []
        if self.observer is not None:
            self.observer.close()
            self.observer = None
        self.state = None
        self.adapter = self.worker = None
        if hasattr(self, "server"):
            self.server.should_exit = True
        if hasattr(self, "thread"):
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                self.server.force_exit = True
                self.thread.join(timeout=1)
            if self.thread.is_alive():
                errors.append("server thread not joined")
        if hasattr(self, "socket"):
            self.socket.close()
        self.registry = None
        if hasattr(self, "prior_capacity"):
            if self.prior_capacity is None:
                os.environ.pop("SIE_MAX_LORAS_PER_MODEL", None)
            else:
                os.environ["SIE_MAX_LORAS_PER_MODEL"] = self.prior_capacity
        gc.collect()
        self.records["cleanup"] = {
            "passed": not errors,
            "errors": errors,
            "model_alive": bool(getattr(self, "weak_model", lambda: None)()),
        }
        if self.records["cleanup"]["model_alive"]:
            errors.append("model reference retained")
            self.records["cleanup"]["passed"] = False
        atomic(self.output / "result.json", self.records)
        require(not errors, "; ".join(errors))

    def __exit__(
        self, exc_type: type[BaseException] | None, error: BaseException | None, _tb: TracebackType | None
    ) -> bool:
        try:
            self.close()
        except BaseException as secondary:
            if error is None:
                raise
            error.add_note(f"cleanup also failed: {secondary!r}")
        return False
