"""Fixed pretrained explicit-LoRA lifecycle proof; independent of latency gates."""

from __future__ import annotations

# Fixed reviewed protocol counts/statuses are intentionally explicit.
# ruff: noqa: PLR2004
import copy
import gc
import itertools
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel
from safetensors.torch import load_file, save_file

from tools.benchmarks.merged_runtime_lora.common import (
    Server,
    atomic,
    close_enough,
    digest,
    memory,
    native_model,
    normalized,
    require,
    tensor_hash,
)
from tools.benchmarks.merged_runtime_lora.timing import run as measure


def fixtures(plan: Any, family: Any, output: Any) -> Any:
    """Deterministic fixtures, completed and hashed before model inference."""
    output.mkdir()
    result = {}
    if family == "gliner2":
        a = Path(plan["gliner2_adapter_A"]["directory"])
        b = output / "b"
        b.mkdir()
        (b / "adapter_config.json").write_bytes((a / "adapter_config.json").read_bytes())
        original = load_file(a / "adapter_model.safetensors")
        require(original and all(".lora_A." in k or ".lora_B." in k for k in original), "unsupported fixture weights")
        weights = {k: (-v if ".lora_B." in k else v).contiguous() for k, v in original.items()}
        require(all(torch.isfinite(v).all() for v in weights.values()), "nonfinite fixture")
        save_file(weights, b / "adapter_model.safetensors")
        reread = load_file(b / "adapter_model.safetensors")
        require(all(torch.equal(reread[k], v) for k, v in weights.items()), "fixture serialization differs")
        paths = {"a": str(a), "b": str(b)}
    else:
        cfg = json.loads((Path(plan["models"][family]["directory"]) / "config.json").read_text())
        prefix = "bert" if family == "bert" else "roberta"
        paths = {}
        for alias, seed in zip(("a", "b"), plan["fixture_ce"]["seeds"], strict=True):
            target = output / alias
            target.mkdir()
            LoraConfig(
                r=4, lora_alpha=8, target_modules=["query", "value"], lora_dropout=0, bias="none", inference_mode=True
            ).save_pretrained(target)
            generator = torch.Generator().manual_seed(seed)
            weights = {}
            width = cfg["hidden_size"]
            for layer in range(cfg["num_hidden_layers"]):
                for name in ("query", "value"):
                    key = f"base_model.model.{prefix}.encoder.layer.{layer}.attention.self.{name}"
                    weights[key + ".lora_A.weight"] = torch.randn(4, width, generator=generator) * plan[
                        "fixture_ce"
                    ].get("scale", 0.04)
                    weights[key + ".lora_B.weight"] = torch.randn(width, 4, generator=generator) * plan[
                        "fixture_ce"
                    ].get("scale", 0.04)
            save_file(weights, target / "adapter_model.safetensors")
            paths[alias] = str(target)
    for alias, directory in paths.items():
        tensors = load_file(Path(directory) / "adapter_model.safetensors")
        result[alias] = {
            "files": {p.name: digest(p) for p in Path(directory).iterdir() if p.is_file()},
            "parameter_bytes": sum(v.numel() * v.element_size() for v in tensors.values()),
            "tensors": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": tensor_hash(v)} for k, v in tensors.items()
            },
        }
    atomic(output / "manifest.json", result)
    return paths, result


def archived_class(plan: Any, family: Any, output: Any) -> Any:
    kind = "gliner2" if family == "gliner2" else "ce"
    info = plan["ordinary_baseline"][kind]
    source = Path(plan["baseline_modules"][kind])
    require(digest(source) == info["sha256"], "archived ordinary source differs")
    return f"{source.resolve()}:{'GLiNER2Adapter' if family == 'gliner2' else 'CrossEncoderAdapter'}"


def compare(actual: Any, actual_dir: Any, expected: Any, expected_dir: Any, atol: Any) -> Any:
    require(actual["wire"][0]["status"] == expected["wire"][0]["status"] == 200, "oracle status differs")
    close_enough(normalized(actual["result"], actual["ids"]), normalized(expected["result"], expected["ids"]), atol)
    close_enough(
        normalized(actual["wire"][0]["body"], actual["ids"]),
        normalized(expected["wire"][0]["body"], expected["ids"]),
        atol,
    )
    require(len(actual["observations"]) == len(expected["observations"]) == 1, "oracle batch absent")
    aa, ee = actual["observations"][0], expected["observations"][0]
    require(aa["texts"] == ee["texts"] and len(aa["forwards"]) == len(ee["forwards"]), "oracle shape/order differs")
    maximum = 0.0
    for a, e in zip(aa["forwards"], ee["forwards"], strict=True):
        require(a["inputs"].keys() == e["inputs"].keys() and a["rows"] == e["rows"], "oracle root inputs differ")
        for key in a["inputs"]:
            require(a["inputs"][key]["sha256"] == e["inputs"][key]["sha256"], "oracle token tensors differ")
        av = np.load(Path(actual_dir) / a["output"]["file"], allow_pickle=False)
        ev = np.load(Path(expected_dir) / e["output"]["file"], allow_pickle=False)
        require(av.shape == ev.shape and av.dtype == ev.dtype, "oracle output shape/dtype differs")
        require(np.isfinite(av).all() and np.isfinite(ev).all(), "nonfinite oracle")
        maximum = max(maximum, float(np.max(np.abs(av.astype(np.float64) - ev.astype(np.float64)))))
    require(maximum <= atol, f"raw root output mismatch: {maximum}")
    actual["oracle_join"] = {"stage": expected["stage"], "root_max_abs_difference": maximum, "atol": atol}
    return maximum


def resident_storage(adapter: Any, family: Any, *, device: Any) -> Any:
    """Join actual attached/detached tensors to the manager's admitted owners."""
    manager = adapter._lora
    parameters = manager._loaded if family == "gliner2" else manager._parameters
    require(set(parameters) == set(manager._loaded), "resident admission sources differ")
    declared = {}
    sources = {}
    for source, values in parameters.items():
        require(values and len({id(p) for p in values}) == len(values), "empty/duplicate resident parameter list")
        require(not set(declared).intersection(id(p) for p in values), "resident parameters have two source owners")
        declared.update({id(p): p for p in values})
        sources[source] = sum(p.numel() * p.element_size() for p in values)
        if family != "gliner2":
            require(sources[source] == manager._loaded[source], "resident admission bytes differ")
    actual = {id(p): p for name, p in native_model(adapter, family).named_parameters() if ".lora_" in name}
    for layer in manager._resident_layers.values():
        actual.update({id(p): p for name, p in layer.named_parameters() if name.startswith("lora_")})
    require(set(actual) == set(declared), "actual resident tensors differ from admitted ownership")
    storages = {}
    expected_device = torch.device(device)
    for p in actual.values():
        require(p.device == expected_device and p.dtype == torch.float32, "resident device/dtype differs")
        require(not p.requires_grad and p.numel() and torch.isfinite(p).all(), "invalid resident parameter")
        s = p.untyped_storage()
        storages[(str(p.device), s.data_ptr())] = s.nbytes()
    return {
        "sources": sorted(parameters),
        "unique_storage_bytes": sum(storages.values()),
        "source_bytes": sources,
        "device": str(expected_device),
        "dtype": "torch.float32",
        "storages": len(storages),
        "parameter_tensors": len(actual),
    }


def wave(server: Any, order: Any, index: Any) -> Any:
    """All three requests queue behind the actual existing dispatch lock."""
    queued = set()
    ready = threading.Event()
    original = server.worker.submit_extract if server.family == "gliner2" else server.worker.submit_score
    owners = {f"wave{index}-{role}-{i}": role for role in order for i in range(4)}
    name = "submit_extract" if server.family == "gliner2" else "submit_score"
    present, prior = name in server.worker.__dict__, server.worker.__dict__.get(name)

    async def observed(*args: Any, **kwargs: Any) -> Any:
        future = await original(*args, **kwargs)
        items = kwargs.get("items")
        if items is None:
            items = args[0]
        matching = {owners[item.id] for item in items if item.id in owners}
        require(len(matching) == 1 and not future.done(), "concurrent queue admission differs")
        queued.update(matching)
        if queued == set(order):
            ready.set()
        return future

    setattr(server.worker, name, observed)
    lock = server.worker._adapter_dispatch_lock
    server.on_loop(lock.acquire())
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(
                    server.call,
                    f"wave{index}-{role}",
                    None if role == "base" else role,
                    4,
                    ids=[f"wave{index}-{role}-{i}" for i in range(4)],
                )
                for role in order
            ]
            try:
                require(ready.wait(15), "three overlapping selections did not queue")
            finally:
                server.loop.call_soon_threadsafe(lock.release)
            return [future.result(timeout=45) for future in futures]
    finally:
        if present:
            setattr(server.worker, name, prior)
        else:
            delattr(server.worker, name)


def run(plan: Any, family: Any, output: Any, *, source: Any = None) -> Any:
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    receipt = {
        "passed": False,
        "functional_only": True,
        "family": family,
        "performance_qualification": False,
        "source_commit": plan["source_commit"],
        "stages": [],
        "memory": [],
        "effects": [],
    }
    atomic(output / "summary.json", receipt)
    plan_path = output / "execution-plan.json"
    atomic(plan_path, plan)
    source = Path(source) if source is not None else Path(__file__).resolve().parents[3]
    paths, fixture = fixtures(plan, family, output / "fixtures")
    paths["a-alias"] = paths["a"]
    spec = plan["models"][family]
    archived = archived_class(plan, family, output)
    oracles = {}
    ordinary = {}
    try:
        receipt["memory"].append({"stage": "context", **memory(plan["environment"]["device"])})
        for role, path in (("archived-ordinary", archived), ("current-ordinary", spec["ordinary"])):
            directory = output / role
            with Server(plan, family, directory, role=role, adapter_path=path) as server:
                require(not hasattr(server.adapter, "_lora"), "ordinary adapter initialized a LoRA manager")
                for rows in (2, 4):
                    row = server.call(f"{role}-base-{rows}", rows=rows)
                    if role == "archived-ordinary":
                        ordinary[rows] = copy.deepcopy(row)
                    else:
                        compare(row, directory, ordinary[rows], output / "archived-ordinary", plan["numeric"]["atol"])
                        oracles["base", rows] = copy.deepcopy(row)
                if role == "current-ordinary":
                    for rows in (2, 4):
                        repeat = server.call(f"repeat-base-{rows}", rows=rows)
                        compare(repeat, directory, oracles["base", rows], directory, plan["numeric"]["repeat_atol"])
                    raw = native_model(server.adapter, family)
                    controller = server.mutate(
                        lambda: PeftModel.from_pretrained(
                            raw,
                            paths["a"],
                            adapter_name="a",
                            is_trainable=False,
                            autocast_adapter_dtype=False,
                            local_files_only=True,
                        )
                    )
                    require(controller.get_base_model() is raw, "static PEFT root differs")
                    server.mutate(
                        lambda: controller.load_adapter(
                            paths["b"],
                            adapter_name="b",
                            is_trainable=False,
                            autocast_adapter_dtype=False,
                            local_files_only=True,
                        )
                    )
                    for alias in ("a", "b"):
                        server.mutate(lambda alias=alias: controller.set_adapter(alias))
                        server.static_selection = alias
                        for rows in (2, 4):
                            oracles[alias, rows] = copy.deepcopy(server.call(f"static-{alias}-{rows}", rows=rows))
                    require(
                        {n: tensor_hash(p) for n, p in server.state.parameters.items()} == server.state.hashes,
                        "static PEFT changed base weights",
                    )
                    server.mutate(controller.unload)
                    del controller, raw
                    server.static_selection = None
                receipt["stages"].append(role)
            require(server.records["cleanup"]["passed"], "ordinary cleanup failed")
            server = None
        oracle_dir = output / "current-ordinary"
        for rows in (2, 4):
            for left, right in (("base", "a"), ("base", "b"), ("a", "b")):
                a = oracles[left, rows]["observations"][0]["forwards"][0]["output"]
                b = oracles[right, rows]["observations"][0]["forwards"][0]["output"]
                av, bv = np.load(oracle_dir / a["file"]), np.load(oracle_dir / b["file"])
                effect = float(np.max(np.abs(av.astype(np.float64) - bv.astype(np.float64))))
                require(
                    effect
                    > max(
                        plan["numeric"]["effect_floor"],
                        plan["numeric"]["effect_drift_multiplier"] * plan["numeric"]["repeat_atol"],
                    ),
                    "fixed adapter effect is below the prospective floor",
                )
                receipt["effects"].append({"rows": rows, "left": left, "right": right, "max_abs": effect})
        for capacity in (2, 1):
            directory = output / f"explicit-capacity{capacity}"
            with Server(
                plan, family, directory, role="explicit", adapter_path=spec["explicit"], paths=paths, capacity=capacity
            ) as server:
                manager = server.adapter._lora
                require(not manager._loaded, "explicit adapter preloaded residents")
                for rows in (2, 4):
                    compare(
                        server.call(f"initial-base-{rows}", rows=rows),
                        directory,
                        oracles["base", rows],
                        oracle_dir,
                        plan["numeric"]["atol"],
                    )
                initial = memory(server.device)
                receipt["memory"].append({"stage": f"capacity{capacity}-warm-base", **initial})
                for index, alias in enumerate(("a", "b") if capacity == 2 else ("a", "b", "a")):
                    server.call(f"cold-{index}-{alias}", alias, expected=503, code="LORA_LOADING")
                    server.ready()
                    require(
                        paths[alias] in manager._loaded and len(manager._loaded) <= capacity, "capacity/source differs"
                    )
                    for selected in (alias, None):
                        for rows in (2, 4):
                            compare(
                                server.call(f"warm-{index}-{selected}-{rows}", selected, rows),
                                directory,
                                oracles[selected or "base", rows],
                                oracle_dir,
                                plan["numeric"]["atol"],
                            )
                    if alias == "a":
                        compare(
                            server.call(f"alias-{index}", "a-alias"),
                            directory,
                            oracles["a", 4],
                            oracle_dir,
                            plan["numeric"]["atol"],
                        )
                    server.mutate(lambda server=server: server.adapter.set_active_lora(None))
                    server.state.check(server.adapter, family)
                    storage = resident_storage(server.adapter, family, device=server.device)
                    expected_bytes = sum(
                        fixture[a]["parameter_bytes"] for a in ("a", "b") if paths[a] in manager._loaded
                    )
                    require(storage["unique_storage_bytes"] == expected_bytes, "resident storage count differs")
                    receipt["memory"].append(
                        {"stage": f"capacity{capacity}-{index}-resident", **memory(server.device), **storage}
                    )
                if capacity == 2:
                    require(len(manager._loaded) == 2, "two residents required")
                    for index, order in enumerate(itertools.permutations(("a", "b", "base"))):
                        for row in wave(server, order, index):
                            compare(
                                row,
                                directory,
                                oracles[row["selection"] or "base", 4],
                                oracle_dir,
                                plan["numeric"]["atol"],
                            )
                    # Separate opt-in path, fixed descriptive sample only. It
                    # does not enter the ordinary six-pair performance gates.
                    server.mutate(lambda server=server: server.adapter.set_active_lora(None))
                    receipt["opt_in_resident_base"] = measure(
                        plan_path, family, source, output / "resident-timing", limit_s=90, existing_server=server
                    )
                server.mutate(lambda server=server: server.adapter.set_active_lora(None))
                parameter_map = manager._loaded if family == "gliner2" else manager._parameters
                refs = [weakref.ref(p) for values in parameter_map.values() for p in values]
                refs += [weakref.ref(layer) for layer in manager._resident_layers.values()]
                for alias in ("a", "b"):
                    if paths[alias] in manager._loaded:
                        server.mutate(lambda alias=alias, server=server: server.adapter.unload_lora(alias))
                server.registry._loaded[server.model_id].loras.clear()
                del parameter_map
                gc.collect()
                require(
                    not manager._loaded and not manager._resident_layers and not any(r() is not None for r in refs),
                    "adapter tensors/wrappers retained",
                )
                server.state.check(server.adapter, family, hashes=True, training=True)
                for rows in (2, 4):
                    compare(
                        server.call(f"final-base-{rows}", rows=rows),
                        directory,
                        oracles["base", rows],
                        oracle_dir,
                        plan["numeric"]["atol"],
                    )
                final = memory(server.device)
                receipt["memory"].append({"stage": f"capacity{capacity}-final-base", **final})
                if initial["allocated"] is not None:
                    require(
                        final["allocated"] - initial["allocated"]
                        <= plan["memory"]["restored_allocated_allowance_bytes"],
                        "final adapter removal allocation exceeds allowance",
                    )
                del manager
                receipt["stages"].append(f"explicit-capacity{capacity}")
            server = None
        final = memory(plan["environment"]["device"])
        receipt["memory"].append({"stage": "full-unload", **final})
        if final["allocated"] is not None:
            require(
                final["allocated"] - receipt["memory"][0]["allocated"]
                <= plan["memory"]["restored_allocated_allowance_bytes"],
                "whole-model allocation not restored",
            )
        receipt["passed"] = True
        return receipt
    finally:
        atomic(output / "summary.json", receipt)
