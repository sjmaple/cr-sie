from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from peft import LoraConfig
from peft.tuners.lora.layer import Linear

from tools.benchmarks.merged_runtime_lora.common import Server, close_enough, require
from tools.benchmarks.merged_runtime_lora.lifecycle import fixtures, resident_storage, run, wave
from tools.benchmarks.merged_runtime_lora.timing import run as timing

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def plan(tmp_path: Any, tmp_path_factory: Any, monkeypatch: Any, request: Any) -> Any:
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SIE_TELEMETRY_DISABLED": "1",
        "SIE_DISK_CACHE_ENABLED": "false",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        monkeypatch.setenv(key, value)
    torch.set_num_threads(1)
    path = ROOT / "packages/sie_server/tests/adapters/test_cross_encoder_lora.py"
    spec = importlib.util.spec_from_file_location("proof_test_fixture_source", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    family = getattr(request, "param", "bert")
    checkpoint = module.checkpoint.__wrapped__(tmp_path_factory, SimpleNamespace(param=family))
    return {
        "environment": {"device": "cpu"},
        "models": {
            family: {
                "repo": "generated-local",
                "revision": None,
                "directory": str(checkpoint),
                "ordinary": "sie_server.adapters.cross_encoder:CrossEncoderAdapter",
                "explicit": "sie_server.adapters.cross_encoder.runtime_lora:CrossEncoderLoraAdapter",
            }
        },
        "family": family,
        "fixture_ce": {"seeds": [20261016, 20261017]},
        "query": "how do i reset my password",
        "documents": [
            "reset your password by email",
            "the capital is paris",
            "email my password",
            "how is the capital",
        ],
        "sequence_limits": {"ce": 64},
        "numeric": {"atol": 1e-4},
        "opt_in_latency": {"warmup_calls": 2, "measured_calls": 4},
    }


@pytest.mark.parametrize("plan", ["bert", "xlmr"], indirect=True)
def test_actual_cold_server_static_inputs_and_overlapping_selections(plan: Any, tmp_path: Any) -> Any:
    family = plan["family"]
    paths, _ = fixtures(plan, family, tmp_path / "fixtures")
    with Server(
        plan, family, tmp_path / "server", role="cpu-test", adapter_path=plan["models"][family]["explicit"], paths=paths
    ) as server:
        base = server.call("base")
        require(base["observations"][0]["forwards"], "actual bound forward missing")
        for alias in ("a", "b"):
            server.call(f"cold-{alias}", alias, expected=503, code="LORA_LOADING")
            server.ready()
            server.call(f"warm-{alias}", alias)
        rows = wave(server, ("a", "b", "base"), 0)
        assert len(rows) == 3
        for row in rows:
            assert row["observations"]
            assert row["wire"][0]["status"] == 200
        server.mutate(lambda: server.adapter.set_active_lora(None))
        server.state.check(server.adapter, family)
        detached = resident_storage(server.adapter, family, device="cpu")
        server.mutate(lambda: server.adapter.set_active_lora("a"))
        assert resident_storage(server.adapter, family, device="cpu") == detached
        server.mutate(lambda: server.adapter.set_active_lora(None))
    assert server.records["cleanup"]["passed"]
    files = list((tmp_path / "server").glob("request-*.json"))
    assert len(files) == len(server.requests)
    assert {json.loads(p.read_text())["stage"] for p in files} == {r["stage"] for r in server.requests}
    assert not list((tmp_path / "server").glob("*.part"))


@pytest.mark.parametrize("family", ["bert", "gliner2"])
@pytest.mark.parametrize("damage", ["missing", "orphan", "device", "dtype"])
def test_resident_storage_rejects_incomplete_or_wrong_tensors(family: Any, damage: Any) -> Any:
    layer = Linear(torch.nn.Linear(8, 8), adapter_name="test", config=LoraConfig(r=4, lora_alpha=8), r=4, lora_alpha=8)
    values: list[Any] = [layer.lora_A["test"].weight, layer.lora_B["test"].weight]
    for value in values:
        value.requires_grad_(False)
    root = torch.nn.Module()
    root.projection = layer.get_base_layer()
    manager = SimpleNamespace(
        _resident_layers={"projection": layer},
        _loaded={"source": values if family == "gliner2" else 256},
        _parameters={"source": values},
    )
    adapter = SimpleNamespace(_lora=manager, _model=root if family == "gliner2" else SimpleNamespace(model=root))
    assert resident_storage(adapter, family, device="cpu")["parameter_tensors"] == 2
    if damage == "missing":
        values.pop()
    elif damage == "orphan":
        layer.lora_A["orphan"] = torch.nn.Linear(8, 4, bias=False)
    elif damage == "dtype":
        values[0].data = values[0].data.double()
    with pytest.raises(AssertionError, match="resident"):
        resident_storage(adapter, family, device="cuda:0" if damage == "device" else "cpu")


@pytest.mark.parametrize(
    ("got", "want"), [(float("nan"), 1.0), (1.1, 1.0), ({"a": 1}, {"b": 1}), ([1], [1, 2]), (1, 1.0)]
)
def test_full_result_mismatch_rejected(got: Any, want: Any) -> Any:
    with pytest.raises(AssertionError):
        close_enough(got, want, 1e-4)


def test_float_only_tolerance() -> Any:
    close_enough({"id": "a", "score": 1.00001, "tokens": 12}, {"id": "a", "score": 1.0, "tokens": 12}, 1e-4)
    with pytest.raises(AssertionError):
        close_enough({"tokens": 12}, {"tokens": 13}, 1e-4)


@pytest.mark.parametrize("plan", ["bert", "xlmr"], indirect=True)
def test_full_static_reference_eviction_restoration(plan: Any, tmp_path: Any) -> Any:
    source = "packages/sie_server/src/sie_server/adapters/cross_encoder/__init__.py"
    blob = subprocess.check_output(["git", "show", f"0521464e:{source}"], cwd=ROOT)  # noqa: S603,S607 -- fixed local git blob
    baseline = tmp_path / "baseline.py"
    baseline.write_bytes(blob)
    plan.update(
        ordinary_baseline={"ce": {"sha256": hashlib.sha256(blob).hexdigest()}},
        baseline_modules={"ce": str(baseline)},
        source_commit="tiny-control",
        memory={"restored_allocated_allowance_bytes": 8388608},
    )
    plan["numeric"].update(repeat_atol=1e-6, effect_floor=1e-5, effect_drift_multiplier=5)
    plan["fixture_ce"]["scale"] = 0.4
    receipt = run(plan, plan["family"], tmp_path / "full")
    assert receipt["passed"]
    assert receipt["performance_qualification"] is False


def test_actual_sdk_only_timing_child(plan: Any, tmp_path: Any) -> Any:
    plan["latency"] = {"warmup_calls": 2, "measured_calls": 4}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    receipt = timing(path, plan["family"], ROOT, tmp_path / "timing", limit_s=40)
    assert receipt["passed"]
    assert receipt["metrics"]["count"] == 4
    assert receipt["cleanup"]["reaped"]


def gliner_plan(tmp_path: Any, monkeypatch: Any) -> Any:
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SIE_TELEMETRY_DISABLED": "1",
        "SIE_DISK_CACHE_ENABLED": "false",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        monkeypatch.setenv(key, value)
    torch.set_num_threads(1)
    path = ROOT / "packages/sie_server/tests/adapters/test_gliner2_lora.py"
    spec = importlib.util.spec_from_file_location("gliner_proof_test_fixture", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.raw_model.__wrapped__(SimpleNamespace(param="deberta"), tmp_path)
    base = tmp_path / "base"
    model.save_pretrained(str(base))
    adapter = module.make_checkpoint(tmp_path, model, "a")
    del model
    relative = "packages/sie_server/src/sie_server/adapters/gliner2/adapter.py"
    blob = subprocess.check_output(["git", "show", f"0521464e:{relative}"], cwd=ROOT)  # noqa: S603,S607 -- fixed local git blob
    archived = tmp_path / "archived.py"
    archived.write_bytes(blob)
    return {
        "environment": {"device": "cpu"},
        "source_commit": "tiny-gliner-control",
        "models": {
            "gliner2": {
                "directory": str(base),
                "repo": "generated-local",
                "revision": None,
                "ordinary": "sie_server.adapters.gliner2.adapter:GLiNER2Adapter",
                "explicit": "sie_server.adapters.gliner2.entity_lora:GLiNER2EntityLoraAdapter",
            }
        },
        "ordinary_baseline": {"gliner2": {"sha256": hashlib.sha256(blob).hexdigest()}},
        "baseline_modules": {"gliner2": str(archived)},
        "gliner2_adapter_A": {"directory": adapter},
        "entity_texts": [
            "Alice met Bob in London.",
            "Bob works with Alice in Paris.",
            "Bob met Alice in London.",
            "Alice works with Bob in Paris.",
        ],
        "entity_labels": ["person", "city"],
        "entity_threshold": 0.0,
        "sequence_limits": {"gliner2": 64},
        "numeric": {"atol": 1e-4, "repeat_atol": 1e-6, "effect_floor": 1e-5, "effect_drift_multiplier": 5},
        "memory": {"restored_allocated_allowance_bytes": 8388608},
        "latency": {"warmup_calls": 2, "measured_calls": 4},
        "opt_in_latency": {"warmup_calls": 2, "measured_calls": 4},
        "query": "unused",
        "documents": ["unused"] * 4,
    }


def test_actual_gliner_static_lifecycle_and_timing(tmp_path: Any, monkeypatch: Any) -> Any:
    plan = gliner_plan(tmp_path, monkeypatch)
    receipt = run(plan, "gliner2", tmp_path / "full")
    assert receipt["passed"]
    assert receipt["performance_qualification"] is False
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    measured = timing(path, "gliner2", ROOT, tmp_path / "timing", limit_s=40)
    assert measured["passed"]
    assert measured["metrics"]["count"] == 4
    assert measured["cleanup"]["reaped"]
