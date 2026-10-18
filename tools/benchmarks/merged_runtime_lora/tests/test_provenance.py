"""Reject valid-looking but wrongly joined child results and source trees."""

import json
from typing import Any

import pytest

from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest
from tools.benchmarks.merged_runtime_lora.provenance import terminal_result, verify_import, verify_tree
from tools.benchmarks.merged_runtime_lora.runner import gate


@pytest.fixture
def joined(tmp_path: Any) -> Any:
    expected = {
        "token": "unique",
        "source": "/exact/source",
        "revision": "a" * 40,
        "family": "bert",
        "kind": "timing",
        "limit_s": 150,
        "plan_sha256": "b" * 64,
        "inputs_manifest_sha256": "c" * 64,
    }
    ownership = {**expected, "pid": 100, "ppid": 99, "pgid": 100, "passed": False, "started_monotonic": 1.0}
    atomic_json(tmp_path / "ownership.json", ownership)
    metrics = {"count": 200, "mean_ms": 12.0, "p95_ms": 13.0, "elapsed_s": 2.4, "rps": 200 / 2.4}
    raw = tmp_path / "proof/client/performance.json"
    raw.parent.mkdir(parents=True)
    atomic_json(raw, metrics)
    result = {
        "passed": True,
        "family": "bert",
        "source": "/exact/source",
        "metrics": metrics,
        "result_sha256": digest(raw),
    }
    atomic_json(tmp_path / "proof/summary.json", result)
    terminal = {
        **ownership,
        "passed": True,
        "finished_monotonic": 4.0,
        "metrics": metrics,
        "result_sha256": digest(tmp_path / "proof/summary.json"),
    }
    atomic_json(tmp_path / "terminal.json", terminal)
    return tmp_path, expected


def test_terminal_exact_join(joined: Any) -> Any:
    output, expected = joined
    assert terminal_result(output, expected, pid=100, ppid=99, returncode=0)["passed"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("token", "other"),
        ("pid", 101),
        ("pgid", 101),
        ("family", "xlmr"),
        ("kind", "lifecycle"),
        ("revision", "d" * 40),
        ("source", "/wrong/source"),
        ("plan_sha256", "e" * 64),
        ("inputs_manifest_sha256", "f" * 64),
    ],
)
def test_terminal_wrong_binding_rejected(joined: Any, key: Any, value: Any) -> Any:
    output, expected = joined
    terminal = json.loads((output / "terminal.json").read_text())
    terminal[key] = value
    atomic_json(output / "terminal.json", terminal)
    with pytest.raises(ValueError, match=r"differs|ownership"):
        terminal_result(output, expected, pid=100, ppid=99, returncode=0)


def test_rehashed_wrong_metrics_rejected(joined: Any) -> Any:
    output, expected = joined
    result = json.loads((output / "proof/summary.json").read_text())
    result["metrics"]["mean_ms"] = 1.0
    atomic_json(output / "proof/summary.json", result)
    terminal = json.loads((output / "terminal.json").read_text())
    terminal.update(result_sha256=digest(output / "proof/summary.json"), metrics=result["metrics"])
    atomic_json(output / "terminal.json", terminal)
    with pytest.raises(ValueError, match="raw performance metrics"):
        terminal_result(output, expected, pid=100, ppid=99, returncode=0)


def test_source_complete_inventory_and_import_binding(tmp_path: Any) -> Any:
    path = tmp_path / "packages/server.py"
    path.parent.mkdir()
    path.write_text("VALUE = 1\n")
    files = {"packages/server.py": digest(path)}
    verify_tree(tmp_path, files)
    assert verify_import(path, tmp_path, files)["sha256"] == files["packages/server.py"]
    path.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="bytes differ"):
        verify_tree(tmp_path, files)
    with pytest.raises(ValueError, match="hash differs"):
        verify_import(path, tmp_path, files)
    path.unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify_tree(tmp_path, files)


def test_gate_neutral_and_planted_slowdown() -> Any:
    limits = {
        "t95_df5": 2.0150483733330233,
        "geometric_ratio_max": 1.01,
        "one_sided_95_upper_max": 1.03,
        "worst_pair_max": 1.10,
    }
    assert gate([1.0] * 6, limits)["passed"]
    assert not gate([1.05] * 6, limits)["passed"]
    assert not gate([0.98] * 5 + [1.11], limits)["passed"]
    for invalid in ([1.0] * 5, [float("nan")] * 6, [0] * 6):
        with pytest.raises(ValueError, match="six finite"):
            gate(invalid, limits)
