"""Small fail-closed source and terminal joins for the fixed proof."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tools.benchmarks.cross_encoder_lora.evidence import digest


def require(value: Any, message: Any) -> Any:
    if not value:
        raise ValueError(message)


def verify_tree(root: Any, expected: Any) -> Any:
    root = Path(root).resolve()
    actual = {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}
    require(actual.keys() == expected.keys(), "source inventory differs")
    for name, path in actual.items():
        require(not path.is_symlink() and path.resolve().is_relative_to(root), "source path escapes archive")
        require(digest(path) == expected[name], f"source bytes differ: {name}")


def verify_import(path: Any, source: Any, files: Any) -> Any:
    path, source = Path(path).resolve(), Path(source).resolve()
    require(path.is_relative_to(source / "packages"), "module outside selected source archive")
    relative = str(path.relative_to(source))
    require(relative in files and digest(path) == files[relative], "imported module hash differs")
    return {"path": str(path), "relative_path": relative, "sha256": files[relative]}


def terminal_result(output: Any, expected: Any, *, pid: Any, ppid: Any, returncode: Any) -> Any:
    """Hash-reopen raw result only after binding the admitted child identity."""
    output = Path(output)
    terminal = json.loads((output / "terminal.json").read_text())
    for key, value in expected.items():
        require(terminal.get(key) == value, f"terminal {key} differs from admitted job")
    require(
        (terminal.get("pid"), terminal.get("ppid"), terminal.get("pgid")) == (pid, ppid, pid),
        "terminal process ownership differs",
    )
    require(
        terminal.get("finished_monotonic", 0) >= terminal.get("started_monotonic", math.inf), "terminal clock differs"
    )
    ownership = json.loads((output / "ownership.json").read_text())
    require(all(terminal.get(k) == v for k, v in ownership.items() if k != "passed"), "terminal ownership changed")
    require(type(terminal.get("passed")) is bool, "terminal verdict absent")
    if not terminal["passed"]:
        require(returncode != 0 and isinstance(terminal.get("error"), dict), "failure receipt/exit differs")
        return terminal
    require(returncode == 0 and "error" not in terminal, "successful terminal/exit contradict")
    result_path = output / "proof/summary.json"
    require(digest(result_path) == terminal["result_sha256"], "result hash differs")
    result = json.loads(result_path.read_text())
    require(
        result.get("passed") is True and result.get("family") == expected["family"], "result identity/verdict differs"
    )
    if expected["kind"] == "timing":
        require(result.get("source") == expected["source"], "timing source differs")
        require(result.get("metrics") == terminal.get("metrics"), "terminal metrics differ from raw result")
        client = output / "proof/client/performance.json"
        require(digest(client) == result["result_sha256"], "raw performance file hash differs")
        raw = json.loads(client.read_text())
        require(all(raw.get(k) == v for k, v in terminal["metrics"].items()), "raw performance metrics differ")
    else:
        require(result.get("source_commit") == expected["revision"], "lifecycle source differs")
    return terminal
