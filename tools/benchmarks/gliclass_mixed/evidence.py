"""Small evidence helpers; every failed or partial phase remains inspectable."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("latency samples must be positive and finite")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must lie in (0, 1]")
    return sorted(values)[math.ceil(len(values) * quantile) - 1]


def error_chain(error: BaseException) -> list[dict[str, str]]:
    result = []
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        result.append({"type": type(error).__name__, "message": str(error)})
        cause = error.__cause__ or error.__context__
        if cause is None:
            break
        error = cause
    return result


class Journal:
    def __init__(self, path: Path) -> None:
        self.stream = path.open("w")
        self.lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        record = {"event": event, "monotonic_ns": time.monotonic_ns(), "time_ns": time.time_ns(), **fields}
        with self.lock:
            self.stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            self.stream.flush()

    def close(self) -> None:
        with self.lock:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


@contextmanager
def parent_guard() -> Iterator[None]:
    """Terminate only this child if its explicitly recorded parent disappears."""
    parent = int(os.environ.get("SIE_BENCH_PARENT_PID", os.getppid()))
    stopped = threading.Event()

    def watch() -> None:
        while not stopped.wait(0.5):
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stopped.set()
        watcher.join(timeout=1)


def process_identity(pid: int) -> dict:
    """Linux process birth identity; local Mac smoke explicitly has no procfs."""
    path = Path(f"/proc/{pid}/stat")
    return {"pid": pid, "start_ticks": int(path.read_text().rsplit(")", 1)[1].split()[19]) if path.exists() else None}
