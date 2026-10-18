"""The public Qwen reranker addendum, using the original study's requests and splits."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sie_sdk import SIEClient
from sie_sdk.types import Item

import study

REVISION = "bdf9f45218b292b60512fa404595b17abc1cacd1"
FOLDER = "evaluations/2026-10-13/request-routing-decision-routers-v1/results"
MANIFEST_SHA256 = "30c06cccd2b5361902c1c59e07d518ccece61c0c61e806b834b51d76d3315560"
MODEL = "Qwen/Qwen3-Reranker-4B"
ARMS = {"q1": "qwen3_rr4b_q1", "q3": "qwen3_rr4b_q3"}
FILES = (
    "results.json",
    "code/specs.json",
    "code/prices.json",
    "signals/qwen3_rr4b_q1.jsonl",
    "signals/qwen3_rr4b_q3.jsonl",
)


def manifest_entries(body: bytes) -> dict[str, tuple[str, int]]:
    return study.manifest_entries(body, digest=MANIFEST_SHA256, files=FILES)


def verify_evidence(directory: Path) -> None:
    study.verify_evidence(directory, digest=MANIFEST_SHA256, files=FILES)


def response_score(response: dict) -> float:
    """Accept exactly the one requested document, with a finite yes/no score."""
    scores = response.get("scores")
    if (
        not isinstance(scores, list)
        or len(scores) != 1
        or not isinstance(scores[0], dict)
        or scores[0].get("item_id") != "d"
    ):
        raise ValueError("reranker must return exactly one score for document d")
    value = scores[0].get("score")
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("reranker score must be finite and between zero and one")
    return float(value)


def score_request(client: SIEClient, text: str, spec: dict) -> float:
    """One native score call; query/instruction are copied from the pinned study."""
    result = client.score(
        MODEL,
        Item(text=spec["query"]["text"]),
        [Item(id="d", text=text)],
        instruction=spec["instruction"],
        options={"profile": "default"},
        wait_for_capacity=False,
        max_oom_retries=0,
        provision_timeout_s=60,
    )
    return response_score(result)


@dataclass(frozen=True)
class Recording:
    scores: np.ndarray
    usd: np.ndarray
    spec: dict[str, Any]
    pairs: dict[str, Any]


def load(directory: Path, ids: list[str], description: str = "q1") -> Recording:
    """Verify hashes and align each signal with the base study by id, never by row position."""
    verify_evidence(directory)
    arm = ARMS[description]
    rows = study.read_jsonl(directory / f"signals/{arm}.jsonl")
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids) or any(row["signal"] != arm for row in rows):
        raise study.EvidenceError(f"{arm}: duplicate, missing or unexpected request/signal")
    scores = np.array([response_score(by_id[i]["reply"]) for i in ids])
    tokens = [by_id[i]["usage"]["input_tokens_details"]["content_tokens"] for i in ids]
    if any(type(t) is not int or t < 0 for t in tokens):
        raise study.EvidenceError(f"{arm}: invalid content token usage")
    prices = json.loads((directory / "code/prices.json").read_text())
    price = prices[MODEL]
    if price["unit"] != "content_input_tokens":
        raise study.EvidenceError("reranker price must use content tokens")
    specs = json.loads((directory / "code/specs.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    return Recording(
        scores=scores,
        usd=np.asarray(tokens) * price["usd_per_token"],
        spec=specs[arm]["body"],
        pairs=results["arms"][arm]["pairs"],
    )
