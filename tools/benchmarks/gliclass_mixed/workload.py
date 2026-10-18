"""Fixed public synthetic inputs for the GLiClass mixed-load investigation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

WINDOW_TOKENS = 1024
STATIC_BUDGET = 16384
ADAPTIVE_FLOOR = 4096
MIXED_SECONDS = 480
RUNNER_CEILING = 3900
MAX_CALLS_PER_LANE = 200000
EXPECTED_VERSIONS = {
    "torch": "2.9.1+cu129",
    "transformers": "4.57.6",
    "gliclass": "0.1.17",
    "gliformer": "0.1.2",
    "safetensors": "0.7.0",
    "huggingface-hub": "0.36.2",
    "httpx": "0.28.1",
    "uvicorn": "0.46.0",
}

MODEL = "knowledgator/gliclass-large-v1.0"
REVISION = "ae8a18644e84cfda53325bd91c7941f2f6fd0ccb"
PARAGRAPHS = (
    "The customer was charged twice for the monthly subscription and asks the billing team to check both payments, "
    "explain the renewal date, and refund the duplicate charge.",
    "The application cannot open a saved project after the latest update. The support team should inspect the error "
    "report and help restore access without losing any work.",
    "A workspace administrator needs to invite a new colleague, reset account access, and understand which permission "
    "allows the colleague to view shared reports.",
    "The buyer is comparing a team subscription with an individual plan and asks about available features, contract "
    "terms, onboarding support, and the renewal schedule.",
)
LABELS = ["billing", "technical support", "account access", "sales", "other"]
GROUPS = {
    "topic": LABELS,
    "urgency": ["low", "medium", "high"],
    "sentiment": ["positive", "neutral", "negative"],
    "customer": ["individual", "team", "enterprise"],
    "channel": ["email", "phone", "chat"],
    "stage": ["new", "investigating", "resolved"],
    "action": ["reply", "refund", "escalate"],
    "region": ["local", "international", "unspecified"],
}
SHORT = "Please check the duplicate payment and refund the extra charge for my subscription."


def templates() -> dict[str, dict[str, Any]]:
    """No request/result-dependent fixture selection; IDs never change the text."""
    documents = {
        band: [f"Ticket {index:02d}. " + " ".join([PARAGRAPHS[index % 4]] * repeats) for index in range(24)]
        for band, repeats in (("medium", 10), ("long", 20))
    }
    return {
        "short": {"texts": [SHORT], "labels": LABELS, "options": {}},
        "steady": {"texts": list(PARAGRAPHS), "labels": LABELS, "options": {}},
        **{
            f"{band}{groups}": {
                "texts": documents[band],
                "labels": None,
                "options": {"group_encoding": "separate", "label_groups": dict(list(GROUPS.items())[:groups])},
            }
            for band in ("medium", "long")
            for groups in (7, 8)
        },
    }


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def items(kind: str, request_id: str, fixtures: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    return [{"id": f"{request_id}/{index}", "text": text} for index, text in enumerate(fixtures[kind]["texts"])]


def mixed_kinds() -> tuple[str, ...]:
    return ("short",) * 12 + ("medium7", "medium8", "long7", "long8")
