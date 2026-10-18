"""Bounded SIE-only inference with a persistent, credential-free request ledger."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

from sie_sdk.types import Item

import reranker
from run import SingleSendClient


class BudgetExceeded(RuntimeError):
    """A local call, token, time or observed-credit limit has been reached."""


class CloudFailure(RuntimeError):
    """A content-free inference failure; the request ledger retains the attempted call."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def usage_of(reply: dict) -> dict:
    usage = reply.get("usage") or {}
    request = reply.get("request") or {}
    request_usage = request.get("usage") or {}
    credits = request.get("credits_debited", request_usage.get("credits_charged", usage.get("credits_charged")))
    if type(credits) not in (int, float) or not math.isfinite(credits) or credits < 0:
        raise CloudFailure("response has no usable billed-credit receipt")
    return {
        "credits": float(credits),
        "rate_book_version": request_usage.get("rate_book_version", usage.get("rate_book_version")),
        **{
            k: v
            for k, v in usage.items()
            if k in ("input_tokens", "prompt_tokens", "completion_tokens", "total_tokens")
        },
    }


class Cloud:
    def __init__(
        self,
        client: SingleSendClient,
        directory: Path,
        *,
        max_score_items=5000,
        max_reflections=6,
        max_credits=20000,
        max_seconds=1800,
        batch_size=16,
        reflection_model="Qwen/Qwen3.8-27B-FP8",
        reflection_client=None,
        self_hosted_usd_per_token=None,
    ):
        self.client, self.directory = client, directory
        self.max_score_items, self.max_reflections = max_score_items, max_reflections
        self.max_credits, self.max_seconds, self.batch_size = max_credits, max_seconds, batch_size
        self.reflection_model = reflection_model
        self.reflection_client = reflection_client or client
        self.self_hosted_usd_per_token = self_hosted_usd_per_token
        if self_hosted_usd_per_token is not None and (
            not math.isfinite(self_hosted_usd_per_token) or self_hosted_usd_per_token <= 0
        ):
            raise ValueError("self-hosted routing estimate requires a positive, finite content-token price")
        self.started = time.monotonic()
        self.score_items = self.reflections = self.requests = 0
        self.credits = 0.0
        self.failed = False
        self.unknown_calls = 0
        self.cache: dict[tuple[str, str], tuple[float, float]] = {}
        self.ledger = (directory / "calls.jsonl").open("x")
        self.scores_file = (directory / "scores.jsonl").open("x")

    def close(self):
        self.ledger.close()
        self.scores_file.close()

    def check(self):
        if self.failed:
            raise CloudFailure("an earlier call failed; start a new run explicitly after inspecting its ledger")
        if self.credits >= self.max_credits or time.monotonic() - self.started >= self.max_seconds:
            raise BudgetExceeded("observed credit or wall-clock budget reached")

    def invalid_reply(self, message):
        self.failed = True
        raise CloudFailure(message)

    def summary(self):
        return {
            "requests": self.requests,
            "score_items": self.score_items,
            "reflections": self.reflections,
            "observed_credits": self.credits if self.self_hosted_usd_per_token is None else None,
            "calls_with_unknown_billing": self.unknown_calls if self.self_hosted_usd_per_token is None else None,
            "elapsed_s": round(time.monotonic() - self.started, 3),
        }

    def receipt(self, reply, kind):
        if self.self_hosted_usd_per_token is None:
            return usage_of(reply)
        usage = reply.get("usage") or {}
        if kind == "score":
            tokens = (usage.get("input_tokens_details") or {}).get("content_tokens")
            if type(tokens) is not int or tokens < 0:
                raise CloudFailure("self-hosted score response has no content-token usage")
            return {"credits": None, "billing": "self_hosted", "content_tokens": tokens}
        tokens = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage}
        if not tokens or any(type(v) is not int or v < 0 for v in tokens.values()):
            raise CloudFailure("self-hosted reflection response has no usable token usage")
        return {"credits": None, "billing": "self_hosted", **tokens}

    def call(self, kind: str, fn, metadata: dict) -> dict:
        self.check()
        self.requests += 1
        row = {"request_number": self.requests, "kind": kind, **metadata}
        started = time.monotonic()
        try:
            reply = fn()
            receipt = self.receipt(reply, kind)
            self.credits += receipt["credits"] or 0
            row.update(receipt)
            row["status"] = "success"
        except Exception as error:  # noqa: BLE001 -- log every failed send, with no response or secret text
            self.failed = True
            self.unknown_calls += 1
            row.update(status="failed_or_unknown", error_type=type(error).__name__)
            status, code = getattr(error, "status_code", None), getattr(error, "code", None)
            if type(status) is int:
                row["http_status"] = status
            if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", code):
                row["error_code"] = code
            raise CloudFailure(f"{kind} call failed: {type(error).__name__}; see calls.jsonl") from None
        finally:
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            self.ledger.write(json.dumps(row, allow_nan=False) + "\n")
            self.ledger.flush()
        return reply

    def score(self, candidate: dict[str, str], items: list[dict]) -> tuple[list[float], list[float]]:
        if set(candidate) != {"description", "instruction"} or any(
            not isinstance(v, str) or not v.strip() or len(v) > 3000 for v in candidate.values()
        ):
            raise ValueError("candidate must contain nonempty description/instruction, each at most 3000 characters")
        candidate_id = digest(candidate)
        missing = {item["id"]: item for item in items if (candidate_id, item["id"]) not in self.cache}
        pending = list(missing.values())
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            if self.score_items + len(batch) > self.max_score_items:
                raise BudgetExceeded("reranker item budget reached")
            self.score_items += len(batch)
            reply = self.call(
                "score",
                lambda batch=batch: self.client.score(
                    reranker.MODEL,
                    Item(text=candidate["description"]),
                    [Item(id=item["id"], text=item["request"]) for item in batch],
                    instruction=candidate["instruction"],
                    options={"profile": "default"},
                    wait_for_capacity=False,
                    max_oom_retries=0,
                    provision_timeout_s=60,
                ),
                {"candidate": candidate_id, "ids": [item["id"] for item in batch], "model": reranker.MODEL},
            )
            values = reply.get("scores") or []
            if not isinstance(values, list) or any(
                not isinstance(row, dict) or not isinstance(row.get("item_id"), str) or "score" not in row
                for row in values
            ):
                self.invalid_reply("reranker returned malformed scores")
            aligned = {row["item_id"]: row["score"] for row in values}
            if (
                len(values) != len(batch)
                or set(aligned) != {item["id"] for item in batch}
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in aligned.values())
            ):
                self.invalid_reply("reranker returned invalid or misaligned scores")
            # Batch receipt divided evenly among documents; not an individual-call tariff.
            receipt = self.receipt(reply, "score")
            usd = (
                receipt["credits"] * 0.00001
                if self.self_hosted_usd_per_token is None
                else receipt["content_tokens"] * self.self_hosted_usd_per_token
            ) / len(batch)
            for item in batch:
                value = float(aligned[item["id"]])
                self.cache[(candidate_id, item["id"])] = (value, usd)
                self.scores_file.write(
                    json.dumps(
                        {
                            "candidate": candidate_id,
                            "id": item["id"],
                            "score": value,
                            "router_usd": usd,
                        }
                    )
                    + "\n"
                )
            self.scores_file.flush()
            meter = (
                f"{self.credits:g} credits observed"
                if self.self_hosted_usd_per_token is None
                else "self-hosted usage; routing cost estimated"
            )
            print(f"Scored {self.score_items} request/prompt pairs; {meter}.", flush=True)
        values = [self.cache[(candidate_id, item["id"])] for item in items]
        return [v[0] for v in values], [v[1] for v in values]

    def reflect(self, prompt: str | list[dict]) -> str:
        self.check()  # GEPA can retry its callback; a failed send must never cause a second send.
        if self.reflections >= self.max_reflections:
            raise BudgetExceeded("reflection-call budget reached")
        self.reflections += 1
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        if len(json.dumps(messages)) > 60000:
            raise BudgetExceeded("reflection prompt exceeds the input-size limit")
        reply = self.call(
            "reflection",
            lambda: self.reflection_client.chat_completions(
                model=self.reflection_model,
                messages=messages,
                max_tokens=1200,
                temperature=0.7,
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=120,
                read_timeout_s=120,
            ),
            {"model": self.reflection_model, "input_sha256": digest(messages), "max_tokens": 1200},
        )
        try:
            text = reply["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            self.invalid_reply("reflection model returned a malformed answer")
        if not isinstance(text, str) or not text.strip():
            self.invalid_reply("reflection model returned no text")
        with (self.directory / f"reflection-{self.reflections:02d}.json").open("x") as handle:
            json.dump({"messages": messages, "answer": text}, handle, indent=2)
        return text
