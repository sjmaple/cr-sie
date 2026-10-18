"""Route a request to a cheap or a strong LLM with embedding kNN or a Qwen reranker.

    export SIE_API_KEY=...
    uv run --frozen python run.py "Solve for x: x/2 + x/3 = 5"
    uv run --frozen python run.py --pair anthropic "Write a limerick about a lighthouse keeper"
    uv run --frozen python run.py --item math500/test/prealgebra/505.json
    uv run --frozen python run.py --item math500/test/prealgebra/505.json --recorded
    uv run --frozen python run.py --router reranker --item math500/test/prealgebra/505.json --recorded
    OPENAI_API_KEY=... uv run --frozen python run.py --answer "Solve for x: x/2 + x/3 = 5"

One call to SIE Serverless embeds the request with BAAI/bge-m3; the study's
calls were billed 1.3 credits on average, at USD 0.00001 a credit. The request
is compared with the study's 729 calibration requests and their labels for the
chosen pair, and goes to the cheap model if its score reaches the pair's
threshold from results.json. Run fetch.py first.

--router reranker uses one Qwen/Qwen3-Reranker-4B score call against the pinned
description instead of an embedding bank; run fetch.py --reranker first.
--item routes a study request by id; --recorded uses its recorded vector/score
without calling SIE. Live routing permits one send with a 60-second timeout.
--answer separately sends the request to the chosen OpenAI model with the
study's settings, needs OPENAI_API_KEY, and is billed outside SIE credits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np
from sie_sdk import SIEClient, SIEError
from sie_sdk.types import Item

import learn
import reranker
import study

OPENAI_RESPONSES = "https://api.openai.com/v1/responses"


class Encoder(Protocol):
    def encode(self, model: str, items: Item, **kwargs: Any) -> Any: ...


class RetryDisabled(RuntimeError):
    """The workshop permits one inference send per request."""


class SingleSendClient(SIEClient):
    def _record_retry(self) -> None:
        raise RetryDisabled


def embed(client: Encoder, text: str) -> np.ndarray:
    return np.asarray(
        client.encode(
            study.EMBEDDER, Item(text=text), wait_for_capacity=False, max_oom_retries=0, provision_timeout_s=60
        )["dense"],
        dtype=np.float64,
    )


def threshold(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("threshold must be a finite number between zero and one")
    return number


def answer_openai(model: str, text: str, api_key: str) -> str:
    """One OpenAI Responses call with the study's generation settings (code/arms.py `gen_body`)."""
    body = {
        "model": model,
        "input": [{"role": "user", "content": text}],
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 32000,
        "store": False,
    }
    response = httpx.post(OPENAI_RESPONSES, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=900)
    response.raise_for_status()
    return "".join(
        part.get("text", "")
        for item in response.json().get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", nargs="?", help="the request text to route")
    parser.add_argument("--pair", choices=sorted(study.PAIRS), default="openai")
    parser.add_argument("--router", choices=("knn", "reranker"), default="knn")
    parser.add_argument("--description", choices=sorted(reranker.ARMS), default="q1")
    parser.add_argument("--policy", type=Path, help="use learn.py's embedding policy or optimize.py's reranker policy")
    parser.add_argument("--threshold", type=threshold, help="override the study cutoff with your calibrated cutoff")
    parser.add_argument("--item", help="route a study request by id instead of REQUEST")
    parser.add_argument(
        "--recorded", action="store_true", help="with --item: use its recorded vector/score, no SIE call"
    )
    parser.add_argument("--answer", action="store_true", help="send the request to the chosen OpenAI model")
    parser.add_argument("--evidence", type=Path, default=study.EVIDENCE, help="directory written by fetch.py")
    args = parser.parse_args()
    if (args.request is None) == (args.item is None):
        parser.error("give either REQUEST or --item")
    if args.recorded and args.item is None:
        parser.error("--recorded needs --item")
    if args.router != "reranker" and args.description != "q1":
        parser.error("--description applies to --router reranker")
    policy = None
    linear_policy = False
    if args.policy:
        try:
            policy = json.loads(args.policy.read_text())
            if not isinstance(policy, dict):
                raise TypeError("policy must be a JSON object")
            linear_policy = policy.get("type") == learn.POLICY_TYPE
            if linear_policy:
                if args.router != "knn":
                    parser.error("a learned embedding policy needs --router knn (the default)")
                learn.validate_policy(policy, args.pair)
            elif args.router != "reranker" or args.recorded or args.description != "q1":
                parser.error("a reranker policy needs --router reranker and fresh scores")
            elif (
                policy["pair"] != args.pair
                or policy["model"] != reranker.MODEL
                or policy["profile"] != "default"
                or any(not isinstance(policy[k], str) or not policy[k].strip() for k in ("description", "instruction"))
            ):
                raise ValueError("policy pair/model/profile/text mismatch")
            if not linear_policy and policy["tau"] is not None:
                threshold(str(policy["tau"]))
        except (OSError, ValueError, TypeError, KeyError, argparse.ArgumentTypeError) as error:
            parser.error(f"invalid policy: {type(error).__name__}")
    if not args.recorded and not any(os.environ.get(name, "").strip() for name in ("SIE_API_KEY", "SIE_BASE_URL")):
        parser.error(
            "set SIE_API_KEY for Serverless or SIE_BASE_URL for a local server; use --item ID --recorded offline"
        )
    if args.answer and args.pair != "openai":
        parser.error("--answer calls OpenAI models only; use --pair openai")
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if args.answer and not openai_key:
        parser.error("--answer needs OPENAI_API_KEY")

    try:
        s = study.load_study(args.evidence)
    except (OSError, ValueError) as error:
        print(f"Evidence check failed: {error}", file=sys.stderr)
        return 1
    bank = study.bank(s, args.pair)
    recording = None
    if args.router == "reranker":
        try:
            recording = reranker.load(args.evidence / "reranker", s.ids, args.description)
        except (OSError, ValueError, KeyError) as error:
            print(f"Reranker evidence check failed: {error}; run fetch.py --reranker", file=sys.stderr)
            return 1

    index = None
    text = args.request
    if args.item is not None:
        try:
            index = s.index(args.item)
        except KeyError as error:
            print(error.args[0], file=sys.stderr)
            return 1
        text = s.prompts[index]

    try:
        if args.recorded:
            value = (
                learn.policy_score(policy, s.raw_vectors[index])
                if linear_policy
                else recording.scores[index]
                if recording
                else bank.score(s.raw_vectors[index], exclude=args.item)
            )
        else:
            with SingleSendClient(timeout_s=60, max_connections=1) as client:
                spec = (
                    {"query": {"text": policy["description"]}, "instruction": policy["instruction"]}
                    if policy and not linear_policy
                    else recording.spec
                    if recording
                    else None
                )
                value = (
                    reranker.score_request(client, text, spec)
                    if recording
                    else learn.policy_score(policy, embed(client, text))
                    if linear_policy
                    else bank.score(embed(client, text), exclude=args.item)
                )
    except (SIEError, ValueError, RetryDisabled) as error:
        print(
            f"Routing call failed: {type(error).__name__}. Check your endpoint, API key and model availability.",
            file=sys.stderr,
        )
        return 1
    tau = policy["tau"] if policy else recording.pairs[args.pair]["tau"] if recording else bank.tau
    if args.threshold is not None:
        tau = args.threshold
    cheap = study.routes_cheap(float(value), tau)
    names = study.PAIRS[args.pair]
    decision = {
        "score": float(value),
        "tau": tau,
        "route": "cheap" if cheap else "strong",
        "model": names.cheap if cheap else names.strong,
    }
    chosen = names.cheap_name if decision["route"] == "cheap" else names.strong_name
    relation = ">=" if decision["route"] == "cheap" else "<"
    print(f"pair   {args.pair}: {names.cheap_name} (cheap) or {names.strong_name} (strong)")
    print(
        f"router {'learned embedding' if linear_policy else args.router}"
        + (f" policy {args.policy}" if policy else f" {args.description}" if recording else "")
    )
    shown_tau = "never cheap" if tau is None else f"{tau:.6f}"
    print(f"score  {decision['score']:.6f} {relation} tau {shown_tau}")
    print(f"route  {decision['model']} ({chosen})")
    if index is not None:
        recorded = s.pairs[args.pair]
        print(f"study  {s.ids[index]} ({'calibration' if s.calibration[index] else 'held-out'})")
        old_score = float(recording.scores[index]) if recording else recorded.score[index]
        old_tau = recording.pairs[args.pair]["tau"] if recording else bank.tau
        old_route = "cheap" if study.routes_cheap(old_score, old_tau) else "strong"
        print(f"       original study reference: score {old_score:.6f}, routed {old_route}")
    if not args.recorded:
        print("One live routing call. Use graded traffic to validate a policy for your models and deployment.")

    if args.answer:
        try:
            reply = answer_openai(decision["model"], text, openai_key)
        except (httpx.HTTPError, ValueError) as error:
            print(f"The OpenAI call failed: {type(error).__name__}", file=sys.stderr)
            return 1
        print(f"\n{reply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
