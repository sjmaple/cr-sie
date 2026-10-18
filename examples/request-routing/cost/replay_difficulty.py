"""Verify GEPA model-routing selection, controls, test results and receipts offline."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import difficulty
import difficulty_evidence
import learning
import study
from cloud import digest
from replay_learning import verify


class Recording:
    def __init__(self, source, directory):
        self.directory = directory
        scores = study.read_jsonl(source / "scores.jsonl")
        self.cache = {(r["candidate"], r["id"]): (r["score"], r["router_usd"]) for r in scores}
        if len(self.cache) != len(scores) or any(
            not math.isfinite(v) or not 0 <= v <= 1 or not math.isfinite(c) or c < 0 for v, c in self.cache.values()
        ):
            raise ValueError("duplicate or invalid score records")
        calls = study.read_jsonl(source / "calls.jsonl")
        inference = json.loads((source / "plan.json").read_text()).get("inference", {})
        self_hosted = inference.get("backend") == "self_hosted"
        if self_hosted and any(r.get("billing") != "self_hosted" or r.get("credits") is not None for r in calls):
            raise ValueError("self-hosted ledger contains inconsistent billing provenance")
        price = inference.get("estimated_router_usd_per_content_token")
        if self_hosted and (type(price) not in (int, float) or not math.isfinite(price) or price <= 0):
            raise ValueError("invalid self-hosted content-token price")
        covered = set()
        for call in calls:
            if call["status"] != "success":
                raise ValueError("a completed replay requires successful call receipts")
            if call["kind"] != "score":
                continue
            amount = call["content_tokens"] * price if self_hosted else call["credits"] * 0.00001
            ids = call["ids"]
            if not ids or len(set(ids)) != len(ids):
                raise ValueError("invalid score receipt IDs")
            for item_id in ids:
                key = (call["candidate"], item_id)
                if key in covered or key not in self.cache:
                    raise ValueError("score receipt coverage mismatch")
                if not math.isclose(self.cache[key][1], amount / len(ids), rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("score receipt cost mismatch")
                covered.add(key)
        if covered != set(self.cache):
            raise ValueError("scores lack matching call receipts")
        self.receipts = {
            "requests": len(calls),
            "score_items": sum(len(r["ids"]) for r in calls if r["kind"] == "score"),
            "reflections": sum(r["kind"] == "reflection" for r in calls),
            "observed_credits": None if self_hosted else sum(r.get("credits", 0) for r in calls),
            "calls_with_unknown_billing": None if self_hosted else sum(r["status"] != "success" for r in calls),
            "elapsed_s": json.loads((source / "results.json").read_text())["cloud"]["elapsed_s"],
        }

    def score(self, candidate, items):
        rows = [self.cache[(digest(candidate), item["id"])] for item in items]
        return [r[0] for r in rows], [r[1] for r in rows]

    def summary(self):
        return self.receipts


def replay(directory, s):
    plan = json.loads((directory / "plan.json").read_text())
    selection = json.loads((directory / "selection.json").read_text())
    expected = json.loads((directory / "results.json").read_text())
    candidates = list(json.loads((directory / "candidates.json").read_text()).values())
    candidates = [{"description": c["description"]} for c in candidates]
    groups = learning.split(s, len(plan["splits"]["initial"]))
    verify({k: [s.ids[i] for i in groups[k]] for k in plan["splits"]}, plan["splits"], "splits")
    verify(plan["study_revision"], study.REVISION, "study revision")
    verify(plan["seed_candidate"], {**difficulty.SEED, "instruction": difficulty.INSTRUCTION}, "seed")
    verify(candidates, selection["candidates"], "candidate coverage")
    with tempfile.TemporaryDirectory(prefix="difficulty-replay-") as temp:
        adapter = difficulty.DifficultyAdapter(s, Recording(directory, Path(temp)), groups)
        computed = difficulty.freeze(adapter, candidates)
        verify(computed, selection, "selection")
        result = difficulty.test_policies(adapter, computed)
        verify(result, expected)
    verify(
        json.loads((directory / "router.json").read_text()),
        {
            "pair": "openai",
            "model": difficulty.reranker.MODEL,
            "profile": "default",
            **selection["policies"]["gepa437"],
        },
        "serving policy",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed difficulty.py output directory")
    args = parser.parse_args()
    if args.run.resolve() == (study.EVIDENCE / "difficulty").resolve():
        difficulty_evidence.verify_evidence(args.run)
    result = replay(args.run, study.load_study())
    print(json.dumps(result, indent=2))
    print("Verified splits, candidate coverage, frozen cutoffs, validation selection, test comparisons and receipts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
