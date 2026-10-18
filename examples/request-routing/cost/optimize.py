"""Evolve a routing description with GEPA, using SIE for scoring AND reflection.

Only archived public training feedback is used. No benchmark answer generation.
Output is a new directory; failed runs retain receipts and are never auto-resumed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import learning
import reranker
import score
import study
from cloud import BudgetExceeded, Cloud, CloudFailure, digest
from run import SingleSendClient

REFLECTION = """Improve a description used as the Query of a yes/no relevance reranker.
The Document is a user's request to an AI assistant. A high relevance score routes it
to the cheaper model; a low score routes it to the stronger model. Do not answer requests.
Current description:
<curr_param>
Training examples and measured routing feedback:
<side_info>
Write a general description of requests where the cheap model preserves strong-model
quality. Capture useful patterns in this feedback, avoiding request IDs, benchmark names,
specific answers or memorized examples. Quality loss is penalized; savings matter only
when quality is preserved. Describe the eligible requests, not instructions to an assistant.
Return ONLY the new description, at most 90 words, inside one plain triple-backtick block.
"""


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path

    def log(self, message: str):
        with self.path.open("a") as handle:
            handle.write(message + "\n")
        if len(message) <= 300:
            print(message, flush=True)


class Adapter:
    propose_new_texts = None

    def __init__(self, s: study.Study, pair: str, cloud: Cloud, fit: list[int], instruction: str):
        self.s, self.o, self.cloud = s, s.pairs[pair], cloud
        self.fit, self.instruction = fit, instruction
        self.thresholds: dict[str, float] = {}
        self.candidates: dict[str, dict] = {}
        self.normalizer = float(self.o.usd_strong[fit].mean())

    def full(self, candidate: dict) -> dict:
        return {"description": candidate["description"], "instruction": self.instruction}

    def signals(self, candidate: dict, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        full = self.full(candidate)
        self.candidates[digest(full)] = full
        write(self.cloud.directory / "candidates.json", self.candidates)
        values, usd = self.cloud.score(full, [{"id": self.s.ids[i], "request": self.s.prompts[i]} for i in indices])
        all_values, all_usd = np.zeros(len(self.s.ids)), np.zeros(len(self.s.ids))
        all_values[indices], all_usd[indices] = values, usd
        return all_values, all_usd

    def threshold(self, candidate: dict) -> float:
        key = digest(candidate)
        if key not in self.thresholds:
            values, usd = self.signals(candidate, self.fit)
            self.thresholds[key] = score.choose_tau(self.o, values, usd, learning.mask(len(values), self.fit))
        return self.thresholds[key]

    def evaluate(self, batch: list[int], candidate: dict, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch  # optional dependency

        tau = self.threshold(candidate)
        values, usd = self.signals(candidate, batch)
        quality, cost, cheap = score.routed(self.o, values, usd, tau)
        # A fixed training-derived scale; validation/test outcomes never define the objective.
        utility = 1 - cost / self.normalizer - 20 * np.maximum(0, self.o.q_strong - quality)
        outputs = [{"score": float(values[i]), "route": "cheap" if cheap[i] else "strong"} for i in batch]
        traces = (
            [
                {
                    "request": self.s.prompts[i],
                    **out,
                    "threshold": None if math.isinf(tau) else tau,
                    "cheap_quality": float(self.o.q_cheap[i]),
                    "strong_quality": float(self.o.q_strong[i]),
                    "cheap_usd": float(self.o.usd_cheap[i]),
                    "strong_usd": float(self.o.usd_strong[i]),
                    "utility": float(utility[i]),
                }
                for i, out in zip(batch, outputs, strict=True)
            ]
            if capture_traces
            else None
        )
        return EvaluationBatch(outputs=outputs, scores=utility[batch].tolist(), trajectories=traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {name: eval_batch.trajectories for name in components_to_update}

    def validation(self, candidate: dict, indices: list[int]) -> dict:
        values, usd = self.signals(candidate, indices)
        return score.summarize(
            self.o, *score.routed(self.o, values, usd, self.threshold(candidate)), learning.mask(len(values), indices)
        )


def experiment(s: study.Study, pair: str, cloud: Cloud, initial_size=64, metric_calls=1200) -> dict:
    import gepa  # optional dependency

    groups = learning.split(s, initial_size)
    spec = reranker.load(study.EVIDENCE / "reranker", s.ids).spec
    seed = {"description": spec["query"]["text"]}
    adapter = Adapter(s, pair, cloud, groups["initial"], spec["instruction"])
    training = [i for i in groups["train"] if i not in set(groups["initial"])]
    write(
        cloud.directory / "plan.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "pair": pair,
            "gepa_version": "0.1.4",
            "study_revision": study.REVISION,
            "seed": 17,
            "seed_candidate": adapter.full(seed),
            "splits": {name: [s.ids[i] for i in indices] for name, indices in groups.items()},
            "limits": {
                "score_items": cloud.max_score_items,
                "reflections": cloud.max_reflections,
                "observed_credits": cloud.max_credits,
                "seconds": cloud.max_seconds,
                "metric_calls": metric_calls,
            },
            "models": {"reranker": reranker.MODEL, "reflection": cloud.reflection_model, "profile": "default"},
            "selection": "Validation only: cheaper than seed, quality >= max(98%, seed quality); otherwise keep seed.",
            "objective": "1 - routed_usd / mean_training_strong_usd - 20 * max(0, strong_quality - routed_quality)",
        },
    )
    result = gepa.optimize(
        seed_candidate=seed,
        trainset=training,
        valset=groups["validation"],
        adapter=adapter,
        reflection_lm=cloud.reflect,
        reflection_prompt_template=REFLECTION,
        reflection_minibatch_size=6,
        skip_perfect_score=False,
        use_merge=False,
        max_metric_calls=metric_calls,
        stop_callbacks=lambda state: cloud.failed or cloud.reflections >= cloud.max_reflections,
        run_dir=str(cloud.directory / "gepa"),
        logger=RunLogger(cloud.directory / "optimization.log"),
        seed=17,
        raise_on_exception=True,
        display_progress_bar=False,
        track_best_outputs=False,
    )
    candidates = result.candidates
    validation = [adapter.validation(candidate, groups["validation"]) for candidate in candidates]
    eligible = [i for i in range(1, len(candidates)) if learning.improvement(validation[i], validation[0])]
    chosen = min(eligible, key=lambda i: validation[i]["cost_per_1k"]) if eligible else 0
    # Freeze selection BEFORE any final calibration or test inference/evaluation.
    write(
        cloud.directory / "selection.json",
        {
            "chosen": chosen,
            "promoted": chosen != 0,
            "candidates": candidates,
            "validation": validation,
            "gepa_best_idx": result.best_idx,
        },
    )
    heldout = {}
    for name, candidate in (("static", seed), ("updated", candidates[chosen])):
        values, usd = adapter.signals(candidate, groups["threshold"] + groups["test"])
        heldout[name] = learning.evaluate(adapter.o, values, usd, groups["threshold"], groups["test"])
    write(
        cloud.directory / "router.json",
        {
            "pair": pair,
            "model": reranker.MODEL,
            "profile": "default",
            **adapter.full(candidates[chosen]),
            "tau": heldout["updated"]["tau"],
        },
    )
    output = {
        "pair": pair,
        "split_counts": {k: len(v) for k, v in groups.items()},
        "embedding": learning.embedding_experiment(s, pair, groups),
        "reranker": {"promoted": chosen != 0, "chosen": chosen, "validation": validation, "heldout": heldout},
        "cloud": cloud.summary(),
        "cost_note": "Historical answer list costs + current batched reranker receipts (credits at $0.00001). "
        "Embedding costs are archived. Optimizer spend is separate; these are estimates, not a live tariff.",
    }
    write(cloud.directory / "results.json", output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=sorted(study.PAIRS), default="openai")
    parser.add_argument("--output", type=Path, default=study.HERE / "run-output/gepa-cloud")
    parser.add_argument("--initial-size", type=int, default=64)
    parser.add_argument("--max-reflections", type=int, default=6)
    parser.add_argument("--max-score-items", type=int, default=5000)
    parser.add_argument("--max-credits", type=float, default=20000)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--metric-calls", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reflection-model", default="Qwen/Qwen3.8-27B-FP8")
    args = parser.parse_args()
    if not os.environ.get("SIE_API_KEY", "").strip():
        parser.error("set SIE_API_KEY from the SIE console")
    if any(
        not math.isfinite(v) or v <= 0
        for v in (
            args.initial_size,
            args.max_reflections,
            args.max_score_items,
            args.max_credits,
            args.max_seconds,
            args.metric_calls,
            args.batch_size,
        )
    ):
        parser.error("all budgets and sizes must be positive and finite")
    try:
        import gepa  # noqa: F401 -- check before creating output or spending credits
    except ImportError:
        parser.error("install GEPA with: uv sync --frozen --group gepa")
    s = study.load_study(study.EVIDENCE)
    learning.split(s, args.initial_size)
    reranker.verify_evidence(study.EVIDENCE / "reranker")
    args.output.mkdir(parents=True, exist_ok=False)
    with SingleSendClient(timeout_s=60, max_connections=1) as client:
        cloud = Cloud(
            client,
            args.output,
            max_score_items=args.max_score_items,
            max_reflections=args.max_reflections,
            max_credits=args.max_credits,
            max_seconds=args.max_seconds,
            batch_size=args.batch_size,
            reflection_model=args.reflection_model,
        )
        try:
            result = experiment(s, args.pair, cloud, args.initial_size, args.metric_calls)
        except (BudgetExceeded, CloudFailure, ValueError) as error:
            write(args.output / "incomplete.json", {"error_type": type(error).__name__, "cloud": cloud.summary()})
            print(
                f"Stopped: {type(error).__name__}. Receipts retained in {args.output}; no automatic retry.",
                file=sys.stderr,
            )
            return 1
        finally:
            cloud.close()
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved results and deployable router.json to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
