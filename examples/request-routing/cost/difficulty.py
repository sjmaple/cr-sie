"""Evolve a cheap-versus-strong routing description with GEPA on SIE Cloud.

Uses the embedding workshop's splits, a calibration-only control, and a prompt
AND cutoff frozen before test. No downstream answer generation is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import learning
import optimize
import reranker
import score
import study
from cloud import BudgetExceeded, Cloud, CloudFailure
from run import SingleSendClient

SEED = {"description": "A request that a small, cheap AI model can answer correctly."}
INSTRUCTION = "Judge whether the Document, a request sent to an AI assistant, fits the description in the Query."
UTILITY_WEIGHT = 8
REFLECTION = """Write a description of requests to send to the CHEAP model.
Your description becomes the Query of a yes/no relevance reranker. The Document
is a user's request. A YES match to your description sends it to GPT-6 Luna
(CHEAP); a NO match sends it to GPT-6.1 Sol (STRONG). The description must name
the CHEAP class, never the class that needs the strong model.
The reranker never sees answers or these training grades.
Current description:
<curr_param>
Paired model grades and routing feedback from TRAINING ONLY:
<side_info>
Learn which requests are adequately handled by the CHEAP model from these grades,
not from absolute difficulty. Both models succeeding OR failing similarly means
little reason to pay more, so those are useful POSITIVE examples of the cheap class.
Some difficult-looking questions have no measured benefit from escalation;
some simple-looking writing requests have a large quality gap and should NOT
match your cheap-class description. Infer the pattern from aggregate grades and
representative failures, rather than assuming it.
The evaluator fits each candidate's cutoff on all 437 training requests to retain
98% of strong-model quality. Raw score scale does not matter; ranking does.
Describe observable task characteristics. Prefer a concise, coherent positive
description with clear boundaries over abstract claims about model capabilities.
Do not include dataset names, IDs, answers, copied requests, or model names.
Do not turn one failure into a blanket ban. Distinguish unnecessary escalation
from harmful cheap routing; both matter. The fixed instruction tests membership
in your described category; do not put instructions to answer requests here.
Return ONLY the replacement description, at most 120 words, in a plain triple-backtick block.
Before answering, check its polarity: a request matching your description will
go to CHEAP. Describing difficult requests that need STRONG reverses the router.
Do not write routing commands such as "route to strong". Describe the observable
features of the CHEAP class in positive terms, starting with "A request".
"""


def row(o, values, usd, tau, indices):
    mask = learning.mask(len(values), indices)
    quality, total, cheap = score.routed(o, values, usd, tau)
    return score.summarize(o, quality, total, cheap, mask) | {
        "routing_per_1k": float(1000 * usd[mask].mean()),
        "generation_per_1k": float(1000 * (total[mask] - usd[mask]).mean()),
        "strong_requests": int(sum(~cheap[mask])),
    }


class DifficultyAdapter(optimize.Adapter):
    def __init__(self, s, cloud, groups):
        super().__init__(s, "openai", cloud, groups["train"], INSTRUCTION)
        self.groups = groups

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        tau = self.threshold(candidate)
        values, usd = self.signals(candidate, batch)
        quality, cost, cheap = score.routed(self.o, values, usd, tau)
        # Signed quality difference matches the retained-quality metric, including
        # requests on which the cheap model outperforms the strong model.
        utility = 1 - cost / self.normalizer + UTILITY_WEIGHT * (quality - self.o.q_strong)
        outputs = [{"score": float(values[i]), "route": "cheap" if cheap[i] else "strong"} for i in batch]
        return EvaluationBatch(
            outputs=outputs, scores=utility[batch].tolist(), trajectories=outputs if capture_traces else None
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        train = self.groups["train"]
        values, usd = self.signals(candidate, train)
        quality, _, cheap = score.routed(self.o, values, usd, self.threshold(candidate))
        lost = self.o.q_strong - quality
        waste = np.where(~cheap & (self.o.q_cheap >= self.o.q_strong), self.o.usd_strong - self.o.usd_cheap, 0)
        harmful = sorted((i for i in train if lost[i] > 0), key=lambda i: -lost[i])[:5]
        unnecessary = sorted((i for i in train if waste[i] > 0), key=lambda i: -waste[i])[:5]
        # Rotate representative rows, without choosing by validation or test outcomes.
        offset = (self.cloud.reflections * 7) % len(train)
        representative = learning.balanced(train[offset:] + train[:offset], self.s.strata, 6)
        examples = list(dict.fromkeys(harmful + unnecessary + representative))
        categories = {
            "chat": "open-ended writing and conversation",
            "math500": "math",
            "mbpp_plus": "coding",
            "supergpqa": "knowledge questions",
        }
        summary = {}
        for category in sorted(set(self.s.strata[train])):
            indices = [i for i in train if self.s.strata[i] == category]
            summary[categories.get(str(category), str(category))] = {
                "n": len(indices),
                "cheap_mean_quality": float(self.o.q_cheap[indices].mean()),
                "strong_mean_quality": float(self.o.q_strong[indices].mean()),
                "cheap_better": int(sum(self.o.q_cheap[indices] > self.o.q_strong[indices])),
                "strong_better": int(sum(self.o.q_strong[indices] > self.o.q_cheap[indices])),
                "equal_quality": int(sum(self.o.q_cheap[indices] == self.o.q_strong[indices])),
                "harmful_cheap": int(sum(lost[indices] > 0)),
                "unnecessary_strong": int(sum(waste[indices] > 0)),
            }
        feedback = [{"aggregate_training_grades": summary}]
        feedback.extend(
            {
                "request": self.s.prompts[i][:1000],
                "truncated_for_reflection": len(self.s.prompts[i]) > 1000,
                "route": "cheap" if cheap[i] else "strong",
                "cheap_quality": float(self.o.q_cheap[i]),
                "strong_quality": float(self.o.q_strong[i]),
                "cheap_is_as_good": bool(self.o.q_cheap[i] >= self.o.q_strong[i]),
                "cheap_usd": float(self.o.usd_cheap[i]),
                "strong_usd": float(self.o.usd_strong[i]),
            }
            for i in examples
        )
        return {name: feedback for name in components_to_update}


def freeze(adapter, candidates):
    """Select on validation only and return the exact policies to evaluate."""
    g, o = adapter.groups, adapter.o
    values, usd = adapter.signals(SEED, g["initial"] + g["validation"])
    initial_tau = score.choose_tau(o, values, usd, learning.mask(len(values), g["initial"]))
    initial = row(o, values, usd, initial_tau, g["validation"])
    validation, thresholds = [], []
    for candidate in candidates:
        tau = adapter.threshold(candidate)
        values, usd = adapter.signals(candidate, g["validation"])
        validation.append(row(o, values, usd, tau, g["validation"]))
        thresholds.append(None if math.isinf(tau) else tau)
    eligible = [i for i in range(1, len(candidates)) if learning.improvement(validation[i], initial)]
    chosen = min(eligible, key=lambda i: validation[i]["cost_per_1k"]) if eligible else None
    policies = {
        "static64": {**adapter.full(SEED), "tau": None if math.isinf(initial_tau) else initial_tau},
        "calibration_only437": {**adapter.full(SEED), "tau": thresholds[0]},
        "gepa437": {**adapter.full(candidates[chosen]), "tau": thresholds[chosen]}
        if chosen is not None
        else {**adapter.full(SEED), "tau": None if math.isinf(initial_tau) else initial_tau},
    }
    selection = {
        "candidates": candidates,
        "validation": validation,
        "initial_validation": initial,
        "thresholds": thresholds,
        "chosen": chosen,
        "policies": policies,
    }
    optimize.write(adapter.cloud.directory / "selection.json", selection)
    return selection


def paired(s, o, indices, before, after):
    rng = np.random.default_rng(67)
    groups = [[i for i in indices if s.strata[i] == k] for k in sorted(set(s.strata[indices]))]
    samples = np.concatenate([rng.choice(g, (4000, len(g))) for g in groups], axis=1)
    qa, ca, _ = score.routed(o, *before)
    qb, cb, _ = score.routed(o, *after)
    cost = 100 * (1 - cb[samples].mean(axis=1) / ca[samples].mean(axis=1))
    quality = 100 * (qb[samples].mean(axis=1) - qa[samples].mean(axis=1)) / o.q_strong[samples].mean(axis=1)
    return {
        "cost_reduction_pct": float(100 * (1 - cb[indices].mean() / ca[indices].mean())),
        "cost_reduction_pct_95ci": np.quantile(cost, [0.025, 0.975]).tolist(),
        "quality_change_pp": float(100 * (qb[indices].mean() - qa[indices].mean()) / o.q_strong[indices].mean()),
        "quality_change_pp_95ci": np.quantile(quality, [0.025, 0.975]).tolist(),
        "method": "paired stratified percentile bootstrap",
        "resamples": 4000,
        "seed": 67,
    }


def test_policies(adapter, selection):
    # Called only after selection.json has been written. No threshold refitting.
    indices, o = adapter.groups["test"], adapter.o
    rows, signals = {}, {}
    for name, policy in selection["policies"].items():
        values, usd = adapter.signals(policy, indices)
        tau = math.inf if policy["tau"] is None else policy["tau"]
        rows[name] = row(o, values, usd, tau, indices)
        signals[name] = (values, usd, tau)
    return {
        "test_n": len(indices),
        "promoted": selection["chosen"] is not None,
        "test": rows,
        "vs_static": paired(adapter.s, o, indices, signals["static64"], signals["gepa437"]),
        "vs_calibration_only": paired(adapter.s, o, indices, signals["calibration_only437"], signals["gepa437"]),
        "cloud": adapter.cloud.summary(),
    }


def experiment(s, cloud, initial_size=64):
    import gepa

    groups = learning.split(s, initial_size)
    plan = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": "gepa-difficulty-v2",
        "study_revision": study.REVISION,
        "gepa_version": "0.1.4",
        "seed": 67,
        "seed_candidate": {**SEED, "instruction": INSTRUCTION},
        "splits": {k: [s.ids[i] for i in groups[k]] for k in ("initial", "train", "validation", "test")},
        "unused": "The former 146 threshold-calibration requests are unused, as in learn.py.",
        "threshold": "Seed on initial64; calibration control and every GEPA candidate on train437. Retain 98% quality. Freeze before test.",
        "selection": "Cheapest validation GEPA candidate cheaper than initial64 and quality >= max(98%, initial64). Otherwise retain initial64.",
        "objective": f"1 - cost / mean_training_strong_cost + {UTILITY_WEIGHT} * (routed_quality - strong_quality)",
        "models": {"reranker": reranker.MODEL, "reflection": cloud.reflection_model, "profile": "default"},
        "inference": {
            "backend": "cloud" if cloud.self_hosted_usd_per_token is None else "self_hosted",
            "estimated_router_usd_per_content_token": cloud.self_hosted_usd_per_token,
            "cost_basis": "Billed batched credits"
            if cloud.self_hosted_usd_per_token is None
            else "Measured content tokens times an explicitly supplied historical tariff; excludes GPU rental",
        },
        "limits": {
            "score_items": cloud.max_score_items,
            "reflections": cloud.max_reflections,
            "credits": cloud.max_credits,
            "seconds": cloud.max_seconds,
        },
        "limitations": "Reused public benchmark; method developed with knowledge of prior runs. Test outcomes do not select this run's prompt or threshold. Batched routing receipts and historical downstream prices are estimates, not live individual-request tariffs.",
    }
    optimize.write(cloud.directory / "plan.json", plan)
    adapter = DifficultyAdapter(s, cloud, groups)
    gepa.optimize(
        seed_candidate=SEED,
        trainset=groups["train"],
        valset=groups["validation"],
        adapter=adapter,
        reflection_lm=cloud.reflect,
        reflection_prompt_template=REFLECTION,
        reflection_minibatch_size=32,
        skip_perfect_score=False,
        use_merge=False,
        max_metric_calls=12000,
        stop_callbacks=lambda state: cloud.failed or cloud.reflections >= cloud.max_reflections,
        run_dir=str(cloud.directory / "gepa"),
        logger=optimize.RunLogger(cloud.directory / "optimization.log"),
        seed=67,
        raise_on_exception=True,
        display_progress_bar=False,
        track_best_outputs=False,
    )
    candidates = [{"description": c["description"]} for c in adapter.candidates.values()]
    selection = freeze(adapter, candidates)
    output = test_policies(adapter, selection)
    optimize.write(cloud.directory / "results.json", output)
    optimize.write(
        cloud.directory / "router.json",
        {"pair": "openai", "model": reranker.MODEL, "profile": "default", **selection["policies"]["gepa437"]},
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=study.HERE / "run-output/gepa-difficulty")
    parser.add_argument("--max-reflections", type=int, default=16)
    parser.add_argument("--max-score-items", type=int, default=16000)
    parser.add_argument("--max-credits", type=float, default=30000)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument(
        "--self-hosted-usd-per-token",
        type=float,
        help="Explicit historical routing-cost estimate; requires SIE_BASE_URL. No Console credits are reported.",
    )
    args = parser.parse_args()
    if args.self_hosted_usd_per_token is not None and not os.environ.get("SIE_BASE_URL", "").strip():
        parser.error("self-hosted mode requires an explicit SIE_BASE_URL")
    if args.self_hosted_usd_per_token is None and not os.environ.get("SIE_API_KEY", "").strip():
        parser.error("set SIE_API_KEY from the SIE console")
    if any(
        not math.isfinite(v) or v <= 0
        for v in (args.max_reflections, args.max_score_items, args.max_credits, args.max_seconds)
    ):
        parser.error("budgets must be positive and finite")
    try:
        import gepa  # noqa: F401 -- fail before spending credits
    except ImportError:
        parser.error("install GEPA with: uv sync --frozen --group gepa")
    s = study.load_study()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for file in [*study.HERE.glob("*.py"), study.HERE / "pyproject.toml", study.HERE / "uv.lock"]:
        shutil.copy2(file, source / file.name)
    with (
        SingleSendClient(timeout_s=120, max_connections=1) as client,
        SingleSendClient(
            base_url=os.environ.get("SIE_REFLECTION_BASE_URL") or os.environ.get("SIE_BASE_URL"),
            api_key=os.environ.get("SIE_REFLECTION_API_KEY"),
            timeout_s=180,
            max_connections=1,
        ) as reflection_client,
    ):
        cloud = Cloud(
            client,
            args.output,
            max_reflections=args.max_reflections,
            max_score_items=args.max_score_items,
            max_credits=args.max_credits,
            max_seconds=args.max_seconds,
            batch_size=32,
            reflection_client=reflection_client,
            self_hosted_usd_per_token=args.self_hosted_usd_per_token,
        )
        try:
            result = experiment(s, cloud)
        except (BudgetExceeded, CloudFailure, ValueError) as error:
            optimize.write(
                args.output / "incomplete.json", {"error_type": type(error).__name__, "cloud": cloud.summary()}
            )
            print(f"Stopped: {type(error).__name__}; receipts retained in {args.output}")
            return 1
        finally:
            cloud.close()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
