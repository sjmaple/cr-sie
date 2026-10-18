"""A larger GEPA experiment with diagnostic training feedback and a reserved audit.

The audit excludes every request scored in the published first GEPA run. It
is fresh to prompt optimization, not a newly collected independent benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import learning
import learning_evidence
import optimize
import reranker
import score
import study
from cloud import BudgetExceeded, Cloud, CloudFailure
from run import SingleSendClient


def stratified(s, indices, count, salt):
    buckets = {name: [i for i in indices if s.strata[i] == name] for name in sorted(set(s.strata[indices]))}
    exact = {name: len(rows) * count / len(indices) for name, rows in buckets.items()}
    sizes = {name: math.floor(value) for name, value in exact.items()}
    for name in sorted(buckets, key=lambda name: (-(exact[name] - sizes[name]), name))[: count - sum(sizes.values())]:
        sizes[name] += 1
    return [
        i
        for name, rows in buckets.items()
        for i in sorted(rows, key=lambda i: hashlib.sha256(f"{salt}:{s.ids[i]}".encode()).digest())[: sizes[name]]
    ]


def partitions(s, previous):
    used = {row["id"] for row in study.read_jsonl(previous / "scores.jsonl")}
    calibration = np.where(s.calibration)[0].tolist()
    fresh = [i for i in calibration if s.ids[i] not in used]
    if len(fresh) < 192:
        raise ValueError("need 192 calibration examples never scored in the first GEPA run")
    audit = stratified(s, fresh, 192, "audit-v2")
    available = [i for i in calibration if i not in set(audit)]
    fit = stratified(s, available, 128, "fit-v2")
    available = [i for i in available if i not in set(fit)]
    validation = stratified(s, available, 160, "validation-v2")
    train = [i for i in available if i not in set(validation)]
    return {"fit": fit, "train": train, "validation": validation, "audit": audit}


TEMPLATE = """Optimize the {component} of a yes/no relevance reranker used to route requests.
High score sends the Document request to GPT-6 Luna (cheap); low score to GPT-6.1 Sol (strong).
The Query describes requests where using the cheap model preserves the strong model's quality.
This is comparative quality: both models failing is not a reason to pay for the stronger one.
The reranker sees only request text plus the query/instruction, never grades or task answers.
Current {component}:
<curr_param>
Diagnostic feedback from TRAINING DATA ONLY:
<side_info>
Improve generalization, avoiding memorized answers, exact requests, benchmark names or IDs.
Identify why expensive unnecessary escalation occurs and why harmful cheap routing occurs.
The other prompt component is shown in the feedback; make the two consistent.
The evaluator refits the cutoff for each prompt on a separate calibration sample. A low raw
score alone is not an error; the ranking of safe versus harmful cheap decisions matters.
Do not turn a few examples into blanket bans on whole task types. Use the aggregate evidence.
{guidance}
Return ONLY the replacement text in one plain triple-backtick block, at most 160 words.
"""


class DiagnosticAdapter(optimize.Adapter):
    def __init__(self, s, cloud, groups):
        super().__init__(s, "openai", cloud, groups["fit"], "")
        self.groups = groups

    def full(self, candidate):
        return dict(candidate)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        train = self.groups["train"]
        values, usd = self.signals(candidate, train)
        tau = self.threshold(candidate)
        quality, _, cheap = score.routed(self.o, values, usd, tau)
        loss = np.maximum(0, self.o.q_strong - quality)
        opportunity = np.where(~cheap & (self.o.q_cheap >= self.o.q_strong), self.o.usd_strong - self.o.usd_cheap, 0)
        harmful = sorted((i for i in train if loss[i] > 0), key=lambda i: -loss[i])
        wasted = sorted((i for i in train if opportunity[i] > 0), key=lambda i: -opportunity[i])
        examples = list(dict.fromkeys(harmful[:4] + wasted[:4] + learning.balanced(train, self.s.strata, 4)))
        summary = {}
        for name in sorted(set(self.s.strata[train])):
            indices = [i for i in train if self.s.strata[i] == name]
            summary[str(name)] = {
                "n": len(indices),
                "cheap_quality": float(self.o.q_cheap[indices].mean()),
                "strong_quality": float(self.o.q_strong[indices].mean()),
                "harmful_cheap": int(sum(loss[indices] > 0)),
                "unnecessary_strong": int(sum(opportunity[indices] > 0)),
            }
        feedback = [
            {
                "current_prompt": candidate,
                "training_summary": summary,
                "threshold": None if math.isinf(tau) else tau,
                "objective": "Reduce cost without reducing quality. A quality point lost costs 20 utility units.",
            }
        ]
        feedback += [
            {
                "request": self.s.prompts[i][:800],
                "truncated_for_reflection": len(self.s.prompts[i]) > 800,
                "route": "cheap" if cheap[i] else "strong",
                "score": float(values[i]),
                "cheap_quality": float(self.o.q_cheap[i]),
                "strong_quality": float(self.o.q_strong[i]),
                "quality_lost": float(loss[i]),
                "unnecessary_spend_usd": float(opportunity[i]),
                "cheap_usd": float(self.o.usd_cheap[i]),
                "strong_usd": float(self.o.usd_strong[i]),
            }
            for i in examples
        ]
        return {name: feedback for name in components_to_update}


def pareto_better(row, baseline):
    return (
        row["cost_per_1k"] < baseline["cost_per_1k"] - 1e-9
        and row["quality_vs_strong"] >= baseline["quality_vs_strong"] - 1e-12
    )


def experiment(s, cloud, groups):
    import gepa

    seed_spec = reranker.load(study.EVIDENCE / "reranker", s.ids).spec
    seed = {"description": seed_spec["query"]["text"], "instruction": seed_spec["instruction"]}
    plan = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": "diagnostic-gepa-v2",
        "pair": "openai",
        "study_revision": study.REVISION,
        "prior_gepa_revision": learning_evidence.REVISION,
        "seed": 29,
        "seed_candidate": seed,
        "splits": {name: [s.ids[i] for i in indices] for name, indices in groups.items()},
        "models": {"reranker": reranker.MODEL, "reflection": cloud.reflection_model, "profile": "default"},
        "selection": "First choose cheapest validation candidate with no quality regression versus seed. "
        "If none, retain seed. Report 98% target separately; deployment still requires it.",
        "threshold": "Fit on 128 only; freeze chosen prompt AND threshold before audit, with no refitting.",
        "audit_note": "192 original calibration requests absent from all published v1 GEPA scores. "
        "Public and previously used in embedding banks; not newly collected or independent data.",
        "limits": {
            "score_items": cloud.max_score_items,
            "reflections": cloud.max_reflections,
            "observed_credits": cloud.max_credits,
            "seconds": cloud.max_seconds,
        },
    }
    optimize.write(cloud.directory / "plan.json", plan)
    adapter = DiagnosticAdapter(s, cloud, groups)
    templates = {
        name: TEMPLATE.format(component=name, guidance=guidance)
        for name, guidance in {
            "description": "Write a positive description of suitable requests, optionally with concise exclusions.",
            "instruction": "Write the criterion the yes/no reranker should apply to the Document and Query. Do not answer the request.",
        }.items()
    }
    result = gepa.optimize(
        seed_candidate=seed,
        trainset=groups["train"],
        valset=groups["validation"],
        adapter=adapter,
        reflection_lm=cloud.reflect,
        reflection_prompt_template=templates,
        reflection_minibatch_size=24,
        skip_perfect_score=False,
        use_merge=False,
        max_metric_calls=8000,
        stop_callbacks=lambda state: cloud.failed or cloud.reflections >= cloud.max_reflections,
        run_dir=str(cloud.directory / "gepa"),
        logger=optimize.RunLogger(cloud.directory / "optimization.log"),
        seed=29,
        raise_on_exception=True,
        display_progress_bar=False,
        track_best_outputs=False,
    )
    # Include rejected minibatch proposals: minibatch utility is not the final validation gate.
    candidates = list(adapter.candidates.values())
    validation = [adapter.validation(c, groups["validation"]) for c in candidates]
    eligible = [i for i in range(1, len(candidates)) if pareto_better(validation[i], validation[0])]
    chosen = min(eligible, key=lambda i: validation[i]["cost_per_1k"]) if eligible else 0
    selection = {
        "chosen": chosen,
        "candidates": candidates,
        "validation": validation,
        "thresholds": [None if math.isinf(adapter.threshold(c)) else adapter.threshold(c) for c in candidates],
        "gepa_accepted_candidates": len(result.candidates),
        "validation_improved": chosen != 0,
    }
    optimize.write(cloud.directory / "selection.json", selection)
    # This is the first point the audit can be scored. Neither audit grades nor scores enter reflection.
    audit = {
        name: adapter.validation(candidates[index], groups["audit"])
        for name, index in (("seed", 0), ("selected", chosen))
    }
    output = {
        "validation": {"seed": validation[0], "selected": validation[chosen]},
        "audit": audit,
        "validation_improved": chosen != 0,
        "audit_improved": pareto_better(audit["selected"], audit["seed"]),
        "audit_meets_quality_target": audit["selected"]["quality_vs_strong"] >= score.TARGET,
        "cloud": cloud.summary(),
    }
    optimize.write(cloud.directory / "results.json", output)
    optimize.write(
        cloud.directory / "router.json",
        {
            "pair": "openai",
            "model": reranker.MODEL,
            "profile": "default",
            **candidates[chosen],
            "tau": selection["thresholds"][chosen],
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=study.HERE / "run-output/gepa-diagnostic")
    parser.add_argument("--reflection-model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--max-reflections", type=int, default=24)
    parser.add_argument("--max-score-items", type=int, default=20000)
    args = parser.parse_args()
    if not os.environ.get("SIE_API_KEY", "").strip():
        parser.error("set SIE_API_KEY")
    if args.max_reflections <= 0 or args.max_score_items <= 0:
        parser.error("budgets must be positive")
    learning_evidence.verify_evidence(study.EVIDENCE / "learning")
    s = study.load_study(study.EVIDENCE)
    groups = partitions(s, study.EVIDENCE / "learning")
    args.output.mkdir(parents=True, exist_ok=False)
    with SingleSendClient(timeout_s=60, max_connections=1) as client:
        cloud = Cloud(
            client,
            args.output,
            max_score_items=args.max_score_items,
            max_reflections=args.max_reflections,
            max_credits=40000,
            max_seconds=1800,
            batch_size=32,
            reflection_model=args.reflection_model,
        )
        try:
            result = experiment(s, cloud, groups)
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
