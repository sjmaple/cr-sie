"""Learn descriptions for the existing eight request handlers using GEPA on SIE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import intent_evidence
import optimize
import reranker
import study
from cloud import BudgetExceeded, Cloud, CloudFailure, digest
from run import SingleSendClient

SOURCE_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
SOURCE_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
HANDLERS = {
    "calendar": "view calendar events",
    "calendar_update": "edit calendar events",
    "meeting_schedule": "view meeting schedule",
    "schedule_meeting": "book a meeting room",
    "reminder": "view reminders",
    "reminder_update": "edit reminders",
    "todo_list": "view to-do list",
    "todo_list_update": "edit to-do list",
}
INSTRUCTION = "Judge whether the Document is a user request for the action described in the Query."
REFLECTION = """Improve ONE description in an eight-handler request router.
A yes/no relevance reranker scores the request against each handler description, with
the fixed instruction to judge whether the request asks for that action. Highest score wins.
Current description:
<curr_param>
TRAINING examples, competing descriptions, and errors:
<side_info>
Write a clear description that includes the intended requests and distinguishes confusable
handlers. Use the labeled examples to resolve ambiguity, including vocabulary and synonyms.
Do not copy exact requests, add new handlers, mention the dataset, or answer the user's request.
Keep it concise: at most 70 words. Return ONLY the new description in one plain ``` block.
"""


def validate_data(data):
    groups = data.get("splits", {})
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("data must have train, validation and test splits")
    ids, texts = set(), set()
    for rows in groups.values():
        if not rows or {r["label"] for r in rows} != set(HANDLERS):
            raise ValueError("each split must contain all eight handlers")
        split_texts = set()
        for row in rows:
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("id", "request", "label")):
                raise ValueError("rows need nonempty id, request and label strings")
            if row["id"] in ids:
                raise ValueError("duplicate request ID across splits")
            ids.add(row["id"])
            normalized = row["request"].strip().casefold()
            if normalized in texts:
                raise ValueError("request text overlaps across splits")
            split_texts.add(normalized)
        texts.update(split_texts)


def prepare(source: Path, destination: Path):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("CLINC150 source does not match its pinned hash")
    source_data = json.loads(raw)
    data = {
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "license": "CC BY 3.0",
        "attribution": "Larson et al. (2019), CLINC150 / clinc/oos-eval",
        "handlers": HANDLERS,
        "splits": {},
    }
    for name, source_split in (("train", "train"), ("validation", "val"), ("test", "test")):
        rows = [
            {"id": f"clinc-{source_split}-{i}", "request": text, "label": label}
            for i, (text, label) in enumerate(source_data[source_split])
            if label in HANDLERS
        ]
        if name == "train":
            rows = [
                row
                for label in HANDLERS
                for row in sorted(
                    (r for r in rows if r["label"] == label),
                    key=lambda r: digest(["intent-gepa-v1", r["id"]]),
                )[:10]
            ]
        data["splits"][name] = rows
    validate_data(data)
    if destination.exists():
        raise FileExistsError(destination)
    optimize.write(destination, data)
    return data


class IntentAdapter:
    propose_new_texts = None

    def __init__(self, cloud, training):
        self.cloud, self.training = cloud, training
        self.candidates = {}

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        self.candidates[digest(candidate)] = dict(candidate)
        optimize.write(self.cloud.directory / "candidates.json", self.candidates)
        matrix = np.array(
            [
                self.cloud.score({"description": candidate[label], "instruction": INSTRUCTION}, batch)[0]
                for label in HANDLERS
            ]
        ).T
        target = np.array([list(HANDLERS).index(row["label"]) for row in batch])
        prediction = matrix.argmax(axis=1)
        tied = (matrix == matrix.max(axis=1, keepdims=True)).sum(axis=1) > 1
        correct = (prediction == target) & ~tied
        competitor = matrix.copy()
        competitor[np.arange(len(batch)), target] = -np.inf
        margin = matrix[np.arange(len(batch)), target] - competitor.max(axis=1)
        # Accuracy dominates; a small margin term gives feedback before argmax flips.
        utilities = correct.astype(float) + 0.05 * margin
        outputs = [
            {
                "id": row["id"],
                "expected": row["label"],
                "predicted": None if tied[i] else list(HANDLERS)[prediction[i]],
                "correct": bool(correct[i]),
                "scores": matrix[i].tolist(),
            }
            for i, row in enumerate(batch)
        ]
        traces = (
            [{"request": row["request"], **out} for row, out in zip(batch, outputs, strict=True)]
            if capture_traces
            else None
        )
        return EvaluationBatch(outputs=outputs, scores=utilities.tolist(), trajectories=traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        training = self.evaluate(self.training, candidate, capture_traces=True)
        result = {}
        for component in components_to_update:
            rows = [r for r in training.trajectories if r["expected"] == component or r["predicted"] == component]
            rows.sort(key=lambda r: (r["correct"], r["id"]))
            result[component] = [{"handler_to_improve": component, "all_descriptions": candidate}] + [
                {k: v for k, v in row.items() if k != "id"} for row in rows[:20]
            ]
        return result


def accuracy(adapter, rows, candidate):
    outputs = adapter.evaluate(rows, candidate).outputs
    return float(np.mean([row["correct"] for row in outputs])), outputs


def experiment(data, cloud):
    import gepa

    validate_data(data)
    groups = data["splits"]
    seed = dict(HANDLERS)
    optimize.write(cloud.directory / "data.json", data)
    optimize.write(
        cloud.directory / "plan.json",
        {
            "task": "CLINC150 eight handlers already used by examples/request-routing/direct",
            "seed_candidate": seed,
            "instruction": INSTRUCTION,
            "seed": 41,
            "source_revision": SOURCE_REVISION,
            "split_counts": {k: len(v) for k, v in groups.items()},
            "selection": "Highest validation accuracy; ties prefer shorter total descriptions, then earlier candidate. "
            "An update requires strictly higher accuracy than seed. Test evaluated after selection only.",
            "models": {"reranker": reranker.MODEL, "reflection": cloud.reflection_model},
            "limits": {
                "reflections": cloud.max_reflections,
                "score_items": cloud.max_score_items,
                "observed_credits": cloud.max_credits,
                "seconds": cloud.max_seconds,
            },
        },
    )
    adapter = IntentAdapter(cloud, groups["train"])
    gepa.optimize(
        seed_candidate=seed,
        trainset=groups["train"],
        valset=groups["validation"],
        adapter=adapter,
        reflection_lm=cloud.reflect,
        reflection_prompt_template=REFLECTION,
        reflection_minibatch_size=16,
        skip_perfect_score=False,
        use_merge=False,
        max_metric_calls=6000,
        stop_callbacks=lambda state: cloud.failed or cloud.reflections >= cloud.max_reflections,
        run_dir=str(cloud.directory / "gepa"),
        logger=optimize.RunLogger(cloud.directory / "optimization.log"),
        seed=41,
        raise_on_exception=True,
        display_progress_bar=False,
        track_best_outputs=False,
    )
    candidates = list(adapter.candidates.values())
    validation = [accuracy(adapter, groups["validation"], c)[0] for c in candidates]
    chosen = min(range(len(candidates)), key=lambda i: (-validation[i], sum(map(len, candidates[i].values())), i))
    if validation[chosen] <= validation[0]:
        chosen = 0
    optimize.write(
        cloud.directory / "selection.json", {"candidates": candidates, "validation": validation, "chosen": chosen}
    )
    baseline, before = accuracy(adapter, groups["test"], seed)
    updated, after = accuracy(adapter, groups["test"], candidates[chosen])
    optimize.write(cloud.directory / "predictions.json", {"seed": before, "selected": after})
    result = {
        "validation_accuracy": {"seed": validation[0], "selected": validation[chosen]},
        "test_accuracy": {"seed": baseline, "selected": updated},
        "test_n": len(before),
        "corrected": sum(not a["correct"] and b["correct"] for a, b in zip(before, after, strict=True)),
        "regressed": sum(a["correct"] and not b["correct"] for a, b in zip(before, after, strict=True)),
        "chosen": chosen,
        "cloud": cloud.summary(),
    }
    optimize.write(cloud.directory / "results.json", result)
    optimize.write(cloud.directory / "handlers.json", {"descriptions": candidates[chosen], "instruction": INSTRUCTION})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=study.EVIDENCE / "intents/data.json")
    parser.add_argument("--prepare-source", type=Path, help="prepare data from the pinned official CLINC150 JSON")
    parser.add_argument("--output", type=Path, default=study.HERE / "run-output/gepa-intents")
    parser.add_argument("--reflection-model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--max-reflections", type=int, default=16)
    args = parser.parse_args()
    if args.prepare_source:
        prepare(args.prepare_source, args.data)
        return 0
    if not os.environ.get("SIE_API_KEY", "").strip():
        parser.error("set SIE_API_KEY")
    if args.max_reflections <= 0:
        parser.error("--max-reflections must be positive")
    if args.data.resolve() == (study.EVIDENCE / "intents/data.json").resolve():
        intent_evidence.verify_evidence(args.data.parent)
    data = json.loads(args.data.read_text())
    validate_data(data)
    args.output.mkdir(parents=True, exist_ok=False)
    with SingleSendClient(timeout_s=60, max_connections=1) as client:
        cloud = Cloud(
            client,
            args.output,
            max_score_items=20000,
            max_reflections=args.max_reflections,
            max_credits=20000,
            max_seconds=1800,
            batch_size=32,
            reflection_model=args.reflection_model,
        )
        try:
            result = experiment(data, cloud)
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
