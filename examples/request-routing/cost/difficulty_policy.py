"""Research recipe: learn a human-rated math difficulty routing policy on SIE.

The reference is Easy2Hard-AMC's human-performance-derived rating, not an LLM
opinion or a claim about which answering model is better. No solutions are sent
to the routers. Freeze all policies before scoring the official evaluation set.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
from sie_sdk.types import Item

import learn
import optimize
import study
from cloud import Cloud, digest, usage_of
from run import SingleSendClient

SOURCE = "furonghuang-lab/Easy2Hard-Bench"
REVISION = "55bc0d2fb10954151e669d2026b87fa896f2fa26"
MODEL = "Qwen/Qwen3-Embedding-4B"
EMBED_INSTRUCTION = "Represent the difficulty of the mathematics problem for classifying it as routine or advanced."
INSTRUCTION = (
    "Judge whether the Document is a mathematics problem matching the level of difficulty described in the Query."
)
SEED = {
    "routine": "A relatively routine competition mathematics problem that a well-prepared student can solve with familiar arithmetic, algebra, geometry, counting or probability techniques, without an unusually clever insight or long chain of deductions.",
    "advanced": "A challenging competition mathematics problem requiring a non-obvious insight, careful multi-step reasoning, an intricate construction, or advanced combinations of mathematical ideas.",
}
REFLECTION = """Improve one description in a two-class mathematics difficulty router.
The reference labels derive from human contest performance, not the teacher's
opinion: ROUTINE is the lower half of Easy2Hard-AMC difficulty ratings; ADVANCED
is the upper half. Both include competition math and both include every subject.
A Qwen relevance reranker compares each problem against both descriptions. The
difference in their log-odds is calibrated on training labels. Better ranking,
not changing score scale, helps. The reranker never sees solutions or metadata.
Current description:
<curr_param>
TRAINING ONLY: current pair of descriptions, aggregate errors and examples:
<side_info>
Write the replacement description for the named class. Learn observable patterns
from the feedback, including counterexamples. Do not assume long text or advanced
vocabulary necessarily makes a problem hard. Distinguish concept recognition
from the amount of insight and reasoning needed to solve it. Do not copy exact
problems, mention contest/dataset names, years, IDs or solutions. Do not answer
the problems. Keep the description coherent and at most 100 words.
Return only the replacement description inside a plain triple-backtick block.
"""


def normalize(text):
    return " ".join(text.casefold().split())


def prepare(source, destination):
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["repo"] != SOURCE or metadata["revision"] != REVISION:
        raise ValueError("unexpected source revision")
    source_hashes = {
        "train": "1686c7df11b60848f05142e5b2215a96a9ab6952b3867827542f2c4d92621baa",
        "eval": "2a04f50a80a16bfc5c1ff5667f9403e238184bbe7c325eff99f684d89f8f97fe",
    }
    # Read the exact pinned Parquet, not an unverified intermediate conversion.
    import pyarrow.parquet as pq

    pool, seen, excluded = {}, set(), 0
    for split in ("train", "eval"):
        path = source / f"{split}.parquet"
        if study.sha256(path.read_bytes()) != source_hashes[split]:
            raise ValueError("source Parquet changed")
        pool[split] = []
        for index, raw in enumerate(pq.read_table(path).to_pylist()):
            text = raw["problem"].strip()
            if normalize(text) in seen:
                excluded += 1
                continue
            seen.add(normalize(text))
            pool[split].append(
                {
                    "id": f"e2h-amc-{split}-{index}",
                    "request": text,
                    "label": int(raw["rating_quantile"] >= 0.5),
                    "rating_quantile": raw["rating_quantile"],
                    "source_tag": raw["tag"],
                }
            )
        pool[split].sort(key=lambda r: digest(["difficulty-policy-v1", r["id"]]))

    def take(rows, start, count):
        by_label = [[r for r in rows if r["label"] == label][start : start + count // 2] for label in (0, 1)]
        if any(len(group) != count // 2 for group in by_label):
            raise ValueError("not enough rows for the predeclared balanced sample")
        return [r for pair in zip(*by_label, strict=True) for r in pair]

    data = {
        "source": SOURCE,
        "source_revision": REVISION,
        "source_parquet_sha256": source_hashes,
        "license": "CC-BY-SA-4.0",
        "attribution": "Ding et al., Easy2Hard-Bench, NeurIPS 2024",
        "label_rule": "0=routine for rating_quantile<0.5, 1=advanced otherwise; retain the full difficulty range",
        "sampling": "Hash order within each class; balanced 512 train,128 validation from official train,512 test from official eval. No performance-based sampling.",
        "excluded_duplicate_texts": excluded,
        "splits": {
            "train": take(pool["train"], 0, 512),
            "validation": take(pool["train"], 256, 128),
            "test": take(pool["eval"], 0, 512),
        },
    }
    validate(data)
    if destination.exists():
        raise FileExistsError(destination)
    learn.write(destination, data)
    return data


def validate(data):
    ids, texts = set(), set()
    if set(data["splits"]) != {"train", "validation", "test"}:
        raise ValueError("need three splits")
    for rows in data["splits"].values():
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError("each split needs both classes")
        for row in rows:
            if not row["request"].strip() or row["id"] in ids or normalize(row["request"]) in texts:
                raise ValueError("empty or duplicate request across splits")
            ids.add(row["id"])
            texts.add(normalize(row["request"]))


def target(rows):
    return np.array([r["label"] for r in rows])


def threshold(values, y):
    # Balanced accuracy; tied scores always travel together. Conservative tie
    # break prefers sending a request to advanced handling.
    choices = np.unique(values)
    candidates = np.r_[choices, np.nextafter(choices[-1], np.inf)]
    predictions = values[:, None] >= candidates
    accuracy = 0.5 * (predictions[y == 1].mean(axis=0) + (~predictions[y == 0]).mean(axis=0))
    return float(candidates[np.argmax(accuracy)])


def metrics(values, y, tau):
    p = values >= tau
    return {
        "n": len(y),
        "accuracy": float((p == y).mean()),
        "balanced_accuracy": float(0.5 * (p[y == 1].mean() + (~p[y == 0]).mean())),
        "advanced_recall": float(p[y == 1].mean()),
        "routine_recall": float((~p[y == 0]).mean()),
        "false_routine": int((~p & (y == 1)).sum()),
        "false_advanced": int((p & (y == 0)).sum()),
    }


def collect_vectors(rows, directory, client):
    directory.mkdir(exist_ok=False)
    metadata = client.get_model(MODEL)
    learn.write(
        directory / "plan.json",
        {"ids": [r["id"] for r in rows], "model": MODEL, "metadata": metadata, "instruction": EMBED_INSTRUCTION},
    )
    cloud = Cloud(client, directory, max_credits=3000, max_seconds=600)
    try:
        with gzip.open(directory / "vectors.jsonl.gz", "xt") as handle:
            for offset in range(0, len(rows), 16):
                batch = rows[offset : offset + 16]

                def send(batch=batch):
                    reply = client.encode(
                        MODEL,
                        [Item(id=r["id"], text=r["request"]) for r in batch],
                        output_types=["dense"],
                        instruction=EMBED_INSTRUCTION,
                        is_query=True,
                        wait_for_capacity=False,
                        max_oom_retries=0,
                        provision_timeout_s=60,
                    )
                    if len(reply) != len(batch) or {r["id"] for r in reply} != {r["id"] for r in batch}:
                        raise ValueError("embedding response IDs do not match")
                    if any(
                        usage_of(r) != usage_of(reply[0]) or r["request"]["id"] != reply[0]["request"]["id"]
                        for r in reply
                    ):
                        raise ValueError("embedding batch receipts differ")
                    return {**reply[0], "items": reply}

                reply = cloud.call("embedding", send, {"ids": [r["id"] for r in batch], "model": MODEL})
                for row in reply["items"]:
                    vector = np.asarray(row["dense"])
                    if (
                        vector.shape != (metadata["dims"]["dense"],)
                        or not np.isfinite(vector).all()
                        or np.linalg.norm(vector) == 0
                    ):
                        raise ValueError("invalid embedding")
                    handle.write(json.dumps({"id": row["id"], "vector": vector.tolist()}) + "\n")
                handle.flush()
                print(f"Encoded {offset + len(batch)}/{len(rows)}; {cloud.credits:g} credits", flush=True)
        learn.write(directory / "collection.json", cloud.summary())
    except Exception as error:
        learn.write(directory / "incomplete.json", {"error_type": type(error).__name__, "cloud": cloud.summary()})
        raise
    finally:
        cloud.close()


def vectors(rows, directory):
    if (directory / "incomplete.json").exists():
        raise ValueError("embedding collection incomplete")
    plan = json.loads((directory / "plan.json").read_text())
    if plan["model"] != MODEL or plan["instruction"] != EMBED_INSTRUCTION:
        raise ValueError("embedding configuration changed")
    records = study.read_jsonl(directory / "vectors.jsonl.gz")
    by_id = {r["id"]: r["vector"] for r in records}
    if len(records) != len(by_id) or set(by_id) != set(plan["ids"]):
        raise ValueError("embedding rows missing or duplicated")
    return study.normalise(np.array([by_id[r["id"]] for r in rows]))


def knn(x, y, queries, *, loo=False, k=5):
    similarities = queries @ x.T
    if loo:
        if len(queries) != len(x):
            raise ValueError("LOO requires bank alignment")
        np.fill_diagonal(similarities, -np.inf)
    return np.array([study.knn_score(row, y, min(k, len(y) - int(loo))) for row in similarities])


def fit_embeddings(data, directory):
    train, val = (data["splits"][s] for s in ("train", "validation"))
    x, v = (vectors(rows, directory / "dev-embeddings") for rows in (train, val))
    y, vy = target(train), target(val)
    policies, validation = {}, {}
    for count in (64, 128, 256, 512):
        for method in ("knn", "ridge"):
            name = f"{method}{count}"
            if method == "knn":
                train_values = knn(x[:count], y[:count], x[:count], loo=True)
                val_values = knn(x[:count], y[:count], v)
                tau = threshold(train_values, y[:count])
                policy = {"method": method, "count": count, "k": 5, "tau": tau}
            else:
                fitted = [learn.fit(x[:count], y[:count], p) for p in learn.PENALTIES]
                # Select regularization by exact training LOO error, not validation.
                chosen = int(np.argmin([np.mean((f[2] - y[:count]) ** 2) for f in fitted]))
                w, b, train_values = fitted[chosen]
                val_values = v @ w + b
                tau = threshold(train_values, y[:count])
                policy = {
                    "method": method,
                    "count": count,
                    "penalty": learn.PENALTIES[chosen],
                    "weights": w.tolist(),
                    "bias": b,
                    "tau": tau,
                }
            policies[name] = policy
            validation[name] = metrics(val_values, vy, tau)
    # Cheap sanity control: difficulty is often correlated with length. Record
    # its explanatory power rather than crediting that signal to embeddings.
    lengths = np.array([len(r["request"].split()) for r in train])
    tau = threshold(lengths, y)
    policies["length512"] = {"method": "length", "count": len(train), "tau": tau}
    validation["length512"] = metrics(np.array([len(r["request"].split()) for r in val]), vy, tau)
    learn.write(
        directory / "embedding-selection.json",
        {
            "policies": policies,
            "validation": validation,
            "selection": "All predeclared sizes and methods retained; primary is knn64 vs ridge512, with knn512 and ridge64 ablations.",
        },
    )
    return validation


def logit(values):
    p = np.clip(np.asarray(values), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


class Adapter:
    propose_new_texts = None

    def __init__(self, cloud, train):
        self.cloud, self.train = cloud, train
        self.candidates, self.thresholds = {}, {}

    def values(self, rows, candidate):
        scored = [
            self.cloud.score(
                {"description": candidate[label], "instruction": candidate.get("instruction", INSTRUCTION)}, rows
            )[0]
            for label in ("routine", "advanced")
        ]
        return logit(scored[1]) - logit(scored[0])

    def tau(self, candidate):
        key = digest(candidate)
        if key not in self.thresholds:
            self.thresholds[key] = threshold(self.values(self.train, candidate), target(self.train))
        return self.thresholds[key]

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        self.candidates[digest(candidate)] = dict(candidate)
        learn.write(self.cloud.directory / "candidates.json", self.candidates)
        values, y = self.values(batch, candidate), target(batch)
        tau = self.tau(candidate)
        predictions = values >= tau
        correct = predictions == y
        scores = correct.astype(float) + 0.05 * np.tanh((2 * y - 1) * (values - tau))
        outputs = [
            {"id": r["id"], "predicted": int(p), "expected": r["label"], "correct": bool(c), "score": float(s)}
            for r, p, c, s in zip(batch, predictions, correct, values, strict=True)
        ]
        return EvaluationBatch(
            outputs=outputs, scores=scores.tolist(), trajectories=outputs if capture_traces else None
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        values, y = self.values(self.train, candidate), target(self.train)
        tau = self.tau(candidate)
        wrong = (values >= tau) != y
        rows = []
        for label in (0, 1):
            errors = [i for i in range(len(y)) if wrong[i] and y[i] == label]
            errors.sort(key=lambda i: -abs(values[i] - tau))
            correct = [i for i in range(len(y)) if not wrong[i] and y[i] == label]
            offset = (self.cloud.reflections * 3) % max(1, len(correct))
            chosen = errors[:6] + (correct[offset:] + correct[:offset])[:3]
            rows.extend(
                {
                    "request": self.train[i]["request"][:1800],
                    "truncated": len(self.train[i]["request"]) > 1800,
                    "expected": "advanced" if y[i] else "routine",
                    "misclassified": bool(wrong[i]),
                }
                for i in chosen
            )
        return {
            component: [
                {"class_to_describe": component, "descriptions": candidate, "training_metrics": metrics(values, y, tau)}
            ]
            + rows
            for component in components_to_update
        }


def optimize_prompts(data, directory, client, reflections):
    import gepa

    dest = directory / "reranker"
    dest.mkdir(exist_ok=False)
    cloud = Cloud(
        client,
        dest,
        max_reflections=reflections,
        max_score_items=30000,
        max_credits=20000,
        max_seconds=1800,
        batch_size=32,
    )
    train, val = (data["splits"][s] for s in ("train", "validation"))
    adapter = Adapter(cloud, train)
    learn.write(
        dest / "plan.json",
        {
            "data_sha256": digest(data),
            "seed": SEED,
            "instruction": INSTRUCTION,
            "reflection_template": REFLECTION,
            "reflector": cloud.reflection_model,
            "max_reflections": reflections,
            "max_credits": cloud.max_credits,
            "max_score_items": cloud.max_score_items,
        },
    )
    try:
        gepa.optimize(
            seed_candidate=SEED,
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_lm=cloud.reflect,
            reflection_prompt_template=REFLECTION,
            reflection_minibatch_size=24,
            skip_perfect_score=False,
            use_merge=False,
            max_metric_calls=10000,
            stop_callbacks=lambda state: cloud.failed or cloud.reflections >= reflections,
            run_dir=str(dest / "gepa"),
            logger=optimize.RunLogger(dest / "optimization.log"),
            seed=83,
            raise_on_exception=True,
            display_progress_bar=False,
            track_best_outputs=False,
        )
        candidates = list(adapter.candidates.values())
        validation = [metrics(adapter.values(val, c), target(val), adapter.tau(c)) for c in candidates]
        chosen = min(
            range(len(candidates)),
            key=lambda i: (-validation[i]["balanced_accuracy"], sum(map(len, candidates[i].values())), i),
        )
        if validation[chosen]["balanced_accuracy"] <= validation[0]["balanced_accuracy"]:
            chosen = 0
        initial_tau = threshold(adapter.values(train[:64], SEED), target(train[:64]))
        selection = {
            "candidates": candidates,
            "validation": validation,
            "chosen": chosen,
            "policies": {
                "static64": {"candidate": SEED, "tau": initial_tau},
                "calibrated512": {"candidate": SEED, "tau": adapter.tau(SEED)},
                "gepa512": {"candidate": candidates[chosen], "tau": adapter.tau(candidates[chosen])},
            },
            "cloud": cloud.summary(),
        }
        learn.write(directory / "reranker-selection.json", selection)
        return {"validation": validation, "chosen": chosen, "cloud": cloud.summary()}
    except Exception as error:
        learn.write(dest / "incomplete.json", {"error_type": type(error).__name__, "cloud": cloud.summary()})
        raise
    finally:
        cloud.close()


class RubricAdapter(Adapter):
    """Keep the class descriptions fixed; evolve their shared decision rule."""

    def values(self, rows, candidate):
        return super().values(rows, {**SEED, **candidate})

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        feedback = super().make_reflective_dataset(candidate, eval_batch, components_to_update)
        feedback["instruction"][0] = {
            "task": "Improve the shared matching instruction; preserve both class meanings",
            "fixed_class_descriptions": SEED,
            "training_metrics": feedback["instruction"][0]["training_metrics"],
        }
        return feedback


RUBRIC_REFLECTION = """Improve the shared instruction for a mathematics difficulty router.
The Query describes either ROUTINE or ADVANCED difficulty; the Document is the
math problem. The relevance reranker should say yes exactly when the problem
matches the Query's difficulty category. Its two scores are compared.
ROUTINE means the lower half and ADVANCED the upper half of independently
estimated human contest difficulty. Both classes include every math subject.
Class descriptions stay fixed. Write a better decision rubric in the instruction.
Current instruction:
<curr_param>
Training errors, successful examples, and fixed descriptions:
<side_info>
Use these examples to identify transferable reasoning patterns and distinctions.
Preserve polarity: a YES should mean matches the difficulty named in the Query.
Do not describe only one class, swap their meanings, ask to solve the problem,
or include answers, IDs, contest names or copied problems. Prefer concise
operational distinctions over vague repetition of 'complex' or 'non-obvious'.
Return only the replacement instruction, at most 160 words, in a plain ``` block.
"""


def optimize_rubric(data, directory, client, reflections):
    import gepa

    old = json.loads((directory / "reranker-selection.json").read_text())
    previous_plan = json.loads((directory / "reranker/plan.json").read_text())
    if previous_plan["data_sha256"] != digest(data):
        raise ValueError("cached baseline belongs to different data")
    if (directory / "reranker-selection-v1.json").exists():
        raise FileExistsError("rubric follow-up already started")
    learn.write(directory / "reranker-selection-v1.json", old)
    dest = directory / "reranker-rubric"
    dest.mkdir(exist_ok=False)
    cloud = Cloud(
        client,
        dest,
        max_reflections=reflections,
        max_score_items=30000,
        max_credits=20000,
        max_seconds=1800,
        batch_size=32,
    )
    source_scores = directory / "reranker/scores.jsonl"
    for row in study.read_jsonl(source_scores):
        cloud.cache[row["candidate"], row["id"]] = (row["score"], row["router_usd"])
        cloud.scores_file.write(json.dumps(row) + "\n")
    cloud.scores_file.flush()
    train, val = (data["splits"][s] for s in ("train", "validation"))
    adapter = RubricAdapter(cloud, train)
    learn.write(
        dest / "plan.json",
        {
            "data_sha256": digest(data),
            "seed": {"instruction": INSTRUCTION},
            "fixed_descriptions": SEED,
            "reflection_template": RUBRIC_REFLECTION,
            "max_reflections": reflections,
            "max_credits": cloud.max_credits,
            "cached_scores_sha256": study.sha256(source_scores.read_bytes()),
            "reason": "Development follow-up: description edits sometimes reversed class meanings. Test remains reserved.",
        },
    )
    try:
        gepa.optimize(
            seed_candidate={"instruction": INSTRUCTION},
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_lm=cloud.reflect,
            reflection_prompt_template=RUBRIC_REFLECTION,
            reflection_minibatch_size=32,
            skip_perfect_score=False,
            use_merge=False,
            max_metric_calls=10000,
            stop_callbacks=lambda state: cloud.failed or cloud.reflections >= reflections,
            run_dir=str(dest / "gepa"),
            logger=optimize.RunLogger(dest / "optimization.log"),
            seed=97,
            raise_on_exception=True,
            display_progress_bar=False,
            track_best_outputs=False,
        )
        candidates = list(adapter.candidates.values())
        validation = [metrics(adapter.values(val, c), target(val), adapter.tau(c)) for c in candidates]
        chosen = min(
            range(len(candidates)), key=lambda i: (-validation[i]["accuracy"], len(candidates[i]["instruction"]), i)
        )
        learn.write(
            directory / "rubric-selection.json",
            {"candidates": candidates, "validation": validation, "chosen": chosen, "cloud": cloud.summary()},
        )
        selected = {"candidate": {**SEED, **candidates[chosen]}, "tau": adapter.tau(candidates[chosen])}
        promoted = validation[chosen]["accuracy"] > old["validation"][old["chosen"]]["accuracy"]
        combined = {
            **old,
            "rubric_followup": {
                "promoted": promoted,
                "chosen": chosen,
                "validation": validation,
                "cloud": cloud.summary(),
            },
            "policies": {**old["policies"], "gepa512": selected if promoted else old["policies"]["gepa512"]},
        }
        learn.write(directory / "reranker-selection.json", combined)
        return combined["rubric_followup"]
    except Exception as error:
        learn.write(dest / "incomplete.json", {"error_type": type(error).__name__, "cloud": cloud.summary()})
        raise
    finally:
        cloud.close()


def freeze(data, directory):
    if (directory / "frozen.json").exists():
        raise FileExistsError("policies already frozen")
    selection = {
        name: json.loads((directory / f"{name}-selection.json").read_text()) for name in ("embedding", "reranker")
    }
    if (directory / "teacher-selection.json").exists():
        selection["teacher"] = json.loads((directory / "teacher-selection.json").read_text())
    plan = {
        "data_sha256": digest(data),
        "selection": selection,
        "embedding_recording_sha256": study.sha256((directory / "dev-embeddings/vectors.jsonl.gz").read_bytes()),
        "primary_comparisons": ["knn64 -> ridge512", "static64 -> gepa512"],
        "controls": ["knn64 -> knn512", "ridge64 -> ridge512", "calibrated512 -> gepa512"],
        "success_gate": "Each primary reduces errors by at least 25% and has a positive paired lower 95% accuracy-gain bound. GEPA must also beat recalibration by at least 3 percentage points.",
        "limits": "Balanced math difficulty policy, not model-benefit prediction, answer correctness or dollar savings. Same-domain public historical problems may have appeared in model pretraining.",
        "teacher_followup_gate": "If present, teacher GEPA must reduce errors by at least 25% versus the fixed teacher prompt, with a positive paired lower 95% accuracy-gain bound. Evaluated separately from the original reranker gate.",
    }
    learn.write(directory / "frozen.json", plan)
    return {"frozen_sha256": digest(plan), "status": "no test inference or metrics yet"}


def paired(y, before, after):
    a, b = before == y, after == y
    rng = np.random.default_rng(89)
    sample = np.concatenate(
        [rng.choice(np.where(y == label)[0], (4000, int((y == label).sum()))) for label in (0, 1)], axis=1
    )
    changes = 100 * (b[sample].mean(axis=1) - a[sample].mean(axis=1))
    fixed, broken = int((~a & b).sum()), int((a & ~b).sum())
    discordant = fixed + broken
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(fixed, broken) + 1)) / 2**discordant)
    return {
        "gain_pp": float(100 * (b.mean() - a.mean())),
        "gain_pp_95ci": np.quantile(changes, [0.025, 0.975]).tolist(),
        "relative_error_reduction_pct": float(100 * (1 - (~b).sum() / (~a).sum())) if (~a).any() else None,
        "corrected": fixed,
        "regressed": broken,
        "mcnemar_exact_p": p,
        "bootstrap": "4000 paired label-stratified resamples, seed 89",
    }


class Recorded:
    def __init__(self, directory):
        self.rows = {(r["candidate"], r["id"]): r for r in study.read_jsonl(directory / "scores.jsonl")}

    def score(self, candidate, items):
        records = [self.rows[digest(candidate), row["id"]] for row in items]
        return [r["score"] for r in records], [r["router_usd"] for r in records]


def test_stage(data, directory, client=None):
    plan = json.loads((directory / "frozen.json").read_text())
    if digest(data) != plan["data_sha256"]:
        raise ValueError("dataset changed after freeze")
    if study.sha256((directory / "dev-embeddings/vectors.jsonl.gz").read_bytes()) != plan["embedding_recording_sha256"]:
        raise ValueError("training vectors changed after freeze")
    rows, train = data["splits"]["test"], data["splits"]["train"]
    if client is not None:
        collect_vectors(rows, directory / "test-embeddings", client)
    dev_plan = json.loads((directory / "dev-embeddings/plan.json").read_text())
    test_plan = json.loads((directory / "test-embeddings/plan.json").read_text())
    if dev_plan["metadata"]["revision"] != test_plan["metadata"]["revision"]:
        raise ValueError("encoder revision changed")
    x, tx = vectors(train, directory / "dev-embeddings"), vectors(rows, directory / "test-embeddings")
    y, ty = target(train), target(rows)
    results, predictions = {}, {}
    for name, policy in plan["selection"]["embedding"]["policies"].items():
        if policy["method"] == "knn":
            values = knn(x[: policy["count"]], y[: policy["count"]], tx, k=policy["k"])
        elif policy["method"] == "ridge":
            values = tx @ np.array(policy["weights"]) + policy["bias"]
        else:
            values = np.array([len(r["request"].split()) for r in rows])
        results[name] = metrics(values, ty, policy["tau"])
        predictions[name] = (values >= policy["tau"]).astype(int)
    dest = directory / "test-reranker"
    if client is not None:
        dest.mkdir(exist_ok=False)
        cloud = Cloud(client, dest, max_score_items=4096, max_credits=5000, max_seconds=600, batch_size=32)
    else:
        if (dest / "incomplete.json").exists():
            raise ValueError("test score collection incomplete")
        cloud = Recorded(dest)
    adapter = Adapter(cloud, train)
    try:
        for name, policy in plan["selection"]["reranker"]["policies"].items():
            values = adapter.values(rows, policy["candidate"])
            results[name] = metrics(values, ty, policy["tau"])
            predictions[name] = (values >= policy["tau"]).astype(int)
        if client is not None:
            learn.write(dest / "collection.json", cloud.summary())
    except Exception as error:
        if client is not None:
            learn.write(dest / "incomplete.json", {"error_type": type(error).__name__, "cloud": cloud.summary()})
        raise
    finally:
        if client is not None:
            cloud.close()
    comparisons = {
        f"{a}_to_{b}": paired(ty, predictions[a], predictions[b])
        for a, b in (
            ("knn64", "ridge512"),
            ("knn64", "knn512"),
            ("ridge64", "ridge512"),
            ("static64", "gepa512"),
            ("calibrated512", "gepa512"),
        )
    }
    passed = all(
        comparisons[name]["relative_error_reduction_pct"] is not None
        and comparisons[name]["relative_error_reduction_pct"] >= 25
        and comparisons[name]["gain_pp_95ci"][0] > 0
        for name in ("knn64_to_ridge512", "static64_to_gepa512")
    )
    passed &= comparisons["calibrated512_to_gepa512"]["gain_pp"] >= 3
    result = {
        "test": results,
        "comparisons": comparisons,
        "success_gate_passed": bool(passed),
        "frozen_sha256": digest(plan),
    }
    prediction_records = {name: values.tolist() for name, values in predictions.items()}
    if client is None:
        if json.loads((directory / "results.json").read_text()) != result:
            raise ValueError("replay differs from saved metrics")
        if json.loads((directory / "predictions.json").read_text()) != prediction_records:
            raise ValueError("replay differs from saved routes")
    else:
        learn.write(directory / "results.json", result)
        learn.write(directory / "predictions.json", prediction_records)
    return result


def route(data, directory, request=None, recorded_id=None, client=None):
    frozen = json.loads((directory / "frozen.json").read_text())
    if digest(data) != frozen["data_sha256"]:
        raise ValueError("dataset changed after freeze")
    chosen = frozen["selection"]["embedding"]["policies"]["ridge512"]
    if recorded_id is not None:
        matches = [(split, r) for split, rows in data["splits"].items() for r in rows if r["id"] == recorded_id]
        if len(matches) != 1:
            raise ValueError("recorded ID missing or duplicated")
        split, item = matches[0]
        vector = vectors([item], directory / ("test-embeddings" if split == "test" else "dev-embeddings"))[0]
        receipt = {"mode": "recorded", "new_inference_calls": 0}
        request = item["request"]
    else:
        metadata = client.get_model(MODEL)
        expected = json.loads((directory / "dev-embeddings/plan.json").read_text())
        if metadata["revision"] != expected["metadata"]["revision"]:
            raise ValueError("live encoder revision differs from the trained classifier")
        reply = client.encode(
            MODEL,
            Item(text=request),
            output_types=["dense"],
            instruction=EMBED_INSTRUCTION,
            is_query=True,
            wait_for_capacity=False,
            max_oom_retries=0,
            provision_timeout_s=60,
        )
        vector = study.normalise(np.asarray(reply["dense"]))
        receipt = {"mode": "live", **usage_of(reply)}
    weights = np.asarray(chosen["weights"])
    if vector.shape != weights.shape:
        raise ValueError("embedding dimension differs from the trained classifier")
    score = float(vector @ weights + chosen["bias"])
    return {
        "request": request,
        "route": "advanced" if score >= chosen["tau"] else "routine",
        "score": score,
        "threshold": chosen["tau"],
        "score_note": "Uncalibrated classifier score, not a probability",
        "encoder": MODEL,
        "classifier": "ridge512",
        "receipt": receipt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "embed", "fit", "optimize", "rubric", "freeze", "test", "replay", "route")
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--reflections", type=int, default=12)
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--request")
    query.add_argument("--recorded-id")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.data)
        print({k: len(v) for k, v in result["splits"].items()})
        return
    data = json.loads(args.data.read_text())
    validate(data)
    args.run.mkdir(exist_ok=True, parents=True)
    if args.command == "fit":
        result = fit_embeddings(data, args.run)
    elif args.command == "freeze":
        result = freeze(data, args.run)
    elif args.command == "replay":
        result = test_stage(data, args.run)
    elif args.command == "route" and args.recorded_id:
        result = route(data, args.run, recorded_id=args.recorded_id)
    else:
        with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
            if args.command == "embed":
                collect_vectors(
                    data["splits"]["train"] + data["splits"]["validation"], args.run / "dev-embeddings", client
                )
                result = fit_embeddings(data, args.run)
            elif args.command == "optimize":
                result = optimize_prompts(data, args.run, client, args.reflections)
            elif args.command == "rubric":
                result = optimize_rubric(data, args.run, client, args.reflections)
            elif args.command == "route":
                if not args.request:
                    parser.error("route needs --request or --recorded-id")
                result = route(data, args.run, request=args.request, client=client)
            else:
                result = test_stage(data, args.run, client)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
