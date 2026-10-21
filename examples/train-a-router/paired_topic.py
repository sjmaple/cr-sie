"""Fixed training-only text-to-topic control; source labels are a separate diagnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import paired_training as core

PROTOCOL = {
    "sklearn": "1.6.1",
    "features": "TF-IDF word unigrams/bigrams, min_df2, max_features30000",
    "classifier": "MultinomialNB alpha1",
    "smoothing_rows": 20,
    "topic": "source:task (MMLU subject, BBH task, KOR category); never rule/family IDs",
    "outcome_cost": "Training-only topic arm means with20 global-mean pseudocounts",
}


def topic(row):
    return row["source"] + ":" + row["task"]


def statistics(rows, y, costs):
    y, costs = np.asarray(y, dtype=float), np.asarray(costs, dtype=float)
    core.require(
        len(rows) > 0
        and y.shape == costs.shape == (len(rows), 2)
        and all(row["role"] == "train" for row in rows)
        and np.isin(y, (0, 1)).all()
        and np.isfinite(costs).all()
        and (costs >= 0).all(),
        "Topic means require finite training rows only",
    )
    names = sorted({topic(row) for row in rows})
    global_y, global_cost = y.mean(axis=0), costs.mean(axis=0)
    classes = []
    for name in names:
        mask = np.array([topic(row) == name for row in rows])
        n = int(mask.sum())
        classes.append(
            {
                "topic": name,
                "n": n,
                "probabilities": ((y[mask].sum(axis=0) + 20 * global_y) / (n + 20)).tolist(),
                "expected_usd": ((costs[mask].sum(axis=0) + 20 * global_cost) / (n + 20)).tolist(),
            }
        )
    return {"classes": classes, "global_probabilities": global_y.tolist(), "global_expected_usd": global_cost.tolist()}


def fit_and_score(run, grades_path, output):
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    core.require(sklearn.__version__ == PROTOCOL["sklearn"], "Use pinned topic classifier version")
    run = Path(run)
    core.require(not (run / "final-access.json").exists(), "Final outcomes already accessed")
    plan, all_rows = core.checked_plan(run)
    rows, y, cost = core.grade_arrays(plan, all_rows, core.read(grades_path), "train")
    means = statistics(rows, y, cost)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    matrix = vectorizer.fit_transform([row["request"] for row in rows])
    classifier = MultinomialNB(alpha=1.0).fit(matrix, [topic(row) for row in rows])
    core.require(
        classifier.classes_.tolist() == [item["topic"] for item in means["classes"]], "Topic class order changed"
    )
    model = {
        "format": "routerbench-text-topic-control-v1",
        "protocol": PROTOCOL,
        "plan_sha256": core.sha(run / "plan.json"),
        "source_sha256": core.sha(__file__),
        "training_grades_sha256": core.sha(grades_path),
        "training_ids": [row["id"] for row in rows],
        "vocabulary": {name: int(index) for name, index in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(),
        "classes": classifier.classes_.tolist(),
        "feature_log_prob": classifier.feature_log_prob_.tolist(),
        "class_log_prior": classifier.class_log_prior_.tolist(),
        "statistics": means,
    }
    core.write(run / "topic-model.json", model)
    probability_means = np.array([item["probabilities"] for item in means["classes"]])
    cost_means = np.array([item["expected_usd"] for item in means["classes"]])
    by_topic = {item["topic"]: item for item in means["classes"]}
    result = {
        "format": "routerbench-input-only-topic-v1",
        "plan_sha256": core.sha(run / "plan.json"),
        "model_sha256": core.sha(run / "topic-model.json"),
        "outcomes_accessed": False,
        "roles": {},
    }
    for role in ("calibration", "final", "development"):
        selected = [row for row in all_rows if row["role"] == role]
        probabilities = classifier.predict_proba(vectorizer.transform([row["request"] for row in selected]))
        diagnostic = [
            by_topic.get(
                topic(row),
                {"probabilities": means["global_probabilities"], "expected_usd": means["global_expected_usd"]},
            )
            for row in selected
        ]
        result["roles"][role] = {
            "ids": [row["id"] for row in selected],
            "probabilities": (probabilities @ probability_means).tolist(),
            "expected_usd": (probabilities @ cost_means).tolist(),
            "source_topic_diagnostic": {
                "metadata_assisted_not_input_only": True,
                "probabilities": [item["probabilities"] for item in diagnostic],
                "expected_usd": [item["expected_usd"] for item in diagnostic],
                "unseen_topics": sum(topic(row) not in by_topic for row in selected),
            },
        }
    core.write(output, result)
    return result
