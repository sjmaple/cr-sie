"""Outcome-head math, grouped selection, price binding and lifecycle guards."""

import importlib.util
import io
import json
import math
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_benefit as fixtures

import benefit
import benefit_outcomes as outcomes
from data import digest, file_hash


def paired_rows(count=20):
    return [{"id": f"row-{i}", "cheap_correct": i % 4 < 2, "strong_correct": i % 3 != 0} for i in range(count)]


def fake_fit(matrix, targets, penalty):
    del penalty
    frequency = float(targets.mean())
    return {
        "weights": [0.0] * matrix.shape[1],
        "bias": math.log(frequency / (1 - frequency)),
        "optimizer": {
            "success": True,
            "iterations": 0,
            "function_evaluations": 1,
            "objective": 0.0,
            "gradient_absolute_max": 0.0,
            "message": "Synthetic test fit",
        },
    }


class OutcomeMathTests(unittest.TestCase):
    def test_signed_benefit_and_train_only_cost_mixture_include_rescues_and_floor(self):
        targets = [[0, 0], [0, 1], [1, 0], [1, 1]]
        costs = [[20, 80], [20, 40], [10, 80], [10, 40]]
        means = outcomes.conditional_costs(targets, costs)
        probabilities = np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
        scores, expected = outcomes.normalized_scores(probabilities, means)
        np.testing.assert_allclose(expected, [[12, 72], [18, 48], [15, 60]])
        np.testing.assert_allclose(scores, [-0.6 / 60, 0.6 / 30, 0])
        # A negative incremental predicted cost uses the predeclared1credit
        # floor; it never reverses the meaning of strong-minus-cheap benefit.
        for state in ("correct", "wrong"):
            means[f"cheap_{state}"]["mean_credits_proxy"] = 100
            means[f"strong_{state}"]["mean_credits_proxy"] = 1
        np.testing.assert_allclose(outcomes.normalized_scores(probabilities, means)[0], [-0.6, 0.6, 0])
        for bad in ([[1.1, 0]], [[np.nan, 0]], [[0, 1, 0]]):
            with self.assertRaises(ValueError):
                outcomes.normalized_scores(bad, means)

    def test_cv_uses_only_other_papers_for_both_heads_and_ties_prefer_larger_penalty(self):
        rows = paired_rows()
        matrix = np.array([[i, i % 3] for i in range(20)], dtype=float)
        groups = [f"paper-{i // 4}" for i in range(20)]
        folds = np.array(benefit.group_folds(groups))
        targets = np.array([[row["cheap_correct"], row["strong_correct"]] for row in rows], dtype=float)
        calls = []

        def traced(x, y, penalty):
            calls.append((x.copy(), y.copy(), penalty))
            return fake_fit(x, y, penalty)

        with patch.object(outcomes, "fit_logistic", side_effect=traced):
            head = outcomes.fit_head(matrix, rows, groups, [[10, 30]] * 20)
        self.assertEqual(len(calls), 42)
        self.assertEqual(head["penalty"], 10)
        for index, penalty in enumerate(outcomes.PROTOCOL["penalties"]):
            for fold in range(5):
                for arm in range(2):
                    x, y, observed_penalty = calls[index * 10 + fold * 2 + arm]
                    np.testing.assert_array_equal(x, matrix[folds != fold])
                    np.testing.assert_array_equal(y, targets[folds != fold, arm])
                    self.assertEqual(observed_penalty, penalty)
                    self.assertFalse(
                        {groups[int(row[0])] for row in x} & {groups[i] for i in np.flatnonzero(folds == fold)}
                    )
        for arm, (x, y, penalty) in enumerate(calls[-2:]):
            np.testing.assert_array_equal(x, matrix)
            np.testing.assert_array_equal(y, targets[:, arm])
            self.assertEqual(penalty, head["penalty"])
        for candidate in head["cv"]:
            p = np.array(candidate["oof_probabilities"])
            expected = float((-targets * np.log(p) - (1 - targets) * np.log1p(-p)).mean(axis=0).sum())
            self.assertEqual(candidate["summed_binary_logloss"], expected)

    def test_bad_labels_features_and_cost_classes_fail_before_optimization(self):
        rows, groups = paired_rows(), [f"paper-{i // 4}" for i in range(20)]
        with patch.object(outcomes, "fit_logistic") as fitting:
            with self.assertRaisesRegex(ValueError, "align"):
                outcomes.fit_head(np.full((20, 2), np.nan), rows, groups, [[10, 30]] * 20)
            rows[0]["cheap_correct"] = 1
            with self.assertRaises(TypeError):
                outcomes.fit_head(np.zeros((20, 2)), rows, groups, [[10, 30]] * 20)
            fitting.assert_not_called()
        with self.assertRaisesRegex(ValueError, "correct and incorrect"):
            outcomes.conditional_costs([[1, 1], [0, 1]], [[2, 4], [3, 5]])

    @unittest.skipUnless(
        importlib.util.find_spec("scipy") is not None, "Optional SciPy fitting dependency not installed"
    )
    def test_real_optimizer_recovers_intercept_and_informative_logistic_direction(self):
        x = np.zeros((20, 2))
        y = np.array([0] * 5 + [1] * 15, dtype=float)
        model = outcomes.fit_logistic(x, y, 0.1)
        np.testing.assert_allclose(outcomes.probability(x, model), 0.75, atol=1e-8)
        x[:, 0] = np.linspace(-2, 2, 20)
        model = outcomes.fit_logistic(x, y, 0.1)
        self.assertGreater(model["weights"][0], 0)
        self.assertLess(outcomes._nll(y, outcomes.probability(x, model)), outcomes._nll(y, np.full(20, 0.75)))


class OutcomeLifecycleTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.BenefitLifecycleTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root, self.baseline, self.original = fixture.root, fixture.baseline, fixture.original
        self.training = fixture.training
        self.run = self.root / "outcome-head"
        v2 = {"outcome_grader_source_sha256": "b" * 64, "outcome_grader_protocol_sha256": "c" * 64}
        fixture.grading.update(v2)
        original = outcomes.read(self.original)
        original.pop("sha256")
        fixtures.save(self.original, benefit.seal(original | v2))
        fixtures.save(fixture.training_plan, outcomes.read(fixture.training_plan) | v2)
        fixtures.save(
            fixture.training_grades,
            outcomes.read(fixture.training_grades) | v2 | {"plan_sha256": file_hash(fixture.training_plan)},
        )
        protocol = patch.dict(
            outcomes.PROTOCOL,
            {"training_pool_sizes": [20, 30], "required_original_exclusions": ["train-0"], "dimensions": 3},
        )
        protocol.start()
        self.addCleanup(protocol.stop)
        self.rows_path = fixtures.save(self.training / "training-rows.json", fixture.dataset["splits"]["train"])
        grades = outcomes.read(fixture.training_grades)
        for index, row in enumerate(grades["rows"]):
            row["strong_correct"] = index % 3 != 0
            for arm in outcomes.ARMS:
                row[f"{arm}_record_sha256"] = digest([arm, row["id"]])
        grades["collection_summary_sha256"] = "d" * 64
        fixtures.save(fixture.training_grades, grades)
        rates = {
            "cheap": {
                "input_tokens": {"numerator": 1, "denominator": 10},
                "output_tokens": {"numerator": 1, "denominator": 5},
            },
            "strong": {
                "input_tokens": {"numerator": 3, "denominator": 10},
                "output_tokens": {"numerator": 1, "denominator": 2},
            },
        }
        price_rows = []
        for arm, model in zip(outcomes.ARMS, ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.8-27B-FP8"), strict=True):
            for unit, rate in rates[arm].items():
                price_rows.append(
                    {
                        "model": model,
                        "unit": unit,
                        "profile": "default",
                        "region": "us",
                        "usd_per_unit": {"numerator": rate["numerator"], "denominator": rate["denominator"] * 100000},
                    }
                )
        self.price_path = fixtures.save(
            self.root / "prices.json",
            {"currency": "usd", "pricing_version": "synthetic-v1", "rate_book_sha256": "f" * 64, "items": price_rows},
        )
        records, cost_rows = [], []
        for row in grades["rows"]:
            record = {"id": row["id"]}
            values = []
            for arm, initial_cost in zip(outcomes.ARMS, (2, 4), strict=True):
                calls = [{"prompt_tokens": 3, "completion_tokens": 5, "credits_proxy": initial_cost}]
                if not row[f"{arm}_correct"]:
                    calls.append({"prompt_tokens": 1, "completion_tokens": 1, "credits_proxy": 1})
                charge = sum(call["credits_proxy"] for call in calls)
                record.update(
                    {
                        f"{arm}_correct": row[f"{arm}_correct"],
                        f"{arm}_record_sha256": row[f"{arm}_record_sha256"],
                        f"{arm}_calls": calls,
                        f"{arm}_credits_proxy": charge,
                    }
                )
                values.append(charge)
            records.append(record)
            cost_rows.append(values)
        targets = [[int(row[f"{arm}_correct"]) for arm in outcomes.ARMS] for row in grades["rows"]]
        self.cost_path = fixtures.save(
            self.root / "cost-proxy.json",
            {
                "format": "sie-training-uncached-cloud-cost-proxy-v1",
                "split": "train",
                "n": len(records),
                "credit_usd": 1e-5,
                "training_plan_sha256": file_hash(fixture.training_plan),
                "training_grades_sha256": file_hash(fixture.training_grades),
                "training_collection_sha256": grades["collection_summary_sha256"],
                "price_source": {
                    "sha256": file_hash(self.price_path),
                    "public_rows": price_rows,
                    "pricing_version": "synthetic-v1",
                    "rate_book_sha256": "f" * 64,
                },
                "rates_credits_per_token": rates,
                "rows": records,
                "conditional_means": outcomes.conditional_costs(targets, cost_rows),
            },
        )
        fixtures.save(self.run / "plan.json", self.prepare())

    def prepare(self):
        fixture = self.fixture
        return outcomes.prepare(
            self.baseline,
            self.original,
            fixture.training_plan,
            fixture.training_inputs,
            self.rows_path,
            fixture.eligibility,
            attest_test_outcomes_unopened=True,
        )

    def fit(self):
        fixture = self.fixture
        with patch.object(outcomes, "fit_logistic", side_effect=fake_fit):
            selected = outcomes.fit(
                self.run,
                self.baseline,
                fixture.training_plan,
                fixture.training_inputs,
                self.rows_path,
                fixture.training_grades,
                self.baseline / "development-embeddings",
                self.cost_path,
                self.price_path,
            )
        fixtures.save(self.run / "selected-model.json", selected)
        return selected

    def freeze(self):
        self.fit()
        result = outcomes.freeze(self.run, self.baseline, self.original)
        fixtures.save(self.run / "frozen.json", result)
        return result

    def test_fit_cannot_open_calibration_or_test_and_freeze_requires_durable_selected_model(self):
        with self.assertRaises(FileNotFoundError):
            outcomes.freeze(self.run, self.baseline, self.original)
        read = outcomes.read
        original = self.original

        def training_only(path):
            self.assertNotEqual(Path(path), original)
            self.assertNotIn(Path(path).name, ("grades-calibration.json", "grades-test.json"))
            return read(path)

        with patch.object(outcomes, "read", side_effect=training_only):
            selected = self.fit()
        self.assertEqual(selected["head"]["training_ids"], [row["id"] for row in self.fixture.eligible_rows])
        self.assertFalse((self.run / "frozen.json").exists())
        frozen = outcomes.freeze(self.run, self.baseline, self.original)
        self.assertEqual(frozen["selected_model_sha256"], file_hash(self.run / "selected-model.json"))
        self.assertEqual(tuple(frozen["selections"]), ("1", "0", "2", "5"))

    def test_cost_proxy_recomputes_every_call_and_rejects_swapped_labels_prices_or_ids(self):
        fixture = self.fixture
        grades = outcomes.read(fixture.training_grades)
        original = outcomes.read(self.cost_path)
        costs, means, _ = outcomes.cost_proxy(
            self.cost_path, self.price_path, fixture.training_plan, fixture.training_grades, grades
        )
        self.assertEqual(means, original["conditional_means"])
        self.assertEqual(len(costs), 19)
        cases = []
        bad = json.loads(json.dumps(original))
        bad["rows"][0]["cheap_calls"][0]["credits_proxy"] += 1
        cases.append(bad)
        bad = json.loads(json.dumps(original))
        bad["rows"][0]["cheap_calls"][0]["prompt_tokens"] = True
        cases.append(bad)
        bad = json.loads(json.dumps(original))
        bad["rows"][0]["cheap_correct"] = not bad["rows"][0]["cheap_correct"]
        cases.append(bad)
        bad = json.loads(json.dumps(original))
        bad["rows"].reverse()
        cases.append(bad)
        bad = json.loads(json.dumps(original))
        bad["price_source"]["public_rows"][0]["usd_per_unit"]["numerator"] = 9
        cases.append(bad)
        for value in cases:
            with self.subTest(value=cases.index(value)):
                fixtures.save(self.cost_path, value)
                with self.assertRaises(ValueError):
                    outcomes.cost_proxy(
                        self.cost_path, self.price_path, fixture.training_plan, fixture.training_grades, grades
                    )

    def test_changed_plan_review_hashes_or_vectors_fail_before_optimizer(self):
        fixture = self.fixture
        grades = outcomes.read(fixture.training_grades)
        for key in outcomes.GRADE_HASHES:
            with self.subTest(field=key):
                fixtures.save(fixture.training_grades, grades | {key: "a" * 64})
                with patch.object(outcomes, "fit_head") as fitting:
                    with self.assertRaises(ValueError):
                        self.fit()
                    fitting.assert_not_called()
        fixtures.save(fixture.training_grades, grades)
        with (self.baseline / "development-embeddings/vectors.jsonl.gz").open("ab") as stream:
            stream.write(b"changed")
        with patch.object(outcomes, "fit_head") as fitting:
            with self.assertRaisesRegex(ValueError, "vectors changed"):
                self.fit()
            fitting.assert_not_called()

    def test_resealed_cv_objective_cost_means_and_normalized_score_tampering_fail(self):
        self.freeze()
        original_model = outcomes.read(self.run / "selected-model.json")
        original_frozen = outcomes.read(self.run / "frozen.json")
        for kind in ("objective", "mean", "penalty"):
            changed = json.loads(json.dumps(original_model))
            changed.pop("sha256")
            if kind == "objective":
                changed["head"]["cv"][0]["summed_binary_logloss"] = 0
            elif kind == "mean":
                changed["head"]["conditional_cost_statistics"]["cheap_correct"]["mean_credits_proxy"] += 1
            else:
                changed["head"]["penalty"] = 0.1
            fixtures.save(self.run / "selected-model.json", benefit.seal(changed))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                outcomes.checked_model(self.run)
        fixtures.save(self.run / "selected-model.json", original_model)
        changed = json.loads(json.dumps(original_frozen))
        changed.pop("sha256")
        changed["scores"]["test"][0] += 1
        fixtures.save(self.run / "frozen.json", benefit.seal(changed))
        with self.assertRaisesRegex(ValueError, "cost-normalized"):
            outcomes.authorize_test(self.run)

    def test_report_checks_access_chain_before_opening_test_and_never_reselects_on_test(self):
        frozen = self.freeze()
        test = self.root / "must-not-open.json"
        with self.assertRaises(FileNotFoundError):
            outcomes.report(self.run, test)
        fixtures.save(self.run / "test-access.json", outcomes.authorize_test(self.run))
        access = outcomes.read(self.run / "test-access.json")
        access.pop("sha256")
        fixtures.save(self.run / "test-access.json", benefit.seal(access | {"frozen_sha256": "a" * 64}))
        with self.assertRaisesRegex(ValueError, "Test-access"):
            outcomes.report(self.run, test)
        fixtures.save(self.run / "test-access.json", benefit.seal(access))
        rows = fixtures.outcomes("test", 8)
        for row in rows:
            row["cheap_correct"] = False
        fixtures.save(
            test,
            {
                "split": "test",
                "plan_sha256": file_hash(self.fixture.original_plan),
                "policies_frozen_sha256": file_hash(self.original),
                **self.fixture.grading,
                "rows": rows,
            },
        )
        selector = outcomes.solver_report.select_cutoff

        def only_calibration(rows, *args, **kwargs):
            self.assertTrue(all(row["id"].startswith("calibration-") for row in rows))
            return selector(rows, *args, **kwargs)

        with patch.object(outcomes.solver_report, "select_cutoff", side_effect=only_calibration):
            report = outcomes.report(self.run, test)
        self.assertEqual(report["ids"], [row["id"] for row in rows])
        self.assertEqual(report["policies"]["1"]["cutoff"], frozen["selections"]["1"]["cutoff"])
        self.assertEqual(report["training_rows"], 19)
        with self.assertRaises(FileExistsError):
            outcomes.fit(self.run, None, None, None, None, None, None, None, None)

    def test_new_source_only_review_explicitly_acknowledges_reused_outcomes(self):
        eligibility = outcomes.read(self.fixture.eligibility)
        eligibility.pop("decided_before_training_outcomes")
        eligibility.update(decided_before_new_training_outcomes=True, reused_training_outcomes_already_available=True)
        fixtures.save(self.fixture.eligibility, eligibility)
        plan = outcomes.read(self.fixture.training_plan)
        fixtures.save(self.fixture.training_plan, plan | {"eligibility_sha256": file_hash(self.fixture.eligibility)})
        revised = self.prepare()
        self.assertTrue(revised["eligibility"]["reused_training_outcomes_already_available"])
        eligibility["source_only"] = False
        fixtures.save(self.fixture.eligibility, eligibility)
        with self.assertRaisesRegex(ValueError, "Eligibility"):
            self.prepare()

    def test_post_collection_review_filters_training_and_held_rows_without_rewriting_collection(self):
        fixture = self.fixture
        source_review = outcomes.source_review
        excluded = {"train-1", "calibration-0", "test-0"}
        records = [
            {
                "id": row["id"],
                "split": split,
                "question": row["request"],
                "question_sha256": digest(row["request"]),
                "request_sha256": digest(row["request"]),
                "original_reference": "1",
                "reference": "1",
                "semantic_kind": "scalar",
                "decision": "exclude" if row["id"] in excluded else "retain",
                "reason": "Synthetic source-only missing-input review",
            }
            for split, rows in fixture.dataset["splits"].items()
            if split != "validation"
            for row in rows
        ]
        review_path = fixtures.save(
            self.root / "source-review.json",
            {
                "format": source_review.FORMAT,
                "protocol": source_review.PROTOCOL,
                "finalized": True,
                "source_revision": outcomes.SOURCE_REVISION,
                "source_hashes": outcomes.SOURCE_HASHES,
                "training_outcomes_already_available": True,
                "solver_test_outcomes_accessed": False,
                "records": records,
            },
        )
        metadata = source_review.metadata(review_path)
        original = outcomes.read(self.original)
        original.pop("sha256")
        original["calibration_rows"] = original["calibration_rows"][1:]
        original["embedding_calibration_credits"] = original["embedding_calibration_credits"][1:]
        fixtures.save(self.original, benefit.seal(original | metadata))
        original_collection_bytes = fixture.training_plan.read_bytes()
        plan = outcomes.prepare(
            self.baseline,
            self.original,
            fixture.training_plan,
            fixture.training_inputs,
            self.rows_path,
            fixture.eligibility,
            source_review_path=review_path,
            attest_test_outcomes_unopened=True,
        )
        self.assertEqual(fixture.training_plan.read_bytes(), original_collection_bytes)
        self.assertEqual(len(plan["source_populations"]["train"]), 19)
        self.assertEqual(len(plan["populations"]["train"]), 18)
        self.assertEqual(outcomes.checked_plan(plan), plan)
        graded = outcomes.read(fixture.training_grades)
        graded.update(
            metadata, collected_ids=[row["id"] for row in plan["source_populations"]["train"]], excluded_ids=["train-1"]
        )
        graded["rows"] = [row for row in graded["rows"] if row["id"] != "train-1"]
        fixtures.save(fixture.training_grades, graded)
        rows, _ = outcomes._training(
            plan, fixture.training_plan, fixture.training_inputs, self.rows_path, fixture.training_grades
        )
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in plan["populations"]["train"]])
        for split in ("calibration", "test"):
            held = outcomes.held_rows(plan, fixture.dataset, split)
            self.assertEqual([row["id"] for row in held], [row["id"] for row in plan["populations"][split]])
            self.assertNotIn(f"{split}-0", [row["id"] for row in held])
        changed = json.loads(json.dumps(plan))
        changed.pop("sha256")
        changed["populations"]["test"].pop()
        with self.assertRaisesRegex(ValueError, "nonexcluded"):
            outcomes.checked_plan(benefit.seal(changed))

    def test_prepare_rejects_unaudited_expansion_or_held_population_overlap(self):
        original = outcomes.read(self.rows_path)
        extra = [
            {**row, "id": f"extra-{i}", "request": f"new training question {i}", "family": f"extra-family-{i}"}
            for i, row in enumerate(original[:10])
        ]
        fixtures.save(self.rows_path, original + extra)
        with self.assertRaisesRegex(ValueError, "expansion audit"):
            self.prepare()
        extra[0]["family"] = self.fixture.dataset["splits"]["calibration"][0]["family"]
        fixtures.save(self.rows_path, original + extra)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_expanded_plan_keeps_source_prefix_and_same_whole_paper_folds(self):
        original = outcomes.read(self.rows_path)
        pool = original + [
            {**row, "id": f"extra-{i}", "request": f"new training question {i}", "family": f"extra-family-{i}"}
            for i, row in enumerate(original[:10])
        ]
        fixtures.save(self.rows_path, pool)
        eligibility = outcomes.read(self.fixture.eligibility)
        eligibility["original_pool_ids"] = [row["id"] for row in pool]
        eligibility["eligible_train_ids"] = [row["id"] for row in pool if row["id"] != "train-0"]
        fixtures.save(self.fixture.eligibility, eligibility)
        inputs = {"train": [{key: row[key] for key in ("id", "request")} for row in pool if row["id"] != "train-0"]}
        fixtures.save(self.fixture.training_inputs, inputs)
        collection = outcomes.read(self.fixture.training_plan)
        collection.update(
            source_counts={"train": 30},
            counts={"train": 29},
            inputs_sha256=digest(inputs),
            eligible_train_ids=eligibility["eligible_train_ids"],
            eligibility_sha256=file_hash(self.fixture.eligibility),
        )
        fixtures.save(self.fixture.training_plan, collection)
        expansion = fixtures.save(
            self.root / "expansion-audit.json",
            {
                "source_revision": outcomes.SOURCE_REVISION,
                "source_hashes": outcomes.SOURCE_HASHES,
                "original_dataset_sha256": digest(self.fixture.dataset),
                "expanded_rows_sha256": digest(pool),
                "original_prefix_rows": 20,
                "training_rows": 30,
                "ids": [row["id"] for row in pool],
                "families": [row["family"] for row in pool],
            },
        )
        plan = outcomes.prepare(
            self.baseline,
            self.original,
            self.fixture.training_plan,
            self.fixture.training_inputs,
            self.rows_path,
            self.fixture.eligibility,
            expansion_audit_path=expansion,
            attest_test_outcomes_unopened=True,
        )
        self.assertEqual(len(plan["populations"]["train"]), 29)
        old = outcomes.read(self.run / "plan.json")
        old_assignments = {
            row["paper_group"]: fold
            for row, fold in zip(old["populations"]["train"], old["training_folds"], strict=True)
        }
        self.assertTrue(
            all(
                old_assignments[row["paper_group"]] == fold
                for row, fold in zip(plan["populations"]["train"], plan["training_folds"], strict=True)
            )
        )
        self.assertEqual(plan["training_artifacts_sha256"]["expansion_audit"], file_hash(expansion))

    def test_cli_preserves_existing_output_before_reading_any_inputs(self):
        destination = self.run / "plan.json"
        before = destination.read_bytes()
        args = [
            "benefit_outcomes.py",
            "prepare",
            "--run",
            str(self.run),
            "--baseline",
            "absent",
            "--original-frozen",
            "absent",
            "--training-plan",
            "absent",
            "--training-inputs",
            "absent",
            "--training-rows",
            "absent",
            "--eligibility",
            "absent",
            "--attest-test-outcomes-unopened",
        ]
        with patch("sys.argv", args), patch.object(outcomes, "prepare") as preparing, redirect_stdout(io.StringIO()):
            with self.assertRaises(FileExistsError):
                outcomes.main()
            preparing.assert_not_called()
        self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
