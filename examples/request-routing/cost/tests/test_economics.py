import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import economics
import economics_judge
import economics_policy
import economics_review
import economics_scoring
import learn
from cloud import digest


class AnswerScoringTests(unittest.TestCase):
    def test_exact_exponent_syntax_audit_keeps_tex_token_boundaries(self):
        canonical = economics_scoring.canonical_latex
        self.assertEqual(canonical("e^e - 1"), canonical("e^{e}-1"))
        self.assertNotEqual(canonical("e^e - 1"), canonical("e^{e}-2"))
        self.assertNotEqual(canonical("x^10"), canonical("x^{10}"))
        self.assertIsNone(canonical(r"\text{a b}"))

    def test_uses_last_box_with_nested_fraction(self):
        self.assertEqual(economics.final_answer(r"First \boxed{2}; finally \boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_incomplete_last_box_does_not_reuse_intermediate_answer(self):
        self.assertIsNone(economics.final_answer(r"Intermediate \boxed{42}. Finally \boxed{\frac{1}"))

    def test_unboxed_work_is_not_a_final_answer(self):
        self.assertIsNone(economics.final_answer("We used 42 but could not solve the problem."))

    def test_empty_box_is_not_an_answer(self):
        self.assertEqual(economics.final_answer(r"\boxed{}"), "")

    def test_clock_format_matches_source_without_meridiem(self):
        self.assertEqual(economics.clock_value("6:0"), economics.clock_value(r"6:00 \text{ PM}"))
        self.assertNotEqual(economics.clock_value("6:0"), economics.clock_value("12:30"))
        self.assertIsNone(economics.clock_value("6:80"))


class EconomicTests(unittest.TestCase):
    def test_zero_observed_harms_does_not_claim_zero_population_risk(self):
        self.assertGreater(economics_policy.harmful_switch_upper_bound(0, 64), 0.01)
        self.assertLess(economics_policy.harmful_switch_upper_bound(0, 512), 0.01)
        self.assertAlmostEqual(economics_policy.harmful_switch_upper_bound(0, 10), 0.2588655509)

    def test_quality_loss_preserves_cases_where_cheap_is_better(self):
        qc = np.array([1, 0, 1, 0])
        qs = np.array([1, 1, 0, 0])
        cheap = np.ones(4)
        strong = np.full(4, 10)
        result = economics_policy.evaluation_metrics(np.zeros(4, dtype=bool), qc, qs, cheap, strong, 1)
        self.assertEqual(result["quality_loss_pp"], 0)
        self.assertEqual(result["lost_correct_answers"], 1)
        self.assertEqual(result["gained_correct_answers"], 1)
        self.assertEqual(result["saving_vs_strong_pct"], 80)
        self.assertEqual(result["usd_per_1000"], 0.02)

    def test_threshold_cannot_buy_savings_by_exceeding_quality_budget(self):
        result = economics_policy.select_threshold(
            np.array([0.1, 0.2, 0.8, 0.9]),
            np.array([1, 1, 0, 0]),
            np.ones(4),
            np.full(4, 10),
            0.1,
            0,
        )
        self.assertEqual(result["tau"], 0.8)
        self.assertEqual(result["validation_relative_quality"], 1)
        self.assertEqual(result["validation_strong_share"], 0.5)

    def test_no_cost_advantage_keeps_strong(self):
        result = economics_policy.select_threshold(
            np.array([0.1, 0.2]),
            np.ones(2),
            np.full(2, 20),
            np.full(2, 10),
            0.1,
            0,
        )
        self.assertIsNone(result["tau"])


class CollectorTests(unittest.TestCase):
    def test_self_hosted_final_answer_retains_tokens_without_inventing_cloud_credits(self):
        replies = iter(
            [
                {
                    "choices": [{"message": {"content": "Partial work"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 30},
                },
                {
                    "choices": [{"message": {"content": '{"answer":"42"}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 8},
                },
            ]
        )

        class Client:
            last_model_revision = "native-pinned"

            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def chat_completions(self, **kwargs):
                return next(replies)

        with patch.object(economics, "SingleSendClient", Client):
            row = economics.send(
                {"id": "native", "request": "Compute the answer"},
                "budget_cheap",
                economics.ANSWER_ARMS["budget_cheap"],
                self_hosted=True,
            )
        self.assertEqual(row["status"], "success")
        self.assertEqual(economics.final_answer(row["answer"]), "42")
        self.assertEqual(row["usage"]["completion_tokens"], 30)
        self.assertEqual(row["continuation"]["usage"]["completion_tokens"], 8)
        self.assertNotIn("credits", row)
        self.assertNotIn("credits", row["continuation"])
        self.assertNotIn("incremental_credits", row)

    def test_failure_drains_paid_inflight_answers_without_submitting_more(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            inputs = {"train": [{"id": str(i), "request": "math"} for i in range(10)]}
            learn.write(directory / "inputs.json", inputs)
            learn.write(
                directory / "plan.json",
                {
                    "inputs_sha256": digest(inputs),
                    "max_inflight": 2,
                    "max_observed_credits": 100,
                    "max_seconds_per_collection": 10,
                    "pilot_arms": {"cheap": {"system": "math"}},
                    "pilot_ids": [str(i) for i in range(10)],
                },
            )
            started = []
            both_started = threading.Barrier(2)

            def send(row, name, arm):
                started.append(row["id"])
                both_started.wait(timeout=2)
                if row["id"] == "0":
                    return {"id": "0", "arm": name, "status": "failed_or_unknown", "error_type": "ExampleError"}
                time.sleep(0.02)
                return {"id": row["id"], "arm": name, "status": "success", "credits": 7}

            with patch.object(economics, "send", send), self.assertRaisesRegex(RuntimeError, "Incomplete collection"):
                economics.collect(directory, "pilot")
            retained = economics.rows(directory / "answers-pilot.jsonl")
            self.assertEqual(sorted(started), ["0", "1"])
            self.assertEqual({r["id"] for r in retained}, set(started))
            self.assertEqual(economics.read(directory / "collection-pilot.json")["credits_all_phases"], 7)

    def test_budget_stop_keeps_inflight_receipts_and_bounds_overshoot(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            inputs = {"train": [{"id": str(i), "request": "math"} for i in range(10)]}
            learn.write(directory / "inputs.json", inputs)
            learn.write(
                directory / "plan.json",
                {
                    "inputs_sha256": digest(inputs),
                    "max_inflight": 2,
                    "max_observed_credits": 5,
                    "max_seconds_per_collection": 10,
                    "pilot_arms": {"cheap": {"system": "math"}},
                    "pilot_ids": [str(i) for i in range(10)],
                },
            )

            def send(row, name, arm):
                return {"id": row["id"], "arm": name, "status": "success", "credits": 7}

            with patch.object(economics, "send", send), self.assertRaisesRegex(RuntimeError, "Incomplete collection"):
                economics.collect(directory, "pilot")
            retained = economics.rows(directory / "answers-pilot.jsonl")
            self.assertEqual(len(retained), 2)
            receipt = economics.read(directory / "collection-pilot.json")
            self.assertTrue(receipt["budget_reached"])
            self.assertEqual(receipt["credits_all_phases"], 14)


class FreezeTests(unittest.TestCase):
    def test_policy_change_after_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            settings, inputs = {"arms": {}}, {"test": []}
            frozen = {
                "answer_settings_sha256": digest(settings),
                "inputs_sha256": digest(inputs),
                "supervision": "teacher_agreement",
                "primary": "ridge512_loss1",
                "policies": {"ridge512_loss1": {"tau": 0.5}},
            }
            learn.write(directory / "answer-settings.json", settings)
            learn.write(directory / "inputs.json", inputs)
            learn.write(directory / "routing-frozen.json", frozen)
            economics_policy.freeze_evaluation(directory, ["teacher_agreement"], "teacher_agreement")
            economics.frozen_policies(directory)
            frozen["policies"]["ridge512_loss1"]["tau"] = 0.6
            learn.write(directory / "routing-frozen.json", frozen)
            with self.assertRaisesRegex(ValueError, "policy changed"):
                economics.frozen_policies(directory)

    def test_cannot_freeze_comparisons_after_test_collection_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "answers-test.jsonl").touch()
            with self.assertRaisesRegex(ValueError, "held-out collection"):
                economics_policy.freeze_evaluation(directory, ["teacher_agreement"], "teacher_agreement")


class ReviewLabelsTests(unittest.TestCase):
    def test_review_shard_does_not_send_unselected_paid_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"rows": [{"id": str(i)} for i in range(8)], "gold_provided": False, "test_provided": False}
            learn.write(root / "input.json", plan)
            sent = []

            def review(row, prepared, model, self_hosted):
                sent.append(row["id"])
                return {"id": row["id"], "status": "success", "valid_label": True, "credits": 7}

            with patch.object(economics_judge, "review", review):
                result = economics_judge.collect(
                    root / "input.json", root / "reviews", "teacher", concurrency=2, start=2, limit=3
                )
            self.assertEqual(sorted(sent), ["2", "3", "4"])
            self.assertEqual(result["observed_credits"], 21)
            frozen = economics.read(root / "reviews/plan.json")
            self.assertEqual(frozen["input_sha256"], digest(plan))
            self.assertEqual(frozen["ids"], ["2", "3", "4"])

    def test_verdict_step_preserves_work_cost_and_candidate_order(self):
        calls = []

        class Client:
            last_model_revision = "pinned"

            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def chat_completions(self, **kwargs):
                calls.append(kwargs)
                return {
                    "choices": [
                        {"message": {"content": '{"correct_a":true,"correct_b":false}'}, "finish_reason": "stop"}
                    ]
                }

        row = {
            "messages": [
                {"role": "system", "content": "verify"},
                {"role": "user", "content": json.dumps({"problem": "math", "candidate_a": "a", "candidate_b": "b"})},
            ],
            "input_sha256": "input",
            "arm_order": ["strong", "cheap"],
        }
        original = {
            "status": "success",
            "input_sha256": "input",
            "valid_label": False,
            "answer": "partial verification",
            "finish_reason": "length",
            "elapsed_s": 1,
            "credits": 7,
        }
        with (
            patch.object(economics_judge, "SingleSendClient", Client),
            patch.object(economics_judge, "usage_of", return_value={"credits": 3}),
        ):
            result = economics_judge.conclude(row, original, "teacher", False)
            again = economics_judge.conclude(row, result, "teacher", False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_tokens"], 64)
        self.assertEqual(result["correct_by_arm"], {"strong": True, "cheap": False})
        self.assertEqual(result["credits"], 10)
        self.assertEqual(result["initial_answer"], original["answer"])
        self.assertFalse(original["valid_label"])
        self.assertEqual(again, result)

    def test_signed_teacher_labels_use_candidate_identity_without_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            settings = {"arms": {}, "cheap_arm": "c", "strong_arm": "s"}
            data = {"train": [{"id": "one", "request": "problem"}], "validation": []}
            records = {("one", arm): {"answer": arm} for arm in ("c", "s")}
            row = {
                "id": "one",
                "messages": [{"role": "user", "content": "problem"}],
                "arm_order": ["s", "c"],
                "source_records_sha256": [digest(records[("one", arm)]) for arm in ("s", "c")],
            }
            row["input_sha256"] = digest(row["messages"])
            plan = {
                "gold_provided": False,
                "test_provided": False,
                "answer_settings_sha256": digest(settings),
                "rows": [row],
            }
            review = row | {
                "status": "success",
                "valid_label": True,
                "correct_by_arm": {"s": False, "c": True},
                "model": "teacher",
                "model_revision": "pinned",
            }
            learn.write(directory / "inputs.json", data)
            learn.write(directory / "answer-settings.json", settings)
            learn.write(directory / "review-input.json", plan)
            (directory / "reviews.jsonl").write_text(json.dumps(review) + "\n")
            with patch.object(economics, "answer_records", return_value=records):
                targets, provenance, _ = economics_review.labels(
                    directory, [directory / "review-input.json"], [directory / "reviews.jsonl"]
                )
                self.assertEqual(targets, {"one": 2})  # Cheap-only win: signed benefit is -1.
                self.assertFalse(provenance["gold_provided"])
                review["source_records_sha256"] = ["wrong", "wrong"]
                (directory / "reviews.jsonl").write_text(json.dumps(review) + "\n")
                with self.assertRaisesRegex(ValueError, "different candidate"):
                    economics_review.labels(directory, [directory / "review-input.json"], [directory / "reviews.jsonl"])
                review["source_records_sha256"] = row["source_records_sha256"]
                review["valid_label"] = False
                del review["correct_by_arm"]
                (directory / "reviews.jsonl").write_text(json.dumps(review) + "\n")
                with self.assertRaisesRegex(ValueError, "Invalid teacher review"):
                    economics_review.labels(directory, [directory / "review-input.json"], [directory / "reviews.jsonl"])
                conservative, provenance, _ = economics_review.labels(
                    directory, [directory / "review-input.json"], [directory / "reviews.jsonl"], invalid_as_strong=True
                )
                self.assertEqual(conservative, {"one": 0})
                self.assertEqual(provenance["invalid_reviews"], 1)


if __name__ == "__main__":
    unittest.main()
