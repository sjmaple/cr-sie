import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import solver
from data import digest, file_hash, write


def make_run(directory, split="calibration", n=2):
    rows = [{"id": f"problem-{index}", "request": f"What is {index} + 1?"} for index in range(n)]
    inputs = {split: rows}
    gold = {row["id"]: "1" for row in rows}
    write(directory / "inputs.json", inputs)
    write(directory / "settings.json", solver.SETTINGS)
    write(directory / "gold-for-scoring-only.json", gold)
    write(directory / "gold-preflight.json", {"needs_review": {}})
    write(
        directory / "gold-review.json",
        {
            "preflight_sha256": file_hash(directory / "gold-preflight.json"),
            "approved_ids": [row["id"] for row in rows],
            "ready": True,
        },
    )
    write(
        directory / "plan.json",
        {
            "inputs_sha256": digest(inputs),
            "settings_sha256": digest(solver.SETTINGS),
            "gold_sha256": digest(gold),
        },
    )
    return rows


def record(row, arm, settings=solver.SETTINGS, *, complete=True, known_rejection=False):
    return {
        "id": row["id"],
        "arm": arm,
        "model": settings["arms"][arm]["model"],
        "catalog_revision": settings["arms"][arm]["revision"],
        "settings_sha256": digest(settings),
        "request_sha256": digest(row["request"]),
        "status": "complete" if complete else "incomplete",
        "answer": r"\boxed{1}" if complete else "",
        "calls": [{"billing": "cloud", "billing_known": True, "receipt": {"credits": 2}}] if complete else [],
        "credits": 2 if complete else None,
        "observed_credits": 2 if complete else 0,
        "has_unknown_charges": not complete and not known_rejection,
        "safe_to_resume": known_rejection,
    }


class CatalogClient:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get_model(self, name):
        return next(arm for arm in solver.ARMS.values() if arm["model"] == name)


class SolverCollectionTests(unittest.TestCase):
    def collect(self, directory, **kwargs):
        return solver.collect(directory, "calibration", None, 1, 1000, 30, **kwargs)

    def test_sdk_retry_is_blocked_after_http_diagnostics_are_captured(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(
                429,
                headers={"retry-after": "1", "x-sie-request-id": "request-123", "x-sie-error-code": "RATE_LIMIT"},
                json={
                    "error": {"code": "rate_limit_exceeded", "message": "secret response text"},
                },
            )

        with solver.RecordingClient(
            base_url="https://test.invalid", api_key="", http_transport=httpx.MockTransport(respond)
        ) as client:
            with self.assertRaisesRegex(RuntimeError, "Automatic retries"):
                solver.call(client, solver.ARMS["cheap"], [{"role": "user", "content": "1+1"}])
            self.assertEqual(len(requests), 1)
            failure = client.http_failures[0]
            self.assertEqual(failure["http_status"], 429)
            self.assertEqual(failure["error_code"], "RATE_LIMIT")
            self.assertTrue(failure["known_pre_execution_rejection"])
            self.assertNotIn("secret", json.dumps(failure))

    def test_error_code_precedence_and_openai_normalization_fail_closed(self):
        for status, header, body, expected, safe in [
            (429, "RATE_LIMIT", {"error": {"code": "rate_limit_exceeded"}}, "RATE_LIMIT", True),
            (503, "RESOURCE_EXHAUSTED", {"error": {"code": "QUEUE_FULL"}}, "RESOURCE_EXHAUSTED", False),
            (429, "private-value", {"error": {"code": "RATE_LIMIT"}}, "unrecognized", False),
            (503, None, {"detail": {"code": "QUEUE_FULL"}}, "QUEUE_FULL", True),
            (503, None, {"error": {"code": "provisioning"}}, "PROVISIONING", True),
            (429, None, {"error": {"code": "rate_limit_exceeded"}}, "RATE_LIMIT", False),
            (503, None, {"error": "malformed", "detail": {"code": "QUEUE_FULL"}}, "unrecognized", False),
            (403, "INSUFFICIENT_SCOPE", {}, "INSUFFICIENT_SCOPE", False),
        ]:
            with self.subTest(status=status, header=header, body=body):
                response = httpx.Response(status, json=body, headers={"x-sie-error-code": header} if header else {})
                failure = solver.safe_http_failure(response)
                self.assertEqual(failure["error_code"], expected)
                self.assertEqual(failure["known_pre_execution_rejection"], safe)
                self.assertNotIn("private-value", json.dumps(failure))

    def test_transport_failure_needs_retry_after_to_be_known_pre_execution(self):
        for status, code, hint, expected in [
            (503, "transport_failure", None, False),
            (503, "transport_failure", "1", True),
            (504, "transport_failure", "1", False),
            (503, "QUEUE_FULL", None, True),
            (503, {"bad": "code"}, None, False),
        ]:
            with self.subTest(status=status, code=code, hint=hint):
                response = httpx.Response(
                    status, json={"error": {"code": code}}, headers={"retry-after": hint} if hint else {}
                )
                self.assertEqual(solver.safe_http_failure(response)["known_pre_execution_rejection"], expected)

    def test_missing_credit_receipt_preserves_answer_and_blocks_unknown_retry(self):
        class MissingReceipt(CatalogClient):
            http_failures = ()
            last_model_revision = "b" * 64

            def chat_completions(self, **kwargs):
                return {"choices": [{"message": {"content": r"\boxed{1}"}, "finish_reason": "stop"}], "usage": {}}

        with patch.object(solver, "RecordingClient", MissingReceipt):
            result = solver.solve({"id": "one", "request": "1+0"}, "cheap", solver.SETTINGS)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["calls"][0]["answer"], r"\boxed{1}")
        self.assertEqual(result["calls"][0]["execution_revision"], "b" * 64)
        self.assertTrue(result["has_unknown_charges"])
        self.assertFalse(result["safe_to_resume"])
        self.assertIsNone(result["credits"])

    def test_resume_preserves_history_and_does_not_resend_completed_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_run(directory)
            first_calls = []

            def initial(row, arm, settings, **kwargs):
                first_calls.append((row["id"], arm))
                return record(row, arm, settings, complete=arm == "cheap")

            with (
                patch.object(solver, "SingleSendClient", CatalogClient),
                patch.object(solver, "solve", initial),
                self.assertRaisesRegex(RuntimeError, "incomplete"),
            ):
                self.collect(directory)
            output = directory / "answers-calibration"
            original_complete = file_hash(output / "problem-0-cheap.json")
            original_failed = file_hash(output / "problem-0-strong.json")
            with patch.object(solver, "SingleSendClient", CatalogClient), patch.object(solver, "solve") as solve:
                with self.assertRaisesRegex(ValueError, "retry-unknown"):
                    self.collect(directory, resume=True)
                solve.assert_not_called()
            resumed_calls = []

            def resumed(row, arm, settings, **kwargs):
                resumed_calls.append((row["id"], arm))
                return record(row, arm, settings)

            with patch.object(solver, "SingleSendClient", CatalogClient), patch.object(solver, "solve", resumed):
                self.collect(directory, resume=True, retry_unknown=True)
            self.assertEqual(first_calls, [("problem-0", "cheap"), ("problem-0", "strong")])
            self.assertNotIn(("problem-0", "cheap"), resumed_calls)
            self.assertEqual(len(resumed_calls), 3)
            self.assertEqual(file_hash(output / "problem-0-cheap.json"), original_complete)
            self.assertEqual(file_hash(output / "attempts/problem-0-strong/attempt-0001.json"), original_failed)
            summary = solver.read(output / "collection.json")
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["completed"], 4)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["failed_attempts"], 1)
            self.assertEqual(summary["unknown_charge_attempts"], 1)
            self.assertFalse(summary["acquisition_cost_complete"])
            self.assertEqual(summary["observed_credits"], 8)
            self.assertEqual(len(list((output / "sessions").glob("*/collection.json"))), 2)

    def test_known_rejection_can_resume_without_unknown_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_run(directory, n=1)

            def rejected(row, arm, settings, **kwargs):
                return record(row, arm, settings, complete=False, known_rejection=True)

            with (
                patch.object(solver, "SingleSendClient", CatalogClient),
                patch.object(solver, "solve", rejected),
                self.assertRaises(RuntimeError),
            ):
                self.collect(directory)
            with (
                patch.object(solver, "SingleSendClient", CatalogClient),
                patch.object(
                    solver, "solve", side_effect=lambda row, arm, settings, **kwargs: record(row, arm, settings)
                ),
            ):
                self.collect(directory, resume=True)
            self.assertTrue(solver.read(directory / "answers-calibration/collection.json")["acquisition_cost_complete"])

    def test_resume_checks_limits_and_saved_input_order_before_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_run(directory, n=1)
            with (
                patch.object(solver, "SingleSendClient", CatalogClient),
                patch.object(
                    solver,
                    "solve",
                    side_effect=lambda row, arm, settings, **kwargs: record(row, arm, settings, complete=False),
                ),
                self.assertRaises(RuntimeError),
            ):
                self.collect(directory)
            with patch.object(solver, "SingleSendClient") as client:
                with self.assertRaisesRegex(ValueError, "Attempt limit"):
                    self.collect(directory, resume=True, retry_unknown=True, max_attempts=1)
                plan_path = directory / "answers-calibration/collection-plan.json"
                plan = solver.read(plan_path)
                plan["ids"] = ["different-problem"]
                plan_path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "prepared split"):
                    self.collect(directory, resume=True, retry_unknown=True)
                client.assert_not_called()

    def test_complete_credit_receipts_and_input_bindings_are_validated(self):
        row = {"id": "one", "request": "1+0"}
        valid = record(row, "cheap")
        solver.validate_record(valid, row, "cheap", solver.SETTINGS)
        for field, value in [("credits", 0), ("observed_credits", 0), ("request_sha256", "changed"), ("id", "two")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                solver.validate_record(valid | {field: value}, row, "cheap", solver.SETTINGS)

    def test_regrade_to_new_file_preserves_archived_grades(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_run(directory, n=1)
            with (
                patch.object(solver, "SingleSendClient", CatalogClient),
                patch.object(
                    solver, "solve", side_effect=lambda row, arm, settings, **kwargs: record(row, arm, settings)
                ),
            ):
                self.collect(directory)
            solver.grade(directory, "calibration")
            original = (directory / "grades-calibration.json").read_bytes()
            reproduced = directory / "regraded/calibration.json"
            solver.grade(directory, "calibration", reproduced)
            self.assertEqual(reproduced.read_bytes(), original)
            self.assertEqual((directory / "grades-calibration.json").read_bytes(), original)
            with patch.object(solver, "grade_reviewed") as grade_reviewed:
                with self.assertRaises(FileExistsError):
                    solver.grade(directory, "calibration", reproduced)
                grade_reviewed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
