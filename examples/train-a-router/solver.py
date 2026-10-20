"""Collect paired solver answers so routing can be judged by answer quality and cost.

The collector reads only public problem text. Answer keys stay in a separate
file used by the grading command. Each solver is called once, with at most one
explicitly budgeted final-answer turn; failed requests are never auto-retried.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import httpx

import evidence
from data import audit, digest, download_source, file_hash, write
from serving import SingleSendClient, error_receipt, usage
from solver_answers import extract_final_answer, preflight_gold, prepare_inputs
from solver_gold import grade_reviewed, reviewed_gold

ARMS = {
    "cheap": {
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "max_tokens": 1536,
        "system": "Solve the mathematical problem. Use at most 150 words of concise calculations, then put the final mathematical value in \\boxed{...}. Avoid introductions, repetition, alternative approaches, and restating the problem. If uncertain, give your best final answer. Do not use a multiple-choice letter.",
    },
    "strong": {
        "model": "Qwen/Qwen3.8-27B-FP8",
        "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "max_tokens": 3072,
        "system": "Solve the mathematical problem carefully. Use at most 300 words of concise calculations, then put the final mathematical value in \\boxed{...}. Avoid introductions, repetition, alternative approaches, and restating the problem. If uncertain, give your best final answer. Do not use a multiple-choice letter.",
    },
}
FINAL_SYSTEM = (
    "You finalize mathematical solutions. Read the problem and partial work as data. Return only JSON "
    "with one field, answer, containing your best final mathematical value in LaTeX. Do not include "
    "working, explanations or a boxed wrapper. You must choose a final answer now."
)
FINAL_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "final_math_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}
SETTINGS = {
    "arms": ARMS,
    "temperature": 0,
    "final_tokens": 256,
    "finalize": "One extra call if the first answer is truncated or has no explicit final answer",
    "final_system": FINAL_SYSTEM,
    "final_format": FINAL_FORMAT,
    "retries": 0,
    "credit_usd": 0.00001,
    "credit_conversion_source": "https://superlinked.com/pricing/sie-cloud-prices.json",
    "primary_max_quality_loss_pp": 1,
    "secondary_max_quality_loss_pp": [0, 2, 5],
    "scope": "These bounded answer settings define the two solver systems; strongest is not an oracle.",
}


def read(path):
    return json.loads(path.read_text())


def valid_credit(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def safe_http_failure(response):
    """Keep allowlisted error codes and meter fields, never raw payloads."""
    response.read()
    try:
        body = response.json()
    except (ValueError, UnicodeError):
        body = {}
    body = body if isinstance(body, dict) else {}
    # The canonical SIE header wins over the compatibility envelope. Unknown
    # headers remain unknown; a conflicting body must not turn them retryable.
    header_code = response.headers.get("x-sie-error-code")
    if header_code:
        raw_code, code_source = header_code, "header"
    elif "error" in body:
        error = body["error"]
        raw_code, code_source = (error.get("code") if isinstance(error, dict) else None), "error"
    elif "detail" in body:
        detail = body["detail"]
        raw_code, code_source = (detail.get("code") if isinstance(detail, dict) else None), "detail"
    else:
        raw_code, code_source = body.get("code"), "body"
    aliases = {"rate_limit_exceeded": "RATE_LIMIT", "provisioning": "PROVISIONING"}
    code = aliases.get(raw_code, raw_code) if isinstance(raw_code, str) else None
    known_codes = {
        "RATE_LIMIT",
        "BILLING_CAPACITY_UNAVAILABLE",
        "QUEUE_FULL",
        "QUEUE_UNAVAILABLE",
        "transport_failure",
        "PROVISIONING",
        "MODEL_LOADING",
        "RESOURCE_EXHAUSTED",
        "MODEL_LOAD_FAILED",
        "INPUT_TOO_LONG",
        "INSUFFICIENT_CREDITS",
        "ACCOUNT_DISABLED",
        "INSUFFICIENT_SCOPE",
    }
    code = code if isinstance(code, str) and code in known_codes else "unrecognized"
    retry_after = response.headers.get("retry-after")
    try:
        retry_after = float(retry_after)
    except (ValueError, TypeError):
        retry_after = None
    if not valid_credit(retry_after):
        retry_after = None
    known_rejection = (
        # The header/native discriminator identifies managed pre-dispatch
        # admission. A bare compatibility rate_limit_exceeded has also been
        # used for queue failures after publication, so normalization alone is
        # insufficient evidence that execution never started.
        (response.status_code == 429 and code == "RATE_LIMIT" and raw_code == "RATE_LIMIT")
        or (
            response.status_code == 503
            and code
            in {
                "BILLING_CAPACITY_UNAVAILABLE",
                "QUEUE_FULL",
                "PROVISIONING",
                "MODEL_LOADING",
            }
        )
        or (
            response.status_code == 503
            and code in {"QUEUE_UNAVAILABLE", "transport_failure"}
            and retry_after is not None
        )
    )
    request_id = response.headers.get("x-sie-request-id")
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id):
        request_id = None
    native_usage = body.get("usage")
    native_usage = native_usage if isinstance(native_usage, dict) else {}
    credits = native_usage.get("credits_charged")
    if not valid_credit(credits):
        header = response.headers.get("x-sie-credits-debited", "")
        credits = int(header) if re.fullmatch(r"0|[1-9][0-9]{0,18}", header) else None
    return {
        "http_status": response.status_code,
        "error_code": code,
        "error_code_source": code_source,
        "request_id": request_id,
        "credits": credits,
        "retry_after_seconds": retry_after,
        "known_pre_execution_rejection": known_rejection,
    }


class RecordingClient(SingleSendClient):
    """Use the SDK's public transport injection to retain errors before it retries."""

    def __init__(self, *, http_transport=None, **kwargs):
        self.http_failures = []
        # Let the SDK resolve its public environment contract. Construction makes
        # no HTTP requests, and the temporary transport is closed immediately.
        with SingleSendClient(**kwargs) as configured:
            base_url = configured.base_url
        kwargs.pop("base_url", None)
        transport = httpx.Client(
            base_url=base_url, transport=http_transport, event_hooks={"response": [self.capture_failure]}
        )
        super().__init__(base_url, http_client=transport, **kwargs)

    def capture_failure(self, response):
        if response.status_code >= 400:
            self.http_failures.append(safe_http_failure(response))


def save_latest(path, value):
    temporary = path.with_suffix(path.suffix + ".new")
    # Immutable attempt receipts remain authoritative after an interrupted
    # replacement; a stale convenience copy can be rebuilt from them.
    temporary.unlink(missing_ok=True)
    write(temporary, value)
    temporary.replace(path)


def prepare(reference, source, output):
    if output.exists():
        raise FileExistsError(output)
    evidence.verify(reference)
    dataset = read(reference / "data.json")
    audit(dataset)
    download_source(source)
    inputs, gold, provenance = prepare_inputs(source, dataset)
    write(output / "inputs.json", inputs)
    write(output / "gold-for-scoring-only.json", gold)
    write(output / "settings.json", SETTINGS)
    write(
        output / "plan.json",
        {
            "dataset_sha256": digest(dataset),
            "inputs_sha256": digest(inputs),
            "gold_sha256": digest(gold),
            "settings_sha256": digest(SETTINGS),
            "counts": {split: len(rows) for split, rows in inputs.items()},
            "source": provenance,
            "inference_reads_gold": False,
            "selection": "Use calibration answers only; freeze routing choices before collecting test answers",
            "grading": "Exact source answers with pinned symbolic equivalence checks; no teacher agreement proxy",
        },
    )
    write(output / "gold-preflight.json", preflight_gold(gold))
    canonical, review = reviewed_gold(gold, source=source)
    review["preflight_sha256"] = file_hash(output / "gold-preflight.json")
    write(output / "gold-reviewed.json", canonical)
    write(output / "gold-adjudication-details.json", review)
    write(
        output / "gold-review.json",
        {
            "preflight_sha256": review["preflight_sha256"],
            "original_gold_sha256": review["original_gold_sha256"],
            "canonical_gold_sha256": review["canonical_gold_sha256"],
            "approved_ids": sorted(set(gold) - set(review["needs_review"])),
            "adjudicated_ids": review["approved_ids"],
            "unresolved_ids": sorted(review["needs_review"]),
            "ready": review["ready"],
        },
    )
    print(json.dumps({"prepared": str(output), "counts": {k: len(v) for k, v in inputs.items()}}))


def checked_inputs(directory):
    plan, inputs, settings = (read(directory / name) for name in ("plan.json", "inputs.json", "settings.json"))
    if plan["inputs_sha256"] != digest(inputs) or plan["settings_sha256"] != digest(settings):
        raise ValueError("Solver inputs or settings changed after preparation")
    if settings != SETTINGS:
        raise ValueError("Solver source settings differ from the saved protocol")
    return plan, inputs, settings


def check_review(directory, selected):
    """Inference checks only reviewed IDs and hashes, never the answer key."""
    review_path = directory / "gold-review.json"
    if not review_path.exists():
        raise ValueError("Resolve the gold preflight before collecting answers")
    review = read(review_path)
    if review["preflight_sha256"] != file_hash(directory / "gold-preflight.json"):
        raise ValueError("Gold review belongs to a different preflight")
    if review.get("ready") is False or set(selected) - set(review["approved_ids"]):
        raise ValueError("Gold review leaves selected problems unresolved")


def call(client, arm, messages, *, final=False, self_hosted=False):
    started = time.monotonic()
    reply = client.chat_completions(
        model=arm["model"],
        messages=messages,
        temperature=0,
        max_tokens=256 if final else arm["max_tokens"],
        response_format=FINAL_FORMAT if final else None,
        wait_for_capacity=False,
        max_oom_retries=0,
        provision_timeout_s=180,
        read_timeout_s=180,
    )
    receipt = usage(reply)
    credits = receipt["credits"]
    choices = reply.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    answer = message.get("content")
    return {
        "messages_sha256": digest(messages),
        "elapsed_seconds": time.monotonic() - started,
        "answer": answer or "",
        "response_valid": isinstance(answer, str),
        "finish_reason": choice.get("finish_reason"),
        "execution_revision": client.last_model_revision,
        "billing": "self_hosted" if self_hosted else "cloud",
        "receipt": receipt,
        "billing_known": self_hosted or (type(credits) in (int, float) and math.isfinite(credits) and credits >= 0),
        "tokens": {
            k: v
            for k, v in reply.get("usage", {}).items()
            if k in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }


def solve(row, arm_name, settings, *, self_hosted=False):
    arm = settings["arms"][arm_name]
    record = initial_record(row, arm_name, settings)
    client = None
    try:
        with RecordingClient(timeout_s=180, read_timeout_s=180) as client:
            messages = [{"role": "system", "content": arm["system"]}, {"role": "user", "content": row["request"]}]
            first = call(client, arm, messages, self_hosted=self_hosted)
            record["calls"].append(first)
            if not first["response_valid"]:
                raise ValueError("Successful response has no text answer")
            if not first["billing_known"]:
                raise ValueError("Successful response has no usable billed-credit receipt")
            answer = first["answer"]
            if first["finish_reason"] == "length" or not extract_final_answer(answer):
                final_messages = [
                    {"role": "system", "content": FINAL_SYSTEM},
                    {"role": "user", "content": json.dumps({"problem": row["request"], "partial_work": answer})},
                ]
                final = call(client, arm, final_messages, final=True, self_hosted=self_hosted)
                record["calls"].append(final)
                if not final["response_valid"]:
                    raise ValueError("Successful final response has no text answer")
                if not final["billing_known"]:
                    raise ValueError("Successful final response has no usable billed-credit receipt")
                try:
                    parsed = json.loads(final["answer"])
                    if (
                        not isinstance(parsed, dict)
                        or set(parsed) != {"answer"}
                        or not isinstance(parsed["answer"], str)
                    ):
                        raise ValueError("Malformed final answer")
                    answer = "\\boxed{" + parsed["answer"] + "}"
                except (ValueError, TypeError):
                    answer = ""
                    record["final_format_error"] = True
            record["answer"] = answer
            record["status"] = "complete"
    except Exception as error:  # noqa: BLE001 -- preserve failed attempts without publishing sensitive error text
        record["error"] = error_receipt(error)
    failures = list(client.http_failures) if client is not None else []
    record["http_failures"] = failures
    safe_rejection = bool(failures) and failures[-1]["known_pre_execution_rejection"]
    if failures and "error" in record:
        record["error"].update(failures[-1])
    record["observed_credits"] = sum(
        c["receipt"]["credits"]
        for c in record["calls"]
        if type(c["receipt"]["credits"]) in (int, float)
        and math.isfinite(c["receipt"]["credits"])
        and c["receipt"]["credits"] >= 0
    )
    record["observed_credits"] += sum(f["credits"] for f in failures if valid_credit(f["credits"]))
    record["has_unknown_charges"] = any(not c["billing_known"] for c in record["calls"]) or (
        "error" in record and not safe_rejection
    )
    record["safe_to_resume"] = safe_rejection and not record["has_unknown_charges"]
    record["credits"] = record["observed_credits"] if record["status"] == "complete" and not self_hosted else None
    return record


def initial_record(row, arm_name, settings):
    arm = settings["arms"][arm_name]
    return {
        "id": row["id"],
        "arm": arm_name,
        "request_sha256": digest(row["request"]),
        "settings_sha256": digest(settings),
        "model": arm["model"],
        "catalog_revision": arm["revision"],
        "status": "incomplete",
        "calls": [],
    }


def validate_record(record, row, arm, settings):
    if (
        record.get("id") != row["id"]
        or record.get("arm") != arm
        or record.get("settings_sha256") != digest(settings)
        or record.get("request_sha256") != digest(row["request"])
        or record.get("model") != settings["arms"][arm]["model"]
        or record.get("catalog_revision") != settings["arms"][arm]["revision"]
        or record.get("status") not in ("complete", "incomplete")
        or not valid_credit(record.get("observed_credits"))
    ):
        raise ValueError("Answer does not match the frozen request, settings and identity")
    if record["status"] == "complete":
        if record.get("has_unknown_charges") or not isinstance(record.get("answer"), str):
            raise ValueError("Complete answer has unresolved execution or billing")
        calls = record.get("calls")
        if not isinstance(calls, list) or not 1 <= len(calls) <= 2:
            raise ValueError("Complete answer has an invalid call count")
        hosted = {call.get("billing") for call in calls}
        if hosted == {"cloud"}:
            charges = [call.get("receipt", {}).get("credits") for call in calls]
            if not all(valid_credit(charge) for charge in charges) or (
                sum(charges) != record["credits"] or record["credits"] != record["observed_credits"]
            ):
                raise ValueError("Complete answer credits do not match its call receipts")
        elif hosted != {"self_hosted"} or record.get("credits") is not None:
            raise ValueError("Complete answer has inconsistent hosting or cost evidence")


def collection_binding(plan, directory, split, rows, settings, frozen_hash, self_hosted):
    if (
        plan["ids"] != [r["id"] for r in rows]
        or len(plan["ids"]) != len(set(plan["ids"]))
        or plan["split"] != split
        or plan["plan_sha256"] != file_hash(directory / "plan.json")
        or plan["settings_sha256"] != digest(settings)
        or plan.get("policies_frozen_sha256") != frozen_hash
        or plan["self_hosted"] != self_hosted
    ):
        raise ValueError("Collection does not match the prepared split, inputs and frozen policies")


def load_attempts(output, rows, settings):
    """Archive old-format records once, and validate immutable attempt receipts."""
    records, attempts, interrupted = {}, {}, set()
    for row in rows:
        for arm in settings["arms"]:
            key = (row["id"], arm)
            name = f"{row['id']}-{arm}"
            canonical = output / f"{name}.json"
            history = output / "attempts" / name
            if canonical.exists() and not list(history.glob("attempt-*.json")):
                history.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(canonical, history / "attempt-0001.json")
            paths = sorted(path for path in history.glob("attempt-*.json") if not path.name.endswith(".intent.json"))
            history_records = []
            for path in paths:
                record = read(path)
                validate_record(record, row, arm, settings)
                history_records.append(record)
            if canonical.exists():
                record = read(canonical)
                validate_record(record, row, arm, settings)
                if not history_records or record != history_records[-1]:
                    raise ValueError("Latest answer differs from its immutable attempt history")
                records[key] = record
            elif history_records:
                # A process may stop after writing the immutable receipt but
                # before replacing the convenience copy. No generation repeats.
                records[key] = history_records[-1]
                save_latest(canonical, history_records[-1])
            intents = sorted(history.glob("attempt-*.intent.json"))
            dangling = [path for path in intents if not path.with_name(path.name.replace(".intent", "")).exists()]
            if dangling:
                interrupted.add(key)
            attempts[key] = history_records
    return records, attempts, interrupted


def collect(
    directory,
    split,
    limit,
    workers,
    max_credits,
    max_seconds,
    *,
    self_hosted=False,
    resume=False,
    retry_unknown=False,
    max_attempts=3,
):
    _plan, inputs, settings = checked_inputs(directory)
    frozen_hash = None
    if split == "test":
        frozen = read(directory / "policies-frozen.json")
        if frozen["plan_sha256"] != file_hash(directory / "plan.json"):
            raise ValueError("Test collection requires routing choices frozen against this plan")
        frozen_hash = file_hash(directory / "policies-frozen.json")
    if limit is not None and limit <= 0:
        raise ValueError("A row limit must be positive")
    rows = inputs[split][:limit] if limit else inputs[split]
    if (
        not rows
        or not 1 <= workers <= 16
        or not math.isfinite(max_credits)
        or not math.isfinite(max_seconds)
        or max_credits <= 0
        or max_seconds <= 0
        or type(max_attempts) is not int
        or max_attempts <= 0
    ):
        raise ValueError("Use nonempty inputs, 1..16 workers and positive collection limits")
    if retry_unknown and not resume:
        raise ValueError("Unknown execution may only be retried with an explicit --resume")
    check_review(directory, [r["id"] for r in rows])
    output = directory / f"answers-{split}"
    if resume:
        plan = read(output / "collection-plan.json")
        collection_binding(plan, directory, split, rows, settings, frozen_hash, self_hosted)
    else:
        output.mkdir(exist_ok=False)
        write(
            output / "collection-plan.json",
            {
                "ids": [r["id"] for r in rows],
                "split": split,
                "plan_sha256": file_hash(directory / "plan.json"),
                "settings_sha256": digest(settings),
                "policies_frozen_sha256": frozen_hash,
                "max_credits": max_credits,
                "credit_limit_scope": "Observed spend; each in-flight task may finish its initial and one finalization call",
                "max_seconds": max_seconds,
                "workers": workers,
                "self_hosted": self_hosted,
            },
        )
    records, history, interrupted = load_attempts(output, rows, settings)
    todo = []
    for row in rows:
        for arm in settings["arms"]:
            key = (row["id"], arm)
            previous = records.get(key)
            if previous and previous["status"] == "complete":
                continue
            attempt_folder = output / "attempts" / f"{row['id']}-{arm}"
            attempt_numbers = {path.name.split(".")[0] for path in attempt_folder.glob("attempt-*.json")}
            if len(attempt_numbers) >= max_attempts:
                raise ValueError("Attempt limit reached; inspect receipts before explicitly raising --max-attempts")
            unknown = key in interrupted or (previous is not None and not previous.get("safe_to_resume", False))
            if unknown and not retry_unknown:
                raise ValueError(
                    "Prior execution or charge is unknown; inspect receipts before --resume --retry-unknown"
                )
            todo.append((row, arm))
    if not todo:
        print(f"{split}: every answer is already complete; rebuilding the summary without requests", flush=True)
    sessions = output / "sessions"
    previous_sessions = sorted(sessions.glob("session-*"))
    if not previous_sessions and (output / "collection.json").exists():
        initial = sessions / "session-0000"
        initial.mkdir(parents=True)
        shutil.copyfile(output / "collection.json", initial / "collection.json")
        shutil.copyfile(output / "collection-plan.json", initial / "plan.json")
        previous_sessions = [initial]
    session = sessions / f"session-{len(previous_sessions) + 1:04d}"
    write(
        session / "plan.json",
        {
            "collection_plan_sha256": file_hash(output / "collection-plan.json"),
            "resume": resume,
            "retry_unknown": retry_unknown,
            "max_attempts": max_attempts,
            "workers": workers,
            "max_credits": max_credits,
            "max_seconds": max_seconds,
            "pending": [{"id": row["id"], "arm": arm} for row, arm in todo],
            "budget_scope": "New observed acquisition charges in this invocation; in-flight requests are drained",
        },
    )
    tasks = iter(todo)
    started, credits, failures_this_session, sent = time.monotonic(), 0.0, 0, 0
    stopped = False
    fatal = None
    try:
        # Check public model identity once before inference; no answer key is read here.
        if todo:
            with SingleSendClient(timeout_s=60) as client:
                for arm in settings["arms"].values():
                    metadata = client.get_model(arm["model"])
                    if metadata["revision"] != arm["revision"]:
                        raise ValueError("Solver revision differs from the prepared protocol")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}

            def schedule():
                nonlocal stopped, sent
                if failures_this_session or credits >= max_credits or time.monotonic() - started >= max_seconds:
                    stopped = True
                    return
                task = next(tasks, None)
                if task is not None:
                    row, arm = task
                    identity = row["id"]
                    folder = output / "attempts" / f"{identity}-{arm}"
                    numbers = [int(path.name.split(".")[0].split("-")[1]) for path in folder.glob("attempt-*.json")]
                    number = max(numbers, default=0) + 1
                    target = folder / f"attempt-{number:04d}.json"
                    write(
                        target.with_name(target.stem + ".intent.json"),
                        {
                            "id": identity,
                            "arm": arm,
                            "session": session.name,
                            "request_sha256": digest(row["request"]),
                            "settings_sha256": digest(settings),
                        },
                    )
                    pending[pool.submit(solve, row, arm, settings, self_hosted=self_hosted)] = (row, arm, target)
                    sent += 1

            for _ in range(workers):
                schedule()
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    row, arm, target = pending.pop(future)
                    identity = row["id"]
                    try:
                        record = future.result()
                    except Exception as error:  # noqa: BLE001 -- retain unknown worker attempts and drain others
                        record = initial_record(row, arm, settings) | {
                            "error": error_receipt(error),
                            "observed_credits": 0,
                            "credits": None,
                            "has_unknown_charges": True,
                            "safe_to_resume": False,
                        }
                    validate_record(record, row, arm, settings)
                    write(target, record)
                    save_latest(output / f"{identity}-{arm}.json", record)
                    records[(identity, arm)] = record
                    history[(identity, arm)].append(record)
                    credits += record["observed_credits"]
                    failures_this_session += int(record["status"] != "complete")
                    completed = sum(r["status"] == "complete" for r in records.values())
                    print(
                        f"{split}: {completed}/{len(settings['arms']) * len(rows)} complete, "
                        f"{failures_this_session} failed this session, {credits:g} new observed credits",
                        flush=True,
                    )
                for _ in done:
                    schedule()
    except Exception as error:  # noqa: BLE001 -- preserve a content-free incomplete-run receipt
        fatal = error_receipt(error)
    completed = sum(r["status"] == "complete" for r in records.values())
    failed = sum(r["status"] != "complete" for r in records.values())
    status = (
        "complete" if completed == len(settings["arms"]) * len(rows) and not failed and fatal is None else "incomplete"
    )
    all_attempts = [record for attempts in history.values() for record in attempts]
    summary = {
        "status": status,
        "completed": completed,
        "failed": failed,
        "stopped": stopped,
        "observed_credits": sum(record["observed_credits"] for record in all_attempts),
        "selected_answer_credits": sum(
            record["observed_credits"] for record in records.values() if record["status"] == "complete"
        ),
        "failed_attempts": sum(record["status"] != "complete" for record in all_attempts),
        "unknown_charge_attempts": sum(record.get("has_unknown_charges", True) for record in all_attempts),
        "prior_interrupted_attempts": len(interrupted),
        "acquisition_cost_complete": not interrupted
        and not any(record.get("has_unknown_charges", True) for record in all_attempts),
        "session": session.name,
        "session_observed_credits": credits,
        "session_attempts": sent,
        "elapsed_seconds": time.monotonic() - started,
        "fatal_error": fatal,
    }
    write(session / "collection.json", summary)
    save_latest(output / "collection.json", summary)
    if status != "complete":
        raise RuntimeError(
            "Collection incomplete; preserve receipts and inspect the recorded failure before continuing"
        )


def grade(directory, split, output=None):
    output = output or directory / f"grades-{split}.json"
    if output.exists():
        raise FileExistsError(output)
    plan, inputs, settings = checked_inputs(directory)
    gold = read(directory / "gold-for-scoring-only.json")
    if digest(gold) != plan["gold_sha256"]:
        raise ValueError("Answer key changed after preparation")
    collection = directory / f"answers-{split}"
    summary = read(collection / "collection.json")
    if summary["status"] != "complete":
        raise ValueError("Refuse an incomplete collection")
    collection_plan = read(collection / "collection-plan.json")
    selected = collection_plan["ids"]
    if (
        collection_plan["plan_sha256"] != file_hash(directory / "plan.json")
        or collection_plan["settings_sha256"] != digest(settings)
        or collection_plan["split"] != split
        or summary["completed"] != len(selected) * len(settings["arms"])
        or summary["failed"] != 0
    ):
        raise ValueError("Collection does not match the prepared plan and count")
    if not selected or selected != [r["id"] for r in inputs[split]][: len(selected)]:
        raise ValueError("Collection IDs differ from the prepared split order")
    if split == "test" and collection_plan.get("policies_frozen_sha256") != file_hash(
        directory / "policies-frozen.json"
    ):
        raise ValueError("Frozen policies changed after test collection")
    check_review(directory, selected)
    by_id = {r["id"]: r for r in inputs[split]}
    result = []
    for identity in selected:
        row = by_id[identity]
        outcome = {"id": identity}
        for arm in settings["arms"]:
            path = collection / f"{identity}-{arm}.json"
            answer = read(path)
            if (
                answer["status"] != "complete"
                or answer["settings_sha256"] != digest(settings)
                or answer["request_sha256"] != digest(row["request"])
                or answer["id"] != identity
                or answer["arm"] != arm
            ):
                raise ValueError("Answer does not match the frozen request and settings")
            validate_record(answer, row, arm, settings)
            measured = grade_reviewed(identity, gold[identity], answer["answer"])
            if measured["status"] in ("gold_unparseable", "grading_error"):
                raise ValueError(f"Grader could not safely score {identity}; inspect the source answer")
            outcome.update(
                {
                    f"{arm}_correct": measured["correct"],
                    f"{arm}_credits": answer["credits"],
                    f"{arm}_grade": measured,
                    f"{arm}_record_sha256": file_hash(path),
                }
            )
        result.append(outcome)
    write(
        output,
        {
            "split": split,
            "plan_sha256": file_hash(directory / "plan.json"),
            "collection_plan_sha256": file_hash(collection / "collection-plan.json"),
            "collection_summary_sha256": file_hash(collection / "collection.json"),
            "policies_frozen_sha256": collection_plan.get("policies_frozen_sha256"),
            "grader_source_sha256": file_hash(Path(__file__).with_name("solver_answers.py")),
            "gold_review_source_sha256": file_hash(Path(__file__).with_name("solver_gold.py")),
            "gold_review_sha256": file_hash(directory / "gold-review.json"),
            "rows": result,
        },
    )
    print(
        json.dumps(
            {
                "n": len(result),
                **{arm: sum(r[f"{arm}_correct"] for r in result) / len(result) for arm in settings["arms"]},
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    child = sub.add_parser("prepare")
    child.add_argument("--evidence", type=Path, default=Path("evidence"))
    child.add_argument("--source", type=Path, default=Path("run-output/solver-source"))
    child.add_argument("--output", type=Path, required=True)
    for command in ("collect", "grade"):
        child = sub.add_parser(command)
        child.add_argument("--run", type=Path, required=True)
        child.add_argument("--split", choices=("validation", "calibration", "test"), required=True)
        if command == "collect":
            child.add_argument("--limit", type=int)
            child.add_argument("--workers", type=int, default=4)
            child.add_argument("--max-credits", type=int, default=1000000)
            child.add_argument("--max-seconds", type=int, default=7200)
            child.add_argument("--self-hosted", action="store_true")
            child.add_argument(
                "--resume", action="store_true", help="Continue missing answers; preserve every earlier attempt"
            )
            child.add_argument(
                "--retry-unknown",
                action="store_true",
                help="Explicitly permit resubmission when prior execution or charge is unknown",
            )
            child.add_argument(
                "--max-attempts",
                type=int,
                default=3,
                help="Maximum recorded attempts per problem and solver across resumptions",
            )
        else:
            child.add_argument("--output", type=Path, help="Write grades to a new file, preserving archived grades")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.evidence, args.source, args.output)
    elif args.command == "collect":
        collect(
            args.run,
            args.split,
            args.limit,
            args.workers,
            args.max_credits,
            args.max_seconds,
            self_hosted=args.self_hosted,
            resume=args.resume,
            retry_unknown=args.retry_unknown,
            max_attempts=args.max_attempts,
        )
    else:
        grade(args.run, args.split, args.output)


if __name__ == "__main__":
    main()
