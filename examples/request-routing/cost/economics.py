"""Measure answer costs and teacher-supervised routing without gold training labels.

Gold answers are exported separately and never read by the inference collector.
Each API request contains one problem. Failed sends are retained and not retried.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import learn
from cloud import digest, usage_of
from run import SingleSendClient

CHEAP = "Qwen/Qwen3.5-4B"
STRONG = "Qwen/Qwen3.8-27B-FP8"
PROMPT = (
    "Solve the mathematical problem. Give concise but sufficient working, then put your final answer "
    "in \\boxed{...}. Use the mathematical value, not a multiple-choice letter. "
    "Do not give several alternative answers unless the problem asks for all solutions."
)
FAST_PROMPT = (
    "Solve the mathematical problem and give ONLY the final mathematical answer in \\boxed{...}. "
    "Use the mathematical value, not a multiple-choice letter. Do not include explanations."
)
ARMS = {
    "cheap": {"model": CHEAP, "max_tokens": 1536, "system": PROMPT},
    "fast": {"model": CHEAP, "max_tokens": 256, "system": FAST_PROMPT},
    "strong": {"model": STRONG, "max_tokens": 3072, "system": PROMPT},
}
ANSWER_ARMS = {
    "budget_cheap": {
        "model": CHEAP,
        "max_tokens": 1536,
        "system": "Solve the mathematical problem. Use at most 150 words of concise calculations, then put the final mathematical value in \\boxed{...}. Avoid introductions, repetition, alternative approaches, and restating the problem. If uncertain, give your best final answer. Do not use a multiple-choice letter.",
        "finalize_on_length_or_no_box": True,
        "final_tokens": 256,
        "final_format": "json",
    },
    "budget_strong": {
        "model": STRONG,
        "max_tokens": 3072,
        "system": "Solve the mathematical problem carefully. Use at most 300 words of concise calculations, then put the final mathematical value in \\boxed{...}. Avoid introductions, repetition, alternative approaches, and restating the problem. If uncertain, give your best final answer. Do not use a multiple-choice letter.",
        "finalize_on_length_or_no_box": True,
        "final_tokens": 256,
        "final_format": "json",
    },
}
FINAL_PROMPT = (
    "Your working budget is exhausted. Conclude now using the work above. "
    "Give ONLY your best final mathematical answer in \\boxed{...}, with no further working or explanation."
)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def failure_metadata(error):
    """Retain only public status/code fields from the SDK's rejected response."""
    result = {"status": "failed_or_unknown", "error_type": type(error).__name__}
    trace = error.__traceback__
    while trace is not None:
        response = trace.tb_frame.f_locals.get("response")
        if response is not None and hasattr(response, "status_code"):
            result["http_status"] = response.status_code
            try:
                code = response.json().get("error", {}).get("code")
                if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", code):
                    result["error_code"] = code
            except (ValueError, AttributeError):
                pass
        trace = trace.tb_next
    return result


def recovered_rejections(directory):
    path = directory / "admission-recovery.json"
    return set(read(path)["approved_record_sha256"]) if path.exists() else set()


def frozen_policies(directory):
    """Resolve exactly the policy comparisons committed before test collection."""
    plan = read(directory / "evaluation-plan.json")
    settings, inputs = read(directory / "answer-settings.json"), read(directory / "inputs.json")
    if plan["answer_settings_sha256"] != digest(settings) or plan["inputs_sha256"] != digest(inputs):
        raise ValueError("Evaluation inputs changed after freeze")
    groups = {}
    for name, entry in plan["groups"].items():
        policy = read(directory / entry["file"])
        if digest(policy) != entry["sha256"] or policy.get("development_only"):
            raise ValueError("Evaluation policy changed after freeze")
        if policy["answer_settings_sha256"] != digest(settings) or policy["inputs_sha256"] != digest(inputs):
            raise ValueError("Evaluation policy belongs to different inputs")
        groups[name] = policy
    return plan, groups


def configure(directory):
    """Freeze the recorded answering recipe for a new, explicitly started run."""
    path = directory / "answer-settings.json"
    if path.exists():
        raise FileExistsError(path)
    result = {
        "plan_sha256": digest(read(directory / "plan.json")),
        "arms": ANSWER_ARMS,
        "cheap_arm": "budget_cheap",
        "strong_arm": "budget_strong",
        "router_credits_per_request": 2,
        "router_cost_note": "Validation allowance only; final evaluation uses actual singleton receipts.",
        "max_inflight": 4,
        "max_observed_credits": 1000000,
        "max_seconds_per_collection": 7200,
        "recipe": "Recorded bounded-working recipe; repeatability run without pilot prompt selection.",
        "teacher_labels": "Mathematical answer agreement with 27B; router training and cutoff selection never read gold answer keys.",
    }
    learn.write(path, result)
    return result


def prepare(directory, reference, posttrain):
    import pyarrow.parquet as pq

    data = read(reference / "main/data.json")
    raw = {split: pq.read_table(reference / f"source/{split}.parquet").to_pylist() for split in ("train", "eval")}
    splits = {"train": data["splits"]["train"], "validation": data["splits"]["validation"]}
    splits["test"] = read(posttrain / "confirmation.json")
    gold = {}
    for items in splits.values():
        for row in items:
            source, number = row["id"].removeprefix("e2h-amc-").rsplit("-", 1)
            original = raw[source][int(number)]
            if original["problem"].strip() != row["request"]:
                raise ValueError("Source problem mismatch")
            gold[row["id"]] = original["answer"]
    directory.mkdir(parents=True, exist_ok=False)
    learn.write(directory / "inputs.json", splits)
    learn.write(directory / "gold-for-independent-scoring-only.json", gold)
    plan = {
        "source_revision": data["source_revision"],
        "inputs_sha256": digest(splits),
        "gold_sha256": digest(gold),
        "counts": {k: len(v) for k, v in splits.items()},
        "pilot_ids": [r["id"] for r in splits["train"][:16]],
        "pilot_arms": ARMS,
        "inference": {"temperature": 0, "one_problem_per_request": True, "automatic_retries": 0},
        "max_observed_credits": 300000,
        "credit_usd": 0.00001,
        "credit_conversion_source": "https://superlinked.com/pricing/sie-cloud-prices.json",
        "max_inflight": 4,
        "max_seconds_per_collection": 3600,
        "primary_quality_loss_pp": 1,
        "secondary_quality_loss_pp": [2, 5],
        "teacher_labels": "Answer agreement with the 27B teacher, verified mathematically; no gold labels used for training or validation selection",
        "evaluation": "Independent source answer keys and frozen old difficulty routers; confirmation inputs reused, answer outcomes newly collected",
        "selection": "Choose answer settings from the 16 development pilot cases; freeze before remaining collection. Freeze learned routing thresholds before test answers are collected or scored.",
        "costs": "Actual per-request billed credits including failures; router overhead separate; training acquisition amortized separately",
        "success": "At least 25% total serving cost saving vs always strong, at most 1 percentage point observed answer-accuracy loss; paired 95% intervals reported without claiming noninferiority unless supported",
    }
    learn.write(directory / "plan.json", plan)
    return plan


def native_usage(reply):
    return {
        "billing": "self_hosted",
        "usage": {
            k: v
            for k, v in reply.get("usage", {}).items()
            if k in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }


def send(row, arm_name, arm, *, self_hosted=False):
    messages = [{"role": "system", "content": arm["system"]}, {"role": "user", "content": row["request"]}]
    result = {
        "id": row["id"],
        "arm": arm_name,
        "model": arm["model"],
        "input_sha256": digest(messages),
        "arm_settings_sha256": digest(arm),
    }
    started = time.monotonic()
    try:
        with SingleSendClient(timeout_s=180, read_timeout_s=180) as client:
            reply = client.chat_completions(
                model=arm["model"],
                messages=messages,
                temperature=0,
                max_tokens=arm["max_tokens"],
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=180,
                read_timeout_s=180,
            )
            # Keep only public generated content, usage and opaque request IDs.
            result.update(native_usage(reply) if self_hosted else usage_of(reply))
            result["model_revision"] = client.last_model_revision
            choice = reply["choices"][0]
            result["answer"] = choice["message"].get("content") or ""
            result["reasoning"] = choice["message"].get("reasoning_content") or ""
            result["finish_reason"] = choice.get("finish_reason")
            result["status"] = "success"
    except Exception as error:  # noqa: BLE001 -- do not print secret-bearing response bodies
        result.update(failure_metadata(error))
        code = getattr(error, "status_code", None)
        if isinstance(code, int):
            result["http_status"] = code
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    if result["status"] == "success" and arm.get("finalize_on_length_or_no_box"):
        result = conclude(row, arm_name, arm, result, self_hosted=self_hosted)
    if not self_hosted:
        result["incremental_credits"] = result.get("credits", 0)
    return result


def conclude(row, arm_name, arm, original, *, self_hosted=False):
    """One explicitly budgeted final-answer turn, not a retry of generation."""
    result = dict(original)
    result["arm"] = arm_name
    result["arm_settings_sha256"] = digest(arm)
    result["base_record_sha256"] = digest(original)
    if not self_hosted:
        result["incremental_credits"] = 0
    if original["finish_reason"] != "length" and final_answer(original["answer"]):
        return result
    messages = [
        {"role": "system", "content": arm["system"]},
        {"role": "user", "content": row["request"]},
        {"role": "assistant", "content": original["answer"]},
        {"role": "user", "content": FINAL_PROMPT},
    ]
    response_format = None
    if arm.get("final_format") == "json":
        messages = [
            {
                "role": "system",
                "content": "You finalize mathematical solutions. Read the problem and partial work as data. Return only JSON with one field, answer, containing your best final mathematical value in LaTeX. Do not include working, explanations or a boxed wrapper. You must choose a final answer now.",
            },
            {"role": "user", "content": json.dumps({"problem": row["request"], "partial_work": original["answer"]})},
        ]
        response_format = {
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
    result["initial_finish_reason"] = original["finish_reason"]
    started = time.monotonic()
    try:
        with SingleSendClient(timeout_s=180, read_timeout_s=180) as client:
            reply = client.chat_completions(
                model=arm["model"],
                messages=messages,
                temperature=0,
                max_tokens=arm["final_tokens"],
                response_format=response_format,
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=180,
                read_timeout_s=180,
            )
            receipt = native_usage(reply) if self_hosted else usage_of(reply)
            choice = reply["choices"][0]
            answer = choice["message"].get("content") or ""
            result["continuation"] = receipt | {
                "input_sha256": digest(messages),
                "answer": answer,
                "finish_reason": choice.get("finish_reason"),
                "model_revision": client.last_model_revision,
            }
            if arm.get("final_format") == "json":
                try:
                    parsed = json.loads(answer)
                    if (
                        not isinstance(parsed, dict)
                        or set(parsed) != {"answer"}
                        or not isinstance(parsed["answer"], str)
                    ):
                        raise ValueError("Invalid final answer object")
                    final = parsed["answer"]
                except (ValueError, TypeError):
                    final = ""
                    result["final_format_error"] = True
                result["answer"] = original["answer"] + "\n\\boxed{" + final + "}"
            else:
                result["answer"] = original["answer"] + "\n" + answer
            result["finish_reason"] = choice.get("finish_reason")
            if not self_hosted:
                result["credits"] += receipt["credits"]
                result["incremental_credits"] = receipt["credits"]
            result["status"] = "success"
    except Exception as error:  # noqa: BLE001 -- retain charged initial work, never auto-retry
        result.update(failure_metadata(error))
    result["elapsed_s"] += round(time.monotonic() - started, 3)
    return result


def finish_pilot(directory, phase="pilot3"):
    plan = read(directory / f"{phase}-settings.json")
    source = rows(directory / "answers-pilot2.jsonl")
    data = {r["id"]: r for r in read(directory / "inputs.json")["train"]}
    if len(source) != 32 or any(r["status"] != "success" for r in source):
        raise ValueError("Concise pilot must finish successfully before its bounded-finalization follow-up")
    with (directory / f"answers-{phase}.jsonl").open("x") as handle, ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(source), 4):
            tasks = []
            for original in source[offset : offset + 4]:
                name = plan["source_arm_mapping"][original["arm"]]
                tasks.append((data[original["id"]], name, plan["arms"][name], original))
            completed = list(pool.map(lambda task: conclude(*task), tasks))
            for result in completed:
                handle.write(json.dumps(result) + "\n")
                handle.flush()
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "completed": offset + len(completed),
                        "incremental_credits": sum(r["incremental_credits"] for r in completed),
                    }
                ),
                flush=True,
            )
            if any(r["status"] != "success" for r in completed):
                raise RuntimeError("Finalization failed; retained all receipts; no retries")


def collect(directory, phase, limit=None, attempt=1):
    plan, inputs = read(directory / "plan.json"), read(directory / "inputs.json")
    if digest(inputs) != plan["inputs_sha256"]:
        raise ValueError("Inputs changed")
    previous = [r for p in directory.glob("answers-*.jsonl") for r in rows(p)]
    recovered = recovered_rejections(directory)
    if any(r["status"] != "success" and digest(r) not in recovered for r in previous):
        raise ValueError("Prior failed/unknown request requires explicit investigation, not an automatic retry")
    credits = sum(r.get("incremental_credits", r.get("credits", 0)) for r in previous)
    max_inflight = plan["max_inflight"]
    max_credits = plan["max_observed_credits"]
    max_seconds = plan["max_seconds_per_collection"]
    if phase in ("pilot", "pilot2"):
        arms = plan["pilot_arms"] if phase == "pilot" else read(directory / "pilot2-settings.json")["arms"]
        items = [r for r in inputs["train"] if r["id"] in plan["pilot_ids"]]
    else:
        frozen = read(directory / "answer-settings.json")
        if frozen["plan_sha256"] != digest(plan):
            raise ValueError("Plan changed after answer settings freeze")
        arms = frozen["arms"]
        items = inputs[phase]
        max_inflight = frozen.get("max_inflight", max_inflight)
        max_credits = frozen.get("max_observed_credits", max_credits)
        max_seconds = frozen.get("max_seconds_per_collection", max_seconds)
        if phase == "test":
            frozen_policies(directory)
    if limit is not None:
        if phase == "test" or limit <= 0 or limit > len(items):
            raise ValueError("Development limits must be positive and within the split; no partial test peeking")
        items = items[:limit]
    output_phase = phase if limit is None else f"{phase}-{limit}"
    if attempt != 1:
        output_phase += f"-attempt{attempt}"
    if (directory / "admission-recovery.json").exists():
        max_inflight = min(max_inflight, read(directory / "admission-recovery.json")["max_inflight"])
    prior = {(r["id"], r["arm"]): r for r in previous if r["status"] == "success"}
    known_arms = dict(plan["pilot_arms"])
    if (directory / "pilot2-settings.json").exists():
        known_arms.update(read(directory / "pilot2-settings.json")["arms"])
    if (directory / "pilot3-settings.json").exists():
        known_arms.update(read(directory / "pilot3-settings.json")["arms"])
    if (directory / "pilot4-settings.json").exists():
        known_arms.update(read(directory / "pilot4-settings.json")["arms"])
    if phase not in ("pilot", "pilot2"):
        known_arms.update(arms)
    tasks = []
    reused = []
    for row in items:
        for name, arm in arms.items():
            old = prior.get((row["id"], name))
            expected = digest(
                [{"role": "system", "content": arm["system"]}, {"role": "user", "content": row["request"]}]
            )
            original_arm = known_arms.get(name)
            if old is not None and old["input_sha256"] == expected and original_arm == arm:
                if old.get("arm_settings_sha256", digest(arm)) != digest(arm):
                    raise ValueError("Cached answer token/finalization settings changed")
                reused.append([row["id"], name])
            else:
                tasks.append((row, name, arm))
    started = time.monotonic()
    count = 0
    failures = []
    budget_reached = False
    with (
        (directory / f"answers-{output_phase}.jsonl").open("x") as handle,
        ThreadPoolExecutor(max_workers=max_inflight) as pool,
    ):
        remaining = iter(tasks)
        active = set()
        exhausted = False
        while active or not exhausted:
            budget_reached = credits >= max_credits or time.monotonic() - started >= max_seconds
            while not failures and not budget_reached and not exhausted and len(active) < max_inflight:
                task = next(remaining, None)
                if task is None:
                    exhausted = True
                else:
                    active.add(pool.submit(send, *task))
            if not active:
                break
            completed, active = wait(active, return_when=FIRST_COMPLETED)
            # After a failure or budget stop, drain and retain every in-flight
            # result. No new work is submitted and no paid result is discarded.
            for future in completed:
                result = future.result()
                handle.write(json.dumps(result, allow_nan=False) + "\n")
                handle.flush()
                credits += result.get("incremental_credits", result.get("credits", 0))
                count += 1
                if result["status"] != "success":
                    failures.append({k: result[k] for k in ("id", "arm", "error_type", "http_status") if k in result})
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "completed": count,
                        "total": len(tasks),
                        "observed_credits_all_phases": credits,
                        "failures": failures,
                    }
                ),
                flush=True,
            )
    receipt = {
        "phase": phase,
        "completed": count,
        "planned": len(tasks),
        "reused": reused,
        "failures": failures,
        "budget_reached": budget_reached,
        "max_inflight": max_inflight,
        "credits_all_phases": credits,
        "elapsed_s": time.monotonic() - started,
    }
    learn.write(directory / f"collection-{output_phase}.json", receipt)
    if failures or count != len(tasks):
        raise RuntimeError("Incomplete collection; see retained receipts")
    return receipt


def clock_value(text):
    text = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", text).replace("\\", "")
    match = re.fullmatch(r"\s*(\d{1,2})\s*:\s*(\d{1,2})\s*(?:[AP]\.?\s*M\.?)?\s*", text, re.IGNORECASE)
    if match and 0 <= int(match[1]) <= 23 and 0 <= int(match[2]) < 60:
        # The source's clock key omits AM/PM, so compare the clock face only.
        return int(match[1]) % 12, int(match[2])
    return None


def grade(answer, gold, problem=""):
    from math_verify import LatexExtractionConfig, parse, verify

    final = final_answer(answer)
    if re.search(r"what is the (?:actual )?time\?", problem, re.IGNORECASE) and ":" in gold:
        reference, candidate = clock_value(gold), clock_value(final or "")
        return {
            "correct": reference is not None and candidate == reference,
            "reference_parsed": reference is not None,
            "answer_parsed": candidate is not None,
        }
    # Parenthesize factorial powers: the source's 15 !^{2} means (15!)^2.
    gold = re.sub(r"(\d+)\s*!\s*(\^\{[^{}]+\})", r"(\1!)\2", gold)
    reference = parse(
        "$" + gold.strip().strip("$") + "$", extraction_config=[LatexExtractionConfig()], fallback_mode="no_fallback"
    )
    candidate = (
        parse("$" + final + "$", extraction_config=[LatexExtractionConfig()], fallback_mode="no_fallback")
        if final
        else []
    )
    return {
        "correct": bool(reference and candidate and verify(reference, candidate)),
        "reference_parsed": bool(reference),
        "answer_parsed": bool(candidate),
    }


def final_answer(text):
    """Only the last complete boxed answer counts, never an intermediate number."""
    offset = text.rfind("\\boxed{")
    if offset < 0:
        return None
    start = offset + len("\\boxed{")
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index].strip()
    return None


def score(directory, phase):
    gold = read(directory / "gold-for-independent-scoring-only.json")
    plan = read(directory / "plan.json")
    if digest(gold) != plan["gold_sha256"]:
        raise ValueError("Answer keys changed")
    records = rows(directory / f"answers-{phase}.jsonl")
    problems = {r["id"]: r["request"] for split in read(directory / "inputs.json").values() for r in split}
    graded = [
        {"id": r["id"], "arm": r["arm"], **grade(r["answer"], gold[r["id"]], problems[r["id"]])}
        for r in records
        if r["status"] == "success"
    ]
    result = {}
    for arm in sorted({r["arm"] for r in records}):
        selected = [r for r in graded if r["arm"] == arm]
        calls = [r for r in records if r["arm"] == arm and r["status"] == "success"]
        result[arm] = {
            "n": len(selected),
            "correct": sum(r["correct"] for r in selected),
            "unparsed_gold": sum(not r["reference_parsed"] for r in selected),
            "unparsed_answer": sum(not r["answer_parsed"] for r in selected),
            "length_stops": sum(r["finish_reason"] == "length" for r in calls),
            "mean_credits": sum(r["credits"] for r in calls) / max(1, len(calls)),
        }
    return {"summary": result, "graded": graded}


def answer_records(directory, ids, arms):
    """Require one successful retained answer for every requested arm and ID."""
    found = {}
    recovered = recovered_rejections(directory)
    for path in sorted(directory.glob("answers-*.jsonl")):
        for row in rows(path):
            key = (row["id"], row["arm"])
            if row["id"] not in ids or row["arm"] not in arms:
                continue
            if row["status"] != "success" and digest(row) in recovered:
                continue
            if key in found or row["status"] != "success":
                raise ValueError("Duplicate or unsuccessful answer")
            expected = digest(
                [
                    {"role": "system", "content": arms[row["arm"]]["system"]},
                    {"role": "user", "content": ids[row["id"]]},
                ]
            )
            if expected != row["input_sha256"]:
                raise ValueError("Answer was generated with different settings")
            if row.get("arm_settings_sha256", digest(arms[row["arm"]])) != digest(arms[row["arm"]]):
                raise ValueError("Answer token/finalization settings changed")
            found[key] = row
    if len(found) != len(ids) * len(arms):
        raise ValueError("Answers missing")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "configure", "collect", "score"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "pilot2", "train", "validation", "test"), default="pilot")
    parser.add_argument("--reference", type=Path, default=Path("evidence/difficulty-policy"))
    parser.add_argument("--limit", type=int, help="Development-only prefix collection; test runs are always complete")
    parser.add_argument(
        "--attempt", type=int, default=1, help="Explicitly named recovery after investigating retained failures"
    )
    parser.add_argument("--posttrain", type=Path, default=Path("evidence/posttrain"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.run, args.reference, args.posttrain)
    elif args.command == "configure":
        result = configure(args.run)
    elif args.command == "collect":
        result = collect(args.run, args.phase, args.limit, args.attempt)
    else:
        result = score(args.run, args.phase)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
