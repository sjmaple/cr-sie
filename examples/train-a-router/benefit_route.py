"""Use a frozen benefit router to answer one new question through SIE.

The selected calibration rule is fixed. A request needs at most one embedding,
one answer-model call and one conditional final-answer call. There are no
automatic retries. Cloud and self-hosted servers use the same public SDK.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sie_sdk.types import Item

import benefit
import benefit_native
import benefit_outcomes
import benefit_outcomes_native
import solver
from data import digest, file_hash, render_request, write
from serving import INSTRUCTION, MODEL, REVISION, SingleSendClient, error_receipt, normalize, usage

CONTESTS = ("AMC8", "AMC10", "AMC12", "AIME", "HMMT-Feb", "HMMT-Nov")


def embed(request, receipt_path=None):
    """Save the billing receipt before validating a successful response body."""
    with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
        metadata = client.get_model(MODEL)
        if metadata.get("revision") != REVISION or metadata.get("name", metadata.get("id")) != MODEL:
            raise ValueError("Live encoder identity or revision differs")
        reply = client.encode(
            MODEL,
            Item(text=request),
            output_types=["dense"],
            instruction=INSTRUCTION,
            is_query=True,
            wait_for_capacity=False,
            max_oom_retries=0,
            provision_timeout_s=60,
        )
        receipt = usage(reply)
        if receipt_path is not None:
            write(receipt_path, receipt)
        vector = normalize(reply["dense"])
        if vector.shape != (2560,):
            raise ValueError("Wrong vector shape")
    return vector, receipt


def choose(frozen, request, target, *, self_hosted=False, receipt_path=None, outcome_head=None):
    """Select one model; bypass embedding calls for an explicit static policy."""
    policy = frozen["selections"][str(target)]
    mode, score, receipt = policy["mode"], None, None
    if mode == "threshold":
        vector, receipt = embed(request, receipt_path)
        if not self_hosted and not solver.valid_credit(receipt.get("credits")):
            raise ValueError("Cloud embedding has no usable charge receipt")
        vector = np.asarray(vector, dtype=float)
        if outcome_head is not None:
            score = float(benefit_outcomes.score(vector[None, :], outcome_head)[0])
        else:
            weights = np.asarray(frozen["head"]["weights"], dtype=float)
            if vector.shape != weights.shape or not np.isfinite(vector).all():
                raise ValueError("Embedding does not match the trained head")
            score = float(vector @ weights + frozen["head"]["bias"])
        if not np.isfinite(score):
            raise ValueError("The trained head returned an invalid score")
        arm = "strong" if score >= policy["cutoff"] else "cheap"
    elif mode in ("always_strong", "always_cheap"):
        arm = mode.removeprefix("always_")
    else:
        raise ValueError("Unknown frozen routing rule")
    return {
        "router": "qwen_embedding_outcomes" if outcome_head is not None else "qwen_embedding_ridge",
        "arm": arm,
        "model": solver.SETTINGS["arms"][arm]["model"],
        "mode": mode,
        "score": score,
        "cutoff": policy["cutoff"],
        "calibration_quality_loss_target_pp": target,
        "embedding_receipt": receipt,
        "note": "The target describes calibration results, not a guarantee for this answer.",
    }


def choose_native(native_run, head_run, request, target, variant, device, *, outcome_mode=False):
    """Apply a selected native model with its original calibration cutoff."""
    if variant not in ("base", "lora") or device not in ("cpu", "mps", "cuda"):
        raise ValueError("Choose a native variant and supported local device")
    routing_started = time.perf_counter()
    trained = None
    if outcome_mode:
        _, trained, frozen = benefit_outcomes_native.checked_frozen(native_run, head_run)
    else:
        _, frozen = benefit_native.checked_frozen(native_run, head_run)
    model = frozen["models"][variant]
    policy = model["policies"][str(target)]
    score, elapsed, verification_elapsed = None, None, None
    if policy["mode"] == "threshold":
        checkpoint = Path(native_run) / variant
        verification_started = time.perf_counter()
        # Metadata validation above checks all manifest paths/hashes. Only the
        # selected optional checkpoint needs to be downloaded for live routing.
        manifest = (
            trained["checkpoints"][variant]["files_sha256"] if trained else model["checkpoint_files_sha256"]
        )
        for relative, expected in manifest.items():
            if file_hash(checkpoint / relative) != expected:
                raise ValueError("Selected native checkpoint bytes differ from the frozen manifest")
        verification_elapsed = time.perf_counter() - verification_started
        started = time.perf_counter()
        score = (
            benefit_outcomes_native.score_checkpoint(
                checkpoint, [{"request": request}], device, trained["conditional_cost_statistics"]
            )[0]
            if trained
            else benefit_native.score_checkpoint(checkpoint, [{"request": request}], device)[0]
        )
        elapsed = time.perf_counter() - started
        arm = "strong" if score >= policy["cutoff"] else "cheap"
    elif policy["mode"] in ("always_strong", "always_cheap"):
        arm = policy["mode"].removeprefix("always_")
    else:
        raise ValueError("Unknown native routing rule")
    return {
        "router": "laya_outcomes_" + variant if outcome_mode else "laya_" + variant,
        "arm": arm,
        "model": solver.SETTINGS["arms"][arm]["model"],
        "mode": policy["mode"],
        "score": score,
        "cutoff": policy["cutoff"],
        "calibration_quality_loss_target_pp": target,
        "embedding_receipt": None,
        "native_model_calls": int(elapsed is not None),
        "native_questions_per_request": 2 if outcome_mode else 1,
        "local_checkpoint_verification_seconds": verification_elapsed,
        "local_model_load_and_score_seconds": elapsed,
        "local_router_end_to_end_seconds": time.perf_counter() - routing_started,
        "latency_note": "Fresh CLI timing includes metadata checks, selected-checkpoint hashing and cold model loading; it is not persistent-server inference latency.",
        "cost_note": "Local native compute is unpriced; reported credits cover the answer API only.",
        "note": "The target describes calibration results, not a guarantee for this answer.",
    }


def answer(
    run, output, contest, question, target=1, *, self_hosted=False, native_run=None, variant="lora", device="cpu"
):
    if output.exists():
        raise FileExistsError(output)
    if contest not in CONTESTS or not question.strip() or len(question) > 20000 or target not in benefit.TARGETS:
        raise ValueError(
            "Provide a supported competition, a nonempty question of at most 20,000 characters, and a declared target"
        )
    outcome_mode = benefit.read(run / "frozen.json").get("format") == benefit_outcomes.FROZEN_FORMAT
    outcome_head = None
    if outcome_mode:
        frozen, selected = benefit_outcomes.checked_frozen(run)
        plan, outcome_head = selected["plan"], selected["head"]
    else:
        frozen = benefit._checked_frozen(run)
        plan = frozen["plan"]
    if plan["solver_settings_sha256"] != digest(solver.SETTINGS):
        raise ValueError("Answer-model settings differ from the training and evaluation protocol")
    request = render_request(contest, question)
    row = {"id": "new-" + digest(request)[:16], "request": request}
    write(
        output / "intent.json",
        {
            "id": row["id"],
            "request_sha256": digest(request),
            "frozen_sha256": file_hash(run / "frozen.json"),
            "target_pp": target,
            "self_hosted": self_hosted,
            "router": "laya_" + variant if native_run else "qwen_embedding_outcomes" if outcome_mode else "qwen_embedding_ridge",
            "outcome_model": outcome_mode,
            "automatic_retries": False,
        },
    )
    try:
        routing = (
            choose_native(native_run, run, request, target, variant, device, outcome_mode=outcome_mode)
            if native_run
            else choose(
                frozen,
                request,
                target,
                self_hosted=self_hosted,
                receipt_path=output / "embedding-receipt.json",
                outcome_head=outcome_head,
            )
        )
        write(output / "routing.json", routing)
        with SingleSendClient(timeout_s=60) as client:
            metadata = client.get_model(routing["model"])
            if (
                metadata.get("revision") != solver.SETTINGS["arms"][routing["arm"]]["revision"]
                or metadata.get("name", metadata.get("id")) != routing["model"]
            ):
                raise ValueError("Answer-model identity or revision differs from the evaluated model")
        record = solver.solve(row, routing["arm"], solver.SETTINGS, self_hosted=self_hosted)
        solver.validate_record(record, row, routing["arm"], solver.SETTINGS)
        write(output / "answer.json", record)
        if record["status"] != "complete":
            raise RuntimeError("Answer collection failed; inspect the preserved receipt before another attempt")
    except Exception as error:
        write(output / "incomplete.json", error_receipt(error))
        raise
    embedding_charge = (routing["embedding_receipt"] or {}).get("credits", 0)
    summary = {
        "model": routing["model"],
        "answer": record["answer"],
        "solver_calls": len(record["calls"]),
        "embedding_calls": int(routing["embedding_receipt"] is not None),
        "native_model_calls": routing.get("native_model_calls", 0),
        "total_api_credits": None if self_hosted else record["credits"] + embedding_charge,
        "local_compute_cost": "unpriced",
        "answer_quality": "Not graded; no reference answer was supplied for this new question",
    }
    write(output / "result.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Frozen embedding router directory")
    parser.add_argument("--output", required=True, type=Path, help="New directory for this request and its receipts")
    parser.add_argument("--contest", required=True, choices=CONTESTS)
    parser.add_argument("--question", required=True)
    parser.add_argument("--target", type=int, choices=benefit.TARGETS, default=1)
    parser.add_argument("--self-hosted", action="store_true")
    parser.add_argument("--native-run", type=Path, help="Use local frozen Laya instead of the embedding router")
    parser.add_argument("--variant", choices=("base", "lora"), default="lora")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            answer(
                args.run,
                args.output,
                args.contest,
                args.question,
                args.target,
                self_hosted=args.self_hosted,
                native_run=args.native_run,
                variant=args.variant,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
