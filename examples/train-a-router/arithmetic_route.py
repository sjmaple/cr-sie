"""Answer a new arithmetic question with a saved router and the public SIE SDK."""

from __future__ import annotations

import math
import time
from pathlib import Path

import arithmetic_replay as public
import benefit_outcomes_native as native
import benefit_route
import solver
from data import digest, file_hash, render_request, write
from serving import SingleSendClient, error_receipt


def selected_model(study, run, population, method):
    root = public.checked_run(run, study) if run else public.reference(study)
    size = public.POPULATIONS[population]
    name = {
        "embedding": f"embedding_{size}", "lora": f"lora_{size}",
        "laya": "base", "teacher": "embedding_teacher_2000",
    }[method]
    if run and method != "laya":
        directory = Path(run)
        policy_path = directory / f"policy-{name}.json"
        policy = public.read(policy_path)
        model_path = Path(policy["model_path"])
        if (
            model_path.is_absolute() or ".." in model_path.parts
            or public.read(directory / "reproduction.json")["policy_files_sha256"].get(policy_path.name)
            != file_hash(policy_path)
            or policy.get("method") != name
            or policy.get("design") != ("native" if method == "lora" else "embedding")
            or (method != "teacher" and policy.get("training_rows") != size)
            or policy["study_plan_sha256"] != file_hash(directory / "plan.json")
            or policy["model_sha256"] != file_hash(directory / model_path)
            or policy["calibration_grades_sha256"] != file_hash(directory / "grades-calibration.json")
        ):
            raise ValueError("Saved live model or calibration policy changed")
    else:
        directory = root / "study"
        policy_path = root / "study-cost-controls/frozen.json" if method == "laya" else directory / "frozen.json"
        frozen = public.read(policy_path)
        policy = frozen["methods"][f"base_arithmetic_{size}" if method == "laya" else name]
    if method in ("embedding", "teacher"):
        path = directory / (f"embedding-{size}.json" if method == "embedding" else "teacher-models.json")
        record = public.read(path)
        head = record["head"] if method == "embedding" else record["heads"][name]
        return policy, policy_path, head, None, None
    directory = root / "reference/native" if method == "laya" else directory / f"native-{size}"
    trained = public.read(directory / "trained.json")
    variant = "base" if method == "laya" else "lora"
    if method == "laya":
        trained = trained | {"conditional_cost_statistics": policy["conditional_cost_statistics"]}
    return policy, policy_path, None, directory / variant, trained


def choose_native(policy, checkpoint, trained, request, target, device):
    selected = policy["policies"][str(target)]
    score, load_seconds, verification_seconds = None, None, None
    started = time.perf_counter()
    if selected["mode"] == "threshold":
        verification_started = time.perf_counter()
        for name, expected in trained["checkpoints"][checkpoint.name]["files_sha256"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or file_hash(checkpoint / relative) != expected:
                raise ValueError("Selected native checkpoint differs; download the optional weights first")
        verification_seconds = time.perf_counter() - verification_started
        scoring_started = time.perf_counter()
        score = float(native.score_checkpoint(
            checkpoint, [{"request": request}], device, trained["conditional_cost_statistics"]
        )[0])
        if not math.isfinite(score):
            raise ValueError("Native router returned an invalid score")
        load_seconds = time.perf_counter() - scoring_started
        arm = "strong" if score >= selected["cutoff"] else "cheap"
    elif selected["mode"] in ("always_strong", "always_cheap"):
        arm = selected["mode"].removeprefix("always_")
    else:
        raise ValueError("Unknown saved routing rule")
    return {
        "router": "laya_outcomes_" + checkpoint.name,
        "arm": arm, "model": solver.SETTINGS["arms"][arm]["model"],
        "mode": selected["mode"], "score": score, "cutoff": selected["cutoff"],
        "calibration_quality_loss_target_pp": target,
        "embedding_receipt": None, "native_model_calls": int(load_seconds is not None),
        "local_checkpoint_verification_seconds": verification_seconds,
        "local_model_load_and_score_seconds": load_seconds,
        "local_router_end_to_end_seconds": time.perf_counter() - started,
        "latency_note": "Fresh CLI timings include checkpoint verification and loading; they are not warm server latency.",
        "cost_note": "Local Laya compute is unpriced; API credits cover answer generation only.",
    }


def answer(
    study, output, question, *, run=None, population="expanded", method="embedding",
    target=1, self_hosted=False, device="cpu",
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not question.strip() or len(question) > 20000 or target not in (0, 1, 2, 5):
        raise ValueError("Use a nonempty question up to 20,000 characters and a declared calibration target")
    if method not in ("embedding", "lora", "laya", "teacher") or device not in ("cpu", "cuda"):
        raise ValueError("Choose a supported router and CPU or CUDA device")
    policy, policy_path, head, checkpoint, trained = selected_model(study, run, population, method)
    request = render_request("GSM8K", question)
    row = {"id": "new-" + digest(request)[:16], "request": request}
    write(output / "intent.json", {
        "id": row["id"], "request_sha256": digest(request), "policy_sha256": file_hash(policy_path),
        "solver_settings_sha256": digest(solver.SETTINGS), "target_pp": target,
        "method": method, "self_hosted": self_hosted, "automatic_retries": False,
    })
    try:
        routing = (
            benefit_route.choose(
                {"selections": policy["policies"]}, request, target, self_hosted=self_hosted,
                receipt_path=output / "embedding-receipt.json", outcome_head=head,
            )
            if head is not None
            else choose_native(policy, checkpoint, trained, request, target, device)
        )
        write(output / "routing.json", routing)
        with SingleSendClient(timeout_s=60) as client:
            metadata = client.get_model(routing["model"])
            if (
                metadata.get("revision") != solver.SETTINGS["arms"][routing["arm"]]["revision"]
                or metadata.get("name", metadata.get("id")) != routing["model"]
            ):
                raise ValueError("Live answer model differs from the evaluated model")
        record = solver.solve(row, routing["arm"], solver.SETTINGS, self_hosted=self_hosted)
        solver.validate_record(record, row, routing["arm"], solver.SETTINGS)
        write(output / "answer.json", record)
        if record["status"] != "complete":
            raise RuntimeError("Answer request failed; inspect the saved receipt before another attempt")
    except Exception as error:
        write(output / "incomplete.json", error_receipt(error))
        raise
    summary = {
        "model": routing["model"], "answer": record["answer"], "solver_calls": len(record["calls"]),
        "embedding_calls": int(routing["embedding_receipt"] is not None),
        "native_model_calls": routing.get("native_model_calls", 0),
        "total_api_credits": None if self_hosted else record["credits"] + (routing["embedding_receipt"] or {}).get("credits", 0),
        "local_compute_cost": "unpriced",
        "answer_quality": "Not graded; no reference answer was supplied for this new question.",
    }
    write(output / "result.json", summary)
    return summary
