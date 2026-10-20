"""Train native Laya on paired solver benefit, with a separate test-access gate.

Preparation and reporting need only core dependencies. Training uses the pinned
optional training environment on MPS or CUDA; no solver API is called here.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import benefit
import finetune
import solver_policies
import solver_report
from data import digest, file_hash, write

QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Predict which solver is more likely to answer this mathematics problem correctly. Compare the cheaper Qwen3.5-4B solver with the stronger Qwen3.8-27B solver. Judge the expected benefit of the stronger solver; do not solve the problem.",
        "criteria": {
            "cheap": "The cheaper solver is at least as likely to produce the correct answer.",
            "strong": "The stronger solver is more likely to produce the correct answer.",
        },
    }
}
PROTOCOL = {
    "target": "Signed paired benefit y=strong_correct-cheap_correct; soft target p_strong=(1+y)/2, including ties=0.5",
    "base": finetune.BASE,
    "revision": finetune.REVISION,
    "recipe": finetune.LORA_RECIPES["original"],
    "seed": 89,
    "selection_fold": 0,
    "selection_epochs": 8,
    "selection": "Minimum training-fold validation MSE of tanh((strong_logit-cheap_logit)/2) versus signed benefit; include epoch0; ties retain earlier epoch",
    "refit": "Fresh base and seed89, all eligible training rows, selected epoch count; cosine schedule spans the selected refit duration; epoch0 skips training",
    "score_order": ["cheap", "strong"],
    "max_len": finetune.MAX_LEN,
    "head_max_len": finetune.HEAD_LEN,
    "export_dtype": "float32",
    "max_merge_margin_error": 0.001,
    "max_export_margin_error": 0.03,
    "primary_max_quality_loss_pp": 1,
    "secondary_max_quality_loss_pp": [0, 2, 5],
    "runtime_profiles": {device: finetune.runtime_settings(device) for device in ("mps", "cuda")},
    "eager_flags": {"TORCH_COMPILE_DISABLE": "1", "TORCHDYNAMO_DISABLE": "1"},
    "versions": {"laya": "0.3.29", "torch": "2.9.1", "transformers": "4.57.6", "peft": "0.18.1"},
    "test_boundary": "Freeze before solver test OUTCOME ACCESS; collection may already exist. Solver test answers never enter native training or scoring.",
    "hosting_cost": "Unpriced native serving; report gross solver savings and break-even router cost only",
}
PLAN_FORMAT = "sie-solver-benefit-native-plan-v1"
FROZEN_FORMAT = "sie-solver-benefit-native-frozen-v1"
ACCESS_FORMAT = "sie-solver-benefit-native-access-v1"


def code_hashes():
    return benefit.code_hashes() | {
        name: file_hash(Path(__file__).with_name(name)) for name in ("benefit_native.py", "finetune.py")
    }


def _unopened(head_run):
    if (Path(head_run) / "test-access.json").exists():
        raise ValueError("Shared solver test access is already authorized; native preparation/training must precede it")


def prepare(head_run):
    head_run = Path(head_run)
    _unopened(head_run)
    head = benefit._checked_plan(benefit.read(head_run / "plan.json"))
    fold = np.asarray(head["training_folds"])
    rows = head["populations"]["train"]
    return benefit.seal(
        {
            "format": PLAN_FORMAT,
            "created_at": benefit.now(),
            "protocol": PROTOCOL,
            "questions": QUESTIONS,
            "head_plan": head,
            "head_plan_sha256": file_hash(head_run / "plan.json"),
            "selection_fit_ids": [
                row["id"]
                for row, assignment in zip(rows, fold, strict=True)
                if assignment != PROTOCOL["selection_fold"]
            ],
            "selection_validation_ids": [
                row["id"]
                for row, assignment in zip(rows, fold, strict=True)
                if assignment == PROTOCOL["selection_fold"]
            ],
            "training_pool_rows": len(head["populations"]["train_pool"]),
            "training_rows": len(rows),
            "source_code_sha256": code_hashes(),
        }
    )


def checked_plan(run, head_run):
    plan = benefit.checked(benefit.read(Path(run) / "plan.json"), PLAN_FORMAT)
    head = benefit._checked_plan(plan["head_plan"])
    rows = head["populations"]["train"]
    if (
        plan["protocol"] != PROTOCOL
        or plan["questions"] != QUESTIONS
        or plan["source_code_sha256"] != code_hashes()
        or plan["head_plan_sha256"] != file_hash(Path(head_run) / "plan.json")
        or benefit.read(Path(head_run) / "plan.json") != head
        or plan["training_pool_rows"] != len(head["populations"]["train_pool"])
        or plan["training_rows"] != len(rows)
        or any(
            plan[key]
            != [row["id"] for row, fold in zip(rows, head["training_folds"], strict=True) if (fold == 0) == held]
            for key, held in (("selection_fit_ids", False), ("selection_validation_ids", True))
        )
    ):
        raise ValueError("Native protocol, paper split, head plan or source code changed")
    return plan


def soft_rows(inputs, targets=None):
    targets = np.zeros(len(inputs)) if targets is None else np.asarray(targets, dtype=float)
    if targets.shape != (len(inputs),) or not np.isfinite(targets).all() or not np.isin(targets, [-1, 0, 1]).all():
        raise ValueError("Need aligned signed paired outcomes; ties must remain zero")
    return [
        {
            "state": row["request"],
            "questions": QUESTIONS,
            "gold": {"route": {"probabilities": {"cheap": float((1 - target) / 2), "strong": float((1 + target) / 2)}}},
        }
        for row, target in zip(inputs, targets, strict=True)
    ]


def benefit_scores(margins):
    margins = np.asarray(margins, dtype=float)
    if margins.ndim != 1 or not len(margins) or not np.isfinite(margins).all():
        raise ValueError("Need finite one-dimensional native margins")
    # Numerically stable identity: 2 * sigmoid(margin) - 1.
    return np.tanh(margins / 2)


def records_to_scores(records, expected_targets):
    logits = np.asarray([record[1] for record in records], dtype=float)
    targets = np.asarray([record[2] for record in records], dtype=float)
    if (
        logits.shape != (len(expected_targets), 2)
        or targets.shape != logits.shape
        or any(record[3] != 2 for record in records)
        or not np.isfinite(logits).all()
        or not np.isfinite(targets).all()
        or not np.allclose(targets.sum(axis=1), 1, rtol=0, atol=1e-12)
        or np.any(targets < 0)
        or not np.allclose(2 * targets[:, 1] - 1, expected_targets, rtol=0, atol=1e-12)
    ):
        raise ValueError("Native records changed the cheap/strong order, soft targets or row count")
    margins = logits[:, 1] - logits[:, 0]
    return margins, benefit_scores(margins)


def selected_epoch(history, targets):
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 1 or not len(targets) or not np.isin(targets, [-1, 0, 1]).all():
        raise ValueError("Selection requires signed paired targets, including zero ties")
    if [row["epoch"] for row in history] != list(range(PROTOCOL["selection_epochs"] + 1)):
        raise ValueError("Selection must retain epoch0 and every declared training epoch")
    scores = [benefit_scores(row["margins"]) for row in history]
    if any(values.shape != np.asarray(targets).shape for values in scores):
        raise ValueError("Validation scores and signed targets are not aligned")
    losses = [float(np.mean((values - targets) ** 2)) for values in scores]
    if any(
        not np.isclose(row["benefit_mse"], loss, rtol=0, atol=1e-12) for row, loss in zip(history, losses, strict=True)
    ):
        raise ValueError("Recorded signed-benefit MSE differs from preserved soft targets")
    return int(min(range(len(losses)), key=lambda epoch: (losses[epoch], epoch)))


def score_checkpoint(checkpoint, rows, device):
    """Return signed benefit scores in input order, using a local merged checkpoint.

    ``rows`` contains request dictionaries; any other keys are ignored. Callers
    loading a frozen policy must verify its checkpoint hashes before calling.
    This helper performs no solver requests or threshold fitting.
    """
    checkpoint = Path(checkpoint)
    if device not in ("cpu", "mps", "cuda") or not rows:
        raise ValueError("Need nonempty request rows and device cpu, mps or cuda")
    if not all((checkpoint / name).is_file() for name in ("rl_agent_config.json", "model.safetensors")):
        raise ValueError("Native inference requires a local complete checkpoint")
    os.environ.update(PROTOCOL["eager_flags"])
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("This protocol requires MPS without CPU fallback")
    import laya.train as lt
    import torch

    runtime_device = torch.device(device)
    model, tok, _ = lt.load_checkpoint(str(checkpoint))
    try:
        model.encoder.config.reference_compile = False
        model.to(runtime_device).eval()
        encoded, skipped = lt.items_from_rows(
            tok, soft_rows(rows), finetune.MAX_LEN, finetune.HEAD_LEN, label_smoothing=0.0
        )
        if skipped or len(encoded) != len(rows):
            raise ValueError(f"Native inference must not drop rows: {skipped}")
        records = lt.calibration_records(
            model,
            tok,
            encoded,
            runtime_device,
            finetune.MAX_LEN,
            finetune.HEAD_LEN,
            batch_size=finetune.runtime_settings(device)["evaluation_batch"],
        )
        return records_to_scores(records, np.zeros(len(rows)))[1].tolist()
    finally:
        del model
        finetune.empty_cache(torch, runtime_device)


def preflight(run, head_run, baseline, original_path, training_plan_path, training_inputs_path, training_grades_path):
    run = Path(run)
    _unopened(head_run)
    for name in ("training-start.json", "frozen.json", "test-access.json", "report.json"):
        if (run / name).exists():
            raise FileExistsError(run / name)
    plan = checked_plan(run, head_run)
    head = plan["head_plan"]
    dataset, original, solver_plan, artifacts = benefit._inputs(baseline, original_path)
    if (
        digest(dataset) != head["dataset_sha256"]
        or file_hash(original_path) != head["original_frozen_sha256"]
        or artifacts != head["baseline_artifacts_sha256"]
        or solver_plan["settings_sha256"] != head["solver_settings_sha256"]
    ):
        raise ValueError("Native data or original solver study changed after preparation")
    grades = benefit._training(head, training_plan_path, training_inputs_path, training_grades_path, original)
    training_plan = benefit.read(training_plan_path)
    for key in ("gold_review_sha256", "training_gold_rules_sha256", "training_gold_review_source_sha256"):
        if grades.get(key) != training_plan.get(key) or not isinstance(grades.get(key), str):
            raise ValueError("Training grade review provenance differs from its source plan")
    by_id = {row["id"]: row for row in dataset["splits"]["train"]}
    training = [by_id[row["id"]] for row in head["populations"]["train"]]
    targets = benefit.signed_targets(grades["rows"])
    folds = np.asarray(head["training_folds"])
    return {
        "plan": plan,
        "original": original,
        "training": training,
        "targets": targets,
        "held": folds == PROTOCOL["selection_fold"],
        "calibration": dataset["splits"]["calibration"],
        "test": dataset["splits"]["test"],
        "training_grade_provenance": {key: value for key, value in grades.items() if key != "rows"},
        "training_artifacts_sha256": {
            "plan": file_hash(training_plan_path),
            "inputs": file_hash(training_inputs_path),
            "grades": file_hash(training_grades_path),
        },
    }


def _calibrated(rows, scores, rate):
    return {
        str(target): solver_report.select_cutoff(rows, scores, target, router_credits=None, credit_usd=rate)
        for target in benefit.TARGETS
    }


def train(
    run,
    head_run,
    baseline,
    original_path,
    training_plan_path,
    training_inputs_path,
    training_grades_path,
    *,
    device,
    local_files_only=False,
):
    context = preflight(
        run, head_run, baseline, original_path, training_plan_path, training_inputs_path, training_grades_path
    )
    if device not in PROTOCOL["runtime_profiles"]:
        raise ValueError("Native training requires mps or cuda")
    return _train_gpu(Path(run), Path(head_run), context, device, local_files_only)


def _train_gpu(run, head_run, context, device_name, local_files_only):
    os.environ.update(PROTOCOL["eager_flags"])
    if device_name == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("This protocol requires MPS without CPU fallback")
    import laya.train as lt
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    versions = {name: importlib.metadata.version(name) for name in PROTOCOL["versions"]}
    if {name: version.split("+", 1)[0] for name, version in versions.items()} != PROTOCOL["versions"]:
        raise ValueError("Native training dependencies differ from the pinned protocol")
    device = torch.device(device_name)
    if not (torch.backends.mps.is_available() if device_name == "mps" else torch.cuda.is_available()):
        raise ValueError("Requested GPU is unavailable")
    runtime, recipe = PROTOCOL["runtime_profiles"][device_name], PROTOCOL["recipe"]
    start = {
        "plan_sha256": file_hash(run / "plan.json"),
        "started_at": benefit.now(),
        "versions": versions,
        "device": device_name,
        "runtime": runtime,
        "eager_flags": PROTOCOL["eager_flags"],
        "training_artifacts_sha256": context["training_artifacts_sha256"],
    }
    write(run / "training-start.json", start)
    started = time.monotonic()
    snapshot = snapshot_download(
        finetune.BASE,
        revision=finetune.REVISION,
        allow_patterns=["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"],
        local_files_only=local_files_only,
    )
    targets, held = context["targets"], context["held"]
    histories, refit_losses = [], []

    def items(tok, inputs, y=None):
        result, skipped = lt.items_from_rows(
            tok, soft_rows(inputs, y), finetune.MAX_LEN, finetune.HEAD_LEN, label_smoothing=0.0
        )
        if skipped or len(result) != len(inputs):
            raise ValueError(f"Native preprocessing must not drop rows: {skipped}")
        return result

    def score(model, tok, encoded, y):
        records = lt.calibration_records(
            model, tok, encoded, device, finetune.MAX_LEN, finetune.HEAD_LEN, batch_size=runtime["evaluation_batch"]
        )
        return records_to_scores(records, y)

    def config(epochs):
        return lt.TrainConfig(
            epochs=epochs,
            micro_batch=runtime["micro_batch"],
            grad_accum=runtime["grad_accum"],
            encoder_lr=recipe["encoder_lr"],
            head_lr=recipe["head_lr"],
            weight_decay=recipe["weight_decay"],
            seed=PROTOCOL["seed"],
            max_len=finetune.MAX_LEN,
            head_max_len=finetune.HEAD_LEN,
            loss="soft-ce",
            shuffle_options=("choice",),
            gradient_checkpointing=runtime["gradient_checkpointing"],
            label_smoothing=0.0,
            log_every=32,
        )

    with tempfile.TemporaryDirectory(prefix=".native-initial-", dir=run) as temporary:
        initial = Path(temporary)
        for name in ("rl_agent_config.json", "model.safetensors"):
            shutil.copy2(Path(snapshot) / name, initial / name)
        for name in ("encoder", "tokenizer"):
            shutil.copytree(Path(snapshot) / name, initial / name, symlinks=False)
        initial_sha = file_hash(initial / "model.safetensors")

        def load(with_lora):
            torch.manual_seed(PROTOCOL["seed"])
            np.random.seed(PROTOCOL["seed"])
            random.seed(PROTOCOL["seed"])
            model, tok, cfg = lt.load_checkpoint(str(initial))
            cfg.update(max_len=finetune.MAX_LEN, head_max_len=finetune.HEAD_LEN)
            model.encoder.config.reference_compile = False
            if with_lora:
                model.encoder = get_peft_model(
                    model.encoder,
                    LoraConfig(
                        r=recipe["rank"],
                        lora_alpha=2 * recipe["rank"],
                        lora_dropout=recipe["dropout"],
                        target_modules=finetune.lora_targets(recipe),
                        bias="none",
                    ),
                )
            return model.to(device), tok, cfg

        model, tok, cfg = load(True)
        fit_rows = [row for row, is_held in zip(context["training"], held, strict=True) if not is_held]
        held_rows = [row for row, is_held in zip(context["training"], held, strict=True) if is_held]
        fitting, validation = items(tok, fit_rows, targets[~held]), items(tok, held_rows, targets[held])
        selection_config = config(PROTOCOL["selection_epochs"])
        write(run / "selection-config.json", asdict(selection_config))

        def observe(epoch, loss, model=model, tok=tok, validation=validation):
            margins, values = score(model, tok, validation, targets[held])
            row = {
                "epoch": epoch + 1,
                "train_loss": loss,
                "margins": margins.tolist(),
                "benefit_mse": float(np.mean((values - targets[held]) ** 2)),
            }
            histories.append(row)
            finetune.write(run / "selection-history.json", histories)
            print(json.dumps({key: row[key] for key in ("epoch", "train_loss", "benefit_mse")}), flush=True)
            model.train()

        observe(-1, None)
        lt.train_model(
            model, tok, fitting, selection_config, device, finetune.MAX_LEN, finetune.HEAD_LEN, on_epoch_end=observe
        )
        observe = None
        epoch = selected_epoch(histories, targets[held])
        write(
            run / "epoch-selected.json",
            {
                "epoch": epoch,
                "history_sha256": file_hash(run / "selection-history.json"),
                "validation_ids": context["plan"]["selection_validation_ids"],
                "signed_targets": targets[held].astype(int).tolist(),
                "criterion": PROTOCOL["selection"],
            },
        )
        del model, fitting, validation, tok
        finetune.empty_cache(torch, device)
        models = {}
        for variant in ("base", "lora"):
            model, tok, cfg = load(variant == "lora")
            validation = items(tok, held_rows, targets[held])
            if variant == "lora":
                refit_config = config(epoch)
                write(run / "refit-config.json", asdict(refit_config))
                if epoch:
                    fitting = items(tok, context["training"], targets)
                    refit_losses = lt.train_model(
                        model, tok, fitting, refit_config, device, finetune.MAX_LEN, finetune.HEAD_LEN
                    )
                    del fitting
            before, _ = score(model, tok, validation, targets[held])
            merge_error = 0.0
            if variant == "lora":
                model.encoder = model.encoder.merge_and_unload()
                merged, _ = score(model, tok, validation, targets[held])
                merge_error = float(np.max(np.abs(merged - before)))
                if merge_error > PROTOCOL["max_merge_margin_error"]:
                    raise ValueError("LoRA merge changed native margins beyond the predeclared tolerance")
            destination = run / variant
            lt.save_checkpoint(model, tok, cfg, str(destination))
            save_file(
                {name: tensor.detach().float().cpu().contiguous() for name, tensor in model.state_dict().items()},
                str(destination / "model.safetensors"),
            )
            del model
            finetune.empty_cache(torch, device)
            model, tok, _ = lt.load_checkpoint(str(destination))
            model.encoder.config.reference_compile = False
            model.to(device).eval()
            exported, _ = score(model, tok, validation, targets[held])
            export_error = float(np.max(np.abs(exported - before)))
            if export_error > PROTOCOL["max_export_margin_error"]:
                raise ValueError("FP32 export changed native margins beyond the predeclared tolerance")
            margins, scores = {}, {}
            for split in ("calibration", "test"):
                unlabeled = context[split]
                values, expected_benefit = score(model, tok, items(tok, unlabeled), np.zeros(len(unlabeled)))
                margins[split], scores[split] = values.tolist(), expected_benefit.tolist()
            models[variant] = {
                "checkpoint_files_sha256": {
                    str(path.relative_to(destination)): file_hash(path)
                    for path in sorted(destination.rglob("*"))
                    if path.is_file()
                },
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "margins": margins,
                "scores": scores,
                "max_merge_margin_error": merge_error,
                "max_export_margin_error": export_error,
                "policies": _calibrated(
                    context["original"]["calibration_rows"], scores["calibration"], context["original"]["credit_usd"]
                ),
            }
            write(run / f"{variant}-scores.json", models[variant])
            del model, tok, validation
            finetune.empty_cache(torch, device)
    _unopened(head_run)
    finetune.synchronize(torch, device)
    frozen = benefit.seal(
        {
            "format": FROZEN_FORMAT,
            "frozen_at": benefit.now(),
            "plan_sha256": file_hash(run / "plan.json"),
            "head_plan_sha256": context["plan"]["head_plan_sha256"],
            "training_start_sha256": file_hash(run / "training-start.json"),
            "initial_checkpoint_sha256": initial_sha,
            "training_artifacts_sha256": context["training_artifacts_sha256"],
            "training_grade_provenance": context["training_grade_provenance"],
            "selection_history": histories,
            "validation_signed_targets": targets[held].astype(int).tolist(),
            "selected_epoch": epoch,
            "refit_losses": refit_losses,
            "grading_provenance": solver_policies._grading_provenance(context["original"]),
            "calibration_rows": context["original"]["calibration_rows"],
            "calibration_grades_sha256": context["original"]["calibration_grades_sha256"],
            "credit_usd": context["original"]["credit_usd"],
            "models": models,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    write(run / "frozen.json", frozen)
    print(
        json.dumps(
            {"frozen": str(run / "frozen.json"), "selected_epoch": epoch, "elapsed_seconds": frozen["elapsed_seconds"]}
        ),
        flush=True,
    )
    return frozen


def checked_frozen(run, head_run, *, weights=False):
    run = Path(run)
    plan = checked_plan(run, head_run)
    frozen = benefit.checked(benefit.read(run / "frozen.json"), FROZEN_FORMAT)
    start = benefit.read(run / "training-start.json")
    if (
        frozen["plan_sha256"] != file_hash(run / "plan.json")
        or frozen["head_plan_sha256"] != plan["head_plan_sha256"]
        or frozen["training_start_sha256"] != file_hash(run / "training-start.json")
        or frozen["selected_epoch"]
        != selected_epoch(frozen["selection_history"], np.asarray(frozen["validation_signed_targets"]))
        or len(frozen["refit_losses"]) != frozen["selected_epoch"]
        or not np.isfinite(frozen["refit_losses"]).all()
        or len(frozen["validation_signed_targets"]) != len(plan["selection_validation_ids"])
        or set(frozen["models"]) != {"base", "lora"}
        or start["plan_sha256"] != frozen["plan_sha256"]
        or start["training_artifacts_sha256"] != frozen["training_artifacts_sha256"]
        or start["eager_flags"] != PROTOCOL["eager_flags"]
        or start["runtime"] != PROTOCOL["runtime_profiles"].get(start["device"])
        or {name: version.split("+", 1)[0] for name, version in start["versions"].items()} != PROTOCOL["versions"]
    ):
        raise ValueError("Native freeze does not match its training-only epoch selection and refit")
    solver_policies._grading_provenance(frozen["grading_provenance"])
    if [row["id"] for row in frozen["calibration_rows"]] != [
        row["id"] for row in plan["head_plan"]["populations"]["calibration"]
    ]:
        raise ValueError("Native calibration rows differ from the fixed population")
    for variant, model in frozen["models"].items():
        if (
            not 0 <= model["max_merge_margin_error"] <= PROTOCOL["max_merge_margin_error"]
            or not 0 <= model["max_export_margin_error"] <= PROTOCOL["max_export_margin_error"]
            or "model.safetensors" not in model["checkpoint_files_sha256"]
        ):
            raise ValueError("Native checkpoint/export parity receipt is invalid")
        for split in ("calibration", "test"):
            count = len(plan["head_plan"]["populations"][split])
            margins = solver_policies.finite_scores(model["margins"][split], count)
            scores = solver_policies.finite_scores(model["scores"][split], count)
            if not np.allclose(benefit_scores(margins), scores, rtol=0, atol=1e-12):
                raise ValueError("Native benefit scores differ from canonical cheap/strong margins")
        if model["policies"] != _calibrated(
            frozen["calibration_rows"], model["scores"]["calibration"], frozen["credit_usd"]
        ):
            raise ValueError("Native cutoffs differ from calibration-only cost selection")
        for relative, expected in model["checkpoint_files_sha256"].items():
            path = Path(relative)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not isinstance(expected, str)
                or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
            ):
                raise ValueError("Native checkpoint manifest contains an invalid path or hash")
            if weights and file_hash(run / variant / path) != expected:
                raise ValueError("Native checkpoint bytes differ from their frozen hashes")
    return plan, frozen


def checkpoint_manifest_hash(frozen):
    return digest({name: model["checkpoint_files_sha256"] for name, model in frozen["models"].items()})


def authorize_test(run, head_run, *, verify_weights=False):
    run, head_run = Path(run), Path(head_run)
    if (run / "test-access.json").exists():
        raise FileExistsError(run / "test-access.json")
    plan, frozen = checked_frozen(run, head_run, weights=verify_weights)
    benefit._checked_frozen(head_run)
    return benefit.seal(
        {
            "format": ACCESS_FORMAT,
            "authorized_at": benefit.now(),
            "native_frozen_sha256": file_hash(run / "frozen.json"),
            "head_frozen_sha256": file_hash(head_run / "frozen.json"),
            "head_plan_sha256": plan["head_plan_sha256"],
            "checkpoint_manifest_sha256": checkpoint_manifest_hash(frozen),
            "checkpoint_verification": "local-bytes" if verify_weights else "manifest-only",
            "source_code_sha256": code_hashes(),
        }
    )


def report(run, head_run, test_grades_path):
    run, head_run = Path(run), Path(head_run)
    plan, frozen = checked_frozen(run, head_run)
    access = benefit.checked(benefit.read(run / "test-access.json"), ACCESS_FORMAT)
    shared = benefit.checked(benefit.read(head_run / "test-access.json"), benefit.ACCESS_FORMAT)
    if (
        access["native_frozen_sha256"] != file_hash(run / "frozen.json")
        or access["head_frozen_sha256"] != file_hash(head_run / "frozen.json")
        or access["head_plan_sha256"] != plan["head_plan_sha256"]
        or access["source_code_sha256"] != code_hashes()
        or access["checkpoint_manifest_sha256"] != checkpoint_manifest_hash(frozen)
        or access["checkpoint_verification"] not in ("local-bytes", "manifest-only")
        or shared["frozen_sha256"] != access["head_frozen_sha256"]
        or shared["plan_sha256"] != plan["head_plan_sha256"]
        or shared["source_code_sha256"] != benefit.code_hashes()
    ):
        raise ValueError("Native/shared test access does not bind the frozen models")
    head = plan["head_plan"]
    ids = [row["id"] for row in head["populations"]["test"]]
    grades = solver_policies._grade_rows(test_grades_path, "test", head["original_solver_plan_sha256"], ids)
    if (
        grades["policies_frozen_sha256"] != head["original_frozen_sha256"]
        or solver_policies._grading_provenance(grades) != frozen["grading_provenance"]
    ):
        raise ValueError("Native test grades differ from the original paired solver/grading experiment")
    measured = {}
    for variant, model in frozen["models"].items():
        measured[variant] = {}
        for target in benefit.TARGETS:
            selection = model["policies"][str(target)]
            mode = selection["mode"]
            routes = (
                (np.asarray(model["scores"]["test"]) >= selection["cutoff"]).tolist() if mode == "threshold" else mode
            )
            measured[variant][str(target)] = {
                "mode": mode,
                "cutoff": selection["cutoff"],
                "metrics": solver_report.summarize(
                    grades["rows"],
                    routes,
                    router_credits=None if mode == "threshold" else 0,
                    credit_usd=frozen["credit_usd"],
                ),
            }
    return {
        "format": "sie-solver-benefit-native-report-v1",
        "n": len(ids),
        "ids": ids,
        "training_pool_rows": plan["training_pool_rows"],
        "training_rows": plan["training_rows"],
        "selected_epoch": frozen["selected_epoch"],
        "frozen_sha256": file_hash(run / "frozen.json"),
        "native_access_sha256": file_hash(run / "test-access.json"),
        "shared_access_sha256": file_hash(head_run / "test-access.json"),
        "test_grades_sha256": file_hash(test_grades_path),
        "primary_max_quality_loss_pp": 1,
        "models": measured,
        "interpretation": "Training-only paper fold selected epoch; refit all eligible rows. Calibration loss targets are not test guarantees. Native hosting remains unpriced; gross solver savings exclude router cost. Frozen before solver test outcome access, not collection.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "authorize-test", "report"):
        child = commands.add_parser(command)
        child.add_argument("--run", type=Path, required=True)
        child.add_argument("--head-run", type=Path, required=True)
        if command == "train":
            for name in ("baseline", "original-frozen", "training-plan", "training-inputs", "training-grades"):
                child.add_argument(f"--{name}", type=Path, required=True)
            child.add_argument("--device", choices=("mps", "cuda"), required=True)
            child.add_argument("--local-files-only", action="store_true")
        if command == "report":
            child.add_argument("--test-grades", type=Path, required=True)
        if command == "authorize-test":
            child.add_argument("--verify-weights", action="store_true", help="Also verify local checkpoint bytes")
    args = parser.parse_args()
    destination = (
        args.run
        / {
            "prepare": "plan.json",
            "train": "frozen.json",
            "authorize-test": "test-access.json",
            "report": "report.json",
        }[args.command]
    )
    if destination.exists():
        raise FileExistsError(destination)
    if args.command == "prepare":
        if args.run.exists():
            raise FileExistsError(args.run)
        result = prepare(args.head_run)
    elif args.command == "train":
        train(
            args.run,
            args.head_run,
            args.baseline,
            args.original_frozen,
            args.training_plan,
            args.training_inputs,
            args.training_grades,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        return
    elif args.command == "authorize-test":
        result = authorize_test(args.run, args.head_run, verify_weights=args.verify_weights)
    else:
        result = report(args.run, args.head_run, args.test_grades)
    write(destination, result)
    print(json.dumps({"output": str(destination), "sha256": file_hash(destination)}))


if __name__ == "__main__":
    main()
