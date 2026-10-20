"""Train one Laya checkpoint to predict both solvers' answer correctness.

The two questions share encoder and decision-head weights. Each source request
produces two items. CUDA training and calibration scoring are separate commands;
preparation, receipt validation and recorded-result replay need only NumPy.
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
import benefit_outcomes as outcomes
import finetune
import solver_policies
import solver_report
from data import digest, file_hash, write

QUESTION_ORDER = ("cheap_correct", "strong_correct")
OPTIONS = ("incorrect", "correct")
QUESTIONS = {
    "cheap_correct": {
        "type": "choice",
        "instructions": "Predict whether the cheaper Qwen3.5-4B solver will answer this mathematics problem correctly. Judge its likely answer quality; do not solve the problem.",
        "criteria": {
            "incorrect": "The solver will give an incorrect answer.",
            "correct": "The solver will give the correct answer.",
        },
    },
    "strong_correct": {
        "type": "choice",
        "instructions": "Predict whether the stronger Qwen3.8-27B solver will answer this mathematics problem correctly. Judge its likely answer quality; do not solve the problem.",
        "criteria": {
            "incorrect": "The solver will give an incorrect answer.",
            "correct": "The solver will give the correct answer.",
        },
    },
}
PROTOCOL = {
    "base": finetune.BASE,
    "revision": finetune.REVISION,
    "recipe": finetune.LORA_RECIPES["original"],
    "seed": 89,
    "question_order": list(QUESTION_ORDER),
    "option_order": list(OPTIONS),
    "target": "Two hard Bernoulli outcomes per request, with incorrect/correct probabilities [1-y,y]",
    "architecture": "One shared Laya encoder and native decision head; rank16 attention LoRA plus the full native head",
    "selection_fold": 0,
    "selection_epochs": 8,
    "selection": "Minimum mean per-request sum of cheap and strong binary NLL on the training-only paper fold0; include epoch0, ties earlier",
    "refit": "Fresh base and seed89; all eligible requests for the selected epoch count; cosine schedule spans that refit duration; epoch0 skips training",
    "batch_unit": "One question-item, two items per source request; effective batch16 items with the unchanged CUDA profile",
    "runtime": finetune.runtime_settings("cuda"),
    "training_device": "cuda",
    "max_len": finetune.MAX_LEN,
    "head_max_len": finetune.HEAD_LEN,
    "export_dtype": "float32",
    "max_merge_margin_error": 0.001,
    "max_export_margin_error": 0.03,
    "eager_flags": {"TORCH_COMPILE_DISABLE": "1", "TORCHDYNAMO_DISABLE": "1"},
    "versions": {"laya": "0.3.29", "torch": "2.9.1", "transformers": "4.57.6", "peft": "0.18.1"},
    "cost_model": outcomes.PROTOCOL["cost_model"],
    "score": outcomes.PROTOCOL["score"],
    "primary_max_quality_loss_pp": 1,
    "secondary_max_quality_loss_pp": [0, 2, 5],
    "test_boundary": outcomes.PROTOCOL["test_boundary"],
    "hosting_cost": "Native hosting unpriced; compare gross solver savings and break-even router cost, never assume native inference is free",
}
PLAN_FORMAT = "sie-solver-outcome-native-plan-v2"
TRAINED_FORMAT = "sie-solver-outcome-native-trained-v1"
FROZEN_FORMAT = "sie-solver-outcome-native-frozen-v1"
ACCESS_FORMAT = "sie-solver-outcome-native-access-v1"
# This revision differs only in replay validation: NumPy CPU kernels can round
# float64 exp/logaddexp by a few ulps. Historical model/score/threshold bytes and
# every other source hash stay authoritative. Never use this exception to train
# or freeze under a historical plan with different executing source bytes.
REPLAY_COMPATIBLE_NATIVE_SOURCES = frozenset({"611bc5c09e5d854c5a1528b683ac7eeca583c00a1e41812ec1e8f902e3e4f68a"})
REPLAY_RTOL = 1e-12
REPLAY_ATOL = 1e-15


def code_hashes():
    return outcomes.code_hashes() | {
        name: file_hash(Path(__file__).with_name(name)) for name in ("benefit_outcomes_native.py", "finetune.py")
    }


def _source_matches(recorded, *, replay=False):
    current = code_hashes()
    if recorded == current:
        return True
    name = "benefit_outcomes_native.py"
    return (
        replay
        and isinstance(recorded, dict)
        and set(recorded) == set(current)
        and recorded[name] in REPLAY_COMPATIBLE_NATIVE_SOURCES
        and all(recorded[key] == value for key, value in current.items() if key != name)
    )


def _matches_recorded(recorded, recomputed):
    """Validate derived float64 arithmetic, without replacing saved values."""
    try:
        saved, checked = np.asarray(recorded, dtype=float), np.asarray(recomputed, dtype=float)
    except (TypeError, ValueError):
        return False
    return (
        saved.shape == checked.shape
        and saved.size > 0
        and np.isfinite(saved).all()
        and np.isfinite(checked).all()
        and np.allclose(saved, checked, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    )


def replay_validation(plan):
    return {
        "format": "sie-native-float64-replay-validation-v1",
        "training_source_code_sha256": plan["source_code_sha256"],
        "validator_source_code_sha256": code_hashes(),
        "known_historical_native_source": plan["source_code_sha256"] != code_hashes(),
        "rtol": REPLAY_RTOL,
        "atol": REPLAY_ATOL,
        "authoritative_values": "Stored scores, cutoffs, selection losses and exact tie decisions; arithmetic replay never replaces them",
    }


def _unopened(run, head_run):
    outcomes._unopened(run)
    outcomes._unopened(head_run)


def prepare(head_run):
    head_run = Path(head_run)
    outcomes._unopened(head_run)
    head = outcomes.checked_plan(outcomes.read(head_run / "plan.json"))
    rows = head["populations"]["train"]
    return benefit.seal(
        {
            "format": PLAN_FORMAT,
            "created_at": benefit.now(),
            "protocol": PROTOCOL,
            "questions": QUESTIONS,
            "head_plan": head,
            "head_plan_sha256": file_hash(head_run / "plan.json"),
            "source_code_sha256": code_hashes(),
            "selection_fit_ids": [
                row["id"] for row, fold in zip(rows, head["training_folds"], strict=True) if fold != 0
            ],
            "selection_validation_ids": [
                row["id"] for row, fold in zip(rows, head["training_folds"], strict=True) if fold == 0
            ],
        }
    )


def checked_plan(run, head_run, *, replay=False):
    plan = benefit.checked(outcomes.read(Path(run) / "plan.json"), PLAN_FORMAT)
    head = outcomes.checked_plan(plan["head_plan"])
    outcomes._require(
        plan["protocol"] == PROTOCOL
        and plan["questions"] == QUESTIONS
        and _source_matches(plan["source_code_sha256"], replay=replay)
        and plan["head_plan_sha256"] == file_hash(Path(head_run) / "plan.json")
        and outcomes.read(Path(head_run) / "plan.json") == head
        and all(
            plan[key]
            == [
                row["id"]
                for row, fold in zip(head["populations"]["train"], head["training_folds"], strict=True)
                if (fold == 0) == held
            ]
            for key, held in (("selection_fit_ids", False), ("selection_validation_ids", True))
        ),
        "Native protocol, source, outcome plan or whole-paper selection fold changed",
    )
    return plan


def _targets(targets, count=None):
    values = np.asarray(targets, dtype=float)
    outcomes._require(
        values.ndim == 2
        and values.shape[1] == 2
        and len(values) > 0
        and (count is None or len(values) == count)
        and np.isfinite(values).all()
        and np.isin(values, (0, 1)).all(),
        "Need aligned cheap/strong binary targets",
    )
    return values


def question_rows(rows, targets=None):
    y = np.zeros((len(rows), 2)) if targets is None else _targets(targets, len(rows))
    return [
        {
            "state": row["request"],
            "questions": QUESTIONS,
            "gold": {
                question: {"probabilities": {"incorrect": float(1 - labels[index]), "correct": float(labels[index])}}
                for index, question in enumerate(QUESTION_ORDER)
            },
        }
        for row, labels in zip(rows, y, strict=True)
    ]


def encode_rows(lt, tokenizer, rows, targets=None):
    y = np.zeros((len(rows), 2)) if targets is None else _targets(targets, len(rows))
    items, skipped = lt.items_from_rows(
        tokenizer, question_rows(rows, y), finetune.MAX_LEN, finetune.HEAD_LEN, label_smoothing=0.0
    )
    outcomes._require(not skipped and len(items) == 2 * len(rows), f"Two native items per request required: {skipped}")
    for row_index, row in enumerate(rows):
        state = lt.encode_state(tokenizer, row["request"], finetune.MAX_LEN)
        for arm, question in enumerate(QUESTION_ORDER):
            item = items[2 * row_index + arm]
            outcomes._require(
                item["state_ids"] == state
                and item["q"]["ins"] == QUESTIONS[question]["instructions"]
                and list(item["q"]["crit"]) == list(OPTIONS)
                and item["k"] == 2
                and item["target"] == [float(1 - y[row_index, arm]), float(y[row_index, arm])],
                "Native item request, question or incorrect/correct target order changed",
            )
    return items


def records_to_margins(records, expected_targets):
    expected = _targets(expected_targets)
    logits = np.asarray([record[1] for record in records], dtype=float)
    targets = np.asarray([record[2] for record in records], dtype=float)
    flat = expected.reshape(-1)
    outcomes._require(
        logits.shape == (2 * len(expected), 2)
        and targets.shape == logits.shape
        and all(record[3] == 2 for record in records)
        and np.isfinite(logits).all()
        and np.array_equal(targets, np.column_stack([1 - flat, flat])),
        "Native calibration records changed request/question/option target order",
    )
    return (logits[:, 1] - logits[:, 0]).reshape(len(expected), 2)


def probabilities(margins):
    values = np.asarray(margins, dtype=float)
    outcomes._require(
        values.ndim == 2 and values.shape[1] == 2 and len(values) > 0 and np.isfinite(values).all(),
        "Need finite native margins ordered cheap then strong",
    )
    return np.exp(-np.logaddexp(0, -values))


def outcome_nll(margins, targets):
    values, y = np.asarray(margins, dtype=float), _targets(targets)
    probabilities(values)
    outcomes._require(values.shape == y.shape, "Native NLL targets do not align")
    return float((np.logaddexp(0, values) - y * values).sum(axis=1).mean())


def selected_epoch(history, targets):
    y = _targets(targets)
    outcomes._require(
        [row["epoch"] for row in history] == list(range(PROTOCOL["selection_epochs"] + 1)),
        "Retain epoch0 and every declared native selection epoch",
    )
    losses = [row["summed_binary_nll"] for row in history]
    recomputed = [outcome_nll(row["margins"], y) for row in history]
    outcomes._require(
        _matches_recorded(losses, recomputed),
        "Preserved native margins do not reproduce the outcome NLL",
    )
    return min(range(len(losses)), key=lambda epoch: (losses[epoch], epoch))


def _config_values(epochs):
    runtime, recipe = PROTOCOL["runtime"], PROTOCOL["recipe"]
    return {
        "epochs": epochs,
        "micro_batch": runtime["micro_batch"],
        "grad_accum": runtime["grad_accum"],
        "encoder_lr": recipe["encoder_lr"],
        "head_lr": recipe["head_lr"],
        "weight_decay": recipe["weight_decay"],
        "seed": PROTOCOL["seed"],
        "max_len": finetune.MAX_LEN,
        "head_max_len": finetune.HEAD_LEN,
        "loss": "soft-ce",
        "shuffle_options": ("choice",),
        "gradient_checkpointing": runtime["gradient_checkpointing"],
        "label_smoothing": 0.0,
        "log_every": 32,
    }


def preflight(run, head_run, baseline, training_plan_path, inputs_path, rows_path, grades_path, cost_path, price_path):
    run = Path(run)
    _unopened(run, head_run)
    for name in ("training-start.json", "trained.json", "frozen.json"):
        if (run / name).exists():
            raise FileExistsError(run / name)
    plan = checked_plan(run, head_run)
    head = plan["head_plan"]
    outcomes._baseline(head, baseline)
    rows, grades = outcomes._training(head, training_plan_path, inputs_path, rows_path, grades_path)
    costs, statistics, proxy = outcomes.cost_proxy(cost_path, price_path, training_plan_path, grades_path, grades)
    targets = _targets([[int(row[question]) for question in QUESTION_ORDER] for row in grades["rows"]])
    return {
        "plan": plan,
        "rows": rows,
        "targets": targets,
        "held": np.array(head["training_folds"]) == 0,
        "costs": costs,
        "statistics": statistics,
        "cost_proxy_provenance": {key: value for key, value in proxy.items() if key != "rows"},
        "training_grade_provenance": {key: value for key, value in grades.items() if key != "rows"},
        "training_artifacts_sha256": {
            "plan": file_hash(training_plan_path),
            "inputs": file_hash(inputs_path),
            "rows": file_hash(rows_path),
            "grades": file_hash(grades_path),
            "cost_proxy": file_hash(cost_path),
            "price_snapshot": file_hash(price_path),
        },
    }


def train(
    run,
    head_run,
    baseline,
    training_plan_path,
    inputs_path,
    rows_path,
    grades_path,
    cost_path,
    price_path,
    *,
    device="cuda",
    local_files_only=False,
):
    outcomes._require(device == "cuda", "This matched study trains on CUDA only")
    context = preflight(
        run, head_run, baseline, training_plan_path, inputs_path, rows_path, grades_path, cost_path, price_path
    )
    return _train_cuda(Path(run), Path(head_run), context, local_files_only)


def _train_cuda(run, head_run, context, local_files_only):
    os.environ.update(PROTOCOL["eager_flags"])
    import laya.train as lt
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    versions = {name: importlib.metadata.version(name) for name in PROTOCOL["versions"]}
    outcomes._require(
        {name: value.split("+", 1)[0] for name, value in versions.items()} == PROTOCOL["versions"]
        and torch.cuda.is_available(),
        "Pinned native dependencies and an available CUDA GPU are required",
    )
    device, runtime, recipe = torch.device("cuda"), PROTOCOL["runtime"], PROTOCOL["recipe"]
    write(
        run / "training-start.json",
        {
            "plan_sha256": file_hash(run / "plan.json"),
            "started_at": benefit.now(),
            "versions": versions,
            "device": "cuda",
            "runtime": runtime,
            "eager_flags": PROTOCOL["eager_flags"],
            "training_artifacts_sha256": context["training_artifacts_sha256"],
        },
    )
    started = time.monotonic()
    print("Loading pinned Laya checkpoint for two-outcome training", flush=True)
    snapshot = snapshot_download(
        finetune.BASE,
        revision=finetune.REVISION,
        allow_patterns=["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"],
        local_files_only=local_files_only,
    )
    targets, held = context["targets"], context["held"]
    fit_rows = [row for row, flag in zip(context["rows"], held, strict=True) if not flag]
    held_rows = [row for row, flag in zip(context["rows"], held, strict=True) if flag]
    history, refit_losses = [], []

    def margins(model, tokenizer, encoded, y):
        records = lt.calibration_records(
            model,
            tokenizer,
            encoded,
            device,
            finetune.MAX_LEN,
            finetune.HEAD_LEN,
            batch_size=runtime["evaluation_batch"],
        )
        return records_to_margins(records, y)

    with tempfile.TemporaryDirectory(prefix=".outcome-initial-", dir=run) as temporary:
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
            model, tokenizer, config = lt.load_checkpoint(str(initial))
            config.update(max_len=finetune.MAX_LEN, head_max_len=finetune.HEAD_LEN)
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
            return model.to(device), tokenizer, config

        model, tokenizer, config = load(True)
        fitting = encode_rows(lt, tokenizer, fit_rows, targets[~held])
        validation = encode_rows(lt, tokenizer, held_rows, targets[held])
        selection_config = lt.TrainConfig(**_config_values(PROTOCOL["selection_epochs"]))
        write(run / "selection-config.json", asdict(selection_config))

        def observe(epoch, loss, model=model, tokenizer=tokenizer, validation=validation):
            values = margins(model, tokenizer, validation, targets[held])
            entry = {
                "epoch": epoch + 1,
                "train_loss": loss,
                "margins": values.tolist(),
                "summed_binary_nll": outcome_nll(values, targets[held]),
            }
            history.append(entry)
            finetune.write(run / "selection-history.json", history)
            print(json.dumps({key: entry[key] for key in ("epoch", "train_loss", "summed_binary_nll")}), flush=True)
            model.train()

        observe(-1, None)
        lt.train_model(
            model,
            tokenizer,
            fitting,
            selection_config,
            device,
            finetune.MAX_LEN,
            finetune.HEAD_LEN,
            on_epoch_end=observe,
        )
        observe = None
        epoch = selected_epoch(history, targets[held])
        write(
            run / "epoch-selected.json",
            {
                "epoch": epoch,
                "history_sha256": file_hash(run / "selection-history.json"),
                "validation_ids": context["plan"]["selection_validation_ids"],
                "targets": targets[held].astype(int).tolist(),
                "criterion": PROTOCOL["selection"],
            },
        )
        del model, tokenizer, fitting, validation
        finetune.empty_cache(torch, device)
        checkpoints = {}
        for variant in ("base", "lora"):
            print(f"Preparing {variant} checkpoint; selected refit epochs={epoch}", flush=True)
            model, tokenizer, config = load(variant == "lora")
            validation = encode_rows(lt, tokenizer, held_rows, targets[held])
            if variant == "lora":
                refit_config = lt.TrainConfig(**_config_values(epoch))
                write(run / "refit-config.json", asdict(refit_config))
                if epoch:
                    fitting = encode_rows(lt, tokenizer, context["rows"], targets)
                    refit_losses = lt.train_model(
                        model, tokenizer, fitting, refit_config, device, finetune.MAX_LEN, finetune.HEAD_LEN
                    )
                    del fitting
            before = margins(model, tokenizer, validation, targets[held])
            merged = before
            if variant == "lora":
                model.encoder = model.encoder.merge_and_unload()
                merged = margins(model, tokenizer, validation, targets[held])
            merge_error = float(np.max(np.abs(merged - before)))
            outcomes._require(
                merge_error <= PROTOCOL["max_merge_margin_error"], "LoRA merge exceeded the margin tolerance"
            )
            destination = run / variant
            lt.save_checkpoint(model, tokenizer, config, str(destination))
            save_file(
                {name: tensor.detach().float().cpu().contiguous() for name, tensor in model.state_dict().items()},
                str(destination / "model.safetensors"),
            )
            del model
            finetune.empty_cache(torch, device)
            model, tokenizer, _ = lt.load_checkpoint(str(destination))
            model.encoder.config.reference_compile = False
            model.to(device).eval()
            exported = margins(model, tokenizer, validation, targets[held])
            export_error = float(np.max(np.abs(exported - before)))
            outcomes._require(
                export_error <= PROTOCOL["max_export_margin_error"], "FP32 export exceeded the margin tolerance"
            )
            checkpoints[variant] = {
                "files_sha256": {
                    str(path.relative_to(destination)): file_hash(path)
                    for path in sorted(destination.rglob("*"))
                    if path.is_file()
                },
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "reference_margins": before.tolist(),
                "merged_margins": merged.tolist(),
                "exported_margins": exported.tolist(),
                "max_merge_margin_error": merge_error,
                "max_export_margin_error": export_error,
            }
            del model, tokenizer, validation
            finetune.empty_cache(torch, device)
    _unopened(run, head_run)
    finetune.synchronize(torch, device)
    trained = benefit.seal(
        {
            "format": TRAINED_FORMAT,
            "completed_at": benefit.now(),
            "plan_sha256": file_hash(run / "plan.json"),
            "head_plan_sha256": context["plan"]["head_plan_sha256"],
            "training_start_sha256": file_hash(run / "training-start.json"),
            "training_artifacts_sha256": context["training_artifacts_sha256"],
            "training_grade_provenance": context["training_grade_provenance"],
            "initial_checkpoint_sha256": initial_sha,
            "training_targets": targets.astype(int).tolist(),
            "training_cost_proxies": context["costs"],
            "conditional_cost_statistics": context["statistics"],
            "cost_proxy_provenance": context["cost_proxy_provenance"],
            "selection_history": history,
            "selected_epoch": epoch,
            "refit_losses": refit_losses,
            "checkpoints": checkpoints,
            "receipts_sha256": {
                name: file_hash(run / name)
                for name in (
                    "selection-config.json",
                    "selection-history.json",
                    "epoch-selected.json",
                    "refit-config.json",
                )
            },
            "elapsed_seconds": time.monotonic() - started,
            "boundary": "Training and FP32 checkpoint exports complete; no calibration outcomes consumed by native training",
        }
    )
    write(run / "trained.json", trained)
    print(
        json.dumps(
            {
                "trained": str(run / "trained.json"),
                "selected_epoch": epoch,
                "elapsed_seconds": trained["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return trained


def _manifest(run, variant, manifest, verify_weights):
    outcomes._require(
        {"model.safetensors", "rl_agent_config.json"}.issubset(manifest), "Incomplete native checkpoint manifest"
    )
    for relative, expected in manifest.items():
        path = Path(relative)
        outcomes._require(
            not path.is_absolute() and ".." not in path.parts and outcomes._sha(expected),
            "Native checkpoint manifest has an invalid path or hash",
        )
        if verify_weights:
            outcomes._require(file_hash(Path(run) / variant / path) == expected, "Native checkpoint bytes changed")


def checked_trained(run, head_run, *, weights=False, replay=True):
    run = Path(run)
    plan = checked_plan(run, head_run, replay=replay)
    head = plan["head_plan"]
    trained = benefit.checked(outcomes.read(run / "trained.json"), TRAINED_FORMAT)
    start = outcomes.read(run / "training-start.json")
    targets = _targets(trained["training_targets"], len(head["populations"]["train"]))
    held = np.asarray(head["training_folds"]) == 0
    epoch = selected_epoch(trained["selection_history"], targets[held])
    outcomes._require(
        trained["plan_sha256"] == file_hash(run / "plan.json")
        and trained["head_plan_sha256"] == plan["head_plan_sha256"]
        and trained["training_start_sha256"] == file_hash(run / "training-start.json")
        and start["plan_sha256"] == trained["plan_sha256"]
        and start["training_artifacts_sha256"] == trained["training_artifacts_sha256"]
        and start["device"] == "cuda"
        and start["runtime"] == PROTOCOL["runtime"]
        and start["eager_flags"] == PROTOCOL["eager_flags"]
        and {name: value.split("+", 1)[0] for name, value in start["versions"].items()} == PROTOCOL["versions"]
        and trained["selected_epoch"] == epoch
        and len(trained["refit_losses"]) == epoch
        and np.isfinite(trained["refit_losses"]).all()
        and set(trained["checkpoints"]) == {"base", "lora"}
        and trained["conditional_cost_statistics"]
        == outcomes.conditional_costs(targets, trained["training_cost_proxies"])
        and all(
            trained["training_grade_provenance"].get(name) == head["training_grade_provenance"][name]
            for name in outcomes.GRADE_HASHES
        ),
        "Native training lifecycle, targets, costs or runtime changed",
    )
    outcomes._require(
        all(
            trained["training_artifacts_sha256"][name] == head["training_artifacts_sha256"][name]
            for name in ("plan", "inputs", "rows")
        )
        and trained["training_grade_provenance"]["plan_sha256"] == trained["training_artifacts_sha256"]["plan"]
        and trained["cost_proxy_provenance"]["training_grades_sha256"] == trained["training_artifacts_sha256"]["grades"]
        and trained["cost_proxy_provenance"]["conditional_means"] == trained["conditional_cost_statistics"],
        "Native training artifacts do not match the frozen source and cost contracts",
    )
    outcomes._require(
        set(trained["receipts_sha256"])
        == {"selection-config.json", "selection-history.json", "epoch-selected.json", "refit-config.json"},
        "Native receipt set changed",
    )
    for name, expected in trained["receipts_sha256"].items():
        outcomes._require(file_hash(run / name) == expected, "Native training receipt changed")
    outcomes._require(
        outcomes.read(run / "selection-history.json") == trained["selection_history"], "Native history differs"
    )
    selected = outcomes.read(run / "epoch-selected.json")
    outcomes._require(
        selected["epoch"] == epoch
        and selected["targets"] == targets[held].astype(int).tolist()
        and selected["validation_ids"] == plan["selection_validation_ids"]
        and selected["history_sha256"] == file_hash(run / "selection-history.json")
        and selected["criterion"] == PROTOCOL["selection"],
        "Native epoch selection receipt changed",
    )
    for name, epochs in (("selection-config.json", PROTOCOL["selection_epochs"]), ("refit-config.json", epoch)):
        config = outcomes.read(run / name)
        for key, value in _config_values(epochs).items():
            outcomes._require(
                config.get(key) == (list(value) if isinstance(value, tuple) else value),
                "Native training config changed",
            )
    for variant, checkpoint in trained["checkpoints"].items():
        _manifest(run, variant, checkpoint["files_sha256"], weights)
        reference, merged, exported = (
            np.asarray(checkpoint[key], dtype=float)
            for key in ("reference_margins", "merged_margins", "exported_margins")
        )
        outcomes._require(
            all(
                value.shape == (int(held.sum()), 2) and np.isfinite(value).all()
                for value in (reference, merged, exported)
            ),
            "Native export parity rows changed",
        )
        for key, value, tolerance in (
            ("max_merge_margin_error", merged, PROTOCOL["max_merge_margin_error"]),
            ("max_export_margin_error", exported, PROTOCOL["max_export_margin_error"]),
        ):
            error = float(np.max(np.abs(value - reference)))
            outcomes._require(checkpoint[key] == error and error <= tolerance, "Native export parity exceeds tolerance")
    return plan, trained


def checkpoint_margins(checkpoint, rows, device):
    """Infer two untempered margins in input order; caller verifies weight bytes."""
    checkpoint = Path(checkpoint)
    outcomes._require(
        device in ("cpu", "cuda", "mps") and bool(rows), "Need nonempty rows and an explicit inference device"
    )
    outcomes._require(
        all((checkpoint / name).is_file() for name in ("rl_agent_config.json", "model.safetensors")),
        "Native inference requires a complete local checkpoint",
    )
    os.environ.update(PROTOCOL["eager_flags"])
    import laya.train as lt
    import torch

    runtime_device = torch.device(device)
    model, tokenizer, _ = lt.load_checkpoint(str(checkpoint))
    try:
        model.encoder.config.reference_compile = False
        model.to(runtime_device).eval()
        items = encode_rows(lt, tokenizer, rows)
        records = lt.calibration_records(
            model,
            tokenizer,
            items,
            runtime_device,
            finetune.MAX_LEN,
            finetune.HEAD_LEN,
            batch_size=finetune.runtime_settings(device)["evaluation_batch"],
        )
        return records_to_margins(records, np.zeros((len(rows), 2)))
    finally:
        del model
        finetune.empty_cache(torch, runtime_device)


def score_checkpoint(checkpoint, rows, device, conditional_cost_statistics):
    """Return one benefit-per-credit score per request with no threshold fitting."""
    return outcomes.normalized_scores(
        probabilities(checkpoint_margins(checkpoint, rows, device)), conditional_cost_statistics
    )[0].tolist()


def _policies(rows, scores, rate):
    return {
        str(target): solver_report.select_cutoff(rows, scores, target, router_credits=None, credit_usd=rate)
        for target in outcomes.TARGETS
    }


def freeze(run, head_run, baseline, original_path, *, device="cuda"):
    run = Path(run)
    _unopened(run, head_run)
    if (run / "frozen.json").exists() or (run / "scoring-start.json").exists():
        raise FileExistsError(run / "frozen.json")
    plan, trained = checked_trained(run, head_run, weights=True, replay=False)
    head = plan["head_plan"]
    embedding = outcomes.checked_model(head_run)
    outcomes._require(
        embedding["training_grades_sha256"] == trained["training_artifacts_sha256"]["grades"]
        and embedding["cost_proxy_sha256"] == trained["training_artifacts_sha256"]["cost_proxy"]
        and embedding["head"]["targets"] == trained["training_targets"]
        and embedding["head"]["conditional_cost_statistics"] == trained["conditional_cost_statistics"],
        "Native and embedding comparison must use identical training outcomes and cost means",
    )
    outcomes._require(file_hash(original_path) == head["original_frozen_sha256"], "Original calibration study changed")
    dataset, original, _, artifacts = benefit._inputs(baseline, original_path)
    outcomes._require(
        digest(dataset) == head["dataset_sha256"] and artifacts == head["baseline_artifacts_sha256"],
        "Native held inputs differ from the prepared population",
    )
    write(
        run / "scoring-start.json",
        {
            "trained_sha256": file_hash(run / "trained.json"),
            "selected_embedding_sha256": file_hash(Path(head_run) / "selected-model.json"),
            "device": device,
            "started_at": benefit.now(),
            "eager_flags": PROTOCOL["eager_flags"],
        },
    )
    models = {}
    for variant in ("base", "lora"):
        margins, scores, expected = {}, {}, {}
        for split in ("calibration", "test"):
            print(f"Scoring frozen {variant} native outcomes: {split} inputs", flush=True)
            rows = outcomes.held_rows(head, dataset, split)
            values = checkpoint_margins(run / variant, rows, device)
            ratio, costs = outcomes.normalized_scores(probabilities(values), trained["conditional_cost_statistics"])
            margins[split], scores[split], expected[split] = values.tolist(), ratio.tolist(), costs.tolist()
        models[variant] = {
            "margins": margins,
            "scores": scores,
            "expected_costs_proxy": expected,
            "policies": _policies(original["calibration_rows"], scores["calibration"], original["credit_usd"]),
        }
    _unopened(run, head_run)
    return benefit.seal(
        {
            "format": FROZEN_FORMAT,
            "frozen_at": benefit.now(),
            "trained_sha256": file_hash(run / "trained.json"),
            "plan_sha256": file_hash(run / "plan.json"),
            "head_plan_sha256": plan["head_plan_sha256"],
            "selected_embedding_sha256": file_hash(Path(head_run) / "selected-model.json"),
            "scoring_start_sha256": file_hash(run / "scoring-start.json"),
            "models": models,
            "grading_provenance": outcomes.grading_provenance(original),
            "calibration_rows": original["calibration_rows"],
            "calibration_grades_sha256": original["calibration_grades_sha256"],
            "credit_usd": original["credit_usd"],
        }
    )


def checked_frozen(run, head_run, *, weights=False):
    run = Path(run)
    plan, trained = checked_trained(run, head_run, weights=weights)
    frozen = benefit.checked(outcomes.read(run / "frozen.json"), FROZEN_FORMAT)
    embedding = outcomes.checked_model(head_run)
    start = outcomes.read(run / "scoring-start.json")
    outcomes._require(
        frozen["trained_sha256"] == file_hash(run / "trained.json")
        and frozen["plan_sha256"] == file_hash(run / "plan.json")
        and frozen["head_plan_sha256"] == plan["head_plan_sha256"]
        and frozen["selected_embedding_sha256"] == file_hash(Path(head_run) / "selected-model.json")
        and frozen["scoring_start_sha256"] == file_hash(run / "scoring-start.json")
        and start["trained_sha256"] == frozen["trained_sha256"]
        and start["selected_embedding_sha256"] == frozen["selected_embedding_sha256"]
        and start["eager_flags"] == PROTOCOL["eager_flags"]
        and trained["training_targets"] == embedding["head"]["targets"]
        and trained["training_artifacts_sha256"]["grades"] == embedding["training_grades_sha256"]
        and trained["training_artifacts_sha256"]["cost_proxy"] == embedding["cost_proxy_sha256"]
        and trained["conditional_cost_statistics"] == embedding["head"]["conditional_cost_statistics"]
        and frozen["grading_provenance"] == plan["head_plan"]["grading_provenance"]
        and set(frozen["models"]) == {"base", "lora"},
        "Native scoring/freeze lifecycle differs",
    )
    outcomes._require(
        [row["id"] for row in frozen["calibration_rows"]]
        == [row["id"] for row in plan["head_plan"]["populations"]["calibration"]],
        "Native calibration IDs differ",
    )
    for model in frozen["models"].values():
        for split in ("calibration", "test"):
            values = np.asarray(model["margins"][split])
            outcomes._require(
                values.shape == (len(plan["head_plan"]["populations"][split]), 2), "Native margins misaligned"
            )
            scores, costs = outcomes.normalized_scores(probabilities(values), trained["conditional_cost_statistics"])
            outcomes._require(
                _matches_recorded(model["scores"][split], scores)
                and _matches_recorded(model["expected_costs_proxy"][split], costs),
                "Native ratio scores differ from canonical margins and training costs",
            )
        outcomes._require(
            model["policies"]
            == _policies(frozen["calibration_rows"], model["scores"]["calibration"], frozen["credit_usd"]),
            "Native cutoffs differ from calibration-only selection",
        )
    return plan, trained, frozen


def authorize_test(run, head_run, *, verify_weights=False):
    outcomes._unopened(run)
    plan, trained, _frozen = checked_frozen(run, head_run, weights=verify_weights)
    outcomes.checked_frozen(head_run)
    return benefit.seal(
        {
            "format": ACCESS_FORMAT,
            "authorized_at": benefit.now(),
            "native_frozen_sha256": file_hash(Path(run) / "frozen.json"),
            "head_frozen_sha256": file_hash(Path(head_run) / "frozen.json"),
            "head_plan_sha256": plan["head_plan_sha256"],
            "checkpoint_manifest_sha256": digest(trained["checkpoints"]),
            "checkpoint_verification": "local-bytes" if verify_weights else "manifest-only",
            "source_code_sha256": code_hashes(),
            "replay_validation": replay_validation(plan),
        }
    )


def report(run, head_run, grades_path):
    run, head_run = Path(run), Path(head_run)
    plan, trained, frozen = checked_frozen(run, head_run)
    outcomes.checked_frozen(head_run)
    access = benefit.checked(outcomes.read(run / "test-access.json"), ACCESS_FORMAT)
    shared = benefit.checked(outcomes.read(head_run / "test-access.json"), outcomes.ACCESS_FORMAT)
    outcomes._require(
        access["native_frozen_sha256"] == file_hash(run / "frozen.json")
        and access["head_frozen_sha256"] == file_hash(head_run / "frozen.json")
        and access["head_plan_sha256"] == plan["head_plan_sha256"]
        and access["checkpoint_manifest_sha256"] == digest(trained["checkpoints"])
        and access["source_code_sha256"] == code_hashes()
        and access["replay_validation"] == replay_validation(plan)
        and shared["frozen_sha256"] == access["head_frozen_sha256"]
        and shared["plan_sha256"] == plan["head_plan_sha256"]
        and shared["selected_model_sha256"] == frozen["selected_embedding_sha256"]
        and shared["source_code_sha256"] == outcomes.code_hashes(),
        "Native/shared test-access receipts differ",
    )
    head = plan["head_plan"]
    ids = [row["id"] for row in head["populations"]["test"]]
    grades = solver_policies._grade_rows(grades_path, "test", head["original_solver_plan_sha256"], ids)
    outcomes._require(
        grades["policies_frozen_sha256"] == head["original_frozen_sha256"]
        and outcomes.grading_provenance(grades) == frozen["grading_provenance"],
        "Native test answers differ from the original solver/grading study",
    )
    measured = {}
    for variant, model in frozen["models"].items():
        measured[variant] = {}
        for target in outcomes.TARGETS:
            policy = model["policies"][str(target)]
            mode = policy["mode"]
            routes = (np.asarray(model["scores"]["test"]) >= policy["cutoff"]).tolist() if mode == "threshold" else mode
            measured[variant][str(target)] = {
                "mode": mode,
                "cutoff": policy["cutoff"],
                "metrics": solver_report.summarize(
                    grades["rows"],
                    routes,
                    router_credits=None if mode == "threshold" else 0,
                    credit_usd=frozen["credit_usd"],
                ),
            }
    return {
        "format": "sie-solver-outcome-native-report-v1",
        "n": len(ids),
        "ids": ids,
        "training_pool_rows": len(head["training_pool"]),
        "training_rows": len(head["populations"]["train"]),
        "selected_epoch": trained["selected_epoch"],
        "frozen_sha256": file_hash(run / "frozen.json"),
        "native_access_sha256": file_hash(run / "test-access.json"),
        "shared_access_sha256": file_hash(head_run / "test-access.json"),
        "test_grades_sha256": file_hash(grades_path),
        "primary_max_quality_loss_pp": 1,
        "models": measured,
        "interpretation": "One shared two-question checkpoint, epoch selected on training papers only and freshly refit. Cost-normalization means match embedding training. Native hosting unpriced; report gross solver savings. Calibration loss targets are empirical, not test guarantees.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "freeze", "authorize-test", "report"):
        child = commands.add_parser(command)
        child.add_argument("--run", type=Path, required=True)
        child.add_argument("--head-run", type=Path, required=True)
        if command in ("train", "freeze"):
            child.add_argument("--baseline", type=Path, required=True)
            child.add_argument("--device", choices=("cuda",), default="cuda")
        if command == "train":
            for name in (
                "training-plan",
                "training-inputs",
                "training-rows",
                "training-grades",
                "cost-proxy",
                "price-snapshot",
            ):
                child.add_argument(f"--{name}", type=Path, required=True)
            child.add_argument("--local-files-only", action="store_true")
        if command == "freeze":
            child.add_argument("--original-frozen", type=Path, required=True)
        if command == "authorize-test":
            child.add_argument("--verify-weights", action="store_true")
        if command == "report":
            child.add_argument("--test-grades", type=Path, required=True)
    args = parser.parse_args()
    destination = (
        args.run
        / {
            "prepare": "plan.json",
            "train": "trained.json",
            "freeze": "frozen.json",
            "authorize-test": "test-access.json",
            "report": "report.json",
        }[args.command]
    )
    if destination.exists() or (args.command == "prepare" and args.run.exists()):
        raise FileExistsError(destination)
    if args.command == "prepare":
        result = prepare(args.head_run)
    elif args.command == "train":
        train(
            args.run,
            args.head_run,
            args.baseline,
            args.training_plan,
            args.training_inputs,
            args.training_rows,
            args.training_grades,
            args.cost_proxy,
            args.price_snapshot,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        return
    elif args.command == "freeze":
        result = freeze(args.run, args.head_run, args.baseline, args.original_frozen, device=args.device)
    elif args.command == "authorize-test":
        result = authorize_test(args.run, args.head_run, verify_weights=args.verify_weights)
    else:
        result = report(args.run, args.head_run, args.test_grades)
    write(destination, result)
    print(json.dumps({"output": str(destination), "sha256": file_hash(destination)}))


if __name__ == "__main__":
    main()
