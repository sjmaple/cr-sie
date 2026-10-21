"""Task-neutral two-question Laya LoRA, explicitly using a 4096-token budget."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
import paired_training as core

NATIVE = core.PROTOCOL["native"]
MAX_LEN, HEAD_LEN = NATIVE["max_len"], NATIVE["head_max_len"]
INITIAL_WEIGHTS_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
OPTION_ORDER = ("incorrect", "correct")


def checkpoint_files(root):
    root = Path(root)
    return {path.relative_to(root).as_posix(): core.sha(path) for path in sorted(root.rglob("*")) if path.is_file()}


def copy_base_checkpoint(source, destination, *, expected_weights_sha256=INITIAL_WEIGHTS_SHA256):
    """Preserve original weights/tokenizer/encoder; change only the context limits."""
    source, destination = Path(source), Path(destination)
    original = checkpoint_files(source)
    core.require(original.get("model.safetensors") == expected_weights_sha256, "Initial weights changed")
    core.require("rl_agent_config.json" in original, "Missing original Laya configuration")
    shutil.copytree(source, destination, symlinks=False)
    config = core.read(source / "rl_agent_config.json")
    config.update(max_len=MAX_LEN, head_max_len=HEAD_LEN)
    config_path = destination / "rl_agent_config.json"
    config_path.unlink()
    core.write(config_path, config)
    exported = checkpoint_files(destination)
    core.require(
        set(original) == set(exported)
        and all(original[name] == exported[name] for name in original if name != "rl_agent_config.json"),
        "Unchanged base export altered non-context checkpoint bytes",
    )
    return {
        "original_files_sha256": original,
        "files_sha256": exported,
        "weights_byte_identical": True,
        "amended_config_fields": {"max_len": MAX_LEN, "head_max_len": HEAD_LEN},
    }


def load_checkpoint_preserving(lt, checkpoint):
    """Laya normalizes tokenizer metadata on load; isolate that from immutable files."""
    checkpoint = Path(checkpoint).resolve()
    with tempfile.TemporaryDirectory(prefix="router-laya-load-") as temporary:
        local = Path(temporary)
        for path in sorted(checkpoint.rglob("*")):
            if not path.is_file():
                continue
            target = local / path.relative_to(checkpoint)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "model.safetensors":
                target.symlink_to(path.resolve())  # The loader only reads weights; avoid a multi-GB copy.
            else:
                shutil.copy2(path, target)
        return lt.load_checkpoint(str(local))


def targets(values, count):
    result = np.asarray(values, dtype=float)
    core.require(
        result.shape == (count, 2) and count > 0 and np.isin(result, (0, 1)).all(), "Wrong paired binary targets"
    )
    return result


def question_rows(rows, pair, labels=None):
    y = np.zeros((len(rows), 2)) if labels is None else targets(labels, len(rows))
    questions = core.questions(pair)
    return [
        {
            "state": row["request"],
            "questions": questions,
            "gold": {
                f"{arm}_correct": {"probabilities": {"incorrect": float(1 - values[i]), "correct": float(values[i])}}
                for i, arm in enumerate(core.ARMS)
            },
        }
        for row, values in zip(rows, y, strict=True)
    ]


def encode_rows(lt, tokenizer, rows, pair, labels=None):
    y = np.zeros((len(rows), 2)) if labels is None else targets(labels, len(rows))
    questions = core.questions(pair)
    items, skipped = lt.items_from_rows(tokenizer, question_rows(rows, pair, y), MAX_LEN, HEAD_LEN, label_smoothing=0.0)
    core.require(not skipped and len(items) == 2 * len(rows), f"No native item may be skipped: {skipped}")
    for index, row in enumerate(rows):
        uncapped = lt.encode_state(tokenizer, row["request"], 1_000_000)
        core.require(len(uncapped) <= MAX_LEN, "State exceeds fixed context; do not silently truncate")
        for arm, name in enumerate(core.ARMS):
            item = items[2 * index + arm]
            core.require(
                item["state_ids"] == uncapped
                and item["q"]["ins"] == questions[f"{name}_correct"]["instructions"]
                and list(item["q"]["crit"]) == list(OPTION_ORDER)
                and item["k"] == 2
                and item["target"] == [float(1 - y[index, arm]), float(y[index, arm])],
                "Native request/question/option/label order changed",
            )
            head, _, _ = lt.build_head(tokenizer, item["q"], HEAD_LEN)
            uncut_head, _, _ = lt.build_head(tokenizer, item["q"], MAX_LEN)
            core.require(head == uncut_head, "Decision instructions/options exceed head cap")
            _, markers, truncation = lt.build_sequence(
                tokenizer, None, item["q"], MAX_LEN, HEAD_LEN, state_ids=uncapped, return_truncation_stats=True
            )
            core.require(
                len(markers) == 2 and not truncation["truncated"] and truncation["state_tokens_dropped"] == 0,
                "Combined native request would be truncated",
            )
    return items


def margins_from_records(records, y):
    y = targets(y, len(y))
    logits = np.asarray([row[1] for row in records], dtype=float)
    actual = np.asarray([row[2] for row in records], dtype=float)
    flat = y.reshape(-1)
    core.require(
        logits.shape == actual.shape == (len(y) * 2, 2)
        and np.isfinite(logits).all()
        and all(row[3] == 2 for row in records)
        and np.array_equal(actual, np.column_stack([1 - flat, flat])),
        "Scoring order or binary targets changed",
    )
    return (logits[:, 1] - logits[:, 0]).reshape(len(y), 2)


def summed_nll(margins, y):
    values = np.asarray(margins, dtype=float)
    y = targets(y, len(y))
    core.require(values.shape == y.shape and np.isfinite(values).all(), "Invalid logits")
    return float(np.logaddexp(0, (1 - 2 * y) * values).sum(axis=1).mean())


def select_epoch(history):
    core.require([row["epoch"] for row in history] == list(range(9)), "Retain all epoch0..8 candidates")
    core.require(all(np.isfinite(row["summed_binary_nll"]) for row in history), "Invalid held-family loss")
    return min(range(9), key=lambda epoch: (history[epoch]["summed_binary_nll"], epoch))


def config_values(epochs):
    return {
        "epochs": epochs,
        "micro_batch": NATIVE["micro_batch"],
        "grad_accum": NATIVE["grad_accum"],
        "encoder_lr": NATIVE["encoder_lr"],
        "head_lr": NATIVE["head_lr"],
        "weight_decay": NATIVE["weight_decay"],
        "seed": 89,
        "max_len": MAX_LEN,
        "head_max_len": HEAD_LEN,
        "loss": "soft-ce",
        "shuffle_options": ("choice",),
        "gradient_checkpointing": NATIVE["gradient_checkpointing"],
        "label_smoothing": 0.0,
        "log_every": 64,
    }


def memory_probe(run, initial_checkpoint, lengths_path, output):
    """One discarded synthetic update at the longest source-only request."""
    run, initial = Path(run), Path(initial_checkpoint)
    plan, metadata = core.checked_plan(run)
    audit = core.read(lengths_path)
    core.require(
        audit["source_only"] and audit["inputs_sha256"] == plan["source_inputs_sha256"],
        "Memory probe must use the pinned source-only length audit",
    )
    core.require(core.sha(initial / "model.safetensors") == INITIAL_WEIGHTS_SHA256, "Initial weights changed")
    longest = max(audit["rows"], key=lambda row: (row["state_tokens"], row["id"]))
    by_id = {row["id"]: row for row in metadata}
    core.require(set(by_id) == {row["id"] for row in audit["rows"]}, "Length audit source population differs")
    row = by_id[longest["id"]]
    core.require(
        hashlib.sha256(row["request"].encode()).hexdigest() == longest["request_sha256"],
        "Longest source request changed",
    )
    os.environ.update(TORCH_COMPILE_DISABLE="1", TORCHDYNAMO_DISABLE="1")
    import laya.train as lt
    import torch
    from peft import LoraConfig, get_peft_model

    versions = {name: importlib.metadata.version(name).split("+", 1)[0] for name in NATIVE["versions"]}
    core.require(versions == NATIVE["versions"] and torch.cuda.is_available(), "Pinned CUDA runtime required")
    torch.manual_seed(89)
    np.random.seed(89)
    random.seed(89)
    model, tokenizer, _ = load_checkpoint_preserving(lt, initial)
    model.encoder.config.reference_compile = False
    model.encoder = get_peft_model(
        model.encoder,
        LoraConfig(
            r=NATIVE["rank"],
            lora_alpha=NATIVE["alpha"],
            lora_dropout=NATIVE["dropout"],
            target_modules=r".*attn\.(Wqkv|Wo)",
            bias="none",
        ),
    )
    device = torch.device("cuda")
    model.to(device)
    count = max(NATIVE["micro_batch"] * NATIVE["grad_accum"], NATIVE["evaluation_batch"]) // 2
    y = np.array([[i % 2, 1 - i % 2] for i in range(count)])
    items = encode_rows(lt, tokenizer, [row] * count, plan["pair"], y)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    losses = lt.train_model(model, tokenizer, items, lt.TrainConfig(**config_values(1)), device, MAX_LEN, HEAD_LEN)
    records = lt.calibration_records(
        model, tokenizer, items, device, MAX_LEN, HEAD_LEN, batch_size=NATIVE["evaluation_batch"]
    )
    margins = margins_from_records(records, y)
    torch.cuda.synchronize()
    core.require(np.isfinite(losses).all() and np.isfinite(margins).all(), "Synthetic memory step was nonfinite")
    receipt = {
        "format": "routerbench-native-memory-probe-v1",
        "plan_sha256": core.sha(run / "plan.json"),
        "length_audit_sha256": core.sha(lengths_path),
        "request_id": row["id"],
        "request_sha256": longest["request_sha256"],
        "state_tokens": longest["state_tokens"],
        "max_len": MAX_LEN,
        "head_max_len": HEAD_LEN,
        "micro_batch": NATIVE["micro_batch"],
        "grad_accum": NATIVE["grad_accum"],
        "evaluation_batch": NATIVE["evaluation_batch"],
        "gpu": torch.cuda.get_device_name(0),
        "total_gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.monotonic() - started,
        "versions": versions,
        "labels": "Synthetic alternating labels only; all updated parameters discarded",
        "actual_outcomes_accessed": False,
        "passed": True,
    }
    del model, tokenizer, items
    gc.collect()
    torch.cuda.empty_cache()
    core.write(output, receipt)
    return receipt


def train(run, grades_path, initial_checkpoint, *, commit=lambda: None):
    run, initial = Path(run), Path(initial_checkpoint)
    core.require(not (run / "final-access.json").exists(), "Final outcomes already accessed")
    plan, metadata = core.checked_plan(run)
    rows, y, costs = core.grade_arrays(plan, metadata, core.read(grades_path), "train")
    cost_statistics = core.conditional_usd(y, costs)
    core.require(core.sha(initial / "model.safetensors") == INITIAL_WEIGHTS_SHA256, "Pinned initial weights changed")
    held = np.asarray([row["fold"] == 0 for row in rows])
    core.require(held.any() and (~held).any(), "Empty fixed native fold")
    out = run / "native"
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCH_COMPILE_DISABLE="1", TORCHDYNAMO_DISABLE="1")
    import laya.train as lt
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    versions = {name: importlib.metadata.version(name) for name in NATIVE["versions"]}
    core.require(
        {name: value.split("+", 1)[0] for name, value in versions.items()} == NATIVE["versions"]
        and torch.cuda.is_available(),
        "Pinned Laya training dependencies and CUDA required",
    )
    device = torch.device("cuda")
    core.write(
        out / "training-start.json",
        {
            "plan_sha256": core.sha(run / "plan.json"),
            "training_grades_sha256": core.sha(grades_path),
            "versions": versions,
            "gpu": torch.cuda.get_device_name(0),
            "config": NATIVE,
            "initial_files_sha256": {
                path.relative_to(initial).as_posix(): core.sha(path)
                for path in sorted(initial.rglob("*"))
                if path.is_file()
            },
            "training_rows": len(rows),
            "training_items": len(rows) * 2,
            "selection_fit_rows": int((~held).sum()),
            "selection_validation_rows": int(held.sum()),
        },
    )
    commit()
    started, history = time.monotonic(), []

    def release():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def load(lora):
        torch.manual_seed(89)
        np.random.seed(89)
        random.seed(89)
        model, tokenizer, configuration = load_checkpoint_preserving(lt, initial)
        configuration.update(max_len=MAX_LEN, head_max_len=HEAD_LEN)
        model.encoder.config.reference_compile = False
        if lora:
            model.encoder = get_peft_model(
                model.encoder,
                LoraConfig(
                    r=NATIVE["rank"],
                    lora_alpha=NATIVE["alpha"],
                    lora_dropout=NATIVE["dropout"],
                    target_modules=r".*attn\.(Wqkv|Wo)",
                    bias="none",
                ),
            )
        return model.to(device), tokenizer, configuration

    def score(model, tokenizer, items, labels):
        records = lt.calibration_records(
            model, tokenizer, items, device, MAX_LEN, HEAD_LEN, batch_size=NATIVE["evaluation_batch"]
        )
        return margins_from_records(records, labels)

    fit_rows, held_rows = (
        [row for row, flag in zip(rows, held, strict=True) if not flag],
        [row for row, flag in zip(rows, held, strict=True) if flag],
    )
    model, tokenizer, configuration = load(True)
    fitting = encode_rows(lt, tokenizer, fit_rows, plan["pair"], y[~held])
    validation = encode_rows(lt, tokenizer, held_rows, plan["pair"], y[held])
    config = lt.TrainConfig(**config_values(8))
    core.write(out / "selection-config.json", asdict(config))

    def observe(epoch, train_loss, model=model, tokenizer=tokenizer, validation=validation):
        values = score(model, tokenizer, validation, y[held])
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "summed_binary_nll": summed_nll(values, y[held]),
            "margins": values.tolist(),
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        core.write(out / f"epoch-{epoch + 1}.json", record)
        commit()
        print(
            json.dumps({key: record[key] for key in ("epoch", "train_loss", "summed_binary_nll", "elapsed_seconds")}),
            flush=True,
        )
        model.train()

    observe(-1, None)
    lt.train_model(model, tokenizer, fitting, config, device, MAX_LEN, HEAD_LEN, on_epoch_end=observe)
    epoch = select_epoch(history)
    core.write(
        out / "selected-epoch.json",
        {
            "epoch": epoch,
            "validation_ids": [row["id"] for row in held_rows],
            "criterion": NATIVE["selection"],
            "history": history,
        },
    )
    observe = None
    del model, tokenizer, fitting, validation
    release()
    checkpoints, refit_losses = {}, []
    for variant in ("base", "lora"):
        model, tokenizer, configuration = load(variant == "lora")
        validation = encode_rows(lt, tokenizer, held_rows, plan["pair"], y[held])
        if variant == "lora":
            config = lt.TrainConfig(**config_values(epoch))
            core.write(out / "refit-config.json", asdict(config))
            if epoch:
                fitting = encode_rows(lt, tokenizer, rows, plan["pair"], y)
                refit_losses = lt.train_model(
                    model,
                    tokenizer,
                    fitting,
                    config,
                    device,
                    MAX_LEN,
                    HEAD_LEN,
                    on_epoch_end=lambda e, loss: (
                        core.write(out / f"refit-epoch-{e + 1}.json", {"epoch": e + 1, "loss": loss}),
                        commit(),
                    ),
                )
                del fitting
        before = score(model, tokenizer, validation, y[held])
        if variant == "lora":
            model.encoder = model.encoder.merge_and_unload()
        merged = score(model, tokenizer, validation, y[held])
        merge_error = float(np.abs(merged - before).max())
        core.require(merge_error <= 0.001, "LoRA merge numerical check failed")
        destination = out / variant
        base_copy = None
        if variant == "base":
            base_copy = copy_base_checkpoint(initial, destination)
        else:
            lt.save_checkpoint(model, tokenizer, configuration, str(destination))
            save_file(
                {name: tensor.detach().float().cpu().contiguous() for name, tensor in model.state_dict().items()},
                str(destination / "model.safetensors"),
            )
        del model
        release()
        model, _, saved = load_checkpoint_preserving(lt, destination)
        core.require(
            saved["max_len"] == MAX_LEN and saved["head_max_len"] == HEAD_LEN, "Export reverted to512/context defaults"
        )
        model.encoder.config.reference_compile = False
        model.to(device).eval()
        exported = score(model, tokenizer, validation, y[held])
        export_error = float(np.abs(exported - before).max())
        core.require(export_error <= 0.03, "Checkpoint reload numerical check failed")
        checkpoints[variant] = {
            "files_sha256": checkpoint_files(destination),
            "base_copy": base_copy,
            "max_merge_margin_error": merge_error,
            "max_export_margin_error": export_error,
            "reference_margins": before.tolist(),
            "exported_margins": exported.tolist(),
        }
        core.write(out / f"{variant}-export.json", checkpoints[variant])
        commit()
        del model, tokenizer, validation
        release()
    trained = {
        "format": "routerbench-paired-native-v1",
        "plan_sha256": core.sha(run / "plan.json"),
        "training_grades_sha256": core.sha(grades_path),
        "grader_sha256": core.read(grades_path)["grader_sha256"],
        "training_ids": [row["id"] for row in rows],
        "selected_epoch": epoch,
        "refit_losses": refit_losses,
        "conditional_cost_usd": cost_statistics,
        "checkpoints": checkpoints,
        "training_start_sha256": core.sha(out / "training-start.json"),
        "elapsed_seconds": time.monotonic() - started,
        "hosting": "Unpriced native inference; report gross answer-model savings until hosting is measured.",
        "boundary": "Training-only epoch selection and refit; no calibration/final outcomes read.",
    }
    core.write(out / "trained.json", trained)
    commit()
    return trained


def checkpoint_probabilities(checkpoint, rows, pair, *, device="cuda"):
    """Input-only score helper; canonical cheap/strong column order, no token truncation."""
    os.environ.update(TORCH_COMPILE_DISABLE="1", TORCHDYNAMO_DISABLE="1")
    import laya.train as lt
    import torch

    model, tokenizer, config = load_checkpoint_preserving(lt, checkpoint)
    core.require(
        config["max_len"] == MAX_LEN and config["head_max_len"] == HEAD_LEN, "Wrong native context configuration"
    )
    model.encoder.config.reference_compile = False
    model.to(torch.device(device)).eval()
    items = encode_rows(lt, tokenizer, rows, pair)
    records = lt.calibration_records(
        model,
        tokenizer,
        items,
        torch.device(device),
        MAX_LEN,
        HEAD_LEN,
        batch_size=NATIVE["evaluation_batch"] if device == "cuda" else 1,
    )
    margins = margins_from_records(records, np.zeros((len(rows), 2)))
    probabilities = np.exp(-np.logaddexp(0, -margins))
    del model, tokenizer, items, records
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return probabilities


def score_native(run, output, *, device="cuda", commit=lambda: None):
    run = Path(run)
    plan, rows = core.checked_plan(run)
    root = run / "native"
    trained = core.read(root / "trained.json")
    core.require(
        trained["plan_sha256"] == core.sha(run / "plan.json")
        and trained["training_ids"] == plan["populations"]["train"],
        "Different trained population/plan",
    )
    selected = [row for row in rows if row["role"] != "train"]
    result = {
        "format": "routerbench-input-only-native-v1",
        "plan_sha256": core.sha(run / "plan.json"),
        "trained_sha256": core.sha(root / "trained.json"),
        "outcomes_accessed": False,
        "variants": {},
    }
    for variant in ("base", "lora"):
        checkpoint = root / variant
        actual = {
            path.relative_to(checkpoint).as_posix(): core.sha(path)
            for path in sorted(checkpoint.rglob("*"))
            if path.is_file()
        }
        core.require(actual == trained["checkpoints"][variant]["files_sha256"], "Checkpoint bytes changed")
        probabilities = checkpoint_probabilities(checkpoint, selected, plan["pair"], device=device)
        expected = core.expected_usd(probabilities, trained["conditional_cost_usd"])
        scored = {
            "method": variant,
            "conditional_cost_usd": trained["conditional_cost_usd"],
            "router_hosting_usd": None,
            "roles": {},
        }
        for role in ("calibration", "final", "development"):
            indices = [i for i, row in enumerate(selected) if row["role"] == role]
            scored["roles"][role] = {
                "ids": [selected[i]["id"] for i in indices],
                "probabilities": probabilities[indices].tolist(),
                "expected_usd": expected[indices].tolist(),
            }
        result["variants"][variant] = scored
        core.write(root / f"{variant}-input-scores.json", scored)
        commit()
    core.write(output, result)
    commit()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = train(args.run, args.grades, args.initial_checkpoint)
    print(json.dumps({"selected_epoch": result["selected_epoch"], "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
