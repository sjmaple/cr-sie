"""Train native Laya decisions on a CUDA or Apple Silicon GPU.

Training reads train.jsonl, validation.jsonl and calibration.jsonl. Evaluation is a separate
command, after checkpoint/threshold selection has been written and hashed.
Run with ``uv run --frozen --group training python finetune.py --help``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

BASE = "convaiinnovations/laya"
REVISION = "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8"
MAX_LEN = 512
HEAD_LEN = 256

LORA_RECIPES = {
    "original": {
        "rank": 16,
        "encoder_lr": 2e-4,
        "head_lr": 1e-4,
        "dropout": 0.05,
        "weight_decay": 0.01,
        "smoothing": 0.0,
        "head": "all",
    },
    "slow": {
        "rank": 16,
        "encoder_lr": 5e-5,
        "head_lr": 1e-5,
        "dropout": 0.05,
        "weight_decay": 0.01,
        "smoothing": 0.0,
        "head": "all",
    },
    "regularized": {
        "rank": 8,
        "encoder_lr": 5e-5,
        "head_lr": 1e-5,
        "dropout": 0.1,
        "weight_decay": 0.1,
        "smoothing": 0.1,
        "head": "all",
    },
    "frozen-head": {
        "rank": 8,
        "encoder_lr": 1e-4,
        "head_lr": 1e-5,
        "dropout": 0.1,
        "weight_decay": 0.1,
        "smoothing": 0.1,
        "head": "none",
    },
    "scorer-only": {
        "rank": 8,
        "encoder_lr": 1e-4,
        "head_lr": 1e-5,
        "dropout": 0.1,
        "weight_decay": 0.1,
        "smoothing": 0.1,
        "head": "scorer",
    },
}
LORA_RECIPES["top-layers"] = {**LORA_RECIPES["slow"], "last_layers": 4}


def lora_targets(recipe):
    if recipe.get("last_layers"):
        # The pinned ModernBERT-large encoder has 28 layers.
        indices = "|".join(str(i) for i in range(28 - recipe["last_layers"], 28))
        return rf".*layers\.({indices})\.attn\.(Wqkv|Wo)"
    return r".*attn\.(Wqkv|Wo)"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cutoff(scores, labels):
    """Calibration balanced accuracy; ties prefer the native zero-margin boundary."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    if len(scores) != len(labels) or not np.isfinite(scores).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("Calibration needs finite aligned scores and both classes")
    values = np.unique(scores)
    candidates = np.r_[values[0] - 1, (values[:-1] + values[1:]) / 2, values[-1] + 1, 0.0]

    def quality(t):
        prediction = scores >= t
        return np.mean([np.mean(prediction[labels == label] == label) for label in (0, 1)])

    return float(max(candidates, key=lambda t: (quality(t), -abs(t), -t)))


def items(lt, tok, path, smoothing=0.0):
    rows = lt.read_jsonl(str(path))
    result, skipped = lt.items_from_rows(tok, rows, MAX_LEN, HEAD_LEN, label_smoothing=smoothing)
    if skipped or len(result) != len(rows):
        raise ValueError(f"Training data must not silently drop rows: {skipped}")
    return result


def runtime_settings(device_type):
    if device_type == "mps":
        # Torch 2.9.1 checkpoint recomputation does not restore MPS dropout RNG.
        # Small batches keep memory bounded without that optimization.
        return {"micro_batch": 1, "grad_accum": 16, "gradient_checkpointing": False, "evaluation_batch": 4}
    return {"micro_batch": 8, "grad_accum": 2, "gradient_checkpointing": True, "evaluation_batch": 16}


def empty_cache(torch, device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def scores(lt, model, tok, data, device):
    records = lt.calibration_records(
        model, tok, data, device, MAX_LEN, HEAD_LEN, batch_size=runtime_settings(device.type)["evaluation_batch"]
    )
    logits = np.array([r[1] for r in records], dtype=np.float64)
    labels = np.array([int(np.argmax(r[2])) for r in records])
    if logits.shape != (len(data), 2) or not np.isfinite(logits).all():
        raise ValueError("Expected finite two-option logits")
    return logits[:, 1] - logits[:, 0], labels


def auc(values, labels):
    """Pairwise ROC AUC, with half credit for equal scores."""
    values, labels = np.asarray(values), np.asarray(labels)
    if len(values) != len(labels) or not np.isfinite(values).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("AUC needs finite aligned scores and both classes")
    difference = values[labels == 1, None] - values[labels == 0][None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def better(candidate, best, metric):
    if best is None:
        return True
    if metric == "auc":
        return candidate["auc"] > best["auc"]
    return candidate["cross_entropy"] < best["cross_entropy"]


def metrics(values, labels, threshold):
    predictions = (values >= threshold).astype(int)
    return {
        "n": len(labels),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean([np.mean(predictions[labels == v] == v) for v in (0, 1)])),
        "errors": int(np.sum(predictions != labels)),
        "false_routine": int(np.sum((predictions == 0) & (labels == 1))),
        "false_advanced": int(np.sum((predictions == 1) & (labels == 0))),
        "scores": values.tolist(),
        "labels": labels.tolist(),
        "predictions": predictions.tolist(),
    }


def train(args):
    import laya.train as lt
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    runtime = runtime_settings(device.type)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = lt.TrainConfig(
        epochs=8,
        micro_batch=runtime["micro_batch"],
        grad_accum=runtime["grad_accum"],
        encoder_lr=2.5e-5,
        head_lr=1e-4,
        seed=args.seed,
        max_len=MAX_LEN,
        head_max_len=HEAD_LEN,
        loss="soft-ce",
        shuffle_options=("choice",),
        gradient_checkpointing=runtime["gradient_checkpointing"],
        log_every=32,
    )
    recipe = LORA_RECIPES[args.recipe]
    plan = {
        "base": BASE,
        "revision": REVISION,
        "config": asdict(config),
        "runtime": runtime,
        "variants": args.variants,
        "selection": (
            "Maximum validation ROC AUC, including epoch zero; ties retain earlier epoch. "
            if args.validation_metric == "auc"
            else "Minimum validation cross entropy; ties retain earlier epoch. "
        )
        + "Cutoff fitted once on separate calibration after selection.",
        "validation_metric": args.validation_metric,
        "recipe": args.recipe,
        "export_dtype": args.export_dtype,
        "lora": {
            **recipe,
            "alpha": 2 * recipe["rank"],
            "targets": lora_targets(recipe),
            "frozen_head_dropout": "Retained as training-time regularization",
            "export": "Merge encoder adapters into the unchanged native architecture",
        },
        "data_sha256": {
            n: sha(args.data / n) for n in ("train.jsonl", "validation.jsonl", "calibration.jsonl", "questions.json")
        },
        "test_access": "None. Separate evaluate command only after frozen.json exists.",
        "versions": {n: importlib.metadata.version(n) for n in ("laya", "torch", "transformers", "peft")},
        "device": torch.cuda.get_device_name() if device.type == "cuda" else str(device),
        "reporting": "All declared seeds and model families; no test-based family or seed selection",
    }
    write(output / "plan.json", plan)
    shutil.copy2(args.data / "questions.json", output / "questions.json")
    snapshot = snapshot_download(
        BASE,
        revision=REVISION,
        allow_patterns=[
            "rl_agent_config.json",
            "model.safetensors",
            "encoder/*",
            "tokenizer/*",
        ],
    )
    base = output / "base"
    # A shared Hub snapshot may already contain unrelated sub-models. Copy only
    # the native root checkpoint, even when those other files are cached.
    base.mkdir()
    for name in ("rl_agent_config.json", "model.safetensors"):
        shutil.copy2(Path(snapshot) / name, base / name)
    for name in ("encoder", "tokenizer"):
        shutil.copytree(Path(snapshot) / name, base / name, symlinks=False)
    summary = {}
    for variant in plan["variants"]:
        torch.manual_seed(args.seed)
        model, tok, cfg = lt.load_checkpoint(str(base))
        cfg.update(max_len=MAX_LEN, head_max_len=HEAD_LEN)
        model.to(device)
        training = items(lt, tok, args.data / "train.jsonl", recipe["smoothing"] if variant == "lora" else 0.0)
        validation = items(lt, tok, args.data / "validation.jsonl")
        destination = base if variant == "base" else output / variant
        if variant == "lora":
            model.encoder = get_peft_model(
                model.encoder,
                LoraConfig(
                    r=recipe["rank"],
                    lora_alpha=2 * recipe["rank"],
                    lora_dropout=recipe["dropout"],
                    target_modules=lora_targets(recipe),
                    bias="none",
                ),
            )
            if args.recipe != "original":
                for name, parameter in model.named_parameters():
                    if not name.startswith("encoder."):
                        parameter.requires_grad_(
                            not name.startswith("act_head.")
                            and (recipe["head"] == "all" or (recipe["head"] == "scorer" and name.startswith("scorer.")))
                        )
        count = {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        history = []
        best = None
        started = time.perf_counter()

        def select(epoch, loss, model=model, tok=tok, validation=validation, history=history, variant=variant):
            nonlocal best
            values, labels = scores(lt, model, tok, validation, device)
            threshold = 0.0
            result = metrics(values, labels, threshold)
            result.update(
                epoch=epoch + 1,
                train_loss=loss,
                threshold=threshold,
                cross_entropy=float(np.mean(np.logaddexp(0, values) - labels * values)),
                auc=auc(values, labels),
            )
            history.append(result)
            if better(result, best, args.validation_metric):
                best = dict(result)
                if variant != "base":
                    # Keep the unmerged PEFT weights private to this training run.
                    torch.save(model.state_dict(), output / f"{variant}-best.pt")
            write(output / f"{variant}-history.json", history)
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "epoch": epoch + 1,
                        "validation_accuracy": result["accuracy"],
                        "validation_ce": result["cross_entropy"],
                        "validation_auc": result["auc"],
                        "loss": loss,
                    }
                ),
                flush=True,
            )
            model.train()

        if variant == "base":
            select(-1, None)
            calibration = items(lt, tok, args.data / "calibration.jsonl")
            cal_values, cal_labels = scores(lt, model, tok, calibration, device)
            best["threshold"] = cutoff(cal_values, cal_labels)
        else:
            config.encoder_lr = recipe["encoder_lr"] if variant == "lora" else 2.5e-5
            config.head_lr = recipe["head_lr"] if variant == "lora" else 1e-4
            config.weight_decay = recipe["weight_decay"] if variant == "lora" else 0.01
            config.label_smoothing = recipe["smoothing"] if variant == "lora" else 0.0
            write(output / f"{variant}-train-config.json", asdict(config))
            if args.validation_metric == "auc":
                select(-1, None)
            lt.train_model(model, tok, training, config, device, MAX_LEN, HEAD_LEN, on_epoch_end=select)
            select = None
            model.load_state_dict(torch.load(output / f"{variant}-best.pt", weights_only=True, map_location=device))
            model.eval()
            before, _ = scores(lt, model, tok, validation, device)
            if variant == "lora":
                model.encoder = model.encoder.merge_and_unload()
                after, _ = scores(lt, model, tok, validation, device)
                merge_error = float(np.max(np.abs(before - after)))
                if merge_error > 1e-3:
                    raise ValueError(f"LoRA merge changed margins: {merge_error}")
                write(output / "lora-merge-parity.json", {"max_margin_error": merge_error})
            lt.save_checkpoint(model, tok, cfg, str(destination))
            if args.export_dtype == "float32":
                save_file(
                    {name: tensor.detach().float().cpu().contiguous() for name, tensor in model.state_dict().items()},
                    str(destination / "model.safetensors"),
                )
            del model
            empty_cache(torch, device)
            model, tok, _ = lt.load_checkpoint(str(destination))
            model.to(device).eval()
            exported, _ = scores(lt, model, tok, validation, device)
            error = float(np.max(np.abs(exported - before)))
            if error > 0.03:
                raise ValueError(f"Native export margin drift exceeds tolerance: {error}")
            # Native serialization uses half-precision weights. Fit the cutoff on
            # the checkpoint users actually load, after merging and serialization.
            calibration = items(lt, tok, args.data / "calibration.jsonl")
            cal_values, cal_labels = scores(lt, model, tok, calibration, device)
            best["threshold"] = cutoff(cal_values, cal_labels)
            changes = int(np.sum((exported >= best["threshold"]) != (before >= best["threshold"])))
            write(
                output / f"{variant}-export-parity.json",
                {
                    "max_margin_error": error,
                    "validation_decision_changes_at_final_cutoff": changes,
                    "calibration": "Cutoff fitted on serialized checkpoint",
                },
            )
        summary[variant] = {
            "threshold": best["threshold"],
            "threshold_source": "separate calibration split, serialized checkpoint",
            "validation_cross_entropy": best["cross_entropy"],
            "validation_auc": best["auc"],
            "seed": args.seed,
            "epoch": best["epoch"],
            "validation_accuracy_at_zero": best["accuracy"],
            "parameters": count,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint_sha256": sha(destination / "model.safetensors"),
        }
        write(output / f"{variant}-calibration.json", metrics(cal_values, cal_labels, best["threshold"]))
        write(output / "selection.json", summary)
        select = None
        del model
        empty_cache(torch, device)
    write(output / "frozen.json", {"plan_sha256": sha(output / "plan.json"), "selected": summary})
    print("FROZEN", json.dumps(summary), flush=True)


def evaluate(args):
    import laya.train as lt
    import torch

    frozen = json.loads((args.output / "frozen.json").read_text())
    plan = json.loads((args.output / "plan.json").read_text())
    if (
        sha(args.output / "plan.json") != frozen["plan_sha256"]
        or sha(args.questions) != plan["data_sha256"]["questions.json"]
    ):
        raise ValueError("Plan or question changed after freeze")
    rows = json.loads(args.evaluation.read_text())
    questions = json.loads(args.questions.read_text())
    data = [
        {
            "state": row["request"],
            "questions": questions,
            "expected": {"difficulty": ("routine", "advanced")[row["label"]]},
        }
        for row in rows
    ]
    result = {
        "frozen_sha256": sha(args.output / "frozen.json"),
        "data_sha256": sha(args.evaluation),
        "ids": [r["id"] for r in rows],
        "models": {},
    }
    device = torch.device(args.device)
    for name, selected in frozen["selected"].items():
        path = args.output / name
        if sha(path / "model.safetensors") != selected["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after freeze")
        model, tok, _ = lt.load_checkpoint(str(path))
        model.to(device).eval()
        encoded, skipped = lt.items_from_rows(tok, data, MAX_LEN, HEAD_LEN)
        if skipped or len(encoded) != len(data):
            raise ValueError(f"Evaluation silently dropped rows: {skipped}")
        values, labels = scores(lt, model, tok, encoded, device)
        measured = metrics(values, labels, selected["threshold"])
        # Same warm model, text, device and batch size across all three variants.
        latencies = []
        for index in range(min(35, len(encoded))):
            synchronize(torch, device)
            start = time.perf_counter()
            scores(lt, model, tok, encoded[index : index + 1], device)
            synchronize(torch, device)
            if index >= 5:
                latencies.append(time.perf_counter() - start)
        measured["batch1_seconds"] = {
            "median": float(np.median(latencies)),
            "p95": float(np.quantile(latencies, 0.95)),
            "samples": len(latencies),
        }
        result["models"][name] = measured
        write(args.output / f"evaluation-{args.evaluation.stem}.json", result)
        print(name, measured["accuracy"], measured["errors"], measured["batch1_seconds"], flush=True)
        del model
        empty_cache(torch, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--seed", type=int, choices=(89, 137, 211), default=89)
    training.add_argument("--recipe", choices=tuple(LORA_RECIPES), default="original")
    training.add_argument("--export-dtype", choices=("float16", "float32"), default="float16")
    training.add_argument("--validation-metric", choices=("cross-entropy", "auc"), default="cross-entropy")
    training.add_argument("--variants", nargs="+", choices=("base", "full", "lora"), default=["base", "full", "lora"])
    testing = sub.add_parser("evaluate")
    testing.add_argument("--evaluation", type=Path, required=True)
    testing.add_argument("--questions", type=Path, required=True)
    for child in (training, testing):
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--device", default="cuda")
    args = parser.parse_args()
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
