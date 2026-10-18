"""Optional GPU lab: full fine-tuning and encoder LoRA for native Laya decisions.

Training reads only train.jsonl and validation.jsonl. Evaluation is a separate
command, after checkpoint/threshold selection has been written and hashed.
Run with ``uv run --frozen --group posttrain python posttrain.py --help``.
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
SEED = 89


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
    """Validation accuracy; ties prefer the native zero-logit-margin boundary."""
    values = np.unique(scores)
    candidates = np.r_[values[0] - 1, (values[:-1] + values[1:]) / 2, values[-1] + 1, 0.0]
    return float(max(candidates, key=lambda t: (np.mean((scores >= t) == labels), -abs(t), -t)))


def items(lt, tok, path):
    rows = lt.read_jsonl(str(path))
    result, skipped = lt.items_from_rows(tok, rows, MAX_LEN, HEAD_LEN)
    if skipped or len(result) != len(rows):
        raise ValueError(f"Training data must not silently drop rows: {skipped}")
    return result


def scores(lt, model, tok, data, device):
    records = lt.calibration_records(model, tok, data, device, MAX_LEN, HEAD_LEN, batch_size=16)
    logits = np.array([r[1] for r in records], dtype=np.float64)
    labels = np.array([int(np.argmax(r[2])) for r in records])
    if logits.shape != (len(data), 2) or not np.isfinite(logits).all():
        raise ValueError("Expected finite two-option logits")
    return logits[:, 1] - logits[:, 0], labels


def metrics(values, labels, threshold):
    predictions = (values >= threshold).astype(int)
    return {
        "n": len(labels),
        "accuracy": float(np.mean(predictions == labels)),
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

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    config = lt.TrainConfig(
        epochs=8,
        micro_batch=8,
        grad_accum=2,
        encoder_lr=2.5e-5,
        head_lr=1e-4,
        seed=SEED,
        max_len=MAX_LEN,
        head_max_len=HEAD_LEN,
        loss="soft-ce",
        shuffle_options=("choice",),
        gradient_checkpointing=True,
        log_every=32,
    )
    plan = {
        "base": BASE,
        "revision": REVISION,
        "config": asdict(config),
        "variants": ["base", "full", "lora"],
        "selection": "Per family, maximum validation accuracy after fitted cutoff; ties retain earlier epoch",
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "targets": r".*attn\.(Wqkv|Wo)",
            "encoder_lr": 2e-4,
            "head": "All native decision-head parameters train; merge encoder for native export",
        },
        "data_sha256": {n: sha(args.data / n) for n in ("train.jsonl", "validation.jsonl")},
        "test_access": "None. Separate evaluate command only after frozen.json exists.",
        "versions": {n: importlib.metadata.version(n) for n in ("laya", "torch", "transformers", "peft")},
        "device": torch.cuda.get_device_name() if device.type == "cuda" else str(device),
        "confirmation_success": "At least 25% fewer errors than calibrated native Laya; report comparison with frozen Qwen ridge separately",
    }
    write(output / "plan.json", plan)
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
        torch.manual_seed(SEED)
        model, tok, cfg = lt.load_checkpoint(str(base))
        cfg.update(max_len=MAX_LEN, head_max_len=HEAD_LEN)
        model.to(device)
        training = items(lt, tok, args.data / "train.jsonl")
        validation = items(lt, tok, args.data / "validation.jsonl")
        destination = base if variant == "base" else output / variant
        if variant == "lora":
            model.encoder = get_peft_model(
                model.encoder,
                LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=r".*attn\.(Wqkv|Wo)",
                    bias="none",
                ),
            )
        count = {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        history = []
        best = {"accuracy": -1.0}
        started = time.perf_counter()

        def select(epoch, loss, model=model, tok=tok, validation=validation, history=history, variant=variant):
            nonlocal best
            values, labels = scores(lt, model, tok, validation, device)
            threshold = cutoff(values, labels)
            result = metrics(values, labels, threshold)
            result.update(epoch=epoch + 1, train_loss=loss, threshold=threshold)
            history.append(result)
            if result["accuracy"] > best["accuracy"]:
                best = result
                if variant != "base":
                    # Keep the unmerged PEFT weights private to this training run.
                    torch.save(model.state_dict(), output / f"{variant}-best.pt")
            write(output / f"{variant}-history.json", history)
            print(
                json.dumps(
                    {"variant": variant, "epoch": epoch + 1, "validation_accuracy": result["accuracy"], "loss": loss}
                ),
                flush=True,
            )
            model.train()

        if variant == "base":
            select(-1, None)
        else:
            config.encoder_lr = 2e-4 if variant == "lora" else 2.5e-5
            lt.train_model(model, tok, training, config, device, MAX_LEN, HEAD_LEN, on_epoch_end=select)
            select = None
            model.load_state_dict(torch.load(output / f"{variant}-best.pt", weights_only=True, map_location=device))
            model.eval()
            before, _ = scores(lt, model, tok, validation, device)
            if variant == "lora":
                model.encoder = model.encoder.merge_and_unload()
                after, _ = scores(lt, model, tok, validation, device)
                merge_error = float(np.max(np.abs(before - after)))
                if merge_error > 1e-3 or not np.array_equal(before >= best["threshold"], after >= best["threshold"]):
                    raise ValueError(f"LoRA merge changed decisions: {merge_error}")
                write(output / "lora-merge-parity.json", {"max_margin_error": merge_error, "decisions_equal": True})
            lt.save_checkpoint(model, tok, cfg, str(destination))
            del model
            torch.cuda.empty_cache()
            model, tok, _ = lt.load_checkpoint(str(destination))
            model.to(device).eval()
            exported, _ = scores(lt, model, tok, validation, device)
            error = float(np.max(np.abs(exported - before)))
            equal = bool(np.array_equal(exported >= best["threshold"], before >= best["threshold"]))
            if error > 0.03 or not equal:
                raise ValueError(f"Native export parity failed: max margin error {error}, decisions_equal={equal}")
            write(output / f"{variant}-export-parity.json", {"max_margin_error": error, "decisions_equal": equal})
        summary[variant] = {
            "threshold": best["threshold"],
            "epoch": best["epoch"],
            "validation_accuracy": best["accuracy"],
            "parameters": count,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint_sha256": sha(destination / "model.safetensors"),
        }
        write(output / "selection.json", summary)
        select = None
        del model
        torch.cuda.empty_cache()
    write(output / "frozen.json", {"plan_sha256": sha(output / "plan.json"), "selected": summary})
    print("FROZEN", json.dumps(summary), flush=True)


def evaluate(args):
    import laya.train as lt
    import torch

    frozen = json.loads((args.output / "frozen.json").read_text())
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
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            scores(lt, model, tok, encoded[index : index + 1], device)
            if device.type == "cuda":
                torch.cuda.synchronize()
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
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
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
