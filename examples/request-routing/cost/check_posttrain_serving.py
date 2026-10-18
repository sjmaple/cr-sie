"""Verify native SIE Laya export and measure matched local adapter latency.

Run in an installed SIE server environment, after posttrain.py evaluate.
This is an optional GPU check, independent of the CPU workshop environment.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np


def main():
    import torch
    from sie_server.adapters.laya.adapter import LayaAdapter
    from sie_server.adapters.laya.questions import Calibration
    from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter
    from sie_server.types.inputs import Item

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--output-name", default="sie-serving.json")
    args = parser.parse_args()
    output = args.run / args.output_name
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    frozen = json.loads((args.run / "frozen.json").read_text())
    evaluation = json.loads((args.run / f"evaluation-{args.evaluation.stem}.json").read_text())
    rows = json.loads(args.evaluation.read_text())
    questions = json.loads(args.questions.read_text())
    results = {
        "device": torch.cuda.get_device_name(),
        "precision": "bfloat16 CUDA projections",
        "torch_threads": torch.get_num_threads(),
        "scope": "Direct native SIE adapters; no network, queueing or billing",
        "models": {},
    }
    order = ("full", "qwen4b", "lora", "lora", "qwen4b", "full") if args.latency_only else ("full", "lora", "qwen4b")
    for round_index, name in enumerate(order):
        if name == "qwen4b":
            adapter = PyTorchEmbeddingAdapter(
                "Qwen/Qwen3-Embedding-4B",
                revision="5cf2132abc99cad020ac570b19d031efec650f2b",
                compute_precision="bfloat16",
                pooling="last_token",
                normalize=True,
                query_template="Instruct: {instruction}\nQuery: {text}",
                trust_remote_code=False,
            )
        else:
            adapter = LayaAdapter(str(args.run / name), compute_precision="bfloat16")
        adapter.load("cuda")
        record = {}
        if name != "qwen4b" and not args.latency_only:
            cfg = json.loads((args.run / name / "rl_agent_config.json").read_text())
            temperature = Calibration.from_config(cfg).temperature_for("choice", 2)
            probability_cutoff = float(1 / (1 + np.exp(-frozen["selected"][name]["threshold"] / temperature)))
            predictions, probabilities = [], []
            for start in range(0, len(rows), 16):
                reply = adapter.extract(
                    [Item(text=r["request"]) for r in rows[start : start + 16]], output_schema=questions
                )
                values = [r["difficulty"]["probabilities"]["advanced"] for r in reply.data]
                probabilities.extend(values)
                predictions.extend(int(v >= probability_cutoff) for v in values)
            reference = evaluation["models"][name]["predictions"]
            changed = int(np.sum(np.array(predictions) != reference))
            record.update(
                probability_cutoff=probability_cutoff,
                predictions=predictions,
                probabilities=probabilities,
                changed_vs_float32_reference=changed,
                accuracy=float(np.mean(np.array(predictions) == [r["label"] for r in rows])),
            )
            if changed > 2:
                raise ValueError(f"Native SIE export changed {changed} decisions for {name}")
        timings = []
        torch.cuda.reset_peak_memory_stats()
        warmup, measured = (20, 128) if args.latency_only else (5, 30)
        # Reuse the same text prefix for every model and both loading orders.
        timed_rows = rows[:warmup] + rows[:measured]
        for index, row in enumerate(timed_rows):
            torch.cuda.synchronize()
            started = time.perf_counter()
            if name == "qwen4b":
                adapter.encode(
                    [Item(text=row["request"])],
                    ["dense"],
                    is_query=True,
                    instruction="Represent the difficulty of the mathematics problem for classifying it as routine or advanced.",
                )
            else:
                adapter.extract([Item(text=row["request"])], output_schema=questions)
            torch.cuda.synchronize()
            if index >= warmup:
                timings.append(time.perf_counter() - started)
        record["batch1_seconds"] = {
            "median": float(np.median(timings)),
            "p95": float(np.quantile(timings, 0.95)),
            "samples": len(timings),
        }
        record["peak_torch_memory_bytes"] = torch.cuda.max_memory_allocated()
        key = f"{round_index + 1}-{name}" if args.latency_only else name
        results["models"][key] = record
        output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            name,
            {key: value for key, value in record.items() if key not in ("predictions", "probabilities")},
            flush=True,
        )
        adapter.unload()
        del adapter
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
