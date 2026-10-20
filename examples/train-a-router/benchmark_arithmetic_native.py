"""Time a resident arithmetic native router on the saved 20 training requests.

One separate warmup is excluded. This measures two question forwards per
request, score normalization and a timing-only comparison with zero. It makes
no answer calls and does not select or change a calibrated routing policy.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import benefit
import benefit_outcomes_native as native
import finetune
from benchmark_outcomes_native import WarmRouter, measure_requests
from data import digest, file_hash, write


def inputs_and_checkpoint(inputs, directory, variants):
    """Check the question-only sample and selected exports before loading Torch."""
    inputs, directory = Path(inputs), Path(directory)
    sample = json.loads(inputs.read_text())
    plan_path, study_path = directory / "plan.json", directory.parent / "plan.json"
    plan, study = json.loads(plan_path.read_text()), json.loads(study_path.read_text())
    trained = benefit.checked(json.loads((directory / "trained.json").read_text()), native.TRAINED_FORMAT)
    rows = sample.get("rows", [])
    if (
        sample.get("salt") != "arithmetic-warm-v1"
        or sample.get("warmup") != 1
        or sample.get("timed") != 20
        or len(rows) != 21
        or any(
            not isinstance(row, dict)
            or set(row) != {"id", "request"}
            or not isinstance(row["id"], str)
            or not row["id"]
            or not isinstance(row["request"], str)
            or not row["request"].strip()
            for row in rows
        )
        or len({row["id"] for row in rows}) != 21
        or [row["id"] for row in rows] != study["warm_probe_ids"]
        or not {row["id"] for row in rows} <= set(study["training_ids"])
        or sample.get("source_plan_sha256") != study["source_plan_sha256"]
    ):
        raise ValueError("Need the saved21 question-only training inputs in the study's fixed order")
    if (
        plan["protocol"] != native.PROTOCOL
        or trained["plan_sha256"] != file_hash(plan_path)
        or trained["head_plan_sha256"] != file_hash(study_path)
        or plan["head_plan_sha256"] != file_hash(study_path)
        or not set(plan["training_ids"]) <= set(study["training_ids"])
    ):
        raise ValueError("Native checkpoint and arithmetic study bindings differ")
    verification = {}
    for variant in variants:
        started = time.perf_counter()
        native._manifest(directory, variant, trained["checkpoints"][variant]["files_sha256"], True)
        verification[variant] = time.perf_counter() - started
    return sample, trained, verification


def _runtime(device):
    # Optional training dependencies; core imports and --help need no Torch.
    import laya.train as lt
    import torch

    versions = {name: importlib.metadata.version(name) for name in (*native.PROTOCOL["versions"], "numpy")}
    if any(versions[name].split("+", 1)[0] != expected for name, expected in native.PROTOCOL["versions"].items()):
        raise ValueError("Use the pinned native inference dependencies")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    runtime_device = torch.device(device)
    metadata = {
        "device": device,
        "gpu": torch.cuda.get_device_name(runtime_device) if device == "cuda" else None,
        "cuda_version": torch.version.cuda,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "python": platform.python_version(),
        "versions": versions,
        "torch_threads": torch.get_num_threads(),
        "eager_flags": native.PROTOCOL["eager_flags"],
        "forward_amp": False,
    }
    return lt, torch, runtime_device, metadata


def _save(output, receipt):
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def run(inputs, directory, output, *, device, variant="lora"):
    inputs, directory, output = map(Path, (inputs, directory, output))
    if output.exists():
        raise FileExistsError(output)
    if device not in ("cpu", "cuda") or variant not in ("base", "lora", "both"):
        raise ValueError("Choose cpu or cuda and base, lora or both")
    variants = ("base", "lora") if variant == "both" else (variant,)
    sample, trained, verification = inputs_and_checkpoint(inputs, directory, variants)
    rows, warmup = sample["rows"][:20], sample["rows"][20]
    receipt = {
        "format": "sie-arithmetic-warm-native-v1",
        "status": "incomplete",
        "source_sha256": file_hash(Path(__file__)),
        "timing_helper_sha256": file_hash(Path(__file__).with_name("benchmark_outcomes_native.py")),
        "inputs_sha256": file_hash(inputs),
        "requests_sha256": digest(sample["rows"]),
        "native_plan_sha256": file_hash(directory / "plan.json"),
        "trained_sha256": file_hash(directory / "trained.json"),
        "study_plan_sha256": file_hash(directory.parent / "plan.json"),
        "timed_ids": [row["id"] for row in rows],
        "warmup_id": warmup["id"],
        "questions_per_request": 2,
        "request_batch_size": 1,
        "question_item_batch_size": 2,
        "excluded_warmup_requests_per_model": 1,
        "quantile_method": "linear",
        "scope": "Resident direct inference: tokenization, two question forwards, normalization and timing-only score>=0 comparison; synchronized before and after each request.",
        "load_scope": "Checkpoint load and device transfer after manifest verification, measured separately; not fresh-process or cold-disk startup.",
        "limitation": "20 training requests, no HTTP, concurrency or answer calls. Different-hardware Cloud timings are not a matched comparison. A static bypass policy skips this model work. Hosting dollars remain unknown.",
        "models": {},
    }
    write(output, receipt)
    os.environ.update(native.PROTOCOL["eager_flags"])
    try:
        started = time.perf_counter()
        lt, torch, runtime_device, runtime = _runtime(device)
        runtime["import_seconds"] = time.perf_counter() - started
        receipt["runtime"] = runtime
        for choice in variants:
            finetune.empty_cache(torch, runtime_device)
            finetune.synchronize(torch, runtime_device)
            started = time.perf_counter()
            router = WarmRouter(
                lt, directory / choice, runtime_device, trained["conditional_cost_statistics"],
                {"mode": "threshold", "cutoff": 0.0},
            )
            finetune.synchronize(torch, runtime_device)
            load_seconds = time.perf_counter() - started
            try:
                measurement = measure_requests(
                    router.score_one, rows, warmup, lambda: finetune.synchronize(torch, runtime_device)
                )
                measurement.update(
                    artifact_verification_seconds=verification[choice],
                    model_load_seconds_after_hash_verification=load_seconds,
                    parameters=trained["checkpoints"][choice]["parameters"],
                    weight_file_bytes=(directory / choice / "model.safetensors").stat().st_size,
                )
                receipt["models"][choice] = measurement
                _save(output, receipt)
                print(
                    f"{choice} ({device}): warm median {measurement['p50_ms']:.2f}ms; "
                    f"p95 {measurement['p95_ms']:.2f}ms; load {load_seconds:.2f}s",
                    flush=True,
                )
            finally:
                del router
                finetune.empty_cache(torch, runtime_device)
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["error_type"] = type(error).__name__
        raise
    finally:
        _save(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True, help="Arithmetic native-500 or native-2000 export directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--variant", choices=("base", "lora", "both"), default="lora")
    args = parser.parse_args()
    run(args.inputs, args.native, args.output, device=args.device, variant=args.variant)
    print(json.dumps({"output": str(args.output), "sha256": file_hash(args.output)}))


if __name__ == "__main__":
    main()
