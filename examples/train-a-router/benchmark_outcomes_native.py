"""Measure warm, single-request inference for a frozen two-question native router.

Hash-ordered training inputs are shared by the base and LoRA measurements. This
is direct local inference; model loading, artifact checks, HTTP transport and
Cloud generation charges are separate from the warm inference measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import benefit_outcomes as outcomes
import benefit_outcomes_native as native
import finetune
from data import digest, file_hash

FORMAT = "sie-native-outcome-warm-inference-v1"
SAMPLE_SALT = "sie-native-outcome-warm-inference-v1"
N = 20
REQUEST_BATCH_SIZE = 1
QUESTION_ITEM_BATCH_SIZE = 2


def select_inputs(original_rows, eligible_ids):
    """Sample reviewed original inputs shared by either training size."""
    identities = [row["id"] for row in original_rows]
    if len(identities) != 512 or len(set(identities)) != len(identities):
        raise ValueError("Expected the exact original512 source pool with unique IDs")
    eligible = set(eligible_ids)
    if len(eligible) != len(eligible_ids):
        raise ValueError("Reviewed training IDs must be unique")
    rows = [row for row in original_rows if row["id"] in eligible]
    if len(rows) < N + 1:
        raise ValueError("Benchmark needs at least21 reviewed original training inputs")
    ordered = sorted(
        rows,
        key=lambda row: (hashlib.sha256(f"{SAMPLE_SALT}\0{row['id']}".encode()).hexdigest(), row["id"]),
    )
    return ordered[:N], ordered[N]


def _measurement(value):
    if (
        not isinstance(value, dict)
        or not np.isfinite(value.get("score", np.nan))
        or np.asarray(value.get("margins"), dtype=float).shape != (2,)
        or not np.isfinite(value["margins"]).all()
        or value.get("route") not in ("cheap", "strong")
    ):
        raise ValueError("Warm inference must return a finite two-outcome score and route")
    return value


def measure_requests(score_one, rows, warmup, synchronize, *, clock=time.perf_counter):
    if len(rows) != N or warmup["id"] in {row["id"] for row in rows}:
        raise ValueError("Need20 timed requests and one separate warmup request")
    synchronize()
    started = clock()
    _measurement(score_one(warmup))
    synchronize()
    warmup_seconds = clock() - started
    timings = []
    for row in rows:
        synchronize()
        started = clock()
        result = _measurement(score_one(row))
        synchronize()
        elapsed = clock() - started
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid monotonic-clock duration")
        timings.append({"id": row["id"], "seconds": elapsed, **result})
    if not np.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("Invalid warmup duration")
    seconds = np.asarray([row["seconds"] for row in timings])
    return {
        "warmup_id": warmup["id"],
        "warmup_seconds_excluded": warmup_seconds,
        "requests": timings,
        "n": len(timings),
        "p50_ms": float(np.percentile(seconds, 50, method="linear") * 1000),
        "p95_ms": float(np.percentile(seconds, 95, method="linear") * 1000),
        "mean_ms": float(seconds.mean() * 1000),
        "total_timed_seconds": float(seconds.sum()),
    }


class WarmRouter:
    """Keep one checkpoint loaded across requests; reuse the frozen scoring path."""

    def __init__(self, lt, checkpoint, device, statistics, policy):
        self.lt, self.device, self.statistics, self.policy = lt, device, statistics, policy
        self.model, self.tokenizer, _ = lt.load_checkpoint(str(checkpoint))
        self.model.encoder.config.reference_compile = False
        self.model.to(device).eval()

    def score_one(self, row):
        items = native.encode_rows(self.lt, self.tokenizer, [row])
        records = self.lt.calibration_records(
            self.model,
            self.tokenizer,
            items,
            self.device,
            finetune.MAX_LEN,
            finetune.HEAD_LEN,
            batch_size=QUESTION_ITEM_BATCH_SIZE,
        )
        margins = native.records_to_margins(records, np.zeros((1, 2)))
        scores, _ = outcomes.normalized_scores(native.probabilities(margins), self.statistics)
        score = float(scores[0])
        mode = self.policy["mode"]
        if mode == "threshold":
            strong = score >= self.policy["cutoff"]
        elif mode in ("always_strong", "always_cheap"):
            strong = mode == "always_strong"
        else:
            raise ValueError("Unknown frozen policy mode")
        return {
            "score": score,
            "margins": margins[0].tolist(),
            "route": "strong" if strong else "cheap",
            "state_tokens": len(items[0]["state_ids"]),
        }


def _save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(run_directory, head_run, baseline, output, *, device="cuda"):
    run_directory, head_run, baseline, output = map(Path, (run_directory, head_run, baseline, output))
    if output.exists():
        raise FileExistsError(output)
    if device not in ("cuda", "mps", "cpu"):
        raise ValueError("Choose an explicit supported inference device")
    plan, trained, frozen = native.checked_frozen(run_directory, head_run)
    dataset = outcomes._baseline(plan["head_plan"], baseline)
    rows, warmup = select_inputs(
        dataset["splits"]["train"], [row["id"] for row in plan["head_plan"]["populations"]["train"]]
    )
    receipt = {
        "format": FORMAT,
        "status": "incomplete",
        "source_sha256": file_hash(Path(__file__)),
        "native_source_sha256": native.code_hashes(),
        "native_plan_sha256": file_hash(run_directory / "plan.json"),
        "native_trained_sha256": file_hash(run_directory / "trained.json"),
        "native_frozen_sha256": file_hash(run_directory / "frozen.json"),
        "dataset_sha256": plan["head_plan"]["dataset_sha256"],
        "sampling": {
            "eligible_original_rows": len(
                {row["id"] for row in dataset["splits"]["train"]}
                & {row["id"] for row in plan["head_plan"]["populations"]["train"]}
            ),
            "ordering": f"sha256({SAMPLE_SALT!r} + NUL + id), then id; first20 timed,21st warmup",
            "timed_ids": [row["id"] for row in rows],
            "warmup_id": warmup["id"],
            "requests_sha256": digest([{key: row[key] for key in ("id", "request")} for row in [*rows, warmup]]),
            "paper_groups": [row["paper_group"] for row in rows],
            "request_characters": [len(row["request"]) for row in rows],
        },
        "questions_per_request": 2,
        "request_batch_size": REQUEST_BATCH_SIZE,
        "question_item_batch_size": QUESTION_ITEM_BATCH_SIZE,
        "excluded_warmup_requests_per_model": 1,
        "quantile_method": "linear",
        "timed_scope": "Tokenization, both native question forwards, CPU score conversion, predicted-cost normalization and primary frozen cutoff; synchronize device before and after each request",
        "load_scope": "Checkpoint load and device transfer, measured separately after weight-file hash verification; this is not a cold-disk or fresh-process startup measurement",
        "limitation": "A small direct-inference probe on20 training inputs, with one resident model and no HTTP or concurrent requests. Different-hardware embedding or Cloud timings are not matched comparisons. No serving dollars are inferred. A bypass policy would skip this model work in deployment.",
        "models": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.environ.update(native.PROTOCOL["eager_flags"])
    started = time.perf_counter()
    try:
        import laya.train as lt
        import torch

        runtime_device = torch.device(device)
        versions = {name: importlib.metadata.version(name) for name in (*native.PROTOCOL["versions"], "numpy")}
        if any(versions[name].split("+", 1)[0] != expected for name, expected in native.PROTOCOL["versions"].items()):
            raise ValueError("Native benchmark requires the pinned training/inference dependencies")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable")
        receipt["runtime"] = {
            "device": device,
            "gpu": torch.cuda.get_device_name(runtime_device) if device == "cuda" else None,
            "cuda_version": torch.version.cuda,
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
            "versions": versions,
            "import_seconds": time.perf_counter() - started,
            "eager_flags": native.PROTOCOL["eager_flags"],
            "forward_amp": False,
        }
        for variant in ("base", "lora"):
            verification_started = time.perf_counter()
            native._manifest(run_directory, variant, trained["checkpoints"][variant]["files_sha256"], True)
            verification_seconds = time.perf_counter() - verification_started
            finetune.empty_cache(torch, runtime_device)
            finetune.synchronize(torch, runtime_device)
            started = time.perf_counter()
            router = WarmRouter(
                lt,
                run_directory / variant,
                runtime_device,
                trained["conditional_cost_statistics"],
                frozen["models"][variant]["policies"]["1"],
            )
            finetune.synchronize(torch, runtime_device)
            load_seconds = time.perf_counter() - started
            try:
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats(runtime_device)
                measurement = measure_requests(
                    router.score_one, rows, warmup, lambda: finetune.synchronize(torch, runtime_device)
                )
                measurement.update(
                    artifact_verification_seconds=verification_seconds,
                    model_load_seconds_after_hash_verification=load_seconds,
                    parameters=sum(parameter.numel() for parameter in router.model.parameters()),
                    parameter_dtypes=sorted({str(parameter.dtype) for parameter in router.model.parameters()}),
                    weight_file_bytes=(run_directory / variant / "model.safetensors").stat().st_size,
                    primary_policy_mode=router.policy["mode"],
                    cuda_peak_allocated_mib=(
                        torch.cuda.max_memory_allocated(runtime_device) / 2**20 if device == "cuda" else None
                    ),
                    cuda_peak_reserved_mib=(
                        torch.cuda.max_memory_reserved(runtime_device) / 2**20 if device == "cuda" else None
                    ),
                )
                receipt["models"][variant] = measurement
                _save(output, receipt)
                print(
                    f"{variant}: warm p50={measurement['p50_ms']:.2f}ms p95={measurement['p95_ms']:.2f}ms; load after hash={load_seconds:.2f}s",
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args.run, args.head_run, args.baseline, args.output, device=args.device)
    print(json.dumps({"output": str(args.output), "sha256": file_hash(args.output)}))


if __name__ == "__main__":
    main()
