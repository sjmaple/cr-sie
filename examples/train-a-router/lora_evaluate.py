"""Evaluate the declared LoRA stages only after the complete study is frozen.

The historical test is an exposed regression benchmark. Test inputs are opened
only after validating every screening and expanded run using development data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from data import write
from evidence import verify as verify_evidence
from finetune import items, scores, sha
from lora_report import check_study, predictions, read
from lora_study import SEEDS

EAGER_ENV = {"TORCH_COMPILE_DISABLE": "1", "TORCHDYNAMO_DISABLE": "1"}
MAX_MARGIN_ERROR = 0.03


def margin_parity(actual, reference, threshold):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if (
        actual.ndim != 1
        or actual.shape != reference.shape
        or not len(actual)
        or not np.isfinite(actual).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Runtime parity needs finite aligned scores")
    return {
        "scores": actual.tolist(),
        "max_margin_error": float(np.max(np.abs(actual - reference))),
        "decision_changes": int(np.sum((actual >= threshold) != (reference >= threshold))),
    }


def parity_references(directory):
    selected = read(directory / "frozen.json")["selected"]["lora"]
    history = read(directory / "lora-history.json")
    validation = next(row for row in history if row["epoch"] == selected["epoch"])
    return selected, {"validation": validation, "calibration": read(directory / "lora-calibration.json")}


def runtime_parity(directory, development, device):
    """Measure only development inputs; called in an isolated, deadline-bound child."""
    if any(os.environ.get(name) != value for name, value in EAGER_ENV.items()):
        raise ValueError("Eager runtime flags must be set before importing the model runtime")

    import laya.train as lt
    import torch

    selected, references = parity_references(directory)
    checkpoint = directory / "lora/model.safetensors"
    if sha(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed before runtime parity")
    model, tokenizer, _ = lt.load_checkpoint(str(directory / "lora"))
    model.to(torch.device(device)).eval()
    result = {"runtime_environment": EAGER_ENV, "frozen_sha256": sha(directory / "frozen.json"), "splits": {}}
    for split, reference in references.items():
        data = items(lt, tokenizer, development / f"{split}.jsonl")
        values, labels = scores(lt, model, tokenizer, data, torch.device(device))
        result["splits"][split] = margin_parity(values, reference["scores"], selected["threshold"]) | {
            "labels": labels.tolist()
        }
    return result


def check_runtime_parity(directory, record):
    selected, references = parity_references(directory)
    if (
        record["runtime_environment"] != EAGER_ENV
        or record["frozen_sha256"] != sha(directory / "frozen.json")
        or set(record["splits"]) != set(references)
    ):
        raise ValueError("Runtime parity receipt differs from its frozen inputs")
    for split, reference in references.items():
        measured = record["splits"][split]
        expected = margin_parity(measured["scores"], reference["scores"], selected["threshold"]) | {
            "labels": reference["labels"]
        }
        if measured != expected or expected["max_margin_error"] > MAX_MARGIN_ERROR or expected["decision_changes"]:
            raise ValueError(f"Eager runtime parity failed for {split}; frozen cutoff decisions must be unchanged")


def evaluation_runs(study, selected_recipe):
    roots = (study / "screening/original", study / "screening" / selected_recipe, study / "expanded")
    return list(dict.fromkeys(root / f"seed-{seed}" for root in roots for seed in SEEDS))


def check_float32(path):
    """Inspect the safetensors header without importing a model runtime."""
    with path.open("rb") as handle:
        size = handle.read(8)
        if len(size) != 8 or not 0 < (length := int.from_bytes(size, "little")) <= 2_000_000:
            raise ValueError("Invalid checkpoint header size")
        header = json.loads(handle.read(length))
    tensors = [entry for name, entry in header.items() if name != "__metadata__"]
    if not tensors or any(entry.get("dtype") != "F32" for entry in tensors):
        raise ValueError("Checkpoint does not contain the declared float32 export")


def checkpoint_manifest(directory):
    frozen = read(directory / "frozen.json")
    model = directory / "lora"
    checkpoint = model / "model.safetensors"
    checkpoint_sha = sha(checkpoint)
    if checkpoint_sha != frozen["selected"]["lora"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint weights changed after freeze")
    check_float32(checkpoint)
    files = {
        str(path.relative_to(model)): sha(path)
        for path in sorted(model.rglob("*"))
        if path.is_file() and path != checkpoint
    }
    files["model.safetensors"] = checkpoint_sha
    required = {
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    }
    if not required <= files.keys():
        raise ValueError("Native checkpoint configuration or tokenizer is missing")
    return {
        "frozen_sha256": sha(directory / "frozen.json"),
        "plan_sha256": sha(directory / "plan.json"),
        "checkpoint_files_sha256": files,
    }


def run(args):
    if args.timeout_seconds <= 0:
        raise ValueError("Evaluation timeout must be positive")
    # Set before optional model imports in this process or either type of child.
    os.environ.update(EAGER_ENV)
    started = time.monotonic()
    # This helper reads only the four development inputs and frozen study metadata.
    study_plan, choice, _, _ = check_study(args.study, args.evidence)
    training_source = Path(__file__).with_name("finetune.py")
    if sha(training_source) != study_plan["training_source_sha256"]:
        raise ValueError("Use the same finetune.py source recorded by the frozen study")
    runs = evaluation_runs(args.study, choice["recipe"])
    targets = [
        args.study / f"{stage}-{status}.json"
        for stage in ("evaluation", "runtime-parity")
        for status in ("plan", "complete", "incomplete")
    ]
    targets.extend(run / "evaluation-test.json" for run in runs)
    targets.extend(run / "runtime-parity.json" for run in runs)
    if any(path.exists() for path in targets):
        raise FileExistsError("Evaluation artifacts already exist; existing observations are preserved")
    checkpoints = {str(run.relative_to(args.study)): checkpoint_manifest(run) for run in runs}

    def remaining_seconds():
        remaining = args.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Evaluation deadline reached")
        return remaining

    parity_plan = {
        "all_frozen_sha256": sha(args.study / "all-frozen.json"),
        "study_plan_sha256": sha(args.study / "study-plan.json"),
        "training_source_sha256": sha(training_source),
        "evaluation_source_sha256": sha(Path(__file__)),
        "development_data_sha256": study_plan["data_sha256"]["512"],
        "runtime_environment": EAGER_ENV,
        "max_margin_error": MAX_MARGIN_ERROR,
        "require_unchanged_frozen_cutoff_decisions": True,
        "references": {
            str(run.relative_to(args.study)): {
                file: sha(run / file) for file in ("lora-history.json", "lora-calibration.json")
            }
            for run in runs
        },
        "runs": checkpoints,
        "device": args.device,
        "timeout_seconds": args.timeout_seconds,
    }

    def check_inputs(directory):
        name = str(directory.relative_to(args.study))
        if (
            sha(args.study / "all-frozen.json") != parity_plan["all_frozen_sha256"]
            or sha(args.study / "study-plan.json") != parity_plan["study_plan_sha256"]
            or sha(training_source) != parity_plan["training_source_sha256"]
            or sha(Path(__file__)) != parity_plan["evaluation_source_sha256"]
            or sha(directory / "frozen.json") != checkpoints[name]["frozen_sha256"]
            or sha(directory / "plan.json") != checkpoints[name]["plan_sha256"]
        ):
            raise ValueError("Evaluation inputs changed after preflight")
        for file, expected in parity_plan["development_data_sha256"].items():
            if sha(args.evidence / file) != expected:
                raise ValueError("Development data changed after preflight")
        for file, expected in parity_plan["references"][name].items():
            if sha(directory / file) != expected:
                raise ValueError("Runtime parity reference changed after preflight")
        for filename, expected in checkpoints[name]["checkpoint_files_sha256"].items():
            # Both children verify the large weights immediately before loading.
            if filename != "model.safetensors" and sha(directory / "lora" / filename) != expected:
                raise ValueError("Checkpoint configuration changed after preflight")

    write(args.study / "runtime-parity-plan.json", parity_plan)
    parity_receipts = {}
    try:
        for index, directory in enumerate(runs, 1):
            name = str(directory.relative_to(args.study))
            check_inputs(directory)
            print(f"Development parity {index}/{len(runs)}: {name}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__)),
                    "--study",
                    str(args.study),
                    "--evidence",
                    str(args.evidence),
                    "--device",
                    args.device,
                    "--runtime-parity-run",
                    str(directory),
                ],
                check=True,
                timeout=remaining_seconds(),
            )
            receipt = directory / "runtime-parity.json"
            parity_receipts[name] = sha(receipt)
            check_inputs(directory)
            check_runtime_parity(directory, read(receipt))
    except Exception as error:
        write(
            args.study / "runtime-parity-incomplete.json",
            {"error_type": type(error).__name__, "parity_receipts_sha256": parity_receipts},
        )
        raise
    write(
        args.study / "runtime-parity-complete.json",
        {
            "runtime_parity_plan_sha256": sha(args.study / "runtime-parity-plan.json"),
            "parity_receipts_sha256": parity_receipts,
        },
    )

    # Only a fully frozen study whose eager decisions passed parity reaches test inputs.
    remaining_seconds()
    files = verify_evidence(args.evidence)
    test = args.evidence / "test.json"
    questions = args.evidence / "questions.json"
    test_sha = files["test.json"]["sha256"]
    rows = read(test)
    plan = {
        "all_frozen_sha256": sha(args.study / "all-frozen.json"),
        "study_plan_sha256": sha(args.study / "study-plan.json"),
        "evidence_manifest_sha256": sha(args.evidence / "manifest.json"),
        "test_sha256": test_sha,
        "questions_sha256": sha(questions),
        "training_source_sha256": sha(training_source),
        "evaluation_source_sha256": sha(Path(__file__)),
        "runtime_environment": EAGER_ENV,
        "runtime_parity_plan_sha256": sha(args.study / "runtime-parity-plan.json"),
        "runtime_parity_complete_sha256": sha(args.study / "runtime-parity-complete.json"),
        "selected_recipe": choice["recipe"],
        "runs": checkpoints,
        "device": args.device,
        "timeout_seconds": args.timeout_seconds,
        "population": "Exposed regression benchmark; no test-based selection or retraining",
    }
    write(args.study / "evaluation-plan.json", plan)
    completed = {}
    try:
        for directory in runs:
            name = str(directory.relative_to(args.study))
            check_inputs(directory)
            if (
                sha(test) != test_sha
                or sha(questions) != plan["questions_sha256"]
                or sha(args.study / "runtime-parity-plan.json") != plan["runtime_parity_plan_sha256"]
                or sha(args.study / "runtime-parity-complete.json") != plan["runtime_parity_complete_sha256"]
                or sha(directory / "runtime-parity.json") != parity_receipts[name]
            ):
                raise ValueError("Evaluation inputs changed after preflight")
            command = [
                sys.executable,
                str(training_source),
                "evaluate",
                "--evaluation",
                str(test),
                "--questions",
                str(questions),
                "--output",
                str(directory),
                "--device",
                args.device,
            ]
            print(f"Frozen test evaluation {len(completed) + 1}/{len(runs)}: {name}", flush=True)
            subprocess.run(command, check=True, timeout=remaining_seconds())
            predictions(directory, rows, test_sha)
            completed[name] = sha(directory / "evaluation-test.json")
    except Exception as error:
        write(
            args.study / "evaluation-incomplete.json",
            {"error_type": type(error).__name__, "completed_evaluations_sha256": completed},
        )
        raise
    write(
        args.study / "evaluation-complete.json",
        {
            "evaluation_plan_sha256": sha(args.study / "evaluation-plan.json"),
            "evaluations_sha256": completed,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(f"Evaluated {len(completed)} unique frozen runs; evaluation-complete.json records their hashes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--runtime-parity-run", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.runtime_parity_run is not None:
        os.environ.update(EAGER_ENV)
        write(
            args.runtime_parity_run / "runtime-parity.json",
            runtime_parity(args.runtime_parity_run, args.evidence, args.device),
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
