"""Bounded development-only comparison of native LoRA recipes, then training scale.

Only train/validation/calibration/questions are accepted as experiment inputs.
The existing published test is an exposed regression benchmark and is evaluated
separately, after every candidate and the recipe choice have been frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from finetune import BASE, LORA_RECIPES, REVISION, sha, write

SEEDS = (89, 137, 211)
FILES = ("train.jsonl", "validation.jsonl", "calibration.jsonl", "questions.json")


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {
        name: {file: sha(path / file) for file in FILES}
        for name, path in (("512", args.data), ("expanded", args.expanded_data))
    }
    for name in FILES[1:]:
        if hashes["512"][name] != hashes["expanded"][name]:
            raise ValueError(f"Development inputs changed between training sizes: {name}")
    small = (args.data / "train.jsonl").read_bytes()
    expanded = (args.expanded_data / "train.jsonl").read_bytes()
    if len(small.splitlines()) != 512 or not expanded.startswith(small) or len(expanded.splitlines()) != 2581:
        raise ValueError("Expected 512-row prefix and 2,581-row expansion")
    plan = {
        "base": BASE,
        "revision": REVISION,
        "recipes": LORA_RECIPES,
        "seeds": SEEDS,
        "epochs": 8,
        "export_dtype": "float32",
        "export_amendment": (
            "Before any test access or recipe selection, preserve float32 weights for every run. "
            "The first original/seed89 attempt completed training but failed the float16 export-margin check "
            "(max drift 0.043626606464385986 > 0.03). Its history is retained as an audit; "
            "no learning settings or candidates changed."
        ),
        "data_sha256": hashes,
        "training_source_sha256": sha(Path(__file__).with_name("finetune.py")),
        "driver_source_sha256": sha(Path(__file__)),
        "selection": (
            "Checkpoint: maximum 2018 validation AUC, epoch zero eligible, earliest epoch wins ties. "
            "Recipe: maximum mean selected validation AUC across all three seeds; declaration order wins ties. "
            "Freeze recipe before training all 2,581 eligible pre-2018 families, with unchanged settings and seeds. "
            "Fit a separate 2019 cutoff after each checkpoint selection; calibration never selects recipe or epoch."
        ),
        "test_access": "No test inputs. Published test is reserved for a later exposed-regression comparison.",
        "timeout_seconds": args.timeout_seconds,
    }
    args.output.mkdir(parents=True)
    write(args.output / "study-plan.json", plan)
    started = time.monotonic()
    env = os.environ.copy()
    env.pop("UV_NO_SYNC", None)
    env["TOKENIZERS_PARALLELISM"] = "false"

    def train(recipe, seed, data, output):
        remaining = args.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Study deadline reached")
        command = [
            sys.executable,
            str(Path(__file__).with_name("finetune.py")),
            "train",
            "--data",
            str(data),
            "--output",
            str(output),
            "--seed",
            str(seed),
            "--recipe",
            recipe,
            "--validation-metric",
            "auc",
            "--variants",
            "lora",
            "--export-dtype",
            "float32",
            "--device",
            args.device,
        ]
        print(json.dumps({"starting": recipe, "seed": seed, "data": data.name}), flush=True)
        subprocess.run(command, env=env, check=True, timeout=remaining)
        frozen = json.loads((output / "frozen.json").read_text())
        result = frozen["selected"]["lora"]
        # Native exports and all histories remain; optimizer-era copies are disposable.
        (output / "lora-best.pt").unlink()
        shutil.rmtree(output / "base")
        return {"frozen_sha256": sha(output / "frozen.json"), **result}

    screening = {}
    for recipe in LORA_RECIPES:
        screening[recipe] = {}
        for seed in SEEDS:
            screening[recipe][str(seed)] = train(
                recipe, seed, args.data, args.output / "screening" / recipe / f"seed-{seed}"
            )
            write(args.output / "screening.json", screening)
    means = {
        recipe: sum(run["validation_auc"] for run in runs.values()) / len(SEEDS) for recipe, runs in screening.items()
    }
    chosen = max(means, key=means.get)
    choice = {
        "recipe": chosen,
        "mean_validation_auc": means,
        "study_plan_sha256": sha(args.output / "study-plan.json"),
        "screening_sha256": sha(args.output / "screening.json"),
    }
    write(args.output / "recipe-frozen.json", choice)
    print("RECIPE_FROZEN", json.dumps(choice), flush=True)
    scaling = {}
    for seed in SEEDS:
        scaling[str(seed)] = train(chosen, seed, args.expanded_data, args.output / "expanded" / f"seed-{seed}")
        write(args.output / "expanded.json", scaling)
    write(
        args.output / "all-frozen.json",
        {
            "study_plan_sha256": sha(args.output / "study-plan.json"),
            "recipe_frozen_sha256": sha(args.output / "recipe-frozen.json"),
            "screening_sha256": sha(args.output / "screening.json"),
            "expanded_sha256": sha(args.output / "expanded.json"),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print("ALL_LORA_RUNS_FROZEN", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expanded-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=3900)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    run(args)


if __name__ == "__main__":
    main()
