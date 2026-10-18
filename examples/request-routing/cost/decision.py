"""Serve and use a frozen native Laya checkpoint through SIE's public SDK."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from sie_sdk.types import Item

from run import SingleSendClient


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probability_cutoff(config, margin):
    # Match SIE's native choice:2 temperature lookup. This preserves the frozen
    # logit-margin decision; the resulting score is not a correctness estimate.
    temperature = (config.get("temperature_by_options") or {}).get("choice:2")
    if temperature is None:
        temperature = config.get("temperature", [1.0, 1.0, 1.0])[0]
    temperature = float(temperature)
    if not math.isfinite(temperature):
        temperature = 1.0
    temperature = max(0.5, min(5.0, temperature))
    return 1 / (1 + math.exp(-margin / temperature))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure", "route"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--variant", choices=("full", "lora"), default="full")
    parser.add_argument("--models-dir", type=Path, default=Path("run-output/decision-models"))
    parser.add_argument("--request")
    args = parser.parse_args()
    frozen = json.loads((args.run / "frozen.json").read_text())
    selected = frozen["selected"][args.variant]
    checkpoint = (args.run / args.variant).resolve()
    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())
    model_id = f"workshop/laya-difficulty-{args.variant}"
    threshold = probability_cutoff(cfg, selected["threshold"])
    if args.command == "configure":
        if checksum(checkpoint / "model.safetensors") != selected["checkpoint_sha256"]:
            raise ValueError("Checkpoint does not match the frozen selection")
        # JSON is valid YAML and safely quotes any spaces in the local path.
        config = {
            "sie_id": model_id,
            "weights_path": str(checkpoint),
            "inputs": {"text": True, "image": False, "audio": False, "video": False},
            "tasks": {"encode": None, "score": None, "extract": {}},
            "max_sequence_length": 512,
            "profiles": {
                "default": {
                    "max_batch_tokens": 8192,
                    "compute_precision": "bfloat16",
                    "adapter_path": "sie_server.adapters.laya.adapter:LayaAdapter",
                    "adapter_options": {"loadtime": {}, "runtime": {}},
                }
            },
        }
        args.models_dir.mkdir(parents=True, exist_ok=True)
        destination = args.models_dir / f"laya-{args.variant}.yaml"
        with destination.open("x") as handle:
            json.dump(config, handle, indent=2)
        print(f"Verified frozen checkpoint. SIE config: {destination}; model ID: {model_id}")
        print(f"Frozen native-score cutoff: {threshold:.9f}")
        return
    if not args.request or not os.environ.get("SIE_BASE_URL"):
        parser.error("route requires --request and SIE_BASE_URL for your checkpoint's SIE server")
    questions = json.loads((args.run / "questions.json").read_text())
    with SingleSendClient(timeout_s=120, read_timeout_s=120) as client:
        reply = client.extract(
            model_id,
            Item(text=args.request),
            output_schema=questions,
            wait_for_capacity=False,
            max_oom_retries=0,
            provision_timeout_s=60,
        )
    value = float(reply["data"]["difficulty"]["probabilities"]["advanced"])
    print(
        json.dumps(
            {
                "route": "advanced" if value >= threshold else "routine",
                "native_score": value,
                "threshold": threshold,
                "model": model_id,
                "new_inference_calls": 1,
                "score_note": "Native normalized option score, not a calibrated probability of correctness",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
