"""Require every configured profile to be loaded at its pinned revision."""

from __future__ import annotations

import argparse

from sie_sdk import SIEClient

MODELS = {
    "cheap": (
        ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
        ("Qwen/Qwen3.5-4B:no-spec", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ),
    "strong": (("Qwen/Qwen3.8-27B-FP8", "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"),),
    "encoder": (("Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b"),),
}


def check(client, arm):
    for model, revision in MODELS[arm]:
        metadata = client.get_model(model)
        if (
            metadata.get("name") != model
            or metadata.get("loaded") is not True
            or metadata.get("revision") != revision
        ):
            raise RuntimeError(f"Pinned model is not loaded: {model}")
        if arm == "encoder" and metadata.get("dims", {}).get("dense") != 2560:
            raise RuntimeError("Encoder dimensions differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=MODELS)
    args = parser.parse_args()
    # Metadata reads never trigger a model load or inference request.
    with SIEClient("http://127.0.0.1:8080", api_key="", timeout_s=3, read_timeout_s=3) as client:
        check(client, args.arm)


if __name__ == "__main__":
    main()
