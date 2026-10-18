"""Serve one authenticated SIE model through a temporary Modal tunnel.

Run inside an owned Modal GPU function with SIE_AUTH_TOKEN supplied as a
Modal Secret. The caller owns the allocation TTL and cleanup. This optional
operator helper is not required for the laptop replay or Console workflow.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time


def main():
    import modal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    if not os.environ.get("SIE_AUTH_TOKEN", "").strip():
        parser.error("supply SIE_AUTH_TOKEN using a Modal Secret")
    if not 1 <= args.seconds <= 7200:
        parser.error("seconds must be between 1 and 7200")
    os.environ["SIE_AUTH_MODE"] = "token"
    versions = {}
    for name in ("sie-server", "torch", "transformers", "sglang", "flash-attn", "modal"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    print("RUNTIME=" + json.dumps(versions, sort_keys=True), flush=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sie_server.cli",
            "serve",
            "--host",
            "0.0.0.0",  # Modal tunnels connect through the container's network interface.
            "--models",
            args.model,
            "--preload",
            args.model,
            "--device",
            "cuda",
        ]
    )
    try:
        with modal.forward(8080) as tunnel:
            print("SERVER_URL=" + tunnel.url, flush=True)
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                status = process.poll()
                if status is not None:
                    return status or 1
                time.sleep(5)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
