"""Check the three pinned workers through one local SIE endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import serving
import solver
from data import file_hash, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if urlparse(os.environ.get("SIE_BASE_URL", "")).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("This smoke test only calls an explicitly configured loopback endpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"format": "sie-router-self-host-smoke-v1", "status": "incomplete", "calls": [],
               "source_sha256": {p.name: file_hash(p) for p in (Path(__file__), Path(solver.__file__), Path(serving.__file__))},
               "observed_cloud_credits": None, "hosting_cost": "unpriced"}
    started = time.monotonic()
    try:
        with serving.SingleSendClient(timeout_s=300, read_timeout_s=300) as client:
            metadata = {}
            for model, revision in [(serving.MODEL, serving.REVISION),
                                    *[(arm["model"], arm["revision"]) for arm in solver.SETTINGS["arms"].values()]]:
                value = client.get_model(model)
                if value["revision"] != revision:
                    raise ValueError("Pinned worker model revision differs")
                metadata[model] = value
            if metadata[serving.MODEL]["dims"]["dense"] != 2560:
                raise ValueError("Encoder dimensions differ")
            receipt["metadata"] = metadata
            write(args.output / "metadata.json", metadata)
            request = "Competition: AMC10\nProblem: A rectangle has perimeter 30 and length 9. What is its area?"
            vector, usage = serving.embed(request)
            if vector.shape != (2560,):
                raise ValueError("Embedding shape differs")
            receipt["embedding"] = {"dimensions": len(vector), "squared_norm": float(vector @ vector), "receipt": usage}
            for name, arm in solver.SETTINGS["arms"].items():
                print(f"Checking {name} initial response and forced structured finalizer", flush=True)
                messages = [{"role": "system", "content": arm["system"]}, {"role": "user", "content": request}]
                first = solver.call(client, arm, messages, self_hosted=True)
                entry = {"arm": name, "model": arm["model"], "initial": first}
                receipt["calls"].append(entry)
                write(args.output / f"{name}-initial.json", entry)
                if not first["response_valid"] or not first["answer"].strip():
                    raise ValueError("Worker returned no initial answer")
                final_messages = [{"role": "system", "content": solver.FINAL_SYSTEM},
                                  {"role": "user", "content": json.dumps({"problem": request, "partial_work": first["answer"]})}]
                final = solver.call(client, arm, final_messages, final=True, self_hosted=True)
                structured = json.loads(final["answer"])
                if not isinstance(structured, dict) or set(structured) != {"answer"} or not isinstance(structured["answer"], str):
                    raise ValueError("Forced finalizer violated the recorded JSON schema")
                entry["forced_finalizer"] = final
                write(args.output / f"{name}.json", entry)
        receipt["status"] = "complete"
    except Exception as error:
        receipt["error"] = serving.error_receipt(error)
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        write(args.output / "smoke.json", receipt)


if __name__ == "__main__":
    main()
