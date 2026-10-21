"""Regenerate the fixed question embeddings with explicit, resumable SIE calls."""

from __future__ import annotations

import json
import os
import tempfile
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

import paired_training as core
from live import SingleSendClient as SingleSendClient, sie_client

DIMENSIONS = 2560
CREDIT_USD = Decimal("0.00001")


def write_json(path, value, *, replace=False):
    """Commit one complete file; a stopped attempt remains visible on resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    core.require(replace or not path.exists(), "Preserve existing receipt: " + path.name)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def charge(value):
    core.require(not isinstance(value, bool) and isinstance(value, (str, int, float)), "Invalid Cloud charge")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid Cloud charge") from error
    core.require(result.is_finite() and result >= 0, "Invalid Cloud charge")
    return result


def batch_metadata(reply, rows, execution, *, self_hosted):
    """SIE repeats request-level usage on each item; count that charge once."""
    core.require(isinstance(reply, list) and len(reply) == len(rows), "Wrong embedding result count")
    ids = [item.get("id") for item in reply]
    core.require(len(set(ids)) == len(ids) and set(ids) == {row["id"] for row in rows}, "Wrong embedding IDs")
    core.require(all(item.get("model") == core.MODEL for item in reply), "Wrong returned encoder model")
    metadata = [item.get("request") or {} for item in reply]
    usage = [item.get("usage") or meta.get("usage") or {} for item, meta in zip(reply, metadata, strict=True)]
    request_ids = [meta.get("id") for meta in metadata]
    core.require(all(value == request_ids[0] for value in request_ids), "Inconsistent batch request IDs")
    core.require(all(value == usage[0] for value in usage), "Inconsistent repeated batch usage")
    values = [
        meta.get("credits_debited", used.get("credits_charged_decimal", used.get("credits_charged")))
        for meta, used in zip(metadata, usage, strict=True)
    ]
    credits = None
    if not self_hosted:
        core.require(isinstance(request_ids[0], str) and bool(request_ids[0]), "Missing Cloud request ID")
        amounts = [charge(value) for value in values]
        core.require(all(value == amounts[0] for value in amounts), "Inconsistent repeated batch charge")
        credits = str(amounts[0])
    return {
        "request_id": request_ids[0],
        "model": core.MODEL,
        "hf_revision": core.REVISION,
        "execution_bundle_sha256": execution,
        "usage": usage[0],
        "cloud_credits": credits,
        "cloud_usd": None if credits is None else str(Decimal(credits) * CREDIT_USD),
        "cost_scope": "Unpriced self-hosted inference"
        if self_hosted
        else "Observed charge for this entire batch, counted once",
    }


def normalized_vectors(reply, rows):
    by_id = {item["id"]: item for item in reply}
    vectors = np.asarray([by_id[row["id"]]["dense"] for row in rows], dtype=np.float64)
    core.require(vectors.shape == (len(rows), DIMENSIONS) and np.isfinite(vectors).all(), "Invalid embedding vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    core.require(np.isfinite(norms).all() and (norms > 0).all(), "Zero/nonfinite embedding norm")
    return (vectors / norms).astype(np.float32)


def checked_vectors(path, count, digest):
    core.require(core.sha(path) == digest, "Completed batch vector bytes changed")
    values = np.load(path, allow_pickle=False, mmap_mode="r")
    core.require(
        values.shape == (count, DIMENSIONS)
        and values.dtype == np.float32
        and np.isfinite(values).all()
        and np.allclose(np.linalg.norm(values, axis=1), 1, rtol=0, atol=1e-5),
        "Completed batch vectors are not normalized float32",
    )
    return values


def request_for(rows):
    return {
        "model": core.MODEL,
        "items": [{"id": row["id"], "text": row["request"]} for row in rows],
        "output_types": ["dense"],
        "instruction": core.INSTRUCTION,
        "output_dtype": "float32",
        "is_query": True,
    }


def inventory(output, batches):
    """Verify every retained attempt and return the single completed attempt per batch."""
    selected, attempts = {}, []
    for index, rows in enumerate(batches):
        directory = output / "batches" / f"{index:06d}"
        previous = sorted(directory.glob("attempt-*"))
        core.require(
            [path.name for path in previous] == [f"attempt-{i:03d}" for i in range(len(previous))],
            "Attempt sequence changed",
        )
        for attempt in previous:
            intent_path = attempt / "intent.json"
            core.require(intent_path.is_file(), "Attempt has no durable intent")
            intent = core.read(intent_path)
            core.require(intent["request"] == request_for(rows), "Attempt request changed")
            receipt_path = attempt / "receipt.json"
            receipt = (
                core.read(receipt_path)
                if receipt_path.exists()
                else {
                    "status": "unknown",
                    "cloud_credits": None,
                }
            )
            if receipt_path.exists():
                core.require(receipt["intent_sha256"] == core.sha(intent_path), "Attempt intent changed")
            core.require(receipt["status"] in ("complete", "failed_or_unknown", "unknown"), "Unknown attempt status")
            if receipt["status"] == "complete":
                core.require(
                    index not in selected and receipt["ids"] == [row["id"] for row in rows],
                    "Duplicate completion or changed batch IDs",
                )
                checked_vectors(attempt / "vectors.npy", len(rows), receipt["vectors_sha256"])
                selected[index] = attempt
            attempts.append({"path": attempt.relative_to(output).as_posix(), **receipt})
    expected_dirs = {f"{i:06d}" for i in range(len(batches))}
    core.require(
        all(path.name in expected_dirs for path in (output / "batches").glob("*")), "Unexpected batch directory"
    )
    return selected, attempts


def summary(output, batches, self_hosted):
    selected, attempts = inventory(output, batches)
    known = sum((charge(row["cloud_credits"]) for row in attempts if row.get("cloud_credits") is not None), Decimal(0))
    unknown = [row["path"] for row in attempts if row.get("cloud_credits") is None and not self_hosted]
    result = {
        "format": "train-a-router-encoding-collection-v1",
        "status": "complete" if len(selected) == len(batches) else "incomplete",
        "complete_rows": sum(len(batches[index]) for index in selected),
        "expected_rows": sum(map(len, batches)),
        "physical_attempts": len(attempts),
        "known_cloud_credits": None if self_hosted else str(known),
        "known_cloud_usd": None if self_hosted else str(known * CREDIT_USD),
        "unknown_charge_attempts": unknown,
        "cost_scope": "Batch acquisition charges; these are not deployment singleton fees. Self-hosted compute is unpriced.",
        "attempts": [
            {
                "path": row["path"],
                "status": row["status"],
                "intent_sha256": core.sha(output / row["path"] / "intent.json"),
                "receipt_sha256": core.sha(output / row["path"] / "receipt.json")
                if (output / row["path"] / "receipt.json").exists()
                else None,
            }
            for row in attempts
        ],
    }
    write_json(output / "collection.json", result, replace=True)
    return selected, result


def collect_rows(rows, output, plan, client, *, retry_unknown=False, max_batches=None):
    """Internal small-batch engine, also used by mocked tests and the bounded smoke."""
    output = Path(output)
    core.require(type(plan["batch_size"]) is int and 1 <= plan["batch_size"] <= 16, "Use batches of 1–16")
    core.require(max_batches is None or type(max_batches) is int and max_batches > 0, "Invalid batch limit")
    core.require(rows and len({row["id"] for row in rows}) == len(rows), "Empty or duplicate input IDs")
    core.require(all(isinstance(row["request"], str) and row["request"].strip() for row in rows), "Empty request")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "encoding-plan.json"
    if plan_path.exists():
        core.require(core.read(plan_path) == plan, "Resume must use the same source, endpoint and encoding recipe")
    else:
        core.require(not any(output.iterdir()), "Use a new output directory; bundled features are immutable")
        write_json(plan_path, plan)
    batches = [rows[i : i + plan["batch_size"]] for i in range(0, len(rows), plan["batch_size"])]
    selected, ledger = summary(output, batches, plan["self_hosted"])
    new_calls, catalog = 0, None
    for index, batch in enumerate(batches):
        if index in selected:
            continue
        if max_batches is not None and new_calls >= max_batches:
            break
        directory = output / "batches" / f"{index:06d}"
        prior = sorted(directory.glob("attempt-*"))
        core.require(
            not prior or retry_unknown,
            "Unsettled attempt retained. Inspect it, then explicitly use --retry-unknown to make another call",
        )
        if catalog is None:
            catalog = client.get_model(core.MODEL)
            core.require(
                catalog.get("revision") == core.REVISION
                and catalog.get("dims", {}).get("dense") == DIMENSIONS
                and catalog.get("name", catalog.get("id", core.MODEL)) == core.MODEL,
                "The serving encoder differs from the fixed model/revision/dimensions",
            )
        attempt = directory / f"attempt-{len(prior):03d}"
        write_json(
            attempt / "intent.json",
            {
                "request": request_for(batch),
                "catalog_hf_revision": catalog["revision"],
                "retry_of_unknown_attempts": [path.relative_to(output).as_posix() for path in prior],
                "automatic_retries": False,
            },
        )
        metadata, started = {}, time.perf_counter()
        try:
            new_calls += 1
            reply = client.encode(
                **request_for(batch), wait_for_capacity=False, provision_timeout_s=150, max_oom_retries=0
            )
            metadata = batch_metadata(reply, batch, client.last_model_revision, self_hosted=plan["self_hosted"])
            core.require(client.last_retry_count == 0, "Unexpected automatic SDK retry")
            vectors = normalized_vectors(reply, batch)
            with (attempt / "vectors.npy").open("xb") as stream:
                np.save(stream, vectors, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            write_json(
                attempt / "receipt.json",
                {
                    **metadata,
                    "status": "complete",
                    "ids": [row["id"] for row in batch],
                    "intent_sha256": core.sha(attempt / "intent.json"),
                    "vectors_sha256": core.sha(attempt / "vectors.npy"),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
        except Exception as error:
            write_json(
                attempt / "receipt.json",
                {
                    **metadata,
                    "status": "failed_or_unknown",
                    "error_type": type(error).__name__,
                    "cloud_credits": metadata.get("cloud_credits"),
                    "intent_sha256": core.sha(attempt / "intent.json"),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            summary(output, batches, plan["self_hosted"])
            raise
    selected, ledger = summary(output, batches, plan["self_hosted"])
    if len(selected) != len(batches):
        return ledger
    if (output / "features.json").exists():
        info = core.read(output / "features.json")
        core.require(
            info["ids_sha256"] == core.sha(output / "ids.json")
            and info["vectors_sha256"] == core.sha(output / "vectors.npy")
            and info["collection_sha256"] == core.sha(output / "collection.json"),
            "Completed features changed",
        )
        return ledger
    ids = [row["id"] for row in rows]
    write_json(output / "ids.json", ids, replace=True)
    pending = output / "vectors.npy.pending"
    matrix = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float32, shape=(len(rows), DIMENSIONS))
    offset = 0
    for index, batch in enumerate(batches):
        matrix[offset : offset + len(batch)] = np.load(selected[index] / "vectors.npy", allow_pickle=False)
        offset += len(batch)
    matrix.flush()
    del matrix
    pending.replace(output / "vectors.npy")
    write_json(
        output / "features.json",
        {
            "format": "routerbench-fixed-embeddings-v1",
            "model": core.MODEL,
            "hf_revision": core.REVISION,
            "instruction": core.INSTRUCTION,
            "dimensions": DIMENSIONS,
            "dtype": "float32",
            "n": len(rows),
            "inputs_sha256": plan["inputs_sha256"],
            "source_plan_sha256": plan["source_plan_sha256"],
            "ids_sha256": core.sha(output / "ids.json"),
            "vectors_sha256": core.sha(output / "vectors.npy"),
            "collection_sha256": core.sha(output / "collection.json"),
            "encoding_plan_sha256": core.sha(plan_path),
            "normalization": "float64 L2 normalization then float32 storage",
            "outcomes_accessed": False,
            "fixed_encoder_no_fitting": True,
            "cost_scope": ledger["cost_scope"],
            "unknown_charge_attempts": ledger["unknown_charge_attempts"],
        },
    )
    return ledger


def collect(
    data, output, *, batch_size=16, sie_base_url=None, self_hosted=False, retry_unknown=False, max_batches=None
):
    """Only the pinned question-only file is read; no labels or answer records."""
    data, output = Path(data), Path(output)
    core.require(output.resolve() != (data / "features").resolve(), "Write regenerated features to a new directory")
    source = data / "source"
    core.require(core.sha(source / "source-plan.json") == core.SOURCE_PLAN_SHA256, "Different fixed source plan")
    source_plan = core.read(source / "source-plan.json")
    inputs = source / "question-only/inputs.json"
    core.require(
        core.sha(inputs) == source_plan["files_sha256"]["question-only/inputs.json"], "Question-only inputs changed"
    )
    original = core.read(inputs)
    core.require(len(original) == 16110, "The fixed source has 16,110 questions")
    rows = [{"id": row["id"], "request": row["request"]} for row in original]
    endpoint = (sie_base_url or os.environ.get("SIE_BASE_URL") or "https://api.superlinked.com").rstrip("/")
    url = urlsplit(endpoint)
    core.require(
        url.scheme in (("http", "https") if self_hosted else ("https",))
        and url.hostname
        and not (url.username or url.password or url.query or url.fragment),
        "Use a plain SIE URL without embedded credentials or query parameters",
    )
    plan = {
        "format": "train-a-router-encoding-plan-v1",
        "source_plan_sha256": core.SOURCE_PLAN_SHA256,
        "inputs_sha256": core.sha(inputs),
        "model": core.MODEL,
        "hf_revision": core.REVISION,
        "instruction": core.INSTRUCTION,
        "dimensions": DIMENSIONS,
        "is_query": True,
        "batch_size": batch_size,
        "self_hosted": bool(self_hosted),
        "endpoint": endpoint,
        "encoder_source_sha256": core.sha(__file__),
        "automatic_retries": False,
    }
    with sie_client(endpoint, self_hosted=self_hosted) as client:
        return collect_rows(rows, output, plan, client, retry_unknown=retry_unknown, max_batches=max_batches)
