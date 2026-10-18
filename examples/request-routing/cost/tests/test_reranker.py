"""Workshop contracts: exact scoring request, paired evidence, and disjoint optimization data."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import msgpack
import numpy as np

import compare
import fetch
import prepare_gepa
import reranker
import run
import study

SPEC = {
    "query": {"text": "A request that a small, cheap AI model can answer correctly."},
    "items": [{"id": "d", "text": "__TEXT__"}],
    "instruction": "Judge whether the Document, a request sent to an AI assistant, fits the description in the Query.",
}


class RerankerTests(unittest.TestCase):
    def test_native_request_preserves_the_study_query_instruction_and_document(self):
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(
                200,
                headers={"Content-Type": "application/msgpack"},
                content=msgpack.packb({"model": reranker.MODEL, "scores": [{"item_id": "d", "score": 0.2, "rank": 0}]}),
            )

        transport = httpx.Client(base_url="https://api.superlinked.com", transport=httpx.MockTransport(respond))
        with (
            mock.patch.dict(os.environ, {"SIE_API_KEY": "test-key"}, clear=True),
            run.SingleSendClient(http_client=transport) as client,
        ):
            self.assertEqual(reranker.score_request(client, "Solve x + 1 = 2", SPEC), 0.2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].url.path, "/v1/score/Qwen/Qwen3-Reranker-4B")
        body = msgpack.unpackb(calls[0].content, raw=False)
        self.assertEqual(body["query"], SPEC["query"])
        self.assertEqual(body["instruction"], SPEC["instruction"])
        self.assertEqual(body["items"], [{"id": "d", "text": "Solve x + 1 = 2"}])
        self.assertEqual(body["options"], {"profile": "default"})

    def test_rejects_missing_duplicate_wrong_document_or_invalid_scores(self):
        cases = [[], [None], [{"item_id": "other", "score": 0.2}], [{"item_id": "d", "score": 0.2}] * 2]
        cases += [[{"item_id": "d", "score": v}] for v in (True, None, float("nan"), float("inf"), -0.1, 1.1)]
        for scores in cases:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                reranker.response_score({"scores": scores})

    def test_model_loading_does_not_trigger_a_second_send(self):
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(
                503, headers={"Retry-After": "1"}, json={"detail": {"code": "MODEL_LOADING", "message": "loading"}}
            )

        transport = httpx.Client(base_url="http://localhost:8080", transport=httpx.MockTransport(respond))
        with (
            run.SingleSendClient(base_url="http://localhost:8080", http_client=transport) as client,
            self.assertRaises(run.RetryDisabled),
        ):
            reranker.score_request(client, "request", SPEC)
        self.assertEqual(len(calls), 1)

    def test_threshold_rejects_nonfinite_and_out_of_range_values(self):
        for value in ("nan", "inf", "-0.1", "1.1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                run.threshold(value)


class EvidenceTests(unittest.TestCase):
    def test_download_addendum_uses_its_own_revision_and_manifest(self):
        files = {name: f"fixture {name}".encode() for name in reranker.FILES}
        manifest = json.dumps(
            {"files": {name: {"sha256": study.sha256(body), "bytes": len(body)} for name, body in files.items()}}
        ).encode()
        addresses = []

        def download(address, max_bytes):
            addresses.append(address)
            name = address.split(reranker.FOLDER + "/", 1)[1]
            return manifest if name == study.MANIFEST else files[name]

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(reranker, "MANIFEST_SHA256", study.sha256(manifest)),
        ):
            destination = Path(temporary) / "reranker"
            self.assertTrue(fetch.fetch(destination, download, source=reranker))
            self.assertFalse(fetch.fetch(destination, download, source=reranker))
            self.assertTrue(all(reranker.REVISION in address for address in addresses))
            (destination / "results.json").write_text("changed")
            with self.assertRaises(study.EvidenceError):
                reranker.verify_evidence(destination)

    def test_misaligned_or_duplicate_ids_cannot_be_silently_compared(self):
        for rows in ([{"id": "a", "signal": "qwen3_rr4b_q1"}] * 2, [{"id": "wrong", "signal": "qwen3_rr4b_q1"}]):
            with (
                mock.patch.object(reranker, "verify_evidence"),
                mock.patch.object(study, "read_jsonl", return_value=rows),
                self.assertRaises(study.EvidenceError),
            ):
                reranker.load(Path("unused"), ["a"])

    def test_comparison_rejects_a_changed_threshold_or_result(self):
        outcomes = study.Outcomes(
            q_cheap=np.array([0.0, 1.0, 0.0, 1.0]),
            q_strong=np.ones(4),
            usd_cheap=np.ones(4),
            usd_strong=np.full(4, 10.0),
            usd_router=np.zeros(4),
            score=np.zeros(4),
            route=[],
        )
        s = SimpleNamespace(pairs={"openai": outcomes}, calibration=np.array([True, True, False, False]))
        recording = reranker.Recording(
            np.array([0.1, 0.9, 0.2, 0.8]),
            np.zeros(4),
            SPEC,
            {"openai": {"tau": 0.1}},
        )
        with self.assertRaises(study.EvidenceError):
            compare.check_reranker(s, "openai", recording)


class OptimizationDataTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_stratified_and_never_exports_heldout(self):
        s = SimpleNamespace(
            ids=[f"id-{i}" for i in range(40)],
            strata=np.array(["math"] * 20 + ["chat"] * 20),
            calibration=np.array([True, False] * 20),
        )
        groups = prepare_gepa.partitions(s)
        self.assertEqual(groups, prepare_gepa.partitions(s))
        flattened = [i for indices in groups.values() for i in indices]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(np.where(s.calibration)[0]))
        self.assertEqual(
            {name: len(ids) for name, ids in groups.items()}, {"train": 12, "validation": 4, "threshold": 4}
        )
        for indices in groups.values():
            self.assertEqual(set(s.strata[indices]), {"math", "chat"})


if __name__ == "__main__":
    unittest.main()
