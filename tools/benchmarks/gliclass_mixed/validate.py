"""Validate the frozen workload using the pinned tokenizer, without model weights."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from gliclass.pipeline import UniEncoderZeroShotClassificationPipeline
from sie_server.adapters.gliclass import GLiClassAdapter
from sie_server.types.inputs import Item
from transformers import AutoTokenizer

from tools.benchmarks.gliclass_mixed.evidence import atomic_json, digest
from tools.benchmarks.gliclass_mixed.workload import (
    ADAPTIVE_FLOOR,
    MODEL,
    REVISION,
    STATIC_BUDGET,
    WINDOW_TOKENS,
    fingerprint,
    templates,
)


class TokenizerOnlyModel:
    """Only the pipeline's formatting/config contract; cannot perform inference."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = SimpleNamespace(**config)
        self.device = torch.device("cpu")

    def eval(self) -> TokenizerOnlyModel:
        return self


def validate(snapshot: Path, output: Path) -> dict:
    fixtures = templates()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    config = json.loads((snapshot / "config.json").read_text())
    pipe = UniEncoderZeroShotClassificationPipeline(
        TokenizerOnlyModel(config),
        tokenizer,
        max_length=WINDOW_TOKENS,
        device="cpu",
        progress_bar=False,
    )
    adapter = GLiClassAdapter(str(snapshot), max_seq_length=WINDOW_TOKENS, cuda_graphs="off")
    adapter._attach(pipe, tokenizer)

    def no_inference(texts: list[str], labels: list, **_: Any) -> list[list[float]]:
        groups = [labels] * len(texts) if isinstance(labels[0], str) else labels
        return [[1 / len(group)] * len(group) for group in groups]

    adapter._score_rows = no_inference  # ty: ignore[invalid-assignment] -- intentional inference-free tokenizer control.
    records = {}
    for kind, fixture in fixtures.items():
        values = [Item(id=f"preflight/{kind}/{index}", text=text) for index, text in enumerate(fixture["texts"])]
        result = adapter.extract(values, labels=fixture["labels"], options=fixture["options"])
        if result.errors is not None and any(error is not None for error in result.errors):
            raise AssertionError(f"{kind} contains invalid or overflow items: {result.errors}")
        groups = fixture["options"].get("label_groups", {"labels": fixture["labels"]})
        lengths = [
            [
                len(tokenizer(pipe.prepare_input(text, labels), truncation=False)["input_ids"])
                for labels in groups.values()
            ]
            for text in fixture["texts"]
        ]
        if any(length > WINDOW_TOKENS or length <= 0 for row in lengths for length in row):
            raise AssertionError(f"{kind} has a truncated/empty inference row")
        costs = adapter.extract_item_costs(values, labels=fixture["labels"], options=fixture["options"])
        records[kind] = {
            "items": len(values),
            "chars": [len(text) for text in fixture["texts"]],
            "full_compiled_tokens_per_item_group": lengths,
            "item_costs": costs,
            "input_token_counts": result.input_token_counts,
            "errors": result.errors,
        }
    cheap_cost = 12 * len(fixtures["short"]["texts"][0])
    for kind in ("medium7", "medium8"):
        costs = records[kind]["item_costs"]
        if not all(cost > ADAPTIVE_FLOOR and cost + cheap_cost <= STATIC_BUDGET for cost in costs):
            raise AssertionError(f"{kind} does not exercise the prespecified cost-budget contrast")
    for kind in ("long7", "long8"):
        if not all(cost > STATIC_BUDGET for cost in records[kind]["item_costs"]):
            raise AssertionError(f"{kind} does not exceed the static budget")
    record = {
        "schema": 1,
        "model": MODEL,
        "revision": REVISION,
        "model_weights_loaded": False,
        "inference_performed": False,
        "fixture_sha256": fingerprint(fixtures),
        "fixtures": fixtures,
        "tokenizer_files": {
            path.name: digest(path)
            for path in sorted(snapshot.iterdir())
            if path.is_file() and path.suffix in (".json", ".model")
        },
        "adapter_source": {
            "path": inspect.getfile(GLiClassAdapter),
            "sha256": digest(Path(inspect.getfile(GLiClassAdapter))),
        },
        "pipeline_source": {
            "path": inspect.getfile(UniEncoderZeroShotClassificationPipeline),
            "sha256": digest(Path(inspect.getfile(UniEncoderZeroShotClassificationPipeline))),
        },
        "validation": records,
        "cheap_backlog_cost": cheap_cost,
        "passed": True,
    }
    atomic_json(output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = validate(args.snapshot, args.output)
    print(
        json.dumps(
            {"passed": record["passed"], "fixture_sha256": record["fixture_sha256"], "validation": record["validation"]}
        )
    )


if __name__ == "__main__":
    main()
