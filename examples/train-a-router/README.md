# Train a request router

Learn a small decision function, measure what changes as you add feedback, and
use it through SIE. This example routes mathematics problems to **routine** or
**advanced**, relative to the named competition. It predicts human-rated
difficulty; it does not choose which LLM can answer correctly.

The progression is **64-example nearest neighbors → learned linear classifier
→ 512 reviewed examples → optional fine-tuned decision model**. The encoder is
Qwen3-Embedding-4B. The GPU extension trains a smaller Laya decision model.
See [results](docs/results.md), [the evaluation protocol](docs/methodology.md),
and [how this differs from the earlier studies](docs/archive.md).

On the corrected held-out set, the main progression improves accuracy from
**59.29% to 72.03%**: **31.3% fewer routing errors**. Qwen plus ridge remains the
recommended policy. In the [LoRA follow-up](docs/lora-study.md), expanding the
same recipe from 512 to 2,581 training examples raises mean accuracy from
**57.34% to 65.00%** across three seeds: **17.9% fewer errors**. That follow-up
uses the same 479 problems as an **exposed regression benchmark**.

## Run the comparison on your laptop

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
Python 3.12 is selected automatically. From a clone of this repository:

```sh
cd examples/train-a-router
uv sync --frozen
uv run --frozen python run.py fetch
uv run --frozen python run.py replay
```

No key or GPU is needed. The 22 MB download contains pinned public data, recorded
embeddings and per-case predictions; weights are optional. The replay trains all
eight embedding policies on CPU, evaluates their frozen decisions, and verifies
the native-model metrics from recordings. It saves policies in
`run-output/policies/` and a report in `run-output/replay.json`. Use a new
`--policies` directory and `--output` file to repeat it; existing runs are preserved.

## Route a fresh request with SIE Cloud

Create an account and API key in [SIE Console](https://console.superlinked.com).
Use the free credits shown in your account, if available. Set the key in your shell, then run:

```sh
export SIE_API_KEY=your-key
unset SIE_BASE_URL
uv run --frozen python run.py route --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
```

This makes one embedding inference call, applies your local learned classifier,
and prints the route and available usage receipt. It does not generate an
answer. A [local SIE server](docs/serving.md) runs the same encoder and code.

## Repeat the learning process

```sh
uv run --frozen python run.py train --policies run-output/my-policies
uv run --frozen python run.py evaluate --policies run-output/my-policies
```

To recollect embeddings using your Cloud key, run `run.py collect --split
development --output run-output/development-embeddings`, then pass that directory
to `train --embeddings`. Collect `--split test` separately after freezing the
policies and pass its directory to `evaluate --embeddings`. Each collection is
bounded, sends batches of 16, disables automatic inference retries, and saves
usage receipts. Cached replay spends no credits.

The [LoRA guide](docs/lora-study.md) replays the improvement on CPU, serves the
downloaded router locally, and trains one router on a CUDA GPU. It also preserves
the full recipe comparison. The [original GPU lab](docs/finetuning.md) compares
full fine-tuning and LoRA with 512 examples. All scripts, dependencies and
configuration live in this folder. Large inputs and artifacts live in an immutable
Hugging Face revision. No other example or private repository is required.
