# Train a router: quickstart

Choose **Qwen3.5-4B** when it should answer correctly and **Qwen3.8-27B-FP8**
when the extra capability is worth paying for. Measure savings against always
using 27B and the **quality gap**: how many percentage points of answer accuracy
you lose. [Full walkthrough and metrics](README.md).

The GSM8K study uses **500 / 2,000 training questions**, **1,000 calibration
questions** to choose the routing cutoff, and **3,000 test questions**.
The primary embedding result is **20.9% savings / 0.47-point quality gap**.
The declared secondary setting gives **33.8% / 0.83 points**: **$6.30 instead
of $9.51 per 10,000 questions**, with about 83 fewer correct answers. Its gap
interval is 0.33–1.30 points; results on other workloads can differ.

| Step | What changes | Savings / quality gap |
| --- | --- | --- |
| [1. Always strong](README.md#baseline) | Send everything to 27B. | **0% / 0 points** |
| [2. Laya](README.md#laya) | Use a fixed decision model. | **21.6% / 1.50 points** |
| [3. LoRA](README.md#lora) | Adapt Laya on the training examples. | **16.4% / 0.80 points** |
| [4. Embeddings](README.md#embedding-router) | Train two small classifiers on fixed Qwen embeddings. | **20.9% / 0.47 points** |
| [5. Teacher labels](README.md#teacher-labels) | Compare teacher and reference labels on 1,539 matched examples. | **19.7% / 0.33 points** |

Solver costs are estimates at the recorded Cloud prices. Embedding savings
include charges shared across each collection batch; Laya savings exclude
router hosting. Cutoffs target a one-point calibration gap; test gaps can differ.

**Requirements:** Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and an **NVIDIA CUDA machine for LoRA**, locally or rented. The reference uses
a Modal H100. CPU works for the other steps with the supplied recordings.

```sh
# From the repository root, install dependencies, fetch evidence, and replay stages 1–5.
cd examples/train-a-router
uv sync --frozen --group outcomes
uv run --frozen --group outcomes python router.py fetch --study evidence
uv run --frozen --group outcomes python router.py replay --study evidence --output run-output/replay
```

The default download is about 159 MB. Read `run-output/replay/comparison.md`;
no API key or model calls are needed.

```sh
# Prepare a separate folder for your own training runs.
uv run --frozen --group outcomes python router.py prepare --study evidence --output run-output/my-router
```

```sh
# Stage 3: on your CUDA machine, adapt Laya and evaluate its saved routing rule.
uv sync --frozen --group outcomes --group training
uv run --frozen --group outcomes --group training python router.py fit-native --run run-output/my-router
uv run --frozen --group outcomes --group training python router.py evaluate --run run-output/my-router --method native --output run-output/lora.json
```

Commands default to `--study evidence --population expanded` (2,000 examples).
Use `--population original` on fit and evaluate for 500 examples.

```sh
# Stages 4–5: train from saved embeddings and teacher references, then evaluate both.
uv run --frozen --group outcomes python router.py fit-embedding --run run-output/my-router
uv run --frozen --group outcomes python router.py fit-teacher --run run-output/my-router
for method in embedding teacher; do
  uv run --frozen --group outcomes python router.py evaluate --run run-output/my-router --method "$method" --output "run-output/$method.json"
done
```

Create a [SIE Console](https://console.superlinked.com) account and API key to
use your available free credits on a new routed answer:

```sh
# Connect to SIE Cloud and answer a new question using your trained embedding router.
export SIE_BASE_URL=https://api.superlinked.com
export SIE_API_KEY='paste-your-key'
uv run --frozen --group outcomes python router.py answer --run run-output/my-router --question 'A rectangle has perimeter 30 and length 9. What is its area?' --target 1 --output run-output/new-answer
```

Use a fresh output folder for each run. [Live Laya and timing commands](docs/serving.md)
include its additional inference dependencies. The [self-hosted setup](self-host/README.md)
uses the same SDK against your own GPU server.
