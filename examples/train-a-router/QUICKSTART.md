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

| Step | What you learn | Savings / quality gap |
| --- | --- | --- |
| [1. Baselines](README.md#baseline) | Compare always-strong with always-cheap to see the opportunity. | **0% / 0 points** for always-strong |
| [2. Laya baseline](README.md#laya) | Compare its replayed cost and quality gap with your goal. | **21.6% / 1.50 points** |
| [3. LoRA](README.md#lora) | Learn from answer outcomes to preserve more quality. | **16.4% / 0.80 points** |
| [4. Embeddings](README.md#embedding-router) | Improve measured cost and quality with small classifiers trained on CPU. | **20.9% / 0.47 points** |
| [5. Without training answer keys](README.md#teacher-labels) | Use a teacher; compare label sources on 1,539 matched questions. | **19.7% / 0.33 points** |
| [6. Serve your choice](README.md#serve) | Choose a cost/quality setting and route a new request. | **20.9–33.8% / 0.47–0.83 points** for embeddings |

LoRA first improves quality; embeddings then improve both measured quality and
cost. Teacher supervision is an alternative when training answer keys are missing.

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
# Stage 4: train classifiers from saved embeddings and evaluate the routing rule.
uv run --frozen --group outcomes python router.py fit-embedding --run run-output/my-router
uv run --frozen --group outcomes python router.py evaluate --run run-output/my-router --method embedding --output run-output/embedding.json
```

Stage 5 explores the alternative when training answer keys are missing.
It reuses recorded teacher references and compares them with reference labels
on the same questions. If using your reference-label router, continue to stage 6.

```sh
# Stage 5: compare teacher-generated training labels with the matched reference-label control.
uv run --frozen --group outcomes python router.py fit-teacher --run run-output/my-router
uv run --frozen --group outcomes python router.py evaluate --run run-output/my-router --method teacher --output run-output/teacher.json
```

For stage 6, choose the primary `--target 1` (**$7.52 per 10,000 questions**, about
47 fewer correct answers than always-strong) or the secondary `--target 2`
(**$6.30**, about 83 fewer). Create a [SIE Console](https://console.superlinked.com) account and API key to
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

Finish with a saved router, your chosen cost/quality setting, and a receipt
showing which answer model was selected and what the new request cost.

Optional: [replay the same routing decisions with Sol and Luna](docs/model-pair-transfer.md)
to see how another answer-model pair changes the cost/quality tradeoff.
