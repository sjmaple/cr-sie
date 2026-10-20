# Train a router: quickstart

Save money by sending a question to **Qwen3.5-4B** when its answer should be
good enough, and to **Qwen3.8-27B-FP8** when the extra capability is useful.
Measure **cost per 10,000 questions**, **savings versus always using 27B**, and
**answer-quality gap**: always-strong accuracy minus routed accuracy, in
percentage points. [Dataset and metrics](README.md#dataset-and-the-measures-of-success).

The reviewed competition dataset has **503 / 2,503 training questions**, **242
calibration questions** for choosing the routing cutoff, and **463 test
questions**. A 1-point calibration target is not a future-quality guarantee.

| Stage | What you compare | Savings / quality gap |
| --- | --- | --- |
| [1. Always strong](README.md#baseline) | Send everything to 27B. | **0% / 0 points** |
| [2. Laya](README.md#laya) | Use the unchanged decision model. | **PENDING / PENDING** |
| [3. Laya + LoRA](README.md#lora) | Train small weight updates; compare both training sizes. | **PENDING / PENDING** |
| [4. Embeddings + classifiers](README.md#embedding-router) | Predict both models' correctness from Qwen4B embeddings. | **PENDING / PENDING** |
| [5. Teacher labels](README.md#teacher-labels) | Fit the same classifiers using recorded teacher references. | **PENDING / PENDING** |

**Draft: result values and the final Hugging Face download pin are pending.**
Embedding savings include embedding charges; Laya savings still need router
hosting deducted. [Serving trade-offs](README.md#serving-trade-offs-and-a-separate-transfer-check).

**Requirements:** Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
CPU works for replay and classifier fitting. **LoRA training requires an NVIDIA
CUDA machine**, local or rented, such as a Modal GPU shell; run its commands
inside that machine's checkout. [Setup](README.md#setup).

```sh
# From the repository root, install dependencies, fetch records, and replay all five stages.
cd examples/train-a-router
uv sync --frozen --group outcomes
uv run --frozen --group outcomes python router.py fetch --study evidence
uv run --frozen --group outcomes python router.py replay --study evidence --output run-output/replay
```

Read `run-output/replay/comparison.md` for stages 1–2 and the recorded progression.
To reproduce training, use a fresh output folder:

```sh
# Prepare your training folders while preserving the downloaded reference results.
uv run --frozen --group outcomes python router.py prepare --study evidence --output run-output/my-router
```

```sh
# Stage 3: on your CUDA machine, train and evaluate LoRA using the expanded training set.
uv sync --frozen --group outcomes --group training
uv run --frozen --group outcomes --group training python router.py fit-native --run run-output/my-router
uv run --frozen --group outcomes --group training python router.py evaluate --run run-output/my-router --method native --output run-output/lora.json
```

The commands default to `--study evidence --population expanded`. Add
`--population original` to both commands for the smaller training set.

```sh
# Stages 4–5: fit both classifier approaches on CPU, then save their evaluated comparisons.
uv run --frozen --group outcomes python router.py fit-embedding --run run-output/my-router
uv run --frozen --group outcomes python router.py fit-teacher --run run-output/my-router
for method in embedding teacher; do
  uv run --frozen --group outcomes python router.py evaluate --run run-output/my-router --method "$method" --output "run-output/$method.json"
done
```

These steps reuse saved embeddings and teacher labels; they make no paid calls.

Create a [SIE Console](https://console.superlinked.com) account and API key;
use the free credits available in your account for a new routed answer:

```sh
# Connect to SIE Cloud and answer one new question with your trained embedding router.
export SIE_BASE_URL=https://api.superlinked.com
export SIE_API_KEY='paste-your-key'
uv run --frozen --group outcomes python router.py answer --run run-output/my-router --contest AMC10 --question 'A rectangle has perimeter 30 and length 9. What is its area?' --target 1 --output run-output/new-answer
```

[Self-hosted endpoints and model requirements](README.md#use-your-router-with-sie-cloud-or-your-own-server)
use the same SDK. Full explanations, source-review details, and the separate
2,000-question arithmetic transfer check are in the [README](README.md).
