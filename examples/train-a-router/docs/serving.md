# Run a trained router on new requests

The router predicts whether paying for the stronger answer model is worthwhile,
then calls either Qwen3.5-4B or Qwen3.8-27B-FP8. Its job is to reduce the answer
bill while keeping answer quality close to always using the stronger model.
Follow the [README](../README.md) to prepare and train the models; this page
covers using them and understanding their serving costs. The
[methodology](methodology.md) explains the labels, cost estimates and evaluation.

## What runs for each request

| | Unchanged Laya or Laya with LoRA | Qwen embeddings + two classifiers |
| --- | --- | --- |
| Main router model | About 421 million parameters | About 4 billion parameters |
| Work before choosing an answer model | Two correctness questions through one shared Laya checkpoint | One embedding call, then two small CPU calculations |
| What is learned | LoRA updates and the decision head; unchanged Laya keeps its published weights | Two logistic classifiers, each with 2,560 weights and a bias; the encoder stays fixed |
| Live routing | Local Python inference on CPU or NVIDIA CUDA | SIE Cloud or a self-hosted encoder; classifiers run locally on CPU |
| Training requirement | NVIDIA CUDA GPU for LoRA; a rented GPU such as Modal works | CPU when the saved embeddings are available |

Both approaches estimate the correctness of each answer model and its expected
cost. The saved cutoff selects the model to call. If calibration selected an
explicit always-strong or always-cheap policy, live routing skips the decision
model or embedding call entirely.

LoRA updates are merged into the exported Laya weights, so serving does not
require a separate adapter step. More training examples do not increase either
router's parameter count or number of model calls. Qwen is roughly ten times
larger than Laya and has different architecture and pretraining. A quality
difference between these recipes does not establish which difference caused it.

## Use SIE Cloud from your computer

Run these commands from `examples/train-a-router` after the README has created
`evidence` and trained `run-output/my-router`. The embedding route needs no
local GPU. Create an API key in the [SIE Console](https://console.superlinked.com)
and use your account's available credits. Set the URL explicitly so the
destination is clear in each terminal.

```sh
# Install the CPU dependencies and point the SDK at your SIE Cloud account.
uv sync --frozen --group outcomes
export SIE_BASE_URL=https://api.superlinked.com
export SIE_API_KEY='paste-your-key'
```

```sh
# Embed a new question, apply your two classifiers, and call the selected answer model.
uv run --frozen --group outcomes python router.py answer \
  --study evidence --run run-output/my-router --population expanded \
  --method embedding --contest AMC10 \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/cloud-answer
```

`--target 1` chooses the saved rule whose calibration quality loss target was
one percentage point. It does not guarantee the correctness of this answer or
the same quality gap on future traffic. Use a new output directory for each
request. The directory saves the routing decision, answer and API receipts;
`result.json` reports the selected model and observed API credits. A new
question is not graded because no reference answer was supplied.

To use your adapted decision model, keep the same Cloud connection for the
answer models and run Laya locally:

```sh
# Load your trained Laya weights on CPU, choose a model, and request its answer from Cloud.
uv run --frozen --group outcomes --group training python router.py answer \
  --study evidence --run run-output/my-router --population expanded \
  --method lora --device cpu --contest AMC10 \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/lora-cloud-answer
```

Use `--method laya` for the unchanged decision model, or `--device cuda` for
local NVIDIA inference. CPU inference does not remove the GPU requirement for
the earlier LoRA training step. Omitting `--run` selects the recorded routers
from the evidence bundle; live Laya then also needs the optional checkpoint
files downloaded by `router.py fetch --weights`.

Each answer uses one initial generation call, plus at most one final-answer
formatting call if the initial response is truncated or lacks a final answer.
There are no automatic inference retries. Failures preserve an incomplete
receipt for inspection before another attempt.

## Connect to an existing self-hosted SIE endpoint

The same public `SIEClient` interface supports your own endpoint. For the
embedding route, it must expose the encoder and both answer models listed
below. A local Laya route only needs the two answer models at that endpoint.

```sh
# Use an already running local SIE endpoint with the required pinned models.
SIE_BASE_URL=http://127.0.0.1:8080 SIE_API_KEY='' \
uv run --frozen --group outcomes python router.py answer \
  --study evidence --run run-output/my-router --population expanded \
  --method embedding --contest AMC10 \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --self-hosted --output run-output/self-hosted-answer
```

Replace the URL and supply your deployment's key if it requires authentication.
The empty key above is only for a local endpoint configured without API-key
authentication. `--self-hosted` accepts responses without Cloud charge receipts;
it keeps the model-revision checks and does not launch a server. Local GPU
costs remain separate from API credits.

### Model weights and execution environments

The live commands check the model identity and immutable Hugging Face revision:

| Purpose | Model | Pinned revision |
| --- | --- | --- |
| Decision model before adaptation | `convaiinnovations/laya` | `aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8` |
| Embedding model | `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| Cheaper answer model | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Stronger answer model | `Qwen/Qwen3.8-27B-FP8` | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |

Laya's adapted checkpoint is checked against its saved file hashes. Embeddings
also depend on the fixed instruction in [serving.py](../serving.py), vector
dimension and normalization. Changing that instruction requires regenerating
features and reevaluating the router.

Matching model weights is only part of reproducing execution. The recorded
answer collection used SGLang 0.5.10.post1 with CUDA 12 for the 4B model and
SGLang 0.5.20 with CUDA 13 for the 27B model. The encoder used SGLang's embedding
runtime. Those environments need isolation. The existing PyTorch embedding
profile in `configs/` has not been checked for numerical equivalence with the
recorded SGLang vectors. Hardware, precision and runtime changes can affect
outputs even with identical weights.

The public [SIE Helm deployment](../../../deploy/helm/sie-cluster/README.md)
supports a gateway with separate GPU worker groups for these runtimes. It is
the supported advanced deployment path. This example does not yet have a
tested compact launcher for all three models, or a tested three-model Helm
overlay. Do not assume one default server process can reproduce their distinct
runtime configurations. The recorded prompts, token limits and conditional
final-answer protocol are defined in [solver.py](../solver.py).

## Measure warm inference separately from startup

The single-question CLI verifies the selected checkpoint and loads Laya anew
on each invocation. Its saved `local_model_load_and_score_seconds` includes
model loading; `local_router_end_to_end_seconds` also includes metadata and
checkpoint checks. Neither is a persistent server's inference latency.

A persistent application keeps verified weights and its SDK client loaded,
then reuses them for new requests. Compare routers after warmup on the same
questions and hardware, with the same concurrency. Report router time
separately from answer generation, and report both the median and the 95th
percentile, which describes the slower end of the sample. Include network time
when measuring Cloud calls. Startup and checkpoint download time belong in a
separate measurement.

The current Cloud probe used the same 20 hash-ordered training inputs as the
native timing probe, one separate warmup, and one request at a time. Embedding
plus classification took **184 ms median** and **274 ms at the 95th percentile**.
The local scoring portion took **0.21 ms median**. The SDK ran on an Apple
Silicon laptop; all embedding-model computation ran in SIE Cloud. Network time
is included. No answer model was called by this probe.

The 20 timed calls used **23 credits**, or **$0.00023**. That scales to about
**$0.115 per 10,000 embedding calls for this small sample**. The main quality
report uses its own measured embedding charges, collected in batches of 16;
the timing probe does not replace those charges.

```sh
# Time 20 Cloud embedding requests and their CPU classifier calculations after one warmup.
uv run --frozen --group outcomes python benchmark_embedding_outcomes.py \
  --run run-output/my-router/heads/expanded --baseline evidence/outcomes/baseline \
  --output run-output/warm-embedding.json
```

Use the same SIE connection variables as the live-answer commands. This probe
makes 21 embedding calls, so it uses API credits. For an existing self-hosted
endpoint, add `--self-hosted` and set its URL and key. A policy that always
chooses one answer model would skip this work in deployment.

Native inference measurements are still pending. Cloud timings and direct
Laya timings on a known GPU describe different serving setups; they are not a
controlled comparison of equal hardware. The example's live CLI does not
implement a persistent Laya service. This small sequential probe also does
not measure concurrent throughput.

## Compare savings after serving costs

The reports express dollars per **10,000 requests** alongside the answer-quality
gap from always using the stronger model. The main evaluation uses observed
Cloud answer charges, including conditional final-answer calls. Embedding
routes also include their observed encoder API charges; the small local CPU
calculation remains unpriced.

For Laya, production hosting cost is still unpriced. Its answer-model savings
are a **hosting allowance**: the maximum additional router spend before those
savings disappear. For example, saving $20 in answer charges per 10,000 requests
leaves at most $20 to run the router before breaking even. That is an
illustration, not a measured result. Idle GPU time and utilization matter when
turning this allowance into a deployment budget. A static policy that skips
the router has zero router inference cost.

Training and teacher labeling are separate one-time costs. Self-hosted GPU
charges also differ from the Cloud token-price estimates used for training
cost prediction and the arithmetic transfer study. Keep these acquisition
expenses, ongoing hosting expenses and per-request API charges separate when
deciding whether a router saves money on your workload.
