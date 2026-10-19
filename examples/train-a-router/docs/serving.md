# Cloud and local serving

**Serving** means keeping a model available so an application can send it new
requests. With the embedding approach, SIE turns the text into numbers and
your local classifier turns those numbers into a routine/advanced decision.
With Laya, SIE scores the two choices and your client applies the cutoff learned
during calibration.

The terminal settings `SIE_BASE_URL` and `SIE_API_KEY` select the server and
authenticate requests. Use `https://api.superlinked.com` with your account key
for Cloud, or a local address such as `http://127.0.0.1:8098` for your own
server. Some SDK examples leave the URL unset because Cloud is the default;
this walkthrough sets it explicitly so a previous local setting cannot change
the destination. `unset SIE_API_KEY` clears the key from that terminal session;
it does not delete or revoke your account's key.

## What each design costs to serve

| | Adapted Laya | Qwen embeddings + ridge classifier |
| --- | --- | --- |
| Large model needed for every new request | About 421 million weights | About 4.02 billion weights |
| Small client-side calculation | Compare the model's option score with a cutoff | Normalize 2,560 numbers, apply 2,561 learned values and a cutoff |
| Model weight storage at 32-bit precision | About 1.57 GiB | About 14.98 GiB |
| Retraining | GPU required for LoRA and decision-layer updates | CPU works when embeddings are already available |
| Deployment in this example | Your own SIE server with your trained weights | Hosted Qwen on SIE Cloud, or your own SIE server; classifier stays local |

The memory row counts **weights only**, not peak memory: both servers also need
space for intermediate calculations and runtime overhead. A GiB is about
1.07 billion bytes. Downloaded Qwen weights use 16-bit numbers and total about
8 GB; the example's portable local profile uses 32-bit computation on Apple
Silicon. Changing the precision is a separate deployment choice that should be
checked for accuracy.

Qwen is about **9.55 times larger** than Laya. It also has different architecture
and pretraining. This experiment compares useful systems; it does not isolate
which of those differences causes Qwen's higher accuracy. The tiny classifier
is cheap, but computing a fresh embedding still requires the large model. Using
more training examples does not enlarge the ridge classifier or add embedding
calls to a new request.

## Measure request latency

We measured both approaches on the same Apple M3 Max with 128 GiB memory,
macOS 26.2, SIE server 0.10.0 and 32-bit model computation. Each server ran
separately. A persistent SDK client sent 20 warmup requests, then the same
32 test problems one at a time. The server's default adaptive batching was
enabled; warmup allowed its initial timing calibration to complete.

| Time from SDK call to routing decision | Laya + expanded LoRA, seed 89 | Qwen4B + ridge512 |
| --- | ---: | ---: |
| Median | 48.77 ms | 171.51 ms |
| 95th percentile | 63.43 ms | 246.43 ms |
| Median local scoring work | 0.0015 ms | 0.87 ms |
| Decisions matching the frozen reference | 32/32 | 32/32 |

The **median** is the middle response time. The **95th percentile** describes
the slower end of this small sample; it is not a service guarantee. These times
include local HTTP, server processing, response decoding and the final decision.
For embeddings, local scoring includes normalization, verifying the saved
classifier and computing its score. Downloads, model loading, process startup
and initial model checks are excluded. This measures neither concurrent
throughput nor Cloud network latency, and does not repeat the 479-problem
accuracy evaluation.

The [recorded requests, environment, server logs and source files](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/12c2865c71dfc94bc9c28ff368134a8fefaced56/evaluations/2026-10-19/train-a-router-serving)
are published at a fixed revision with a file-hash manifest. All 17 files were
downloaded and verified without an account after publication.

To measure your own Laya model, first train and evaluate it, then start its
server using [the main walkthrough](../README.md#serve-your-router). In another
terminal in this example directory:

```sh
# Time your warmed local Laya router and compare its decisions with your saved evaluation.
uv run --frozen python benchmark_serving.py --approach laya \
  --url http://127.0.0.1:8100 --run run-output/my-lora2581 \
  --output run-output/laya-latency.json
```

Stop that server. Start Qwen using the next section, then run:

```sh
# Time the same problems through local Qwen and the ridge512 classifier from step 4.
uv run --frozen python benchmark_serving.py --approach embedding \
  --url http://127.0.0.1:8098 --policy run-output/policies/ridge512.json \
  --output run-output/embedding-latency.json
```

Each output file records per-request times, input and local artifact hashes, the timing
protocol and reference agreement. Use a new output filename for another run.
The script only accepts a server running on your own computer and sends no
Cloud key. It disables inference retries; a failed request leaves an incomplete
record instead of silently dropping the failure. Embedding decisions are
compared with those from the downloaded vectors, so numerical differences
between local and Cloud inference remain visible.

## Run the embedding model locally

From the example directory, start SIE with the provided pinned model profile:

```sh
# Download/load the pinned Qwen embedding model and start a local SIE server.
SIE_IPC_SOCKET_PATH=/tmp/sie-example-embedding.sock \
uvx --from 'sie-server[local]==0.10.0' sie-server serve \
  --host 127.0.0.1 --port 8098 --models-dir configs \
  --models Qwen/Qwen3-Embedding-4B --preload Qwen/Qwen3-Embedding-4B
```

Wait for the model to finish downloading and loading. In another terminal, run
the same learned policy:

```sh
# Send a problem to your local embedding server and apply the saved classifier.
export SIE_BASE_URL=http://127.0.0.1:8098
unset SIE_API_KEY  # The local server needs no Cloud key.
uv run --frozen python run.py route --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
```

SIE Cloud avoids running Qwen on your own machine. The CPU training steps can
also use the downloaded embeddings without a running server. Device and
precision can change numerical scores, so fresh local requests need their own
checks. For bulk local collection, add
`--self-hosted` to `run.py collect`; local responses need not have Cloud receipts.

## Collect embeddings and retrain

Set your Cloud key as in [serve your router](../README.md#serve-your-router),
or start a local server and set its endpoint as above. Collect development
vectors and fit the classifier before collecting the test vectors:

```sh
# Compute embeddings for development examples, then train new routing rules.
uv run --frozen python run.py collect --split development \
  --output run-output/fresh-development
uv run --frozen python run.py train --embeddings run-output/fresh-development \
  --policies run-output/fresh-policies
```

Once `run-output/fresh-policies/frozen.json` exists, collect and evaluate test
vectors. Use the **same policy directory** as the training command:

```sh
# After fixing the routing rules, compute test embeddings and measure accuracy.
uv run --frozen python run.py collect --split test \
  --output run-output/fresh-test
uv run --frozen python run.py evaluate --embeddings run-output/fresh-test \
  --policies run-output/fresh-policies
```

For local collection, add `--self-hosted` to both `collect` commands. Collection
sends batches of 16, disables automatic inference retries and saves receipts.
Use new output directories for another run. To serve the newly fitted classifier,
pass `--policy run-output/fresh-policies/ridge512.json` to `run.py route`.

## Serve the adapted decision model

For the main walkthrough's 2,581-example LoRA model, follow
[serve your trained router](lora-study.md#serve-your-trained-router).
Those commands configure SIE with `run-output/my-lora2581`, route a request,
and check 32 decisions against your own frozen evaluation. The guide also
provides a published checkpoint for trying serving without training first.

### Serve the historical 512-example checkpoints

The commands below serve the original full-fine-tuning comparison, which uses
different weights from the main LoRA stage. Download the predeclared seed-89
checkpoint or substitute your own frozen run:

```sh
# Download the full-fine-tuned comparison model and start its local SIE server.
uv run --frozen python run.py fetch --weights full
uv run --frozen python decision.py configure --run evidence/native/seed-89 \
  --variant full --models-dir run-output/decision-models
SIE_IPC_SOCKET_PATH=/tmp/sie-example-decision.sock \
uvx --from 'sie-server[local]==0.10.0' sie-server serve \
  --host 127.0.0.1 --port 8099 --models-dir run-output/decision-models \
  --models example/laya-difficulty-full --preload example/laya-difficulty-full
```

In a second terminal:

```sh
# Route a problem through the local full-fine-tuned comparison model.
export SIE_BASE_URL=http://127.0.0.1:8099
unset SIE_API_KEY  # The local server needs no Cloud key.
uv run --frozen python decision.py route --run evidence/native/seed-89 \
  --variant full --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
```

For the historical LoRA checkpoint, use `--weights lora`, `--variant lora`, and model ID
`example/laya-difficulty-lora`. The adapter is already merged into native weights.
The SDK request supplies the same typed question as training. The script converts
the frozen logit-margin cutoff to SIE's option-score scale; choosing the raw top
option would deploy a different classifier. Seed 89 is the documented serving
example by convention, not a seed selected from test performance.

Stop each local server with Ctrl-C. These custom checkpoints use self-hosted
SIE; the Cloud exercise uses the publicly served Qwen encoder.

To compare the first 32 test decisions with the recorded GPU reference, run
`uv run --frozen python check_serving.py --variant full --output
run-output/serving-full.json` while the matching server is running. Repeat with
`--variant lora` and another output file. Both checks matched 32/32 in the
recorded Apple Silicon verification. This checks serving consistency; it does
not make the native model more accurate than the embedding classifier.
