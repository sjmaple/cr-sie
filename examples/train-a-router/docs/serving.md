# Cloud and local serving

The embedding policy runs locally and calls SIE once per request. Set only
`SIE_API_KEY` to use SIE Cloud. Set `SIE_BASE_URL` to use your own SIE endpoint.
The script checks the model revision before embedding a fresh request.

## Run the embedding model locally

From the example directory, start SIE with the provided pinned model profile:

```sh
SIE_IPC_SOCKET_PATH=/tmp/sie-example-embedding.sock \
uvx --from 'sie-server[local]==0.10.0' sie-server serve \
  --host 127.0.0.1 --port 8098 --models-dir configs \
  --models Qwen/Qwen3-Embedding-4B --preload Qwen/Qwen3-Embedding-4B
```

Wait for the model to finish downloading and loading. In another terminal, run
the same learned policy:

```sh
export SIE_BASE_URL=http://127.0.0.1:8098
unset SIE_API_KEY
uv run --frozen python run.py route --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
```

Qwen4B weights are about 8 GB; this portable profile can use about 16 GB for
float32 weights plus runtime memory on Apple Silicon. Cloud inference and cached
CPU replay suit machines with less memory. Device and
precision can change numerical scores. Fresh serving observations are separate
from replay of the pinned recordings. For bulk local collection, add
`--self-hosted` to `run.py collect`; local responses need not have Cloud receipts.

## Collect embeddings and retrain

Set your Cloud key as in [serve your router](../README.md#serve-your-router),
or start a local server and set its endpoint as above. Collect development
vectors and fit the classifier before collecting the test vectors:

```sh
uv run --frozen python run.py collect --split development \
  --output run-output/fresh-development
uv run --frozen python run.py train --embeddings run-output/fresh-development \
  --policies run-output/fresh-policies
```

Once `run-output/fresh-policies/frozen.json` exists, collect and evaluate test
vectors. Use the **same policy directory** as the training command:

```sh
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
export SIE_BASE_URL=http://127.0.0.1:8099
unset SIE_API_KEY
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
