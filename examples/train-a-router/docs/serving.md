# Run your router on new requests

A router adds work before answer generation. Whether it is worthwhile depends
on the money it saves, the quality it preserves, and the time and infrastructure
needed to make each decision. Follow the [README](../README.md) to train it;
this page covers live use and serving trade-offs.

## What runs for each request

| | Laya decision model | Qwen embeddings + two classifiers |
| --- | --- | --- |
| Router size | About 421 million parameters | About 4 billion encoder parameters, plus two tiny classifiers |
| Decision work | Two correctness questions through one shared checkpoint | One embedding call, then two weighted sums on CPU |
| What training changes | Rank-16 attention LoRA updates and the decision head | Two sets of 2,560 weights and a bias; the encoder stays fixed |
| Where inference runs | Your CPU or NVIDIA CUDA machine | Encoder in SIE Cloud or your SIE server; classifiers on your CPU |
| Training requirement | NVIDIA CUDA GPU | CPU with the downloaded embeddings |

The LoRA updates are merged into the saved checkpoint. More training examples
do not increase either router's model size or number of inference calls. Qwen
is roughly ten times larger than Laya, with different architecture and
pretraining. This comparison does not isolate parameter count as the cause
of a quality difference.

The saved cutoff chooses the answer model. An always-strong or always-cheap
policy skips router inference entirely.

## Use SIE Cloud

Run these commands from `examples/train-a-router` after preparing and training
`run-output/my-router`. Create an account and API key in the
[SIE Console](https://console.superlinked.com), then use the credits available
in your account. The embedding route runs from a laptop without a local model.

```sh
# Install the CPU dependencies and connect the SDK to your SIE Cloud account.
uv sync --frozen --group outcomes
export SIE_BASE_URL=https://api.superlinked.com
export SIE_API_KEY='paste-your-key'
```

```sh
# Embed one new arithmetic question, select an answer model, and request its answer.
uv run --frozen --group outcomes python router.py answer \
  --study evidence --run run-output/my-router --method embedding \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/cloud-answer
```

The output directory contains the decision, answer, and API receipts.
`result.json` reports the selected model and observed credits. The new answer
is not graded because you did not supply an answer key. Use a fresh output
folder for each request. `--target 1` selects the rule calibrated for a
one-point quality gap; it does not guarantee this particular answer is correct.

```sh
# Run your adapted Laya checkpoint on CPU and use SIE Cloud for the selected answer model.
uv run --frozen --group outcomes --group training python router.py answer \
  --study evidence --run run-output/my-router --method lora --device cpu \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/lora-cloud-answer
```

Use `--device cuda` for NVIDIA inference. `--method teacher` uses the classifiers
trained from teacher labels; `--method laya` uses the unchanged decision model.
Omit `--run` to use the recorded routers from the download. Recorded Laya
inference also needs the optional model files:

```sh
# Download the optional verified decision-model checkpoints for local inference.
uv run --frozen --group outcomes python router.py fetch --study evidence --weights
```

Each answer uses one initial call and, only when needed, one final-answer
formatting call. Inference is not automatically retried. A failed request
leaves a receipt to inspect before trying again. Both answer models and their
formatting calls passed separate live production checks.

## Run the same models on your own SIE server

The [self-hosted setup](../self-host/README.md) supplies this example's pinned
model configurations and a Docker Compose launcher. It requires Linux and
three NVIDIA GPUs; the validation hardware is H100 80 GB. The encoder, smaller
answer model, and stronger answer model run in separate environments because
their CUDA and SGLang requirements differ. This is the full inference setup;
LoRA training alone needs one CUDA GPU.

Once your endpoint is running, the same public `SIEClient` calls work:

```sh
# Send a routed question to the local three-model SIE deployment.
SIE_BASE_URL=http://127.0.0.1:8080 SIE_API_KEY='' \
uv run --frozen --group outcomes python router.py answer \
  --study evidence --run run-output/my-router --method embedding \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --self-hosted --output run-output/self-hosted-answer
```

The empty key is for the supplied trusted, local deployment. Use your own
endpoint and key when connecting to an authenticated deployment.
`--self-hosted` accepts responses without Cloud billing receipts; it keeps
model-revision checks and does not start the server. Your GPU hosting costs
remain separate from the report's equivalent Cloud-price estimates.

| Purpose | Model | Pinned revision |
| --- | --- | --- |
| Decision model before adaptation | `convaiinnovations/laya` | `aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8` |
| Embedding model | `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| Cheaper answer model | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Stronger answer model | `Qwen/Qwen3.8-27B-FP8` | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |

Embedding results also depend on the fixed instruction, dimensions, and
normalization in [serving.py](../serving.py). Changing the encoder or instruction
requires new embeddings, classifier training, and evaluation. Matching model
weights alone does not guarantee identical outputs across runtimes or hardware.

## Latency: compare warm decisions separately from startup

We measured the same 20 training questions, one at a time, after one separate
warmup question. No answer generation is included.

| Router and serving setup | Median | 95th percentile |
| --- | ---: | ---: |
| Unchanged Laya, resident on a Modal H100 | 14.65 ms | 18.74 ms |
| LoRA with 2,000 examples, resident on the same H100 | 15.23 ms | 16.00 ms |
| The same LoRA checkpoint, resident on a Modal CPU worker | 436.77 ms | 550.16 ms |
| SIE Cloud embedding + local classifiers, client on Modal CPU | 107.67 ms | 128.58 ms |

The local classifier work in the embedding route took **0.36 ms median**.
Most of its measured time is the embedding request and network round trip.
The 21 Cloud calls used 21 credits, or **$0.00021**. The main quality report
uses its own measured embedding charges; this small timing sample does not
replace them.

Laya's measurement includes tokenization, both correctness predictions, and
score calculation, with the model already loaded. It excludes HTTP. Cloud
measurements include network time from a Modal CPU worker. These are different
serving setups, not a comparison on equal hardware, and 20 sequential requests
do not measure concurrent throughput. The native CPU probe used eight Torch
threads and FP32 weights on a Modal worker; it is not a laptop measurement.

```sh
# Time Cloud embedding decisions on the same saved 20 questions plus one warmup.
uv run --frozen --group outcomes python benchmark_arithmetic.py \
  --inputs evidence/arithmetic/timing/inputs.json \
  --model run-output/my-router/embedding-2000.json \
  --output run-output/warm-embedding.json
```

This command makes 21 embedding calls. Add `--self-hosted` and set your endpoint
variables to measure your own server.

```sh
# On your CUDA machine, time your resident Laya router on the same questions.
uv run --frozen --group outcomes --group training python benchmark_arithmetic_native.py \
  --inputs evidence/arithmetic/timing/inputs.json \
  --native run-output/my-router/native-2000 --device cuda --variant lora \
  --output run-output/warm-laya.json
```

Use `--device cpu` to measure your CPU. The timing comparison uses a fixed zero
score threshold; it does not select or change a routing policy.

The single-question answer CLI verifies and loads a Laya checkpoint each time.
Its cold startup time is therefore larger than the warm numbers above. A
persistent application keeps the verified model loaded and reuses it. Report
checkpoint download, loading, router inference, and answer generation separately.

## Turn savings into a hosting budget

An embedding call is cheap, but the router's **choice of answer model** can
change the bill substantially. Sending more questions to 27B can increase the
total cost even when embedding costs stay almost unchanged.

The report's embedding savings include measured encoder credits. Laya's
answer-model savings leave an allowance for hosting the router: if the answer
bill falls by $20 per 10,000 requests, you can spend at most $20 on router
hosting before those savings disappear. That is an example calculation;
use the measured allowance for your chosen policy. Idle GPU time and utilization
matter. Tiny local classifier CPU work is also unpriced.

Training, teacher labeling, and collecting evaluation answers are separate
one-time costs. The [methodology](methodology.md) explains which charges were
observed and which are estimates from recorded token counts.
