# Request-routing workshop: static policies and learning from feedback

Follow the [one-page walkthrough](../WORKSHOP.md). The required lab is a complete
CPU exercise: verified data download → train → freeze → evaluate → route. A
Console key or a local SIE server adds fresh inference. There is no dependency
on sie-web, sie-internal, private packages, or instructor credentials.

## Course structure

| Stage | Students do | What changes |
| --- | --- | --- |
| Define the task | Inspect routine/advanced math problems and independent labels | Explicit routing objective and a fixed test split |
| Establish a baseline | Replay a 64-example cosine kNN router | Nothing; this is the static policy |
| Learn a better boundary | Fit ridge on the same 64 examples | 2,560 weights, bias and cutoff |
| Add reviewed feedback | Repeat with 128, 256 and 512 nested training labels | More training evidence; same frozen Qwen4B encoder |
| Deploy a policy | Route recorded text, then a new request through Cloud or local SIE | One embedding call plus local arithmetic |
| Compare alternatives | Replay Qwen4B reranking and GEPA; inspect decision-model post-training | Prompt learning versus weight learning |
| Economic capstone | Train from blinded teacher reviews; score real paired model answers | Learn the value of using the larger model and measure its cost–quality tradeoff |

Suggested timing: 5 minutes for the task, 10 for baseline and learning, 10 for
feedback curves, 10 for deployment, and 10–20 for comparison/discussion. GPU
post-training is an extension; participants can replay its evidence.

## Static versus improving

| Approach | Static mode | Improving mode |
| --- | --- | --- |
| Embedding neighbors | Fixed example bank and cutoff | Add reviewed labeled examples and recalibrate |
| Embedding classifier | Fixed encoder, learned weights and cutoff | Refit the small classifier as labels arrive |
| Relevance reranker | Fixed class descriptions and cutoff | Recalibrate, or use GEPA to evolve descriptions |
| Generative rubric | Fixed Qwen27B rubric | GEPA proposes rubric updates from training errors |
| Native decision model | Fixed Laya checkpoint and question | Supervised full fine-tuning or encoder LoRA plus head training |
| Teacher-supervised cost router | Frozen answering models, classifier and cutoff | Collect paired answers, review them without answer keys, refit and validate |

Improvement is a deployment decision, not an automatic property of retraining.
The **demonstrated gain** is embedding learning: 71.68% → 89.65% test accuracy,
145 → 53 errors; the same ridge learner improves 79.49% → 89.65% as feedback grows.
GEPA did not improve held-out performance here. See [all results](./DIFFICULTY_POLICY.md)
and the [post-training experiment](./POSTTRAIN.md).

## Run your own SIE server

The supplied local profile uses the same pinned Qwen4B checkpoint, last-token
pooling, normalization and instruction as the Cloud experiment. It uses SIE's
PyTorch adapter for CPU, Apple Silicon or CUDA; it does not require SGLang.
The 4B model is a substantial download and needs much more memory than the
CPU replay. Use replay or Cloud on machines that cannot hold the encoder.

In a separate terminal, from this directory:

```sh
HF_HOME="$HOME/.cache/huggingface" SIE_IPC_SOCKET_PATH=/tmp/sie-routing-workshop.sock uvx --from 'sie-server[local]==0.10.0' sie-server serve --host 127.0.0.1 --port 8098 --models-dir local-models --models Qwen/Qwen3-Embedding-4B --preload Qwen/Qwen3-Embedding-4B
```

Then in the workshop terminal:

```sh
export SIE_BASE_URL=http://127.0.0.1:8098
unset SIE_API_KEY
uv run --frozen python workshop.py route --self-hosted --policy run-output/policy512.json --request 'Find the number of integer solutions to x^2 + y^2 = 2025.'
```

An authenticated remote SIE server can use its own key. Stop a locally started
server with Ctrl-C. Return to Cloud with `unset SIE_BASE_URL` and your Console
key. The CLI checks model revision and vector dimensions before applying the
policy. Different hardware/precision can produce small numerical differences;
verify predictions before replacing an embedding backend in deployment.

## Fresh data collection and optional GEPA

The published vectors let everyone train without collection charges. To recreate
the development embeddings through your SIE Cloud account, download evidence,
set `SIE_API_KEY`, and choose a new run directory:

```sh
uv run --frozen python difficulty_policy.py embed --data evidence/difficulty-policy/main/data.json --run run-output/fresh-study
```

This collects all 640 development requests in 40 batches, then fits every
predeclared embedding policy. The original collection used 510 credits. Full
1,152-case vector collection used 911 credits; these are observed receipts, not
a free-tier allowance or future tariff. Failed calls are recorded, retries are
off, and a failed run is not silently resumed.

GEPA 0.1.4 uses SIE for both Qwen4B reranking and Qwen3.8-27B reflection:

```sh
uv sync --frozen --group gepa
uv run --frozen python difficulty_policy.py optimize --data evidence/difficulty-policy/main/data.json --run run-output/fresh-study --reflections 12
uv run --frozen python difficulty_policy.py freeze --data evidence/difficulty-policy/main/data.json --run run-output/fresh-study
uv run --frozen python difficulty_policy.py test --data evidence/difficulty-policy/main/data.json --run run-output/fresh-study
```

This is an optional, separately budgeted research run, not a required starter-credit
step. Inspect the bounded call/credit limits in `cloud.py` before starting. Keep
all prompts and checkpoints selected on development data; evaluate only after
freezing. [Recorded GEPA outcomes](./DIFFICULTY_POLICY.md#gepa-status) include both
the reranker and generative-rubric experiments. No closed-model teacher generated
the reference labels. Everything needed for generation is at most 27B.

## Dataset, reproducibility and scope

The Easy2Hard-AMC dataset is attributed to Ding et al. and licensed CC-BY-SA-4.0.
Our derived data retains that license; example code is Apache-2.0. Public evidence
includes original source files, the split, embeddings, policies, predictions and
receipts, pinned by revision and SHA-256. `workshop.py download` verifies every
required artifact. Generated data and checkpoints stay outside Git.

The core task predicts human-relative competition difficulty. The
[economic capstone](./ECONOMICS.md) separately measures downstream answers and
actual credits, including a teacher-supervised path whose fitter cannot read
the benchmark answer keys. Difficulty accuracy is never substituted for answer quality.
Public historical problems may have appeared in pretraining. The held-out results
are conditional on this dataset and these trained policies.

The [earlier dollar-cost study](./LEGACY_COST.md) and
[handler cascade](../CASCADE.md) answer different questions and are retained as
separate references. They are not mixed into this workshop's headline result.
