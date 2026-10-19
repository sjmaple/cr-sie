# Train a request router

Build a router that reads a mathematics problem and decides whether it needs
routine or advanced handling. Start with **Laya**, a model designed to choose
between options; adapt it to this task; then compare it with a larger text
embedding model and a small classifier that you train yourself.

The progression below reaches **53.65% → 57.34% → 65.00% → 67.64% → 72.03%**
accuracy on the same 479 test problems. You will train models, measure their
errors, and serve a router through SIE. This example predicts difficulty; it
does not solve the problems or measure the quality of generated answers.

## Dataset and metrics

Given a mathematics problem and its competition, predict **routine** or
**advanced** difficulty. The labels—the expected routine/advanced categories
used for training and evaluation—come from human difficulty ratings in
[Easy2Hard-Bench](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench),
using each competition's historical median rating as the dividing line. These
categories describe difficulty **within a competition**. The model sees only
the problem and competition name; it never receives the answer, solution or
rating as input.

We keep four groups of problems separate so that learning and measuring results
use different examples:

| Group | Examples | What it is used for |
| --- | ---: | --- |
| Training | 512, later expanded to 2,581 | Teach the router to recognize difficulty. The embedding experiment also uses smaller subsets. |
| Validation | 247 | Choose training settings or which saved model version to keep. |
| Calibration | 246 | Choose the score cutoff that separates routine from advanced. |
| Test | 479 | Measure the selected model after its settings are fixed. |

Older problems are used for development and newer problems for testing. Copies
and near-duplicates of a problem are kept from crossing those groups. See the
[data checks](docs/methodology.md) for the exact construction.

- **Accuracy:** correct routine/advanced decisions divided by all 479 test
  problems. This is the quality metric in the step headings.
- **Balanced accuracy:** average of the success rate on routine problems and
  the success rate on advanced problems, giving both groups equal weight.
- **Error counts:** an advanced problem labeled routine may receive too little
  computation; a routine problem labeled advanced may receive more than it
  needs. We report both mistakes separately.
- **Validation AUC:** measures how well scores rank advanced problems above
  routine problems, before choosing a cutoff. Laya training uses it to select the
  saved model; it is separate from the accuracy reported in the headings.

The scripts print accuracy as a fraction; the headings show percentages.
Training records also report elapsed time, and Cloud requests save a record of
API usage and charges. Those measurements are separate from classification quality.

The bracketed accuracies are recorded references. LoRA training was repeated
with three random seeds (89, 137 and 211), which change initialization and
training order; the headings show their average. The commands below run seed
89 once. Hardware can also affect your result. This public test set has already
been used in several experiments, so it is useful for reproducing these results
but is no longer a previously unseen final exam for new research. Model sizes
and training budgets differ; [results](docs/results.md) and
[methodology](docs/methodology.md) explain the limits of the comparison.

## What you will build

1. [Measure calibrated Laya [53.65%]](#1-measure-calibrated-laya-5365).
2. [Adapt Laya on 512 examples [57.34%]](#2-adapt-laya-on-512-examples-5734).
3. [Expand LoRA training to 2,581 examples [65.00%]](#3-expand-lora-training-to-2581-examples-6500).
4. [Train an embedding classifier on 64 examples [67.64%]](#4-train-an-embedding-classifier-on-64-examples-6764).
5. [Compare the 512-example embedding classifier [72.03%]](#5-compare-the-512-example-embedding-classifier-7203).

## Prerequisites and shared setup

Use either an **Apple Silicon Mac** or an **NVIDIA GPU**, on your own computer
or a rented GPU machine, for the Laya stages. Install Git and
[uv](https://docs.astral.sh/uv/getting-started/installation/), which installs
Python and the example's dependencies. Allow disk space for several saved model
versions, also called **checkpoints**. Choose `mps` for Apple's GPU support in
PyTorch, or `cuda` for NVIDIA's. See [Mac setup and measurements](docs/mac-training.md)
for tested hardware and software. The reference accuracies above were measured
on an NVIDIA H100.
On an M3 Max, the complete 512-example Mac run took **10.3 minutes** and reached
**57.62%** accuracy, with **9.1 GiB** peak sampled GPU allocation. That machine
had 128 GiB unified memory; smaller-memory Macs have not been validated.
The full 2,581-example Mac training run has not yet been measured.

The embedding-classifier stages run on **CPU** using previously computed Qwen
embeddings: lists of numbers that represent each problem's text. For a new
request, those numbers must be computed by SIE Cloud or your own server. A SIE
Cloud key lets you call a hosted model; you still need your own GPU machine for
the Laya training steps.

Start in the root of your cloned repository. This creates the example's Python
environment with the exact dependency versions recorded in `uv.lock`;
`--group training` includes the packages needed for LoRA training:

```sh
# Enter the example, select your GPU, and install the recorded dependencies.
cd examples/train-a-router
export ROUTER_DEVICE=mps  # Apple Silicon; use cuda for an NVIDIA GPU
uv sync --frozen --group training
```

Download the data, precomputed embeddings and recorded reference results. Each
download is checked against its recorded file hash to detect changes:

```sh
# Download and verify the shared dataset and the expanded-training study.
uv run --frozen --group training python run.py fetch
uv run --frozen --group training python run.py fetch --lora-study
```

Prepare two input folders: one for the 512-example run and one for the expanded
run. These commands copy data; they do not train a model:

```sh
# Prepare training inputs while keeping test problems out of both folders.
mkdir run-output/lora-small-inputs run-output/lora-expanded-inputs
cp evidence/train.jsonl evidence/validation.jsonl evidence/calibration.jsonl \
  evidence/questions.json run-output/lora-small-inputs/
cp evidence/validation.jsonl evidence/calibration.jsonl evidence/questions.json \
  run-output/lora-expanded-inputs/
cp run-output/lora-study/expansion/train-expanded.jsonl \
  run-output/lora-expanded-inputs/train.jsonl
```

The data and recordings total about 28 MB. Training downloads the model weights
separately. `evidence/` holds downloaded reference files; `run-output/` holds
your inputs and results. Use a new output directory when repeating a training
run, so the previous run remains intact. All code lives in this example; large
data and model files live on Hugging Face.

## 1. Measure calibrated Laya [53.65%]

**Apple or NVIDIA GPU.** Laya has about 421 million learned values, called model
weights, and is built on the ModernBERT text model. It scores the two choices defined in
`evidence/questions.json`. First, leave its weights unchanged and use the
calibration examples to choose a score cutoff. Then measure its test accuracy:

```sh
# Choose a cutoff for unchanged Laya weights, then measure test accuracy.
uv run --frozen --group training python finetune.py train \
  --data run-output/lora-small-inputs --output run-output/my-base \
  --variants base --seed 89 --device "$ROUTER_DEVICE"
uv run --frozen --group training python finetune.py evaluate \
  --evaluation evidence/test.json --questions evidence/questions.json \
  --output run-output/my-base --device "$ROUTER_DEVICE"
```

Read `run-output/my-base/evaluation-test.json`. This measures the starting
decision model with a calibrated cutoff; it performs no weight updates.

## 2. Adapt Laya on 512 examples [57.34%]

**Apple or NVIDIA GPU.** **LoRA** adapts a model by training small added weight
updates while keeping most of the original model fixed. This recipe also trains
Laya's final decision layers, which turn its text representation into option
scores. Make eight passes through the training data (eight **epochs**). The
validation examples choose the saved model; calibration chooses its cutoff.

```sh
# Adapt Laya on 512 training examples, then evaluate the selected model.
uv run --frozen --group training python finetune.py train \
  --data run-output/lora-small-inputs --output run-output/my-lora512 \
  --variants lora --recipe original --seed 89 \
  --validation-metric auc --export-dtype float32 --device "$ROUTER_DEVICE"
uv run --frozen --group training python finetune.py evaluate \
  --evaluation evidence/test.json --questions evidence/questions.json \
  --output run-output/my-lora512 --device "$ROUTER_DEVICE"
```

The recorded NVIDIA H100 run with seed 89 scored **57.41%**; the heading gives the three-seed
mean. Training combines the learned updates with Laya's original weights and
saves the result in `run-output/my-lora512/lora/`. It writes `frozen.json` to
record the selected model and cutoff before test evaluation; test results go
to `run-output/my-lora512/evaluation-test.json`. The commands export weights as
32-bit numbers (`float32`) to avoid rounding them to a lower-precision format.
Calibration checks this saved version before choosing its cutoff. The
[training guide](docs/lora-study.md) explains the settings.

## 3. Expand LoRA training to 2,581 examples [65.00%]

**Apple or NVIDIA GPU.** Keep the recipe and change the training data. The first 512 rows are
unchanged, as are validation, calibration and test. Keep these settings fixed
through the walkthrough rather than adjusting them after inspecting test errors.

```sh
# Repeat the same training recipe with 2,581 examples and measure its accuracy.
uv run --frozen --group training python finetune.py train \
  --data run-output/lora-expanded-inputs --output run-output/my-lora2581 \
  --variants lora --recipe original --seed 89 \
  --validation-metric auc --export-dtype float32 --device "$ROUTER_DEVICE"
uv run --frozen --group training python finetune.py evaluate \
  --evaluation evidence/test.json --questions evidence/questions.json \
  --output run-output/my-lora2581 --device "$ROUTER_DEVICE"
```

The recorded NVIDIA H100 run with seed 89 scored **66.60%**. Across all three seeds, mean accuracy rose
from 57.34% to 65.00%: **17.9% fewer errors**. The estimated improvement has a
95% uncertainty interval of **+3.20 to +12.04 percentage points**, calculated
by repeatedly resampling the test problems. Eight passes through more data
also means more training updates, so this improvement combines more examples
with more training work. The [training guide](docs/lora-study.md) includes the
full study and commands to inspect the recorded comparison on CPU.

## 4. Train an embedding classifier on 64 examples [67.64%]

**CPU.** Change the design. **Qwen3-Embedding-4B** converts each problem into
2,560 numbers without changing the model's weights. Train a **ridge classifier**:
a weighted sum of those numbers, with a penalty that discourages overly large
weights and helps limit overfitting. It learns just 2,561 values, including a
constant offset. Use the embeddings that were downloaded during setup:

```sh
# Fit small classifiers on the downloaded embeddings; no GPU or API call is needed.
uv run --frozen python run.py train
```

The command trains classifiers at 64, 128, 256 and 512 examples. It also prepares
a nearest-neighbor comparison, which predicts from the most similar training
problems. `run-output/policies/ridge64.json` stores the 64-example classifier's
weights and cutoff. A **policy** in these commands is this saved routing rule.
The next step evaluates it alongside the 512-example version.

## 5. Compare the 512-example embedding classifier [72.03%]

**CPU.** Measure the test accuracy of the routing rules saved in step 4:

```sh
# Evaluate the saved classifiers on the separate test set and print their scores.
uv run --frozen python run.py evaluate
```

Compare `ridge64` (**67.64%**) with `ridge512` (**72.03%**). The embedding model
stays fixed; only the small classifier changes. Both use one embedding call and
the same number of classifier weights per new request. The measured gain is
4.38 percentage points, but its uncertainty interval includes zero: this test
does not establish that the extra examples reliably improve accuracy. All
training sizes are retained in [results](docs/results.md). The 512-example ridge
classifier is the strongest measured design here.

## Quality and serving tradeoffs

Choosing a router also means trading accuracy against response time, memory and
the work needed to operate it:

| | Laya + LoRA, 2,581 examples | Qwen embeddings + classifier, 512 examples |
| --- | ---: | ---: |
| Recorded test accuracy | 65.00% (three-run mean) | 72.03% |
| Median local request time | 49 ms | 172 ms |
| Model size | 421 million weights | 4.02 billion weights + 2,561 classifier values |
| Memory for 32-bit weights alone | 1.57 GiB | 14.98 GiB |
| Where the model runs | Your own SIE server | SIE Cloud or your own server |

The timing check used the **same 32 requests**, one at a time, on an M3 Max
with 128 GiB memory, after 20 warmup requests. It includes local HTTP and the
final routing decision. Laya used the expanded seed-89 model; both approaches
matched all 32 reference decisions. These are local measurements, not Cloud
latency or a test of how many simultaneous users a server can handle. The
memory figures exclude runtime overhead.

Qwen is about **9.55× larger**, and was pretrained differently. This comparison
does not tell us how much of its accuracy advantage comes from size alone.
Computing the embedding dominates its request time: the small classifier and
its checks took less than 1 ms at the median. More classifier training examples
do not add per-request embedding calls. See [serving measurements and how to
repeat them](docs/serving.md#measure-request-latency) for the recorded timings
and commands.

## Serve your router

For Cloud embeddings, create an account and key in
[SIE Console](https://console.superlinked.com). Use the free credits shown in
your account, if available. `SIE_BASE_URL` tells the client which server to call;
`SIE_API_KEY` authenticates your requests. Setting the Cloud URL explicitly also
replaces any local server address left over from an earlier exercise:

```sh
# Send a new problem to SIE Cloud for embedding, then apply your local classifier.
export SIE_API_KEY=your-key
export SIE_BASE_URL=https://api.superlinked.com
uv run --frozen python run.py route \
  --policy run-output/policies/ridge512.json --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
```

This makes one embedding call and applies your locally trained classifier.
To serve **your trained LoRA model**, start a local SIE server and keep it
running so another program can send it requests:

```sh
# Configure SIE with your trained LoRA model and start its local server.
uv run --frozen python decision.py configure \
  --run run-output/my-lora2581 --variant lora \
  --models-dir run-output/my-router-models
SIE_IPC_SOCKET_PATH=/tmp/sie-example-my-router.sock \
uvx --from 'sie-server[local]==0.10.0' sie-server serve \
  --host 127.0.0.1 --port 8100 --models-dir run-output/my-router-models \
  --models example/laya-difficulty-lora --preload example/laya-difficulty-lora
```

After the model loads, open another terminal in this example directory:

```sh
# Route through your local Laya server; it does not need a Cloud API key.
export SIE_BASE_URL=http://127.0.0.1:8100
unset SIE_API_KEY  # Clear the Cloud credential for this local terminal session.
uv run --frozen python decision.py route \
  --run run-output/my-lora2581 --variant lora --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
```

Stop the server with Ctrl-C. [Serving instructions](docs/serving.md) also cover
running Qwen locally, checking serving consistency, and collecting fresh
embeddings for retraining. This example measures difficulty classification;
downstream answer quality and money savings require their own measurements.
