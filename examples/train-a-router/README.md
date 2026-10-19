# Train a request router

Adapt a decision model with LoRA, improve it with more training examples, then
compare it with a classifier over frozen embeddings. Each step uses the same
difficulty task and evaluation metrics.

## Dataset and metrics

Given a mathematics problem and its competition, predict **routine** or
**advanced** difficulty. The labels come from human ratings in
[Easy2Hard-Bench](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench),
relative to each competition's training-period median. Inputs contain the problem
and competition name; answers, solutions and ratings are excluded.

Training uses 512 examples, expanded to 2,581 for the larger LoRA run. Separate
splits provide **247 validation**, **246 calibration**, and **479 test** problems.
Duplicate families are separated across chronological partitions.

- **Accuracy:** correct routine/advanced decisions divided by all 479 test
  problems. This is the quality metric in the step headings.
- **Balanced accuracy:** average recall for the two difficulty classes, so both
  classes receive equal weight.
- **Error counts:** advanced problems sent to routine, and routine problems sent
  to advanced. Applications may assign different costs to these errors.
- **Validation AUC:** ranks native checkpoints during training; separate
  calibration data chooses the decision cutoff before test evaluation.

The scripts print accuracy as a fraction; the headings show percentages.
Training records also report elapsed time, and Cloud collection saves usage
receipts. Those operational measurements are separate from classification quality.

The bracketed accuracies are recorded references. LoRA values average seeds
89, 137 and 211; the commands train seed 89 once, so your result is not a
three-seed mean. Hardware can also affect results. This public test has been
reused for the LoRA study and is an **exposed regression benchmark**. Model sizes
and training budgets differ between approaches; see [results](docs/results.md)
and [methodology](docs/methodology.md) for the comparisons and leakage checks.

## What you will build

1. [Measure calibrated Laya [53.65%]](#1-measure-calibrated-laya-5365).
2. [Adapt Laya on 512 examples [57.34%]](#2-adapt-laya-on-512-examples-5734).
3. [Expand LoRA training to 2,581 examples [65.00%]](#3-expand-lora-training-to-2581-examples-6500).
4. [Train an embedding classifier on 64 examples [67.64%]](#4-train-an-embedding-classifier-on-64-examples-6764).
5. [Compare the 512-example embedding classifier [72.03%]](#5-compare-the-512-example-embedding-classifier-7203).

## Prerequisites and shared setup

Use either an **Apple Silicon Mac** or a **CUDA GPU**, locally or on a GPU VM,
for the Laya stages. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, and allow
disk space for several checkpoints. Apple Silicon uses its GPU through PyTorch
MPS. See [Mac setup and measurements](docs/mac-training.md) for the tested
hardware and runtime. The reference accuracies above were measured on an H100.
On an M3 Max, the complete 512-example Mac run took **10.3 minutes** and reached
**57.62%** accuracy, with **9.1 GiB** peak sampled GPU allocation. That machine
had 128 GiB unified memory; smaller-memory Macs have not been validated.

The embedding-classifier stages run on **CPU** using cached Qwen vectors. For
fresh requests, use SIE Cloud or run the encoder on your own server. A SIE Cloud
key covers inference; it does not provide the GPU training machine.

From a clone of this repository:

```sh
cd examples/train-a-router
export ROUTER_DEVICE=mps  # Apple Silicon; use cuda for an NVIDIA GPU
uv sync --frozen --group training
uv run --frozen --group training python run.py fetch
uv run --frozen --group training python run.py fetch --lora-study
mkdir run-output/lora-small-inputs run-output/lora-expanded-inputs
cp evidence/train.jsonl evidence/validation.jsonl evidence/calibration.jsonl \
  evidence/questions.json run-output/lora-small-inputs/
cp evidence/validation.jsonl evidence/calibration.jsonl evidence/questions.json \
  run-output/lora-expanded-inputs/
cp run-output/lora-study/expansion/train-expanded.jsonl \
  run-output/lora-expanded-inputs/train.jsonl
```

Python 3.12 and dependencies are pinned. The hash-checked data and recordings
total about 28 MB; native training downloads the pinned model separately.
Both input directories contain only development data. Use new output directories
for repeated training runs. All code lives in this example; large artifacts live
on Hugging Face.

## 1. Measure calibrated Laya [53.65%]

**Apple or CUDA GPU.** Laya is a roughly 421M-parameter ModernBERT-based decision model. It
scores the two options in `evidence/questions.json`. The `base` variant leaves
its weights unchanged and fits a cutoff using calibration examples, then the
evaluation command runs fresh inference on the test set:

```sh
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

**Apple or CUDA GPU.** Train rank-16 attention adapters **and the native decision head** for
eight epochs. Validation AUC selects the checkpoint; separate calibration sets
its cutoff after FP32 export. Evaluate only after training writes `frozen.json`.

```sh
uv run --frozen --group training python finetune.py train \
  --data run-output/lora-small-inputs --output run-output/my-lora512 \
  --variants lora --recipe original --seed 89 \
  --validation-metric auc --export-dtype float32 --device "$ROUTER_DEVICE"
uv run --frozen --group training python finetune.py evaluate \
  --evaluation evidence/test.json --questions evidence/questions.json \
  --output run-output/my-lora512 --device "$ROUTER_DEVICE"
```

The recorded seed-89 run scored **57.41%**; the heading gives the three-seed
mean. Your merged model is in `run-output/my-lora512/lora/` and its measured
results are in `evaluation-test.json` beside it.

## 3. Expand LoRA training to 2,581 examples [65.00%]

**Apple or CUDA GPU.** Keep the recipe and change the training data. The first 512 rows are
unchanged, as are validation, calibration and test. Keep these settings fixed
through the walkthrough rather than adjusting them after inspecting test errors.

```sh
uv run --frozen --group training python finetune.py train \
  --data run-output/lora-expanded-inputs --output run-output/my-lora2581 \
  --variants lora --recipe original --seed 89 \
  --validation-metric auc --export-dtype float32 --device "$ROUTER_DEVICE"
uv run --frozen --group training python finetune.py evaluate \
  --evaluation evidence/test.json --questions evidence/questions.json \
  --output run-output/my-lora2581 --device "$ROUTER_DEVICE"
```

Recorded seed 89 scored **66.60%**. Across all three seeds, mean accuracy rose
from 57.34% to 65.00%: **17.9% fewer errors**, with a paired 95% gain interval of
**+3.20 to +12.04 percentage points**. Eight epochs on more data also means more
optimizer updates. The [native training guide](docs/lora-study.md) explains the
full study, export checks and CPU replay of the recorded comparison.

## 4. Train an embedding classifier on 64 examples [67.64%]

**CPU.** Change the design: keep **Qwen3-Embedding-4B** frozen and learn a ridge
classifier over its 2,560-dimensional vectors. The classifier has one weight per
dimension and a bias, with no hidden layers. Train using cached embeddings:

```sh
uv run --frozen python run.py train
```

The existing command fits the complete suite once: ridge and nearest-neighbor
controls at 64, 128, 256 and 512 training examples. Inspect
`run-output/policies/ridge64.json` for the 64-example coefficients and calibrated
cutoff. The next step evaluates this policy alongside the 512-example version.

## 5. Compare the 512-example embedding classifier [72.03%]

**CPU.** Evaluate the policies frozen in step 4:

```sh
uv run --frozen python run.py evaluate
```

Compare `ridge64` (**67.64%**) with `ridge512` (**72.03%**). The encoder stays
fixed; only the learned classifier changes. Both use one embedding call and
the same number of coefficients when serving. The 4.38-point gain has a 95%
interval that includes zero; all budgets are retained in [results](docs/results.md).
The 512-example ridge policy is the strongest measured design here. Its larger
encoder remains part of the computation required for each fresh request.

## Serve your router

For Cloud embeddings, create an account and key in
[SIE Console](https://console.superlinked.com). Use the free credits shown in
your account, if available:

```sh
export SIE_API_KEY=your-key
unset SIE_BASE_URL
uv run --frozen python run.py route \
  --policy run-output/policies/ridge512.json --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
```

This makes one embedding call and applies your locally trained classifier.
To serve **your trained LoRA model** through your own SIE server:

```sh
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
export SIE_BASE_URL=http://127.0.0.1:8100
unset SIE_API_KEY
uv run --frozen python decision.py route \
  --run run-output/my-lora2581 --variant lora --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
```

Stop the server with Ctrl-C. [Serving instructions](docs/serving.md) also cover
running Qwen locally, checking serving consistency, and collecting fresh
embeddings for retraining. This example measures difficulty classification;
downstream answer quality and money savings require their own measurements.
