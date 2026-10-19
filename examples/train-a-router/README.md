# Train a request router

Start with a pretrained decision model, adapt it to your task with LoRA, then
compare it with a frozen embedding model and a small learned classifier. This
example routes mathematics problems to **routine** or **advanced**, relative to
the named competition, using human difficulty labels from Easy2Hard-Bench.

## What you will build

1. [Measure the starting decision model](#1-measure-the-starting-decision-model).
   Calibrated **Laya: 53.65%** accuracy.
2. [Adapt Laya with LoRA](#2-adapt-laya-with-lora).
   Train on **512 examples: 57.34%** mean accuracy.
3. [Improve the decision model with more examples](#3-improve-the-decision-model-with-more-examples).
   Keep the recipe and grow to **2,581 examples: 65.00%** mean accuracy.
4. [Train a classifier on frozen embeddings](#4-train-a-classifier-on-frozen-embeddings).
   Qwen3-Embedding-4B plus ridge, **64 examples: 67.64%** accuracy.
5. [Improve the embedding classifier with more examples](#5-improve-the-embedding-classifier-with-more-examples).
   Grow to **512 examples: 72.03%** accuracy, the strongest result here.
6. [Route fresh requests through SIE](#6-route-fresh-requests-through-sie).
   Use Cloud embeddings or a local SIE server with the same learned policy.

LoRA is part of the main walkthrough. **A laptop can replay every stage without
a key or GPU and train the embedding classifiers on CPU.** To repeat the native
LoRA training itself, follow the CUDA commands in the linked training guide.

All accuracies use the same 479 problems; LoRA results average three seeds. This
is a comparison of practical designs with different training budgets and model
sizes. Model selection and cutoff calibration use separate development splits.
The benchmark is now public and was reused for the LoRA study; it is an exposed regression
benchmark, not a new prospective test. See [results](docs/results.md) for all
runs and uncertainty.

## Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
From a clone of this repository, run:

```sh
cd examples/train-a-router
uv sync --frozen
uv run --frozen python run.py fetch
uv run --frozen python run.py fetch --lora-study
```

Python 3.12 is selected automatically. These downloads total about **28 MB** of
pinned public data, embeddings and prediction records, with hashes checked on
download. All code and configuration live in this folder; large artifacts live
on Hugging Face. Model weights are downloaded only when requested.

## 1. Measure the starting decision model

Laya is a roughly 421M-parameter ModernBERT-based decision model. It scores a
typed question with two options, without generating an answer. Inspect
`evidence/questions.json` to see the question and route definitions, then replay
the published baselines:

```sh
uv run --frozen python run.py replay \
  --policies run-output/baseline-policies --output run-output/baseline-report.json
```

Look for `base-89`: **53.65%** accuracy. This is the pretrained model with a
cutoff fitted on separate calibration examples; its weights are unchanged.
The other base rows repeat that same checkpoint. The replay also rebuilds the
embedding baselines used later, using cached vectors, and verifies the original
native results from recorded scores. It runs no encoder or Laya inference.

## 2. Adapt Laya with LoRA

Train attention adapters **and the native decision head** on 512 examples.
Choose the checkpoint on validation AUC, then fit its cutoff on the separate
calibration split. Replay the recorded training comparison:

```sh
uv run --frozen python lora_report.py \
  --study run-output/lora-study --evidence evidence \
  --output run-output/lora-report.json
```

The report checks the frozen selections and recomputes metrics from recorded
scores for every seed.
`stages.selected_auc_512` averages **57.34%**. Follow
[train on 512 examples](docs/lora-study.md#train-on-512-examples) to run this
stage yourself on CUDA. The guide includes the locked dependencies and exact
settings; replaying its records requires only CPU.

## 3. Improve the decision model with more examples

Keep that recipe and expand its training set from 512 to 2,581 examples. In the
same report, compare `selected_auc_512` with `selected_auc_2581`: mean accuracy
rises to **65.00%**, with **17.9% fewer errors**. Every declared seed improves.
The eight-epoch schedule also means more optimizer updates at the larger size.

Plot the recorded improvement alongside every training seed:

```sh
uv run --frozen --group plots python plot.py \
  --lora-study run-output/lora-study --evidence evidence \
  --output run-output/lora-figures
```

Open `run-output/lora-figures/lora-study.png` to compare the stages.
Follow [train on 2,581 examples](docs/lora-study.md#train-on-2581-examples) using
the downloaded expanded dataset. Validation, calibration and test stay fixed.
The guide also shows how to [serve the trained router locally](docs/lora-study.md#serve-the-downloaded-router-locally)
using its published checkpoint, without training it first.

## 4. Train a classifier on frozen embeddings

Now change the design: keep **Qwen3-Embedding-4B** frozen and learn a ridge
classifier over its vectors. The classifier has 2,560 weights and a bias, with
no hidden layers. Fit it locally from the cached embeddings:

```sh
uv run --frozen python run.py train
```

This trains and freezes the complete declared suite at 64, 128, 256 and 512
examples, including nearest-neighbor controls. Inspect
`run-output/policies/ridge64.json`: it contains the coefficients and calibrated
cutoff. With 64 training examples, this design reaches **67.64%** accuracy.
Its encoder is larger than Laya; the small classifier alone is not the whole
inference cost.

## 5. Improve the embedding classifier with more examples

Evaluate the frozen suite, including the version trained on 512 examples:

```sh
uv run --frozen python run.py evaluate
```

Compare `ridge64` (**67.64%**) with `ridge512` (**72.03%**). Only the local
classifier changes; the encoder stays frozen. At serving time both use one
embedding call and the same number of classifier coefficients. Adding training
examples does not add inference calls. The 4.38-point gain is a point estimate;
its 95% interval includes zero. All intermediate budgets are in [results](docs/results.md).

To repeat learning with fresh Cloud or local embeddings, follow
[collect embeddings and retrain](docs/serving.md#collect-embeddings-and-retrain).
It keeps development and test collection separate and saves usage receipts.
Cached replay and CPU classifier training spend no credits. Training requires
new policy directories; use new output paths when repeating runs.

## 6. Route fresh requests through SIE

Create an account and API key in [SIE Console](https://console.superlinked.com).
Use the free credits shown in your account, if available. Set the key in your
shell, then use the 512-example classifier:

```sh
export SIE_API_KEY=your-key
unset SIE_BASE_URL
uv run --frozen python run.py route \
  --policy run-output/policies/ridge512.json --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
```

The script makes one embedding call, applies your local classifier, and prints
the route and available usage receipt. For the same flow on your own computer,
follow [local SIE serving](docs/serving.md). Custom LoRA checkpoints also run on
your own SIE server; a Cloud inference key does not provision training.

You now have two update paths: retrain Laya adapters or retrain a small classifier
on frozen embeddings. Both require reviewed examples, development selection and
a frozen evaluation. This task measures difficulty classification; downstream
answer quality and monetary savings need their own measurements. See
[methodology](docs/methodology.md) for the data and leakage checks, and
[earlier assets](docs/archive.md) for the separate original routing examples.
