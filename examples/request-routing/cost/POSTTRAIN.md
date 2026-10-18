# Post-training a native decision model

**Completed experiment:** a 421M-parameter Laya model learns the same difficulty
routing task as the Qwen4B embedding classifier. Full fine-tuning and encoder
LoRA both improve substantially over the calibrated base model. The embedding
classifier remains more accurate on a fresh confirmation set.

| Policy | Original test: accuracy / errors | Fresh confirmation: accuracy / errors |
| --- | ---: | ---: |
| Qwen4B + static kNN64 | 71.68% / 145 | 74.22% / 132 |
| Qwen4B + ridge64 | 79.49% / 105 | 82.23% / 91 |
| Qwen4B + ridge512 | **89.65% / 53** | **91.60% / 43** |
| Calibrated base Laya | 53.71% / 237 | 53.32% / 239 |
| Laya full fine-tune | **89.65% / 53** | 87.11% / 66 |
| Laya encoder LoRA + trained head | 88.09% / 61 | 88.09% / 61 |

Each test contains 512 problems. On fresh confirmation, full tuning gains
**33.79 points** over base (paired 95% interval **30.08–37.50**) and reduces errors
**72.38%**. LoRA gains **34.77 points** (interval **31.05–38.29**) and reduces errors
**74.48%**. Full tuning was the better validation candidate; we report both
predeclared families, without selecting a winner from the test results.

The confirmation set excludes every previously used ID and normalized problem
text. It was balanced and hash-selected before training, without performance
filtering. Labels come from the same pinned Easy2Hard-AMC human-difficulty
ratings; no GLM, Haiku, Sonnet or Qwen teacher supplied them. We ran one training
seed. Paired intervals describe test-case uncertainty, not training-run variance.

[Pinned public evidence and checkpoints](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/85c9d93a73d5559a6c69a22145f5f86f6b450fbc/evaluations/2026-10-18/request-routing-course-posttrain-v2) ·
[Results figure](https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/85c9d93a73d5559a6c69a22145f5f86f6b450fbc/evaluations/2026-10-18/request-routing-course-posttrain-v2/course-results.png).
All 103 files (including both checkpoints) passed anonymous SHA-256 verification.

## Two different classifier architectures

The embedding path is:

`text → frozen Qwen3-Embedding-4B → L2-normalized 2,560-vector → ridge score → cutoff`

Only 2,560 weights and one bias learn. Regularization and the cutoff are selected
from training leave-one-out predictions. There are no hidden layers, sigmoid,
or encoder updates. A score is not a calibrated probability.

[Laya](https://huggingface.co/convaiinnovations/laya) is ModernBERT-large plus a
native decision head: two additional transformer layers score option markers.
The complete checkpoint has **421,293,827 parameters**. It reads the problem and
a typed two-option question in one forward pass and does not generate text.

Full tuning updates the complete checkpoint. Our LoRA variant trains rank-16
adapters on encoder attention `Wqkv` and `Wo`, plus the complete decision head:
**29,264,643 trainable parameters**, including 2,752,512 encoder adapter parameters.
We merge the adapters into an ordinary Laya checkpoint. This is not a claim that
SIE's cross-encoder runtime LoRA contract supports replacement decision heads.

## Recipe and selection

Both variants use the same 512 training and 128 validation examples, seed 89,
eight epochs, supervised cross-entropy, shuffled option order, effective batch
16 (microbatch 8 × accumulation 2), 512-token context and 256-token question budget.
Full encoder/head learning rates are 2.5e-5/1e-4; LoRA uses 2e-4/1e-4.
Regularization is AdamW weight decay 0.01 with a cosine learning-rate schedule.
Validation selects epoch and logit-margin cutoff, with ties retaining the earlier
epoch. Full tuning selects epoch 2; LoRA selects epoch 6. Test inputs were sent
to a separate GPU job only after all checkpoint hashes and cutoffs were frozen.

Measured training/selection/export sections on one H100 took **97.7 seconds**
(full) and **116.5 seconds** (LoRA). Those durations exclude dependency installation,
initial model loading and GPU allocation. They do not show a LoRA training speedup.
Base checkpoint: `convaiinnovations/laya` at
`aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8`; Laya 0.3.29, Torch 2.9.1,
Transformers 4.57.6 and PEFT 0.18.1. All are pinned in the optional dependency group.

## Replay on any laptop

From this directory, after the main workshop evidence download:

```sh
uv run --frozen python posttrain_evidence.py
uv run --frozen python replay_posttrain.py
```

This downloads the small evidence subset and reconstructs both test comparisons,
including Qwen's fresh confirmation predictions, without a GPU or API key.
The bundle includes the original plan, validation histories, frozen choices,
per-request scores, paired comparisons and serving receipts. Checkpoint weights
are optional downloads; each native checkpoint is about 846 MB with its tokenizer.
Data adaptations are CC-BY-SA-4.0; code and Laya weights retain Apache-2.0 with
upstream attribution. The source dataset and original experiment are linked in
[DIFFICULTY_POLICY.md](./DIFFICULTY_POLICY.md).

## Repeat the training on your GPU

```sh
uv sync --frozen --group posttrain
uv run --frozen --group posttrain python posttrain.py train --data evidence/posttrain --output run-output/my-posttrain --device cuda
uv run --frozen --group posttrain python posttrain.py evaluate --evaluation evidence/posttrain/confirmation.json --questions evidence/posttrain/questions.json --output run-output/my-posttrain --device cuda
uv run --frozen --group posttrain python posttrain.py evaluate --evaluation evidence/posttrain/reference.json --questions evidence/posttrain/questions.json --output run-output/my-posttrain --device cuda
```

The training command reads only `train.jsonl` and `validation.jsonl`; it writes
`frozen.json` after native export and merge parity checks. Evaluation refuses a
changed checkpoint. For a remote worker, transfer only those development files
and `questions.json` before training, then transfer evaluation inputs after freeze.
Fresh training may differ across hardware; evaluate your own frozen outputs.

## Serve the downloaded checkpoint through SIE

```sh
uv run --frozen python posttrain_evidence.py --weights full --output evidence/posttrain-full
uv run --frozen python decision.py configure --run evidence/posttrain-full --models-dir run-output/decision-models
HF_HOME="$HOME/.cache/huggingface" SIE_IPC_SOCKET_PATH=/tmp/sie-routing-decision.sock uvx --from 'sie-server[local]==0.10.0' sie-server serve --host 127.0.0.1 --port 8099 --models-dir run-output/decision-models --models workshop/laya-difficulty-full --preload workshop/laya-difficulty-full
```

In another terminal, from the same directory:

```sh
export SIE_BASE_URL=http://127.0.0.1:8099
unset SIE_API_KEY
uv run --frozen python decision.py route --run evidence/posttrain-full --request 'Find the number of integer solutions to x^2 + y^2 = 2025.'
```

Use `--weights lora` and `--variant lora` to try the merged LoRA checkpoint.
`decision.py` applies the frozen cutoff to the native option scores; using the
raw top option would silently deploy a different policy. Stop the server with
Ctrl-C. This exact public SIE 0.10.0 path passed on Apple Silicon: **512/512 routes
matched the GPU reference for each checkpoint**, through the HTTP API.

## Serving tradeoffs and Cloud

Native SIE CUDA inference preserves all 512 full-model routes. For LoRA, 510/512
routes match the float32 reference, with unchanged aggregate accuracy. The LoRA
merge and fp16 file export separately preserve all validation routes.

On one H100, a controlled repeat with 8 CPU threads, 20 warmup requests and 128
measured texts per model gives second-pass median direct-adapter latencies of
**11.49 ms full Laya**, **11.53 ms LoRA Laya**, and **21.03 ms Qwen4B**. First-pass
medians varied (30.50, 11.51 and 57.49 ms respectively), so report both passes;
these are warm local measurements, not an HTTP SLA or universal speedup.
Peak PyTorch allocation was **1.00 GB versus 8.30 GB**. The smaller model has a
measured memory advantage; it gives up 3.5–4.5 accuracy points on confirmation.

The core Cloud lab uses production Qwen4B embeddings and your locally learned
classifier; GEPA also ran through production SIE using Qwen3.8-27B. This course's
custom Laya checkpoints are not registered in the production Console. Training
and custom-checkpoint serving are the optional self-hosted GPU extension, with
CPU replay available to every attendee. A Console inference key is sufficient
for the main learning loop, not for launching a training job.
