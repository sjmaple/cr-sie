# Original native-model training comparison

For a compact-model training walkthrough, start with the [LoRA guide](lora-study.md).
Keeping its selected recipe fixed and expanding training from 512 to 2,581
examples improved mean accuracy from **57.34% to 65.00%** across three seeds,
with **17.9% fewer errors** on the exposed regression benchmark. The guide covers
CPU replay, training one router and local SIE serving. Qwen plus ridge remains
the recommended policy at 72.03%; see [results](results.md) for the comparison.

The commands below reproduce the original 512-example, cross-entropy-selected
comparison of full fine-tuning and LoRA. Those recipes average 57.48% and 55.74%
accuracy respectively. The follow-up also changes export precision, so its
512-example control is separate from this historical baseline.

Use a CUDA GPU and the same downloaded dataset as the CPU exercise. The
reference environment pins Laya 0.3.29, Torch 2.9.1, Transformers 4.57.6 and
PEFT 0.18.1. A CUDA installation compatible with Torch is required. The recorded
run uses one H100; CPU replay does not require these dependencies.

```sh
uv sync --frozen --group training
uv run --frozen --group training python finetune.py train \
  --data evidence --seed 89 --output run-output/seed-89
uv run --frozen --group training python finetune.py train \
  --data evidence --seed 137 --output run-output/seed-137
uv run --frozen --group training python finetune.py train \
  --data evidence --seed 211 --output run-output/seed-211
```

Each command reads only `train.jsonl`, `validation.jsonl`, `calibration.jsonl`
and `questions.json`, trains both variants, checks merge/export margin drift, and
writes `frozen.json`. On a remote training machine, transfer only those four
development inputs until all runs are frozen. Each output needs several GB of
disk space because it preserves checkpoints and intermediate best weights.

After all three runs have frozen, evaluate each:

```sh
uv run --frozen --group training python finetune.py evaluate \
  --evaluation evidence/test.json --questions evidence/questions.json \
  --output run-output/seed-89
```

Repeat for seeds 137 and 211. The evaluator verifies the frozen checkpoint,
plan and question hashes before inference. Evaluate your own outputs: fresh
training can differ across hardware. Keep every declared seed in your report.

`base-history.json`, `full-history.json` and `lora-history.json` preserve learning
curves. Compare training loss with validation cross-entropy to identify
overfitting. The selected checkpoint can precede the final epoch. Calibration
observations are separate from both training and checkpoint selection.
The cutoff is fitted on the serialized checkpoint after export, so half-precision
serialization cannot silently leave the cutoff calibrated to different weights.

To deploy an output, follow [serving](serving.md). A Console inference key covers
the embedding exercise; it does not provision a GPU training job or register
custom fine-tuned weights in SIE Cloud.
