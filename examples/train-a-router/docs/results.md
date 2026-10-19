# Results on the corrected evaluation

[Pinned data, predictions and checkpoints](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/4136e68feaa90b5f50dce9f30df212982d393356/evaluations/2026-10-18/train-a-router-grouped-v2)
· [Comparison figure](https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/4136e68feaa90b5f50dce9f30df212982d393356/evaluations/2026-10-18/train-a-router-grouped-v2/figures/comparison.png)
· [Learning curves](https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/4136e68feaa90b5f50dce9f30df212982d393356/evaluations/2026-10-18/train-a-router-grouped-v2/figures/learning-curves.png).

**The frozen Qwen4B encoder plus a learned linear classifier is the main result.**
On 479 chronological, duplicate-family-isolated test problems, kNN with 64
training examples reaches **59.29%** accuracy; ridge with 512 reaches **72.03%**.
That is **+12.73 percentage points**, paired 95% interval **+7.10 to +18.37**,
and **31.3% fewer errors** (195 → 134). The descriptive competition-year
bootstrap gives +7.80 to +18.89 points over 11 groups.

| Training examples | kNN accuracy | Ridge accuracy |
| ---: | ---: | ---: |
| 64 | 59.29% | 67.64% |
| 128 | 57.20% | 67.22% |
| 256 | 59.29% | 71.40% |
| 512 | 65.97% | 72.03% |

With the same 64 labels, learning the linear classifier gains **8.35 points**
(95% interval +2.51 to +13.99). Adding labels to ridge gives a further **4.38
points**, with an interval of **−0.84 to +9.60**: the point estimate improves,
but this smaller increment alone is uncertain. More feedback is not guaranteed
to improve every intermediate version. With the same 512 labels, ridge beats
kNN by 6.05 points (+0.84 to +11.27).

Ridge512 balanced accuracy is 71.56%. Of 479 problems it incorrectly routes 85
advanced problems to routine and 49 routine problems to advanced. These errors
may have different application costs; the benchmark uses a symmetric decision
criterion. Relative to kNN64, false-routine errors rise from 67 to 85 while
false-advanced errors fall from 128 to 49. A risk-sensitive routing objective
would need a different development criterion and its own evaluation.
A training-only contest-name majority control has 47.39% accuracy
and 50% balanced accuracy, so contest identity alone no longer explains the task.

## Original fine-tuning comparison: 512 training examples

| Native policy | Seed 89 | Seed 137 | Seed 211 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Calibrated base Laya | 53.65% | 53.65% | 53.65% | 53.65% |
| Full fine-tuning | 57.20% | 58.04% | 57.20% | **57.48%** |
| Encoder LoRA + trained head | 56.99% | 53.03% | 57.20% | **55.74%** |

**The original recipe does not establish a LoRA advantage or a competitive compact
replacement for the embedding classifier.** Full tuning gains 3.83 points over
base on average and remains 14.54 points behind ridge512. LoRA gains 2.09 points
over base on average and remains 16.28 points behind ridge512. The base values
repeat the same deterministic checkpoint; they are not three independently
trained baselines.

Full tuning selects epoch 2 for all three seeds. LoRA selects epochs 2, 1 and 1.
Later epochs reduce training loss while substantially increasing validation
cross-entropy. The selected checkpoint is restored; final-epoch weights are not
used. Calibration is separate and uses the saved checkpoint after serialization.
All three per-seed paired LoRA-minus-full case intervals include zero; seed
variation and the small test population do not establish a universal ranking.

These numbers replace the earlier near-90% global-difficulty results, which had
duplicate-family overlap and a label strongly confounded with contest family.
The target and partitions changed, so the old and new accuracies are not an
apples-to-apples before/after comparison. The [archive](archive.md) preserves the
old evidence; [methodology](methodology.md) gives the corrected protocol.

## LoRA follow-up: add examples to the same recipe

[Pinned follow-up evidence and checkpoints](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/d884d137386ccc4c89eb6d78449197a4e4be9892/evaluations/2026-10-18/train-a-router-lora-followup)
· [Comparison figure](https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/d884d137386ccc4c89eb6d78449197a4e4be9892/evaluations/2026-10-18/train-a-router-lora-followup/figures/lora-study.png).

Expanding training from **512 to 2,581 examples** improves the native router from
**57.34% to 65.00%** mean accuracy across three seeds: **+7.65 percentage points**
and **17.9% fewer errors**. The paired problem-bootstrap 95% interval is
**+3.20 to +12.04 points**; the competition-year interval is +2.66 to +11.70.
This follow-up uses the same 479 problems as an **exposed regression benchmark**,
so it measures progress on that benchmark rather than a new prospective test.

| Training examples, same LoRA recipe | Seed 89 | Seed 137 | Seed 211 | Mean |
| --- | ---: | ---: | ---: | ---: |
| 512, AUC selection, FP32 export | 57.41% | 57.83% | 56.78% | **57.34%** |
| 2,581, AUC selection, FP32 export | 66.60% | 62.42% | 65.97% | **65.00%** |

Six declared recipes were screened on validation AUC across all three seeds.
The original rank-16 attention-adapter recipe with a trained native head won;
the alternatives did not improve mean validation AUC. That recipe was frozen
before the expanded training runs. Epoch selection uses 2018 validation data,
and each cutoff uses separate 2019 calibration data. Seed 89 was fixed as the
serving example before test evaluation; it was not chosen for its final accuracy.
The unchanged eight-epoch schedule gives the larger dataset more optimizer
updates, so this improvement includes additional training compute.

The historical LoRA baseline above scores 55.74%. Its new 512-example control
scores 57.34%, but changes both checkpoint selection from cross-entropy to AUC
and export precision from FP16 to FP32. Evaluation also switches to eager
execution after checking development-score parity. The +1.60-point difference
cannot be attributed to AUC selection alone; its paired 95% interval includes
zero (−1.32 to +4.59). The 512-to-2,581 comparison keeps these settings fixed.
Follow-up intervals average seed correctness before resampling problems or
competition-year groups and condition on these three trained seeds.

**Qwen plus ridge remains the recommended policy at 72.03%**, 7.03 points above
the expanded native model, using 512 training examples. The follow-up provides
a working compact-model training extension and a measurable improvement within
that approach. Full fine-tuning was only run on 512 examples, so the expanded
LoRA result does not establish a LoRA advantage at matched training sizes.
Difficulty accuracy does not establish downstream answer quality, latency or
money savings. Follow the [LoRA guide](lora-study.md) to replay the comparison,
train one router, or reproduce the complete study.

## Reproduce and use

`run.py replay` retrains the embedding policies and recomputes the original
comparison from verified public recordings. `lora_report.py` separately verifies
the follow-up's frozen selections, runtime-parity receipts and per-case results;
the [LoRA guide](lora-study.md) gives its download and replay commands.
`finetune.py` repeats the native training experiments.
The main README shows fresh Cloud and local SIE inference. Collected Qwen4B
embeddings used **883 credits for 1,005 development problems** and **418 credits
for 479 test problems**, in batches of 16; a fresh single-request Cloud smoke
used one credit. Cached replay uses none. These are observed embedding charges,
not downstream LLM savings or GPU training costs.

The same sample request returned routine through Cloud Qwen4B, local Qwen4B,
and both original native checkpoints served locally. The first 32 test decisions
also matched the GPU reference through local SIE for each original checkpoint.
The expanded seed-89 LoRA checkpoint separately matched **32/32** reference
decisions through public SIE 0.10.0 on Apple Silicon with FP32 computation.
Numerical scores can differ across serving backends; the example does not
promise identical live scores or generalize this prefix check to every input.

Regenerate the comparison and overfitting figures with:

```sh
uv run --frozen --group plots python plot.py
```

Figures are written to `run-output/figures/`. Source data, all six native
checkpoints, hashes, histories, calibration observations and per-case predictions
are retained in the pinned evidence downloaded by `run.py fetch`.
