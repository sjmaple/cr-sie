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

## The fine-tuning comparison

| Native policy | Seed 89 | Seed 137 | Seed 211 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Calibrated base Laya | 53.65% | 53.65% | 53.65% | 53.65% |
| Full fine-tuning | 57.20% | 58.04% | 57.20% | **57.48%** |
| Encoder LoRA + trained head | 56.99% | 53.03% | 57.20% | **55.74%** |

**This recipe does not establish a LoRA advantage or a competitive compact
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

## Reproduce and use

`run.py replay` retrains the embedding policies and recomputes this report from
verified public recordings. `finetune.py` repeats the optional native experiment.
The main README shows fresh Cloud and local SIE inference. Collected Qwen4B
embeddings used **883 credits for 1,005 development problems** and **418 credits
for 479 test problems**, in batches of 16; a fresh single-request Cloud smoke
used one credit. Cached replay uses none. These are observed embedding charges,
not downstream LLM savings or GPU training costs.

The same sample request returned routine through Cloud Qwen4B, local Qwen4B,
and both locally served native checkpoints. The first 32 test decisions also
matched the GPU reference through local SIE for each native checkpoint.
Numerical scores can differ across serving backends; the example does not
promise identical live scores or generalize this prefix check to every input.

Regenerate the comparison and overfitting figures with:

```sh
uv run --frozen --group plots python plot.py
```

Figures are written to `run-output/figures/`. Source data, all six native
checkpoints, hashes, histories, calibration observations and per-case predictions
are retained in the pinned evidence downloaded by `run.py fetch`.
