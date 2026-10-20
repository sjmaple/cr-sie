# What other Laya adaptations tell us

Laya can learn task-specific decisions through post-training. The useful
question for this example is whether that adaptation improves the money and
answer-quality tradeoff on new requests. Results on a different benchmark
provide ideas to test; they do not predict our router's improvement.

The following are the authors' reported results, checked on 2026-10-19.
We did not reproduce these outside experiments.

| Example | What was adapted | Reported result |
| --- | --- | --- |
| [Official typed-decisions walkthrough](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md) | Full tuning on 1,200 cases containing 6,000 teacher-labeled decisions, with a separate calibration slice. | Base accuracy 36.2%; published adapted checkpoint 76.6% on 2,000 test decisions. This measures agreement with the benchmark's teacher labels. |
| [Official three-seed loss comparison](https://github.com/NandhaKishorM/laya/blob/main/research/scripts/step4_benchmark/GATE_D_RESULTS.md) | Same data and full-tuning recipe; compare RLCD, Laya's reinforcement-learning-style objective, with supervised cross-entropy. | Mean accuracy 77.62% versus 77.63%. The difference changes sign across seeds; the report retains RLCD as the default. |
| [Community duplicate-question LoRA](https://github.com/vishalmysore/layaMOE/blob/main/results/qqp_lora.json) | Rank-16 adapters in the last four encoder layers, including attention and feed-forward layers, plus decision-head training. Starts from the typed-decisions checkpoint. | Accuracy 77.2% to 78.0%; F1 0.767 to 0.791. Its paired significance test reports p = 0.299, so this small experiment does not establish a reliable accuracy gain. |

The official loss comparison is useful because it qualifies an earlier
single-seed cross-entropy win. Across three seeds, the mean difference is
negligible and the supervised runs vary more. It does not show that one
objective will consistently improve this binary LoRA router. Both losses in
that panel use the same teacher probability targets; it is not a comparison
of hard labels with soft labels.

## What we use in this example

We use one fixed recipe so that the stages remain understandable:

- Start from the English Laya checkpoint and keep the input questions fixed.
- Train rank-16 attention adapters and the complete native decision head.
- Learn two actual answer-correctness targets, one for each answer model.
- Choose the number of epochs on held-out training papers, then refit all
  training examples and choose the routing cutoff on separate calibration data.
- Repeat the same recipe with 503 and 2,503 training questions.
- Check that merging and saving the adapters preserves the decision scores.

Our labels are hard correctness outcomes from independently reviewed answers.
The teacher-label stage substitutes a teacher-generated reference and compares
it with a matched reference-label control. It does not treat the teacher's
self-reported confidence as a verified probability of correctness.

The official training interface supports both hard expected answers and teacher
probability targets. That makes either kind of supervision technically possible;
it does not establish that purchasing probability labels would improve this
task. The recorded example uses the pinned training implementation and settings
described in [methodology](methodology.md).

## What the comparison can establish

LoRA limits how many weights are updated, which can help with small datasets.
Full fine-tuning has more capacity but is not automatically better: learning
rates, regularization, stopping, and data size also matter. Our current study
compares unchanged Laya, its LoRA adaptation, and embedding classifiers. It does
not establish that LoRA generally outperforms full fine-tuning.

Model accuracy on another benchmark and teacher self-agreement are not ceilings
on this router's independently measured answer quality. The final comparison
must show the actual quality gap and dollar savings, using the same recorded
answers for every routing method.
