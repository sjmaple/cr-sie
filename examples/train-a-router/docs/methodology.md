# How the comparison is measured

The router chooses between two answer models. We measure whether its choices
save money while keeping final-answer accuracy close to always using the
stronger model. Predicting a difficulty label is not the goal.

## Questions, labels, and independent evaluation

The questions come from [GSM8K](https://github.com/openai/grade-school-math),
pinned to revision `3101c7d5072418e28b9008a6636bde82a006892c`.
The download contains the source license and attribution.

| Role | Questions | What can use their answers |
| --- | ---: | --- |
| Small training set | 500 | Fit models and choose training settings. This is a subset of the expanded set. |
| Expanded training set | 2,000 | Fit models and choose training settings. |
| Calibration | 1,000 | Choose the routing cutoff after the models are trained. |
| Test | 3,000 | Measure the saved models and cutoffs once. |

The 2,000 training questions had already been used for development. They are
not counted as fresh test evidence: they contain all 1,319 official GSM8K test
questions and 681 official training questions. The calibration and test questions were
selected from unused families in the original GSM8K training split. Thus our
study roles differ from the original benchmark's train/test split; the results
are a custom routing evaluation, not standard GSM8K benchmark scores.

A **question family** groups exact duplicates and similar question templates.
Before collecting fresh answers, we grouped normalized text and word shingles,
then checked character four-gram similarity both with the original numbers and
with numbers replaced by a placeholder. Similarity thresholds are 0.80 and
0.85 respectively. The latter helps catch the same problem with different
numbers. Every detected family stays within one study role; each fresh role
uses only one representative per family. The 500-question subset also keeps
whole families together.

These checks found 6,751 unused families, enough for the 4,000 fresh questions.
The source audit preserves the detected pairs and exclusions. Lexical checks
cannot establish that all semantic relatives have been found, or that a model
never saw GSM8K during its original pretraining.

## Record both possible answers

For each question, we record one answer from each model using fixed prompts,
token limits, model revisions, and runtimes:

| Answer model | Initial output limit | Runtime used for fresh collection |
| --- | ---: | --- |
| Qwen3.5-4B | 1,536 tokens | SGLang 0.5.10.post1, CUDA 12 |
| Qwen3.8-27B-FP8 | 3,072 tokens | SGLang 0.5.20, CUDA 13 |

An initial answer that is truncated or lacks a final answer receives at most
one additional 256-token call to extract its final answer. The corresponding
charges are included. There is no retry based on an answer being incorrect.
The recorded settings and model revisions are in [solver.py](../solver.py).

Both answers are checked against the dataset's reference answer. Equivalent
numeric forms count as equal. A small arithmetic adapter handles clock-hour
answers; durations remain ordinary quantities. Unresolved parsing or unit
comparisons are retained for review instead of silently dropping a question
or treating the answer as wrong. The one calibration-format adjudication
preserves the original verdict and binds the reviewed answer to its file hash.
It was resolved before choosing the cutoffs.

Every router is evaluated against these same recorded answer pairs. We do not
regenerate answers separately for each router, which would mix routing changes
with variation in answer generation.

## Learn correctness and the value of an upgrade

Each question supplies two labels: whether the cheaper model was correct and
whether the stronger model was correct. Both can succeed; both can fail; the
cheaper model can occasionally be the only one to succeed.

The routers predict both correctness probabilities. Their routing score is:

```text
predicted improvement in correctness
-----------------------------------
predicted extra cost of the stronger answer
```

The denominator has a floor of one credit. Expected costs use four means from
training only: each model's cost when correct and when incorrect. This matters
because long, unsuccessful answers can cost more than short, successful ones.
The stronger model is selected when this score is at or above the saved cutoff.

The embedding approach keeps **Qwen3-Embedding-4B** fixed. Its 2,560-number
representation feeds two logistic classifiers: each is a weighted sum plus a
bias, followed by a conversion to a probability. Only those small classifiers
are trained. Five folds of training families choose one shared regularization
strength from `0.01, 0.1, 1, 10`, using prediction loss. Stronger regularization
wins an exact tie. The final classifiers are refit on all training rows.

The decision-model approach uses one shared **Laya** checkpoint, based on
ModernBERT-large, and asks it the same two correctness questions. The unchanged
checkpoint is the baseline. Adaptation trains rank-16 attention LoRA updates
and the complete decision head. The recipe uses seed 89, adapter learning rate
0.0002, head learning rate 0.0001, dropout 0.05, and weight decay 0.01.

We check epochs 0 through 8 on one reserved training-family fold. **Epoch 0**
means no adaptation. The epoch with the lowest combined prediction loss is
selected, with earlier epochs winning ties. We then restart from the original
checkpoint and train on all rows for that many epochs. Calibration and test
outcomes never select the training duration. Both arithmetic training sizes
selected one epoch. Adapter merging and checkpoint export are checked for
score drift before use.

For each training size, a matched unchanged-Laya control uses the same
training-derived cost estimates as its adapted counterpart. Its weights stay
fixed, and calibration chooses its own cutoff. This separates changing the
weights from updating the expected answer costs. The controls were saved in a
separate, hash-checked supplement before opening test answers; they do not
replace or modify the original frozen comparison.

This comparison does not establish that LoRA generally beats full fine-tuning,
or that a model's parameter count alone explains its performance.

## Choose a cutoff, then open the test results

The primary target is an empirical **one-percentage-point quality gap on
calibration**. We choose the lowest-cost qualifying cutoff, also considering
always-cheap and always-strong policies. Secondary targets of 0, 2, and 5
points are declared in advance and retained in the report.

The source plan aimed for at least 30% savings near a one-point quality gap.
The primary 2,000-example embedding rule saves 20.87% with a 0.47-point test
gap, so it does not reach that savings aim. The declared two-point calibration
setting saves 33.77% with a 0.83-point test gap; its gap interval extends to
1.30 points. We report that as a secondary operating point, without changing
the primary comparison or claiming a guaranteed one-point bound.

All model files, input-only test scores, and calibration cutoffs are saved
before test answers are graded. A test result can exceed its calibration
target: the target is not a guarantee. The public download contains the frozen
decisions and test-access record. Replaying these published results is an
**exposed reproduction**, not another blind experiment.

The report keeps additional controls: models trained on competition problems
with their original cutoffs, and those same weights with arithmetic-only
cutoff calibration. These distinguish adjusting the cutoff from training
weights on the new workload. Their original training-derived cost means stay
fixed. They are diagnostic comparisons, not extra stages learners must run.

Separate experiments used only the 2,000 exposed development questions. An
answer-aware cascade, which inspects the cheaper model's answer before deciding
whether to upgrade, was screened with training-fold validation. Two fixed
changes to the cheaper solver's generation settings were also tested on all
2,000 questions. Neither solver change met the improvement rule specified
before collecting those answers. These diagnostics did not change the final
model pair, trained routers, cutoffs, or test questions; their executed sources
and results are retained as optional evidence, outside the walkthrough.

## Teacher labels when answer keys are unavailable

An independent **GPT-6 Astra** teacher receives the training questions
using high reasoning effort. It sees only each question, never the benchmark
reference or either answer model's output. The answer contract requests one
number in the requested units, without unit text or currency symbols.

The numeric-format collection completed **1,539 of the planned 2,000
references** before the provider account exhausted its credits. We closed
that acquisition and preserved its incomplete collection record. The remaining
461 questions are missing because of acquisition availability, not an answer
quality filter. All 1,539 completed references satisfied the numeric contract;
both teacher and reference-label classifiers use exactly those same questions.

A valid numeric reference is used to grade both answer models. Incorrect
teacher answers remain in the training labels. We do not replace them with
benchmark answers. A matched control trains the same classifiers with benchmark
labels on exactly the same eligible questions and family folds. Comparing the
two isolates the effect of the labeling pipeline, including the conditional
cost means derived from those labels. A separate benchmark audit measures
the teacher's reliability; it does not decide which teacher labels to retain.
The teacher's references agree with the benchmark on **98.25%** of these
training questions. The derived correctness labels agree on **98.70%** of
cheaper-model answers and **98.51%** of stronger-model answers. These are
training-label checks, separate from the held-out routing comparison.

An earlier teacher run used a general mathematics answer format. Its unit text
exposed limitations in the generic parser, so we preserved that attempt and
collected the numeric-format run before opening the test results. The label
cost report keeps these acquisition expenses separate from inference savings.
Known numeric-run usage is **$6.40**; the earlier general-format acquisition
used **$8.75**. Attempts without verified usage remain recorded separately,
so these figures are usage-based estimates rather than a complete invoice.

A numerically stable, algebraically equivalent logistic-loss calculation is
used for both matched heads. It avoids floating-point cancellation; the
objective, regularization grid, folds, and selection rule are unchanged.

The teacher stage replaces **training** answer keys. Calibration and test still
use trusted benchmark answers. In a deployment without any labels, reserve a
small independently reviewed sample for quality measurement; agreement with
the training teacher alone cannot establish real answer quality.

<a id="selection-check"></a>

## Check selection separately from the fraction sent to the cheaper model

A router can preserve more quality simply by using the stronger model more
often. To check its selection, compare it with the expected result of randomly
sending the **same fraction** of questions to the cheaper model.

If that fraction is `p`, the random expectation is:

- quality gap = `p × always-cheap quality gap`;
- answer cost = `p × always-cheap cost + (1 − p) × always-strong cost`.

These are algebraic expectations from the same 3,000 recorded answer pairs,
not another model run, a sampled random policy, or new cutoff selection.
The following descriptive check uses the primary saved setting:

| Router, 2,000 training examples or cost records | Cheap-model share | Random expected gap | Router's measured gap | Random expected answer cost / 10,000 | Router's answer cost / 10,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laya, unchanged | 26.2% | 1.28 points | 1.50 points | $7.61 | $7.46 |
| Laya + LoRA | 21.7% | 1.05 points | 0.80 points | $7.94 | $7.95 |
| Embeddings + classifiers | 34.9% | 1.70 points | 0.47 points | $6.99 | $7.46 |

Unchanged Laya does not beat this random expectation on answer quality.
LoRA shows a modest selection advantage in these point estimates; the
embedding router shows a larger one. At the secondary setting, the embedding
router uses 4B for **52.8%** of questions: the random expected gap is **2.57
points**, compared with its measured **0.83 points**.

Equal cheap-model shares do **not** imply equal spending. Answer length and
cost vary by question. The embedding router's primary answer bill is about
**$0.47 per 10,000 higher** than the random expectation: it preserves more
quality while keeping different, more expensive questions on 27B. The table
excludes all router charges to compare answer costs consistently; the main
embedding result also includes its roughly $0.059 embedding charge per 10,000.
This check explains selection behavior; it does not establish a statistically
significant improvement or dominance at equal cost.

## Money, uncertainty, and serving costs

Fresh answer generation runs on Modal. We estimate its equivalent Cloud cost
from recorded physical-call token counts and a saved Cloud price table. These
are **Cloud-price estimates**, not observed Cloud bills or Modal invoices.
Embedding charges are observed Cloud credits, including the original collection
batch size. One credit corresponds to $0.00001 in this report.

For each router, the report sums the selected answer charges and, where
applicable, the embedding charges. Training, data collection, teacher labeling,
and idle hosting are separate expenses. Laya hosting is unpriced, so its
reported answer-model savings are the maximum allowance for router hosting
before breaking even. Tiny local classifier CPU work is also unpriced. If the
selected policy always uses one model, it skips router inference entirely.

**Quality gap** is always-strong accuracy minus routed accuracy, in percentage
points. It includes both answers lost by choosing the cheaper model and answers
recovered when only that model succeeds. A one-point gap means 100 fewer
correct answers per 10,000 requests, on average.

The training recipe uses one seed. More examples require more training work;
this is not an experiment with a fixed compute budget across training sizes.

The 95% intervals resample questions while keeping each pair of answers and
its routing decision together. They describe sampling uncertainty on this
workload, conditional on the saved models and rules. They do not include
training-seed variation or adjust jointly for all comparisons, and do not
protect against future domain changes. With 3,000 test
questions, one changed answer is about 0.033 points; the actual uncertainty
also depends on how often the two policies disagree in correctness.

Warm latency is measured separately from checkpoint loading and answer
generation. The [serving guide](serving.md) records hardware, network inclusion,
and the limits of the small sequential timing sample.
