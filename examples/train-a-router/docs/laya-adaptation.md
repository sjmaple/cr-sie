# What other Laya adaptations teach us

Source review: 2026-10-18. The results below are the authors' reports, not
experiments reproduced by this example. They motivate the LoRA study without
setting its expected accuracy.

| Adaptation | Supervision and approach | Reported result and limitation |
| --- | --- | --- |
| [Official typed-decisions](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md) | 1,200 cases yield 6,000 decisions with teacher probability targets; full tuning, four epochs, effective batch 64, encoder/head rates 2.5e-5/1e-4. Approximately 400 decisions are held for calibration. | Base 36.2% → published checkpoint 76.6% on 2,000 test decisions. This measures agreement with the benchmark teacher. |
| [Early loss comparison](https://github.com/NandhaKishorM/laya/issues/887) | Same typed-decisions task; compare RLCD with supervised soft cross-entropy. | RLCD 77.15%; soft CE 79.0% in one seed. A [second single-seed comparison](https://github.com/NandhaKishorM/laya/issues/741) also favors soft CE; these do not establish a consistent advantage. |
| [Later three-seed loss comparison](https://github.com/NandhaKishorM/laya/blob/main/research/scripts/step4_benchmark/GATE_D_RESULTS.md) | Same pinned training contract and teacher distributions; only the loss changes, across seeds 0, 1, and 2. | Mean accuracy: RLCD .7762, soft CE .7763. The paired difference changes sign; seed standard deviation is .0016 versus .0155. The authors retain RLCD as the default. Neither objective has a demonstrated mean-accuracy advantage here. |
| [Community QQP LoRA](https://github.com/vishalmysore/layaMOE/blob/main/results/qqp_lora.json) | Start from `laya-typed-decisions`; 1,500 duplicate-question pairs; rank 16 adapters in the last four encoder layers, attention and MLP; train the native head too. | Accuracy 77.2% → 78.0%; F1 .767 → .791. Reported McNemar p=.299: a working recipe, without a statistically established accuracy gain. |
| [Browser decision model](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md) | Thousands of browser states and goals, successful actions and trajectory negatives; full tuning; revise candidate representation to avoid truncation. | The walkthrough reports element selection around 10% → 66% on held-out pages. This is a narrower task than complete browser-task success; later model versions and harnesses have different results. |
| [DynaSent and SNLI](https://github.com/rsk2327/decision-model-benchmarks) | Full tuning of `laya-multilingual` on 80,488 sentiment examples or 100,000 inference examples; development data for calibration, followed by frozen test evaluation. | Reported macro F1: DynaSent .4524 → .7558; SNLI .8593 → .8824. These use a different base checkpoint, different tasks, and much more training data than this router. |

The [QQP notebook's code](https://github.com/vishalmysore/layaMOE/blob/main/notebooks/laya_lora_qqp.ipynb)
adapts `attn.Wqkv`, `attn.Wo`, `mlp.Wi`, and `mlp.Wo` in layers 24–27, with
rank 16, alpha 32, dropout .05, both learning rates 2e-4, batch 16 and three
epochs. It holds out 150 of the 1,500 generated training pairs for validation,
uses ordinary hard-label cross-entropy, AdamW with weight decay .01, and a
6% warmup followed by linear learning-rate decay. Our `top-layers` candidate isolates
attention-layer scope using the same slower settings as our `slow` candidate.
It is inspired by that example, not a reproduction of its recipe or checkpoint.
The [QQP input builder](https://github.com/vishalmysore/layaMOE/blob/main/scripts/gen_qqp_data.py)
also uses a boolean `noul` question with three instruction wordings, whereas
this router uses one fixed two-option `choice` rubric. That is a separate input
design choice, not an incompatibility between the checkpoints.

## Checkpoint and input differences

The [English checkpoint configuration](https://huggingface.co/convaiinnovations/laya/blob/aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8/rl_agent_config.json)
and [typed-decisions configuration](https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/main/rl_agent_config.json)
both specify ModernBERT-large and two native head layers. Their default input/head
budgets differ: 512/192 for English, 1,024/256 for typed-decisions. This example
explicitly uses 512/256. Both accept the same typed `state` and `questions`
contract; typed-decisions is not a different classifier architecture.

The [typed-decisions model card](https://huggingface.co/convaiinnovations/laya-typed-decisions)
describes additional training on four synthetic business workflows. It cautions
that unrelated tasks can perform like the English base or worse. Starting from
that checkpoint is therefore a separate transfer-learning candidate, not an
established upgrade for mathematics difficulty.

## Decisions for this example

Retain supervised cross-entropy as the simplest controlled objective for this
experiment. The later three-seed panel qualifies the early single-seed wins:
mean accuracy is essentially equal, while RLCD is more stable in that reported
full-fine-tuning experiment. It does not show that CE is superior, or that RLCD
will improve this binary LoRA router.

The [training loader](https://github.com/NandhaKishorM/laya/blob/main/laya/train.py)
supports both `gold` probability targets and one-hot `expected` labels with the
same soft cross-entropy objective.
The loss comparisons above keep the same teacher distributions; they are not
soft-target versus hard-target ablations. Our examples have hard human
difficulty labels. Label smoothing is explicit regularization, not a teacher's
estimated probability distribution, and the QQP LoRA example also uses hard
labels. None of these sources establishes that buying teacher probabilities is
necessary for this task.

Test adaptation capacity and more data. The original recipe trained the whole
native head alongside encoder adapters. New candidates slow the updates,
regularize them, freeze the head, train only the scorer beside the adapters,
or restrict adapters to the last four encoder layers. Then the
validation-selected recipe gets all 2,581 eligible training families instead
of 512. No new teacher calls are needed for this particular dataset.

Finish that declared study before adding candidates. If another recipe is
needed, the attention-plus-MLP scope and warmup in the actual QQP code are
concrete, inexpensive changes to investigate on development data. Keep the
English base fixed when testing those changes; a checkpoint change is a separate
factor. The public QQP result supports feasibility, not an expected accuracy
gain on this dataset.

Select checkpoints for the routing decision. Validation ranking continued to
improve after raw cross-entropy deteriorated in the initial runs. The LoRA study
therefore selects maximum validation ROC AUC, retaining the earlier epoch on
ties, and fits the decision cutoff on separate calibration data. It retains
cross-entropy and training loss as diagnostics. AUC selection is a declared
protocol change; the original CE-selected results remain available.

## Keep the evaluation boundaries

The [typed-decisions dataset card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
describes teacher-generated labels. Teacher agreement is a different claim
from independently verified business outcomes. A teacher self-agreement
statistic is not a mathematical ceiling on a student's score.

The [official notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
splits calibration after flattening cases into decisions; sibling questions
can therefore appear in both training and calibration. This does not itself
establish test leakage. We retain this example's stricter grouped,
chronological partitioning and pinned inputs. We also measured truncation:
only 9/512 training and 4/247 validation examples exceed the native input
budget, so context length is not the first intervention.

The already-published 479-case test is an **exposed regression benchmark** for
this LoRA study. New recipes are chosen without its outcomes, but a fresh
temporal generalization claim still requires additional evaluation data.
