# How we measure a money-saving router

A router is useful when it lowers the bill while keeping answer quality close
to always using the stronger model. This example compares the same two answer
models on the same questions: **Qwen3.5-4B** and **Qwen3.8-27B-FP8**.

The router sees the problem text and competition name. It never sees either
model's answer, a reference solution, the year, the question number, or a human
difficulty rating when making a decision.

## Data and separation between training and evaluation

The questions come from a pinned copy of
[Easy2Hard-Bench](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench).
We collect both answer models' responses, then check their final answers against
reviewed references. Mathematical equivalents such as `1/2` and `0.5` count as
equal. The metric measures final-answer correctness, not the completeness of a
proof or the quality of the writing.

| Group | Reviewed questions | What it controls |
| --- | ---: | --- |
| Original training | 503 | A modest initial labeling budget. |
| Expanded training | 2,503 | More examples from the same training papers. |
| Calibration | 242 | The cutoff between the cheaper and stronger model. |
| Main test | 463 | The final cost and answer-quality comparison. |

Training uses competitions through 2017, calibration uses 2019, and the main
test uses 2020 onward. The source data also retains a 2018 validation group;
this recipe does not use it to choose models. Instead, model settings are
selected on held-out papers within the training group. All problems from the
same competition and year stay together in those folds.

Duplicate checks group exact text matches, similar word sequences, and text
that matches after numbers are masked. Families crossing chronological groups
are removed, and one representative is selected per remaining family. The
original training set is a subset of the expanded set. Neither overlaps the
calibration or test families. These checks cover our router training; they
cannot establish what public problems the underlying models saw in pretraining.

An explicit source review corrects references and excludes questions that
cannot be answered from the supplied text, including incomplete tasks. That
review was amended after exploratory training, before the corrected fits and
before access to the held-out answer models' test results. The downloadable
review retains the decisions and supporting proofs. Source exclusions do not
remove a question merely because a model answered incorrectly.

## What the router learns

Each training question provides two labels:

| Smaller model | Stronger model | Meaning for routing |
| --- | --- | --- |
| Correct | Correct | The smaller model can save money. |
| Wrong | Correct | Paying for the stronger model helps. |
| Wrong | Wrong | Paying more does not recover this answer. |
| Correct | Wrong | The smaller model also gives a better answer. |

We keep all four cases. The models predict the probability that each answer
model will be correct. Their difference estimates the stronger model's quality
advantage. We divide that difference by the predicted extra answer cost to
obtain the routing score:

```text
predicted extra correct answers / predicted extra answer-model credits
```

The cost estimates come only from training records. For each answer model, we
calculate its mean token cost when correct and when wrong, then weight those
two means by its predicted correctness. A new request does not reveal its
future answer length or correctness to the router. A one-credit denominator
floor keeps the score defined when the estimated cost difference is very small.

## The decision-model approach

Laya has roughly 421 million parameters. It combines a ModernBERT-large text
encoder with a decision head: layers that score supplied choices instead of
generating an answer. We ask two questions about the same problem, one for
each answer model, with the choices `incorrect` and `correct`. Both questions
use one shared Laya checkpoint.

The unchanged baseline uses the published weights. Training examples supply
its cost estimates, and calibration supplies its routing cutoff. Its predicted
probabilities should not be assumed to be perfectly calibrated.

LoRA adds rank-16 updates to the attention layers while most encoder weights
stay fixed. We also train the complete decision head. Training uses the two
correctness labels and cross-entropy, which penalizes confident wrong
predictions. One held-out training-paper fold selects the number of epochs
from zero through eight; an epoch is a pass through the training examples.
The selection criterion is the summed negative log likelihood of both labels.
Ties prefer the earlier epoch.

We then start again from the original checkpoint and fit all eligible training
questions for the selected number of epochs. The seed is 89. The original and
expanded populations use the same recipe. Training runs on NVIDIA CUDA; no
Apple-specific training result is substituted into this comparison.

LoRA updates are merged into ordinary model weights. We save those weights at
32-bit precision and check scores before merging, after merging, and after
reloading. This catches export changes that could otherwise look like a
training-quality difference.

## The embedding approach

The fixed **Qwen3-Embedding-4B** model converts each request into 2,560 numbers.
Two independent logistic classifiers predict the two correctness labels from
those numbers. Each classifier has 2,560 weights and a bias; there are no
hidden layers and Qwen's weights are not updated.

Five training-paper folds select a shared regularization strength from
`0.01`, `0.1`, `1`, and `10`. Regularization discourages large classifier
weights. Selection minimizes the combined negative log likelihood on the
held-out folds; ties prefer stronger regularization. Both classifiers are then
fitted to all training examples. The saved vectors make this fitting step
possible on CPU.

The encoder, text instruction, and vector normalization stay fixed for both
training sizes and both label sources. A new request normally needs one
embedding call and two small CPU calculations. More training examples do not
increase the number of embedding calls or classifier parameters.

The encoder is roughly ten times larger than Laya. This is a comparison of
practical recipes, with different model sizes and pretraining. It does not
isolate which architectural difference causes a quality difference.

## When reference labels are unavailable

The teacher comparison uses **gpt-6-astra** to solve each training problem from
the question alone. It does not receive the benchmark answer or either smaller
model's answer. Its final answer becomes the proposed reference for the two
correctness labels.

Generic formatting checks determine whether a teacher reference can be used.
For example, an unsupported answer form or an unresolved unit is recorded as
unusable. Eligibility depends on the teacher reference itself, not on whether
it agrees with the benchmark or makes either answer model look good. There is
no fallback from a teacher label to the benchmark label.

The matched reference-label control uses exactly the same eligible questions,
vectors, paper folds, and training settings. Only the label source changes.
Each version derives its cost estimates under its own labels. Calibration and
test use the reviewed benchmark references for both versions, so teacher
agreement is not mistaken for independently measured answer quality.

The public commands reuse recorded labels. Acquiring new labels is a separate
one-time expense. A different teacher or a self-hosted 27B model is a new
label-source experiment, not an established equivalent to this teacher.

## Choose a cutoff before seeing test outcomes

Calibration chooses the cheapest routing cutoff whose observed quality gap is
within the declared target. The primary target is **one percentage point**;
zero-, two-, and five-point targets are also reported. Always using the stronger
model and always using the smaller model are explicit alternatives. Those
fixed choices skip the router's inference call.

Every model and cutoff is saved before the held-out answer outcomes are opened.
The answer recordings were collected earlier; access to their outcomes is the
relevant boundary. The published receipts document that sequence. Once the
results are public, rerunning the example is a reproduction on a known test
set. Further development needs a fresh test for new generalization claims.

## Translate the result into money and quality

**Quality gap** is always-strong accuracy minus routed accuracy, in percentage
points. A change from 90% to 89% is a one-point gap: ten fewer correct answers
per 1,000 requests. We report both lost correct answers and rescues where the
smaller model succeeds while the stronger model fails. Losing five and rescuing
two produces a net loss of three.

**Answer-model savings** compare the selected answer charges with the
always-strong charges. **Savings after known router charges** also subtract
embedding API charges. Dollar tables show costs and savings per 10,000
requests, scaled from this measured sample. They are workload estimates, not
a quote for an arbitrary production workload.

Laya hosting has not been priced as a production service. For a threshold
policy, its answer-model savings therefore provide a **hosting allowance**:
the maximum extra router spend before those savings disappear. They are not
net savings. Training and teacher labeling are separate one-time expenses.
Embedding-router results include its measured API overhead, while the small
local classifier calculation remains unpriced.

The main test uses measured SIE Cloud charges, including a conditional
final-answer formatting call when needed. Training answers were collected
through self-hosted SIE servers; their cost model uses actual token counts and
recorded Cloud list prices. Those estimates are neither Cloud bills nor Modal
GPU charges. Runtime and model revisions are recorded, but matching weights
does not guarantee identical responses across serving environments.

## Uncertainty and transfer to new questions

A one-point calibration target is not a promise of at most one point of loss
on future requests. The main test has 463 questions. Paired bootstrap intervals
resample those questions with the routing policy fixed. They describe this
sample's uncertainty but do not account for shared papers, model selection,
solver randomness, or changing prices. One training seed does not establish
stability across seeds.

A separate 2,000-question arithmetic check applies the same saved routers and
cutoffs without retraining. It contains 1,319 official GSM8K test questions and
681 fixed-hash-selected official training questions unused by our routers.
This is a custom transfer study, not a standard GSM8K leaderboard score or an
extension of the competition test. Its generation costs are token-based
Cloud-price estimates; its embedding charges are observed.

Future improvements follow the same cycle: collect requests and independently
verifiable outcomes, update training data, retrain, choose settings on
development data, then evaluate on new questions. Both LoRA and the embedding
classifiers support that cycle. Adding data is an intervention to measure,
not a guarantee that the next version will improve.
