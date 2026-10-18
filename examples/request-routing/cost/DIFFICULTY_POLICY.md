# A routing policy that improves with feedback

The [decision-model post-training follow-up](./POSTTRAIN.md) explains the classifier architecture and reports completed full/LoRA fine-tuning
and a fresh 512-problem confirmation set.

**Research result, 2026-10-17:** a small classifier trained on fixed
`Qwen/Qwen3-Embedding-4B` vectors reaches **89.6% held-out accuracy**, up from
71.7% for a 64-example nearest-neighbor router. Errors fall **145 → 53 (63%)**.
Keeping the classifier method the same, increasing feedback from 64 to 512
labels cuts errors **105 → 53 (50%)**. This is the strongest demonstrated
workshop recipe. The [one-page walkthrough](../WORKSHOP.md) now runs the complete
learn → evaluate → deploy flow.

The task is explicit: send a mathematics problem to **routine** or **advanced**
handling according to human-performance-derived difficulty. These labels do
not identify which answering LLM is best. This experiment measures routing
accuracy, not answer quality or downstream generation savings.

## Evidence and method

The source is [Ding et al., Easy2Hard-Bench](https://arxiv.org/abs/2409.18433),
specifically the [Easy2Hard-AMC dataset](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench).
Published contest difficulty ratings derive from human performance. Our binary
policy uses `rating_quantile < 0.5` for routine, and the upper half for advanced;
we keep the full difficulty range, including the ambiguous middle. Both labels
refer to competition mathematics, not everyday request difficulty.

Pinned source revision: `55bc0d2fb10954151e669d2026b87fa896f2fa26`.
Data and adaptations retain **CC-BY-SA-4.0**, with attribution to the source.
The bundle includes the original dataset card and Parquet files, our derived
split, frozen policies, per-request predictions, vectors, API receipts,
prompts, failed trials and a code/lockfile snapshot. Code is Apache-2.0.

[Pinned public evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/002087323da76e925235e0c6acb12f402bf8fd9e/evaluations/2026-10-17/request-routing-difficulty-policy-v1):
486 files, 30.3 MB, each verified by anonymous download. Manifest SHA-256:
`894d168a197116963c1e55a6e3baac1988ffc89d97460bbe6532a5b91ff5f7e2`.
[Learning-curve figure](https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/002087323da76e925235e0c6acb12f402bf8fd9e/evaluations/2026-10-17/request-routing-difficulty-policy-v1/main/learning-curve.png).

We deterministically select balanced **512 train / 128 validation / 512 test**
examples. Train and validation come from official train; test comes from
official eval. Normalized duplicate problem texts are excluded across splits.
Only problem text goes to the models: no solutions, ratings, source tags or
labels. Text can contain LaTeX or Asymptote diagram markup; no images are rendered.
These are public historical problems and may have appeared in model pretraining.

For each predeclared training size, we fit cosine-weighted 5-neighbor routing
and a ridge classifier. Ridge regularization is chosen from four values using
exact training leave-one-out mean squared error; classification cutoffs use
training leave-one-out scores. All sizes are reported. Validation selects GEPA
candidates. Every policy and cutoff is frozen before its test evaluation.

## Held-out results

| Policy | Training labels | Accuracy | Errors / 512 |
| --- | ---: | ---: | ---: |
| Nearest neighbors | 64 | 71.68% | 145 |
| Nearest neighbors | 128 | 74.61% | 130 |
| Nearest neighbors | 256 | 78.52% | 110 |
| Nearest neighbors | 512 | 78.52% | 110 |
| Learned linear classifier | 64 | 79.49% | 105 |
| Learned linear classifier | 128 | 84.96% | 77 |
| Learned linear classifier | 256 | 88.87% | 57 |
| Learned linear classifier | 512 | **89.65%** | **53** |
| Word-count-only control | 512 | 58.01% | 215 |
| Fixed Qwen4B reranker, cutoff from 64 | 64 | 75.20% | 127 |
| Same reranker, recalibrated cutoff | 512 | 73.44% | 136 |
| GEPA-evolved reranker | 512 | 74.61% | 130 |
| Fixed Qwen27B classification rubric | 128 calibration; 4 in-context | 75.78% | 124 |
| GEPA-evolved Qwen27B rubric | 128 | 72.46% | 141 |

The main kNN64 → linear512 comparison gains **17.97 percentage points**,
paired 95% interval **14.06–22.07**, with **63.45% fewer errors**. It corrects
113 requests and regresses 21. The same-method linear64 → linear512 control
gains **10.16 points**, interval **6.84–13.67**, with **49.52% fewer errors**.
Intervals use 4,000 paired label-stratified bootstrap samples, seed 89.
The comparisons describe these trained policies on this test population;
they do not include uncertainty across fresh training samples.

The 512-label classifier misroutes 23 of 256 advanced problems as routine and
30 of 256 routine problems as advanced. It learns a useful boundary; it is not
a guarantee of perfect escalation. Comparing it with the 27B rubric is a
comparison of these recipes, not an equal-training-data model benchmark.

## What changes, and what it costs

Static mode freezes the example bank or classifier. Improving mode adds reviewed
feedback, refits the local classifier, validates the proposed policy, then
deploys the accepted weights and threshold together. Students can simulate that
process by revealing the nested 64, 128, 256 and 512 training labels. Existing
vectors are reused; a new labeled request needs one additional embedding.

The encoder weights never change. Serving uses **one Qwen4B embedding followed
by a local dot product**, regardless of the number of training labels. One
observed local CPU fit of all four regularization candidates on 512 vectors
took **0.039 seconds**, excluding data loading and embedding collection.

Production SIE receipts: embedding all 1,152 train/validation/test problems
used **72 batched API calls and 911 credits** (510 development, 401 test).
Two additional single-request production routing checks agreed with recorded
routes and used **2 credits each**. These are observed receipts, not a future
tariff or a promise about an account's free-credit allowance. Feedback acquisition
is separate; this experiment reuses public reference labels.

## GEPA status

The reranker is `Qwen/Qwen3-Reranker-4B`; reflection and generative classification
use `Qwen/Qwen3.8-27B-FP8` through production SIE. No GLM, Haiku or Sonnet generated
the reference labels. GEPA 0.1.4 tries 12 description updates, then an 8-update
shared-instruction follow-up. The latter produces no better validation candidate.
The selected reranker loses 0.59 points on test versus the original policy.

A separate 6-update GEPA run improves the generative rubric's validation accuracy
75.78% → 79.69%, but test accuracy falls **75.78% → 72.46%** (paired difference
−3.32 points, interval −5.86 to −0.59). Optimization used 118 calls and 10,099
credits; evaluating both frozen teacher policies used 64 calls and 4,393 credits.

We preserve two discarded teacher attempts: one stopped on incomplete JSON;
another exposed alternating class order in generative batches. The latter was
invalidated before inspecting its test metrics. The corrected run orders batches
by a label-independent hash; a regression test verifies that ordering and score
alignment. Only `teacher/teacher-results.json` is valid teacher evidence.
The earlier `main/teacher-*` and `main/test-teacher/` records are audit-only.

The original combined success gate required both embedding learning and GEPA to
reduce errors by at least 25%, with positive lower confidence bounds, plus a
GEPA gain of at least 3 points beyond recalibration. **The embedding comparison
passes; the combined gate fails.** Its recorded result remains false. GEPA has
not earned a success claim on this task and should not be the workshop headline.

## Research replay and live use

From this repository's `examples/request-routing/cost/`, download the pinned
replay subset (no key required):

```sh
uv sync --frozen --group gepa
uv run --frozen python policy_evidence.py
uv run --frozen python difficulty_policy.py replay --data evidence/difficulty-policy/main/data.json --run evidence/difficulty-policy/main
uv run --frozen python gepa_teacher.py replay --run evidence/difficulty-policy/teacher
```

From the public bundle's `code/` directory, after downloading the bundle:

```sh
uv sync --frozen --group gepa --group data --group analysis
uv run --frozen python difficulty_policy.py replay --data ../main/data.json --run ../main
uv run --frozen python gepa_teacher.py replay --run ../teacher
uv run --frozen python difficulty_policy.py route --data ../main/data.json --run ../main --recorded-id e2h-amc-eval-1602
uv run --frozen python plot_policy.py --run ../main
```

These commands make no inference calls. Rebuilding the dataset is also local:

```sh
uv run --frozen python difficulty_policy.py prepare --source ../source --data ../rebuilt-data.json
```

With a production SIE Console API key supplied as `SIE_API_KEY` and
`SIE_BASE_URL` unset, the same trained policy routes a new request:

```sh
uv run --frozen python difficulty_policy.py route --data ../main/data.json --run ../main --request 'Find the number of integer solutions to x^2 + y^2 = 2025.'
```

This returns a difficulty route; it does not call a downstream answering model.
The offline replay, production path and local SIE 0.10.0 Apple Silicon path are
verified. All 32 checked local embedding routes matched the recorded Cloud routes;
all selected models are at most 27B. Full GEPA optimization is optional research work with separate credit
budgets, not a required free-tier attendee step.

## Fresh confirmation and the completed course

The unchanged embedding policies were also tested on 512 previously unused
official-eval problems, reserved before decision-model training. kNN64 scores
74.22%, ridge64 82.23%, and ridge512 **91.60%**. The main update reduces errors
**132 → 43 (67.42%)**, gaining **17.38 points** (paired 95% interval **13.28–21.68**).
The same-learner update reduces errors **91 → 43 (52.75%)**. This confirms the
embedding learning result on a fresh sample without changing its policy.

The [post-training chapter](./POSTTRAIN.md) demonstrates a separate positive
result: native Laya rises from 53.32% to 87.11% with full tuning, or 88.09% with
encoder LoRA plus head training, on the same confirmation cases. It gives up
accuracy to Qwen4B ridge in exchange for a much smaller serving model. GEPA's
recorded negative result and original failed combined gate remain unchanged.

[Confirmation and decision-model evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/85c9d93a73d5559a6c69a22145f5f86f6b450fbc/evaluations/2026-10-18/request-routing-course-posttrain-v2) includes both exported checkpoints
and the final local/Cloud verification receipts.
