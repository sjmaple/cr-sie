# Result check before workshop packaging — 2026-10-17

**Decision: keep this in the experiment stage.** The Qwen encoder reduces
answering cost, but the large development gain did not transfer to held-out
requests. Qwen27B can generate usable teacher labels through production SIE;
those labels have only modest predictive value for escalation benefit so far.
There is no new GEPA improvement claim from this experiment.

## Frozen embedding confirmation

Frozen before test collection: Qwen/Qwen3-Embedding-4B, the existing task
instruction, cosine-weighted 20-neighbor routing, a 98% training quality target,
and training-only cutoffs. Compare 64 versus 437 feedback examples; average the
64-example result over all ten previously specified feedback orders. The same
728 test requests were held out of this development exercise, although earlier
BGE/reranker studies already reported their outcomes. They are not a fresh,
independently collected test set.

| Haiku 5.5 / Sonnet 5.5 router | Historical cost / 1,000 | Quality relative to always Sonnet |
| --- | ---: | ---: |
| Qwen embedding, 64 examples; mean of ten banks | $8.7300 | 97.89% |
| Same banks, thresholds recalibrated with 437 outcomes | $9.0506 | 98.77% |
| Qwen embedding, expanded to 437 examples | $7.4052 | 98.15% |
| BGE embedding, 437 examples | $8.9151 | 98.55% |

Expansion saves **15.18%** versus the initial-bank mean (paired 95% interval
8.23–23.86%). Quality changes by **+0.26 percentage points** (−1.06 to +1.49).
Against recalibration alone it saves 18.18%, with −0.62 points of measured
quality. Against BGE437 it saves 16.94%, with −0.40 points of quality.

The predeclared success gate required at least 25% savings, at least 98% retained
quality, positive lower cost-savings confidence bound, and a lower quality-change
bound of at least −1 point. **It did not pass.** The OpenAI secondary pair also
failed: its frozen full-bank cutoff sends everything to the stronger model.

These are historical downstream answer costs plus a common archived embedding
allowance of $0.01324 / 1,000, used to isolate routing decisions. They exclude
training and feedback acquisition. Fresh Qwen test collection took 46 batched
API calls, 22.18 seconds, and 927 billed credits for 728 requests. The larger
bill in a routing comparison comes from selected answers, not an expanding
per-query embedding computation.

## Qwen27B teacher pilot

Model: `Qwen/Qwen3.8-27B-FP8`, production SIE. It labeled the fixed 64 initial
training requests using only request text and a recorded rubric. Each label
contains a difficulty level, an estimate of escalation benefit, and a brief
rationale. These are model judgments, not human-reviewed gold labels.

For Haiku/Sonnet, the stronger model had a better recorded answer on 13 of 64
requests. Teacher difficulty predicts that event with **AUC 0.599** (95% interval
0.455–0.741); explicitly estimating escalation benefit gives **AUC 0.662**
(0.507–0.804). AUC 0.5 is chance ranking; these are not accuracy percentages.
The OpenAI secondary pair gives 0.633 and 0.692, respectively. This is a small
development diagnostic, not held-out evidence for a distilled router.

The completed pilot used 8 API calls and 1,471 credits. Two earlier format
probes cost another 198 credits: JSON-object mode returned `{}`; plain-text mode
returned a valid bare label array. The final parser accepts an object or array
envelope, and still rejects missing IDs, extra fields and invalid label values.
All attempts are retained; there are no automatic inference retries.

Do not use teacher agreement as evidence of downstream answer quality. Before
distilling or optimizing these labels, the next experiment must specify whether
it is learning an explicit difficulty policy or predicting real model benefit,
and evaluate against an independent reference appropriate to that objective.

## Research replay

The public evidence bundle includes frozen policies, vectors, all labels and
receipts, and a source snapshot. Once downloaded, use its `code/` directory:

[Pinned Hugging Face evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/3fce9668dea29fb8c23390a7b450925e5826ea94/evaluations/2026-10-17/request-routing-confirmation-v1)
contains 78 files (19.1 MB), all verified by anonymous download. Manifest SHA-256:
`3ef874683994bddd1a3d3c198bec9e9f04ec4c713a03f611c1c6fe54cd8fe22c`.

```sh
uv sync --frozen --group gepa --group dev
uv run --frozen python fetch.py --reranker
uv run --frozen python confirm.py --development ../development --run ../confirmation
uv run --frozen python teacher_pilot.py --run ../teacher
```

Replay makes no inference calls. Fresh collection explicitly requires `--collect`
and a new directory; `confirm.py --freeze` must precede test collection. This
research note is not the attendee one-pager. Class packaging remains deferred.
