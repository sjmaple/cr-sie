# Where a stronger learning result could come from

The development analysis finds substantial theoretical headroom, but no strong
feedback-learning result for the OpenAI pair. The most promising next experiment
is the **Haiku/Sonnet growing embedding bank with independent calibration**.
The apparent gain below is a development lead, not a new workshop headline.

The subsequent [Qwen3-Embedding-4B comparison](./QWEN.md) tests whether the encoder
is limiting this result, including a fixed task instruction and training-fitted
cutoffs. It is an additional development result, not a replacement test claim.

[Published results, figures and source snapshot](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/788e8bdb5e994a51aff99f974c4f7495aa60c270/evaluations/2026-10-17/request-routing-headroom-v1)
contain all 2,450 comparisons and the initial diagnostic. All 42 files were
downloaded anonymously and verified against the manifest (SHA-256
`6fd230cf31c737f55c33b03d071fd382e8d8b1088b1810c6416ed0565f151da5`).

## What was measured

`headroom.py` uses the existing 437 training and 146 validation requests. It
does not use the 728 test outcomes or the other 146 calibration requests.
It compares 32, 64, 128, 256 and 437 feedback examples across ten nested,
stratum-balanced orders. Seed zero retains the original 64-example bank.
All runs use historical answer costs and recorded routing costs; no API calls.

Two views answer different questions:

- **Ranking potential:** sweep cutoffs on validation and find the cheapest
  result retaining at least the target quality. This is optimistic and cannot
  be presented as a frozen, deployable-policy result. Discrete thresholds may
  overshoot the target slightly.
- **Calibration transfer:** fit the cutoff on training, then apply it unchanged
  to validation. This exposes how often the requested quality is actually met.

The ridge learning curves select regularization by training leave-one-out mean
squared error. That differs from `learn.py`'s validation cost/quality selection;
these curves do not replace the previously published 16.7% test result.
The GEPA endpoint was already selected on this validation set and is optimistic.
There is no GEPA learning curve: its saved run used all 437 training examples.

## Headroom at 98% retained quality

Costs below are USD per 1,000 validation requests, not live tariffs.
Quality is a ratio to always-strong quality, not accuracy; it can exceed 100%
when the cheap model answers some requests better.

| Comparison | GPT-6 Luna / GPT-6.1 Sol | Haiku 5.5 / Sonnet 5.5 |
| --- | ---: | ---: |
| Always strong, 100% quality | $4.305 | $9.999 |
| Random routing, expected 98% quality | $3.922 | $8.879 |
| Fixed reranker, validation-swept cutoff | $2.459 | $7.356 |
| Perfect-information fractional lower bound | $1.287 | $2.168 |

The bound knows both models' eventual grades **and** answer costs, excludes
routing overhead, and can fractionally route one request. It implies theoretical
savings of 70.1% and 78.3% versus always strong, not achievable model performance.
The data has headroom; these numbers do not establish that embeddings can learn it.

## Learning curves: the useful lead

At at least 98% validation quality, averaging over ten feedback orders:

| Router | 64 examples | 437 examples | Cost change |
| --- | ---: | ---: | ---: |
| OpenAI ridge predictor | $2.570 | $2.506 | 2.5% lower |
| Anthropic growing kNN bank | $7.743 | $5.186 | **33.0% lower** |

The Anthropic bank is also 29.5% cheaper than the fixed reranker's $7.356 at this
quality floor. However, the 64-example banks range from $4.082 to $10.012 across
orders: the starting sample matters substantially. All ten 437-example runs use
the same final training set; these are not ten independent evaluation datasets.

The practical gap is calibration. The 437-example Anthropic bank with its
**training-fitted** 98% cutoff costs $6.653 and retains **97.46%**, below target.
Its $5.186 result uses a validation-swept cutoff and retains 98.31%.
Only 24 validation requests favor Sonnet; near 98%, a few changed decisions
can move the result substantially. A separate calibration and evaluation
population is necessary before calling this a reliable improvement.

A development follow-up also predicts incremental answer cost from embeddings
and ranks expected quality benefit per predicted dollar. It did not produce a
stronger learning curve. The GEPA endpoint's validation-swept cost is $2.384
versus the original description's $2.459, about 3%; its published test gain
remains 0.89%. These findings do not justify another identical GEPA run.

## Reproduce on a laptop

```sh
uv sync --frozen --group analysis
uv run --frozen python fetch.py --difficulty
uv run --frozen python headroom.py --difficulty evidence/difficulty --output run-output/headroom
uv run --frozen --group analysis python plot_headroom.py --run run-output/headroom
```

The output contains the split plan, every comparison, a Markdown report, and
PNG/SVG figures for ranking potential and calibration transfer. No credentials,
GPU, GEPA installation or inference credits are needed. Without `--difficulty`,
the diagnostic uses the earlier archived fixed reranker and omits GEPA.

The next experiment should freeze an Anthropic kNN policy using independent
calibration data, compare against both fixed kNN and fixed reranking, and assess
it on newly collected requests. Keep the full learning curve and a matched-quality
comparison; do not select a favorable starting bank or tune on the existing test.
