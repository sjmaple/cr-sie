# Use the same routing decisions with Sol and Luna

This optional case study asks whether the routers from the
[main walkthrough](../README.md) still offer useful savings with a different
answer-model pair. The embedding router saves **29.8% with a 0.27-point quality
gap** at the original primary setting, or **46.2% with a 0.43-point gap** at
the original secondary setting. Embedding charges are included.

We collected answers from **`gpt-6.1-sol`** and **`gpt-6-luna`** on the same
3,000 test questions. Both used medium reasoning, a 4,096-token output limit
including reasoning, and a structured numeric final answer. The API received
the question without its reference answer.

All routing decisions were already fixed by the Qwen study. A choice of the
cheaper Qwen model selects Luna; a choice of the stronger Qwen model selects
Sol. We reused unchanged Laya, the 2,000-example LoRA model, and the
2,000-example embedding classifiers. Their scores, model-specific prompts,
cost estimates, and cutoffs were not changed. This measures transfer of the
old decisions, rather than training or calibrating a Sol/Luna router.
It uses a new answer-model pair, not a new population of questions.

## Start with both simple alternatives

Always using Sol answers **97.80%** correctly at **$7.41 per 10,000 questions**.
Always using Luna costs about **$0.51**, saving about **93.1%**, with
**96.53–96.57%** accuracy: a **1.23–1.27-point** quality gap. The small ranges
account for one unavailable Luna response, described below.

If that quality loss is acceptable, always using Luna is the simpler and
cheapest option measured here. A router is useful when preserving more of
Sol's answer quality justifies the extra spending.

## Compare the original primary rules

The labels `target 1` and `target 2` refer to the original Qwen calibration
goals. They are not quality guarantees for Sol/Luna.

| Router, original target 1 | Cost / 10,000 | Savings | Quality gap, points (95% interval) |
| --- | ---: | ---: | ---: |
| Laya, unchanged | $5.56 + hosting | 25.0% before hosting | 0.20 (−0.10–0.50) |
| Laya + LoRA | $5.92 + hosting | 20.1% before hosting | 0.13 (−0.07–0.33) |
| Embeddings + classifiers | $5.20 | 29.8% | 0.27 (0.03–0.50) |

LoRA has the smallest measured quality gap at this setting; embeddings cost
less. These results do not show embeddings improving both measures over LoRA
for every answer-model pair.

## Compare the original secondary rules

| Router, original target 2 | Cost / 10,000 | Savings | Quality gap, points (95% interval) |
| --- | ---: | ---: | ---: |
| Laya, unchanged | $5.09 + hosting | 31.3% before hosting | 0.27 (−0.03–0.60) |
| Laya + LoRA | $4.73 + hosting | 36.2% before hosting | 0.33 (0.03–0.63) |
| Embeddings + classifiers | $3.99 | 46.2% | 0.43 (0.10–0.77) |

The embedding rule sends **52.8%** of questions to Luna. Compared with always
using Sol, it saves **$3.42 per 10,000 questions** and loses about **43 correct
answers**. Its savings interval is **44.4–47.8%**. Relative to always using
Luna, it recovers about **80–83 correct answers per 10,000**, for roughly
**$3.48** more. The intervals describe this fixed comparison, not a promise
about future traffic.

The percentage saving is larger than the Qwen result, but the absolute saving
on these short answers is similar: **$3.42 versus $3.21 per 10,000** at the
two studies' secondary embedding settings.

The report also compares each router with sending the same fraction of
questions to Luna at random. All six routing rules have a better quality
point estimate, but every uncertainty interval for that advantage includes
zero. This transfer study therefore does not establish that the routers
select questions better than that random comparator. The
[main Qwen study](methodology.md#selection-check) remains the teaching example
for learning useful selection from answer outcomes.

## What the prices include

We price recorded token usage at the saved Standard API tariffs. Sol's input
and output rates were **$2 and $10 per million tokens**, versus Luna's
**$0.10 and $0.50**. Reasoning is already included in the output-token count.
These are usage-based estimates, not an invoice. See the
[OpenAI pricing documentation](https://developers.openai.com/api/docs/pricing).

Embedding totals include the original SIE batch embedding charge, about
**$0.059 per 10,000 questions**, as an estimate of routing overhead. Laya's
hosting remains unpriced. New answer lengths, cache usage, embedding batch
sizes, and hosting choices can change deployment costs.

One interrupted Luna call has no recorded response. It remains in the
3,000-question population, with its unknown charge bounded rather than
treated as free. Every primary and secondary router selected Sol for that
question, so all router rows above have complete selected answers and costs.
The always-Luna ranges cover its possible answer and charge. The evidence
preserves the original incomplete report and the separately declared analysis
that counts the unavailable answer as a service failure.

## Reproduce the comparison on CPU

No OpenAI account, API key, GPU, or new model calls are needed. Install Python
3.12 and `uv`, then run this from `examples/train-a-router`:

```sh
# Download the pinned answer records, evaluator, and replay instructions.
uvx --from huggingface-hub==0.36.2 hf download \
  superlinked/sie-task-evidence --repo-type dataset \
  --revision c72dacc8c54e3024198d6f25cb312e8de7eecac1 \
  --include 'evaluations/2026-10-20/train-a-router-sol-luna/*' \
  --local-dir run-output/sol-luna
```

```sh
# Enter the downloaded study; its README contains the complete CPU replay commands.
cd run-output/sol-luna/evaluations/2026-10-20/train-a-router-sol-luna
```

Follow the downloaded `README.md`, or open the
[pinned replay instructions](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/c72dacc8c54e3024198d6f25cb312e8de7eecac1/evaluations/2026-10-20/train-a-router-sol-luna/README.md).
The approximately 24 MB bundle includes readable answer records and token usage, frozen rules,
all four targets, grading code, cost calculations, and file hashes. Unneeded
provider metadata has been removed; an independent rerun confirmed identical
answers, grades, costs, and uncertainty intervals.

**Your task:** decide whether always-Luna or one of the routing rules best
matches your quality budget, then explain its cost and serving overhead.
Return to the [SIE serving step](../README.md#serve) for the executable
application using the Qwen pair.
