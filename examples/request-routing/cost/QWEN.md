# Qwen3 embedding development comparison

The development lead has now been checked against 728 held-out answer outcomes.
The Haiku/Sonnet cost reduction was 15.18%, below the predeclared 25% gate;
the OpenAI pair did not improve. See [the confirmation and Qwen27B teacher pilot](CONFIRMATION.md)
before using the development numbers below as a workshop claim.

The original study used BGE-M3. This follow-up encodes the same 437 training
and 146 validation requests with production `Qwen/Qwen3-Embedding-4B`
(revision `5cf2132abc99cad020ac570b19d031efec650f2b`, 2,560 dimensions).
It does not collect embeddings or evaluate outcomes for the 728 test requests.

[Published recordings, receipts, results and source](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/a5ca52b9e4d7218202fa65bf956c7fdc57186dc4/evaluations/2026-10-17/request-routing-qwen4b-v1)
contain both arms. All 62 files were anonymously downloaded and hash-verified;
manifest SHA-256: `af996f1f17b941eab3121dba4c4e7fd354253feab1afaae8791881e3c6a0d3e0`.

Two development runs are retained: plain document embeddings, then a fixed
task instruction applied to both training-bank and validation requests:

> Represent the user request for classifying whether a small language model can answer it as well as a more capable model.

The second run uses `is_query=True` and SIE's native query template. Qwen's
[model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B) recommends task-specific
instructions. This was a development follow-up after inspecting the plain arm,
not an independently held-out confirmation or a GEPA-optimized instruction.

## Result worth following up

On the Haiku/Sonnet pair, with cutoffs fitted on training and applied unchanged
to validation, the instruction-aware Qwen kNN learning curve is:

| Training examples | Mean cost / 1,000 | Retained quality |
| --- | ---: | ---: |
| 32 | $8.968 | 99.03% |
| 64 | $8.417 | 98.69% |
| 128 | $8.197 | 99.41% |
| 256 | $8.384 | 99.96% |
| 437 | $5.765 | 98.73% |

The 64-to-437 change is **31.51% lower cost** at similar measured quality.
Rows average ten nested, stratum-balanced feedback orders on the same 146
validation requests. All orders converge to the same 437-example bank. They
are correlated development comparisons, not ten independent tests. The original
seed's 64-example policy sends everything to strong; the average avoids choosing
that especially weak starting point for the headline.

At 437 examples, the BGE kNN policy costs $6.653 at 97.46% quality. Qwen with
the task instruction costs $5.765 at 98.73%. The fixed reranker recalibrated on
437 costs $7.624 at 98.31%. These are development point estimates, not a newly
validated production policy. OpenAI-pair results and plain Qwen results are
mixed; larger embeddings are not established as universally better for routing.

Both encoders use the same archived BGE routing-cost allowance, isolating the
effect of their vectors on downstream decisions. These totals are **not a Qwen
serving-price claim**. Actual collection took 37 calls and 678 credits for the
plain arm, plus 37 calls and 780 credits for the instructed arm (about 17 seconds
each after warmup). One earlier two-item smoke call cost one credit. Production
receipts use rate book `2026-10-16-production-bootstrap-v38`.

## Reproduce or collect

`encoder_comparison.py` collects only development embeddings, records each batch
receipt once, and compares the two encoders with identical feedback orders,
training labels, ridge choices and cutoff rules. It does not change `learn.py`'s
published BGE policy or introduce a new test result.

```sh
uv sync --frozen
uv run --frozen python fetch.py --difficulty
# Set SIE_API_KEY from Console; a new directory is required.
uv run --frozen python encoder_comparison.py --collect --run run-output/qwen4b \
  --difficulty evidence/difficulty \
  --instruction 'Represent the user request for classifying whether a small language model can answer it as well as a more capable model.'
```

Collection limits are 2,000 observed credits and 20 minutes, checked before
each batch; one batch can cross a limit. Retries are disabled. Do not restart
an incomplete run automatically. To replay downloaded recordings, pass their
directory with `--run`, omit `--collect`, and use `--output run-output/replayed`
to write new results while preserving the recording.

Haiku/Sonnet here are the archived answer-model pair, not label generators.
For a separate difficulty-classification workshop, a strong SIE-hosted teacher
can supply training labels; its agreement with reviewed evaluation labels is a
different claim from preserving downstream answer quality. The fresh production
GLM-5.3-Flash chat probe returned `404 model_not_found`; no GLM labels were produced.
