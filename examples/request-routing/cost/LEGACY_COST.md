# Train and use a cost router

**Latest research:** [learn a difficulty policy from feedback](./DIFFICULTY_POLICY.md)
reports 71.7% → 89.6% held-out accuracy with Qwen4B embeddings. It is a separate,
independently labeled routing task; the historical answer-cost exercise follows.

Start with the [ODSC West one-pager](../WORKSHOP.md). The core exercise stays on
one task: choose a cheap or strong LLM for each request, then improve that
choice with graded feedback. It works from this folder alone with public
locked dependencies and hash-verified Hugging Face evidence.

```sh
uv sync --frozen
uv run --frozen python fetch.py --reranker
uv run --frozen python learn.py --output run-output/my-router
uv run --frozen python run.py --policy run-output/my-router/policy-openai.json --item math500/test/prealgebra/505.json --recorded
```

No key, GPU or GEPA installation is needed for these commands. The download is
about 11 MB; training and evaluation run locally in seconds. Output directories
must be new. The original recorded comparison remains available through
`python compare.py`; the learning exercise below is a separate experiment with
explicitly smaller training banks.

## One task, two fixed routers, one learned update

The embedding baseline embeds each request with `BAAI/bge-m3`, finds its nearest
20 graded examples in a frozen bank of 64, and predicts whether the cheap model
is sufficient. The fixed `Qwen/Qwen3-Reranker-4B` baseline scores the request
against the original Q1 description of work suitable for a small cheap model.
Both baselines use the same 64 initial examples to set their routing cutoff.

The update uses 437 graded training requests to fit ridge regression on the same
normalized embeddings. Its target is `1 + quality_cheap - quality_strong`.
This signed difference preserves cases where the cheap model does better.
The encoder weights stay frozen; only a small linear predictor is trained.
At inference, its entire learned computation is `embedding @ weights + bias`.
It needs one embedding call, regardless of training-bank size. A score is not a
calibrated probability and can fall outside zero to one.

| Mode | What changes with feedback? | Inference |
| --- | --- | --- |
| Static embedding kNN | Nothing; bank and threshold frozen | One embedding + local neighbor lookup |
| Learned embedding router | Fit a predictor from paired grades | One embedding + local dot product |
| Static reranker | Nothing; description and threshold frozen | One reranker call |
| GEPA reranker extension | Evolve the description from feedback | One reranker call with the learned description |

The core measured learning result is the embedding predictor. The [GEPA lab](./GEPA.md)
now includes `difficulty.py` for prompt learning on this same model-routing task,
with a calibration-only control. Its completed self-hosted run changes test cost
from $3.079 to $3.051/1,000 and quality from 94.63% to 95.03%; uncertainty includes
no improvement. Replay it with `fetch.py --difficulty` and
`replay_difficulty.py --run evidence/difficulty`. The Console path also passed a
separate score/generation smoke check. Earlier cost-router experiments and the
separate handler exercise remain documented there.

## Results on the same 728 test requests

Costs include routing and downstream generation, in dollars per 1,000 requests.
Quality is relative to always using the strong model on those same requests.

| Model pair / router | Routing | Generation | Total | Quality retained |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Luna / GPT-6.1 Sol: frozen kNN 64 | $0.0132 | $3.6411 | $3.6543 | 96.25% |
| Same pair: learned predictor 437 | $0.0132 | $3.0318 | $3.0450 | 96.74% |
| Same pair: fixed reranker Q1 | $0.0090 | $2.9990 | $3.0080 | 94.06% |
| Haiku 5.5 / Sonnet 5.5: frozen kNN 64 | $0.0132 | $8.2777 | $8.2909 | 97.27% |
| Same pair: learned predictor 437 | $0.0132 | $8.2700 | $8.2832 | 97.51% |
| Same pair: fixed reranker Q1 | $0.0090 | $6.4067 | $6.4157 | 93.65% |

For the OpenAI pair, the learned update costs **16.67% less** than frozen kNN,
with a paired 95% interval of **10.71–22.68%**. The retained-quality difference
is **+0.49 percentage points**, interval **−1.15 to +2.12**: similar point estimates,
not proof of a quality gain or a formal non-inferiority result. The update saves
41.38% versus always using the strong model. The Anthropic update saves only
0.09% versus frozen kNN, interval −5.24 to +5.31%; no clear benefit for that pair.

These estimates reuse the public benchmark, not a new independent evaluation
population. The method was developed after earlier results were known. The
four-penalty selection uses validation only; test results do not choose a
penalty or cutoff. Answer costs are the original study's dated list prices;
router costs are archived embedding receipts and reranker content-token prices.
Training/feedback acquisition is separate from per-request serving cost.

For context, the original full 729-example kNN study reports $3.19 and 95.8%
quality for the OpenAI pair on these same test requests. The 16.7% improvement
above is specifically against the correctly frozen 64-example starting policy,
not against that larger reference bank.

## Exact training and selection procedure

The original 729 calibration requests retain their earlier deterministic,
stratified split: 437 train, 146 validation and 146 former threshold-calibration
requests. The latter 146 are unused here. A fixed hash-balanced 64-example
subset of training forms the frozen kNN baseline.

For each model pair:

1. Fit the baseline cutoff using leave-one-out predictions on its 64 examples.
2. Fit four ridge predictors on 437 examples with penalties 0.01,0.1,1 and 10.
   An unpenalized intercept and exact leave-one-out predictions fit each cutoff
   to a 98% training-quality target. Unit tests compare this analytic calculation
   with refitting every fold.
3. Select the cheapest candidate on 146 validation requests that costs less than
   the baseline, retains at least 98% of strong quality, and does not reduce
   baseline validation quality. Otherwise keep the baseline.
4. Save selection, weights and threshold before evaluating 728 test requests.
   **Do not recalibrate after validation.**

The selected penalties are 10 for OpenAI and 0.01 for Anthropic. The 98% validation
rule is not a test guarantee; measured test quality is shown above. `plan.json`,
`selection.json`, `results.json`, and `policy-*.json` make the choices reviewable.
Paired stratified bootstrap uses 4,000 resamples with seed 61.

## Use the learned policy with SIE

Create an account and API key at [SIE Console](https://console.superlinked.com).
Set `SIE_API_KEY` and leave `SIE_BASE_URL` unset for Serverless:

```sh
uv run --frozen python run.py --policy run-output/my-router/policy-openai.json "Solve x + 2 = 5"
uv run --frozen python run.py --router reranker --policy run-output/my-router/policy-reranker-openai.json "Solve x + 2 = 5"
```

Each command makes one inference send, with a 60-second timeout and automatic
retries disabled. It prints the selected LLM; no OpenAI or Anthropic key is
required and no answer is generated. `--answer` is a separate optional OpenAI
call billed by OpenAI. `--pair anthropic` selects the other pair; use its
matching policy file.

For an existing local SIE server, set `SIE_BASE_URL=http://localhost:8080`.
The embedding path uses BAAI/bge-m3. Qwen reranking needs its native yes/no
scoring adapter; the current CPU/MPS generic CrossEncoder fallback is not
equivalent. Use cloud or compatible CUDA serving for fresh reranker scores.
Offline replay works on ordinary laptops. A model/profile change requires
revalidation and calibration.

## Why the earlier growing bank cost more

The earlier pilot's embedding charge was identical: **$0.01324/1,000**.
Its downstream LLM bill rose from $2.36379 to $2.72893/1,000. Of 728 requests,
23 switched cheap→strong and seven switched strong→cheap; the strong-model
count rose 80→96. Chat requests accounted for most of the extra spend.
The new examples changed model choices, not the price of computing embeddings.
`python learning.py` now prints that cost decomposition explicitly.

That pilot also recalibrated thresholds after validation, changing the policy
being tested. Its historical evidence and replay remain intact in
[EXPERIMENTS.md](./EXPERIMENTS.md); the current exercise freezes the validated
policy. Consequently its frozen baseline numbers differ from the earlier pilot.
The successful new update changes both the amount of feedback and the learning
rule; it is not a claim that adding neighbors alone produced the gain.

## Data and verification

For the development-only oracle bound, matched-quality comparisons and feedback
learning curves, see [HEADROOM.md](./HEADROOM.md). It distinguishes ranking
potential from cutoffs fitted on training and introduces no new test claim.

The [base study](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/95d34b775b61bf1d584cada099bfed053710fc2b/evaluations/2026-10-11/request-routing-cost-results-v1)
contains 1,457 requests from SuperGPQA, MATH-500, MBPP+ and Arena-Hard, both
models' grades/costs, vectors and routing receipts. `study.py` pins the revision
and manifest hash. `reranker.py` separately pins the Q1/Q3 reranker addendum.
No private checkout or credential is needed to download either.

The [frozen learned-router run](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/9de9ff567785f063b67ae895bbded42102f568cc/evaluations/2026-10-16/request-routing-learned-router-v1)
contains the selected policies, split and selection records, results, source
snapshot, and live SIE receipt. Its manifest SHA-256 is
`6efafa39dcce5b65b5160a42a6dfeb4b32265a383ca3c61e62877ab1f9399213`.
One live embedding call cost one credit and reproduced the archived policy
score and route exactly. `learn.py` regenerates the policies locally from the
pinned base evidence; this additional artifact is for inspection.

```sh
uv run --frozen python score.py
uv run --frozen python compare.py
uv run --frozen --group gepa python -m unittest discover -s tests -v
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

The optional GEPA group enables its integration tests; it is not needed to
train or serve the embedding predictor. Keep new traffic labels for training,
select on separate validation data, and collect a fresh temporal test sample
before drawing conclusions about your own application.
