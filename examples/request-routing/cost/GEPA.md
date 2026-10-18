# GEPA: learn which requests need the stronger model

The [main workshop](../WORKSHOP.md) improves cheap-versus-strong model routing
with an embedding predictor. `difficulty.py` applies GEPA to **the same task and
the same dataset**: evolve the reranker's description from paired model grades
while leaving its weights fixed. Here, difficulty means that the stronger model
improves the answer enough to justify its cost; a difficult question on which
both models fail offers little reason to escalate.

Both inference roles use SIE: Qwen3-Reranker-4B scores requests;
Qwen3.8-27B-FP8 proposes prompt revisions. The Console path needs no second provider key.
The tested endpoint did not expose GLM-5.3-Flash by its canonical ID.
The completed experiment used isolated self-hosted SIE workers. A separate
Console smoke check successfully scored and generated using these same two
model IDs, spending two billed credits.

## Run the difficulty-routing experiment

Replay the completed result on a CPU laptop, without a key or GEPA:

```sh
uv sync --frozen
uv run --frozen python fetch.py --difficulty
uv run --frozen python replay_difficulty.py --run evidence/difficulty
```

From this folder, fetch the public study and set `SIE_API_KEY` from the Console:

```sh
uv sync --frozen --group gepa
uv run --frozen python fetch.py --reranker
uv run --frozen --group gepa python difficulty.py --output run-output/my-difficulty
uv run --frozen python replay_difficulty.py --run run-output/my-difficulty
uv run --frozen python run.py --router reranker --policy run-output/my-difficulty/router.json "Explain why the sky is blue"
```

The fresh optimization uses SIE credits; replay requires neither a key nor GEPA.
The final command makes one reranker call and prints the chosen downstream LLM.
Its policy is the exact description and cutoff evaluated by the experiment.

The comparison has three arms:

| Arm | Description | Cutoff fitted on |
| --- | --- | --- |
| Static | Original Q1 description | Initial 64 examples |
| Calibration-only control | Same Q1 description | All 437 training examples |
| GEPA | Description evolved from training feedback | All 437 training examples |

The 437 train / 146 validation / 728 test split matches `learn.py`; the former
146 threshold-calibration rows stay unused. Each candidate fits a cutoff on
training data to retain 98% of strong-model quality. GEPA sees aggregate training
grades and examples of unnecessary escalation and harmful cheap routing.
It receives no validation or test requests in reflection prompts.

All proposals, including those rejected by GEPA's minibatch gate, are checked on
validation. Select the cheapest proposal that beats the initial router's cost
and retains at least `max(98%, initial validation quality)`. Otherwise keep the
initial policy. Save the description **and cutoff** before scoring test requests;
there is no post-selection recalibration. The calibration-only arm distinguishes
the effect of new cutoff data from the effect of changing the description.

The output includes `plan.json`, every proposed description and reflection,
`selection.json`, per-request scores, billed-call receipts, `results.json`, and
the deployable `router.json`. The offline verifier recomputes selection, both
controls, test metrics, paired bootstrap intervals, and credit totals. It rejects
inconsistent recordings. Quality intervals and historical-price limitations
apply to this public benchmark, as in the embedding exercise.

Default limits are 16 reflections, 16,000 scored request/description pairs,
30,000 observed credits and 30 minutes. `--max-reflections`, `--max-score-items`,
`--max-credits` and `--max-seconds` lower these limits. The credit limit is checked
before each call; one call can cross it. Automatic retries are disabled. A failed
run retains its receipts and `incomplete.json`; a new attempt needs a new output
directory. Failed sends without receipts have unknown billing.

### Use your own SIE servers

The same experiment accepts self-hosted SIE endpoints. Serve the reranker's
native `default` CUDA profile and the reflection model's CUDA 13 SGLang profile;
these require separate compatible runtime environments. Set `SIE_BASE_URL` and,
if authentication is enabled, `SIE_API_KEY`. For a separate reflection endpoint,
also set `SIE_REFLECTION_BASE_URL` and `SIE_REFLECTION_API_KEY`.

Pass `--self-hosted-usd-per-token RATE` to `difficulty.py`, where `RATE` is an
explicit historical USD price per reranker **content input token**. The pinned
study's price is in `evidence/reranker/code/prices.json`. This mode records token
usage, estimates routing cost at that price, and reports Console credits as
not applicable. GPU rental and optimization spend are separate from the
per-request routing estimate. All data splits, selection rules and offline
replay checks remain the same. Do not compare these estimates with a billed
cloud run without accounting for the different cost basis.

`serve_worker.py` is an optional operator helper for an owned Modal GPU
function. It starts an authenticated SIE server and a temporary HTTPS tunnel;
it requires a Modal Secret containing `SIE_AUTH_TOKEN`. The allocation owner
must set a hard function timeout and stop the exact owned app after use.

## Completed result

The 16-reflection run took **12.4 minutes** after worker warmup, with 385 successful
calls and 11,367 scored request/description pairs. Candidate 9 passed validation:
$2.554 → $2.407/1,000 at the same 99.115% retained quality. Its description and
cutoff were saved before evaluating the 728 test requests:

| Policy | Routing / 1,000 | Generation / 1,000 | Total / 1,000 | Quality retained |
| --- | ---: | ---: | ---: | ---: |
| Original description, cutoff on 64 | $0.0090 | $3.0698 | $3.0788 | 94.63% |
| Original description, cutoff on 437 | $0.0090 | $3.5504 | $3.5594 | 96.74% |
| GEPA description, cutoff on 437 | $0.0132 | $3.0383 | $3.0515 | 95.03% |

Against the original policy, cost falls **0.89%** (95% interval −4.92 to +5.95%)
and quality rises **0.41 percentage points** (−1.16 to +2.07). This does not
establish a material improvement. Against recalibration alone, cost falls
**14.27%** (10.14–19.15%), with **1.71 percentage points lower quality**
(−3.48 to 0.00). That is a cost/quality tradeoff, not a constant-quality saving.
The embedding exercise remains the workshop's stronger measured improvement.

All three policies were scored in the same self-hosted run. Its new batched
seed scores produce a different cutoff from the earlier archived comparison;
use this run's $3.0788 baseline when assessing GEPA. The longer evolved query
increases the reranker's own token cost slightly. These are measured-token
estimates at the pinned historical tariff, plus archived answer prices;
GPU rental and optimizer spend are excluded. They are not Console receipts.

A four-reflection preparatory pilot produced descriptions of the strong class
although positive matches select cheap. It was stopped before selection or any
test call. The completed run makes that polarity explicit and adds
`cheap_is_as_good` to training feedback. The public evidence retains the pilot,
source snapshots, dependency lock, runtime versions, and every reflection.
Live reruns may differ; the saved result replays deterministically.

[Published evidence and source snapshots](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/dc85e8a7e4d70c53b4680febdda8fab9e66d9c0a/evaluations/2026-10-17/request-routing-gepa-difficulty-v2)
are pinned by revision and manifest SHA-256 in `difficulty_evidence.py`.

Use the recorded policy through your SIE endpoint:

```sh
uv run --frozen python run.py --router reranker --policy evidence/difficulty/router.json "Solve for x: x/2 + x/3 = 5"
```

## Earlier handler accuracy exercise

This is a separate task: choose one of eight calendar, meeting, reminder and
to-do handlers. With 80 train / 160 validation / 240 test CLINC150 requests, GEPA
raised test accuracy **83.33% → 87.50%**. There were 19 corrections and 9 regressions;
exact paired p=0.087. The run took 5.8 minutes and 4,039 credits. It demonstrates
an observed accuracy gain, not a cheap-versus-strong cost saving.

```sh
uv run --frozen python fetch.py --intents
uv run --frozen python replay_intents.py
# Set SIE_API_KEY before running a fresh optimization:
uv run --frozen --group gepa python intent_gepa.py --output run-output/my-intents
uv run --frozen python replay_intents.py --run run-output/my-intents
```

The learned descriptions are in `handlers.json`. Model weights are not trained.
Replay is offline; live optimization spends credits. Limits, splits, provenance,
uncertainty and public evidence are in the [recorded method](./EXPERIMENTS.md#gepa-improves-handler-routing).

## Earlier cost-router research

The earlier `optimize.py` pilot and `improve.py` diagnostic remain reproducible.
Neither established a cost-routing gain. They are research extensions rather
than the core workshop's improvement claim. `improve.py` freezes both prompt and
threshold before audit; the older pilot is preserved as historical evidence.

```sh
uv run --frozen python fetch.py --learning --intents
uv run --frozen python replay_learning.py
uv run --frozen python replay_improvement.py
```

See [full experimental records](./EXPERIMENTS.md) for those commands and budgets.
The current `difficulty.py` experiment supplies the matched frozen-policy
comparison; newly collected traffic would still be needed for an independent
evaluation population. The handler result is not substituted for that evidence.
