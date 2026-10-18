# Prior comparisons and GEPA experiments

Historical records and reproducible research extensions. Start with the current
[core workshop](../WORKSHOP.md) for the corrected learning procedure.
The newer matched difficulty-routing experiment, its self-hosted result,
calibration-only control, and public replay are in the [GEPA lab](./GEPA.md#completed-result).

For the ODSC West session, use the [one-page workshop instructions](../WORKSHOP.md).
Everything needed is in this folder, including the locked dependencies;
larger datasets download from pinned public Hugging Face revisions. No
`sie-web` checkout, GPU or provider account is needed for the offline lab.

## Compare embedding kNN and a reranker

```sh
uv sync --frozen
uv run --frozen python fetch.py --reranker
uv run --frozen python compare.py
```

This adds the public [decision-router addendum](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/bdf9f45218b292b60512fa404595b17abc1cacd1/evaluations/2026-10-13/request-routing-decision-routers-v1/results)
to `evidence/reranker/`, about 1.5 MB beyond the base study. Both manifests
are pinned by SHA-256. Every downloaded file is checked before use.

`compare.py` recalculates kNN scores from embeddings and reads Qwen's recorded
scores, aligns both by request ID, chooses each threshold on the same 729
calibration requests, and recomputes quality and cost on the same 728 held-out
requests. It fails if thresholds or point estimates differ from the studies.
It does not regenerate answers or confidence intervals.

| Pair | Router | Cost / 1,000 | Saving | Quality retained |
| --- | --- | ---: | ---: | ---: |
| OpenAI | bge-m3 kNN20 | $3.19 | 38.6% | 95.8% |
| OpenAI | Qwen reranker Q1 | $3.31 | 36.3% | 95.9% |
| OpenAI | Qwen reranker Q3 | $3.17 | 38.9% | 95.9% |
| Anthropic | bge-m3 kNN20 | $9.15 | 11.7% | 99.0% |
| Anthropic | Qwen reranker Q1 | $8.02 | 22.7% | 97.1% |
| Anthropic | Qwen reranker Q3 | $7.79 | 24.9% | 97.6% |

The reranker scores the request against a description of work suitable for a
cheap model. It needs no nearest-neighbour bank at inference time. Q1 asks
whether a small, cheap model can answer correctly. Q3 asks for a short,
checkable answer and was written knowing the first study's results. Q3 is
exploratory; neither table proves one router is better. The current website
uses Q1; presentation notes also discuss Q3. Exact descriptions and the shared
instruction come from the downloaded `code/specs.json`.

All routed costs include routing. kNN uses the study's billed credits; the
reranker uses its dated price per content token. They are historical cost
estimates, not a quote for today's endpoint. Quality is relative to the strong
model's score, not absolute accuracy. The 98% calibration target is not a
held-out guarantee. No latency from a different GPU/profile is mixed into this
comparison.

### Try the same request offline or with console credits

```sh
uv run --frozen python run.py --router knn --item math500/test/prealgebra/505.json --recorded
uv run --frozen python run.py --router reranker --item math500/test/prealgebra/505.json --recorded
uv run --frozen python run.py --router reranker --description q3 --item math500/test/prealgebra/505.json --recorded
```

Create an account at [SIE Console](https://console.superlinked.com), check your
starter-credit balance, and create an API key. Set `SIE_API_KEY` locally and
remove `--recorded` to make one fresh routing call. `SIE_BASE_URL` must be unset
to select Serverless automatically. You can also pass your own request:

```sh
uv run --frozen python run.py --router reranker "Solve for x: x/2 + x/3 = 5"
uv run --frozen python run.py --router knn "Solve for x: x/2 + x/3 = 5"
```

Each command permits one inference send with a 60-second timeout and no
automatic retry. A cold model or capacity failure can therefore require
rerunning the command. Routing stops at choosing a model; the workshop does
not need paid OpenAI/Anthropic keys. `--answer` is a separate, optional OpenAI
call billed by OpenAI, outside SIE credits.

For an existing local SIE server, set `SIE_BASE_URL=http://localhost:8080` and
use the same commands. Local kNN needs `BAAI/bge-m3`; the reranker needs
`Qwen/Qwen3-Reranker-4B` with its Qwen yes/no scoring adapter. The current
CPU/MPS generic CrossEncoder fallback is not equivalent, so use Serverless
or a compatible CUDA server for that arm. Offline replay runs on CPU laptops.
The example requests the reranker's `default` profile. New hardware, profiles,
models or prompts require fresh calibration; `--threshold` accepts your
calibrated cutoff. A live call can differ from the archived score.

## Static and feedback-updated modes

| Router | Static | Feedback-updated challenger |
| --- | --- | --- |
| Embedding kNN | Freeze a 64-example bank | Add 373 graded examples; recalibrate |
| Qwen reranker | Freeze the Q1 description | GEPA revises the description using graded examples; recalibrate |

Both are feedback-driven: collecting requests alone does not teach either
router which model is good enough. Each added example needs both models'
quality grades. The embedding variant updates its example bank without
training model weights; the reranker variant updates its prompt without
training model weights. Updates must pass validation before promotion.

```sh
uv run --frozen python learning.py
uv run --frozen python learning.py --pair anthropic
```

This offline experiment starts with 64 examples and simulates receiving the
remaining training examples. It compares the frozen and expanded banks, shows
the promotion decision, and evaluates both on the final test set. It is a new
small-data experiment, separate from the original 729-example results above.

## Run GEPA entirely through SIE Cloud

For a completed run that improved accuracy, start with
[handler routing](#gepa-improves-handler-routing) below. The cost-router
experiment in this section is retained as a useful negative result.

Set `SIE_API_KEY` from the [SIE Console](https://console.superlinked.com).
Both the reranker and GEPA's reflection model run on SIE; no OpenAI or Anthropic
account is needed. Your laptop runs the optimizer and the scoring calculations.

```sh
uv sync --frozen --group gepa
uv run --frozen --group gepa python optimize.py --output run-output/my-gepa
uv run --frozen python replay_learning.py --run run-output/my-gepa
uv run --frozen python run.py --router reranker --policy run-output/my-gepa/router.json "Solve x + 2 = 5"
```

The optional dependency is pinned to `gepa==0.1.4`. The custom
[GEPA adapter](https://github.com/gepa-ai/gepa) calls native SIE scoring with
`Qwen/Qwen3-Reranker-4B` and asks `Qwen/Qwen3.8-27B-FP8` to revise only the
routing description. The scoring instruction stays fixed. GEPA receives
request text, routing decisions, both models' archived grades and costs as
feedback; the reranker itself receives only the request and routing prompt.
The workshop reuses graded answers instead of paying to generate them again.

The original 729 calibration requests are split deterministically within task
type into 437 training, 146 validation and 146 final threshold-calibration
requests. The initial 64 training examples fit provisional thresholds; the
remaining 373 are available for GEPA reflection. Training utility is cost
saving minus a quality-loss penalty. Validation chooses among accepted GEPA
candidates. A candidate must cost less while retaining at least 98% of strong
quality **and** matching the seed's validation quality. Otherwise retain the
seed. Then freeze selection, recalibrate on the reserved 146, and evaluate on
the original 728 test requests. Test results never select a candidate.

`plan.json` records the split, objective and budgets before scoring;
`selection.json` records the decision before final calibration/test.
`router.json` contains the selected prompt and cutoff, which may still be the
seed. `calls.jsonl` records billed credits and errors; `scores.jsonl` enables
an offline replay of validation, selection, thresholds and results. Outputs
live under ignored `run-output/`; choose a fresh directory for each run.

Defaults permit six reflection calls, 5,000 reranker document scores, 1,200
GEPA metric evaluations, 20,000 **observed** credits and 30 minutes. All apply
together. Requests are sequential, reranking batches up to 32 documents.
Scoring has a 60-second timeout; reflection has 120 seconds and a 1,200-token
output cap. Limits are checked before calls, so a final call can exceed a time
or observed-credit limit. A timeout may have been billed without a receipt.
The failure latch prevents both SDK and GEPA from resending; failures stop the
run and retain receipts. There is no automatic resume of an uncertain run.
Use `--max-reflections`, `--max-score-items`, `--max-credits` and `--max-seconds`
to tighten the budget. `--reflection-model` selects another available SIE model.

A rejected update is a valid result. The small validation/calibration sets can
produce unstable cutoffs, and a 98% calibration target is not a test guarantee.
These are workshop experiments, not production-quality certification. For
ongoing learning, collect new paired grades, keep similar request templates
together, maintain a separate temporal validation sample, and repeat this
promotion process. Refresh the final test set before using its results to
change the method. Serving-model/profile changes also require recalibration.

`prepare_gepa.py` remains available to export the three calibration-data
partitions for another optimizer; it excludes the final test requests and
makes no API calls.

### Replay the completed cloud experiment

```sh
uv run --frozen python fetch.py --learning
uv run --frozen python replay_learning.py
```

The [published workshop evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/dfe4f4b19c4900c70518e4b4d58768cd5b769582/evaluations/2026-10-16/request-routing-gepa-workshop-v1)
is revision- and hash-pinned, about 0.5 MB beyond the other evidence. Replay
checks the split, promotion decision, thresholds, final metrics and credit
receipts offline. The full archive also preserves the earlier development
pilot and source snapshot. No generated data needs to be committed to Git.

On October 16, 2026, the corrected run completed in **117 seconds**, with
**75 successful calls, six prompt proposals, 1,832 document/prompt scores and
2,189 credits** (about $0.022 at $0.00001/credit). Both inference roles ran on
SIE. There were no failed calls or retries in this run. The earlier pilot
exposed GEPA's callback retry after a timeout; the failure latch fixes that.
Its separate ledger records 2,704 observed credits and one call with unknown
billing. Two initial capability probes used another four credits.

Neither update earned promotion: the embedding expansion raised validation
quality but cost more; GEPA's validation candidates sacrificed too much quality.
Both update modes therefore retained their static baselines. On 728 test requests:

| Router / candidate | Cost / 1,000 | Saving | Quality retained |
| --- | ---: | ---: | ---: |
| Static kNN, 64 examples | $2.38 | 54.2% | 91.7% |
| Expanded kNN, 437 examples, rejected | $2.74 | 47.2% | 93.6% |
| Static reranker / retained after GEPA | $3.03 | 41.7% | 94.2% |

These smaller-calibration results miss the 98% target. They demonstrate the
feedback loop and why held-out evaluation matters; they do not establish a
winning router or an independent replication of the original study. Serving
model names/profile were fixed, but the cloud API did not pin weight revisions.
Prompt generation is stochastic, so a fresh run can produce different candidates.

## GEPA improves handler routing

This separate exercise uses the eight handlers already in `../direct/run.py`:
calendar, calendar updates, meeting schedules, meeting booking, reminders,
reminder updates, to-do lists and to-do updates. Qwen3-Reranker-4B scores each
request against each description; the highest score chooses the handler.
Ties count as incorrect. The unchanged original short descriptions form the
static baseline. This measures intent accuracy, not cheap-versus-strong cost.

```sh
uv run --frozen python fetch.py --intents
uv run --frozen python replay_intents.py
# Set SIE_API_KEY to run a new experiment; use a fresh output directory.
uv run --frozen --group gepa python intent_gepa.py --output run-output/my-intents
uv run --frozen python replay_intents.py --run run-output/my-intents
```

The data is a subset of [CLINC150](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de),
Larson et al. (2019), under [CC BY 3.0](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE).
We select ten training requests per handler in fixed hash order, then use all
20 validation and 30 test requests per handler from the official splits:
**80 train / 160 validation / 240 test**. There is no normalized exact-text
overlap across these splits. `intent_gepa.py --prepare-source` can rebuild the
subset from the hash-pinned official `data_full.json`. The downloadable archive
includes the source revision, license, subset, prompts, scores and receipts.
The [published run and cost diagnostic](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/54bda087d98fa2bd7f1883f194cd534276a904b2/evaluations/2026-10-16/request-routing-gepa-improvement-v1)
are pinned by revision and manifest hash in `intent_evidence.py`.

GEPA 0.1.4 uses Qwen3.8-27B-FP8 on SIE for reflection. It revises one handler
description at a time, receiving training errors and competing descriptions.
The scoring instruction stays fixed. Search utility is accuracy plus 0.05 times
the true-handler score margin. Final selection evaluates every proposal on
validation and maximizes accuracy, breaking ties by shorter total descriptions,
then earlier proposal. It keeps the seed unless accuracy strictly improves.
Selection is written before any test scoring. No test results choose a prompt.

| Split | Static descriptions | GEPA-selected descriptions |
| --- | ---: | ---: |
| Validation, 160 | 76.25% | 86.875% |
| Test, 240 | 83.33% (200 correct) | 87.50% (210 correct) |

The October 16 run made 16 reflection calls and 8,128 request/description
scores: **296 successful calls, 349.845 seconds and 4,039 billed credits**.
There were no failed calls. An earlier validation-only baseline check used
180 additional credits and did not inspect test results. GEPA selected four
longer descriptions that distinguish calendar/meeting actions; the reminder
and to-do descriptions remained unchanged. `handlers.json` contains the result.

The paired gain is **4.17 percentage points**, with 19 corrections and nine
regressions. Replay reports a stratified percentile bootstrap 95% interval of
**+0.42 to +8.33 points** (4,000 resamples, seed 53). The exact two-sided McNemar
test gives **p=0.087**; this small single run is not conclusive evidence of a
general improvement. This task was chosen after cost routing failed to show
a clear gain. The handlers and official test split were not chosen from test
results. Public benchmarks can occur in model training; this is no guarantee
for private traffic. Repeated optimizer seeds and a new temporal test set would
strengthen the result. Do not tune against these now-published test outcomes.

Live defaults allow 16 reflections, 20,000 document scores, 20,000 observed
credits and 30 minutes. Limits share the failure/receipt protections described
above. `--max-reflections` reduces search and `--reflection-model` selects an
available model. The endpoint/key used here returned `404 model_not_found` for
`zai-org/GLM-5.3-Flash`; no GLM relay was used or verified. Model names/profile
are recorded, but cloud weight revisions are not pinned. Scores for unchanged
descriptions are cached. Eight comparisons are needed per request; this
experiment makes no serving latency or cost advantage claim over classification.

### Stronger cost-routing diagnostic

`improve.py` additionally evolves both the cost description and instruction
using training error summaries. It reserves 192 original calibration requests
that were absent from all earlier GEPA scores, then uses 249 for training,
128 for threshold fitting and 160 for validation. Those audit requests were
public and previously used in embedding banks, so this is not independent new
data. Prompt and threshold are both frozen before audit, without recalibration.

```sh
uv run --frozen python fetch.py --learning --intents
uv run --frozen python replay_improvement.py --run evidence/intents/cost-diagnostic
# Optional fresh cost search, billed to SIE:
uv run --frozen --group gepa python improve.py --output run-output/my-cost-search
```

Validation selected a slightly cheaper candidate with no quality regression.
On audit, cost rose **0.32%** ($1.909 → $1.915 per 1,000), while retained quality
rose **94.91% → 96.41%**. Both paired intervals include zero, and the quality
target still fails. This is not a cost-routing improvement. The search stopped
on its 18th reflection attempt after 17 proposals. Its saved candidates were
evaluated in an explicit completion phase, without resending that failed call
or generating more prompts. Total observed credits were 11,677 plus one call
with unknown billing; the complete ledger is retained. The example now limits
reflection feedback to 12 shorter requests and stops promptly on a latched
failure. The archive preserves the actual prompts from the earlier run.

## What this shows

Cost routing picks a model for each request before you call one. A request
that a cheap model answers as well as a strong one goes to the cheap model;
the rest go to the strong one. The router has to be much cheaper and faster
than the calls it saves.

This router embeds the request with `BAAI/bge-m3` on SIE Serverless and finds
the 20 most similar requests in a bank of 729 whose answers from both models
were graded. It weights each neighbour by its cosine similarity and averages
whether the cheap model held up on it. If that score reaches a threshold, the
request goes to the cheap model. The threshold was chosen on the bank to keep
98% of the strong model's quality at the lowest cost.

On 728 held-out requests, routing between GPT-6 Luna and GPT-6.1 Sol cost
**$3.19 per 1,000 requests, 38.6% less than always using GPT-6.1 Sol at 95.8%
of its quality**. It sent 77.2% of requests to Luna. Between Claude Haiku 5.5
and Claude Sonnet 5.5 it cost 11.7% less than always using Sonnet, at 99.0% of
its quality. The router itself costs $0.013 per 1,000 requests, included in
those figures.

| GPT-6 Luna → GPT-6.1 Sol | Cost per 1,000 requests | Saving vs always-Sol | Quality vs always-Sol | Sent to Luna |
| --- | ---: | ---: | ---: | ---: |
| Always GPT-6.1 Sol | $5.19 | — | 100% | 0% |
| **SIE router** | **$3.19** | **38.6% (33.6–44.2)** | **95.8% (93.5–98.0)** | **77.2%** |
| Random routing at 77.2% to Luna | $1.51 | 71.0% | 85.2% | 77.2% |
| Always GPT-6 Luna | $0.42 | 92.0% | 80.9% | 100% |

| Claude Haiku 5.5 → Claude Sonnet 5.5 | Cost per 1,000 requests | Saving vs always-Sonnet | Quality vs always-Sonnet | Sent to Haiku |
| --- | ---: | ---: | ---: | ---: |
| Always Claude Sonnet 5.5 | $10.36 | — | 100% | 0% |
| **SIE router** | **$9.15** | **11.7% (9.4–14.5)** | **99.0% (97.7–100.3)** | **34.9%** |
| Random routing at 34.9% to Haiku | $6.99 | 32.5% | 93.8% | 34.9% |
| Always Claude Haiku 5.5 | $0.70 | 93.3% | 82.2% | 100% |

Quality is routed quality as a share of always-strong quality on the same
requests. Bracketed ranges are the study's 95% intervals from a stratified
paired bootstrap. Costs are list prices accessed 2026-10-11, with no caching or
batch discount.

The OpenAI pair did not hold the 98% target on held-out: the threshold that
kept 98% of Sol's quality on the bank kept 95.8% on new requests. Random routing
at the same share to the cheap model keeps 10.6 fewer points of quality for the
OpenAI pair and 5.2 fewer for the Anthropic pair.

## The study

The study was pre-registered before any request was sent
([pre-registration](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/456601521d3eccee3830c29f0f5ee2e97920c212/evaluations/2026-10-11/request-routing-cost-prereg-v1),
[amendment 1](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/a003af2e392964468a107cc6a13c9d8b933f50d4/evaluations/2026-10-11/request-routing-cost-prereg-v1/amendment-1),
published after the pilot and before the main run). Its 1,457 requests come
from SuperGPQA (hard), MATH-500, MBPP+ and Arena-Hard v2.0 chat, alternating
in hash order between 729 calibration and 728 held-out requests. Every model answered every
request at its default reasoning setting. Benchmark answers were graded
automatically; chat answers were judged by a model from the other provider.

Each calibration request is labelled `y = 1 − max(0, q_strong − q_cheap)`: 1
when the cheap answer is as good, lower by the quality it loses. Calibration
chose the threshold for each pair and chose `knn20` from six SIE router
variants. Every figure above is on the held-out requests. The
[results](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/95d34b775b61bf1d584cada099bfed053710fc2b/evaluations/2026-10-11/request-routing-cost-results-v1)
hold every arm, router and record, and the study's own
[`analyze.py`](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/95d34b775b61bf1d584cada099bfed053710fc2b/evaluations/2026-10-11/request-routing-cost-results-v1/code/analyze.py).

## Verify the result

From this directory:

```sh
uv sync --frozen
uv run --frozen python fetch.py
uv run --frozen python score.py
```

Neither step needs a key. `fetch.py` downloads about 9 MB from the public
Hugging Face dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence)
at revision `95d34b775b61bf1d584cada099bfed053710fc2b` into `evidence/`. It
checks `MANIFEST.json` against the SHA-256 pinned in [`study.py`](./study.py),
and every other file against the size and SHA-256 the manifest lists:

- `inputs/items.jsonl`: the 1,457 requests and their split;
- `records/routers/bge.jsonl.gz`: the bge-m3 vector of each request, with the
  credits SIE Serverless billed for it;
- `items-openai.jsonl` and `items-anthropic.jsonl`: each request's graded
  quality and list cost for both models of the pair, and its recorded router
  score and route;
- `results.json`: the thresholds, the selected router and every strategy on
  held-out.

`score.py` checks the files again, then recomputes every knn20 score from the
vectors, re-chooses both thresholds on calibration, and rebuilds the held-out
rows from the recorded quality and costs. It uses the rules of `analyze.py`
(`router_scores`, `routed`, `choose_tau` and `summarize`) and makes no model
calls. It exits non-zero unless every score, route, threshold and figure
matches `results.json` and the headline above. It takes under a second:

```
GPT-6 Luna (cheap) -> GPT-6.1 Sol (strong), 728 held-out requests
knn20 threshold 0.9457; router $0.013 per 1,000, included

  Strategy                   per 1,000   Saving  Quality  To cheap
  Always GPT-6.1 Sol             $5.19        -   100.0%      0.0%
  SIE router (knn20)             $3.19    38.6%    95.8%     77.2%
  Random at the same 77.2%       $1.51    71.0%    85.2%     77.2%
  Always GPT-6 Luna              $0.42    92.0%    80.9%    100.0%

Claude Haiku 5.5 (cheap) -> Claude Sonnet 5.5 (strong), 728 held-out requests
knn20 threshold 0.9516; router $0.013 per 1,000, included

  Strategy                   per 1,000   Saving  Quality  To cheap
  Always Claude Sonnet 5.5      $10.36        -   100.0%      0.0%
  SIE router (knn20)             $9.15    11.7%    99.0%     34.9%
  Random at the same 34.9%       $6.99    32.5%    93.8%     34.9%
  Always Claude Haiku 5.5        $0.70    93.3%    82.2%    100.0%

Quality is routed quality as a share of always-strong quality on the same requests.
Cost is list price, with no caching or batch discount.

Recomputed 1,457 knn20 scores per pair and both thresholds. Every figure matches results.json.
```

The intervals come from the study's bootstrap in `results.json`; `score.py`
does not resample them.

## Route your own requests

`run.py` embeds a request on SIE Serverless, scores it against the study's 729
calibration requests and their labels for the pair, and prints the score, the
threshold from `results.json` and the model to call. Run `fetch.py` first.

```sh
export SIE_API_KEY=your-key
uv run --frozen python run.py "Solve for x: x/2 + x/3 = 5"
uv run --frozen python run.py --pair anthropic "Write a limerick about a lighthouse keeper"
```

The embedding is one call:

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item

client = SIEClient()  # SIE Serverless when SIE_API_KEY is set
vector = client.encode("BAAI/bge-m3", Item(text=request))["dense"]
```

`SIEClient()` reads `SIE_API_KEY` and connects to SIE Serverless. Set
`SIE_BASE_URL` as well to use your own SIE server instead. That default is
newer than the `sie-sdk` release on PyPI, so `pyproject.toml` pins the SDK to
the [superlinked/sie](https://github.com/superlinked/sie) commit that added it.
In the study a call was billed 1.3 credits on average, and 1 for most requests;
longer requests cost more. A credit is $0.00001. From the study's client host
the router took 345 ms at the median, mostly network: SIE reported a median
inference time of 20 ms.

`--item` routes one of the study's requests by id and prints its recorded
score beside the new one. `--recorded` uses its recorded vector instead of
calling SIE, so it needs no key:

```sh
uv run --frozen python run.py --item mbpp_plus/161 --recorded
```

```
pair   openai: GPT-6 Luna (cheap) or GPT-6.1 Sol (strong)
router knn
score  0.901633 < tau 0.945733
route  gpt-6.1-sol (GPT-6.1 Sol)
study  mbpp_plus/161 (held-out)
       recorded score 0.901633, routed strong
```

`--answer` also sends the request to the chosen OpenAI model with the study's
settings and prints the reply. It needs `OPENAI_API_KEY`, and the call is
billed by OpenAI.

### Build a bank from your own traffic

The bank here is the study's: benchmark requests, labelled by how the two
models answered them. It learns partly from the task type the benchmark
templates reveal, so it is not a router for your traffic. To route your own
requests, you need your own sample of requests answered by both models:

1. Answer each request in the sample with both models and grade both answers
   from 0 to 1.
2. Embed each request with `BAAI/bge-m3` and label it with
   `study.labels(q_cheap, q_strong)`.
3. Hold part of the sample back. Choose the threshold on the rest with
   `score.leave_one_out_scores` and `score.choose_tau`, and check it on the part
   you held back.
4. Route new requests with `study.Bank(pair, ids, study.normalise(vectors), labels, tau).route(vector)`.

## Test without inference

```sh
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

The tests check the kNN score and the routing rule on a tiny synthetic bank,
the threshold rule, the SDK request through a mock transport, and that
`fetch.py` rejects a manifest or file that does not match. They need no
network and no evidence.

## What this does not establish

- **Nothing about your traffic.** The requests are benchmark items, and the
  saving depends on the mix: the chat stratum is about a sixth of the requests
  but most of the always-strong spend, and the strong model wins most of it.
  The oracle saving on this mix is 41.6% for the OpenAI pair and 37.5% for the
  Anthropic pair.
- **Not the same quality for the OpenAI pair.** It kept 95.8% of GPT-6.1 Sol's
  quality, below the 98% target. Only the Anthropic pair passed the
  pre-registered rule for "same quality".
- **Grading limits.** Claude Haiku 5.5 often left out the `\boxed{}` the MATH
  prompt asked for; the pre-registered grader marked 35 of those replies wrong,
  about 23 of them correct in another form. The chat judges seldom called ties.
- **Latency elsewhere.** Router latency was measured from one client host.
- **Prices other than list prices.** Caching and batch discounts were not
  applied.
