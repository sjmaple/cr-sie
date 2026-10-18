# From routing accuracy to answer quality and dollars

The difficulty lab teaches a router to recognize human-rated problem difficulty.
This lab asks the operational question: **when does the larger answering model
actually earn its extra cost?** It uses two real answers to each problem and
recorded SIE Cloud credit receipts, including the embedding used to route it.

The answering models are **Qwen3.5-4B** and **Qwen3.8-27B-FP8**. A separate,
blinded Qwen3.8-27B review checks both candidate solutions. Candidate order is
randomized deterministically; neither answer is presented as a trusted reference.
The teacher sees the question and solutions, never the benchmark answer key.
It can mark both answers correct, both incorrect, or either one correct.

## Measured result

All 512 held-out questions have a completed answer from each model. The table uses the policies chosen on validation, before test collection.

| Policy | $ / 1,000 requests | Saving | Answer accuracy | Loss vs 27B |
| --- | ---: | ---: | ---: | ---: |
| Always 27B | $4.918 | 0.00% | 73.24% | +0.00 pp |
| Always 4B | $1.595 | 67.57% | 43.75% | +29.49 pp |
| Difficulty ridge · 512 | $3.774 | 23.27% | 61.13% | +12.11 pp |
| Teacher review · 1pp budget | $4.686 | 4.72% | 71.29% | +1.95 pp |
| Teacher review · 5pp budget | $3.717 | 24.42% | 66.21% | +7.03 pp |
| Answer-key control · 1pp budget | $3.926 | 20.17% | 67.97% | +5.27 pp |

The frozen primary did not meet the target of at least 25% savings with at most 1pp observed quality loss. It lost 10 previously correct answers and gained 0; the paired 95% interval for the net gap is 0.78 to 3.32pp. The conservative 1pp noninferiority check is not supported. The 5pp row is a separately predeclared secondary tradeoff, not a replacement primary.

At one million requests with this same workload mix, the recorded rates imply $4,918 for always-27B, $4,686 for the primary, and $3,717 for the 5pp policy. This scales the observed rates; it is not a forecast or a volume-price quote.

Embedding overhead was only **$0.012 per 1,000 requests**. The material cost is which answering model runs and how many output tokens it uses. Learning better difficulty labels does not by itself optimize that choice.

[Immutable evidence, per-question outcomes, receipts and figure](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/fc8ac96c8492c30945a133dd6669013ba559e494/evaluations/2026-10-18/request-routing-economics-v1). The publication contains both the successful responses and the investigated failed attempt. `economics_lab.py replay` verifies the reported policies and metrics.

## Run the lab on a laptop

From `examples/request-routing/cost`, with Python 3.12 and
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv sync --frozen --group grading
uv run --frozen python economics_lab.py download
uv run --frozen --group grading python economics_lab.py learn --run run-output/my-economic-router
uv run --frozen --group grading python economics_lab.py evaluate --run run-output/my-economic-router
```

Downloading requires no API key. Training uses recorded teacher reviews and
Qwen3-Embedding-4B vectors. It writes a new directory containing development
answers and teacher labels, without copying the answer-key file. Evaluation
then brings in held-out answers and independent answer keys. No model weights,
GPU or inference calls are needed for this path.

To reconstruct every published comparison and verify the numerical results:

```sh
uv run --frozen --group grading python economics_lab.py replay --run run-output/economic-replay
```

Choose a new output directory for each training or replay run. The scripts
reject changed evidence, mismatched input identities and altered frozen policies.

## Run a new request through Cloud

Create an account at [SIE Console](https://console.superlinked.com), check the
available starter credits, and create an API key. Keep the key in your shell,
outside source files:

```sh
unset SIE_BASE_URL
export SIE_API_KEY=your-key
uv run --frozen python economics_lab.py route --run run-output/my-economic-router \
  --request 'Find the number of integer solutions to x^2 + y^2 = 2025.' --answer
```

This embeds the question, makes the local routing decision, and calls the chosen
answering model. Omitting `--answer` makes only the embedding call. The output
contains actual credit receipts. The bounded answering recipe permits one
short final-answer turn if the first response runs out of tokens or omits its
final answer; its cost is included.

The default serves the frozen primary. Add `--quality-budget 2` or
`--quality-budget 5` to serve the corresponding policy selected on validation.
These numbers name validation settings, not guarantees about live answer quality;
compare their independently measured test gaps before choosing a policy.

## Collect teacher feedback without an answer key

Try four reviews using the published development questions and candidate answers:

```sh
uv run --frozen python economics_judge.py prepare --run evidence/economics \
  --input run-output/my-teacher-input.json --count 2 --finalize-on-invalid
uv run --frozen python economics_judge.py collect --input run-output/my-teacher-input.json \
  --output run-output/my-teacher-reviews --concurrency 4
```

`--count 2` selects two training and two validation questions. This tests the
live teacher path; it is not enough data to fit the full router. A review is capped at
1,024 output tokens, with one 64-token final-verdict turn when necessary.
Failed requests are retained and are not retried automatically.

To reproduce all 640 labels, prepare `--count 512` and collect five bounded
128-case shards. Each shard has a 20-minute and 100,000-observed-credit stop;
in-flight calls are retained. This keeps a full run within normal key concurrency
limits without pretending the four-case smoke test is the full experiment:

```sh
uv run --frozen python economics_judge.py prepare --run evidence/economics \
  --input run-output/full-teacher-input.json --count 512 --finalize-on-invalid
for start in 0 128 256 384 512; do
  uv run --frozen python economics_judge.py collect --input run-output/full-teacher-input.json \
    --output "run-output/teacher-$start" --start "$start" --limit 128 --concurrency 4 || break
done
uv run --frozen --group grading python economics_lab.py learn --run run-output/live-teacher-router \
  --teacher-reviews run-output/teacher-0/reviews.jsonl \
  --teacher-reviews run-output/teacher-128/reviews.jsonl \
  --teacher-reviews run-output/teacher-256/reviews.jsonl \
  --teacher-reviews run-output/teacher-384/reviews.jsonl \
  --teacher-reviews run-output/teacher-512/reviews.jsonl
uv run --frozen --group grading python economics_lab.py evaluate --run run-output/live-teacher-router
```

The fit rejects missing, duplicated, invalid or mismatched reviews. A fresh teacher
run can produce different labels and results; published numbers describe the
recorded run. Do not tune your teacher after looking at the test results.

The teacher target is the quality advantage of strong over cheap: **+1** if only
strong is correct, **0** if their correctness matches, and **−1** if only cheap
is correct. The small classifier learns this target from the question embedding.
A second linear model predicts the potential credit saving. Validation chooses
a routing cutoff for each declared quality budget. At serving time, teacher
review is unnecessary: only one embedding and the selected answering model run.

Simply treating disagreement with the larger model as a failure is a separate
baseline. Disagreement can also mean the larger model is wrong, or both are
wrong. The independently scored experiment measures the difference.

## Recollect the paired-answer experiment

This optional Cloud run generates 2,304 answers plus 640 teacher reviews and
any bounded final-answer turns. Budget time and credits separately from the
short classroom exercise; starter credits are not a guarantee that a full
research run is covered. The answer collector stops submitting work at
1,000,000 observed credits across phases, or two hours per collection, and
retains in-flight results. Each teacher shard has the smaller bounds above.
After `economics_lab.py download` above, these commands reuse the pinned
questions and development embeddings but obtain new model answers and reviews:

```sh
uv sync --frozen --group data --group grading
uv run --frozen --group data python economics.py prepare --run run-output/fresh-economics
uv run --frozen python economics.py configure --run run-output/fresh-economics
uv run --frozen python economics.py collect --run run-output/fresh-economics --phase train
uv run --frozen python economics.py collect --run run-output/fresh-economics --phase validation
uv run --frozen python economics_judge.py prepare --run run-output/fresh-economics \
  --input run-output/fresh-economics/teacher-input.json --count 512 --finalize-on-invalid
for start in 0 128 256 384 512; do
  uv run --frozen python economics_judge.py collect \
    --input run-output/fresh-economics/teacher-input.json \
    --output "run-output/fresh-economics/teacher-$start" --start "$start" --limit 128 || break
done
uv run --frozen --group grading python economics_review.py train --run run-output/fresh-economics \
  --input run-output/fresh-economics/teacher-input.json \
  --reviews run-output/fresh-economics/teacher-0/reviews.jsonl \
  --reviews run-output/fresh-economics/teacher-128/reviews.jsonl \
  --reviews run-output/fresh-economics/teacher-256/reviews.jsonl \
  --reviews run-output/fresh-economics/teacher-384/reviews.jsonl \
  --reviews run-output/fresh-economics/teacher-512/reviews.jsonl
uv run --frozen python economics_policy.py --run run-output/fresh-economics \
  --freeze --sources teacher_review --primary-source teacher_review
uv run --frozen python economics_router.py --run run-output/fresh-economics
uv run --frozen python economics.py collect --run run-output/fresh-economics --phase test
uv run --frozen --group grading python economics_policy.py --run run-output/fresh-economics --evaluate
uv run --frozen python economics_scoring.py --run run-output/fresh-economics
uv run --frozen python economics_report.py --run run-output/fresh-economics
```

Run each step only after the preceding step succeeds. Failed or unknown paid
requests require investigation; rerunning a command does not silently retry
them. The final test collection refuses to start without frozen policies.
Answer keys are kept separately for final scoring; collection, review and
teacher-supervised fitting do not read them. The optional answer-key control
in the published comparison is deliberately excluded from this teacher-only run.

## Use your own SIE server

Set `SIE_BASE_URL` to your server, unset the Cloud key, and add `--self-hosted` to
the `route` or teacher `collect` command. The same SDK is used. Native token usage
is reported without inventing Cloud credits or treating GPU compute as free.

The [embedding server instructions](README.md#run-your-own-sie-server) support the
laptop routing path. Running the 27B teacher requires a suitable GPU server;
our full labeling run used an H100 80 GB. The public CUDA 13 image is
`ghcr.io/superlinked/sie-server:v0.10.1-cuda13-sglang-cu130`.
For JSON-constrained teacher reviews with a single explicit model profile, serve
`Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec` and pass that exact ID to
`economics_judge.py collect --model ... --self-hosted`. The parent batch profile
can direct JSON requests to the non-speculative sibling, so that sibling must
actually be served. Ordinary Cloud calls use the default bare model ID.

On the GPU host, with Docker, NVIDIA Container Toolkit and a CUDA-13-compatible
driver (580 or newer), the public image's entrypoint accepts:

```sh
docker run --rm --gpus all --ipc=host -p 127.0.0.1:8098:8080 \
  -e SIE_AUTH_MODE=none -v sie-odsc-hf:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:v0.10.1-cuda13-sglang-cu130 \
  serve --host 0.0.0.0 --port 8080 --models-dir /app/models \
  --models Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec \
  --preload Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec --device cuda
```

The tag was verified anonymously against manifest digest
`sha256:b9262c56066e97d6d1892c57edf96140d951316aa1fd7929bf6eb1424d9d6c81`.
For an immutable pull, append `@sha256:b9262c56066e97d6d1892c57edf96140d951316aa1fd7929bf6eb1424d9d6c81`
to the image reference above.

In a second terminal on that host:

```sh
unset SIE_API_KEY
export SIE_BASE_URL=http://127.0.0.1:8098
uv run --frozen python economics_judge.py collect --input run-output/my-teacher-input.json \
  --output run-output/my-native-reviews --self-hosted \
  --model Qwen/Qwen3.8-27B-FP8:h100-256k-batch-no-spec
```

The server port is exposed only on the host's loopback interface. Stop the
container with Ctrl-C. Our native verification used the public SIE server with
the same engine versions; the image and hardware compatibility are recorded in
the repository's [compatibility guide](../../../docs/compatibility.md).

The public server source, image and example are sufficient; the lab does not
require our internal GPU orchestration repository. To run the complete live
answering path, make the embedding model and both answering model IDs available
at the configured endpoint with enough capacity for their weights and caches.

The local economic CLI check served six development questions successfully on
Apple Silicon. Five decisions matched the recorded Cloud vectors; one kNN score
moved from −0.000079 to +0.000017 across the zero cutoff. Replaying recorded vectors
is deterministic; moving a frozen policy to another inference backend needs
validation and, where necessary, new vectors and recalibration.

## What the measurements mean

The split contains 512 training, 128 validation and 512 held-out questions.
Routing weights, cutoffs and the primary comparison are frozen before collecting
the held-out answers. All 64/128/256/512-label policies and 1/2/5-percentage-point
validation quality budgets are retained as declared comparisons. The separate
gold-label control measures what the same learner can do with correct labels.

Answer quality means correctness against the source mathematical answer key,
using the pinned `math-verify` grader and explicit normalization. It is not
agreement with the teacher, routing accuracy, or a judgment by another LLM.
An audit after frozen scoring found that the parser rejected `e^e - 1`, although
it equals the reference `e^{e}-1`. Both models gave that answer to one question.
`economics_scoring.py` applies exact textual equivalence after bracing elementary
single-token exponents and removing whitespace; it retains the original scores
in `economic-results-unadjusted.json` and logs every correction. This adds
0.195 percentage points to every policy's accuracy. All costs, net quality gaps,
paired intervals and policy choices are unchanged. The published figures use
the audited scores, and classroom replay reproduces both stages.
The recorded replay can also print `Timeout during comparison` from the bounded
symbolic grader. The audited expression subtracts a strictly positive sum from
the reference value, so its incorrect score is retained. This is a grading
diagnostic, not an API call or an incomplete replay.
Both lost and gained correct answers are shown. The paired confidence interval
describes uncertainty in the net quality gap; the conservative harmful-switch
bound prevents zero observed losses from being presented as proof of zero risk.

Dollar amounts convert actual recorded credits at 100,000 credits per dollar.
They describe this recorded experiment, not a promise about future pricing.
Serving totals include actual singleton embedding charges. Training answer
acquisition, batched embeddings and teacher review are reported separately.
Self-hosted teacher GPU cost is not included in Cloud-credit break-even figures.

The held-out collection encountered one HTTP 500 and retained the original
failure plus one explicitly investigated successful reissue; see
[production issue #800](https://github.com/superlinked/sie/issues/800).
That failed attempt returned no settled credit receipt. Its charge is unknown
and is excluded from the recorded-credit totals; serving comparisons use the
successful paired answers. No failed question is dropped from the evaluation.

The initial answer prompt was selected using a small training pilot with answer
keys. The precise no-label claim is that **teacher review, router fitting and
cutoff calibration do not read those keys**. Independent research diagnostics
and final scoring do. The test questions were previously used to confirm human
difficulty classification; their paired answering-model outcomes were unseen
when the economic policies were frozen.

Source data: [Easy2Hard-Bench](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench),
AMC subset, pinned revision `55bc0d2fb10954151e669d2026b87fa896f2fa26`,
CC-BY-SA 4.0. Preserve that attribution when redistributing the questions.
