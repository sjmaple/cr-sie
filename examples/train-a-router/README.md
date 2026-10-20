# Train a router that saves money

Send each request to a smaller model when it is likely to give a good answer,
and pay for the stronger model when its extra capability is useful. This
example measures the result in **dollars saved and answer quality lost**,
compared with always using the stronger model.

For the command-only walkthrough, use the [one-page quickstart](QUICKSTART.md).

**Draft: final experiment results below are PENDING.**

Follow these five steps. Each uses the same answer models and evaluation
questions, so you can see what each change contributes:

1. **Establish the always-strong baseline** [savings **0%**; quality gap **0 points**].
2. **Use the unchanged Laya decision model** [answer cost savings **PENDING**; gap **PENDING**].
3. **Adapt Laya with LoRA, then add more training examples** [answer cost savings **PENDING**; gap **PENDING**].
4. **Train two small classifiers on Qwen embeddings** [net savings **PENDING**; gap **PENDING**].
5. **Replace reference-based training labels with a teacher's labels** [net savings **PENDING**; gap **PENDING**].

All code is in this folder. The downloadable data, recorded answers, embeddings,
and optional model weights belong to this example's Hugging Face bundle.

## Dataset and the measures of success

We use competition mathematics problems from
[Easy2Hard-Bench](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench).
For each question, we record answers from **Qwen3.5-4B** and
**Qwen3.8-27B-FP8**, then check whether each final answer matches a reviewed
reference. The router learns two practical questions: **Will the smaller model
answer correctly? Will the stronger model answer correctly?**

The router sees only the problem text and competition name. Reference answers
and correctness labels are used for training and evaluation.

| Group | Reviewed questions | Purpose |
| --- | ---: | --- |
| Original training set | 503 | Fit a router with a modest labeling budget. |
| Expanded training set | 2,503 | Add examples from the same training competitions and years. |
| Calibration | 242 | Choose when the stronger model is worth its extra cost. |
| Main test | 463 | Measure the saved decisions on later competition problems. |

Separate training exam papers select training settings. Calibration chooses
the **cutoff**, the score above which a request goes to the stronger model.
The models and cutoffs are saved before examining test answers. Duplicate
families stay out of different groups. This separation covers our router's
training; we cannot establish what the original models saw during pretraining.
After exploratory training, we amended the source review to exclude
missing-context questions and correct references, before opening the held-out
answer-model results. The downloadable review records those decisions.

- **Answer accuracy:** percentage of correct final answers. Equivalent forms
  such as `1/2` and `0.5` count as the same answer.
- **Quality gap:** always-strong accuracy minus routed accuracy, in percentage
  points. A change from 90% to 89% is a **1-point** gap, or 100 fewer correct
  answers per 10,000 requests. We also count answers recovered when the smaller
  model succeeds and the stronger one fails.
- **Cost per 10,000 requests:** the selected answer-model charges plus the known
  cost of deciding where to send each request.
- **Savings:** the reduction from the always-strong bill. A drop from $100 to $70
  per 10,000 requests is **30% savings**. For locally hosted Laya, we report
  answer-model savings and the remaining allowance for router hosting; its
  unmeasured hosting cost is not treated as zero.

The primary setting allows a **1-point gap on calibration**, then measures the
actual test gap. The 463-question test cannot establish a guaranteed 1-point
gap on future requests. The report also keeps 0-, 2-, and 5-point targets and
shows 95% uncertainty intervals, estimated by resampling questions while
keeping each pair of model answers together. Dollar tables scale the
recorded workload to 10,000 requests; the JSON retains per-1,000 values.

## Setup

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
**CPU is sufficient to replay every recorded result and fit the embedding
classifiers. To perform the LoRA stage yourself, use an NVIDIA CUDA machine**,
locally or rented, including a Modal GPU shell. The reference uses an H100.
Run the training commands inside that machine's checkout; SIE Cloud supplies
inference, while your GPU machine performs training.

From the root of a clone of this repository:

```sh
# Enter this example and install the locked CPU replay and classifier dependencies.
cd examples/train-a-router
uv sync --frozen --group outcomes
```

```sh
# Download this example's verified dataset, embeddings, and recorded comparison.
uv run --frozen --group outcomes python router.py fetch --study evidence
```

`evidence/` holds references; `run-output/` holds your results. Repeated runs
need fresh output directories.

**PENDING publication:** the download command is implemented but its final
Hugging Face revision and manifest must be pinned before this draft is released.

<a id="baseline"></a>

## 1. Establish the always-strong baseline [savings 0%; quality gap 0 points]

**CPU.** Recompute the complete comparison from saved answers and decisions.
This makes no model calls and needs no API key.

```sh
# Recompute the cost and answer-quality tables from the downloaded evidence.
uv run --frozen --group outcomes python router.py replay \
  --study evidence --output run-output/replay
```

Open `run-output/replay/comparison.md`. The first row sends every question to
Qwen 27B: accuracy **PENDING**, cost **PENDING per 10,000 requests**. The
always-Qwen-4B row shows the cheapest fixed choice and its quality gap.

Every router selects between the same two recorded answers per question.
The main comparison uses measured SIE Cloud answer charges; replaying it adds
no inference cost.

<a id="laya"></a>

## 2. Use Laya [answer cost savings PENDING; quality gap PENDING]

**CPU for replay; optional local CPU or NVIDIA GPU for new decisions.** Laya
is a decision model: it scores choices instead of writing a full answer.
Here it estimates each answer model's chance of being correct. Compare the
**Laya unchanged** rows in the report from step 1.

Its weights remain unchanged. Training examples supply expected costs, and
calibration sets the cutoff. Laya has roughly 421 million parameters—the
numerical weights stored in a model.

<a id="lora"></a>

## 3. Adapt Laya with LoRA [answer cost savings PENDING; quality gap PENDING]

**NVIDIA CUDA GPU for training; CPU for replaying the resulting table.**
LoRA trains small added weight updates while keeping most of Laya fixed.
We also train its final decision layers. Both correctness questions share one
model. Separate training papers select how long to train.

```sh
# Create your own training folders without changing the downloaded reference results.
uv run --frozen --group outcomes python router.py prepare \
  --study evidence --output run-output/my-router
```

```sh
# On your CUDA machine, install the locked LoRA training dependencies.
uv sync --frozen --group outcomes --group training
```

```sh
# Train and evaluate Laya on the original set, then repeat with the larger set.
for population in original expanded; do
  uv run --frozen --group outcomes --group training python router.py fit-native \
    --study evidence --run run-output/my-router --population "$population"
  uv run --frozen --group outcomes --group training python router.py evaluate \
    --study evidence --run run-output/my-router --population "$population" \
    --method native --output "run-output/laya-$population.json"
done
```

Compare unchanged Laya with LoRA, then compare 503 with 2,503 training examples
under the same recipe. **Training time, peak GPU memory, and training cost:
PENDING.** Training is a one-time expense; hosting recurs for each request.

<a id="embedding-router"></a>

## 4. Train the embedding router [net savings PENDING; quality gap PENDING]

**CPU with the supplied embeddings.** Qwen3-Embedding-4B turns each question
into a list of 2,560 numbers describing its text. Two small **logistic
classifiers** learn from those numbers: one predicts whether Qwen 4B will be
correct, and the other predicts whether Qwen 27B will be correct.

The routing score is the predicted quality improvement divided by expected
extra cost. Cost estimates come from training records; they never use a new
question's future answer length or correctness.

```sh
# Fit and evaluate both training sizes on CPU using the saved Qwen embeddings.
for population in original expanded; do
  uv run --frozen --group outcomes python router.py fit-embedding \
    --study evidence --run run-output/my-router --population "$population"
  uv run --frozen --group outcomes python router.py evaluate \
    --study evidence --run run-output/my-router --population "$population" \
    --method embedding --output "run-output/embedding-$population.json"
done
```

Requests scored by this router need one embedding call; both classifiers then
run on CPU. A fixed choice skips that call. Net savings include the API charge.
Local application and classifier CPU costs are unpriced.
The 4B encoder is roughly ten times larger than Laya, so this comparison includes
model capacity as well as training design.

<a id="teacher-labels"></a>

## 5. Train from teacher labels [net savings PENDING; quality gap PENDING]

**CPU for the recorded comparison; a teacher API or hosted model for new
labels.** When reference answers are unavailable, ask a stronger teacher to
solve each training question independently. Compare both answer models' saved
answers with that reference to obtain the two correctness labels, then train
the same classifiers.

The recorded teacher is **gpt-6-astra**. Of the 2,503 training questions,
**2,426** have a teacher reference that the fixed grader can use; 77 have
unsupported answer forms or units. This rule uses the teacher reference and
question, without looking at either answer model's response. The matched
reference-label control uses exactly those same 2,426 questions, embeddings,
and training settings, isolating the label change. The teacher-derived labels
agree with the reviewed reference labels on **99.6%** of smaller-model answers
and **99.4%** of stronger-model answers. This is a training-data diagnostic;
the held-out routing result is measured separately. Known teacher usage for
all 2,503 training questions was **$39.05**, including unusable references and
known failed attempts. Provider calls with unknown usage are recorded
separately, so this is not a complete invoice. Check a sample and disagreements
before trusting generated references.

```sh
# Refit the teacher-label router and its matched reference-label control on CPU.
uv run --frozen --group outcomes python router.py fit-teacher \
  --study evidence --run run-output/my-router --population expanded
uv run --frozen --group outcomes python router.py evaluate \
  --study evidence --run run-output/my-router --population expanded \
  --method teacher --output run-output/teacher-comparison.json
```

These commands reuse the recorded teacher answers and incur no labeling
charges. New teacher labels require a separate acquisition run with an explicit
provider/model and spending limit. Calibration and test still use the reviewed
benchmark references for both routers.

## Use your router with SIE Cloud or your own server

Create an account and API key in the [SIE Console](https://console.superlinked.com).
Use the available free credits shown in your account for model calls. Replaying
and fitting from downloaded records uses no Cloud credits.

```sh
# Point the SDK at SIE Cloud and provide the API key from your console account.
export SIE_BASE_URL=https://api.superlinked.com
export SIE_API_KEY='paste-your-key'
```

```sh
# Route a new problem with your trained embedding router and save its answer and charges.
uv run --frozen --group outcomes python router.py answer \
  --study evidence --run run-output/my-router --population expanded \
  --method embedding --contest AMC10 \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/cloud-answer
```

The output records the chosen model, answer, and API charges. `--target 1`
selects the calibrated cutoff. A new answer is not automatically graded.

For your own running SIE deployment, use the same command with its endpoint and
`--self-hosted`. It must expose the pinned 4B and 27B answer models and the
Qwen4B embedding model, including structured final-answer support.

```sh
# Send the same routed request to a locally running, unauthenticated SIE deployment.
SIE_BASE_URL=http://127.0.0.1:8080 SIE_API_KEY='' \
  uv run --frozen --group outcomes python router.py answer \
  --study evidence --run run-output/my-router --population expanded \
  --method embedding --contest AMC10 \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --self-hosted --output run-output/local-answer
```

Use your deployment's key instead of the empty value if it requires one.
Changing the URL selects a server; it does not start the models. Self-hosted
compute needs separate hardware-cost accounting. **PENDING: the complete
public multi-model launch recipe must accompany this example.**

## Serving trade-offs and a separate transfer check

| Design | Work for a new routed request | Costs to keep separate |
| --- | --- | --- |
| Always strong | One Qwen 27B answer, plus a final-answer formatting call when needed | Answer generation |
| Laya / Laya + LoRA | Two correctness decisions through one shared local Laya model, then the chosen answer model | Router hosting, answer generation; LoRA training once |
| Qwen embeddings + classifiers | One Qwen4B embedding API call, two small CPU calculations, then the chosen answer model | Embedding API, answer generation; classifier fitting once |

The classifier is small, but new embeddings still require the 4B model. Laya
has a smaller serving footprint. In a 20-question sequential Cloud probe,
embedding plus classification took **184 ms median and 274 ms at the 95th
percentile**, including network time; the local classifier calculation took
**0.21 ms median**. This small probe is separate from the quality evaluation.
Native inference measurements are **PENDING**. These are serving setups with
different hardware, not a controlled hardware comparison. Command-line startup
and concurrent throughput need separate measurement.

We also apply the saved routers and cutoffs to a separate **2,000-question
arithmetic transfer sample** from
[GSM8K](https://huggingface.co/datasets/openai/gsm8k), without retraining.
It combines 1,319 official test questions with 681 official training questions
never used for our routers. It tests a different problem distribution and does
not enlarge the competition test. This custom study has its own
`arithmetic-transfer.md`; quality gap and savings are **PENDING**. Its
self-hosted answers use token-based Cloud-price estimates rather than measured
Cloud bills. It is not a standard GSM8K leaderboard score.

See [how the experiment is designed](docs/methodology.md) and
[how to run and measure a live router](docs/serving.md) for more detail.

Published results support reproduction. Use new evaluation questions when
assessing further development after studying this test set.
