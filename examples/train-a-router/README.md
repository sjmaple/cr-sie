# Train a router that saves money

Use a smaller answer model when it is likely to succeed, and pay for the stronger
model when its extra capability is useful. This example measures **dollars
saved and correct answers lost**, compared with always using the stronger model.

The workload is **GSM8K arithmetic word problems**. The two answer models are
**Qwen3.5-4B** and **Qwen3.8-27B-FP8**; their final answers are checked against
the dataset's answer key. A **quality gap** is a difference in answer accuracy:
96% falling to 95% is a one-percentage-point gap.

On **3,000 separate test questions**, the embedding router trained on 2,000
examples saves **20.9%** with a **0.47-point quality gap** at the primary
setting. A second setting, also chosen before testing, saves **33.8%** with
a **0.83-point gap**. That reduces the estimated bill from **$9.51 to $6.30
per 10,000 questions**, with about **83 fewer correct answers**.
The two settings trade money against answer quality; neither guarantees
the same result on a different workload.

Build a router, improve its choices, and select a cost/quality trade-off for
new requests. Follow these steps, or use the [one-page quickstart](QUICKSTART.md):

1. **Measure the opportunity:** compare always using the stronger model with
   always using the cheaper one [always-strong savings **0%**; quality gap **0 points**].
2. **Inspect unchanged Laya:** establish what a compact decision model can
   do before learning this workload [savings **21.6% before hosting**; gap **1.50 points**].
3. **Adapt Laya with LoRA:** preserve more correct answers by learning from the
   two answer models' outcomes [savings **16.4% before hosting**; gap **0.80 points**].
4. **Train small classifiers on Qwen embeddings:** improve measured cost and
   quality, with classifier training on CPU [savings **20.9%**; gap **0.47 points**].
5. **When training answer keys are missing, use a teacher:** repeat the classifier
   recipe with generated references [savings **19.7%**; gap **0.33 points**].
6. **Choose a setting and serve new requests:** use your trained router through
   SIE Cloud or your own server [embedding savings **20.9–33.8%**; measured gap **0.47–0.83 points**].

Steps 1–4 compare routing methods. Step 5 is an alternative source of training
labels; step 6 turns the chosen method into a working application. LoRA's first
benefit is better answer quality. The embedding approach then improves both
measured quality and cost, at the expense of using a larger encoder.
Both training stages compare 500 with 2,000 examples, showing how retraining
with more observed answer outcomes changes the tradeoff.

All code is in this folder. The example's [Hugging Face download](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/71e0d5109e46f8b9c698806e7d3e47b2aa0da87d/evaluations/2026-10-19/train-a-router-arithmetic) contains the
questions, answer keys, recorded predictions, embeddings, and optional model
weights. You can reproduce the comparison without paying for those model calls
again. LoRA is a full training stage and requires an NVIDIA CUDA machine.

## Dataset and the measures of success

We use arithmetic word problems from [GSM8K](https://github.com/openai/grade-school-math).
For every question, we record answers from **Qwen3.5-4B** and
**Qwen3.8-27B-FP8**, then check their final answers against the dataset's answer
key. A router learns when the 27B model is worth its extra cost.

| Group | Questions | Purpose |
| --- | ---: | --- |
| Small training set | 500 | Start with fewer labeled examples. |
| Expanded training set | 2,000 | Repeat the same recipe with more examples; includes the 500-question set. |
| Calibration | 1,000 | Choose the score cutoff for sending a question to 27B. |
| Test | 3,000 | Measure the saved models and cutoffs on separate questions. |

The routers see the question text, not the answer key or either model's answer.
Training labels record whether each answer model was correct:

| Recorded answers | What switching from 4B to 27B would do |
| --- | --- |
| Both correct | Keep the same answer quality. |
| Only 27B correct | Recover a correct answer. |
| Only 4B correct | Lose a correct answer. |
| Both incorrect | Leave the answer incorrect. |

The router learns to predict the benefit of switching and compares it with
the expected extra cost. It cannot see these
outcomes when choosing a model for a new question.

We group duplicates and similar question templates before splitting the data.
No detected family crosses training, calibration, and test. The 2,000 training
questions had already been used during development; calibration and test use
previously unused families. These study groups differ from GSM8K's official
split: training contains its 1,319 official test questions and 681 official
training questions; our calibration and test questions come from the remaining
official training split. These are custom routing results, not standard GSM8K
benchmark scores. The checks do not establish what the foundation models saw during
pretraining. See the [measurement method](docs/methodology.md) for the complete
split, grading, and review rules.

- **Answer accuracy:** the percentage of correct final answers. Equivalent
  numeric forms, such as `1/2` and `0.5`, count as equal.
- **Quality gap:** always-strong accuracy minus routed accuracy, in percentage
  points. Dropping from 96% to 95% is a **1-point gap**: 100 fewer correct answers
  per 10,000 questions. Gains when only the smaller model succeeds count too.
- **Savings:** the percentage reduction in cost versus always using 27B. A
  drop from $100 to $70 is **30% savings**.
- **Cost per 10,000 questions:** the chosen answer-model charges plus known
  router charges. The report also keeps costs per 1,000 for other traffic levels.

The main comparison chooses cutoffs for a **1-point gap on calibration**, then
measures the actual test gap. In practical terms: on the 1,000 calibration
questions, choose the cheapest rule that loses at most 10 correct answers
relative to always using 27B. Save that rule, then evaluate it on the 3,000 test
questions. That setting is not a guarantee on new questions.
The report includes 95% uncertainty intervals and all four declared targets:
0, 1, 2, and 5 points.

**Primary setting: a 1-point calibration target, evaluated on 3,000 test questions.**

| Method | Answer accuracy | Quality gap, points (95% interval) | Cost / 10,000 | Savings |
| --- | ---: | ---: | ---: | ---: |
| Always Qwen 27B | 95.90% | 0.00 (0.00–0.00) | $9.51 | 0.0% |
| Always Qwen 4B | 91.03% | 4.87 (3.97–5.93) | $2.29 | 75.9% |
| Laya, unchanged | 94.40% | 1.50 (0.97–2.07) | $7.46 + hosting | 21.6% |
| Laya + LoRA, 2,000 examples | 95.10% | 0.80 (0.40–1.23) | $7.95 + hosting | 16.4% |
| Embeddings + classifiers, 2,000 examples | 95.43% | 0.47 (0.07–0.87) | $7.52 | 20.9% |
| Embeddings + teacher labels, 1,539 examples | 95.57% | 0.33 (0.03–0.67) | $7.63 | 19.7% |
| Same 1,539 questions, reference labels | 95.30% | 0.60 (0.20–1.03) | $7.41 | 22.0% |

Answer costs use recorded physical token counts and a saved SIE Cloud price
table; they are Cloud-price estimates. Embedding savings include observed
embedding credits, shared across the questions in each collection batch:
about **$0.059 per 10,000 questions**. The one-question timing calls used one
credit each, equivalent to **$0.10 per 10,000** at that rate; individual calls
can therefore cost slightly more than the recorded batches. Laya's hosting cost
is unpriced, so its savings are **before router hosting**. Training and teacher
labeling are separate one-time expenses.

At the primary setting, LoRA reduces the unchanged decision model's
quality gap from **1.50 to 0.80 points**, while saving less on answer calls.
The embedding router then reaches **0.47 points** and saves **20.9%**,
including its embedding bill. These are the recorded point estimates;
the intervals show the uncertainty in each quality gap.

**Secondary setting: a 2-point calibration target.** The same saved models
use a more permissive cutoff, chosen before opening the test answers.

| Method | Savings | Measured test gap, points (95% interval) |
| --- | ---: | ---: |
| Laya, unchanged | 26.9% before hosting | 1.77 (1.17–2.43) |
| Laya + LoRA, 2,000 examples | 29.4% before hosting | 1.53 (0.97–2.17) |
| Embeddings + classifiers, 2,000 examples | 33.8% | 0.83 (0.33–1.30) |
| Embeddings + teacher labels, 1,539 examples | 35.2% | 1.03 (0.50–1.53) |
| Same 1,539 questions, reference labels | 36.7% | 1.13 (0.60–1.67) |

For the 2,000-example embedding router, secondary savings are **33.8%**
(95% interval **32.0–35.6%**). Its **0.83-point** test gap has an interval
of **0.33–1.30 points**; this does not establish a guaranteed gap below
one point. The primary setting remains the main comparison. Both are
available as saved rules: `--target 1` and `--target 2`.

The full comparison plots retain both training sizes and the controls:
[primary setting](https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/71e0d5109e46f8b9c698806e7d3e47b2aa0da87d/evaluations/2026-10-19/train-a-router-arithmetic/arithmetic/figures/cost-and-quality-1pp.png)
and [secondary setting](https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/71e0d5109e46f8b9c698806e7d3e47b2aa0da87d/evaluations/2026-10-19/train-a-router-arithmetic/arithmetic/figures/cost-and-quality-2pp.png).

The headings below show the primary setting, using 2,000 training
examples except for the matched 1,539-example teacher comparison.

## Setup

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
For the complete walkthrough, use **an NVIDIA CUDA machine**, locally or rented.
The reference LoRA training ran on a Modal H100; a
[Modal GPU shell](https://modal.com/docs/cli/latest/shell) is one way to rent a
machine. Run the training commands inside that machine's checkout. The remaining training and
replay steps work on CPU with the saved embeddings. SIE Cloud supplies the
embedding and answer models for new requests; the CUDA machine trains Laya.

From the root of a clone of this repository:

```sh
# Enter the example and install its locked replay and classifier dependencies.
cd examples/train-a-router
uv sync --frozen --group outcomes
```

```sh
# Download the verified data, recorded results, and embeddings for this example.
uv run --frozen --group outcomes python router.py fetch --study evidence
```

`evidence/` contains the downloaded reference material. `run-output/` contains
your results. The default download is about **159 MB**. Use a fresh output
directory when repeating a command.

<a id="baseline"></a>

## 1. Establish the always-strong baseline [savings 0%; quality gap 0 points]

**Purpose:** find out whether the cheaper model loses enough quality to justify
a router, and how much spending there is to save.

**CPU.** Recompute the comparison from saved answers and routing decisions.
This makes no model calls and needs no API key.

```sh
# Recompute costs, answer accuracy, and uncertainty intervals for the saved decisions.
uv run --frozen --group outcomes python router.py replay \
  --study evidence --output run-output/replay
```

Open `run-output/replay/comparison.md`. Always using 27B answers **95.90%** correctly at an estimated **$9.51 per
10,000 questions**. Always using 4B costs **$2.29**, but accuracy falls to
**91.03%**: a **4.87-point gap**, or about 487 fewer correct answers per
10,000. At its saved secondary setting, the final embedding router reduces
that loss to about **83 answers**, while costing **$3.21 less per 10,000**
than always using 27B.

Every router chooses between the same two recorded answers for each question.
That isolates the effect of its routing choices from variation in newly
generated answers. Replaying the published results is a reproduction;
further experiments need new test questions.

<a id="laya"></a>

## 2. Inspect unchanged Laya [savings **21.6% before hosting**; quality gap **1.50 points**]

**Purpose:** try a small model designed for decisions before adapting it to the
successes and failures of these two answer models.

**CPU for replay; CPU or NVIDIA GPU for new decisions.** Laya is a decision
model: it scores possible choices instead of generating an answer. Here it
predicts whether each of the two answer models will be correct.

Start with **Laya, unchanged; costs from 2,000 examples** in the report. Its roughly 421 million
model weights stay fixed. Training records supply expected answer costs, and
calibration chooses the cutoff. The baseline uses the same cost estimates as
the corresponding LoRA run. This keeps the cost assumptions matched when
comparing the unchanged and trained routers; each has its own chosen cutoff.

**Your task:** use the replayed report to record this baseline's cost and
quality gap, then compare the gap with the one-point goal.

The unchanged decision model saves **21.6% before hosting**, but its
**1.50-point** test gap exceeds the one-point calibration target. It
provides a concrete baseline for learning the decisions from our own
answer outcomes.

<a id="lora"></a>

## 3. Adapt Laya with LoRA [savings **16.4% before hosting**; quality gap **0.80 points**]

**Purpose:** recover answer quality while retaining some of the saving. This
stage teaches the decision model which questions deserve the stronger model.

**NVIDIA CUDA GPU for training; CPU for replay.** LoRA, or low-rank adaptation,
trains small added weight updates while keeping most of Laya fixed. We also
train its final decision layers. One shared model learns both correctness
questions.

```sh
# Create your own training folder while preserving the downloaded reference results.
uv run --frozen --group outcomes python router.py prepare \
  --study evidence --output run-output/my-router
```

```sh
# On your CUDA machine, install the locked LoRA training dependencies.
uv sync --frozen --group outcomes --group training
```

The first training run also downloads the public Laya checkpoint. No SIE API
key is needed for this stage.

```sh
# Train and evaluate LoRA with 500 examples, then repeat the same recipe with 2,000.
for population in original expanded; do
  uv run --frozen --group outcomes --group training python router.py fit-native \
    --study evidence --run run-output/my-router --population "$population"
  uv run --frozen --group outcomes --group training python router.py evaluate \
    --study evidence --run run-output/my-router --population "$population" \
    --method native --output "run-output/laya-$population.json"
done
```

One reserved portion of the training data, keeping whole question families
together, chooses how many passes, or **epochs**, to make through the data.
Selection checks zero through eight epochs, then starts
again from the original checkpoint and trains on all training rows for the
chosen duration. Calibration and test do not select that duration.

With 500 examples, LoRA saves **14.9% before hosting** with a **1.17-point
gap**. With 2,000 it saves **16.4%** with a **0.80-point gap**. At this
primary setting, adaptation preserves more correct answers than
unchanged Laya but spends more on 27B. At the secondary setting, the
2,000-example LoRA reaches **29.4% savings / 1.53-point gap**, compared
with unchanged Laya's **26.9% / 1.77 points**. The full report retains
both training sizes and every declared cutoff target.

The reference runs selected one epoch for both training sizes. Selecting,
fitting, and exporting took **2.4 minutes for 500 examples** and **8.1 minutes
for 2,000**, excluding environment startup. The allocated H100 lifetime, including
startup and input scoring, was **18.4 minutes**: about **$1.48** at the recorded
hardware rate, an estimate rather than an invoice.

<a id="embedding-router"></a>

## 4. Train the embedding router [savings **20.9%**; quality gap **0.47 points**]

**Purpose:** test whether a larger, fixed text encoder and tiny trained
classifiers can make better choices with simpler training.

**CPU with saved embeddings.** Qwen3-Embedding-4B converts each question into
2,560 numbers representing its text. Two small probability models, called
**logistic classifiers**, learn from those numbers: one predicts correctness
for Qwen 4B, the other for Qwen 27B. The encoder stays fixed.

```sh
# Fit and evaluate the two classifier training sizes using the saved Qwen embeddings.
for population in original expanded; do
  uv run --frozen --group outcomes python router.py fit-embedding \
    --study evidence --run run-output/my-router --population "$population"
  uv run --frozen --group outcomes python router.py evaluate \
    --study evidence --run run-output/my-router --population "$population" \
    --method embedding --output "run-output/embedding-$population.json"
done
```

The routing score is the predicted improvement in correctness divided by the
expected extra cost of using 27B. Expected costs come from training records;
the router never sees a new question's future answer length or correctness.

With 500 examples, this router saves **18.2%** with a **0.70-point gap**.
With 2,000 it reaches **20.9% / 0.47 points**: lower measured cost and
higher answer accuracy than the 2,000-example LoRA run. The corresponding
secondary setting reaches **33.8% / 0.83 points**.

**Check what the router learned.** At the primary setting it sends **34.9%**
of questions to 4B. Sending the same fraction at random would have an expected
**1.70-point** gap on these recorded answer pairs; the trained router's gap is
**0.47 points**. Its choices preserve more correct answers at that cheap-model
share. Equal shares do not mean equal spending: it retains more expensive
questions on 27B. See the [selection check](docs/methodology.md#selection-check)
for both the quality and cost comparison.

More data does not automatically increase savings at every setting:
the 500-example secondary rule saves **43.9%** but loses **1.73 points**.
Inspect both measures together; a higher saving alone can simply mean
sending more questions to the cheaper model.

New requests need one embedding call and two small CPU calculations. Adding
training data does not increase that work. It changes which questions go to
27B, which can increase or decrease the answer bill. A saved rule that always
chooses one model skips the embedding call. Classifier fitting took **6 seconds
for 500 examples** and **11 seconds for 2,000** on the reference CPU worker.

<a id="teacher-labels"></a>

## 5. When training answer keys are missing, use a teacher [savings **19.7%**; quality gap **0.33 points**]

**Purpose:** make the training recipe usable when you have questions and model
outputs but no trusted training answers. This is an alternative to the answer
keys used above; compare label sources on the same questions.

**CPU with recorded teacher answers.** When answer keys are unavailable, ask a
strong teacher to solve the training questions independently. Compare both
answer models' outputs with the teacher's reference, then train the same two
classifiers. The teacher sees neither the benchmark answer nor the two outputs.

The recorded teacher is **GPT-6 Astra**, which supplied 1,539 independent numeric
references. Both teacher-label and
reference-label classifiers use exactly those same 1,539 questions, with the
same settings and training groups. The remaining questions were unavailable;
they were not removed based on correctness.

Teacher references agree with the answer key on **98.25%** of these training
questions. The derived correctness labels agree on **98.70%** of cheap-model
answers and **98.51%** of strong-model answers. These training-label checks
are separate from the held-out routing comparison. Incorrect teacher answers
remain in the teacher labels.

```sh
# Refit the teacher-label classifiers and a reference-label control on the same questions.
uv run --frozen --group outcomes python router.py fit-teacher \
  --study evidence --run run-output/my-router
uv run --frozen --group outcomes python router.py evaluate \
  --study evidence --run run-output/my-router --method teacher \
  --output run-output/teacher-comparison.json
```

The teacher-label router saves **19.7%** with a **0.33-point gap** at
the primary setting. The same-question reference-label control saves
**22.0%** with a **0.60-point gap**. At the secondary setting, the pair
reaches **35.2% / 1.03 points** and **36.7% / 1.13 points**, respectively.
Teacher labels support a useful router here; these differing cutoffs
do not establish that teacher labels are better than reference labels.

These commands reuse the recorded references and incur no labeling charges.
The numeric teacher run used **$6.40** of known API usage; an earlier formatting
run used **$8.75**. Unknown usage is retained separately, so this is not a
complete invoice. Calibration and test still use the benchmark answer key.
Without any existing labels, keep a small independently reviewed sample to
check quality; agreement with the training teacher alone is insufficient.

<a id="serve"></a>

## 6. Choose a setting and serve new requests [embedding savings **20.9–33.8%**; quality gap **0.47–0.83 points**]

**Purpose:** turn the trained router into a working application and choose how
much answer quality you are willing to trade for a lower bill.

For the embedding router trained on 2,000 examples, choose between the two
saved settings introduced above:

- **`--target 1`:** estimated **$7.52 per 10,000 questions**, with about
  **47 fewer correct answers** than always using 27B.
- **`--target 2`:** estimated **$6.30 per 10,000 questions**, with about
  **83 fewer correct answers**.

These are measured averages on this test set. Use the uncertainty intervals
above when choosing; check new workloads on a separate sample. The commands
below use the primary setting, `--target 1`.

Create an account and API key in the [SIE Console](https://console.superlinked.com).
Use the free credits shown in your account for new model calls. Replaying
and training from the downloaded records uses no SIE credits.

```sh
# Connect the SDK to SIE Cloud using the API key from your console account.
export SIE_BASE_URL=https://api.superlinked.com
export SIE_API_KEY='paste-your-key'
```

```sh
# Select an answer model for a new question and save its answer and observed API charges.
uv run --frozen --group outcomes python router.py answer \
  --study evidence --run run-output/my-router --method embedding \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/cloud-answer
```

The result records the chosen model, answer, and API charges. `--target 1`
selects the saved calibration rule; it does not guarantee a correct answer.
Use `--method teacher` for the classifiers trained from teacher labels. For
your adapted decision model, include its inference dependencies:

```sh
# Load your adapted Laya router on CPU and use SIE Cloud for the chosen answer model.
uv run --frozen --group outcomes --group training python router.py answer \
  --study evidence --run run-output/my-router --method lora --device cpu \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/lora-cloud-answer
```

Use `--device cuda` for NVIDIA inference. Omit `--run` to use recorded routers.
Recorded Laya inference needs the optional weights download; your own trained
checkpoint already contains its weights. The optional recorded checkpoint
bundle is about **10.1 GB**:

```sh
# Fetch the optional decision-model checkpoints for using recorded Laya routers.
uv run --frozen --group outcomes python router.py fetch --study evidence --weights
```

The [self-hosted setup](self-host/README.md) supplies a three-model SIE deployment
for Linux and NVIDIA GPUs. Its validation hardware uses three H100 80 GB GPUs,
one per model. That full inference setup is larger than the single GPU needed
for LoRA training. After starting your own endpoint:

```sh
# Use the same router against the supplied trusted local SIE deployment.
SIE_BASE_URL=http://127.0.0.1:8080 SIE_API_KEY='' \
uv run --frozen --group outcomes python router.py answer \
  --study evidence --run run-output/my-router --method embedding \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --self-hosted --output run-output/self-hosted-answer
```

Use your deployment's API key if it requires authentication. The flag accepts
responses without Cloud billing receipts and preserves model checks. It does
not start the server. Model components and the routed SDK request passed
Modal GPU validation; the complete Docker Compose stack has not been run as
one deployment.

## Serving cost, speed, and model size

| Router | Work per routed request | What you train |
| --- | --- | --- |
| Laya / Laya + LoRA | Two correctness decisions through one shared 421-million-weight model | Small weight updates and its decision layers |
| Qwen embeddings + classifiers | One 4-billion-weight encoder call, then two small CPU calculations | Two sets of 2,560 weights and a bias |

The embedding encoder is roughly ten times larger than Laya. Architecture,
pretraining, and model size all differ; this comparison does not isolate which
causes a performance difference. Laya is smaller to host. Using the Cloud
embedding endpoint avoids hosting the encoder yourself.

We timed the same 20 training questions sequentially after one warmup:

| Router and setup | Median | 95th percentile |
| --- | ---: | ---: |
| LoRA on a resident Modal H100 | 15.23 ms | 16.00 ms |
| The same LoRA on a Modal CPU worker | 436.77 ms | 550.16 ms |
| SIE Cloud embedding + classifiers, including network time | 107.67 ms | 128.58 ms |

The classifier calculation alone took **0.36 ms median**. Native timings
exclude model loading and HTTP transport. These are different serving setups,
not equal-hardware or concurrent-throughput comparisons. The one-question CLI
loads Laya each time; its startup is slower than resident-model inference.

Laya's answer-model savings are the maximum amount you can spend on hosting
the router before breaking even. A GPU kept idle between requests can consume
those savings. The embedding API has a per-use charge, already included in
the embedding-router table; the small classifier's CPU work remains unpriced.

Use the [serving guide](docs/serving.md) to repeat the timing checks and account
for router hosting. The [methodology](docs/methodology.md) explains the frozen
decisions, teacher comparison, cost estimates, and uncertainty intervals.

Your final result is a trained router, a successful routed request, and a
short explanation of your chosen setting: its measured savings, quality gap,
router overhead, and serving requirements.

**Optional transfer case study:** [Use the same decisions with Sol and
Luna](docs/model-pair-transfer.md). Without retraining, the embedding router
saves **29.8% with a 0.27-point gap** at the original primary setting and
**46.2% with a 0.43-point gap** at the original secondary setting. The case study
compares unchanged Laya, LoRA, embeddings, and always using the cheaper model.
It reuses the same questions and supports CPU replay without another API account.
