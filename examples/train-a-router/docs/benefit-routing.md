# Train a router to predict when the stronger solver helps

The useful question is **whether a more expensive model will improve the
answer**. This guide trains that decision from recorded answers, then uses the
router to answer a new question. It compares unchanged Laya, adapted Laya,
and a small regression model trained on frozen Qwen embeddings. A
competition-name-only control checks how much the problem's source explains.

The application is mathematics question answering. **Answer accuracy** means
correct final answers, rather than agreement with a human difficulty rating.
The **quality gap** is always-strong accuracy minus routed accuracy, measured
in percentage points. Savings compare the selected answer's cost with always
using the stronger model. Recorded embedding charges are included where that
router runs; native router hosting remains unpriced.

## What the training labels mean

Each problem receives one recorded answer from each solver. Both final answers
are graded against the reviewed benchmark reference:

| Smaller solver | Stronger solver | Training value |
| --- | --- | ---: |
| Wrong | Correct | +1 |
| Correct | Correct | 0 |
| Wrong | Wrong | 0 |
| Correct | Wrong | −1 |

This is **signed benefit**: stronger-correct minus smaller-correct. Keeping
both-fail cases and smaller-model wins matters. A router that sends every hard
problem to the stronger model can still waste money on problems both models
miss. A high learned score sends the request to the stronger solver; the score
is not a probability that its answer is correct.

The training pool contains 512 problems. One question lacks the multiple-choice
options needed to grade its answer unambiguously; a source-only review excluded
it before collecting responses. Training therefore uses **511 problems**.
Another **246 calibration problems** choose the routing cutoff, and **479
test problems** measure the frozen choices. Problems from the same examination
paper stay together. The [methodology](methodology.md) explains the split,
duplicate checks, answer reviews, training folds and uncertainty estimates.

These are automated labels **derived from benchmark references**. No separate
teacher judges the answers. Without reference answers, a teacher could compare
the two candidate answers, but that would be a different labeling experiment:
check teacher judgments against a separately reviewed sample and evaluate on
independent references or human judgments. This example does not establish
teacher-only labeling accuracy.

## The two solver systems

| Setting | Smaller solver | Stronger solver |
| --- | --- | --- |
| Model | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3.8-27B-FP8` |
| Initial output limit | 1,536 tokens | 3,072 tokens |
| Requested working length | At most 150 words | At most 300 words |
| Temperature | 0 | 0 |

Both receive the problem and competition name, without a reference answer,
solution or difficulty rating. If the initial response is truncated or lacks
an explicit final answer, the same solver gets at most one additional
256-token final-answer call. Both calls count toward serving cost. Automatic
inference retries are disabled. Exact prompts, model revisions and grading
rules are saved in the evidence and checked by the code.

The recorded training pairs use self-hosted SIE; calibration and test pairs use
SIE Cloud. They share the declared model revisions and request settings, while
hardware, serving versions and scheduling differ. Runtime receipts preserve
those differences; identical deployment bytes are not claimed. Every compared
router selects from the **same recorded test answer pairs**, so reporting
does not generate another answer or incur another solver charge.

## Download the experiment once — CPU

Start at the repository root. Python 3.12 and `uv` are required. Fitting the
embedding router and replaying reports need only CPU. Native Laya training
needs an Apple Silicon GPU or an NVIDIA GPU, on your computer or a VM. Training
Laya on a Mac does not also host the two large answer models.

```sh
# Install the example and download the pinned labels, vectors, plans and reports.
cd examples/train-a-router
uv sync --frozen --group evaluation
uv run --frozen python run.py fetch --solver-study
```

The default destination is `run-output/solver-study`. Its root contains the
original solver `plan.json`, `settings.json` and `policies-frozen.json`; keep
them there because later stages verify their hashes. Under `benefit/`:

| Folder | Contents |
| --- | --- |
| `training/` | The 511 eligible inputs, source reviews and paired outcome grades. |
| `baseline/` | The pinned original dataset and cached Qwen4B embeddings. |
| `head/` | The fitted embedding regression model and frozen cutoffs. |
| `native/` | Laya training plans, scores and cutoffs; weights are optional. |
| `control/` | The fixed competition-name-only comparison. |
| `comparison/` | The combined reports and available figures. |

This download makes no inference calls and needs no Cloud key. It excludes
native weights and raw response archives. The experiment's `REPLAY.md` also
provides commands using its exact archived source and lock file.

## Reproduce the training

These commands reproduce a **published experiment**, whose test questions and
answers are already available. They copy its historical plan bytes rather
than create a new assertion that you have never seen the test outcomes.
Every new stage receipt and evaluation is marked
`public_regression_reproduction`. A new generalization claim requires a fresh
test split; repeatedly improving this published score is not that test.

### Prepare a new reproduction — CPU

```sh
# Copy the historical plans into a new run without fitting either router yet.
uv run --frozen python benefit_replay.py prepare --output run-output/my-benefit
```

The command prepares both model families. Neither has been fitted, and no
Cloud call or model download is needed.

### Train native Laya — Apple or NVIDIA GPU

Select **one** device. On NVIDIA, use `cuda`; on Apple Silicon, use `mps`.
The [Mac guide](mac-training.md#runtime-settings) explains the Apple GPU
memory settings. Its historical timings are not timing estimates for this
different training objective.

```sh
# Install native training dependencies and select your NVIDIA GPU.
uv sync --frozen --group training --group evaluation
export ROUTER_DEVICE=cuda
```

```sh
# On Apple Silicon, replace the device selection with the Apple GPU.
export ROUTER_DEVICE=mps
```

```sh
# Train and export the native benefit router using the same recorded labels.
uv run --frozen --group training python benefit_replay.py fit-native \
  --run run-output/my-benefit --device "$ROUTER_DEVICE"
```

The command downloads the pinned public Laya checkpoint if needed. It records
unchanged Laya and trains rank-16 attention adapters plus the decision head.
Targets −1, 0 and +1 become strong-option training weights 0, 0.5 and 1.
A training-only paper fold selects among the initial model and eight epochs;
the selected epoch count is then fitted from a fresh base on all 511 examples.
The initial model is eligible to win. Calibration chooses the cutoff afterward.

Merged checkpoints are exported as ordinary 32-bit Laya weights and checked
for changes in scores. Outputs are under `my-benefit/native/base/` and `lora/`.
Allow several GB of disk space per run. Hardware can change numerical results
and the selected epoch; the guide does not promise identical GPU training bytes.

### Fit the embedding router — CPU

```sh
# Fit a small regression model and the competition-name control from cached vectors.
uv run --frozen python benefit_replay.py fit-embedding --run run-output/my-benefit
```

Qwen3-Embedding-4B stays fixed. The command learns 2,560 feature weights and a
bias from the saved vectors and signed-benefit labels. Paper-grouped training
folds choose the regularization strength; calibration then chooses cutoffs.
It also fits the fixed competition-name control. No embedding or solver API
calls are needed. Outputs are under `my-benefit/head/` and `my-benefit/control/`.
This CPU fit and native training are independent; evaluation needs both.

### Evaluate your frozen models — CPU

```sh
# Apply every saved cutoff to the published paired answers and print the comparison.
uv run --frozen python benefit_replay.py evaluate --run run-output/my-benefit
```

The command writes `my-benefit/evaluation.json` and `evaluation.md`. It reports
always-strong, always-cheap and all four router variants. The primary cutoff
targets at most **1 percentage point of quality loss on calibration**; 0, 2
and 5 are retained as additional choices. These are empirical calibration
targets, not guarantees for test data or future requests. Reporting never
selects a different cutoff from the test table.

Existing outputs are not overwritten. If a stage fails, preserve its records
and use a new reproduction directory for another complete attempt. No
`authorize-test` command is needed or appropriate for this public replay.

## Answer a new question through SIE Cloud

Create an API key in the [SIE Console](https://console.superlinked.com), then
set the endpoint explicitly. The regression model stays on your computer;
SIE Cloud supplies embeddings and the selected answer model.

```sh
# Select SIE Cloud and authenticate with your own account key.
export SIE_BASE_URL=https://api.superlinked.com
export SIE_API_KEY='paste-your-key'
```

```sh
# Use your fitted embedding router to choose one solver and save its answer and charges.
uv run --frozen python benefit_route.py --run run-output/my-benefit/head \
  --contest AMC10 --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/my-benefit-answer
```

For the native router, compute the decision locally and use Cloud for the
selected answer:

```sh
# Use your trained Laya router locally, then call only its selected Cloud solver.
uv run --frozen --group training python benefit_route.py \
  --run run-output/my-benefit/head --native-run run-output/my-benefit/native \
  --variant lora --device "$ROUTER_DEVICE" --contest AMC10 \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/my-benefit-native-answer
```

`--variant base` uses unchanged Laya instead. Each invocation needs a new output
directory and makes a new paid answer request; the two commands above are two
requests. The route uses at most one embedding or native decision, one solver
answer and one conditional final-answer turn. A static always-strong/cheap
calibration choice bypasses the routing model. `result.json` records the
selected model, answer and available API charges. A new question without a
reference answer is **not automatically graded**.

To try published native weights without training them, download the optional
checkpoints and use the published head/native pair together:

```sh
# Download the native checkpoints and use the published LoRA router on local CPU.
uv run --frozen python run.py fetch --solver-study --solver-weights
uv run --frozen --group training python benefit_route.py \
  --run run-output/solver-study/benefit/head \
  --native-run run-output/solver-study/benefit/native --variant lora --device cpu \
  --contest AMC10 --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/published-benefit-answer
```

CPU scoring is supported here; native **training** still requires a GPU.
Use a matching head/native plan pair. These commands select both from the
published study.

## Use your own answer-model deployment

An existing SIE deployment must expose both pinned answer-model IDs above and
support the structured final-answer call. The embedding route also needs
`Qwen/Qwen3-Embedding-4B` at its pinned revision on that endpoint; the local
native route does not. The [SIE deployment guide](https://superlinked.com/docs/deployment/)
covers server setup. Changing the client URL does not start these models, and
the single-model embedding server in the introductory guide cannot answer
the solver requests.

```sh
# Send the native router's selected answer request to your own running SIE endpoint.
export SIE_BASE_URL=http://127.0.0.1:8080
unset SIE_API_KEY  # The local server needs no Cloud key.
uv run --frozen --group training python benefit_route.py \
  --run run-output/my-benefit/head --native-run run-output/my-benefit/native \
  --variant lora --device "$ROUTER_DEVICE" --self-hosted --contest AMC10 \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --output run-output/my-local-benefit-answer
```

Set your deployment's own key if it requires authentication. `--self-hosted`
permits missing Cloud billing receipts; it does not mean free computation.
Measure hosting costs separately before claiming net savings for this setup.

## Inspect records or collect new labels

The default download contains grades sufficient for fitting and reporting.
The optional archives add the original responses, call receipts and training
acquisition history:

```sh
# Download and unpack the recorded answers without issuing new model calls.
uv run --frozen python run.py fetch --solver-study --solver-records
tar -xzf run-output/solver-study/solver-records.tar.gz -C run-output/solver-study
tar -xzf run-output/solver-study/embedding-records.tar.gz -C run-output/solver-study
tar -xzf run-output/solver-study/benefit/training-records.tar.gz -C run-output/solver-study
```

```sh
# Regrade the training responses and check exact agreement with the published labels.
uv run --frozen --group evaluation python benefit_data.py grade \
  --run run-output/solver-study/benefit/training --output run-output/regraded-benefit-train.json
cmp run-output/regraded-benefit-train.json run-output/solver-study/benefit/training/grades-train.json
```

`cmp` prints nothing when the files match. Original calibration/test regrading
and full CPU report replay are documented in the downloaded `REPLAY.md`.
Runtime receipts retain the actual source/model revisions and serving versions.
Private launcher scripts are represented by verified hashes and an explicit
omission reason; the public collector is sufficient for a new collection.

An optional fresh training-label collection requests both solvers on all 511
eligible problems: **1,022 initial calls**, plus conditional finalizers. This
is separate spending from serving one selected answer. The published replay
above needs none of these calls.

```sh
# Select Cloud again and reconstruct eligible inputs with their reviewed reference answers.
export SIE_BASE_URL=https://api.superlinked.com
export SIE_API_KEY='paste-your-key'
uv run --frozen --group evaluation python benefit_data.py prepare \
  --evidence run-output/solver-study/benefit/baseline --output run-output/my-benefit-labels
```

```sh
# Verify the catalog and collect both arms, stopping if any phase fails.
(
  for arm in cheap strong; do
    uv run --frozen --group evaluation python benefit_acquire.py \
      --directory run-output/my-benefit-labels --arm "$arm" --phase initial \
      --workers 4 --max-seconds 7200 --max-credits 200000 || exit
    uv run --frozen --group evaluation python benefit_acquire.py \
      --directory run-output/my-benefit-labels --arm "$arm" --phase final --resume \
      --workers 4 --max-seconds 3600 --max-credits 200000 || exit
  done
)
```

```sh
# Combine complete paired records and derive labels using the frozen answer grader.
uv run --frozen --group evaluation python benefit_collect.py gather --directory run-output/my-benefit-labels
uv run --frozen --group evaluation python benefit_data.py grade --run run-output/my-benefit-labels
```

The credit ceiling stops new scheduling after that invocation observes the
limit; in-flight calls can finish. An incomplete collection preserves receipts.
Inspect those records before an explicit `--resume`; uncertain previous
execution also requires `--retry-unknown` and a larger declared attempt limit.
Before each phase, `benefit_acquire.py` checks the live catalog's model name
and revision, saves a receipt, and stops before generation if they differ.
This check does not assert identical serving hardware or library versions.
For a compatible self-hosted endpoint, add `--self-hosted` to both collection
commands.

New responses can produce different labels. The reproduction wrapper
deliberately uses the **published** labels and fixed training/calibration recipe;
it does not silently substitute this new collection. To improve the method with
new labels, create a separately declared training/calibration experiment and a fresh test split,
then freeze every model and cutoff before reading those test outcomes.
