# Earlier experiment: adapt Laya to difficulty labels

This supplementary experiment adapts Laya to human difficulty categories, then
adds training examples. The [main walkthrough](../README.md) instead learns
when the stronger solver improves the final answer; use its benefit-trained
weights and cutoffs for the savings comparison.
The model starts from the pinned pretrained Laya checkpoint and learns rank-16
attention adapters together with its native decision head.

With the same recipe, mean accuracy improves from **57.34% to 65.00%**, or
**17.9% fewer errors**, as training grows from 512 to 2,581 examples. All three
declared seeds improve; the paired 95% interval for the gain is **+3.20 to
+12.04 percentage points**. These are results on the same published 479-problem
regression benchmark. The corresponding Qwen4B plus ridge classifier reaches
**72.03%** difficulty accuracy with 512 training examples. See
[results](results.md) for every seed and the limits of these comparisons.

Run commands from `examples/train-a-router`. This guide trains both sizes,
evaluates their frozen checkpoints, and serves a trained router. Training uses
your GPU; SIE Cloud credits cover embedding inference. Use new output directories
for each training run. These are different training inputs and targets from
the benefit experiment; keep their output folders separate.

## Choose the training device

Use a CUDA GPU locally or on a VM. Apple Silicon uses the Apple GPU through
PyTorch MPS. The reference accuracy values were measured on CUDA; a fresh run
on another device can produce different scores and checkpoint selections.

For an NVIDIA GPU:

```sh
# Select the NVIDIA GPU for native training and evaluation.
export ROUTER_DEVICE=cuda
```

For an Apple Silicon Mac:

```sh
# Select the Apple GPU through PyTorch MPS.
export ROUTER_DEVICE=mps
```

Both paths use the same model, labels, eight-epoch schedule and effective batch
size of 16. The Mac runtime uses FP32, one example per micro-batch, accumulation
over 16 micro-batches, and evaluation batches of four. It disables gradient
checkpointing: a local Torch 2.9.1 check found that recomputing checkpointed
MPS dropout changed gradients despite identical forward outputs. Smaller batches
allow ordinary backpropagation without that optimization. The runtime settings
are saved in each training plan. See [Mac validation](mac-training.md) for the
measured hardware, runtime and memory requirements.

## Prepare development inputs

Download the pinned data and make a directory for each training size. Both
contain only the four development inputs. On a remote training machine, transfer
these directories and keep test inputs for later evaluation.

```sh
# Download verified evidence and prepare inputs for both training sizes.
uv sync --frozen --group training
uv run --frozen --group training python run.py fetch
uv run --frozen --group training python run.py fetch --lora-study
mkdir run-output/lora-small-inputs run-output/lora-expanded-inputs
cp evidence/train.jsonl evidence/validation.jsonl evidence/calibration.jsonl \
  evidence/questions.json run-output/lora-small-inputs/
cp evidence/validation.jsonl evidence/calibration.jsonl evidence/questions.json \
  run-output/lora-expanded-inputs/
cp run-output/lora-study/expansion/train-expanded.jsonl \
  run-output/lora-expanded-inputs/train.jsonl
```

Training requires the selected GPU and several GB of output space per run.
The locked environment uses Laya 0.3.29, Torch 2.9.1, Transformers 4.57.6 and
PEFT 0.18.1. The recorded runs used an H100; fresh results can vary with hardware.

## Train on 512 examples [57.34% reference accuracy]

First complete [prepare development inputs](#prepare-development-inputs), which
creates the input directories and installs the training dependencies.
Run the adaptation stage with the declared `original` recipe:

```sh
# Train and freeze the 512-example LoRA router.
uv run --frozen --group training python finetune.py train \
  --data run-output/lora-small-inputs --output run-output/my-lora512 \
  --seed 89 --recipe original --variants lora \
  --validation-metric auc --export-dtype float32 --device "$ROUTER_DEVICE"
```

This trains attention adapters and the native decision head for eight epochs,
keeps the checkpoint with the best validation AUC, and fits a cutoff on separate
calibration examples. It writes the merged checkpoint to
`run-output/my-lora512/lora/` and freezes its selection and cutoff in
`frozen.json`. The published three-seed mean is **57.34%**; seed 89 alone scored
**57.41%**. A single training run does not reproduce a three-seed average.

## Train on 2,581 examples [65.00% reference accuracy]

First complete [prepare development inputs](#prepare-development-inputs) if you
arrived here directly from the main walkthrough.
Keep the recipe, selection and calibration rules, and change the training data:

```sh
# Train the same LoRA recipe on the expanded 2,581-example dataset.
uv run --frozen --group training python finetune.py train \
  --data run-output/lora-expanded-inputs --output run-output/my-lora2581 \
  --seed 89 --recipe original --variants lora \
  --validation-metric auc --export-dtype float32 --device "$ROUTER_DEVICE"
```

The first 512 rows are exactly the smaller training set. Validation, calibration
and test remain unchanged. The published three-seed mean is **65.00%**; seed 89
alone scored **66.60%**. At eight epochs, growing the data also grows the number
of optimizer updates from 256 to 1,296. This is the practical recipe of adding
examples and training for the same number of passes, not a fixed-compute study.

After both runs have written `frozen.json`, evaluate the fixed models:

```sh
# Evaluate both frozen routers on the unchanged test set.
uv run --frozen --group training python finetune.py evaluate \
  --evaluation evidence/test.json --questions evidence/questions.json \
  --output run-output/my-lora512 --device "$ROUTER_DEVICE"
uv run --frozen --group training python finetune.py evaluate \
  --evaluation evidence/test.json --questions evidence/questions.json \
  --output run-output/my-lora2581 --device "$ROUTER_DEVICE"
```

The serving commands below use `run-output/my-lora2581`. Choose a new
`--models-dir` in both the configuration and server commands when switching
checkpoints. Seed 89 is the fixed demonstration seed; the complete study
uses seeds 89, 137 and 211 and reports all of them. For a new application, reserve
new evaluation data before training and define labels for the outcome you need.
The published benchmark supports reproduction, not repeated tuning against its
answers.

On a fresh CUDA machine, training and direct evaluation can start slowly:
Transformers may compile ModernBERT blocks, and evaluation also runs timing
calls before printing results. The complete study's guarded evaluator below
checks development-score parity before evaluating test cases.

## Serve your trained router

Configure SIE with the expanded checkpoint you trained and evaluated above:

```sh
# Configure and start SIE with the router you trained.
uv run --frozen python decision.py configure \
  --run run-output/my-lora2581 --variant lora \
  --models-dir run-output/my-lora-models
SIE_IPC_SOCKET_PATH=/tmp/sie-example-lora.sock \
uvx --from 'sie-server[local]==0.10.0' sie-server serve \
  --host 127.0.0.1 --port 8100 --models-dir run-output/my-lora-models \
  --models example/laya-difficulty-lora --preload example/laya-difficulty-lora
```

Wait for the model to load. In another terminal, from the same example folder:

```sh
# Route a fresh problem and check 32 decisions against your frozen evaluation.
export SIE_BASE_URL=http://127.0.0.1:8100
unset SIE_API_KEY # The local server needs no Cloud key.
uv run --frozen python decision.py route \
  --run run-output/my-lora2581 --variant lora --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
uv run --frozen python check_serving.py \
  --run run-output/my-lora2581 --variant lora \
  --evaluation evidence/test.json --limit 32 \
  --output run-output/my-lora-serving.json
```

The check compares 32 decisions from this checkpoint with its own frozen
`evaluation-test.json`. It checks serving consistency, not new accuracy. Stop
the server with Ctrl-C. Custom checkpoints run on your own SIE server; the
[Cloud exercise](../README.md#serve-your-router) uses the public Qwen embedding
model and your learned classifier.

To try serving without training first, fetch the published seed-89 checkpoint
(about **1.7 GB**):

```sh
# Download the published expanded checkpoint to try serving without training.
uv run --frozen python run.py fetch --lora-study --weights lora
```

Then use `--run run-output/lora-study/expanded/seed-89` in the commands above
and choose a new `--models-dir`. Seed 89 was fixed before test evaluation.
`fetch --lora-study --weights lora` fetches this expanded FP32 model; omitting
`--lora-study` selects the separate original 512-example checkpoint.

## Rebuild the dataset from its source

The training steps above use the downloaded, hash-checked data. To audit its
construction, rebuild from the pinned public Parquet source:

```sh
# Rebuild the training data and audit from the pinned public source.
uv run --frozen --group data python data.py \
  --source run-output/lora-source \
  --output run-output/lora-rebuilt \
  --previously-used evidence/prior-used-ids.json
uv run --frozen --group data python expand_training.py \
  --source run-output/lora-source --evidence evidence \
  --output run-output/lora-expansion
```

The expansion uses all 2,581 eligible pre-2018 duplicate-family representatives.
Its first 512 rows must exactly match the original training file. It preserves
the original human-rating thresholds and removes families that cross temporal
partitions. Validation, calibration, and test populations stay unchanged.
`training-expansion-audit.json` records the source hashes, row identities,
family identities, and generated-file hash.

Compare the regenerated files with the downloaded expansion:

```sh
# Check that the rebuilt training data and audit match the published files.
cmp run-output/lora-expansion/train-expanded.jsonl \
  run-output/lora-study/expansion/train-expanded.jsonl
cmp run-output/lora-expansion/training-expansion-audit.json \
  run-output/lora-study/expansion/training-expansion-audit.json
```

## Replay the recorded comparison

```sh
# Download verified records and recompute the LoRA comparison on CPU.
uv sync --frozen
uv run --frozen python run.py fetch
uv run --frozen python run.py fetch --lora-study
uv run --frozen python lora_report.py \
  --study run-output/lora-study --evidence evidence \
  --output run-output/lora-report.json
```

The first download contains the original experiment in `evidence/`. The second
contains the LoRA study in `run-output/lora-study/`: the declared plan, every
candidate's learning curves and selection records, the training-expansion audit,
and the recorded evaluations. This adds about **6 MB** without model weights.
Both downloads check immutable manifest and file
hashes. To choose another destination, pass `fetch --lora-study --evidence` with
that directory. A directory containing another experiment's manifest is rejected
before download.

`lora_report.py` recomputes validation selection, checks the frozen plans and
calibration cutoffs, and recomputes the comparison from recorded per-problem
scores. It needs neither weights nor a GPU. Its stages separate the original
cross-entropy selection, AUC selection with the original recipe, recipe choice
at the same training size, and the increase in training data. Reported seeds are
averaged before resampling problems or competition-year groups; three seeds do
not create three times as many independent test cases.

Generate the LoRA comparison figure from the same verified records:

```sh
# Plot the validation and test results from verified LoRA records.
uv run --frozen --group plots python plot.py \
  --lora-study run-output/lora-study --evidence evidence \
  --output run-output/lora-figures
```

This writes `lora-study.png` and `lora-study.svg`: validation AUC for every recipe,
then regression accuracy for each comparison stage. Both panels show every seed
and the mean across seeds; the regression panel includes the unchanged ridge
reference. The figure recomputes the strict report rather than reading a saved
`report.json`. Use a new output directory for another rendering. Running
`plot.py` without `--lora-study` continues to generate the original baseline
figures.

## Reproduce the complete recipe comparison

The main training path uses the recipe already selected on development data.
To repeat that selection, use the two development directories prepared above:

```sh
# Run every declared recipe and seed, then train the winner on more data.
uv sync --frozen --group training
uv run --frozen --group training python lora_study.py \
  --data run-output/lora-small-inputs \
  --expanded-data run-output/lora-expanded-inputs \
  --output run-output/my-lora-study --device "$ROUTER_DEVICE" \
  --timeout-seconds 14400
```

The command sets a four-hour wall-clock limit; choose a limit appropriate to
your hardware before starting. It retains 21 native FP32 checkpoints, requiring
tens of GB of disk space. The output directory must be new. An interrupted study
preserves partial records but does not resume in place; use a fresh output for a
new complete run.

Every candidate starts from the same pinned Laya checkpoint and uses eight
epochs, effective batch size 16, a 512-token input limit, and a 256-token question
budget. Training uses supervised cross-entropy, AdamW with cosine learning-rate
decay, and shuffled choice-option order with aligned targets. The exact recipes
are declared in `finetune.py`:

| Recipe | Encoder attention layers adapted | Rank / alpha | Adapter / head learning rates | Native head updates | Adapter dropout / weight decay / smoothing |
| --- | --- | --- | --- | --- | --- |
| `original` | All | 16 / 32 | 2e-4 / 1e-4 | Entire head | .05 / .01 / 0 |
| `slow` | All | 16 / 32 | 5e-5 / 1e-5 | Entire head | .05 / .01 / 0 |
| `regularized` | All | 8 / 16 | 5e-5 / 1e-5 | Entire head | .1 / .1 / .1 |
| `frozen-head` | All | 8 / 16 | 1e-4 / none | Frozen | .1 / .1 / .1 |
| `scorer-only` | All | 8 / 16 | 1e-4 / 1e-5 | Scorer only | .1 / .1 / .1 |
| `top-layers` | Layers 24–27 | 16 / 32 | 5e-5 / 1e-5 | Entire head | .05 / .01 / 0 |

Adapters target `attn.Wqkv` and `attn.Wo`; no candidate adapts encoder MLP
layers. `scorer-only` still trains the encoder adapters. The five new recipes
freeze the unused action head. Label smoothing applies only to training targets.
Native-head dropout remains active during training, including in frozen-head
runs; evaluation disables dropout. The [source review](laya-adaptation.md)
explains the motivation and differences from public Laya adaptation recipes.

Selection has three separate steps:

1. Train each of the six recipes on 512 examples for seeds **89, 137, and 211**.
   For each run, choose the maximum ROC AUC on the 2018 validation set. Epoch zero
   is eligible; ties retain the earliest epoch. Cross-entropy and training loss
   remain diagnostics.
2. Choose the recipe with the highest mean selected validation AUC across all
   three seeds. Ties retain recipe declaration order, shown in the table. Write
   `recipe-frozen.json` before starting the larger training runs.
3. Train that same recipe on 2,581 examples for all three seeds, with unchanged
   learning settings. For every selected checkpoint, fit its cutoff by balanced
   accuracy on the separate 2019 calibration set. Calibration does not choose the
   recipe or epoch. Write `all-frozen.json` when every run is complete.

The 18 screening runs plus three expanded runs are all retained. No best seed
is selected. The driver checks that both training directories have identical
validation, calibration, and question files, and that the smaller training file
is the exact prefix of the expanded one. It never reads test inputs.

Keeping eight epochs also increases the training budget: 512 examples produce
256 optimizer updates, while 2,581 produce 1,296. This compares the practical
recipe of adding examples and training for the same number of passes; it does
not isolate dataset size at fixed compute.

## Why this study exports FP32 weights

The original Laya serializer writes half-precision weights. In the first
LoRA-study attempt, that conversion changed validation logit margins beyond the
declared export tolerance. Before recipe selection or test evaluation, the
study was amended to preserve FP32 weights for every candidate. The initial
attempt is retained as an audit; the training recipes and data did not change.

The study passes `--export-dtype float32` to `finetune.py`. This changes the
saved checkpoint precision, not the training objective or native architecture.
Encoder adapters are merged, validation margins are checked before and after
merge and export, and calibration uses the reloaded exported checkpoint. All
candidates in this study use the same export rule. The original baseline keeps its
original export and selection protocol. Consequently, the historical CE baseline
versus the new original-recipe AUC control changes both checkpoint selection and
export precision. That comparison cannot attribute a difference to selection
alone. Recipe and training-size comparisons within the new study all use FP32.

## Evaluate only after the study is frozen

Wait for `ALL_LORA_RUNS_FROZEN` and `run-output/my-lora-study/all-frozen.json`.
Then make the original evidence available on the evaluation machine and run:

```sh
# Check runtime parity, evaluate the frozen study, and recompute its report.
uv run --frozen --group training python lora_evaluate.py \
  --study run-output/my-lora-study --evidence evidence \
  --device "$ROUTER_DEVICE" --timeout-seconds 1800
uv run --frozen python lora_report.py \
  --study run-output/my-lora-study --evidence evidence \
  --output run-output/my-lora-report.json
```

The evaluator checks the completed study and development selections, then sets
`TORCH_COMPILE_DISABLE=1` and `TORCHDYNAMO_DISABLE=1` before loading any model.
Each frozen checkpoint must reproduce its selected validation scores and saved
calibration scores within a maximum absolute margin difference of **0.03**, with
**zero changed decisions** at its existing cutoff on both splits. These checks
run in isolated processes under the same overall timeout. The declared checks,
per-checkpoint scores, and completion hashes are saved in `runtime-parity`
receipts. A failed check stops the run before test inputs are opened.

In the recorded run, all six checkpoints passed: maximum development-margin
drift was 0.000174, with no changed decisions. The first evaluation attempt had
produced a plan but no outcome files during a prolonged startup. That plan and
the runtime amendment are retained in the public audit. The startup cause was
not localized; disabling compilation did not eliminate the initial delay.

After every checkpoint passes, it evaluates the original recipe at 512 examples, the
selected recipe at 512, and the selected recipe at 2,581, each for all three
seeds. If the original recipe wins, its 512-example stage is evaluated once.
Existing parity and evaluation outputs are preserved; rerunning an interrupted
attempt in place is refused. The runtime change keeps weights and cutoffs fixed;
latency measurements from eager and compiled execution are different comparisons.
Fresh GPU evaluation needs the
trained weights; downloading only the published core records supports CPU report
replay instead.

The report compares these stages with the original CE-selected native runs and
the unchanged `ridge512` embedding reference. The published test is an **exposed
regression benchmark** in this study. Development selection is separate from
its outcomes, but a new claim about future generalization requires a new test
population. Difficulty accuracy alone does not establish downstream answer
quality or monetary savings; those need paired solver outcomes and serving costs.
